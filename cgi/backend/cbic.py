# import embed

import sys
sys.path.append("../../facenet/src")

import mock_embed as embed
import uuid
import tensorflow as tf
import numpy as np
import math
import random
import os
import scipy as sp
import logging
from time import time

logging.basicConfig(level=logging.INFO)
_CONF = {
    "dims": 128
}

class CBIC:
    def __init__(self, model_dir, target_db_dir, wild_db_dir=None):
    
        start = time()
        logging.info("Loading pre-trained embeding model...")
        logging.info("Done (in %.1f s)." % (time() - start))
        
        self.embeder = embed.Embedding(model_dir)
        self.sessions = {}
        
        # compute embedding of all images in target_db_dir
        # these embeddings are shared across sessions
        paths = [file for file in os.listdir(target_db_dir) if os.path.splitext(file)[1] in [".jpg", ".png", ".jpeg"]]
        
        self.hypos_path = [os.path.join(target_db_dir, file) for file in paths]
        nums = len(self.hypos_path)
        self.hypos_pos = np.zeros((nums, _CONF["dims"]))
        
        logging.info("Found %d files in target_db_dir" % nums)
        logging.info("Computing their embeddings...")
        
        start = time()
        for i in range(nums):
            # img = sp.misc.imread(self.hypos_path[i])
            self.hypos_pos[i, :] = self.embeder.embed_one_by_path(self.hypos_path[i])[:]
        
        # center of hypotheses
        self.hypos_mean = np.mean(self.hypos_pos, axis=0)
        
        logging.info("Done (in %.1f s)." % (time() - start))
        
        # wild_db_dir: a large images dataset, which is required by some policies
        # leave it unimplemented for now.
    
    def Session(self, name=None):
        def random_name():
            return str(uuid.uuid1())
    
        if name == None:
            name = random_name();
        self.sessions[name] = _Session(self, name)
        # self.sessions[name].embeder = self.embeder
        # self.sessions[name].hypos_path = self.hypos_path
        # self.sessions[name].hypos_pos = self.hypos_pos
        # self.sessions[name].hypos_mean = self.hypos_mean
        
        logging.info("New Session[name=%s]" % name)
        
        return self.sessions[name]

class _Session():
    def __init__(self, parent, name, mode="GBS"):
        self.embeder = parent.embeder
        self.hypos_path = parent.hypos_path
        self.hypos_pos = parent.hypos_pos
        self.hypos_mean = parent.hypos_mean
    
        self.name = name
        self.mode = mode
        self.tests = None
        self.answers = None
        self.p = None
        
        self.tf_sess = tf.Session()
        self.tf_graph = self.tf_sess.graph
        self.tf_answers = None
        self.tf_h_tilde = None
        self.tf_loss = None
        self.tf_train = None
    
        with self.tf_graph.as_default():
            # answers[:, 0] are user's choices
            self.tf_answers = answers = tf.placeholder(tf.float32, shape=(None, 2, _CONF["dims"]))
            self.tf_h_tilde = h_tilde = tf.Variable(self.hypos_mean, dtype=tf.float32)
            # loss function desired property. t is a test:
            #   L(t) = 1 , when h ~= t_T
            #   L(t) = 0 , when h ~= t_F
            #   L(t) = .5, whne d(h, t_T) ~= d(h, t_F)
            # (deleted) --- here, we adopt L(t) = sigmoid(2 * d(h, t_T) / d(h, t_F)), where d(x) = sum(x ** 2) / 2 (l2 norm)    --- 
            # here, we adopt L(t) = d(h, t_F) / ( d(h, t_T) + d(h, t_F) ), where d(x) = sum(x ** 2) (l2 norm)
            dhT = tf.nn.l2_loss(h_tilde - answers[:, 0, :])
            dhF = tf.nn.l2_loss(h_tilde - answers[:, 1, :])
            self.tf_loss = loss = -tf.reduce_sum(tf.log( dhF / (dhT + dhF)))
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            self.tf_train = optimizer.minimize(loss)
        
        self.reset()
    
    def reset(self):
        self.tf_sess.run(tf.global_variables_initializer())
        # shape: [:, 2], records test image tuples as [(index1, index2)]
        self.tests = []
        # shape: [:, 2, 128], records embedding test images, with [:, 0, :] being the user's choice
        self.answers = []
        # shape: [:], records P(h)
        self.p = np.zeros(len(self.hypos_path))
        self.compute_p()
        
    def compute_p(self):
        nums = len(self.hypos_path)
        h_tilde = self.tf_sess.run(self.tf_h_tilde)
        # any monotonically increasing function will work
        # maybe gaussian distribution is a better choice?
        g = lambda x: x 
        self.p = g(np.linalg.norm(h_tilde - self.hypos_pos, axis=1))
        self.p /= np.sum(self.p) 
    
    def next_test(self):
        # generate a test
        logging.info("Computing optimal test...")
        start = time()
        
        test_pair = None
        if self.mode == "GBS":
            test_pair = self.GBS()
        else:
            raise Exception("Unexpected test choosing policy '%s'" % self.mode)
        self.tests.append(test_pair)
        
        logging.info("Done (in %.1f s)." % (time() - start))
        return test_pair
    
    def next_test_with_path(self):
        test_pair = self.next_test()
        return (self.hypos_path[test_pair[0]], self.hypos_path[test_pair[1]])
    
    def answer(self, choice):
        # choice: 0->left, 1->right
        (T, F) = (self.tests[-1][0], self.tests[-1][1])
        if choice == 1:
            (T, F) = (F, T)
            self.tests[-1] = (T, F)
        answer = np.zeros((2, _CONF["dims"]))
        answer[0, :] = self.hypos_pos[T, :]
        answer[1, :] = self.hypos_pos[F, :]
        self.answers.append(answer)
        
        # Feed model with user's answer to last test to update estimated hypothesis
        with self.tf_sess.as_default():
            # Minimize loss
            threshold = 0.01
            max_steps = 1000
            old_h_tilde = self.tf_h_tilde.eval()
            step = 0
            
            while step < max_steps:
                step += 1
                self.tf_train.run({self.tf_answers: self.answers})
                if(np.sum((self.tf_h_tilde.eval() - old_h_tilde) ** 2) < threshold):
                    break
            
            self.compute_p()
            return self.tf_loss.eval({self.tf_answers: self.answers})
        
    # Test generation policies
    def GBS(self):
        tests_threshold = 1000
        
        samples_num = min(int(math.sqrt(tests_threshold)), len(self.hypos_path))
        samples_ind = random.sample(range(0, len(self.hypos_path)), samples_num)
        best_test_loss = np.inf
        best_test_pair = (None, None)
        
        saved_h_tilde = self.tf_sess.run(self.tf_h_tilde)
        restore_h_tilde = self.tf_h_tilde.assign(saved_h_tilde)
        
        for i in samples_ind:
            for j in samples_ind:
                if j > i:
                    (p0, p1) = (self.p[i], self.p[j])
                    self.tests.append((i, j))
                    
                    self.tf_sess.run(restore_h_tilde)
                    loss0 = self.answer(0)
                    self.answers.pop()
                    
                    self.tf_sess.run(restore_h_tilde)
                    loss1 = self.answer(1)
                    self.answers.pop()
                    
                    self.tests.pop()
                    loss = p0 * loss0 + p1 * loss1
                    if(loss < best_test_loss):
                        best_test_loss = loss
                        best_test_pair = (i, j)
        
        return best_test_pair