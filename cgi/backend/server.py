from flask import Flask, session, jsonify, send_from_directory, redirect, request
from cbic import CBIC
import os

app = Flask(__name__, static_folder="../frontend", static_url_path="/static")
app.secret_key = 'ASDFasjaf&(*31___;;[[\''
cbic = CBIC(model_dir="../../model/20170512-110547", target_db_dir="../../datasets/cbic-test")
cbic_session = {}

@app.route("/")
def hello(): 
    cnt = 0 if not 'count' in session else int(session['count']) + 1
    session['count'] = cnt
    return 'Hello for ' + str(cnt) + ' times!'
    
@app.route("/cbic")
def main():
    if not 'uuid' in session:
        cs = cbic.Session()
        cbic_session[cs.name] = cs
        session["uuid"] = cs.name
        print("Set cookie[uuid=%s" % cs.name)
    else:
        uuid = session["uuid"]
        if not uuid in cbic_session:
            session.clear()
            return redirect("/cbic")
    return send_from_directory('../frontend', "index.html")
    
@app.route("/cbic/images/<path:file>")    
def images(file):
    return send_from_directory("../../datasets/cbic-test", file)

@app.route("/cbic/new")
def new():
    uuid = session["uuid"]
    sess = cbic_session[uuid]
    sess.reset()
    pair = sess.next_test_with_path()
    pair = (os.path.split(pair[0])[1], os.path.split(pair[1])[1])
    return jsonify(pair)

@app.route("/cbic/next")    
def next():
    choice = request.args.get('choice', type=int)
    uuid = session["uuid"]
    sess = cbic_session[uuid]
    sess.answer(choice)
    pair = sess.next_test_with_path()
    pair = (os.path.split(pair[0])[1], os.path.split(pair[1])[1])
    return jsonify(pair)
    
if __name__ == '__main__':
    app.run()
    