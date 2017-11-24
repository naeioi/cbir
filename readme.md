# Comparison-based information retrieval
This is an experimental project aiming at pinpoint a target in a large dataset with unknown high dimensional representations within minimum queries on resemblance between two candidates. 

A simple web-based interface for query is implemented. The hypothesis update and selection module is implemented via module `CBIC` in `cgi/backend/cbic.py`. The prototype forked [davidsandberg/facenet](https://github.com/davidsandberg/facenet) and uses [LFW](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk) as its experimental dataset. 

Currently the algorithm sees no converge, thus the document is defered until I see the viability of the project. 