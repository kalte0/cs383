Step 1: Lab

pickle_server.py 
model.pkl

To run: run python pickle_server.py

Step 2: Assignment

user-side: simple_form.html
--- To run, set up with python (to avoid both coming from the same localhost -- explained in write-up_ 

server-side: 
can run make_bigram_model.py to refresh the bigram model / retrain 
can run make_trigram_model.py to refresh the trigram model / retrain    
  Both of these will create pickle files for their respective models. 

run ngram-server.py to set up the server-side to which POST requests from the form will be sent. 
