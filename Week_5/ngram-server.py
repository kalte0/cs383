import sys # System commands ( for command line arguments )
from collections import defaultdict
import numpy as np
from flask import Flask, request, jsonify
import pickle, random
from flask_cors import CORS
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__) # Initialize the Flask application 


def main(): # to run when this is run directly from the command line. 
  print("Number of arguments:", len(sys.argv), "arguments.")
  if len(sys.argv) <= 1: # No command line arguments
    CORS(app)
    app.run(debug= True) # Run the app in debug mode
  elif len(sys.argv) == 2: # One command line argument
    filepath = sys.argv[1] # Get the filepath from the command line
    print("Filepath:", filepath) # print the filepath to the console. 
    model = pickle.load(open(filepath, 'rb')) # Load the pre-trained model
  else:
    print("Usage: python ngram-server.py <model_filepath>")
    sys.exit(1)

def generate_text(model, tokenizer, max_sequence_len, starting_word, num_words=50):
  generated_text = starting_word
  for _ in range(num_words):
    encoded = tokenizer.texts_to_sequences([generated_text])[0]
    encoded = pad_sequences([encoded], maxlen=max_sequence_len, padding='pre')
    predicted_index = model.predict_classes(encoded, verbose=0)
    predicted_word = ""
    for word, index in tokenizer.word_index.items():
      if index == predicted_index:
        predicted_word = word
        break
    generated_text += " " + predicted_word
  return generated_text
  
@app.route('/')
def home(): 
  return 'Hello, World! This is the ML model API.'

@app.route('/predict', methods=['POST'])
def predict_sentence():
  #Get the data from POST request
  data = request.get_json(force=True)
  
  try: 
    filepath = data['filepath']
    word = data['initial_word']
  except KeyError: 
    return jsonify(error="A key is missing from the request payload. Please include a 'filepath' and 'initial_word' for prediction in your JSON"), 400

  # Open the model and tokenizer
  model = pickle.load(open(filepath, 'rb'))
  model = np.array(model)
  tokenizer = model['tokenizer'] # Dot accessor for tokenizer
  max_sequence_len = model.max_sequence_len
  generated_sentence = generate_text(model['model'], tokenizer, max_sequence_len, starting_word)
  print("Generated Sentence:", generated_sentence)
  
  # Return the prediction 
  response = jsonify(prediction=generated_sentence)
  response.headers.add('Access-Control-Allow-Origin', 'http://127.0.0.1:8000') # Allow only requests from the local host 8000 (the default port for python -m http.server)
  return response

@app.route('/check_server_status', methods=['GET']) 
def check_server_status(): 
    return 'Server is active'; 

if __name__ == '__main__':
    main()
    