from collections import defaultdict
from flask import Flask, request, jsonify
import pickle, random
from flask_cors import CORS

app = Flask(__name__) # Initialize the Flask application 
CORS(app)

def nested_default_float():
    return defaultdict(float)

def nested_default_int():
    return defaultdict(int)

# Given a bigram model and a starting word, predict a sentence using the model.
def generate_sentence(bigram_model, start_word, max_length=20):
    sentence = [start_word]

    current_word = start_word
    for _ in range(max_length):
        next_word_candidates = list(bigram_model[current_word].keys())
        if not next_word_candidates:
            break
        next_word = random.choice(next_word_candidates)
        sentence.append(next_word)
        current_word = next_word

    return ' '.join(sentence)

# Given a trigram model and a starting word, predict a sentence using the model.
def generate_sentence_trigram(trigram_model, start_word1, start_word2, max_length=20):
    sentence = [start_word1, start_word2]

    current_word1 = start_word1
    current_word2 = start_word2

    for _ in range(len(trigram_model.keys())):
        next_word_candidates = list(trigram_model[(current_word1,
                                                   current_word2)].keys())
        if not next_word_candidates:
            break
        next_word = random.choice(next_word_candidates)
        sentence.append(next_word)
        current_word1 = current_word2
        current_word2 = next_word

    return ' '.join(sentence)

# Load the pre-trained model
bigram_model = pickle.load(open('bigram_model.pkl', 'rb'))
trigram_model = pickle.load(open('trigram_model.pkl', 'rb'))

@app.route('/')
def home(): 
  return 'Hello, World! This is the ML model API.'

@app.route('/predict/bigram', methods=['POST'])
def predict_bigram():
  #Get the data from POST request
  data = request.get_json(force=True)

  # Ensure that we received the expected array of features
  try: 
    word = data['word']
  except KeyError: 
    return jsonify(error="The 'word' key is missing from the request payload. Add a single word and the bigram_model will predict the sentence that follows"), 400

  # Convert features into the right format and make a prediction 
  prediction = generate_sentence(bigram_model, word, )

  # Return the prediction 
  return jsonify(prediction=prediction)

@app.route('/predict/trigram', methods=['POST'])
def predict_trigram():
  #Get the data from POST request
  data = request.get_json(force=True)

  # Ensure that we received the expected array of features
  try: 
    word1 = data['word1']
  except KeyError: 
    return jsonify(error="The 'word1' key is missing from the request payload. Add a single word and the bigram_model will predict the sentence that follows"), 400

  try: 
    word2 = data['word2']
  except KeyError: 
    return jsonify(error="The 'word2' key is missing from the request payload. Add a single word and the bigram_model will predict the sentence that follows"), 400

  # Convert features into the right format and make a prediction 
  prediction = generate_sentence_trigram(trigram_model, word1, word2)

  # Return the prediction 
  return jsonify(prediction=prediction)

@app.route('/check_server_status', methods=['GET']) 
def check_server_status(): 
    return 'Server is active'; 

if __name__ == '__main__':
    app.run(debug=True)
    