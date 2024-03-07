import pandas as pd
import tensorflow as tf
import pickle
import nltk 
import numpy as np
from gensim.models import Word2Vec
from collections import deque
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras import losses


print("TensorFlow Version:", tf.__version__)

print("\nHello! Welcome to Ray's RNN model construction Helper, or R'sRRN for short.")
print("I'll ask you a few questions, and give you some options. Let's get started!")

text = []
while (text == [] or (choice < 1 or choice > 4)): 
    print("\nWhat dataset would you like to use?");
    print("\t1. Brown Corpus");
    print("\t2. Alice in Wonderland");
    print("\t3. Twitter ")
    choice = int(input("Please enter the number of your choice: "));
    try:
        if choice == 1:
            text = nltk.corpus.brown.sents() # extract data sentence by sentence
        elif choice == 2:
            text = nltk.corpus.gutenberg.sents('carroll-alice.txt')
        elif choice == 3: 
            text = nltk.corpus.twitter_samples.sents()
        else: 
            print("Sorry, that's not an option on the list. Please try again.")
    except:
        print("Sorry, I couldn't find that file. Please try again, or check the file's location")
        exit(1)
print(text[:3])
sentences = [[word.lower() for word in sentence] for sentence in text]
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count =1, workers = 4)
vocab_size = len(word2vec_model.wv.key_to_index) + 1  # Adding 1 for the padding token
embedding_dim = word2vec_model.wv.vector_size

sequences = [[word2vec_model.wv.key_to_index[word] for word in sentence if word in word2vec_model.wv.key_to_index]
             for sentence in sentences]

# Pad sequences to ensure uniform length
max_sequence_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Create input and output data for the model
X = padded_sequences[:, :-1]  # Input sequences (remove the last word)
y = padded_sequences[:, 1:]   # Output sequences (remove the first word)

#print("\nTokenizing and padding sequences...")
#tokenizer = Tokenizer() # tool we will use to preprocess
##tokenizer.fit_on_texts(text) # fit the tokenizer to the text
#print(X[:10])
#print(y[:10])
#print("\nExtracting sequences from text...")
#X_sequences = tokenizer.texts_to_sequences(X) 

#print("\nPadding Sequences..")
#max_sequence_length = max([len(seq) for seq in sequences]) # find the length of the longest sequence  
#X_padded = pad_sequences(sequences, maxlen = max_sequence_length, padding='post') # pad the sequences to the same length

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, idx in word2vec_model.wv.key_to_index.items():
    embedding_matrix[idx] = word2vec_model.wv[word]
    
print("\nWhat would you like the embedding dimension to be?")
embed_dim = int(input("Please enter a number: "));

activation_functions = ["sigmoid", "tanh", "relu"];
activation_function = 0;
while (activation_function < 1 or activation_function > 3):
    print("\nWhich activation function would you like to use?");
    print("\t1. Sigmoid");
    print("\t2. Hyperbolic Tangent");
    print("\t3. Rectified Linear Unite (ReLU)")
    activation_function = int(input("Please enter a number: "));
    if (activation_function < 1 or activation_function > 3):
        print("Sorry, that's not an option on the list. Please try again."); 

recurring = 0;
while (recurring < 1 or recurring > 2):
    print("\nWould you like to make this a Recurrent Neural Network (Activate RNN layer)?");
    print("\t1. Yes");
    print("\t2. No");
    recurring = int(input("Please enter a number: "));
    if (recurring < 1 or recurring > 2):
        print("Sorry, that's not an option on the list. Please try again.");
    

model = None; 
if recurring == 1: 
    model = tf.keras.models.Sequential([
        layers.Embedding(input_dim = vocab_size, 
                         output_dim = embedding_dim, 
                         weights=[embedding_matrix], 
                         input_length=max_sequence_length-1, 
                         trainable=False),
        layers.SimpleRNN(embed_dim, return_sequences=True), # Added the simpleRNN layer to the original basic model from assignment 3
        layers.Dropout(0.2), 
        layers.GlobalAveragePooling1D(), 
        layers.Dense(1)
    ])
else: 
    model = tf.keras.models.Sequential([
        layers.Embedding(input_dim = vocab_size, 
                         output_dim = embedding_dim, 
                         weights=[embedding_matrix], 
                         input_length=max_sequence_length-1, 
                         trainable=False),
        layers.Dropout(0.2), 
        layers.GlobalAveragePooling1D(), 
        layers.Dense(1)
])

 
loss = 0
while (loss < 1 or loss > 2):
    print("\nLoss function?");
    print("\t1. Binary Cross Entropy (keras)");
    print("\t2. Means Squared Error (keras)");
    loss = int(input("Please enter a number: "));
    if (loss < 1 or loss > 2):
        print("Sorry, that's not an option on the list. Please try again.");


optimizers = ["adam", "sgd", "adagrad", "rmsprop", "adadelta", "adamax", "nadam", "ftrl"];
optimizer = 0; 
while (optimizer < 1 or optimizer > 8): 
    print("\nOptimizer?");
    print("\t1. Adam");
    print("\t2. Stochastic Gradient Descent");
    print("\t3. Adagrad");
    print("\t4. RMSprop");
    print("\t5. Adadelta");
    print("\t6. Adamax");
    print("\t7. Nadam");
    print("\t8. Ftrl");
    optimizer = int(input("Please enter a number: "));
    if (optimizer < 1 or optimizer > 8):
        print("Sorry, that's not an option on the list. Please try again.");
        
if (optimizer == 1):
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True), 
                  optimizer=optimizers[optimizer-1], 
                  metrics=['accuracy']);
else: 
    model.compile(loss=losses.MeanSquaredError(),
                  optimizer=optimizers[optimizer-1],
                  metrics=['accuracy']);

epochs = 0; 
while (epochs < 1):
    epochs = int(input("\nHow many epochs would you like to use? Please enter a number: "));
    if (epochs < 1):
        print("Sorry, that's not a valid number. Please try again.");

print("\nFinally, what would you like to name the model? I will add the proper .pkl extension after, so be sure not to include .pkl at the end!");
model_name = input("Please enter a name: ");

model.fit(X_train, X_train, epochs=epochs, validation_data=(X_test, y_test), verbose=2);
pickle.dump(model, open(model_name + '.pkl', 'wb')); # dump the model int model.pkl

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2);

print(model_name + ":")
print(model.summary()); 
print("\nAccuracy: ", test_acc);
print("Loss: ", test_loss);

# Define a function to generate a sentence
def generate_sentence(model, seed_word, max_length=10):
    generated_sentence = [seed_word]
    for _ in range(max_length):
        # Tokenize the seed word
        encoded = [[word2vec_model.wv.key_to_index[word]] for word in generated_sentence]
        encoded = pad_sequences(encoded, maxlen=max_sequence_length-1, padding='pre')
        # Predict the next word
        predicted_index = model.predict_classes(encoded, verbose=0)
        predicted_word = ""
        for word, index in word2vec_model.wv.key_to_index.items():
            if index == predicted_index:
                predicted_word = word
                break
        # Append the predicted word to the sentence
        generated_sentence.append(predicted_word)
        # If the predicted word is a punctuation mark or the end of a sentence, stop generating
        if predicted_word in ['.', '!', '?']:
            break
    return ' '.join(generated_sentence)

# Use the function to generate a sentence
seed_word = "the"
generated_sentence = generate_sentence(model, seed_word)
print("Generated Sentence:", generated_sentence)

print("\nModel has been saved as", model_name + ".pkl. Thank you for using R'sRRN! Goodbye!");
