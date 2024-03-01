import pandas as pd
import tensorflow as tf
import pickle
import nltk 
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras import losses


print("TensorFlow Version:", tf.__version__)

print("\nHello! Welcome to Ray's RNN model construction Helper, or R'sRRN for short.")
print("I'll ask you a few questions, and give you some options. Let's get started!")

text = []
while (text == [] or (choice < 1 or choice > 3)): 
    print("\nWhat dataset would you like to use?");
    print("\t1. Brown Corpus");
    print("\t2. Alice in Wonderland");
    print("\t3. Twitter ")
    choice = int(input("Please enter the number of your choice: "));
    try:
        if choice == 1:
            text = nltk.corpus.brown.raw()
        elif choice == 2:
            text = nltk.corpus.gutenberg.raw('carroll-alice.txt')
        elif choice == 3: 
            text = nltk.corpus.twitter_samples.raw()
        else: 
            print("Sorry, that's not an option on the list. Please try again.")
    except:
        print("Sorry, I couldn't find that file. Please try again, or check the file's location")
        exit(1)
print(text[:50])

print("\nTokenizing and padding sequences...")
tokenizer = Tokenizer() # tool we will use to preprocess
tokenizer.fit_on_texts(text) # fit the tokenizer to the text
total_words = len(tokenizer.word_index) + 1

print("\nExtracting sequences from text...")
sequences = np.array([], dtype=float)
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        sequences = np.append(sequences, n_gram_sequence)
        
# Pad sequences
max_sequence_length = max([len(seq) for seq in sequences]) # find the length of the longest sequence
padded_sequences = pad_sequences(sequences, maxlen = max_sequence_length, padding='post') # pad the sequences to the same length

# Create predictors and label
X, y = sequences[:, :-1], sequences[:, -1]

# One-hot encode the labels. 
y = np.eye(total_words)[y]


print("\nSplitting data into training and testing sets ... ", end= "")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


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

class MyDenseLayer(tf.keras.layers.Layer): 
    def __init__(self, num_outputs):
        super(MyDenseLayer, self).__init__()
        self.W = self.add(weight_variable([input_dim, output_dim]));
        self.b = self.add_weight([1, output_dim]); 
        
    def call(self, inputs):
        z = tf.matmul(inputs, self.W) + self.b;
        
        # Select which activation function based on user input
        if (activation_function == 1):
            output = tf.math.sigmoid(z);  
        elif (activation_function == 2):
            output = tf.math.tanh(z);
        elif (activation_function == 3):
            output = tf.nn.relu(z);
        else: 
            print("there is no activation function corresponding to not an option-- defaulting to Sigmoid :)");
            output = tf.math.sigmoid(z);
            
        return output;

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
        layers.Embedding(input_dim = len(tokenizer.word_index) + 1, 
                         output_dim = embed_dim, 
                         input_length = max_sequence_length),
        layers.SimpleRNN(embed_dim, return_sequences=True), # Added the simpleRNN layer to the original basic model from assignment 3
        layers.Dropout(0.2), 
        layers.GlobalAveragePooling1D(), 
        layers.Dense(1)
    ])
else: 
    model = tf.keras.models.Sequential([
        layers.Embedding(input_dim = len(tokenizer.word_index) + 1, 
                         output_dim = embed_dim, 
                         input_length = max_sequence_length),
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

model.fit(train_data, train_labels, epochs=epochs, validation_data=(test_data, test_labels), verbose=2);
pickle.dump(model, open(model_name + '.pkl', 'wb')); # dump the model int model.pkl

test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2);

print(model_name + ":")
print(model.summary()); 
print("\nAccuracy: ", test_acc);
print("Loss: ", test_loss);
# Assuming the model is already defined and named 'model'

# Define the generate_text function
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

# Assuming 'tokenizer' and 'max_sequence_len' are also defined somewhere earlier in the code

# Generate sentence
starting_word = "The"
generated_sentence = generate_text(model, tokenizer, max_sequence_length, starting_word)
print("Generated Sentence:", generated_sentence)g

print("\nModel has been saved as", model_name + ".pkl. Thank you for using R'sRRN! Goodbye!");
