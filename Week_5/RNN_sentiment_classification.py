import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import layers
from tensorflow.keras import losses


print("TensorFlow Version:", tf.__version__)

print("\nHello! Welcome to Ray's RNN model construction Helper, or R'sRRN for short.")
print("I'll ask you a few questions, and give you some options. Let's get started!")
train_df = pd.DataFrame()
while (train_df.empty or (choice < 1 or choice > 3)): 
    print("\nWhat dataset would you like to use?");
    print("\t1. IMDB Reviews");
    print("\t2. Financial Sentiment Data");
    print("\t3. Twitter Data")
    choice = int(input("Please enter the number of your choice: "));
    train_df = ""; 
    validation_df = "";
    try:
        if choice == 1:
            train_df = pd.read_csv('imdb_reviews.csv')
        elif choice == 2:
            train_df = pd.read_csv('financial_sentiments.csv')
        elif choice == 3: 
            train_df = pd.read_csv('twitter/twitter_training.csv')
            validation_df = pd.read_csv('twitter/twitter_validation.csv')
        else: 
            print("Sorry, that's not an option on the list. Please try again.")
    except:
        print("Sorry, I couldn't find that file. Please try again, or check the file's location")
        exit(1);
print(train_df.head())

dataset_columns = ['review', 'Sentence', 'text']

print("\nTokenizing and padding sequences...")
tokenizer = Tokenizer() # tool we will use to preprocess
tokenizer.fit_on_texts(train_df[dataset_columns[choice-1]]) # fit the tokenizer to the text
sequences = tokenizer.texts_to_sequences(train_df[dataset_columns[choice-1]]) # convert the text to sequences    
max_sequence_length = max([len(seq) for seq in sequences]) # find the length of the longest sequence
padded_sequences = pad_sequences(sequences, maxlen = max_sequence_length, padding='post') # pad the sequences to the same length


# convert the target classifications to numeric labels 
label_mapping = {'positive': 1, 'negative': 0}
numeric_labels = train_df['sentiment'].map(label_mapping) # map the labels to the numeric values


print("\nSplitting data into training and testing sets ... ", end= "")
#Split the data into training and testing sets
if choice <= 2: #If using IMDB or Financial data, split the given data into training and testing. 
    train_data, test_data, train_labels, test_labels = train_test_split(padded_sequences, 
                                                                        numeric_labels, 
                                                                        test_size=0.2, 
                                                                        random_state=42)
else: # choice is 3, using Twitter Data. 
    train_data = tokenizer.fit_on_texts(validation_df['text']) # fit the tokenizer to the text
    sequences = tokenizer.texts_to_sequences(validation_df['text']) # convert the text to sequences
    padded_sequences = pad_sequences(sequences, maxlen = max_sequence_length, padding='post')
    numeric_labels = validation_df['sentiment'].map(label_mapping) # map the labels to the numeric values

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

texts = [
    "I love this movie. It's the best movie I've ever seen!",
    "This movie is terrible. I would not recommend it to anyone."
]

tokenizer = Tokenizer(num_words=1000) 
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Convert padded sequences to numpy array
X_pred = np.array(padded_sequences)

positive_predictions = model.predict(X_pred);
negative_predictions = model.predict(X_pred);
predicted_sentiment_pos = "Positive" if positive_predictions[0] > 0.5 else "Negative"
predicted_sentment_neg = "Positive" if positive_predictions[0] > 0.5 else "Negative"

print("\nModel has been saved as", model_name + ".pkl. Thank you for using R'sRRN! Goodbye!");
