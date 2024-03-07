import numpy as np
import pickle
from gensim.models import Word2Vec
from nltk.corpus import brown
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SimpleRNN, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from random import randint

news_sentences = brown.sents(categories = 'news') # import the brown dataset for news

# Train Word2Vec model
word2vec_model = Word2Vec(news_sentences, 
                          vector_size=100, 
                          window=5, 
                          min_count=1, 
                          workers=4) 

# Tokenize and encode sentences
word2idx = {word: idx + 1 for idx, word in enumerate(word2vec_model.wv.index_to_key)}
idx2word = {idx: word for word, idx in word2idx.items()}

sequences = []
for sentence in news_sentences:
    sequence = [word2idx[word] for word in sentence if word in word2idx]
    sequences.append(sequence)

# Pad sequences
max_sequence_len = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_sequence_len, padding='pre')

# Prepare X and y
X = sequences[:, :-1]
y = sequences[:, -1]
y = to_categorical(y, num_classes=len(word2idx) + 1)

# Split into train and test data. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # should come after processing the sequences but before compiling the model. 

embedding_dim = 32; # 64 is default. 

# Define RNN model
model = Sequential()
print(len(word2idx))
model.add(Embedding(input_dim= len(word2idx) + 1, 
                    input_length=max_sequence_len-1,
                    output_dim= embedding_dim)) # 101 for brown news
print("Shape after Embedding layer:", model.output_shape) 
model.add(Dropout(0.2)) # to prevent overfitting
print("Shape after Dropout layer:", model.output_shape) 
model.add(SimpleRNN(64)) # 150 is the number of neurons in the LSTM layer
#model.add(Flatten()); 
model.add(Dense(len(word2idx) + 1, activation='softmax'))
print("Shape after Dense layer:", model.output_shape) 

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print('model compiled!')

print('\n Training model...')
# Train model
print(X_train.shape)
print(y_train.shape)
model.fit(X_train, y_train, epochs=5, verbose=1)

# Generate a sentence
def generate_sentence(model, word2idx, idx2word, max_sequence_len):
    seed_text = []
    for _ in range(max_sequence_len):
        seed_text.append(idx2word[randint(1, len(word2idx))])

    generated_text = ' '.join(seed_text)

    for _ in range(20):  # Generate 20 words
        encoded = [word2idx[word] for word in seed_text if word in word2idx]
        encoded = pad_sequences([encoded], maxlen=max_sequence_len-1, padding='pre')
        y_pred = np.argmax(model.predict(encoded), axis=-1)
        next_word = idx2word[y_pred[0]]
        seed_text.append(next_word)
        generated_text += ' ' + next_word
    return generated_text

loss, accuracy = model.evaluate(X_test, y_test, verbose=0) # evaluate based on unseen data. 
print("Loss on Test set:", loss)
print("Accuracy on Test set:", accuracy)
train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0) # also evaluate based on the given data for funsies!
print("Loss on Training set:", train_loss)
print("Accuracy on Training set:", train_accuracy)

generated_sentence = generate_sentence(model, word2idx, idx2word, max_sequence_len)
print("Generated Sentence:", generated_sentence)

pickle.dump(model, open('rnn_model.pkl', 'wb')) # dump the model int model.pkl
