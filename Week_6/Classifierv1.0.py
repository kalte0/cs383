import torch # torch library
import torch.nn as nn # neural network module
import torch.optim as optim # optimization module
import numpy as np # numpy library
from torch.utils.data import DataLoader, Dataset #  data loading and processing
import torch.nn.functional as F # functional module
import pandas as pd # pandas library
import json # json library
import torchtext # torchtext library
from tensorflow.keras.preprocessing.text import Tokenizer # keras text preprocessing
from tensorflow.keras.preprocessing.sequence import pad_sequences # keras sequence padding
from sklearn.model_selection import train_test_split # sklearn train test split
from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator

# Load the configuration file
try: 
    with open('config.json') as f: 
        config = json.load(f) 
        embed_size = config['embed_size']
        num_classes = config['num_classes']
        max_pool = config['max_pool']
        seq_len = config['seq_len']
        batch_size = config['batch_size']
except (FileNotFoundError): 
    print('config.json not found')
    exit()

class SelfAttention(nn.Module): 
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        # Initialize linear layers for queries, keys, and values
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # Generate queries, keys, and values by applying the respective linear transformations
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores by performing scaled dot-product attention
        # Scaling by sqrt(embed_size) for stability
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(embed_size)

        # Apply softmax to get probabilities, ensuring that scores sum to 1
        attention = torch.softmax(attention_scores, dim=-1)

        # Multiply the attention scores with the values to get a weighted sum,
        # which signifies the output of the self-attention layer
        out = torch.matmul(attention, V)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, forward_expansion):
        super(TransformerBlock, self).__init__()
        # Self-attention layer
        self.attention = SelfAttention(embed_size)
        # Normalization layer 1
        self.norm1 = nn.LayerNorm(embed_size)
        # Feedforward layers
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        # Normalization layer 2
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        # Apply self-attention
        attention = self.attention(x)
        # Add & normalize (residual connection)
        x = self.norm1(attention + x)
        # Apply feedforward layers
        forward = self.feed_forward(x)
        # Add & normalize (residual connection)
        out = self.norm2(forward + x)
        return out
    
class SentimentClassifier(nn.Module):
    def __init__(self, embed_size, num_classes, max_pool=True):
        super(SentimentClassifier, self).__init__()
        # Transformer block (acting as the encoder)
        self.transformer = TransformerBlock(embed_size, forward_expansion=4)
        self.pool = max_pool
        # Output fully connected layer for classification
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        # Pass input through the transformer block
        transformed = self.transformer(x)
        print("Shape of transformed:") 
        print(transformed.shape)
        
        # Apply global max pooling if specified, else use the last token's output
        if self.pool:
            pooled = F.max_pool1d(transformed.transpose(1,2), kernel_size=transformed.shape[1]).squeeze(2)
        else:
            pooled = transformed[:, -1, :]
        # Pass through the output layer to get class logits
        out = self.fc(pooled)

        return out


if __name__ == "__main__":
    # Simple use: 
    model = SentimentClassifier(embed_size=embed_size, num_classes=num_classes, max_pool=max_pool)
    adjusted_batch_size = 100 #(int)(batch_size/(0.8))
    
    reviews_df = pd.read_csv('imdb_reviews.csv')
    tokenizer = Tokenizer() # tool we will use to preprocess data
    tokenizer.fit_on_texts(reviews_df["review"]) # updates the internal vocabulary of the tokenizer
    sequences = tokenizer.texts_to_sequences(reviews_df['review']) # turn into sequences
    max_sequence_length = max([len(seq) for seq in sequences])
    padded_sequences = pad_sequences(sequences, maxlen=embed_size, padding='post')
    padded_sequences = torch.tensor(padded_sequences, dtype=torch.float32)
    print("padded_sequences", padded_sequences)
    print("Shape of padded_sequences:", padded_sequences.shape)
    #print("tensorized:", torch.tensor(padded_sequences))
    
    # split data into batches: 
    #for i in range(0, len(padded_sequences), adjusted_batch_size):
        #batch = padded_sequences[i:i+adjusted_batch_size]
        #print("Batch shape:", batch.shape)

    # Put batches together into a tensor of shape (500, 100, 128)
    batches = torch.stack([batch for batch in padded_sequences.split(adjusted_batch_size)], dim=0)
    print("Shape of batches:", batches.shape)
    
    #Convert the target classifications to numeric labels
    label_mapping = {'positive': 1.0, 'negative': 0.0}
    numeric_labels = reviews_df['sentiment'].map(label_mapping).tolist() ## tolist makes it a list afterwards. 
    numeric_labels = torch.tensor(numeric_labels, dtype= torch.float32)
    print("numeric_labels", numeric_labels)
    print("Shape of numeric_labels:", numeric_labels.shape) # 50000 long. 

    #Split those labels into batches as well:
    label_batches = torch.stack([batch for batch in numeric_labels.split(adjusted_batch_size)], dim=0)
    print("Shape of label_batches:", label_batches.shape)
    print("label_batches:", label_batches[:10])

    
    # Split the data into training and testing sets
    X_train, X_validation, Y_train, Y_validation = train_test_split(batches,
                                                                    label_batches,
                                                                   test_size=0.2,
                                                                  random_state=42)
    #print(X_train.dtype)
    #print(X_validation.dtype)
    #print(Y_train.dtype)
    #print(Y_validation.dtype)
    #X_train = torch.tensor(X_train, dtype=torch.float32)
    #X_validation = torch.tensor(X_validation, dtype=torch.float32)
    #Y_train = torch.tensor(Y_train, dtype = torch.float32)
    #Y_validation = torch.tensor(Y_validation, dtype = torch.float32)
    
    x = torch.rand(adjusted_batch_size, seq_len, embed_size) # (batch_size, sequence_length, embed_size)
    y = torch.randint(low=0, high=2, size=(adjusted_batch_size,)) # Generate random labels (0 or 1) for sentiment classification
    print("x shape:")
    print(x.shape)
    print("y shape:")
    print(y.shape)
    #print("x:")
    #print(x)
    #print("y:")
    #print(y)
    
        
    d_X_train, d_X_validation, d_Y_train, d_Y_validation = train_test_split(x, y, test_size=0.2, random_state=42)
    print("Shapes of OTHER X_train, X_validation, Y_train, Y_validation:")
    print(d_X_train.shape) 
    print(d_X_validation.shape)
    print(d_Y_train.shape)
    print(d_Y_validation.shape)
    
    print("Shapes of THIS X_train, X_validation, Y_train, Y_validation:")
    print(X_train.shape) 
    print(X_validation.shape)
    print(Y_train.shape)
    print(Y_validation.shape)

    output = model(X_train) # (batch_size, num_classes)
    validation = model(X_validation) # (batch_size / 4, num_classes)
    
    print("output:")
    print(output[:10]) 
    
    probabilities = F.softmax(output, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1) 
    print("probabilities:")
    print("probabilities shape:", probabilities.shape)
    print(probabilities[:10])
    print("predicted_classes:")
    print("predicted_classes shape:", predicted_classes.shape)
    print(predicted_classes[:10])
    
    validation_probabilities = F.softmax(validation, dim=1) 
    validation_predicted_classes = torch.argmax(validation_probabilities, dim=1) 
    print("validation_probabilities:")
    print(validation_probabilities[:10])
    print("validation_predicted_classes:")
    print(validation_predicted_classes[:10])

    Y_train_tensor = torch.tensor(Y_train)
    Y_validation_tensor = torch.tensor(Y_validation)

# Step 3: Calculate Accuracy
def calculate_accuracy(predicted_classes, true_labels):
    print("Shape of predicted_classes tensor:", predicted_classes.shape)
    print("predicted_classes tensor:", predicted_classes)
    print("Shape of true_labels tensor:", true_labels.shape)
    print("true_labels tensor:", true_labels)
    
    correct_predictions = (predicted_classes == true_labels).sum().item()
    total_samples = len(true_labels)
    accuracy = correct_predictions / total_samples
    return accuracy

# For training set
train_accuracy = calculate_accuracy(predicted_classes, Y_train)
print("Training Accuracy:", train_accuracy)

# For validation set
validation_accuracy = calculate_accuracy(validation_predicted_classes, Y_validation_tensor)
print("Validation Accuracy:", validation_accuracy)

    