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
from transformers import BertTokenizer

# Load the configuration file
try: 
    with open('config.json') as f: 
        config = json.load(f) 
        embed_size = config['embed_size']
        num_classes = config['num_classes']
        max_pool = config['max_pool']
        seq_len = config['seq_len']
        batch_size = config['batch_size']
        epochs = config['epochs']
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
        #print("Shape of transformed:") 
        #print(transformed.shape)
        
        # Apply global max pooling if specified, else use the last token's output
        if self.pool:
            pooled = F.max_pool1d(transformed.transpose(1,2), kernel_size=transformed.shape[1]).squeeze(2)
        else:
            pooled = transformed[:, -1, :]
        # Pass through the output layer to get class logits
        out = self.fc(pooled)

        return out
    
# Define a simple function to preprocess and tokenize the reviews
def preprocess_text(text, tokenizer, max_length):
    # Tokenize the text
    tokens = tokenizer.encode_plus(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return tokens

# A function to create PyTorch datasets
def create_dataset(df, tokenizer, max_length):
    dataset = []
    for index, row in df.iterrows():
        review = row['review']
        label = row['sentiment']
        
        # Preprocess text and tokenize review
        tokens = preprocess_text(review, tokenizer, max_length)
        
        # Convert label to numerical format (assuming binary classification)
        label_numeric = 1 if label == 'positive' else 0
        
        # Convert tokenized review to tensor
        input_tensor = {
            'input_ids': torch.tensor(tokens['input_ids']),
            'attention_mask': torch.tensor(tokens['attention_mask'])
        }
        
        # Convert label to tensor
        label_tensor = torch.tensor(label_numeric)
        
        # Create sample dictionary with tensors
        sample = {'inputs': input_tensor, 'label': label_tensor}
        
        # Append sample to dataset
        dataset.append(sample)
    
    return dataset

# A function to evaluate the performance of a model. 
def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            print("Type of inputs:", type(inputs))
            print("Type of labels:", type(labels))
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

if __name__ == "__main__":

    # Define the randomly generated input tensor and sentiment labels
    random_input = torch.randn(batch_size, seq_len, embed_size)
    sentiment_labels = torch.randint(0, 2, (batch_size,))

    # Read in IMDB data: 
    df = pd.read_csv('imdb_reviews.csv')
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_length = 256
    
    train_dataset = create_dataset(train_df, tokenizer, max_length)
    print("Train dataset type:", type(train_dataset), "of: ", type(train_dataset[1]))
    test_dataset = create_dataset(test_df, tokenizer, max_length)    
    print("Test dataset type:", type(test_dataset), "of: ", type(test_dataset[1]))
          
    # Define model and train: 
    model = SentimentClassifier(embed_size=embed_size, num_classes=num_classes, max_pool=max_pool)
    
    criterion = nn.CrossEntropyLoss() # Selecting the loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Train loader type:", type(train_loader))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Test loader type:", type(test_loader))
    
    for i in range (0, epochs):
        model.train() # Set model to training mode. 
        print("Epoch " + str(i) + ": ------------------------------- ")
        outputs = model(train_dataset) # Forward pass
        print("First 10 outputs: ", outputs[:10])
        loss = criterion(outputs, sentiment_labels) # Compute the loss
        print("Loss:", loss.item())
        
        # Backpropagation: adjust the model's weights to minimize the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #model.eval()
        #accuracy = evaluate_model(model, test_loader)

    