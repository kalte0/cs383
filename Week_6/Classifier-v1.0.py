# Basic imports
import torch # pytorch library
import torch.nn as nn # neural network module
import torch.optim as optim # optimization module
import numpy as np # numpy library
from torch.utils.data import DataLoader, Dataset # data loading and processing
import torch.nn.functional as F # functional module

embed_size = 128

class SelfAttention(nn.Module): 
    embed_size = 128
    def __init__(self, embed_size=128):
        super(SelfAttention, self).__init__() 
        # Initialize linear layers for queries, keys, and values
        self.query = nn.Linear(embed_size, embed_size) # Q
        self.key = nn.Linear(embed_size, embed_size) # K
        self.value = nn.Linear(embed_size, embed_size) # V

    def forward(self, x):
        # Generate queries, keys, and values by applying the respective linear transformations
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Calculate attention scores by performing scaled dot-product attention
        # Scaling by sqrt(embed_size) for stability
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embed_size)

        # Apply softmax to get probabilities, ensuring that scores s um to 1
        attention = torch.softmax(attention_scores, dim=-1)

        # Multiply the attention scores with the values to get a weighted sum,
        # which signifies the output of the self-attention layer
        out = torch.matmul(attention, V)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, forward_expansion, embed_size=128):
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
    def __init__(self, num_classes, max_pool=True, embed_size=128):
        super(SentimentClassifier, self).__init__()
        # Transformer block (acting as the encoder)
        self.transformer = TransformerBlock(embed_size=embed_size, forward_expansion=4)
        self.pool = max_pool
        # Output fully connected layer for classification
        self.fc = nn.Linear(embed_size, num_classes)

    def forward(self, x):
        # Pass input through the transformer block
        transformed = self.transformer(x)
        # Apply global max pooling if specified, else use the last token's output
        if self.pool:
            pooled = F.max_pool1d(transformed.transpose(1,2), kernel_size=transformed.shape[1]).squeeze(2)
        else:
            pooled = transformed[:, -1, :]
        # Pass through the output layer to get class logits
        out = self.fc(pooled)
        return out
    
    
if __name__ == '__main__':
    # Example usage
    model = SentimentClassifier(embed_size=128, num_classes=2, max_pool=True)
    x = torch.rand(64, 100, 128)
    output = model(x) 
    print(output)
    
    probabilities = F.softmax(output, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)
    print(probabilities)
    print(predicted_classes)

  