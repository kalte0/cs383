# We will use the same model as in the previous example, but we will add positional encoding to the input embeddings.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Positional encoding function
def positional_encoding(seq_len, embed_size):
    position = np.arange(0, seq_len, dtype=np.float32)
    div_term = np.exp(np.arange(0, embed_size, 2) * -(np.log(10000.0) / embed_size))
    pos_encoding = np.zeros((seq_len, embed_size))
    pos_encoding[:, 0::2] = np.sin(position[:, np.newaxis] * div_term)
    pos_encoding[:, 1::2] = np.cos(position[:, np.newaxis] * div_term)
    return torch.tensor(pos_encoding)

class SelfAttention(nn.Module):
    def __init__(self, embed_size=128):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.embed_size)
        attention = torch.softmax(attention_scores, dim=-1)
        out = torch.matmul(attention, V)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, forward_expansion, embed_size=128):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        attention = self.attention(x)
        x = self.norm1(attention + x)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out

class SentimentClassifier(nn.Module):
    def __init__(self, num_classes, max_pool=True, embed_size=128, seq_len=100):
        super(SentimentClassifier, self).__init__()
        self.transformer = TransformerBlock(embed_size=embed_size, forward_expansion=4)
        self.pool = max_pool
        self.fc = nn.Linear(embed_size, num_classes)
        self.pos_encoding = positional_encoding(seq_len, embed_size).float() # Cast to float 

    def forward(self, x):
        # Add positional encoding to the input embeddings
        x = x + self.pos_encoding.to(x.device)
        
        transformed = self.transformer(x)
        if self.pool:
            pooled = F.max_pool1d(transformed.transpose(1, 2), kernel_size=transformed.shape[1]).squeeze(2)
        else:
            pooled = transformed[:, -1, :]
        out = self.fc(pooled)
        return out


if __name__ == '__main__':
    model = SentimentClassifier(embed_size=128, num_classes=2, max_pool=True)
    x = torch.rand(64, 100, 128)
    output = model(x) 
    print(output)
    
    probabilities = F.softmax(output, dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)
    print(probabilities)
    print(predicted_classes)

