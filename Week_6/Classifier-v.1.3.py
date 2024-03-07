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

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        assert embed_size % num_heads == 0, "Embedding size must be divisible by number of heads"
        self.head_dim = embed_size // num_heads
        
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Split the embedding into num_heads
        Q = self.query(query)
        K = self.key(key)
        V = self.value(value)
        
        # Reshape Q, K, and V
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # Transpose to get dimensions batch_size x num_heads x seq_len x head_dim
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        
        # Compute attention scores
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))
        
        attention = torch.softmax(energy, dim=-1)
        
        # Apply attention to value
        out = torch.matmul(attention, V)
        
        # Reshape and concatenate heads
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(batch_size, -1, self.embed_size)
        
        # Apply final linear layer
        out = self.fc_out(out)
        
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, forward_expansion, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention = self.attention(x, x, x, mask)
        x = self.dropout(self.norm1(attention + x))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class SentimentClassifier(nn.Module):
    def __init__(self, num_classes, max_pool=True, embed_size=128, num_heads=8, forward_expansion=4, dropout=0.1):
        super(SentimentClassifier, self).__init__()
        self.transformer_block = TransformerBlock(embed_size, num_heads, forward_expansion, dropout)
        self.pool = max_pool
        self.fc = nn.Linear(embed_size, num_classes)
        self.pos_encoding = positional_encoding(seq_len, embed_size).float() # Cast to float -- had to add, because the size of the positional encoding was off, was throwing an error. 

    def forward(self, x, mask=None):
        # Add positional encoding to the input embeddings
        x = x + self.pos_encoding.to(x.device)
        
        transformed = self.transformer_block(x, mask)
        if self.pool:
            pooled = F.max_pool1d(transformed.transpose(1, 2), kernel_size=transformed.shape[1]).squeeze(2)
        else:
            pooled = transformed[:, -1, :]
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
