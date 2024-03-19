import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as f
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, embed_sz, max_len=512):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_sz, 2).float() * -(np.log(10000.0) / embed_sz))
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, embed_sz))
        self.positional_encoding[:, :, 0::2] = torch.sin(position * div_term)
        self.positional_encoding[:, :, 1::2] = torch.cos(position * div_term)

    def forward(self, x_f):
        return x_f + self.positional_encoding[:, :x_f.size(1)].detach()


class MultiheadAttention(nn.Module):
    def __init__(self, embed_sz, num_head):
        super(MultiheadAttention, self).__init__()
        self.num_head = num_head
        self.head_dim = embed_sz // num_head
        assert (
            self.head_dim * num_head == embed_sz
        ), "Embedding size needs to be divisible by num_heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_head * self.head_dim, embed_sz)

    def forward(self, values, keys, query, mask):
        # Get number of training examples
        n = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.num_heads different pieces
        values = values.reshape(n, value_len, self.num_head, self.head_dim)
        keys = keys.reshape(n, key_len, self.num_head, self.head_dim)
        queries = query.reshape(n, query_len, self.num_head, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.nn.functional.softmax(energy / (self.embed_sz ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            n, query_len, self.num_head * self.head_dim
        )
        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_sz, num_head, forward_expan):
        super(TransformerBlock, self).__init__()
        self.attention = MultiheadAttention(embed_sz, num_head)
        self.norm1 = nn.LayerNorm(embed_sz)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_sz, forward_expan * embed_sz),
            nn.ReLU(),
            nn.Linear(forward_expan * embed_sz, embed_sz),
        )
        self.norm2 = nn.LayerNorm(embed_sz)

    def forward(self, x_f, mask):
        attention = self.attention(x_f, x_f, x_f, mask)
        x_f = self.norm1(attention + x_f)
        forward = self.feed_forward(x_f)
        out = self.norm2(forward + x_f)
        return out


class SentimentClassifier(nn.Module):
    def __init__(self, embed_sz, num_classe, max_poo, num_head, forward_expan):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(1, embed_sz, sparse=False)
        self.positional_encoding = PositionalEncoding(embed_sz)
        self.transformer = TransformerBlock(embed_sz, num_head, forward_expan)
        self.pool = max_poo
        self.fc = nn.Linear(embed_sz, num_classe)

    def forward(self, x_f, mask):
        x_f = self.embedding(x_f)
        x_f = x_f + self.positional_encoding(x_f)
        x_f = self.transformer(x_f, mask)
        if self.pool:
            x_f = f.max_pool1d(x_f.transpose(1, 2), kernel_size=x_f.shape[1]).squeeze(2)
        else:
            x_f = x_f[:, -1, :]
        out = self.fc(x_f)
        return out


class SentimentDataset(Dataset):
    def __init__(self, reviews, label):
        self.reviews = reviews
        self.labels = label

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        # Map string labels to integers (assuming 'negative' is 0 and 'positive' is 1)
        label_mapping = {'negative': 0, 'positive': 1}

        return {
            'input_ids': self.reviews[idx]['input_ids'],
            'attention_mask': self.reviews[idx]['attention_mask'],
            'labels': torch.tensor(label_mapping[self.labels[idx]], dtype=torch.long)
        }


def tokenize_text(text):
    return tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors='pt')


def create_attention_mask(input_id):
    # define masking for attention (to ignore padded tokens during training)
    return (input_id != tokenizer.pad_token_id).float()


# Parameters
embed_size = 768  # BERT embedding size
num_classes = 2
max_pool = True
num_heads = 8
forward_expansion = 4
seq_len = 128
batch_size = 64

# Import Dataset for IMDB reviews
df = pd.read_csv('IMDB Dataset.csv', encoding='utf-8')

# Preprocess/tokenize the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
df['tokenized_reviews'] = df['review'].apply(tokenize_text)

# Split the dataset into training and validation sets
train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

# convert dataframe to pytorch dataset
train_dataset = SentimentDataset(reviews=train_df['tokenized_reviews'].tolist(), label=train_df['sentiment'].tolist())
valid_dataset = SentimentDataset(reviews=valid_df['tokenized_reviews'].tolist(), label=valid_df['sentiment'].tolist())
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 5

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    # Use tqdm to create a progress bar
    train_dataloader_with_progress = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')

    for batch in train_dataloader:
        input_ids = batch['input_ids'].squeeze(1)
        attention_mask = batch['attention_mask'].squeeze(1)
        labels = batch['labels']

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)
        correct_predictions += torch.sum(predictions == labels).item()
        total_samples += labels.size(0)

    average_loss = total_loss / len(train_dataloader)
    accuracy = correct_predictions / total_samples

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_loss:.4f}, Train Accuracy: {accuracy:.4f}')

    # Validation
    model.eval()
    total_samples = 0
    correct_predictions = 0

    for batch in tqdm(valid_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs} Validation', unit='batch'):
        input_ids = batch['input_ids'].squeeze(1)
        attention_mask = batch['attention_mask'].squeeze(1)
        labels = batch['labels']

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        correct_predictions += torch.sum(predictions == labels).item()
        total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    print(f'Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {accuracy:.4f}')

# Optionally, save the trained model
torch.save(model.state_dict(), 'sentiment_model.pth')
