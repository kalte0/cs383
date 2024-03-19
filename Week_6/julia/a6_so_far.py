import torch.nn.functional as f
from transformers import BertTokenizerFast
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import nltk


class PositionalEncoding(nn.Module):
    def __init__(self, embed_sz, max_len=512):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_sz, 2).float() * -(np.log(10000.0) / embed_sz))
        positional_encoding = torch.zeros(1, max_len, embed_sz, requires_grad=False)  # Detach here
        positional_encoding[:, :, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, :, 1::2] = torch.cos(position * div_term)
        self.positional_encoding = nn.Parameter(positional_encoding)

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
    def __init__(self, embed_sz, num_classe, max_poo, num_head, forward_expan, tokenizer):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.EmbeddingBag(1, embed_sz, sparse=False)
        self.positional_encoding = PositionalEncoding(embed_sz)
        self.transformer = TransformerBlock(embed_sz, num_head, forward_expan)
        self.pool = max_poo
        self.fc = nn.Linear(embed_sz, num_classe)
        self.tokenizer = tokenizer

    def forward(self, x_f, msk):
        print("X_F: ", x_f)
        x_f = self.embedding(x_f)
        x_f = x_f + self.positional_encoding(x_f)
        x_f = self.transformer(x_f, msk)
        if self.pool:
            x_f = f.max_pool1d(x_f.transpose(1, 2), kernel_size=x_f.shape[1]).squeeze(2)
        else:
            x_f = x_f[:, -1, :]
        out = self.fc(x_f)
        return out


def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return tokens


def build_tokenizer(tokens):
    # Build tokenizer from tokens
    tokenizer = {token: idx + 1 for idx, token in enumerate(set(tokens))}
    # Add OOV token
    tokenizer['<OOV>'] = len(tokenizer) + 1
    return tokenizer


class SentimentDataset(Dataset):
    def __init__(self, review, label, max_len=512):
        self.reviews = review
        self.labels = label
        self.max_len = max_len
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.labels)
        self.tokenizer = custom_tokenizer  # Initialize tokenizer

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        text = self.reviews[idx]
        label = self.labels[idx]

        # Tokenize text using custom tokenizer
        tokenized_text = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len,
                                        return_tensors="pt")

        # Extract input_ids and attention_mask from tokenized text
        input_id = tokenized_text["input_ids"].squeeze(0)
        attention_mas = tokenized_text["attention_mask"].squeeze(0)

        # Convert label to tensor
        label_tenso = torch.tensor(self.label_encoder.transform([label])[0], dtype=torch.long).to(device)
        print(f"Text: {text}")
        print(f"Input IDs: {input_id}")
        return input_id, attention_mas, label_tenso


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
print("head: ", df.head())

# Prepare dataset
reviews = df['review'].tolist()
labels = df['sentiment'].tolist()

# Split dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(reviews, labels, test_size=0.2, random_state=42)

# Load the base tokenizer
base_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Fine-tune the tokenizer with a larger vocabulary
larger_vocab_size = 60000  # Choose the desired vocabulary size
custom_tokenizer = base_tokenizer.train_new_from_iterator(
    text_iterator=train_texts,  # Assuming train_texts is your training data
    vocab_size=larger_vocab_size,
    min_frequency=2  # Adjust as needed
)

# convert dataframe to pytorch dataset
train_dataset = SentimentDataset(review=train_texts, label=train_labels)
valid_dataset = SentimentDataset(review=val_texts, label=val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
print("train dataset: ", train_dataset)
print("train dataloader: ", train_dataloader)

# model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)
model = SentimentClassifier(embed_size, num_classes, max_pool, num_heads, forward_expansion, custom_tokenizer)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
num_epochs = 5
print("Starting Training...")

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    for batch_inputs in train_dataloader:
        optimizer.zero_grad()

        # Unpack the batch inputs
        input_ids, attention_mask, labels_tensor = batch_inputs

        # Move tensors to the appropriate device and convert to float
        text_tensor = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label_tensor = labels_tensor.to(device)

        batch_outputs = model(labels_tensor, attention_mask)  # Pass mask to model

        # Calculate loss and perform backpropagation
        loss = criterion(batch_outputs, labels_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(batch_outputs, 1)
        correct_predictions += torch.sum((predicted == label_tensor).int()).item()
        total_samples += label_tensor.size(0)

    average_loss = total_loss / len(train_dataloader)
    accuracy = correct_predictions / total_samples

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_loss:.4f}, Train Accuracy: {accuracy:.4f}')

"""    for batch in train_dataloader:
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
"""
