import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AdamW, get_linear_schedule_with_warmup
import nltk
from nltk.corpus import gutenberg
import random
# Ray: I'm copying these in, but we will want to go through and remove extraneous imports before we submit. 

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

dataset = load_dataset("ehovy/race", "all")
dataset_features = ['example_id', 'article', 'answer', 'question', 'options']
train_examples = dataset['train']  # Access the training examples


# List to hold tokenized and embedded representations of inputs
embedded_inputs = []

# Process each example in the training dataset
for example in train_examples:
    article = example['article']
    question = example['question']
    options = example['options']

    # Tokenize article, question, and options with padding
    tokenized_article = tokenizer(article, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    tokenized_question = tokenizer(question, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    tokenized_options = tokenizer(options, padding="max_length", truncation=True, max_length=64, return_tensors="pt")

    # Extract input_ids and attention_mask from tokenized inputs
    article_input_ids = tokenized_article['input_ids'].flatten()
    article_attention_mask = tokenized_article['attention_mask'].flatten()
    question_input_ids = tokenized_question['input_ids'].flatten()
    question_attention_mask = tokenized_question['attention_mask'].flatten()
    options_input_ids = tokenized_options['input_ids'].flatten()
    options_attention_mask = tokenized_options['attention_mask'].flatten()

    # Create dictionary for this example
    embedded_input = {
        'article_input_ids': article_input_ids,
        'article_attention_mask': article_attention_mask,
        'question_input_ids': question_input_ids,
        'question_attention_mask': question_attention_mask,
        'options_input_ids': options_input_ids,
        'options_attention_mask': options_attention_mask
    }

    # Append the embedded input to the list
    embedded_inputs.append(embedded_input)

# Now `embedded_articles` contains the tokenized and embedded representations of all articles in the training dataset
# Let's print an example to see the structure
print(embedded_inputs[0])

# Convert the list of embedded inputs into a PyTorch TensorDataset
# This will allow you to easily create DataLoader for training
article_input_ids = torch.stack([example['article_input_ids'] for example in embedded_inputs])
article_attention_masks = torch.stack([example['article_attention_mask'] for example in embedded_inputs])
question_input_ids = torch.stack([example['question_input_ids'] for example in embedded_inputs])
question_attention_masks = torch.stack([example['question_attention_mask'] for example in embedded_inputs])
options_input_ids = torch.stack([example['options_input_ids'] for example in embedded_inputs])
options_attention_masks = torch.stack([example['options_attention_mask'] for example in embedded_inputs])

# Create a TensorDataset
train_dataset = TensorDataset(article_input_ids, article_attention_masks,
                              question_input_ids, question_attention_masks,
                              options_input_ids, options_attention_masks)

# Now `train_dataset` is a PyTorch TensorDataset containing the input_ids and attention_masks for all inputs
# You can use `train_dataset` to create a DataLoader for training your model

# Assuming 'answers' is a list of answer choices (e.g., ['A', 'B', 'C', 'D'])
answers = ['A', 'B', 'C', 'D']

# Extract answers from your dataset
answers_dataset = [example['answer'] for example in dataset['train']]

# Map each answer to its index
answer_indices = [answers.index(answer) for answer in answers_dataset]

# Convert answer indices list to a tensor
labels = torch.tensor(answer_indices, dtype=torch.long)

print("HOW TO ACCESS INPUT IDS AND ATTENTION MASKS:")
print("Article input IDs:", train_dataset[0][0])
print("Article attention masks:", train_dataset[0][1])
print("Question input IDs:", train_dataset[0][2])
print("Question attention masks:", train_dataset[0][3])
print("Options input IDs:", train_dataset[0][4])
print("Options attention masks:", train_dataset[0][5])

print("HOW TO ACCESS LABELS (ANSWERS):")
print(labels)


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
model = GPT2LMHeadModel.from_pretrained('gpt2')

#device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")


def fine_tune_model(model, dataloader, epochs=4, learning_rate=0.00002):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader)*epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_examples = 0
        for batch in dataloader:
            input_ids, attention_mask = batch
            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits.view(-1, tokenizer.vocab_size), input_ids.view(-1))
            total_loss += loss.item()

            # Compute accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct = (predictions == input_ids).sum().item()
            total_correct += correct
            total_examples += input_ids.size(0)

            loss.backward()
            optimizer.step()
            scheduler.step()

        epoch_loss = total_loss / len(dataloader)
        epoch_accuracy = total_correct / total_examples
        print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}')
    return model

fine_tuned_model = fine_tune_model(model, dataloader)

sequence = ("He began his permiership by forming a five-man war cabinet which included Chamberlain as Lord President of the Council,"
"Labor leader Clement Attlee as Lord Privy Seal (later as Deputy Prime Minister), Halifax as Foreign Secretary," 
"And Arther Greenwood as a minister without portfolio. In practice,")

inputs = tokenizer.encode(sequence, return_tensors='pt')
outputs = fine_tuned_model.generate(inputs, max_length=200, do_sample=True, temperature=.7, top_k=50)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)


