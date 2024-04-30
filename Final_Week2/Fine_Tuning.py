import torch
import torch.nn as nn
from transformers import BertTokenizer, BertForMaskedLM
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AdamW, get_linear_schedule_with_warmup
import random
# Ray: I'm copying these in, but we will want to go through and remove extraneous imports before we submit.
from torch.utils.data import TensorDataset
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
dataset = load_dataset("ehovy/race", "all")
dataset_features = ['example_id', 'article', 'answer', 'question', 'options']
train_examples = dataset['train']  # Access the training examples


# List to hold tokenized and embedded representations of inputs
embedded_inputs = []
count = 0

# Process each example in the training dataset
for example in train_examples:
    count = count + 1
    if count == 101:
        break

    article = example['article']
    question = example['question']
    options = example['options']

    # Tokenize article, question, and options with padding
    tokenized_article = tokenizer(article, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    tokenized_question = tokenizer(question, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    tokenized_options = tokenizer(options, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    article_token = tokenizer('article', padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    question_token = tokenizer('question', padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    options_token = tokenizer('options', padding="max_length", truncation=True, max_length=128, return_tensors="pt")

    # Extract input_ids and attention_mask from tokenized inputs
    article_input_ids = torch.cat((article_token['input_ids'].flatten(), tokenized_article['input_ids'].flatten()), 0)
    article_attention_mask = torch.cat((article_token['attention_mask'].flatten(), tokenized_article['attention_mask'].flatten()), 0)
    question_input_ids = torch.cat((question_token['input_ids'].flatten(), tokenized_question['input_ids'].flatten()), 0)
    question_attention_mask = torch.cat((question_token['attention_mask'].flatten(), tokenized_question['attention_mask'].flatten()), 0)
    options_input_ids = torch.cat((options_token['input_ids'].flatten(), tokenized_options['input_ids'].flatten()), 0)
    options_attention_mask = torch.cat((options_token['attention_mask'].flatten(), tokenized_options['attention_mask'].flatten()), 0)

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
# print(embedded_inputs[0])

# Convert the list of embedded inputs into a PyTorch TensorDataset
# This will allow you to easily create DataLoader for training
article_input_ids = torch.stack([example['article_input_ids'] for example in embedded_inputs])
article_attention_masks = torch.stack([example['article_attention_mask'] for example in embedded_inputs])
question_input_ids = torch.stack([example['question_input_ids'] for example in embedded_inputs])
question_attention_masks = torch.stack([example['question_attention_mask'] for example in embedded_inputs])
options_input_ids = torch.stack([example['options_input_ids'] for example in embedded_inputs])
options_attention_masks = torch.stack([example['options_attention_mask'] for example in embedded_inputs])

# concat all input ids and attentions together so we can have only 1 stream of input instead of 3
inputs = [torch.cat((example['article_input_ids'], example['question_input_ids'], example['options_input_ids']), 0) for example in embedded_inputs]
inputs_ids = torch.stack(inputs)
attentions = [torch.cat((example['article_attention_mask'], example['question_attention_mask'], example['options_attention_mask']), 0) for example in embedded_inputs]
attentions_ids = torch.stack(attentions)

# Create a TensorDataset
train_dataset = TensorDataset(inputs_ids, attentions_ids)

# Now `train_dataset` is a PyTorch TensorDataset containing the input_ids and attention_masks for all inputs
# You can use `train_dataset` to create a DataLoader for training your model

# Assuming 'answers' is a list of answer choices (e.g., ['A', 'B', 'C', 'D'])
answers = ['A', 'B', 'C', 'D']

# Extract answers from your dataset
answers_dataset = [example['answer'] for example in dataset['train']][:100]

# Map each answer to its index
answer_indices = [answers.index(answer) for answer in answers_dataset]

# Convert answer indices list to a tensor
labels = torch.tensor(answer_indices, dtype=torch.long)

print("HOW TO ACCESS LABELS (ANSWERS):")
print(labels)


model = GPT2LMHeadModel.from_pretrained('gpt2')
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)

# device = torch.device ("cuda" if torch.cuda.is_available() else "cpu")


def fine_tune_model(model, dataloader, epochs=1, learning_rate=0.00002):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(dataloader)*epochs)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_examples = 0
        for batch in dataloader:
            print((batch))
            input_ids, attention_mask = batch
            optimizer.zero_grad()
            print(len(inputs_ids))
            print(len(attention_mask))
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

