# Load model directly
from ctransformers import AutoModelForCausalLM
from datasets import load_dataset

model_id = "TheBloke/llama-2-7b-Guanaco-QLoRA-GGUF"

llm = AutoModelForCausalLM.from_pretrained(model_id, model_file="llama-2-7b.Q4_K_M.gguf", model_type="llama", gpu_layers=50, context_length=4096)

dataset = load_dataset("ehovy/race")
dataset_features = ['example_id', 'article', 'answer', 'question', 'options']
train_examples = dataset['train'] # Access training rather than validation set.

dataloader = DataLoader(train_examples, batch_size=64, shuffle=True)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token':'[PAD]'})

embedded_inputs = []

for example in train_examples: 
  article = example['article']
  tokenized_article = tokenizer(article, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
  article_inputs_ids = tokenized_article['input_ids'].flatten()
  article_attention_mask = 

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
            print(batch)
            print(type(batch)) 
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

#inputs = tokenizer.encode(sequence, return_tensors ='p')
#outputs = model.generate(inputs, max_length=200, do_sample=True, temperature=.7, top_k=50)

print(llm(sequence))
#text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#print(text) 
sequence2 = (
"Read the given passage, then answer the multiple choice question that follows with (A, B, C, or D):\nPassage:\nLast week I talked with some of my students about what they wanted to do after they graduated, and what kind of job prospects they thought they had. Given that I teach students who are training to be doctors, I was surprised do find that most thought that they would not be able to get the jobs they wanted without \"outside help\". \"What kind of help is that?\" I asked, expecting them to tell me that they would need a or family friend to help them out. \"Surgery ,\" one replied. I was pretty alarmed by that response. It seems that the graduates of today are increasingly willing to go under the knife to get ahead of others when it comes to getting a job . One girl told me that she was considering surgery to increase her height. \"They break your legs, put in special extending screws, and slowly expand the gap between the two ends of the bone as it re-grows, you can get at least 5 cm taller!\" At that point, I was shocked. I am short, I can't deny that, but I don't think I would put myself through months of agony just to be a few centimetres taller. I don't even bother to wear shoes with thick soles, as I'm not trying to hide the fact that I am just not tall! It seems to me that there is a trend towards wanting \"perfection\" , and that is an ideal that just does not exist in reality. No one is born perfect, yet magazines, TV shows and movies present images of thin, tall, beautiful people as being the norm. Advertisements for slimming aids, beauty treatments and cosmetic surgery clinics fill the pages of newspapers, further creating an idea that \"perfection\" is a requirement, and that it must be purchased, no matter what the cost. In my opinion, skills, rather than appearance, should determine how successful a person is in his/her chosen career.\n Question:\n Which' s the best title for the passage?.\n [A:\"Young Graduates Have Higher Expectations\", B:\"Young Graduates Look to Surgery for Better Jobs\", C:\"Young Graduates' Opinion About Cosmetic Surgery\", D:\"Young Graduates Face a Different Situation in Job-hunting\" ]. Give your answer in the format 'Answer: [#]', then explain your choice.")

#inputs = tokenizer.encode(sequence, return_tensors ='p')
#outputs = model.generate(inputs, max_length=200, do_sample=True, temperature=.7, top_k=50)

print(llm(sequence))
#text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#print(text) 

sequence2 = (
"Read the given passage, then answer the multiple choice question that follows with (A, B, C, or D):\nPassage:\nLast week I talked with some of my students about what they wanted to do after they graduated, and what kind of job prospects they thought they had. Given that I teach students who are training to be doctors, I was surprised do find that most thought that they would not be able to get the jobs they wanted without \"outside help\". \"What kind of help is that?\" I asked, expecting them to tell me that they would need a or family friend to help them out. \"Surgery ,\" one replied. I was pretty alarmed by that response. It seems that the graduates of today are increasingly willing to go under the knife to get ahead of others when it comes to getting a job . One girl told me that she was considering surgery to increase her height. \"They break your legs, put in special extending screws, and slowly expand the gap between the two ends of the bone as it re-grows, you can get at least 5 cm taller!\" At that point, I was shocked. I am short, I can't deny that, but I don't think I would put myself through months of agony just to be a few centimetres taller. I don't even bother to wear shoes with thick soles, as I'm not trying to hide the fact that I am just not tall! It seems to me that there is a trend towards wanting \"perfection\" , and that is an ideal that just does not exist in reality. No one is born perfect, yet magazines, TV shows and movies present images of thin, tall, beautiful people as being the norm. Advertisements for slimming aids, beauty treatments and cosmetic surgery clinics fill the pages of newspapers, further creating an idea that \"perfection\" is a requirement, and that it must be purchased, no matter what the cost. In my opinion, skills, rather than appearance, should determine how successful a person is in his/her chosen career.\n Question:\n Which' s the best title for the passage?.\n [A:\"Young Graduates Have Higher Expectations\", B:\"Young Graduates Look to Surgery for Better Jobs\", C:\"Young Graduates' Opinion About Cosmetic Surgery\", D:\"Young Graduates Face a Different Situation in Job-hunting\" ]. Give your answer in the format 'Answer: [#]', then explain your choice.")

#inputs = tokenizer.encode(sequence, return_tensors ='p')
#outputs = model.generate(inputs, max_length=200, do_sample=True, temperature=.7, top_k=50)

print(llm(sequence))
#text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#print(text) 
sequence2 = (
"Read the given passage, then answer the multiple choice question that follows with (A, B, C, or D):\nPassage:\nLast week I talked with some of my students about what they wanted to do after they graduated, and what kind of job prospects they thought they had. Given that I teach students who are training to be doctors, I was surprised do find that most thought that they would not be able to get the jobs they wanted without \"outside help\". \"What kind of help is that?\" I asked, expecting them to tell me that they would need a or family friend to help them out. \"Surgery ,\" one replied. I was pretty alarmed by that response. It seems that the graduates of today are increasingly willing to go under the knife to get ahead of others when it comes to getting a job . One girl told me that she was considering surgery to increase her height. \"They break your legs, put in special extending screws, and slowly expand the gap between the two ends of the bone as it re-grows, you can get at least 5 cm taller!\" At that point, I was shocked. I am short, I can't deny that, but I don't think I would put myself through months of agony just to be a few centimetres taller. I don't even bother to wear shoes with thick soles, as I'm not trying to hide the fact that I am just not tall! It seems to me that there is a trend towards wanting \"perfection\" , and that is an ideal that just does not exist in reality. No one is born perfect, yet magazines, TV shows and movies present images of thin, tall, beautiful people as being the norm. Advertisements for slimming aids, beauty treatments and cosmetic surgery clinics fill the pages of newspapers, further creating an idea that \"perfection\" is a requirement, and that it must be purchased, no matter what the cost. In my opinion, skills, rather than appearance, should determine how successful a person is in his/her chosen career.\n Question:\n Which' s the best title for the passage?.\n [A:\"Young Graduates Have Higher Expectations\", B:\"Young Graduates Look to Surgery for Better Jobs\", C:\"Young Graduates' Opinion About Cosmetic Surgery\", D:\"Young Graduates Face a Different Situation in Job-hunting\" ]. Give your answer in the format 'Answer: [#]', then explain your choice.")

#inputs = tokenizer.encode(sequence, return_tensors ='p')
#outputs = model.generate(inputs, max_length=200, do_sample=True, temperature=.7, top_k=50)

print(llm(sequence))
#text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#print(text) 
