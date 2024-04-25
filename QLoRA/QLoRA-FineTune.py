import transformers
import torch
from ctransformers import AutoModelForCausalLM
from datasets import load_dataset

model_id = "TheBloke/llama-2-7b-Guanaco-QLoRA-GGUF"

print(torch.cuda.is_available())
gpu = torch.device("cpu") 

model = AutoModelForCausalLM.from_pretrained(model_id, model_file="llama-2-7b.Q4_K_M.gguf", model_type="llama", gpu_layers=50, context_length=4096)

dataset = load_dataset("ehovy/race", 'high')
dataset_features = ['example_id', 'article', 'answer', 'question', 'options']
train_examples = dataset['train'] #Access training rather than test or validation set. 

torch.cuda.init() # initialize the gpu device. 

training_args = transformers.TrainingArguments(
    auto_find_batch_size=True, # Try to auto-find a batch size. 
    # Also see https://huggingface.co/google/flan-ul2/discussions/16#64c8bdaf4cc48498134a0271
    learning_rate=2e-4,
    # bf16=True, # Only on A100
    fp16=True, # On V100
    save_total_limit=4,
    # warmup_steps=2,
    num_train_epochs=30, # Total number of training epochs.It stablised after 30.
    output_dir='checkpoints',
    save_strategy='epoch',
    report_to="none",
    logging_steps=25, # Number of steps between logs.
    save_safetensors=True,
    #load_best_model_at_end=True,
    metric_for_best_model='accuracy',
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_examples,
    # eval_dataset=dataset["test"], # 16GB GPU not big enough
    args=training_args,
    #data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    # compute_metrics=compute_metrics,
)

model.config.use_cache = False

trainer.train(resume_from_checkpoint=False)
trainer.save_model("final_model")
transformers.logging.set_verbosity_error() 
 

