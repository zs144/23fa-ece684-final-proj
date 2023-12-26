from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import nltk
from datasets import load_metric
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

# Download necessary NLTK data
nltk.download('punkt')
# Load ROUGE metric
metric = load_metric("rouge")

# Load dataset
dataset = load_dataset("cnn_dailymail", '3.0.0')

# Model and Tokenizer
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess_function(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)

    # Tokenize the targets
    labels = tokenizer(examples["highlights"], max_length=150, padding="max_length", truncation=True)

    # Make sure that the labels' attention mask is correctly sized
    # Adjust attention masks: 1 for tokens, 0 for padding
    labels_attention_masks = [[1 if token != tokenizer.pad_token_id else 0 for token in label] for label in labels["input_ids"]]

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["decoder_attention_mask"] = labels_attention_masks
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# Train the model
trainer.train()
logging.info("Training is Complete")

# Evaluation function for computing ROUGE score
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {key: value.mid.fmeasure * 100 for key, value in result.items()}

# Evaluate the model
trainer.evaluate(compute_metrics)
