from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import nltk
from datasets import load_metric
import logging
import torch
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)

# Download necessary NLTK data
nltk.download('punkt')
# Load ROUGE metric
metric = load_metric("rouge")

# Load dataset
dataset = load_dataset("cnn_dailymail", '3.0.0')

# subset_size_train = 1000
# subset_size_validation = 1000
# train_subset = dataset["train"].select(range(subset_size_train))
# validation_subset = dataset["validation"].select(range(subset_size_validation))

# Model and Tokenizer
model_name = 't5-small'
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def preprocess_logits_for_metrics(logits, labels):
    # Check if logits is a tuple and extract the logits tensor
    if isinstance(logits, tuple):
        logits = logits[0]  # Assuming the first element is the logits tensor

    # Convert logits to predicted token IDs
    preds = torch.argmax(logits, dim=-1) if logits is not None else None
    return preds, labels



def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    logging.debug(f"Predictions type: {type(predictions)}")
    logging.debug(f"Number of elements in {type(predictions)} : {len(predictions)}")
    for i, tensor in enumerate(predictions):
        logging.debug(f"Shape of element {i}: {tensor.shape}")

    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Formatting the predictions and labels for ROUGE evaluation
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {key: value.mid.fmeasure * 100 for key, value in result.items()}


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


tokenized_datasets =  dataset.map(preprocess_function, batched=True)


# tokenized_validation_subset = validation_subset.map(preprocess_function, batched=True)


# tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="/scratch/railabs/ld258/output/summarizer_models/results",
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",        # Save checkpoints at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="rouge1",  # Specify the metric to use for best model (change as needed)
    greater_is_better=True,  # Set to True if higher metric score is better
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
   compute_metrics=compute_metrics,
   preprocess_logits_for_metrics=preprocess_logits_for_metrics
)

# Train the model
trainer.train()
logging.info("Training is Complete")


logging.info("Starting Evaluation !! ")
# Evaluate the model
trainer.evaluate()
