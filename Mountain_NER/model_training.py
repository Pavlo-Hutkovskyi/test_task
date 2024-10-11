import argparse
import ast

import numpy as np
import pandas as pd
from datasets import DatasetDict, Dataset
from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForTokenClassification, TrainingArguments, Trainer

# Argument parser for command-line arguments
parser = argparse.ArgumentParser(description="Training Named Entity Recognition with BERT")

# Adding a command-line argument for the path to the model directory
parser.add_argument("--path_to_model",
                    type=str,
                    required=True,
                    default="model_save/",
                    help="Path to model ('model_save/')")

# Parse command-line arguments
args = parser.parse_args()

# Load the dataset from a CSV file
df = pd.read_csv("data/processed_mountain_dataset.csv")

# Convert string representations of lists back into actual lists
df['tokens'] = df['tokens'].apply(ast.literal_eval)
df['ner_tags'] = df['ner_tags'].apply(ast.literal_eval)

# Split the dataset into training, validation, and test sets
SEED = 42
train_data, temp_data = train_test_split(df, test_size=0.20, random_state=SEED)
val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=SEED)

# Create a list of unique labels and a mapping from labels to indices
label_list = list(sorted(set([tag for sublist in train_data['ner_tags'] for tag in sublist])))
label_map = {label: idx for idx, label in enumerate(label_list)}

# Initialize the BERT tokenizer and model for token classification
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(label_list))

# Set the output directory for saving the model
output_dir = args.path_to_model


def align_labels(labels, word_ids):
    # Function to align the labels with the tokenized word IDs
    new_labels = [-100] * len(word_ids)  # Initialize labels with -100 for ignored tokens
    label_index = 0

    for i, word_id in enumerate(word_ids):
        if word_id is not None:  # Only consider non-None word IDs
            if label_index < len(labels):
                new_labels[i] = labels[label_index]  # Assign the label to the corresponding token
            # Move to the next label if the word ID changes
            if i == 0 or word_id != word_ids[i - 1]:
                label_index += 1

    return new_labels


def prepare_dataset(df):
    # Function to prepare the dataset for training by tokenizing and aligning labels
    tokens = df['tokens'].tolist()  # Get the token lists
    ner_tags = df['ner_tags'].tolist()  # Get the NER tags
    tokenized_inputs = tokenizer(tokens, is_split_into_words=True, padding=True, truncation=True,
                                 return_offsets_mapping=True, max_length=512)  # Tokenize the inputs

    labels_aligned = []
    for i in range(len(df)):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Get the word IDs for the current input
        labels = [label_map[tag] for tag in ner_tags[i]]  # Map NER tags to their indices
        labels_aligned.append(align_labels(labels, word_ids))  # Align the labels with word IDs

    tokenized_inputs['labels'] = labels_aligned  # Add the aligned labels to the tokenized inputs
    return Dataset.from_dict(tokenized_inputs)  # Create a Dataset object from the tokenized inputs


# Prepare the training, validation, and test datasets
train_dataset = prepare_dataset(train_data)
val_dataset = prepare_dataset(val_data)
test_dataset = prepare_dataset(test_data)

# Create a DatasetDict to hold the datasets
datasets = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset,
})


def compute_metrics(eval_prediction):
    # Function to compute evaluation metrics
    predictions, labels = eval_prediction
    predictions = np.argmax(predictions, axis=2)  # Get the predicted labels

    # Extract true predictions and labels, ignoring -100 (ignored tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Calculate and return precision, recall, F1 score, and accuracy
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
        "accuracy_score": accuracy_score(true_labels, true_predictions),
    }


# Set training arguments for the Trainer
training_args = TrainingArguments(
    output_dir='./results',  # Directory to save the model and results
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=16,  # Batch size for training
    per_device_eval_batch_size=16,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Weight decay for optimization
    logging_dir='logs',  # Directory for storing logs
    logging_steps=100,  # Logging frequency
    learning_rate=2e-4,  # Learning rate
    adam_epsilon=1e-8,  # Epsilon for Adam optimizer
    save_total_limit=3,  # Limit the total amount of checkpoints
    eval_strategy='steps',  # Evaluation strategy
    save_strategy='steps',  # Save strategy
    save_steps=500,  # Frequency of saving model checkpoints
    eval_steps=500,  # Frequency of evaluating the model
    load_best_model_at_end=True,  # Load the best model when finished training
    optim='adamw_torch',  # Optimizer to use
    seed=SEED,  # Random seed for reproducibility
    metric_for_best_model="f1",  # Metric to use for determining the best model
)

# Initialize the Trainer with model, arguments, datasets, and metrics
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Evaluate the model on the test dataset
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test result: ", test_results)  # Print the test results

# Save the model and tokenizer to the specified output directory
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
