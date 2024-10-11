import argparse
from transformers import BertTokenizerFast, BertForTokenClassification
import torch

# Mapping from label indices to label names
id2label = {0: 'B-MOUNTAIN', 1: 'I-MOUNTAIN', 2: 'O'}


def predict(text, tokenizer, model):
    # Function to make predictions on the input text
    # Tokenize the input text with padding and truncation
    tokenized_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**tokenized_input)  # Forward pass through the model

    # Get the predicted labels by finding the index of the maximum logit
    predicted_labels = outputs.logits.argmax(dim=-1)[0]

    named_entities = []  # List to store the named entities

    # Iterate over the input tokens and their corresponding predicted labels
    for token, label in zip(tokenized_input["input_ids"][0], predicted_labels):
        label_id = label.item()  # Get the integer label ID
        label_name = id2label[label_id]  # Map the label ID to its name

        # Append the decoded token and its label name to the named_entities list
        named_entities.append((tokenizer.decode([token]), label_name))

    return named_entities  # Return the list of named entities


def main():
    # Main function to handle argument parsing and model inference
    parser = argparse.ArgumentParser(description="Inference Named Entity Recognition with BERT")

    # Command-line argument for the path to the model directory
    parser.add_argument("--path_to_model",
                        type=str,
                        required=True,
                        default="model_save/",
                        help="Path to model ('model_save/')")

    # Command-line argument for the input text for prediction
    parser.add_argument("--text",
                        type=str,
                        required=True,
                        default="Alps is the tallest mountain in the world, attracting climbers from all over the globe.",
                        help="Input text for NER prediction")

    args = parser.parse_args()  # Parse the command-line arguments
    model_dir = args.path_to_model  # Get the model directory from the arguments

    # Load the tokenizer and model from the specified directory
    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    model = BertForTokenClassification.from_pretrained(model_dir)

    # Make predictions using the provided text and print the results
    token_label_pairs = predict(args.text, tokenizer, model)
    print(token_label_pairs)  # Print the token-label pairs


if __name__ == "__main__":
    main()
