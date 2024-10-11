# NER with BERT for mountains names

## Overview

This repository contains code for training and using a Named Entity Recognition (NER) model based on the BERT architecture. The model is designed to identify mountain names in sentences.

## Requirements

You need to unzip the archive 'result/' and 'model _save/' .

Make sure to install the required packages. You can use `pip` to install the necessary libraries:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
project-directory/
├── data/
│   ├── mountain_dataset_with_markup.csv    # Raw dataset with markers
│   └── processed_mountain_dataset.csv      # Processed dataset for training
├── model_save/                             # Directory where the trained model is saved
├── model_training.py                       # Script for training the NER model
├── model_inference.py                      # Script for making predictions with the trained model
├── data_preparation.ipynb                  # Notebook for preparation data from Kaggle
├── demo_mountain_ner.ipynb                 # Notebook for an example of using the model
└── README.md                               # Project documentation
```

## Data Preparation
### Reading data
The dataset was taken from [Kaggle](https://www.kaggle.com/datasets/geraygench/mountain-ner-dataset) which is called Mountain NER dataset. The raw dataset (mountain_dataset_with_markup.csv) contains sentences with markers indicating the locations of mountain names. The processing script reads this dataset and generates a processed version with tokens and corresponding NER tags.

### Tagging Sentences
Processes each sentence, identifying mountain names based on the markers and assigns appropriate tags (B-MOUNTAIN, I-MOUNTAIN, O).

## Training the Model

To train the NER model, run the following command:
```bash
python model_training.py --path_to_model 'model_save/'
```

## Making Predictions

To use the trained model for predictions, run the following command:
```bash
python model_inference.py --path_to_model 'model_save/' --text 'sentence'
```



