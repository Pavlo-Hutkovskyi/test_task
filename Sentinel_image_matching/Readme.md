#  Sentinel-2 image matching

## Overview

This project focuses on developing an algorithm for matching satellite images, specifically utilizing data from the Sentinel-2 satellite. The primary objective is to create a model that can accurately match images taken in different seasons, addressing the challenges posed by variations in lighting, vegetation, and atmospheric conditions. But the dataset was not uploaded to the repository because it is very large, the repository only has a folder with already processed data.

## Dataset Link

You can access the dataset used for training and testing the algorithm here: [Dataset Link](https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine)

## Requirements

* Python 3.x
* OpenCV
* PyTorch (for deep learning models)
* Additional libraries (e.g., NumPy, Matplotlib) as needed

Make sure to install the required packages. You can use `pip` to install the necessary libraries:

```bash
pip install -r requirements.txt
```

## Directory Structure

```
project-directory/
├── data/                                   # Directory with dataset with satellite images
├── SuperGluePretrainedNetwork/             # Directory with SuperGluePretrainedNetwork
├── models_training.py                      # Script for define algorithm
├── models_inference.py                     # Script for making a match between images
├── data_preparation.ipynb                  # Notebook for preparation data from Kaggle
├── demo_image_matching.ipynb               # Notebook for an example of using the algorithms
└── README.md                               # Project documentation
```

## Data Preparation
The dataset was taken from [Kaggle](https://www.kaggle.com/) which is called Deforestation in Ukraine from Sentinel2 data. From this dataset, I read three-channel images, reduce their size, since their size is too large, we will reduce it for further work, and also increase the contrast for better detection of key points.

## Algorithm Functionality
The algorithm is designed to:

* Detect and extract features from satellite images.
* Match features across images from different seasons.
* Provide visualization of matched features to assess performance.


## Making Predictions

To use the especial algorithm for matching image, run the following command:
```bash
 python models_inference.py algorithm_name 'first_image_path.jpg' 'second_image_path.jpg'
```
Algorithm name:
* SIFT
* ORB
* SuperGlue


