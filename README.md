# Comic Art Classifier: Manga or Classic

Welcome to the Comic Art Classifier repository! This project aims to classify comic art images into two categories: Manga and Classic. The classifier is built using the VGG16 convolutional neural network architecture trained on custom collected data.

## Overview

This repository contains the necessary code and resources to train and evaluate the comic art classifier. Below is an outline of the contents:

- `images/`: This directory contains the dataset used for training and testing the classifier. It is organized into subdirectories for each class (Manga and Classic).

  Manga:

  <img src="https://github.com/rastogi17/comic_art_classifier/assets/143191304/a86d0867-e2b5-46a6-ba20-41be53ae5008" alt="Manga" width="200" height="200">
  <img src="https://github.com/rastogi17/comic_art_classifier/assets/143191304/61807062-73c4-4359-86dc-37521d859a48" alt="Manga" width="200" height="200">
  
  Classic:

  <img src="https://github.com/rastogi17/comic_art_classifier/assets/143191304/c7b9775e-b840-47f9-82c2-444f89f2a023" alt="Classic" width="200" height="200">
  <img src="https://github.com/rastogi17/comic_art_classifier/assets/143191304/f5de7e25-df07-4ef1-a7ef-4eb490b38128" alt="Classic" width="200" height="200">
  


  
- `best_model.h5`: This file contains the best-performing model.

- `training.ipynb`: This Jupyter Notebook contains the code for training the VGG16 model on the provided dataset and evaluating the trained model's performance, including generating confusion matrix and other evaluation metrics.

- `app.py`: This file contains the code for deploying the trained model as a web application using Streamlit for easy interaction.

## Usage

### Training

To train the comic art classifier, follow these steps:

1. Ensure you have the required dependencies installed. You can install them via `pip install -r requirements.txt`.

2. Prepare your dataset by organizing images into two folders: `manga` and `classic`, each containing respective class images.

3. Run the `training.ipynb` notebook to train the VGG16 model on your dataset.

4. Once training is complete, the trained model will be saved as `best_model.h5`.

### Evaluation

To evaluate the trained model, follow these steps:

1. Ensure you have the trained model file (`best_model.h5`) available.

2. Prepare a separate test dataset (with images organized similarly to the training dataset).

3. Run the `training.ipynb` notebook to evaluate the model's performance on the test dataset. This notebook will generate various evaluation metrics including the confusion matrix.

### Deployment

To deploy the model as a web application, follow these steps:

1. Ensure you have Streamlit installed (`pip install streamlit`).

2. Run the `app.py` script using Streamlit (`streamlit run app.py`). This will launch a web application where users can upload images and get predictions from the model.

## Confusion Matrix

The confusion matrix provides valuable insights into the performance of the classifier. Below is a placeholder for the confusion matrix results on test dataset:

<img src="https://github.com/rastogi17/comic_art_classifier/assets/143191304/add356b9-229b-400d-af37-bafc2a142282" alt="Conf Matrix" width="491" height="421">

## Contributors

- [Kaustubh Rastogi](https://github.com/rastogi17)

Feel free to contribute to this project by forking the repository, making improvements, and submitting pull requests.

Happy classifying! 📚✏️
