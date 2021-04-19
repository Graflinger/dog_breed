# Dog Breed Classifier

## Overview
Repository for the Dog breed classification project for the Udacity Data Science Nanodegree.

This projects goal is to build a CNN to categorize Dog Breeds in user given pictures.

This project is distributed in 2 parts:
- Notebook (Preparation of the CNN used to classify dog breeds)
- App (Hosting a Flask App to use the trained model on user given images)

### Project Definition
Thise project aims to be able to recognize dog breeds in images using deep learning.
Additionally, it should be detected if there is a human face instead of a dog or neither in the image
For this purpose, a CNN defined in a Jupyter notebook is used. 
This CNN is then hosted via a web app and can be populated with images to classify the images.
If a dog is detected, its breed will be classified. If a human face is detected, the most similar dog breed to this dog is returned.

### Problem statement
We are given a collection of images containing a dog and the belonging label containing the dog's breed.

## Installation

This code was implemented using python 3.8.x

Used packages are:

- Pandas
- Tensorflow
- Keras
- Flask
- Numpy

### Instructions for excecuting the program:
1. Run the following command in the app's directory to run your web app.
    `python run.py`

2. Go to http://0.0.0.0:3001/

## Relevant Files

### /Notebook Folder
This is where the jupyter notebook is located, in which the CNN used in the flask app is trained

### /app Folder
run.py is used to launch the flask web application.

