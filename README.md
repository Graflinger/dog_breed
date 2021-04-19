# Dog Breed Classifier

## Overview
Repository for the Dog breed classification project for the Udacity Data Science Nanodegree.

This projects goal is to build a CNN to categorize Dog Breeds in user given pictures.

This project is distributed in 2 parts:
- Notebook (Preparation of the CNN used to classify dog breeds)
- App (Hosting a Flask App to use the trained model on user given images)

## Project Definition
Thise project aims to be able to recognize dog breeds in images using deep learning.
Additionally, it should be detected if there is a human face instead of a dog or neither in the image
For this purpose, a CNN defined in a Jupyter notebook is used. 
This CNN is then hosted via a web app and can be populated with images to classify the images.
If a dog is detected, its breed will be classified. If a human face is detected, the most similar dog breed to this dog is returned.

### Problem statement
We are given a collection of images containing a dog and the belonging label containing the dog's breed. 
With the help of these files, a neural network should be trained in order to be able to classify the breed of new images containing a dog.

### Metrics
The loss function for fitting the models is categorical_crossentropy.
To evaluate which model performs best for this task, the accuracy of correct classifications is calculated. (number of correctly_classified)/(all testcases)

### Data Exploration
The dataset contains 8351 dog images with 133 different dog categories. 
The different dog categories are imbalanced as you can see here:
![imbalanced](https://github.com/Graflinger/dog_breed/blob/main/Notebook/imbalanced.png)


Pictures look like:
![dog](https://github.com/Graflinger/dog_breed/blob/main/Notebook/Labrador_retriever_06457.jpg)

The labels look like:
ages/train/001.Affenpinscher

## Methodology

### Data Preprocessing
The images cant be handled by the used neural networks in their base format. To be suitable for the CNNs used, they are being transformed to a 4D tensor suitable for supplying to a Keras CNN

### Implementation
This is mainly done in the path_to_tensor function, where a (223, 223, 1) shaped 3d tensor is transformed into a (1, 223, 223, 3) 4d tensor.

### Refinement
There was no need in improving anything from the Data Preprocessing part

## Result

### Model Evaluation and Validation
The first try was a self build CNN model with a poor test accuracy of under 10%

Afterwards I tried a pretrained model with VGG16 as base. Unfortunately it also performed poorly with a slightly better accuracy of 34.4498% 

The final model is build with transfer model too. As a pretrained model a Resnet50 model is being used. It resulted in a 82.6555% accuracy with the testing dataset, which is very good value at a usecase like this.

### Justification
The ResNet50 model provides a really good performance for this kind of task. The other ones aren't delivering an acceptable performance, therefore they wont be used in the finished web application

## Conclusion

### Reflection
It was very interesting to go through all steps in a image classification task.
From preprocessing the images to building your CNN models, finetuning and evaluating them.

Using transfer learning with state of the art models like ResNet50 delivers a very appealing performance.
Building a self written model doesn't look like a suitable alternative.

### Improvement
One possible improvenment is to add more dense layers and other layers like dropouts to the final model.
Collecting more dog pictures to get a larger training dataset could be also very helpful.

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

