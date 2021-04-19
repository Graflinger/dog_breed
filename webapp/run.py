
from flask import send_from_directory
from keras.applications.resnet50 import ResNet50
import json
import pandas as pd
from extract_bottleneck_features import *
from flask import render_template, request, jsonify
import numpy as np

import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

import cv2
from tensorflow import keras

from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from tqdm import tqdm


UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads/"
app = Flask(__name__)


# define ResNet50 model for dog detection
ResNet50_model_dog_detection = ResNet50(weights='imagenet')

# load ResNet50 model for dog breed detection
Resnet50_model_breed = keras.models.load_model("resnet50dogbreed")

# load dogbreed labels
df = pd.read_csv("data/dog_breeds.csv")
dog_names = list(df["dog_breeds"])


def path_to_tensor(img_path):
    '''
    prepares a image for deep learning
            Parameters:
                    img_path (str): path to image which should get prepared
            Returns:
                    tensor4d (4D tensor): 4d tensor with shape (1, 224, 244, 3)
    '''
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    tensor4d = np.expand_dims(x, axis=0)
    return tensor4d


def paths_to_tensor(img_paths):
    '''
    prepares a list of images for deep learning
            Parameters:
                    img_paths (list):list of paths to images which should get prepared
            Returns:
                    np.vstack (stack of 4D tensor): stack 4d tensor with shape (1, 224, 244, 3)
    '''
    list_of_tensors = [path_to_tensor(img_path)
                       for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)


def ResNet50_predict_labels(img_path):
    '''
    Using resnet50 to predict if a given picture is a dog
            Parameters:
                    img_path (str): path to image which should get inspected
            Returns:
                    np.argmax(int): returns result of prediction
    '''
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model_dog_detection.predict(img))


def dog_detector(img_path):
    '''
    detects if a given picture is a dog using resnet50
            Parameters:
                    img_path (str): path to image which should get detected
            Returns:
                    (bool): returns result of detection
    '''
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


def face_detector(img_path):
    '''
    detects if a given picture has a face using resnet50
            Parameters:
                    img_path (str): path to image which should get detected
            Returns:
                    (int): returns number of detected faces
    '''
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


def Resnet50_predict_breed(img_path):
    '''
   classifies the breed of a dog in a given picture
            Parameters:
                    img_path (str): path to image which should get classified
            Returns:
                    (string): returns the result of the prediction
    '''
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet50_model_breed.predict(bottleneck_feature)
    # # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)].split(".")[1]


def dog_breed(img_path):
    '''
    Classifies which kind of picture is given. If it is a dog or a human face, it return a matching dog breed.
            Parameters:
                    img_path (str): path to image which should get classified
            Returns:
                    (string): answer string for the application
    '''
    if face_detector(img_path) == True:
        return "This human needs following breed as a pet: " + Resnet50_predict_breed(img_path)

    elif dog_detector(img_path) == True:
        return "This dog looks like a: " + Resnet50_predict_breed(img_path)

    else:
        return "Neither a dog or a humans was found in this image"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
@app.route('/index')
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join("uploads", filename))
            return redirect(url_for('predict',
                                    filename=filename))
    return render_template('master.html')


@app.route('/upload/<filename>')
def predict(filename):

    message = dog_breed("uploads/" + filename).replace("_", " ")
    return render_template('go.html', filename=filename,  message_string=message)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory("uploads/",
                               filename)


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
