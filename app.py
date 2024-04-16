import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

import tensorflow as tf

import time
import numpy as np
import cv2
from flask import Flask, render_template, request
from keras.models import load_model

app = Flask(__name__, template_folder='template')

model = load_model("./Final_Resnet50.h5")

# Function to classify a single image
def classify_single_image(image):
    # Read the image
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
    # Resize the image to match the desired input shape
    resized_img = cv2.resize(img, (224, 224))
    # Preprocess the image
    processed_img = resized_img.astype('float32') / 255.0
    processed_img = np.expand_dims(processed_img, axis=0)  # Add batch dimension
    # Predict probabilities for the positive class
    probability_positive = model.predict(processed_img)
    # Set a threshold for classification
    threshold = 0.5
    # Convert probability to binary prediction
    predicted_class = 'COVID-19 positive' if probability_positive > threshold else 'COVID-19 negative'
    return predicted_class

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded image file
        image = request.files['image']

        # Check for valid image extension
        allowed_extensions = {'jpg', 'jpeg', 'png'}  # Adjust as needed
        if image.filename.split('.')[-1].lower() not in allowed_extensions:
            return render_template('error.html', message="Invalid image format")

        # Use the image for model prediction
        predicted_class = classify_single_image(image)
        return render_template('second_page.html', predicted_class=predicted_class)
    return render_template('page1.html')

if __name__ == "__main__":
    app.run(debug=True)
