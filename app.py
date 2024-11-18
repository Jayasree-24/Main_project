from flask import Flask,render_template,request
import cv2
import numpy as np
import tensorflow as tf
import h5py

app = Flask(__name__,template_folder='template' )

def load_h5_model(filepath):
 """Loads a Keras model from an HDF5 file."""
 with h5py.File(filepath, 'r') as f:
   model = tf.keras.models.load_model(f)
 return model

# Load the model using load_h5_model
model1 = load_h5_model(r"ROI_Resnet50.h5")
model2 = load_h5_model(r"ROI_Densenet201.h5")
model3 = load_h5_model(r"vgg16_roi.h5")
model4 = load_h5_model(r"ROI_Vgg19.h5")
model5 = load_h5_model(r"Inceptionv3_roi.h5")
model6 = load_h5_model(r"Xception_roi.h5")
model7 = load_h5_model(r"ROI_mobilenetV2.h5")
# Function to classify a single image
def classify_single_image(image_path):
 # Read the image
 img = cv2.imread(image_path)
 # Resize the image to match the desired input shape
 resized_img = cv2.resize(img, (224, 224))
 # Preprocess the image
 processed_img = resized_img.astype('float32') / 255.0
 processed_img = np.expand_dims(processed_img, axis=0) # Add batch dimension
 # Predict probabilities 
 
 pred1 = model1.predict(processed_img)
 pred2 = model2.predict(processed_img)
 pred3 = model3.predict(processed_img)
 pred4 = model4.predict(processed_img)
 pred5 = model5.predict(processed_img)
 pred6 = model6.predict(processed_img)
 pred7 = model7.predict(processed_img)
 # Extract the first element (assuming single value)
 predicted_value1 = pred1.item()
 predicted_value2 = pred2.item()
 predicted_value3 = pred3.item()
 predicted_value4 = pred4.item()
 predicted_value5 = pred5.item()
 predicted_value6 = pred6.item()
 predicted_value7 = pred7.item()
# Set a threshold for classification
 threshold = 0.5
 predictions = { 
 "model1": f"COVID-19 ({predicted_value1:.2f})" if predicted_value1> threshold 
else f"Normal ({(1 - predicted_value1):.2f})",
 "model2": f"COVID-19 ({predicted_value2:.2f})" if predicted_value2> threshold 
else f"Normal ({(1 - predicted_value2):.2f})",
"model3": f"COVID-19 ({predicted_value3:.2f})" if predicted_value3> threshold 
else f"Normal ({(1 - predicted_value3):.2f})",
 "model4": f"COVID-19 ({predicted_value4:.2f})" if predicted_value4> threshold 
else f"Normal ({(1 - predicted_value4):.2f})",
 "model5": f"COVID-19 ({predicted_value5:.2f})" if predicted_value5> threshold 
else f"Normal ({(1 - predicted_value5):.2f})",
 "model6": f"COVID-19 ({predicted_value6:.2f})" if predicted_value6> threshold 
else f"Normal ({(1 - predicted_value6):.2f})",
 "model7": f"COVID-19 ({predicted_value7:.2f})" if predicted_value7> threshold 
else f"Normal ({(1 - predicted_value7):.2f})"
 }
 return predictions

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
        return render_template('page2.html', predicted_class=predicted_class)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)

































'''
from flask import Flask, render_template, request
import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
import h5py

app = Flask(__name__, template_folder='template')

# Load the model using load_model
model1 = load_model("ROI_Resnet50.h5")
model2 = load_model("ROI_Densenet201.h5")
model3 = load_model("vgg16_roi.h5")
model4 = load_model("ROI_Vgg19.h5")
model5 = load_model("Inceptionv3_roi.h5")
model6 = load_model("Xception_roi.h5")
model7 = load_model("ROI_mobilenetV2.h5")

# Function to classify a single image
def classify_single_image(image):
    # Resize the image to match the desired input shape
    resized_img = cv2.resize(image, (224, 224))
    # Preprocess the image
    processed_img = resized_img.astype('float32') / 255.0
    processed_img = np.expand_dims(processed_img, axis=0)  # Add batch dimension
    # Predict probabilities
    pred1 = model1.predict(processed_img)
    pred2 = model2.predict(processed_img)
    pred3 = model3.predict(processed_img)
    pred4 = model4.predict(processed_img)
    pred5 = model5.predict(processed_img)
    pred6 = model6.predict(processed_img)
    pred7 = model7.predict(processed_img)
    # Extract the first element (assuming single value)
    predicted_value1 = pred1.item()
    predicted_value2 = pred2.item()
    predicted_value3 = pred3.item()
    predicted_value4 = pred4.item()
    predicted_value5 = pred5.item()
    predicted_value6 = pred6.item()
    predicted_value7 = pred7.item()
    # Set a threshold for classification
    threshold = 0.5
    predictions = {
        "model1": f"COVID-19 ({predicted_value1:.2f})" if predicted_value1 > threshold else f"Normal ({1 - predicted_value1:.2f})",
        "model2": f"COVID-19 ({predicted_value2:.2f})" if predicted_value2 > threshold else f"Normal ({1 - predicted_value2:.2f})",
        "model3": f"COVID-19 ({predicted_value3:.2f})" if predicted_value3 > threshold else f"Normal ({1 - predicted_value3:.2f})",
        "model4": f"COVID-19 ({predicted_value4:.2f})" if predicted_value4 > threshold else f"Normal ({1 - predicted_value4:.2f})",
        "model5": f"COVID-19 ({predicted_value5:.2f})" if predicted_value5 > threshold else f"Normal ({1 - predicted_value5:.2f})",
        "model6": f"COVID-19 ({predicted_value6:.2f})" if predicted_value6 > threshold else f"Normal ({1 - predicted_value6:.2f})",
        "model7": f"COVID-19 ({predicted_value7:.2f})" if predicted_value7 > threshold else f"Normal ({1 - predicted_value7:.2f})"
    }
    return predictions

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Get the uploaded image file
        image = request.files['image']
        # Read the image
        img = cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_COLOR)
        # Check for valid image
        if img is None:
            return render_template('error.html', message="Invalid image")

        # Use the image for model prediction
        predicted_class = classify_single_image(img)
        return render_template('page2.html', predicted_class=predicted_class)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
'''
