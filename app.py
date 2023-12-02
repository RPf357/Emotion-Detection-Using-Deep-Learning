from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = r"C:\UNT AI\SD for AI\Project 2\Emotion-Detection-Using-Deep-Learning\model\Expdect_model.h5"
model = load_model(model_path)

# Define expression names
expression_names = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Function to preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(48, 48))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the expression in the image
def predict_expression(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    return expression_names.get(predicted_class, 'Unknown')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        # Save the uploaded file to a temporary location
        temp_folder = 'static/images/'
        os.makedirs(temp_folder, exist_ok=True)
        uploaded_file_path = os.path.join(temp_folder, 'temp.jpg')

        file.save(uploaded_file_path)

        # Predict the expression
        predicted_expression = predict_expression(uploaded_file_path)

        # Display the uploaded image and predicted expression
        return render_template('index.html', prediction=f'Predicted Expression: {predicted_expression}', image_path=uploaded_file_path)

if __name__ == '__main__':
    app.run(debug=True)
