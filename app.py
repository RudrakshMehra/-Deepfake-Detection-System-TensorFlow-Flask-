import os
from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, redirect, url_for

# Initialize Flask app
app = Flask(__name__)

# Load the model without the optimizer
model = load_model("final_model.keras", compile=False)

# Recompile the model with the optimizer
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define a function to preprocess the uploaded image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to predict if the uploaded image is real or fake
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # save upload
    img_path = os.path.join('static', file.filename)
    file.save(img_path)

    # predict
    img_array = preprocess_image(img_path)
    pred = model.predict(img_array)[0][0]
    result = 'Fake' if pred > 0.5 else 'Real'

    # return JSON for AJAX
    return jsonify({
        'prediction': result,
        'image_path': img_path.replace('\\','/')   # normalize slashes
    })


if __name__=='__main__':
    app.run(debug=True, use_reloader=False)


