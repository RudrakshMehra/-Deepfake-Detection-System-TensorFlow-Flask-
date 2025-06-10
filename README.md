
# 🧠 Deepfake Detection System

A Flask-based web application that uses a Convolutional Neural Network (CNN) to classify facial images as either **Real** or **Fake** (deepfake). Built using TensorFlow and OpenCV, the model was trained on a dataset of real and fake faces (~250MB in size) and deployed via a lightweight REST API.

---

## 🚀 Features

- ✅ Real-time image upload and prediction
- 🧠 Custom CNN model trained on preprocessed face images
- 📦 REST API for classification (`/predict`)
- 📸 Image preprocessing using OpenCV (resize, normalize)
- 🌐 Flask-based web interface for local or cloud deployment

---

## 🧠 Model Overview

- 3 Convolutional layers with ReLU activation and MaxPooling
- Dropout layers to prevent overfitting
- Fully connected Dense layer with Sigmoid output
- Trained using binary cross-entropy loss and Adam optimizer
- Achieved **~60% test accuracy** on a limited dataset

---

## 🖼️ Web App Demo

Upload a face image via the UI or POST an image file to `/predict`.  
The server responds with a prediction: `"Real"` or `"Fake"`.

---

## 📁 Folder Structure

```
deepfake-detector/
│
├── app.py                  # Flask backend + REST API
├── model.py                # CNN training script
├── prep-processing.py      # Data preprocessing script
├── final_model.keras       # Trained model
├── dataset/                # Preprocessed .npy files
├── Dataset/                # Raw images (real/ & fake/)
├── templates/              # index.html (upload form)
├── static/                 # Uploaded images
├── requirements.txt
└── README.md
```

---

## 📦 Installation & Usage

```bash
git clone https://github.com/yourusername/deepfake-detector.git
cd deepfake-detector
pip install -r requirements.txt
python app.py
```

Visit `http://127.0.0.1:5000` in your browser to test the web app.

---

## 🔗 API Usage

**Endpoint:** `POST /predict`  
**Form Data:** `image` (file)  
**Returns:**
```json
{
  "prediction": "Fake",
  "image_path": "static/your_uploaded_image.jpg"
}
```

---

## 📊 Future Improvements

- Improve accuracy using a larger dataset
- Deploy on Render/Heroku for public access
- Add face detection and cropping before classification
- Add deepfake video detection

---

## 👨‍💻 Author

Rudraksh Mehra  
📧 mehrarudraksh29070@gmail.com 
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)
