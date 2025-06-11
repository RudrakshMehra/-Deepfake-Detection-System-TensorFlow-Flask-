
# 🧠 Deepfake Detection System

A Flask-based web application that uses a Convolutional Neural Network (CNN) to classify facial images as either **Real** or **Fake** (deepfake). Built using TensorFlow and OpenCV, the model was trained on a dataset of real and fake faces (~980MB) and deployed via a lightweight REST API.

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
├── prep-processing.py      # Converts raw images to .npy arrays
├── final_model.keras       # Trained Keras model
├── download_dataset.py     # Downloads Dataset/ from Google Drive (raw images)
├── Dataset/                # Raw images (real/ and fake/), extracted from ZIP
├── dataset/                # Preprocessed .npy files (created by prep-processing.py)
├── templates/              # index.html for image upload form
├── static/                 # Uploaded images shown on result page
├── requirements.txt        # Required Python packages
└── README.md               # Project documentation

```

---

## 📥 Dataset Setup

This project uses a dataset (~980MB) hosted on Google Drive.  
The ZIP file contains:
- `Dataset/` folder with raw images (`real/` and `fake/`)
- `dataset/` folder with preprocessed NumPy arrays (`.npy` files)

---

### ✅ Prerequisites:
- Python 3.x
- `gdown` (installed via requirements)

---

### 📦 Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

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

- Improve accuracy using transfer learning (e.g., MobileNetV2)
- Deploy on Render/Heroku for public access
- Add face detection and cropping before classification
- Add confidence score to the response

---

## 👨‍💻 Author

Rudraksh Mehra  
📧 mehrarudraksh29070@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/rudraksh-mehra-2025282b2/) | [GitHub](https://github.com/RudrakshMehra)
