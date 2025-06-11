
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
├── prep-processing.py      # Data preprocessing script
├── final_model.keras       # Trained model
├── download_dataset.py     # Google Drive auto-download script
├── dataset/                # Preprocessed .npy files (after extraction)
├── Dataset/                # Raw images (if using original data)
├── templates/              # index.html (upload form)
├── static/                 # Uploaded images
├── requirements.txt
└── README.md
```

---

## 📥 Dataset Setup

This project uses a preprocessed `.npy` dataset (~980MB), hosted on Google Drive.

### ✅ Prerequisites:
- Python 3.x
- `gdown` (installed via requirements)

### 📦 Step 1: Install Requirements
```bash
pip install -r requirements.txt
```

Make sure your `requirements.txt` includes:
```
gdown
```

### 🚀 Step 2: Run the Download Script
```bash
python download_dataset.py
```

This will:
- Download `dataset.zip` (~980MB) from Google Drive
- Unzip it into your project root as:
```
./dataset/
├── X_train.npy
├── X_test.npy
├── y_train.npy
└── y_test.npy
```

✅ Your training code will work out-of-the-box with this structure.

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

- Improve accuracy using transfer learning (e.g., MobileNetV2)
- Deploy on Render/Heroku for public access
- Add face detection and cropping before classification
- Add confidence score to the response

---

## 👨‍💻 Author

Rudraksh Mehra  
📧 your-email@example.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)
