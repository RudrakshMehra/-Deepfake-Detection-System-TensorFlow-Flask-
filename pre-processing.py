import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Parameters
DATASET_DIR = 'Dataset'
IMG_SIZE = 128

X = []
y = []

# Function to load and process images from a folder
def load_images_from_folder(folder_path, label):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                X.append(img)
                y.append(label)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

# Load real images
load_images_from_folder(os.path.join(DATASET_DIR, 'real'), label=0)

# Load fake images
load_images_from_folder(os.path.join(DATASET_DIR, 'fake'), label=1)

# Convert and normalize
X = np.array(X) / 255.0
y = np.array(y)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save processed data
np.save('dataset/X_train.npy', X_train)
np.save('dataset/X_test.npy', X_test)
np.save('dataset/y_train.npy', y_train)
np.save('dataset/y_test.npy', y_test)

print(f"✅ Done! Loaded {len(X)} images.")
