#AUTHOR : NATHISHWAR
#DATASET : DOWNLOAD FROM KAGGLE(DOGS & CATS)

import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set dataset path (Update the path accordingly)
dataset_path = "path_to_kaggle_dataset/train"  # Change this to your dataset path

# Image Preprocessing Parameters
IMG_SIZE = 64  # Resize all images to 64x64

# Prepare dataset
X, y = [], []

for label, category in enumerate(["cat", "dog"]):
    folder = os.path.join(dataset_path, category)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read in grayscale
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize image
        X.append(img.flatten())  # Flatten image into a vector
        y.append(label)  # 0 for cat, 1 for dog

# Convert lists to NumPy arrays
X = np.array(X) / 255.0  # Normalize pixel values
y = np.array(y)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM Classifier with RBF Kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")
