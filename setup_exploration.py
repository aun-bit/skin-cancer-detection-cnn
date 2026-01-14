# ===============================
# 1. INSTALL REQUIRED LIBRARIES
# ===============================
# Run this ONCE if libraries are missing
# Comment it out after successful install

import sys
import subprocess

libs = [
    "tensorflow",
    "opencv-python",
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn"
]

for lib in libs:
    subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

print("✅ All libraries installed")

# ===============================
# 2. IMPORT LIBRARIES
# ===============================
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===============================
# 3. DATASET PATH (EDIT IF NEEDED)
# ===============================
DATASET_DIR = "data/melanoma_cancer_dataset"

train_dir = os.path.join(DATASET_DIR, "train")
val_dir   = os.path.join(DATASET_DIR, "test")

assert os.path.exists(train_dir), "❌ Training directory not found"
assert os.path.exists(val_dir), "❌ Validation directory not found"

print("✅ Dataset paths verified")

# ===============================
# 4. DATA GENERATORS
# ===============================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary"
)

# ===============================
# 5. MODEL DEFINITION
# ===============================
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================
# 6. TRAIN MODEL
# ===============================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# ===============================
# 7. SAVE MODEL & HISTORY
# ===============================
import pickle

model.save("skin_cancer_model.h5")

with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✅ Model & training history saved")

