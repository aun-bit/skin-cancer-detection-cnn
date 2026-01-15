
import os
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
# Dataset paths
BASE_DIR = "data/melanoma_cancer_dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")

assert os.path.exists(TRAIN_DIR), "❌ Train directory missing"
assert os.path.exists(TEST_DIR), "❌ Test directory missing"
# Data generators
train_gen = ImageDataGenerator(rescale=1./255)
test_gen  = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary"
)

test_data = test_gen.flow_from_directory(
    TEST_DIR,
    target_size=(224,224),
    batch_size=32,
    class_mode="binary",
    shuffle=False # for confusion matrix later
)
# Baseline CNN Model
model = Sequential([
    Conv2D(16, (3,3), activation="relu", input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
# Train
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)

# =========================
# CONFUSION MATRIX
# =========================

# True labels
y_true = test_data.classes

# Predicted probabilities
y_pred_prob = model.predict(test_data)

# Convert probabilities to class labels (0 or 1)
y_pred = (y_pred_prob > 0.5).astype(int).ravel()

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:\n", cm)

# Classification report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=test_data.class_indices.keys(),
    yticklabels=test_data.class_indices.keys()
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Skin Cancer Detection")
plt.tight_layout()
plt.show()


# Save
model.save("baseline_skin_cancer_model.h5")

with open("baseline_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✅ Baseline model saved")
