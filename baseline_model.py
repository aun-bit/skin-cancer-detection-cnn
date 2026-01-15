# baseline_model_final_tracked.py
# Finalized baseline CNN for skin cancer detection
# Previous lines are preserved as comments for tracking

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# Dataset paths
# =========================
BASE_DIR = "data/melanoma_cancer_dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR  = os.path.join(BASE_DIR, "test")

assert os.path.exists(TRAIN_DIR), "Error: Train directory missing"
assert os.path.exists(TEST_DIR), "Error: Test directory missing"

# =========================
# Data generators
# =========================
# Previous simple generator (commented)
# train_gen = ImageDataGenerator(rescale=1./255)
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Previous test generator (kept)
# test_gen  = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

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
    shuffle=False  # for confusion matrix later
)

# =========================
# Baseline CNN Model
# =========================
# Previous simple baseline (commented)
"""
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
"""

# Improved baseline with BatchNormalization & Dropout
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# Training
# =========================
# Previous training code (commented)
"""
history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=5
)
"""

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=20,
    callbacks=[early_stop]
)

# =========================
# Plot training accuracy & loss
# =========================
plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()

# =========================
# Evaluate on test data
# =========================
# Previous confusion matrix code (kept as comment)
"""
y_true = test_data.classes
y_pred_prob = model.predict(test_data)
y_pred = (y_pred_prob > 0.5).astype(int).ravel()
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))
"""

y_true = test_data.classes
y_pred_prob = model.predict(test_data)
y_pred = (y_pred_prob > 0.5).astype(int).ravel()

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

class_labels = list(test_data.class_indices.keys())
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Plot and save confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix – Skin Cancer Detection")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# =========================
# Save model and history
# =========================
# Previous save code (kept as comment)
"""
model.save("baseline_skin_cancer_model.h5")
with open("baseline_history.pkl", "wb") as f:
    pickle.dump(history.history, f)
"""

model.save("baseline_skin_cancer_model.h5")
with open("baseline_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("Baseline model saved successfully (tracked version)")
