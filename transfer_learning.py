# week3_transfer_learning.py
# Executable version of the Jupyter notebook for skin cancer detection

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

print("✅ Libraries imported!")

# Create data generators
train_gen = ImageDataGenerator(rescale=1./255, 
                               horizontal_flip=True,
                               rotation_range=20,
                               zoom_range=0.2,
                               width_shift_range=0.2,
                               height_shift_range=0.2,
                               validation_split=0.2)  # Use 20% of training data for validation

# Load training and validation data
train_data = train_gen.flow_from_directory(
    'data/melanoma_cancer_dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'  # Use this subset for training
)

val_data = train_gen.flow_from_directory(
    'data/melanoma_cancer_dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'  # Use this subset for validation
)

print("✅ Data loaded!")
print(f"Training samples: {train_data.samples}")
print(f"Validation samples: {val_data.samples}")

# Load MobileNetV2 (already trained on millions of images)
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze it (don't retrain it)
base_model.trainable = False

print("✅ MobileNetV2 loaded!")

# Build transfer learning model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()
print("✅ Model ready!")

print("Training... wait 10-15 minutes")

# Calculate steps per epoch for training and validation
steps_per_epoch = train_data.samples // train_data.batch_size
validation_steps = val_data.samples // val_data.batch_size

history = model.fit(
    train_data,
    steps_per_epoch=steps_per_epoch,
    epochs=10,
    validation_data=val_data,
    validation_steps=validation_steps
)

print("✅ Training done!")

# Plot training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Training')
plt.plot(epochs, val_acc, 'r', label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training')
plt.plot(epochs, val_loss, 'r', label='Validation')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_results.png')
plt.show()

print(f"Week 3 Training Accuracy: {acc[-1]*100:.2f}%")
print(f"Week 3 Validation Accuracy: {val_acc[-1]*100:.2f}%")

# Load test data
test_gen = ImageDataGenerator(rescale=1./255)

test_data = test_gen.flow_from_directory(
    'data/melanoma_cancer_dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print(f"Test samples: {test_data.samples}")

# Evaluate
test_steps = test_data.samples // test_data.batch_size
test_loss, test_acc = model.evaluate(test_data, steps=test_steps)

print(f"\nWeek 3 Test Accuracy: {test_acc*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")

# Make predictions on test data
print("\nMaking predictions on test data...")
test_data.reset()  # Reset generator
predictions = model.predict(test_data, steps=test_steps)
predicted_classes = (predictions > 0.5).astype("int32")

# Get true labels
true_classes = test_data.classes
class_labels = list(test_data.class_indices.keys())

# Calculate and display confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, 
            yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

# Save the model
model.save('skin_cancer_model.h5')
print("✅ Model saved as 'skin_cancer_model.h5'")

# Display sample predictions
print("\nSample predictions from test set:")
sample_indices = np.random.choice(len(predictions), 5, replace=False)
for idx in sample_indices:
    actual = class_labels[true_classes[idx]]
    predicted = class_labels[predicted_classes[idx][0]]
    confidence = predictions[idx][0] * 100 if predicted_classes[idx][0] == 1 else (1 - predictions[idx][0]) * 100
    print(f"Sample {idx}: Actual = {actual}, Predicted = {predicted}, Confidence = {confidence:.2f}%")