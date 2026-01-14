from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import pickle
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
    class_mode="binary"
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
# Save
model.save("baseline_skin_cancer_model.h5")

with open("baseline_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✅ Baseline model saved")
