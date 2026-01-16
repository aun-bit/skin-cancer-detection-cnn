# train.py
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from model import build_baseline_model
from utils.data_loader import load_data

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

BASE_DIR = "data/melanoma_cancer_dataset"

train_data, test_data = load_data(BASE_DIR)

model = build_baseline_model()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=20,
    callbacks=[early_stop]
)

model.save("baseline_skin_cancer_model.h5")

with open("baseline_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

print("✅ Training complete and model saved")
