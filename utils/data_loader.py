# utils/data_loader.py
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(base_dir, batch_size=32):
    train_dir = os.path.join(base_dir, "train")
    test_dir  = os.path.join(base_dir, "test")

    assert os.path.exists(train_dir), "Train directory missing"
    assert os.path.exists(test_dir), "Test directory missing"

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode="binary"
    )

    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=(224,224),
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    return train_data, test_data
