import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import numpy as np

# Define constants
IMG_SIZE = 224
BATCH_SIZE = 32

# Define data directories
TRAIN_DIR = "/train"
VALIDATION_DIR = "/validation"
TEST_DIR = "test"

# Define class labels
CLASS_LABELS = ["cat", "dog", "bird", "fish"]

# Load data
train_data = []
train_labels = []
for label in CLASS_LABELS:
    for file in os.listdir(os.path.join(TRAIN_DIR, label)):
        img = Image.open(os.path.join(TRAIN_DIR, label, file))
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 255.0
        train_data.append(img)
        train_labels.append(CLASS_LABELS.index(label))

validation_data = []
validation_labels = []
for label in CLASS_LABELS:
    for file in os.listdir(os.path.join(VALIDATION_DIR, label)):
        img = Image.open(os.path.join(VALIDATION_DIR, label, file))
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 255.0
        validation_data.append(img)
        validation_labels.append(CLASS_LABELS.index(label))

test_data = []
test_labels = []
for label in CLASS_LABELS:
    for file in os.listdir(os.path.join(TEST_DIR, label)):
        img = Image.open(os.path.join(TEST_DIR, label, file))
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img) / 255.0
        test_data.append(img)
        test_labels.append(CLASS_LABELS.index(label))

# Convert data to numpy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)
validation_data = np.array(validation_data)
validation_labels = np.array(validation_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

# Split data into training and validation sets
train_data, validation_data, train_labels, validation_labels = train_test_split(
    train_data, train_labels, test_size=0.2, random_state=42
)

# Define model
model = keras.Sequential(
    [
        keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)
        ),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(len(CLASS_LABELS), activation="softmax"),
    ]
)

# Compile model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train model
model.fit(
    train_data,
    train_labels,
    epochs=10,
    batch_size=BATCH_SIZE,
    validation_data=(validation_data, validation_labels),
)

# Evaluate model
test_loss, test_acc = model.evaluate(test_data, test_labels)
print(f"Test accuracy: {test_acc:.2f}")

# Use model to make predictions
predictions = model.predict(test_data)

# Print predictions
for i, prediction in enumerate(predictions):
    print(f"Image {i+1}: {CLASS_LABELS[np.argmax(prediction)]}")

# Folder structure:
# train/
#     cat/
#         image1.jpg
#         image2.jpg
#         ...
#     dog/
#         image1.jpg
#         image2.jpg
#         ...
#     bird/
#         image1.jpg
#         image2.jpg
#         ...
#     fish/
#         image1.jpg
#         image2.jpg
#         ...
# validation/
#     cat/
#         image1.jpg
#         image2.jpg
#         ...
#     dog/
#         image1.jpg
#         image2.jpg
#         ...
#     bird/
#         image1.jpg
#         image2.jpg
#         ...
#     fish/
#         image1.jpg
#         image2.jpg
#         ...
# test/
#     cat/
#         image1.jpg
#         image2.jpg
#         ...
#     dog/
#         image1.jpg
#         image2.jpg
#         ...
#     bird/
#         image1.jpg
#         image2.jpg
#         ...
#     fish/
#         image1.jpg
#         image2.jpg
#         ...
