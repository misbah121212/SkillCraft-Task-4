import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -----------------------------
# Step 1: Define paths
# -----------------------------
DATASET_PATH = "dataset/leapGestRecog/00"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_gesture_model.h5")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# -----------------------------
# Step 2: Prepare dataset
# -----------------------------
img_width, img_height = 128, 128
batch_size = 32
epochs = 10  # You can increase this for higher accuracy

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

labels = list(train_gen.class_indices.keys())
np.save(LABELS_PATH, labels)
print(f"✅ Labels saved: {labels}")

# -----------------------------
# Step 3: Build CNN Model
# -----------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(labels), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# Step 4: Train Model
# -----------------------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

# -----------------------------
# Step 5: Save Model
# -----------------------------
model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
print(f"✅ Training complete! Validation Accuracy: {history.history['val_accuracy'][-1] * 100:.2f}%")
