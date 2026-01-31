# -----------------------------
# INSTALL REQUIRED PACKAGES
# -----------------------------
# pip install tensorflow pillow matplotlib

# -----------------------------
# IMPORTS
# -----------------------------
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1️⃣ CONFIG
# -----------------------------
data_dir = "data"  # your dataset folder
img_height = 224
img_width = 224
batch_size = 32
epochs = 15

# -----------------------------
# 2️⃣ LOAD DATA (ALL IMAGES)
# -----------------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Classes:", class_names)

# -----------------------------
# 3️⃣ DATA AUGMENTATION
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# -----------------------------
# 4️⃣ BUILD MODEL
# -----------------------------
num_classes = len(class_names)

model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    data_augmentation,
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# 5️⃣ TRAIN MODEL
# -----------------------------
history = model.fit(
    train_ds,
    epochs=epochs
)

# -----------------------------
# 6️⃣ SAVE MODEL
# -----------------------------
model.save("cattle_breed_classifier.h5")

# -----------------------------
# 7️⃣ OPTIONAL: PLOT TRAINING HISTORY
# -----------------------------
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
