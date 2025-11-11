import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Path to your dataset folders
data_dir = "data"
img_size = (48, 48)  # FER images are typically 48x48
batch_size = 32

# Step 1: Load images using ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,          # normalize images
    validation_split=0.2     # 20% for validation
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",  # change if your images are RGB
    class_mode="categorical",
    subset="training"
)

val_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

# Step 2: Build a simple CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_generator.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 3: Train and save the model
checkpoint = ModelCheckpoint("face_emotionModel.h5", save_best_only=True, monitor="val_accuracy", mode="max")
history = model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[checkpoint]
)

print("Training complete. Best model saved as face_emotionModel.h5")
