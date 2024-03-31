import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Directory path
data_directory = "C:/Users/muska/Downloads/Sign_Detection/Data" 

# Function to load images and assign labels
def load_data(directory):
    images = []
    labels = []
    label_map = {}
    label_index = 0
    
    for label_name in os.listdir(directory):
        label_dir = os.path.join(directory, label_name)
        
        if os.path.isdir(label_dir):
            label_map[label_index] = label_name
            
            for filename in os.listdir(label_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img = tf.keras.preprocessing.image.load_img(os.path.join(label_dir, filename), target_size=(224, 224))
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    images.append(img_array)
                    labels.append(label_index)
            
            label_index += 1
    
    return images, labels, label_map

# Load data and assign labels
images, labels, label_map = load_data(data_directory)

# Convert labels to one-hot encoding
labels = tf.keras.utils.to_categorical(labels, num_classes=len(label_map))

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

X_train = np.array(X_train)
X_val = np.array(X_val)

X_train = X_train.reshape(-1, 224, 224, 3)
X_val = X_val.reshape(-1, 224, 224, 3)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(label_map), activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint("model_weight.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping])

# Save the model
model.save("C:/Users/muska/Downloads/Sign_Detection/Model/models.h5")

# Save label map
with open("C:/Users/muska/Downloads/Sign_Detection/Model/label.txt", "w") as file:
    for index, label_name in label_map.items():
        file.write(f"{label_name}\n")

# Evaluate the model
test_loss, test_acc = model.evaluate(X_val, y_val)
print("Validation Loss:", test_loss)
print("Validation Accuracy:", test_acc)
