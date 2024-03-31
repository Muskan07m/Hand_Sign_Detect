import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def load_data_from_folder(folder_path, label):
    images = []
    labels = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = load_img(os.path.join(folder_path, filename), target_size=(224, 224))
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
    
    return np.array(images), np.array(labels)

def train_model(folder_path):
    images, labels = load_data_from_folder(folder_path, 0)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Define CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    optimizer = Adam(learning_rate=0.001)  # Updated optimizer configuration
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Define callbacks
    checkpoint = ModelCheckpoint("model_weight.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping])

    # Save the model
    model.save("mod.h5")

    with open("lab.txt", "w") as file:
        file.write("0:E\n")

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print("Validation Loss:", test_loss)
    print("Validation Accuracy:", test_acc)

if __name__ == '__main__':
    folder_path = "C:/Users/muska/Downloads/Sign_Detection/E"  # Replace with the path to your "E" folder
    train_model(folder_path)
