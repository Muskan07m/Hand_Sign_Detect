import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def load_data_from_folders(root_folder):
    images = []
    labels = []
    label_map = {}

    for index, folder_name in enumerate(os.listdir(root_folder)):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            label_map[index] = folder_name  # Mapping each folder to a unique label
            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img = load_img(os.path.join(folder_path, filename), target_size=(224, 224))
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(index)
    
    return np.array(images), np.array(labels), label_map

def train_and_convert_model(root_folder):
    images, labels, label_map = load_data_from_folders(root_folder)

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
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define callbacks
    checkpoint = ModelCheckpoint("model_weight.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train the model
    history = model.fit(X_train, y_train, epochs=1, batch_size=32, validation_data=(X_val, y_val), callbacks=[checkpoint, early_stopping])

    # Convert the model to TensorFlow Lite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model
    tflite_model_path = "C:/Users/muska/Downloads/Sign_Detection/Model/models.tflite"
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    # Save label map
    with open("C:/Users/muska/Downloads/Sign_Detection/Model/label.txt", "w") as file:
        for index, label_name in label_map.items():
            file.write(f"{label_name}\n")
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_val, y_val)
    print("Validation Loss:", test_loss)
    print("Validation Accuracy:", test_acc)

if __name__ == '__main__':
    folder_path = "C:/Users/muska/Downloads/Sign_Detection/Data"  
    train_and_convert_model(folder_path)
