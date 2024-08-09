import cv2
import numpy as np
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
import math

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="C:/Users/muska/Downloads/Sign_Detection/Model/models.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Read labels from the file
with open("C:/Users/muska/Downloads/Sign_Detection/Model/label.txt", "r") as file:
    labels = file.read().splitlines()

# Initialize hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

def predict_tflite(image):
    # Preprocess the image to fit the model's input size
    image = cv2.resize(image, (224, 224))  # Resize if necessary
    image = image.astype(np.float32) / 255.0  # Normalize the image

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(image, axis=0))

    # Run the inference
    interpreter.invoke()

    # Get the output tensor
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

while True:
    success, img = cap.read()
    imgoutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Crop the hand from the frame
        imgcrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        if imgcrop.size != 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            ratio = h / w

            if ratio > 1:
                k = imgSize / h
                wcal = math.ceil(k * w)
                imgResize = cv2.resize(imgcrop, (wcal, imgSize))
                WGap = (imgSize - wcal) // 2
                imgWhite[:, WGap:WGap + wcal] = imgResize
            else:
                k = imgSize / w
                hcal = math.ceil(k * h)
                imgResize = cv2.resize(imgcrop, (imgSize, hcal))
                HGap = (imgSize - hcal) // 2
                imgWhite[HGap:HGap + hcal, :] = imgResize

            # Predict using TFLite model
            prediction = predict_tflite(imgWhite)
            index = np.argmax(prediction)

            cv2.rectangle(imgoutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgoutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgoutput, (x - offset, y - offset), (x + offset + w, y + h + offset), (255, 0, 255), 4)

    cv2.imshow("Image", imgoutput)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' key to close the window
        break

cap.release()
cv2.destroyAllWindows()
