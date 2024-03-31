import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tensorflow as keras

# 0 is for webcam
cap=cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier =Classifier("C:/Users/muska/Downloads/Sign_Detection/Model/models.h5" ,"C:/Users/muska/Downloads/Sign_Detection/Model/label.txt")

offset = 20
imgSize=300

# Read labels from the file
with open("C:/Users/muska/Downloads/Sign_Detection/Model/label.txt", "r") as file:
    labels = file.read().splitlines()

while True:
    success, img = cap.read()
    
    imgoutput = img.copy()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgcrop = img[y-offset:y + h+ offset, x-offset:x+ w + offset]
        
        ratio = h/w
        
        if ratio > 1:
            k = imgSize/h
            wcal = math.ceil(k*w)
            imgResize = cv2.resize(imgcrop, (wcal, imgSize))
            imgResizeShape = imgResize.shape
            WGap = math.ceil((imgSize-wcal)/2)
            imgWhite[:, WGap:wcal+WGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        else:
            k = imgSize/h
            hcal = math.ceil(k*h)
            imgResize = cv2.resize(imgcrop, (hcal, imgSize))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hcal)/2)
            imgWhite[hGap: hGap+hcal, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        
        cv2.rectangle(imgoutput, (x-offset, y-offset-50), (x-offset+90, y-offset), (255,0,255), cv2.FILLED)
        cv2.putText(imgoutput, labels[index], (x, y-26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255,255,255), 2)
        cv2.rectangle(imgoutput, (x-offset, y-offset), (x+offset+w, y + h + offset), (255,0,255), 4) 
    
    cv2.imshow("Image", imgoutput)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
