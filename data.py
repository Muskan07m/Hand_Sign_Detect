import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# 0 is for webcam
cap=cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize=300

folder ="C:/Users/muska/Downloads/Sign_Detection/Data/F"
if not os.path.exists(folder):
    os.makedirs(folder)
counter = 0

while True:
    success, img = cap.read()
    hands, img=detector.findHands(img)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        
        imgWhite = np.ones((imgSize, imgSize ,3),np.uint8)*255
        imgcrop = img[y-offset:y + h+ offset ,x-offset:x+ w + offset]
        
        ratio =h/w
        
        if ratio >1:
            k = imgSize/h
            wcal = math.ceil(k*w)
            imgResize = cv2.resize(imgcrop,(wcal,imgSize))
            imgResizeShape = imgResize.shape
            WGap = math.ceil((imgSize-wcal)/2)
            imgWhite[: , WGap:wcal+WGap] =imgResize
            
        else:
            k = imgSize/h
            hcal = math.ceil(k*h)
            imgResize = cv2.resize(imgcrop,(hcal,imgSize))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hcal)/2)
            imgWhite[hGap : hGap+hcal , :] =imgResize
            
        
        cv2.imshow("Imagewhite" , imgWhite)
        
    cv2.imshow("Image" , img)
    key = cv2.waitKey(1)
    
    try:
        
      if key == ord("s"):
           counter +=1
           cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
           print(counter)
           print(f"Image saved: {counter}")
    except Exception as e:
       print(f"Failed to save image: {e}")
         
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
    