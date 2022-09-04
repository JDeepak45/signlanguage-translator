
import math
import time
from cvzone.ClassificationModule import Classifier
import tensorflow
import cv2
import mediapipe
from cvzone.HandTrackingModule import HandDetector
import numpy as np
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
ofset = 20
sixe = 500
classifier = Classifier("model/keras_model.h5", "model/labels.txt")
# folder = "data/9"
counter=0
labels=['A','B','C','zero','one','two','three','four','five','six','seven','eight','nine']
while True:
    success, img = cap.read()
    imgoutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, z, w = hand['bbox']

        imgsize = np.ones((sixe, sixe, 3), np.uint8) * 255
        img1 = img[y-ofset: y + w+ofset, x-ofset:x + z+ofset]

        aspect_ratio = w/z
        if aspect_ratio > 1:
            k = sixe/w
            wcal = math.ceil(k*z)
            imgresize = cv2.resize(img1, (wcal, sixe))
            imgresizeshape = imgresize.shape
            wgap= math.ceil((sixe-wcal)/3)
            imgsize[:, wgap:wcal+wgap] = imgresize

            prediction, index = classifier.getPrediction(imgsize,draw=False)
            # print(prediction,index)
        else:
            k = sixe / z
            hcal = math.ceil(k * w)
            imgresize = cv2.resize(img1, (sixe, hcal))
            imgresizeshape = imgresize.shape
            hgap = math.ceil((sixe - hcal)/3)
            imgsize[hgap:hcal + hgap, :] = imgresize

            prediction, index = classifier.getPrediction(imgsize,draw=False)
        txt=cv2.FONT_HERSHEY_PLAIN
        cv2.putText(imgoutput, labels[index], (x,y-20),txt,2,(255,0,255),2)
        cv2.rectangle(imgoutput, (x-ofset,y-ofset),(x+z+ofset,y+w+ofset),(255,0,255),4)
        cv2.imshow("imgg", img1)
        # cv2.imshow("Img1", imgsize)
        # cv2.waitKey(1)
        # if hands[1] is not None:
        #     hand1 = hands[1]
        #     a, b, c, d = hand1['bbox']
        #     img2 = img[b: b + d, a:a + c]
        #     cv2.imshow("img2", img2)
    cv2.imshow("Image", imgoutput)
    cv2.waitKey(1)