
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from captchahelper import preprocess
from imutils import contours
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os


def capture_digit(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def main(input_digit):
    capture_digit("training_data/" + str(input_digit))
    x, y, w, h = 0, 0, 300, 300
    max_pics = 10
    image_no = 0
    blueUpper = np.array([140,255,255], dtype = "uint8")
    blueLower = np.array([100,50,50], dtype = "uint8")
    cap=cv2.VideoCapture(0)
    if cap.isOpened():
        ret,frame=cap.read()
    else :
        ret = False

    while ret :
        ret,frame=cap.read()
        
        hsv= cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        # Thresholding Range
        
        
        # # Threshold the HSV image to get only blue colors
        blue=cv2.inRange(hsv,blueLower,blueUpper)
        # Gaussian Blur
        blur = cv2.GaussianBlur(blue, (3, 3), 0)
        ret, thresh = cv2.threshold(blur, 175, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = thresh[y:y + h, x:x + w]
        #contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        ans=''
        
        (_, contours, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #print("length:{}".format(len(contours))) 
        if len(contours) > 0:
            for c in contours:
                if cv2.contourArea(c)>200:
                    #contour = max(contours, key=cv2.contourArea)
                    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2 )
                    #print("Area:{}".format(cv2.contourArea(c)))
                    (x1, y1, w1, h1) = cv2.boundingRect(c)
                    roi = thresh[y1:y1+h1, x1:x1+w1]
                    roi = preprocess(roi, 28, 28)
                    image_no+=1
                    cv2.imwrite("Training_Data/" + str(input_digit)+"/"+str(image_no)+ ".jpg",roi)
                    print("Image Count : {}".format(image_no))
                    
                    
                    
                    
                    
                    cv2.rectangle(frame, (x1, y1), (x1+w1, y1+h1), (0,255,0),2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)            
        cv2.imshow('frame', frame)
        cv2.imshow('thresh', thresh)
        if image_no==1000:
            break
        if cv2.waitKey(1)==27:
            break

    cv2.destroyAllWindows()
    cap.release()

input_digit=input('Enter digit to capture :')
main(input_digit)