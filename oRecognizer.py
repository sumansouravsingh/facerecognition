import cv2
import os
import numpy as np
import sys
from PIL import Image

detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec=cv2.createFisherFaceRecognizer()
cam=cv2.VideoCapture(0)
rec.load('recognizer/trainingData.yml')
imgId=0

font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,1,1,0,1)
while(True):
	global tTime
	ret,img=cam.read()
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces=detector.detectMultiScale(gray,1.3,5)
	for(x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		imgId,conf=rec.predict(cv2.resize(gray[y:y+h,x:x+w],(250,250)))
		cv2.cv.PutText(cv2.cv.fromarray(img),str(imgId),(x,y+h),font,255)
	cv2.imshow("Face",img)	
	if(cv2.waitKey(1)==ord('q') or cv2.waitKey(1)==ord('e')):
		break;

cam.release()
cv2.destroyAllWindows()


