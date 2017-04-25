import cv2 as cv
import numpy as np
from PIL import Image
detector= cv.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv.VideoCapture(0)

id=raw_input("Enter ID");
sampleNumber=0;
while(True):
	ret, img = capture.read()
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	faces = detector.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		sampleNumber=sampleNumber+1;
		cv.imwrite("dataset/user_"+str(id)+"_"+str(sampleNumber)+".jpg",cv.resize(gray[y:y+h,x:x+w],(250,250)))
		cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		cv.waitKey(500)
	cv.imshow('frame',img)
	cv.waitKey(1)
	if(sampleNumber>20):
		break
capture.release()
cv.destroyAllWindows()
