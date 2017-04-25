import cv2
import os
import numpy as np
import sys
from PIL import Image
from datetime import datetime

if(len(sys.argv)==1):
	print "Enter path: ex - python eigenTrainer.py <path of folder to read>"
	sys.exit()
path = sys.argv[1]

detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec=cv2.createFisherFaceRecognizer()
rec.load('recognizer/trainingData.yml')
imgId=0
imgPath=[]
total = 0
corr = 0
tTime=datetime.now()
x=tTime
def getImageName(iPath):
	global total
	global corr
	global tTime
	for path, subdirs, files in os.walk(iPath):
	    for name in files:
	        imgPath.append(os.path.join(path, name))
	faceId=[]
	imgId=[]
	tmpId=1
	for imagePath in imgPath:
		dataImg = cv2.imread(imagePath,cv2.CV_LOAD_IMAGE_GRAYSCALE)
		total=total+1
		start=datetime.now()		
		label,conf=rec.predict(dataImg)
		end=datetime.now()
		diff=end-start
		tTime=tTime+diff
		print diff
		val=imagePath.split('/')[1].split('s')[1]
		if(int(label)==int(val)):
			corr=corr+1
		if(cv2.waitKey(1)==ord('q') or cv2.waitKey(1)==ord('e')):
			break;

getImageName(path)
print total
print corr
print "% of correct faces detected: {}, Time taken: {}".format(corr*100/total, tTime-x)
cv2.destroyAllWindows()


