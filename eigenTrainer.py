import os
import cv2 as cv
import numpy as np
import sys
from datetime import datetime
from PIL import Image

if(len(sys.argv)==1):
	print "Enter path: ex - python eigenTrainer.py <path of folder to read>"
	sys.exit()
path = sys.argv[1]
imgPath=[]
total=0
recognizer=cv.createEigenFaceRecognizer();
def getImageName(iPath):
	global total
	for path, subdirs, files in os.walk(iPath):
	    for name in files:
	        imgPath.append(os.path.join(path, name))	
	faces=[]
	imgId=[]
	for imagePath in imgPath:
		dataImg = Image.open(imagePath).convert('L')
		faceNp=np.array(dataImg,'uint8')
		getId=int(imagePath.split('/')[1].split('s')[1])
		faces.append(faceNp)
		imgId.append(getId)
		cv.imshow("training",faceNp)
		cv.waitKey(10)
	return faces, imgId


faces, imgIds = getImageName(path)
start=datetime.now()
recognizer.train(faces,np.array(imgIds))
end=datetime.now()
diff=start-end
print diff.microseconds
recognizer.save('recognizer/trainingData.yml')
cv.destroyAllWindows()
