import os
import cv2 as cv
import numpy as np
from PIL import Image

imgPath=[]
recognizer=cv.createFisherFaceRecognizer();
def getImageName(iPath):
	
	imgPath = [os.path.join(iPath,imgid) for imgid in os.listdir(iPath)]
	faceId=[]
	imgId=[]
	#tmpId=1
	for imagePath in imgPath:
		print imagePath
		dataImg = Image.open(imagePath).convert('L')
		faceNp=np.array(dataImg,'uint8')
		getId=int(os.path.split(imagePath)[-1].split('_')[1])
		faceId.append(faceNp)
		imgId.append(getId)
		cv.imshow("training",faceNp)
		cv.waitKey(10)
	return faceId, imgId

faces, imgIds = getImageName("dataset")
recognizer.train(faces,np.array(imgIds))
recognizer.save('recognizer/trainingData.yml')
cv.destroyAllWindows()
