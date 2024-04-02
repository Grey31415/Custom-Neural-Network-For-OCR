import cv2
import string
import imutils
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
from operator import itemgetter

model = load_model("/Users/greysonwiesenack/Desktop/Programming/vscode/Python/AI/OCR_kaggleAZ_Network_generation:deployment/Neural Networks/PT_2_V1.h5")

nr_to_letter = {k:v.upper() for k,v in enumerate(list(string.ascii_lowercase))}

def prepImg(data):
    return cv2.resize(data,(28,28)).reshape(28,28,1)/255.0

image = cv2.imread("/Users/greysonwiesenack/Desktop/Programming/vscode/Python/AI/OCR_kaggleAZ_Network_generation:deployment/Testimages/input 13.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
ret, im_th = cv2.threshold(blurred, 140 , 500 , cv2.THRESH_BINARY_INV)
#cv2.imshow('imageTH', im_th)
#cv2.waitKey(0)
edged = cv2.Canny(im_th, 30, 150)
#cv2.imshow('imageEG', edged)
#cv2.waitKey(0)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]
# initialize the list of contour bounding boxes and associated
# characters that we'll be OCR'ing
chars = []
completePrediction = []

# loop over the contours
for c in cnts:

	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0), int(w/50))

	if (w >= 5 and w <= 700) and (h >= 15 and h <= 700):

		roi = gray[y:y + h, x:x + w]
		thresh = cv2.threshold(roi, 0, 255,
			cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		(tH, tW) = thresh.shape
		# if the width is greater than the height, resize along the
		# width dimension
		if tW > tH:
			thresh = imutils.resize(thresh, width=32)
		# otherwise, resize along the height
		else:
			thresh = imutils.resize(thresh, height=32)
			
		(tH, tW) = thresh.shape
		dX = int(max(0, 28 - tW) / 2.0)
		dY = int(max(0, 28 - tH) / 2.0)
		padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
        left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
		padded = cv2.resize(padded, (28, 28))
		padded = padded.astype("float32") / 255.0
		padded = np.expand_dims(padded, axis=-1)
		chars.append((padded, (x, y, w, h)))

		#plt.imshow(padded)
		#plt.show()
		
boxes = [b[1] for b in chars]
chars = np.array([c[0] for c in chars], dtype="float32")
# OCR the characters using our handwriting recognition model
preds = model.predict(chars)
# define the list of label names
labelNames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

for (pred, (x, y, w, h)) in zip(preds, boxes):
	# find the index of the label with the largest corresponding
	# probability, then extract the probability and label
	i = np.argmax(pred)
	prob = pred[i]
	label = labelNames[i]
	# draw the prediction on the image
	print("[INFO] {} - {:.2f}%".format(label, prob * 100))
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
	cv2.putText(image, label, (x - 10, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
	# show the image
	cv2.imshow("Image", image)
	cv2.waitKey(0)