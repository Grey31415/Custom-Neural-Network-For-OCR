import cv2
import string
import imutils
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from imutils.contours import sort_contours
from operator import itemgetter

#Tolerances for width and height of contours
wMin = 20
wMax = 300
hMin = 70
hMax = 300

#Load the trained neural network
model = load_model('ANN_OCR_V3.h5')

#Set parameters for camera stream
cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

#Converts numerical predictions of the network to letters
nr_to_letter = {k:v.upper() for k,v in enumerate(list(string.ascii_lowercase))}

#Camera feed loop
while True:
    check, frame = cap.read()
    cv2.imshow("Camera Feed", frame)
    key = cv2.waitKey(1)
    if key != -1:
        cv2.imwrite("input" + ".jpg", frame)
        break

#Read the image taken (load)
image = cv2.imread('input.jpg')

#Image preprocessing (grayscale, blurr, binary thresholding, edge detection)
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(grey, (5, 5), 0)
ret, im_th = cv2.threshold(blurred, 100 , 500 , cv2.THRESH_BINARY_INV)
edged = cv2.Canny(im_th, 30, 150)

#Contour recognition and sorting (left to right)
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sort_contours(cnts, method="left-to-right")[0]

#Initiation of the prediction arrays
chars = []
completePrediction = []

#Contour selection (Threshholds, size), resizing of the selected area (padding) 
#and prediction with neural network => looped for every contour in 'cnts'
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)

	if (w >= wMin and w < wMax) and (h >= hMin and h <= hMax):
		cv2.rectangle(image,(x,y),(x+w,y+h),(0, 255, 0), 3 )
		roi = grey[y:y + h, x:x + w]
		thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		(tH, tW) = thresh.shape
		
		if tW > tH:
			thresh = imutils.resize(thresh, width=28)
	
		else:
			thresh = imutils.resize(thresh, height=28)

		(tH, tW) = thresh.shape
		dX = int(max(0, 28 - tW) / 2.0)
		dY = int(max(0, 28 - tH) / 2.0)

        #Adds padding (= Black borders) to make selected area square 
        #without stretching the original image
		padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX,
                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

        #Adds extra black border for better recognition around entire image
		padded = cv2.copyMakeBorder(padded, top=3, bottom=3, left=3, right=3,
                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

        #Resizing of the Imgae to 28x28 pixels and reconfiguration 
        #of the image meet network requirements
		padded = cv2.resize(padded, (28, 28))
		padded = padded.astype("float32") / 255.0
		padded = np.expand_dims(padded, axis=-1)

        #Adds selected areas to the 'chars' array with their original position
		chars.append((padded, (x, y, w, h)))

		padded = np.reshape(padded, (-1, 28, 28, 1))

		#Character Prediction (prediction as number => converted to letter)
		prediction = np.argmax(model.predict(padded), axis=1)

        #Adds letter prediction to image
		cv2.putText(image, str(nr_to_letter[prediction[0]]), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6, cv2.LINE_AA)


		completePrediction.append((x, (str(nr_to_letter[prediction[0]]))))


completePrediction = sorted(completePrediction, key=itemgetter(0))

completePredictionStr = ""

for item in completePrediction:
    completePredictionStr += item[1]

#cv2.putText(image, completePredictionStr, (int(x/2), y - 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, ), 3, cv2.LINE_AA)

print(completePredictionStr)

#plt.imshow(image)
#plt.show()
 

cv2.imwrite("output.jpg", image)
cv2.imshow("IMAGE", image)
cv2.waitKey(0)
cv2.destroyAllWindows()