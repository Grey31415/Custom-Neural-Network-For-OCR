import cv2
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout

model = load_model("/Users/greysonwiesenack/Desktop/Programming/Python/AI/OCR_kaggleAZ_Network_generation:deployment/Neural Networks/PT_3_V1.h5")

nr_to_letter = {k:v.upper() for k,v in enumerate(list(string.ascii_lowercase))}

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

def prepImg(data):
    return cv2.resize(data,(28,28)).reshape(28,28,1)/255.0


img = cv2.imread("/Users/greysonwiesenack/Desktop/Programming/Python/AI/OCR_kaggleAZ_Network_generation:deployment/Documentation/Image Processing/Screenshot 2022-05-30 at 18.09.43 copy.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5) , 0)
ret, im_th = cv2.threshold(blur, 180 , 400 , cv2.THRESH_BINARY_INV)
ctrs , hier = cv2.findContours(im_th.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

cv2.imshow('Threshhold',im_th)
cv2.waitKey(0)

from tensorflow.keras.preprocessing.image import img_to_array
for x,y,w,h in rects :
    
    if y>=3:
        y-=3
    else :
        y=0
    if x>=3:
        x-=3
    else:
        x=0
    w+=3
    h+=3

    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    sliced = im_th[y:y+h,x:x+w]
    sliced = img_to_array(sliced,dtype='float32')
    sliced = prepImg(sliced)
    sliced = np.expand_dims(sliced , axis = 0)
    prediction = np.argmax(model.predict(sliced), axis=-1)
    cv2.putText(img, str(nr_to_letter[prediction[0]]), (x+w,y+int(h/2)), cv2.FONT_HERSHEY_SIMPLEX ,  1, (0,255,0) , 2, cv2.LINE_AA) 

cv2.imshow("IMAGE",img)
cv2.waitKey(0)
cv2.destroyAllWindows()