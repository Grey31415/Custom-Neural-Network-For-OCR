import cv2 
import imutils
import numpy

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

def sort_contours(cnts, method='left -to-right'):
    reverse = False
    i = 0 
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes)), key=lambda b:b[1][i], reverse=reverse)

    return (cnts, boundingBoxes)

while True:
    #check, frame = cap.read()
    frame = cv2.imread("pic.jpg")
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #cnts = sort_contours(cnts, method='left-to-right')[0]
    
    chars = []



    cv2.imshow('Video', edged)


    key = cv2.waitKey(1)
    if key != -1:
        break
    

cap.release()
cv2.destroyAllWindows()
    
