#OpenCV 2 face detection example programmed by Greyson WIESENACK
#
#Uses the 'haarcascade_frontalface_default' trained model
#
#Press 'q' when running to exit & shutdown kernel in JN

import cv2
import os

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)

while True:

    try:
        check, frame = cap.read()
        cv2.imshow("Camera Feed", frame)
        key = cv2.waitKey(1)
        if key != -1:
            cv2.imwrite("pic" + ".jpg", frame)
            break
 
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

    
    