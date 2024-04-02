#OpenCV camera programm with photo-taking function for neural network deployment by Greyson Wiesenack

#Paste under imports
#[INFO]: Sets camera and resolution of video stream

cap = cv2.VideoCapture(0)
cap.set(3, 1080)
cap.set(4, 720)


#Paste between function definitions and image processing
#[INFO]: Opens camera feed window. When a key is pressed => captures image and saves it as 'input.jpg'

while True:
    check, frame = cap.read()
    cv2.imshow("Camera Feed", frame)
    key = cv2.waitKey(1)
    if key != -1:
        cv2.imwrite("input" + ".jpg", frame)
        break


#Paste in image processing, use 'img' as input for network deployment
#[Info]: Sets input variable 'img' to the picture taken in the camera feed window ('input.jpg')

img = cv2.imread("input.jpg")