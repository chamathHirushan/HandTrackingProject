import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
pTime = 0  # Previous time
cTime = 0  # Current time

cap = cv2.VideoCapture(0)  # Capture from webcam
detector = htm.handDetector()

while True:
    success, img = cap.read()  # Read a frame
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[0])    # change the id for landmark position you want

    cTime = time.time()  # Logic to calculate FPS
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)
    cv2.imshow("image", img)  # Show the image
    cv2.waitKey(1)  # Wait for 1 ms
