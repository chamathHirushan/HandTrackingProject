import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # capture from webcam 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # only takes RGB color images as inputs
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()  # read a frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # get the RGB Image
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)

    if (results.multi_hand_landmarks):
        for handLandMarks in results.multi_hand_landmarks:  # extract the information of each hand
            # draw on the original image
            mpDraw.draw_landmarks(img, handLandMarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow("image", img)  # show the image
    cv2.waitKey(1)  # wait for 1ms
