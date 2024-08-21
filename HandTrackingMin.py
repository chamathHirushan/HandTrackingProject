import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # capture from webcam 0

mpHands = mp.solutions.hands
hands = mpHands.Hands()  # this only takes RGB color images as inputs
mpDraw = mp.solutions.drawing_utils

pTime = 0  # previous time
cTime = 0  # current time

while True:
    success, img = cap.read()  # read a frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # get the RGB Image
    results = hands.process(imgRGB)

    # print(results.multi_hand_landmarks)

    if (results.multi_hand_landmarks):
        for handLandMarks in results.multi_hand_landmarks:  # extract the information of each hand
            for id, lm in enumerate(handLandMarks.landmark):

                h, w, c = img.shape  # height width chanel of the image
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)

                # if id == 4:
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            # draw on the original image
            mpDraw.draw_landmarks(img, handLandMarks, mpHands.HAND_CONNECTIONS)

    cTime = time.time()  # logic to calculate the fps
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

    cv2.imshow("image", img)  # show the image
    cv2.waitKey(1)  # wait for 1ms
