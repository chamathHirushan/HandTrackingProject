import numpy as np
import cv2
import time
import HandTrackingModule as htm
import math

from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

##############################################
wCam, hCam = 640, 480
##############################################

cap = cv2.VideoCapture(0)  # Capture from webcam
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volBar = 400

while True:
    success, img = cap.read()  # Read a frame

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        #   print(lmList[4], lmList[8])
        #   get the coordinates of the both tips
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        # draw circle for first tip
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        # draw circle for second tip
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        # draw circle in middle
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # draw a line between them
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # calculating the length
        length = math.hypot(x2-x1, y2-y1)

        if length < 50:
            # change the color of the middle if length is lower
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

        # print(length)   # HandRange from 50 to 300

        vol = np.interp(length, [30, 295], [minVol, maxVol])
        volBar = np.interp(length, [30, 295], [400, 150])
        # print(vol)
        volume.SetMasterVolumeLevel(vol, None)

    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)  # outer boarder
    # print(int(volBar))
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0),
                  cv2.FILLED)  # filled volume part

    cTime = time.time()  # Logic to calculate FPS
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    cv2.imshow("Img", img)
    cv2.waitKey(1)
