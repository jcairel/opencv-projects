import mediapipe as mp
import cv2
from math import hypot
import numpy as np
import screen_brightness_control as sbc
import os

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)
Draw = mp.solutions.drawing_utils

vs = cv2.VideoCapture(0)
W, H = None, None
while True:
    ret, frame = vs.read()
    if frame is None:
        break
    if W is None or H is None:
        H, W = frame.shape[:2]
    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    Process = hands.process(frameRGB)
    landmarkList = []
    if Process.multi_hand_landmarks:
        for handlm in Process.multi_hand_landmarks:
            for _id, landmarks in enumerate(handlm.landmark):
                x,y = int(landmarks.x * W), int(landmarks.y * H)
                landmarkList.append([_id, x, y])

            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    if landmarkList:
        # Store coordinates of tib of thumb
        x1, y1 = landmarkList[4][1], landmarkList[4][2]
        # Tip of index finger
        x2, y2 = landmarkList[8][1], landmarkList[8][2]

        # Draw circle on fingertips, then line between them
        cv2.circle(frame, (x1, y1), 7, (0,255,0), cv2.FILLED)
        cv2.circle(frame, (x2, y2), 7, (0,255,0), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (0,255,0), 3)

        L = hypot(x2-x1, y2-y1)

        b_level = np.interp(L, [15, 220], [0, 100])

        #os.system(f"brightness {float(b_level) / 100}")
        print(f"brightness {float(b_level) / 100}")

        #sbc.set_brightness(int(b_level))


    cv2.imshow('Image',frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
vs.release()
cv2.destroyAllWindows()


