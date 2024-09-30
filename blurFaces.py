import cv2
import argparse
import time
import imutils

# Model for face detection
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', type=str, help='path to optional input video file')
args = vars(ap.parse_args())

# If no video path, use webcam
if not args.get("input", False):
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
    time.sleep(1.0)
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args['video'])

while True:
    ret, frame = vs.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width=500)
    greyscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(greyscale, scaleFactor=1.3, minNeighbors=4)

    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        frame[y:y + h, x:x + w] = cv2.medianBlur(frame[y:y + h, x:x + w], 35)

    cv2.imshow('Face blurred', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

vs.release()
cv2.destroyAllWindows()

