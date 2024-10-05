from centroidtracker import CentroidTracker
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


class TrackableObject:
    def __init__(self, objectID, centroid):
        self.objectID = objectID
        self.centroids = [centroid]
        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False


ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', type=str, required=True, help='path to input video file')
ap.add_argument("-c", "--confidence", type=float, default=0.4,
                help="minimum probability to filter weak detections")
ap.add_argument("-s", '--skipFrames', type=int, default=30,
                help="# of skip frames between detections")
args = vars(ap.parse_args())

model = "mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
prototxt = "mobilenet_ssd/MobileNetSSD_deploy.prototxt"

# Class labels MobileNet SSD was trained to detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# If no video path, use webcam
if args.get("input", False):
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
    time.sleep(1.0)
else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args['video'])

W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
totalDown, totalUp = 0, 0
fps = FPS().start()

while True:
    ret, frame = vs.read()
    if frame is None:
        break
    # Resize image and convert color for dlib
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if W is None or H is None:
        H, W = frame.shape[:2]

    status = 'Waiting'
    rects = []
    # Only perform detection on set intervals
    if totalFrames % args['skipFrames'] == 0:
        status = 'Detecting'
        trackers = []
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W,H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < args['confidence']:
                continue
            classIdx = int(detections[0, 0, i, 1])
            if CLASSES[classIdx] != 'person':
                continue
            # Rescale coordinates for bounding box
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            x1, y1, x2, y2 = box.astype('int')

            # convert to dlib objects and start correlation tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(x1, y1, x2, y2)
            tracker.start_track(rgb, rect)
            trackers.append(tracker)
    # Not on a detection frame
    else:
        for tracker in trackers:
            status = 'Tracking'
            tracker.update(rgb)
            pos = tracker.get_position()
            x1, x2 = int(pos.left()), int(pos.right())
            y1, y2 = int(pos.top()), int(pos.bottom())
            rects.append((x1, y1, x2, y2))

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'

        cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
        objects = ct.update(rects)

        for objectID, centroid in objects.items():
            to = trackableObjects.get(objectID, None)
            if to is None:
                to = TrackableObject(objectID, centroid)
                trackableObjects[objectID] = to
                continue
            # Difference between current and previous y coord determines direction of motion
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            if not to.counted:
                # If direction is up and centroid is above the center line
                if direction < 0 and centroid[1] < H // 2:
                    totalUp += 1
                    to.counted = True
                elif direction > 0 and centroid[1] > H // 2:
                    totalDown += 1
                    to.counted = True

            trackableObjects[objectID] = to
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    info = [
        ("Up", totalUp),
        ("Down", totalDown),
        ("Status", status),
    ]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # Increment the total number of frames processed
    totalFrames += 1
    fps.update()

# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
vs.release()
cv2.destroyAllWindows()
