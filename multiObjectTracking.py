import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', type=str, help='Path to input video file')
ap.add_argument('-t', '--tracker', type=str,default='kcf', help='OpenCV object tracker type')
args = vars(ap.parse_args())

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.legacy.TrackerCSRT_create,
	"kcf": cv2.legacy.TrackerKCF_create,
	"boosting": cv2.legacy.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.legacy.TrackerTLD_create,
	"medianflow": cv2.legacy.TrackerMedianFlow_create,
	"mosse": cv2.legacy.TrackerMOSSE_create
}
trackers = cv2.legacy.MultiTracker_create()

# No video command, use webcam
if not args.get('video', False):
	print("[INFO] starting video stream. . .")
	vs = cv2.VideoCapture(0)
	time.sleep(1.0)
else:
	vs = cv2.VideoCapture(args['video'])

while True:
	ret, frame = vs.read()
	if frame is None:
		break
	frame = imutils.resize(frame, width=400)

	success, boxes = trackers.update(frame)

	for box in boxes:
		x,y,w,h = [int(v) for v in box]
		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
	cv2.imshow('Frame', frame)

	key = cv2.waitKey(1) & 0xFF
	# If 's' key selected, then select a bounding box to track, add to trackers
	if key == ord('s'):
		box = cv2.selectROI('Frame', frame, fromCenter=False, showCrosshair=True)
		tracker = OPENCV_OBJECT_TRACKERS[args['tracker']]()
		trackers.add(tracker, frame, box)
	# If 'q' key selected, quit
	elif key == ord('q'):
		break

# Release the pointer to video
vs.release()
cv2.destroyAllWindows()
