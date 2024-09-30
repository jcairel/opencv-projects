from imutils.video import FPS
import argparse
import imutils
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', type=str, help='Path to video file')
ap.add_argument('-t', '--tracker', type=str, default='kcf', help='OpenCV object tracker type')
args = vars(ap.parse_args())


# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]
# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
	tracker = cv2.Tracker_create(args["tracker"].upper())
# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
	OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.legacy.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.legacy.TrackerTLD_create,
		"medianflow": cv2.legacy.TrackerMedianFlow_create,
		"mosse": cv2.legacy.TrackerMOSSE_create
	}
	tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
initBB = None

# No video command, use webcam
if not args.get('video', False):
	print("[INFO] starting video stream. . .")
	vs = cv2.VideoCapture(0)
	time.sleep(1.0)
# get video file
else:
	vs = cv2.VideoCapture(args['video'])

fps = None

# Loop over all frames in video
while True:
	ret, frame = vs.read()
	if frame is None:
		break
	frame = imutils.resize(frame, width=500)
	H,W = frame.shape[:2]

	if initBB is not None:
		# grab new bounding box coordinates
		success,box = tracker.update(frame)
		if success:
			x,y,w,h = [int(v) for v in box]
			cv2.rectangle(frame, (x, y), (x+w,y+h), (0,255,0), 2)

			fps.update()
			fps.stop()
			info = [('Tracker', args['tracker']),
					('Success', 'Yes' if success else 'No'),
					('FPS', "{:.2f}".format(fps.fps()))]
			for i,(k,v) in enumerate(info):
				text = "{}: {}".format(k,v)
				cv2.putText(frame,text,(10, H - ((i * 20) + 20)),
							cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
	cv2.imshow('Frame', frame)
	key = cv2.waitKey(1) & 0xFF

	# If 's' key selected, then select a bounding box to track
	if key == ord('s'):
		# Select bounding box to track then press ENTER or SPACE
		initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
		tracker.init(frame, initBB)
		fps = FPS().start()
	# If 'q' key selected, quit
	elif key == ord('q'):
		break
# Release the pointer to video
vs.release()
cv2.destroyAllWindows()

