#!/usr/bin/env python
# import the necessary packages
from __future__ import print_function
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
from threading import Thread

from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
        help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] camera sensor warming up...")

#vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)

# 400x225 to 1024x576
frame_width = 400
frame_height = 225

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
#out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

# loop over the frames from the video stream
#2D image points. If you change the image, you need to change vector
image_points = np.array([
                            (259, 291),     # Nose tip 34
                            (399, 261),     # Chin 9
                            (337, 297),     # Left eye left corner 37
                            (513, 301),     # Right eye right corne 46
                            (345, 265),     # Left Mouth corner 49
                            (453, 269)      # Right mouth corner 55
                        ], dtype="double")

# 3D model points.
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip 34
                            (0.0, -330.0, -65.0),        # Chin 9
                            (-225.0, 170.0, -135.0),     # Left eye left corner 37
                            (225.0, 170.0, -135.0),      # Right eye right corne 46
                            (-150.0, -150.0, -125.0),    # Left Mouth corner 49
                            (150.0, -150.0, -125.0)      # Right mouth corner 55

                        ])
# created a *threaded *video stream, allow the camera sensor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from `picamera` module...")
vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()

# loop over some frames...this time using the threaded stream
while fps._numFrames < args["num_frames"]:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# check to see if the frame should be displayed to our screen
	if args["display"] > 0:
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
while True:
    frame = vs.read()
    frame = imutils.resize(image, width=frame_width, height=frame_height)
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = frame.shape

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # check to see if a face was detected, and if so, draw the total
    # number of faces on the frame
    if len(rects) > 0:
            text = "{} face(s) found".format(len(rects))
            cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

    # loop over the face detections
    for rect in rects:
        # compute the bounding box of the face and draw it on the
        # frame
                (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw each of them
                for (i, (x, y)) in enumerate(shape):
                        if i == 33:
                                #something to our key landmarks
            # save to our new key point list
            # i.e. keypoints = [(i,(x,y))]
                                image_points[0] = np.array([x,y],dtype='double')
            # write on frame in Green
                                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                        elif i == 8:
            #something to our key landmarks
            # save to our new key point list
            # i.e. keypoints = [(i,(x,y))]
                                image_points[1] = np.array([x,y],dtype='double')
            # write on frame in Green
                                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                        elif i == 36:
                                #something to our key landmarks
                                # save to our new key point list
                                # i.e. keypoints = [(i,(x,y))]
                                image_points[2] = np.array([x,y],dtype='double')
                                # write on frame in Green
                                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                        elif i == 45:
                                #something to our key landmarks
            # save to our new key point list
            # i.e. keypoints = [(i,(x,y))]
                                image_points[3] = np.array([x,y],dtype='double')
            # write on frame in Green
                                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                        elif i == 48:
            #something to our key landmarks
            # save to our new key point list
            # i.e. keypoints = [(i,(x,y))]
                                image_points[4] = np.array([x,y],dtype='double')
            # write on frame in Green
                                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                        elif i == 54:
            #something to our key landmarks
            # save to our new key point list
            # i.e. keypoints = [(i,(x,y))]
                                image_points[5] = np.array([x,y],dtype='double')
            # write on frame in Green
                                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                                cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
                        else:
            #everything to all other landmarks
            # write on frame in Red
                                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                                cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                focal_length = size[1]
                center = (size[1]/2, size[0]/2)
                camera_matrix = np.array([[focal_length,0,center[0]],[0, focal_length, center[1]],[0,0,1]], dtype="double")

                #print "Camera Matrix :\n {0}".format(camera_matrix)

                dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)#flags=cv2.CV_ITERATIVE)

                #print "Rotation Vector:\n {0}".format(rotation_vector)
                #print "Translation Vector:\n {0}".format(translation_vector)
                # Project a 3D point (0, 0 , 1000.0) onto the image plane
                # We use this to draw a line sticking out of the nose_end_point2D
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]),rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                for p in image_points:
                        cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)

                p1 = ( int(image_points[0][0]), int(image_points[0][1]))
                p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                cv2.line(frame, p1, p2, (255,0,0), 2)

    # show the frame
    (h,w) = frame.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(frame, M, (w,h))

    cv2.imshow("rotated", rotated)
    key = cv2.waitKey(1) & 0xFF

    # clear the stream for the next frame
    rawCapture.truncate(0)
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
            break
print(image_points)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


