'''
Sources: 
https://docs.opencv.org/4.x/df/d9d/tutorial_py_colorspaces.html
https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

Changes:
I used the three sources above to create a program that converts each frame of a video to HSV,
applies a mask to find parts of the frame that are blue, and finds the largest of the contours and
draws it on the frame.
'''

import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

# define range of blue color in HSV
lower_blue = np.array([110,50,50])
upper_blue = np.array([130,255,255])

while True:
    # Take each frame
    _, frame = cap.read()

    # Convert BGR to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Apply thresholding
    mask = cv.inRange(hsv, lower_blue, upper_blue)

    # Find contours in the binary image
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Loop through the detected contours and draw them
    if len(contours) == 0:
        continue

    max_area = 0
    max_area_cnt = contours[0]

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_area_cnt = cnt

    x, y, w, h = cv.boundingRect(max_area_cnt)
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Display the processed frame
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
