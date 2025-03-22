# track how pixels move between two frames
# Shi-Tomasi corner detection algorithm
# Lucas-Kanade method for optical flow

import cv2
import numpy as np

def edge_detection():

    # detect prominent corners or features using the Shi-Tomasi corner detection algorithm
    # Syntax: corners = cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, mask[, blockSize[, useHarrisDetector[, k]]]])
    # 'image': The input grayscale image
    # 'maxCorners': The maximum number of corners to return. Set to 0 to return all detected corners.
    # 'qualityLevel': A value between 0 and 1 representing the minimum accepted quality of corners.
    #                 Higher values mean only stronger corners are detected (e.g., 0.01 for 1% of the strongest corner).
    # 'minDistance': The minimum Euclidean distance (in pixels) between detected corners to avoid clustering.
    # 'mask' (optional): A mask image specifying the region of interest (only white regions are considered).
    # Returns: A NumPy array of shape (N, 1, 2), where N is the number of detected corners, and each corner is represented as (x, y).
    # returns nested array, array of arrays = [[x1, y1], [x2, y2], ...]

    # do the same as color detection with edges
    # obtener features de los edges
    corners1 = cv2.goodFeaturesToTrack(frame1_masked, maxCorners=0, qualityLevel=0.1, minDistance=10)
    corners2 = cv2.goodFeaturesToTrack(frame2_masked, maxCorners=0, qualityLevel=0.1, minDistance=10)

    # convert to integers
    corners1_int = np.int32(corners1)
    corners2_int = np.int32(corners2)

    # draw second 2 image circles in the image
    for corner in corners2_int:
        #  Flatten the nested array [[x, y]] into a simple array [x, y]
        x, y = corner.ravel() 
        # Syntax: cv2.circle(image, center, radius, color, thickness)
        cv2.circle(frame2, (x, y), 5, (0, 255, 0), -1)

    # optical flow
    nextPts, status, err = cv2.calcOpticalFlowPyrLK(frame1_gray, frame2_gray, corners1, None)

    nextPts = np.int32(nextPts)
    #corners1_int = np.int32(corners1)

    # 3. Filter only tracked points (status == 1)
    good_new = nextPts[status == 1]
    good_old = corners1_int[status == 1]

    # 4. Draw them
    for newPt, oldPt in zip(good_new, good_old):
        x2, y2 = newPt.ravel()
        x1, y1 = oldPt.ravel()

        # Circle at new location
        cv2.circle(frame1, (x2, y2), 5, (0, 0, 255), -1)
        cv2.circle(frame2, (x2, y2), 5, (0, 0, 255), -1)

        # Circle at old location
        cv2.circle(frame1, (x1, y1), 5, (255, 0, 0), -1)
        
        # Line from old to new (motion vector)
        cv2.line(frame1, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #cv2.imshow("Optical Flow", frame2)

    # draw second 2 image circles in the image
    for corner in nextPts:
        #  Flatten the nested array [[x, y]] into a simple array [x, y]
        x, y = corner.ravel() 
        # Syntax: cv2.circle(image, center, radius, color, thickness)
        cv2.circle(frame2, (x, y), 5, (255, 0, 0), -1)

    # mostrar frame2 detecciones
    cv2.imshow("Optical Flow vs Detected", frame2)
    cv2.imshow("Optical Flow", frame1)



# get camera
cap = cv2.VideoCapture(0)

# get 2 frames
ret1, frame1 = cap.read()
cv2.waitKey(10)
ret2, frame2 = cap.read()

# flip images
frame1 = cv2.flip(frame1, 1)
frame2 = cv2.flip(frame2, 1)

# creo un mask de amarillo
# convert to gray
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

# converto to hsv
frame1_hsv = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
frame2_hsv = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

# apply mask to the image
lowYellow = (20, 73, 150)
upperYellow = (40, 255, 255)
frame1_mask = cv2.inRange(frame1_hsv, lowYellow, upperYellow)
frame2_mask = cv2.inRange(frame2_hsv, lowYellow, upperYellow)

# get mask from frame1
frame1_masked = cv2.bitwise_and(frame1_gray, frame1_gray, mask=frame1_mask)
# get mask from frame2
frame2_masked = cv2.bitwise_and(frame2_gray, frame2_gray, mask=frame2_mask)

try:
    
    edge_detection()

    
    


except TypeError:
    print("no features found")

# Wait indefinitely until a key is pressed
cv2.waitKey(0)
