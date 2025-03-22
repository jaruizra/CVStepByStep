# track pixels between 
# Shi-Tomasi corner detection algorithm

import cv2
import numpy as np

# get camera object
cap = cv2.VideoCapture(0)

# bucle
while True:
    # get one frame
    ret, frame1 = cap.read()

    # framerate = 1/30 = 3,3ms
    cv2.waitKey(3)

    # flip the image
    frame1 = cv2.flip(frame1, 1)
    frame1_cp = frame1.copy()


    # convert to gray
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # create yellow mask
    # make hsv image
    hsv_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the Yellow color in HSV
    # hue(color type), saturation(intensity of color), value(brightness)
    lowYellow = (20, 73, 150)
    upperYellow = (40, 255, 255)
    mask1 = cv2.inRange(hsv_frame1, lowYellow, upperYellow)

    # apply the mask to the gray and black image to extract red regions
    gray_frame1_masked = cv2.bitwise_and(gray_frame1, gray_frame1, mask=mask1)

    # cojer el error de que no detecte features 
    try:
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
        
        # OBTENER FEATURES de la imagen principal pero solo en las areas destacadas en mask
        corners1 = cv2.goodFeaturesToTrack(gray_frame1, maxCorners=0, qualityLevel=0.1, minDistance=10, mask=mask1)

        # OBTENER FEATURES directamente de la imagen gray_scale que es la imagen original con la mascara
        corners1_masked = cv2.goodFeaturesToTrack(gray_frame1_masked, maxCorners=0, qualityLevel=0.4, minDistance=10)

        # Convert corners to integer values
        corners1 = np.int32(corners1)
        corners1_masked = np.int32(corners1_masked)

        # Draw the detected corners on the original image
        for corner in corners1:
            #  Flatten the nested array [[x, y]] into a simple array [x, y]
            x, y = corner.ravel() 
            # Syntax: cv2.circle(image, center, radius, color, thickness)
            cv2.circle(frame1, (x, y), 5, (0, 0, 255), -1)

         # Draw the detected corners on the original image
        for corner in corners1_masked:
            x, y = corner.ravel()  # Flatten the corner array
            cv2.circle(frame1_cp, (x, y), 5, (0, 0, 255), -1)       

    except TypeError:
        print("No detections")

    # display image
    cv2.imshow("corners", frame1)
    cv2.imshow("mask", mask1)
    cv2.imshow("corners1_masked", frame1_cp)

    


