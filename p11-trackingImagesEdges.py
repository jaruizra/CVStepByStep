# track how pixels move between two frames
# Shi-Tomasi corner detection algorithm

import cv2
import numpy as np

def color_detection():
    # OBTENER FEATURES directamente de la imagen gray_scale que es la imagen original con la mascara
    corners1_masked = cv2.goodFeaturesToTrack(frame1_masked, maxCorners=0, qualityLevel=0.1, minDistance=10)
    corners2_masked = cv2.goodFeaturesToTrack(frame2_masked, maxCorners=0, qualityLevel=0.1, minDistance=10)
    
    # convert to integers
    corners1_masked_int = np.int32(corners1_masked)
    corners2_masked_int = np.int32(corners2_masked)

    # draw first 1  image circles in the image
    for corner in corners1_masked_int:
        #  Flatten the nested array [[x, y]] into a simple array [x, y]
        x, y = corner.ravel() 
        # Syntax: cv2.circle(image, center, radius, color, thickness)
        cv2.circle(frame1, (x, y), 5, (0, 0, 255), -1)

    # mostrar frame1 detecciones
    #cv2.imshow("Frame 1", frame1)

    # draw second 2 image circles in the image
    for corner in corners2_masked_int:
        #  Flatten the nested array [[x, y]] into a simple array [x, y]
        x, y = corner.ravel() 
        # Syntax: cv2.circle(image, center, radius, color, thickness)
        cv2.circle(frame2, (x, y), 5, (0, 255, 0), -1)

    # mostrar frame2 detecciones
    #cv2.imshow("Frame 2", frame2)

    # mostrar detecciones sobre la imagen 1
    for corner in corners2_masked_int:
        #  Flatten the nested array [[x, y]] into a simple array [x, y]
        x, y = corner.ravel() 
        # Syntax: cv2.circle(image, center, radius, color, thickness)
        cv2.circle(frame1, (x, y), 5, (0, 255, 0), -1)
    
    # mostrar frame combinado
    cv2.imshow("Frame combinado", frame1)



# get camera
cap = cv2.VideoCapture(0)

# get 2 frames
ret1, frame1 = cap.read()
cv2.waitKey(10)
ret2, frame2 = cap.read()

# flip images
frame1 = cv2.flip(frame1, 1)
frame2 = cv2.flip(frame2, 1)
frame1_cp = frame1.copy()
frame2_cp = frame2.copy()

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

# get features from frame1
frame1_masked = cv2.bitwise_and(frame1_gray, frame1_gray, mask=frame1_mask)
# get features from frame2
frame2_masked = cv2.bitwise_and(frame2_gray, frame2_gray, mask=frame2_mask)

# get edges of frame 1
frame1_mask_color = cv2.bitwise_and(frame1_cp, frame1_cp, mask=frame1_mask)
frame1_mask_gray = cv2.cvtColor(frame1_mask_color, cv2.COLOR_BGR2GRAY)
edges1 = cv2.Canny(frame1_mask_gray, 90, 200)

# get edges of frame 2
frame2_mask_color = cv2.bitwise_and(frame2_cp, frame2_cp, mask=frame2_mask)
frame2_mask_gray = cv2.cvtColor(frame2_mask_color, cv2.COLOR_BGR2GRAY)
edges2 = cv2.Canny(frame2_mask_gray, 90, 200)

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
    
    #color_detection()
    
    # do the same as color detection with edges
    # obtener features de los edges
    corners1_edges = cv2.goodFeaturesToTrack(edges1, maxCorners=0, qualityLevel=0.1, minDistance=10)
    corners2_edges = cv2.goodFeaturesToTrack(edges2, maxCorners=0, qualityLevel=0.1, minDistance=10)

    # convert to integers
    corners1_edges_int = np.int32(corners1_edges)
    corners2_edges_int = np.int32(corners2_edges)

    # draw first 1  image circles in the image
    for corner in corners1_edges_int:
        #  Flatten the nested array [[x, y]] into a simple array [x, y]
        x, y = corner.ravel() 
        # Syntax: cv2.circle(image, center, radius, color, thickness)
        cv2.circle(frame1_cp, (x, y), 5, (0, 0, 255), -1)

    # mostrar frame1 detecciones
    cv2.imshow("Frame 1", frame1_cp)

    # draw second 2 image circles in the image
    for corner in corners2_edges_int:
        #  Flatten the nested array [[x, y]] into a simple array [x, y]
        x, y = corner.ravel() 
        # Syntax: cv2.circle(image, center, radius, color, thickness)
        cv2.circle(frame2_cp, (x, y), 5, (0, 255, 0), -1)

    # mostrar frame2 detecciones
    cv2.imshow("Frame 2", frame2_cp)

    # mostrar detecciones sobre la imagen 1
    for corner in corners2_edges_int:
        #  Flatten the nested array [[x, y]] into a simple array [x, y]
        x, y = corner.ravel() 
        # Syntax: cv2.circle(image, center, radius, color, thickness)
        cv2.circle(frame1_cp, (x, y), 5, (0, 255, 0), -1)
    
    # mostrar frame combinado
    cv2.imshow("Frame combinado edges", frame1_cp)

except TypeError:
    print("no features found")

# Wait indefinitely until a key is pressed
cv2.waitKey(0)
