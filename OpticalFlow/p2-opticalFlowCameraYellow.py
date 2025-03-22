import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# 1) Read the first frame and find corners
ret, old_frame = cap.read()
if not ret:
    print("Failed to grab initial frame.")
    exit()

old_frame = cv2.flip(old_frame, 1)

# get hsv and grey image
old_gray  = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_hsv  = cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)

# get the yello mask
upperYellow = (40, 255, 255)
lowYellow = (20, 73, 150)
old_mask = cv2.inRange(old_hsv, lowYellow, upperYellow)

# apply the mask to grey
old_masked_grey = cv2.bitwise_and(old_gray, old_gray, mask=old_mask)

# Detect initial corners (Shi-Tomasi)
old_corners = cv2.goodFeaturesToTrack(old_masked_grey, maxCorners=0, qualityLevel=0.5, minDistance=7)
    
while True:
    # 2) Read next frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    # get hsv and gray image
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for yellow
    new_frame_mask = cv2.inRange(frame_hsv, lowYellow, upperYellow)

    frame_masked = cv2.bitwise_and(frame_gray, frame_gray, mask=new_frame_mask)

    # 3) Optical flow calculation
    # But corners come from old_masked_gray
    if old_corners is not None:

        # get predicted values
        new_corners, status, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray,
            old_corners, None
        )
    
        # Filter out valid points
        if new_corners is not None and status is not None:
            good_new = new_corners[status == 1]
            good_old = old_corners[status == 1]
            
            # 4) Draw motion vectors
            for (new, old) in zip(good_new, good_old):
                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()
                
                # Draw line from old to new
                cv2.line(frame, (int(x_old), int(y_old)), (int(x_new), int(y_new)), (0, 255, 0), 2)
                
                # Draw circle at new position
                cv2.circle(frame, (int(x_new), int(y_new)), 5, (0, 0, 255), -1)
            
            # Update old points
            # Reshape the array of points from shape (N, 2) to (N, 1, 2)
            # - Each point has 2 values: (x, y)
            # - The extra '1' dimension is required by OpenCV for certain functions (like optical flow)
            # - '-1' lets NumPy automatically figure out how many points there are (based on the total size)
            old_corners = good_new.reshape(-1, 1, 2)
    
    cv2.imshow("Optical Flow - Yellow only", frame)
    
    # 5) Prepare for next iteration
    old_gray = frame_gray.copy()
    
    # OPTIONAL: Re-detect corners if you want to keep them fresh every N frames
    # old_corners = cv2.goodFeaturesToTrack(...)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
