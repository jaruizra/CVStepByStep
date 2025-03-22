import cv2
import numpy as np


# detect corners in a mask
def detectCorners(hsv_frame, gray_image):
    # get values for yellow mask
    upperYellow = (40, 255, 255)
    lowYellow = (20, 60, 255)
    kernel = np.ones((4, 4), np.uint8)

    # create the msak in the frame
    mask = cv2.inRange(hsv_frame, lowYellow, upperYellow)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # apply mask to the gray image
    masked_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    cv2.imshow("mask", mask)
    #cv2.imshow("mask_gray", masked_gray)

    # Detect initial corners (Shi-Tomasi)
    corners = cv2.goodFeaturesToTrack(masked_gray, maxCorners=100, qualityLevel=0.6, minDistance=10) 

    return corners


def main():
    # get the camera
    cap = cv2.VideoCapture(0)

    # for refresh
    frame_number = 0

    # 1) Read the first frame and find corners
    ret, old_frame = cap.read()
    if not ret:
        print("Failed to grab initial frame.")
        exit()

    # get hsv and grey image
    old_frame = cv2.flip(old_frame, 1)
    old_gray  = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    old_hsv  = cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)

    # Detect initial corners (Shi-Tomasi)
    old_corners = detectCorners(old_hsv, old_gray)
        
    while True:
        # 2) Read next frame
        ret, frame = cap.read()
        # check error
        if not ret:
            break
        
        # get hsv and gray image
        frame = cv2.flip(frame, 1)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        # 2. Redetect if needed
        # 
        if old_corners is None or len(old_corners) <= 2 or frame_number % 30 == 0:
            # Re-detect corners on the new frame
            old_corners = detectCorners(frame_hsv, frame_gray)
            
            # Also update old_gray to keep in sync
            old_gray = frame_gray.copy()
            # Reset frame_counter or let it keep counting
            # frame_counter = 0

        # 3) Optical flow calculation
        if old_corners is not None:

            # too little ones
            if len(old_corners) >= 2:                

                # get predicted values
                new_corners, status, err = cv2.calcOpticalFlowPyrLK(
                    old_gray, frame_gray,
                    old_corners, None
                )
            
                # check if I have detected corners in the new frame
                if new_corners is not None and status is not None:
                    # get only valid points
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
                    
                    # Update old_corners with the newly tracked positions
                    old_corners = good_new.reshape(-1, 1, 2)
        
        # 5) display the image 
        cv2.imshow("Optical Flow - Hybrid Approach", frame)
        


        # 5) Prepare for next iteration
        old_gray = frame_gray.copy()
        frame_number += 1

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
