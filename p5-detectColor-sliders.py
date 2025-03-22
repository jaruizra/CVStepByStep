# Apply the Canny edge detection algorithm

import cv2

# capture camera object
cap = cv2.VideoCapture(0)

# create trackbars window
cv2.namedWindow("Trackbars")

# Hue range in OpenCV is [0..179], Saturation/Value range [0..255]
cv2.createTrackbar("H Min", "Trackbars", 0, 179, lambda x: None)
cv2.createTrackbar("H Max", "Trackbars", 179, 179, lambda x: None)
cv2.createTrackbar("S Min", "Trackbars", 0, 255, lambda x: None)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, lambda x: None)
cv2.createTrackbar("V Min", "Trackbars", 0, 255, lambda x: None)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, lambda x: None)

# bucle
while True:
    # ret = error y frame leer un frame de la camara
    ret, frame = cap.read()
    
    # comprobar error
    if not ret:
        break  # Exit if there’s an error 

    # flip the frame
    # 1 flips horizontally, 0 flips vertically, -1 flips both
    flipFrame = cv2.flip(frame, 1)

    # Syntax: cv2.cvtColor(src, code)
    # Conversion options include cv2.COLOR_BGR2GRAY (used here), cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2HSV, etc.
    #gray_frame = cv2.cvtColor(flipFrame, cv2.COLOR_BGR2GRAY)
    hsv_frame = cv2.cvtColor(flipFrame, cv2.COLOR_BGR2HSV)

    # 4. Read the current positions of all trackbars
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")
    
    # creo una mascara con los range minimo y maximo
    lower = (h_min, s_min, v_min)
    upper = (h_max, s_max, v_max)
    mask  = cv2.inRange(hsv_frame, lower, upper)

    # Apply the mask to the original frame to extract only the red regions
    # Syntax: result = cv2.bitwise_and(src1, src2, mask=mask)
    # 'mask': The binary mask to apply
    masked_result = cv2.bitwise_and(flipFrame, flipFrame, mask=mask)

    # Syntax: cv2.imshow(windowName, image)
    # 'windowName': The name of the window
    # 'image': The image to display
    cv2.imshow('Webcam', masked_result)

    # salir
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()