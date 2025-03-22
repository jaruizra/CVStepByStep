# Apply the Canny edge detection algorithm

import cv2

# capture camera object
cap = cv2.VideoCapture(0)

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

    # Define the lower and upper bounds for the Yellow color in HSV
    # hue(color type), saturation(intensity of color), value(brightness)
    lowYellow = (20, 73, 150)
    upperYellow = (40, 255, 255)
    mask = cv2.inRange(hsv_frame, lowYellow, upperYellow)

    # Find contours in the binary mask
    # Syntax: contours, hierarchy = cv2.findContours(image, mode, method)
    # 'image': The input binary image (non-zero pixels are treated as part of objects).
    # 'mode': Contour retrieval mode. Options:
    #         - cv2.RETR_EXTERNAL: Retrieves only the outermost contours.
    #         - cv2.RETR_LIST: Retrieves all contours without hierarchy.
    #         - cv2.RETR_TREE: Retrieves all contours and reconstructs the hierarchy.
    # 'method': Contour approximation method. Options:
    #         - cv2.CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, and diagonal segments, keeping only endpoints.
    #         - cv2.CHAIN_APPROX_NONE: Stores all points along the contour.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Apply the mask to the original frame to extract only the red regions
    # Syntax: result = cv2.bitwise_and(src1, src2, mask=mask)
    # 'mask': The binary mask to apply
    mask_combined = cv2.bitwise_and(flipFrame, flipFrame, mask=mask)

    # Color boxes
    for contour in contours:
        # Filter small areas
        if cv2.contourArea(contour) > 1000:
            # get rectangle of the contour for drawing  
            x, y, w, h = cv2.boundingRect(contour)
            # draw the boxes
            cv2.rectangle(flipFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box, 2px thick
            cv2.rectangle(mask_combined, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box, 2px thick

    # Syntax: cv2.imshow(windowName, image)
    # 'windowName': The name of the window
    # 'image': The image to display
    cv2.imshow('OG_Webcam', mask_combined)

    # original image with boxes
    cv2.imshow('OG_Boxes', flipFrame)

    # salir
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()