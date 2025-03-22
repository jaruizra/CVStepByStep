# Color masking and edge detection

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


    # Apply the mask to the original frame to extract only the red regions
    # Syntax: result = cv2.bitwise_and(src1, src2, mask=mask)
    # 'mask': The binary mask to apply
    mask_color = cv2.bitwise_and(flipFrame, flipFrame, mask=mask)

    # convert og_frame to grayscale
    gray_frame = cv2.cvtColor(mask_color, cv2.COLOR_BGR2GRAY)
    # apply the mask to the gray and black image to extract red regions
    mask_gray = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

    # Aply edges to the gray image frame (mask_gray)
    # Syntax: edges = cv2.Canny(image, threshold1, threshold2)
    # 'image': The input image (should be a single-channel grayscale image)
    # 'threshold1' (th1): The lower threshold for edge detection. 
    # Pixels with a gradient magnitude below this value are ignored (not considered edges).
    # 'threshold2' (th2): The upper threshold for edge detection. 
    # Pixels with a gradient magnitude above this value are considered strong edges.
    edges = cv2.Canny(gray_frame, 90, 200)

    # Syntax: cv2.imshow(windowName, image)
    # 'windowName': The name of the window
    # 'image': The image to display
    cv2.imshow('Mask_colo', mask_color)
    cv2.imshow('Mask_gray', mask_gray)
    cv2.imshow('Edge_Detection', edges)

    # salir
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()