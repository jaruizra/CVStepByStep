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

    # Define the lower and upper bounds for the red color in HSV
    # Lower red range (hue: 0-10)
    # hue(color type), saturation(intensity of color), value(brightness)
    lower_red = (0, 120, 70)
    upper_red = (10, 255, 255)
    # Create a binary mask for the lower red range (hue between lower[0] and upper[0])
    # Syntax: mask = cv2.inRange(src, lowerb, upperb)
    mask1 = cv2.inRange(hsv_frame, lower_red, upper_red)

    # Upper red range (hue: 170-180)
    # hue(color type), saturation(intensity of color), value(brightness)
    lower_red = (170, 120, 70)
    upper_red = (180, 255, 255)
    # Create a binary mask for the upper red range (hue between lower[0] and upper[0])
    mask2 = cv2.inRange(hsv_frame, lower_red, upper_red)

    # Yellow
    lowYellow = (20, 100, 100)
    upperYellow = (35, 255, 255)
    mask3 = cv2.inRange(hsv_frame, lowYellow, upperYellow)

    # Combine the two masks using a bitwise OR operation
    # Syntax: result = cv2.bitwise_or(src1, src2)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.bitwise_or(mask, mask3)



    # Apply the mask to the original frame to extract only the red regions
    # Syntax: result = cv2.bitwise_and(src1, src2, mask=mask)
    # 'mask': The binary mask to apply
    mask_combined = cv2.bitwise_and(flipFrame, flipFrame, mask=mask)

    # Syntax: cv2.imshow(windowName, image)
    # 'windowName': The name of the window
    # 'image': The image to display
    cv2.imshow('Webcam', mask_combined)

    # salir
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()