# Get two frames and compare differences

import cv2

# capture camera object
cap = cv2.VideoCapture(0)

# bucle
while True:
    # ret = error y frame leer un frame de la camara
    ret1, frame1 = cap.read()
    # sleep for 500ms
    cv2.waitKey(100)
    ret2, frame2 = cap.read()
    
    # comprobar error
    if not ret1 or not ret2:
        break  # Exit if there’s an error 

    # Syntax: cv2.cvtColor(src, code)
    # Conversion options include cv2.COLOR_BGR2GRAY (used here), cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2HSV, etc.
    #gray_frame = cv2.cvtColor(flipFrame, cv2.COLOR_BGR2GRAY)
    gray_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # find differences between two images
    diff = cv2.absdiff(gray_frame1, gray_frame2)

    # convert difference image into a binary image
    _, diff_bin_image = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)

    # Syntax: cv2.imshow(windowName, image)
    # 'windowName': The name of the window
    # 'image': The image to display
    cv2.imshow('OG_Webcam', diff)

    # original image with boxes
    cv2.imshow('OG_Boxes', diff_bin_image)

    # salir
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()