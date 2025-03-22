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
    gray_frame = cv2.cvtColor(flipFrame, cv2.COLOR_BGR2GRAY)

    # Syntax: edges = cv2.Canny(image, threshold1, threshold2)
    # 'image': The input image (should be a single-channel grayscale image)
    # 'threshold1' (th1): The lower threshold for edge detection. 
    # Pixels with a gradient magnitude below this value are ignored (not considered edges).
    # 'threshold2' (th2): The upper threshold for edge detection. 
    # Pixels with a gradient magnitude above this value are considered strong edges.
    edges = cv2.Canny(gray_frame, 90, 150)

    # sacar la imagen girada
    cv2.imshow('Flipped Webcam', edges)

    # salir
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()