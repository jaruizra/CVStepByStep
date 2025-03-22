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

    # Show the frame in a window
    #cv2.imshow('Webcam Feed', frame)  

    # flip the frame
    # 1 flips horizontally, 0 flips vertically, -1 flips both
    flipFrame = cv2.flip(frame, 1)

    # Convert the flipped frame from BGR to Grayscale using cv2.cvtColor()
    # Syntax: cv2.cvtColor(src, code)
    # Conversion options include cv2.COLOR_BGR2GRAY (used here), cv2.COLOR_BGR2RGB, cv2.COLOR_BGR2HSV, etc.
    gray_frame = cv2.cvtColor(flipFrame, cv2.COLOR_BGR2GRAY)

    # sacar la imagen girada
    cv2.imshow('Flipped Webcam', gray_frame)

    # salir
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()