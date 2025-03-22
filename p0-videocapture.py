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
    cv2.imshow('Webcam Feed', frame)  

    # salir
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()