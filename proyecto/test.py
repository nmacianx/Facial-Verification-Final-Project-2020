import cv2 as cv

cap = cv.VideoCapture(0)

print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
print(cap.get(cv.CAP_PROP_FRAME_WIDTH))

while True:        

    hasFrame, frame = cap.read()
    frame = frame[32:448, 112:528]
    print(frame.shape)
    # print()

    key = cv.waitKey(1)
    cv.imshow('Sistema de deteccion facial', frame)