import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture()
cap.open(0, cv2.CAP_DSHOW)



# Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()



while True:
    # Image
    ret, image = cap.read()
    if ret is not True:
        break
    height, width, _ = image.shape
    black_frame = np.zeros((height, width, 3), np.uint8)
    black_frame.fill(255)
    #print("Height, width", height, width)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Facial landmarks
    result = face_mesh.process(rgb_image)

    for facial_landmarks in result.multi_face_landmarks:
        for i in range(0, 468):
            pt1 = facial_landmarks.landmark[i]
            x = int(pt1.x * width)
            y = int(pt1.y * height)

            cv2.circle(black_frame, (x, y), 2, (100, 100, 0), -1)
            #cv2.putText(image, str(i), (x, y), 0, 1, (0, 0, 0))

    cv2.imshow("Image", image)
    cv2.imshow("Black", black_frame)
    cv2.waitKey(1)
