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
    ret, img = cap.read()
    if ret is not True:
        break
    height, width, _ = img.shape
    white_frame = np.zeros((height, width, 3), np.uint8)
    white_frame.fill(255)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Facial landmarks
    result = face_mesh.process(rgb_img)
    try:
        for facial_landmarks in result.multi_face_landmarks:
            for i in range(0, 468):
                pt1 = facial_landmarks.landmark[i]
                x = int(pt1.x * width)
                y = int(pt1.y * height)
                cv2.circle(white_frame, (x, y), 2, (100, 100, 0), -1)
    except Exception as e:
        pass
    cv2.imshow("Image", img)
    cv2.imshow("Black", white_frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
        
cv2.destroyAllWindows()        
cap.release()
