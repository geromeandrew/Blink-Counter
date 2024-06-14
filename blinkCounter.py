import cv2
import numpy as np
from faceMeshModule import faceMeshDetection

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    C = np.linalg.norm(eye[0] - eye[3])

    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture('Blinking Morse Code _ Hello.mp4') 
detector = faceMeshDetection()

leftEyeIndices = [33, 160, 158, 133, 153, 144]
rightEyeIndices = [362, 385, 387, 263, 373, 380]

EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3

blinkCount = 0
counter = 0
color = (255, 0, 255)

while True:
    success, img = cap.read()
    if not success:
        break  

    img, faces = detector.findFaceMesh(img, draw=False)
    
    if faces:
        face = faces[0]

        leftEye = np.array([face[i] for i in leftEyeIndices])
        rightEye = np.array([face[i] for i in rightEyeIndices])

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        if ear < EYE_AR_THRESH:
            counter += 1
        else:
            if counter >= EYE_AR_CONSEC_FRAMES:
                blinkCount += 1
            counter = 0

        for point in leftEye:
            cv2.circle(img, tuple(point), 2, color, cv2.FILLED)
        for point in rightEye:
            cv2.circle(img, tuple(point), 2, color, cv2.FILLED)

    cv2.putText(img, f'Blink Count: {blinkCount}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()

final_img = np.zeros((360, 640, 3), dtype=np.uint8)
cv2.putText(final_img, f'Final Blink Count: {blinkCount}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow('Final Count', final_img)
cv2.waitKey(0) 
cv2.destroyAllWindows()

print(f'Final Blink Count: {blinkCount}')
