
# Blink Detection Project

This project is designed to detect and count the number of times a person blinks in a given video file using OpenCV and MediaPipe.

## Prerequisites

### Download Visual Studio Code

Visual Studio Code (VS Code) is a powerful and free code editor. Download and install it from the link below:

[Download Visual Studio Code](https://code.visualstudio.com/Download)

### Download and Setup Git

Git is a version control system that helps manage project versions and collaboration. Download and install Git from the link below:

[Download Git](https://git-scm.com/downloads)

After installing Git, set it up by following these instructions:

1. Open a terminal or command prompt.
2. Configure your Git username:

   ```sh
   git config --global user.name "Your Name"
   ```

3. Configure your Git email:

   ```sh
   git config --global user.email "your.email@example.com"
   ```

### Download Python

Python is the programming language used for this project. Download and install Python from the link below:

[Download Python](https://www.python.org/downloads/)

### Install Project Requirements

Ensure all the necessary packages are installed by using the `requirements.txt` file.

1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the following command to install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

## Configuration

### Replace the Path for the Video

To use your own video file for blink detection, follow these steps:

1. Place your video file in the project directory.
2. Open `blinkCounter.py` in your code editor.
3. Locate the line that specifies the video file path:

   ```python
   cap = cv2.VideoCapture('Video.mp4')
   ```

4. Replace `'Video.mp4'` with the name of your video file, ensuring it is in quotes. For example:

   ```python
   cap = cv2.VideoCapture('your_video.mp4')
   ```

## Running the Code

To run the blink detection program, follow these detailed steps:

1. Open a terminal or command prompt.
2. Navigate to the project directory.
3. Run the `blinkCounter.py` file using Python:

   ```sh
   python blinkCounter.py
   ```

4. The program will process the video and display the blink count on the screen.
5. When the video ends, the final blink count will be displayed, and the program will pause. Press any key to close the final count display.

## Code

### `blinkCounter.py`
```python
import cv2
import numpy as np
from faceMeshModule import faceMeshDetection

def eye_aspect_ratio(eye):
    # Compute the distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # Compute the distance between the horizontal eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

cap = cv2.VideoCapture('Video.mp4')  # Use the video file as the video source
detector = faceMeshDetection()

# Indices for the left and right eyes in the face mesh
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
        break  # Break the loop if the video has ended

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

    # Display the blink count on the screen
    cv2.putText(img, f'Blink Count: {blinkCount}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()

# Display the final blink count
final_img = np.zeros((360, 640, 3), dtype=np.uint8)
cv2.putText(final_img, f'Final Blink Count: {blinkCount}', (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow('Final Count', final_img)
cv2.waitKey(0)  # Wait indefinitely until a key is pressed

cv2.destroyAllWindows()

# Log the final number of blinks
print(f'Final Blink Count: {blinkCount}')
```

### `faceMeshModule.py`
```python
import cv2
import mediapipe as mp
import numpy as np

class faceMeshDetection:
    def __init__(self, staticMode=False, maxFaces=1, minDetectionCon=0.5, minTrackCon=0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,
                                                 max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceMesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks:
            for faceLms in results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               self.drawSpec, self.drawSpec)
                face = np.array([[int(lm.x * img.shape[1]), int(lm.y * img.shape[0])] for lm in faceLms.landmark])
                faces.append(face)
        return img, faces

    @staticmethod
    def findDistance(p1, p2, img=None):
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = np.hypot(x2 - x1, y2 - y1)
        info = (x1, y1, x2, y2, cx, cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length, info, img
        else:
            return length, info

def main():
    cap = cv2.VideoCapture(0)
    detector = faceMeshDetection(maxFaces=2)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if faces:
            print(faces[0])
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
```
