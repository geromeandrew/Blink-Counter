import cv2
import numpy as np
from tkinter import Tk, Label, Button, filedialog, messagebox, Canvas
from PIL import Image, ImageDraw, ImageFont
from faceMeshModule import faceMeshDetection

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def select_video_file():
    file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4;*.avi;*.mov;*.mkv")])
    if not file_path:
        messagebox.showerror("Error", "No video file selected. Exiting...")
        root.destroy()
    else:
        run_blink_detection(file_path)

def run_blink_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = faceMeshDetection()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    frame_delay = int(1000 / fps)

    leftEyeIndices = [33, 160, 158, 133, 153, 144]
    rightEyeIndices = [362, 385, 387, 263, 373, 380]

    EYE_AR_THRESH = 0.25
    EYE_AR_CONSEC_FRAMES = 3

    blinkCount = 0
    counter = 0
    color = (255, 0, 255)
    end_of_video = False

    while True:
        success, img = cap.read()
        if not success:
            end_of_video = True
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

        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        font = ImageFont.truetype("arial.ttf", 32)
        draw.text((10, 30), f'Blink Count: {blinkCount}', font=font, fill=(0, 255, 0, 255))
        img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        
        cv2.imshow('Image', img)
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            end_of_video = False
            break

    cap.release()
    cv2.destroyAllWindows()

    final_img = Image.new("RGB", (640, 360), color="#34495e")
    draw = ImageDraw.Draw(final_img)
    font = ImageFont.truetype("arial.ttf", 48)
    text = f'Final Blink Count: {blinkCount}'
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    draw.rectangle(((320 - text_width // 2 - 10, 180 - text_height // 2 - 10), 
                    (320 + text_width // 2 + 10, 180 + text_height // 2 + 10)), fill="#1abc9c")
    draw.text((320 - text_width // 2, 180 - text_height // 2), text, font=font, fill="white")

    final_img = cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)
    cv2.imshow('Final Count', final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    messagebox.showinfo("Final Blink Count", f'Total blinks: {blinkCount}')

root = Tk()
root.title("BlinkDetect")
root.geometry("600x400")
root.configure(bg='#34495e')

canvas = Canvas(root, width=600, height=400, bg='#34495e', highlightthickness=0)
canvas.pack()

label = Label(root, text="BlinkDetect", font=("Helvetica", 28, "bold"), fg="white", bg="#34495e")
canvas.create_window(300, 150, window=label)

select_button = Button(root, text="Select Video File", command=select_video_file, font=("Helvetica", 16, "bold"), fg="white", bg="#1abc9c", activebackground="#16a085", activeforeground="white", relief="flat", padx=20, pady=10)
canvas.create_window(300, 250, window=select_button)

root.mainloop()
