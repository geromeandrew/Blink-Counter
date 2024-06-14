
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

### Clone the Project

To get a copy of the project, clone it using Git:

1. Open a terminal or command prompt.
2. Run the following command to clone the project:

   ```sh
   git clone <repository-url>
   ```

3. Navigate to the project directory:

   ```sh
   cd <repository-directory>
   ```

This is the repository-url: https://github.com/geromeandrew/Blink-Counter.git

### Install Project Requirements

Ensure all the necessary packages are installed by using the `requirements.txt` file.

1. Open a terminal or command prompt.
2. Navigate to the project directory (if not already done).
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
5. When the video ends, press the `q` key to stop the program because it loops. The final blink count will be displayed, and the program will pause. Press any key to close the final count display.
