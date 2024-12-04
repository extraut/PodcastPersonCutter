# 🎭 PodcastPersonCutter (YOLOFaceCutter) 🌟

PodcastPersonCutter is a powerful tool that enables easy video editing, focusing on specific faces.

## How to Use / Установка

1. **Dependencies Installation & Run**:
   - Python 3.x
   - Downloads last releases:
     ```
     https://github.com/extraut/PodcastPersonCutter/releases
     ```
   - Download CMake :
     ```
     https://cmake.org/download/
     ```
   - Next run install.bat file.
   - After install start run.bat file.
   - Enjoy!🌟

## How It Works

- PodcastPersonCutter uses the face_recognition face detection model and face_recognition to identify faces in a video.
- The program pre-processes frames, applies the face detection model and tracks them throughout the video.
- Detected faces are saved and can be used for further editing or processing.
- The desired face from the video will be cut into the final video with audio. Ideal for creating voice models later on).

## Key Features

- 📹 Face Detection and Tracking: Automatically detects and tracks faces in videos.
- 🎨 Zoom and Save: Zooms in on detected faces and saves them as individual images for further processing.
- 📝 Logging: Logs information about detected faces and the processing pipeline.
