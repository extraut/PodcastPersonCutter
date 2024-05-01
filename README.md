# 🎭 PodcastPersonCutter (YOLOFaceCutter) 🌟

PodcastPersonCutter is a powerful tool that enables easy video editing, focusing on specific faces.

## How to Use / Установка

1. **Dependencies Installation**:
   - Python 3.x
   - Clone the repository and navigate to the directory:
     ```
     git clone https://github.com/SpaceVX/PodcastPersonCutter.git
     cd PodcastPersonCutter
     ```
   - Goto link -
     ```
     https://drive.google.com/file/d/1nPhmy4N0PWAcdHgCUCKc6wPlNkzI32Cj/view?usp=drive_link
     ```
     (FFMPEG)
     ```
     avcodec-60.dll
     avdevice-60.dll
     avfilter-9.dll
     avformat-60.dll
     avutil-58.dll
     ffmpeg.exe
     ffplay.exe
     ffprobe.exe
     postproc-57.dll
     swresample-4.dll
     swscale-7.dll
     ```
   - Next goto install folder - run .bat file.
   - Enjoy!🌟

2. **Use**:
   - Run the program:
     ```
     python main.py
     ```

## How It Works

- PodcastPersonCutter uses the face_recognition face detection model and face_recognition to identify faces in a video.
- The program pre-processes frames, applies the face detection model and tracks them throughout the video.
- Detected faces are saved and can be used for further editing or processing.
- The desired face from the video will be cut into the final video with audio. Ideal for creating voice models later on).

## Key Features

- 📹 Face Detection and Tracking: Automatically detects and tracks faces in videos.
- 🎨 Zoom and Save: Zooms in on detected faces and saves them as individual images for further processing.
- 📝 Logging: Logs information about detected faces and the processing pipeline.
