from moviepy.editor import concatenate_videoclips, ImageSequenceClip, VideoFileClip

from collections import defaultdict
from statistics import mean
import subprocess
import logging
import io
import os

from rembg import remove
import face_recognition
from PIL import Image
import gradio as gr
import numpy as np
import cv2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


logging.getLogger('asyncio').setLevel(logging.CRITICAL)
logging.getLogger('httpx').setLevel(logging.CRITICAL)

is_processing_faces = True

def preprocess_frame(frame, target_size=(640, 640)):
    frame_resized = cv2.resize(frame, target_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_tensor = torch.from_numpy(frame_rgb).to(device)
    frame_tensor = frame_tensor.half() 
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
    return frame_tensor

def compute_color_histogram(image, bins=8):
    """Compute color histogram of an image."""
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def is_similar(image1, image2, duplicate_rate_threshold):
    image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

    hist1 = compute_color_histogram(image1_rgb)
    hist2 = compute_color_histogram(image2_rgb)

    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return correlation > duplicate_rate_threshold

def remove_face_background(face_frame_bgr: np.array):
    face_frame_rgb = cv2.cvtColor(face_frame_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_frame_rgb)
    
    output_image = remove(face_pil)
    
    processed_face_frame_rgb = output_image.convert("RGB")
    processed_face_frame_bgr = cv2.cvtColor(np.array(processed_face_frame_rgb), cv2.COLOR_RGB2BGR)
    
    return processed_face_frame_bgr

def process_frame(frame_count, frame, padding, existing_faces, duplicate_rate_threshold, faces_directory, use_rem_bg):
    faces_detected = 0
    face_images_frame = []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(frame_rgb)
    
    for face_location in face_locations:
        top, right, bottom, left = face_location

        zoom_factor = 3
        face_height, face_width = bottom - top, right - left
        face_center_x = left + (face_width // 2)
        face_center_y = top + (face_height // 2)

        zoomed_width = int(face_width * zoom_factor)
        zoomed_height = int(face_height * zoom_factor)

        left_x = max(0, face_center_x - zoomed_width // 2)
        right_x = min(frame.shape[1] - 1, face_center_x + zoomed_width // 2)
        top_y = max(0, face_center_y - zoomed_height // 2)
        bottom_y = min(frame.shape[0] - 1, face_center_y + zoomed_height // 2)

        zoomed_face_frame = frame[top_y:bottom_y, left_x:right_x]
        zoomed_face_frame_rgb = cv2.cvtColor(zoomed_face_frame, cv2.COLOR_BGR2RGB)

        is_new_face = True
        for existing_face in existing_faces:
            if is_similar(zoomed_face_frame_rgb, existing_face, duplicate_rate_threshold):
                is_new_face = False
                break

        if is_new_face:
            existing_faces.append(zoomed_face_frame_rgb)

            if use_rem_bg:
                zoomed_face_frame_no_bg = remove_face_background(zoomed_face_frame)
                zoomed_face_frame_no_bg_rgb = cv2.cvtColor(zoomed_face_frame_no_bg, cv2.COLOR_BGR2RGB)
            else:
                zoomed_face_frame_no_bg_rgb = zoomed_face_frame_rgb

            faces_detected += 1
            face_image = Image.fromarray(zoomed_face_frame_no_bg_rgb)
            face_image_path = os.path.join(faces_directory, f"face_{frame_count}_{faces_detected}.jpg")
            face_image.save(face_image_path)
            face_images_frame.append(face_image)
            
    logger.info(f"Number of faces detected on frame #{frame_count}: {faces_detected}")
    return existing_faces, face_images_frame



def track_faces_in_frames(uploaded_video_path, selected_faces, score_detect_threshold, fps_value, use_rem_bg):
    selected_face_images = [face_recognition.load_image_file(face) for face in selected_faces]
    selected_face_encodings = [face_recognition.face_encodings(face_image)[0] for face_image in selected_face_images if face_recognition.face_encodings(face_image)]
    
    if not selected_face_encodings:
        logger.info("Лицо не найдено, выберете другое лицо с лучшим ракурсом.")
        return None, "Лицо не найдено, выберете другое лицо с лучшим ракурсом."

    logger.info(f"Number of selected person: {len(selected_face_encodings)}")

    video_clip = VideoFileClip(uploaded_video_path)
    fps = fps_value or video_clip.fps
    frame_time = 1.0 / fps
    
    video_segments = []

    frame_num = 0
    logger.info("Processing video frames...")
    for frame in video_clip.iter_frames(fps=fps, dtype="uint8"):
        frame_locations = face_recognition.face_locations(frame)
        frame_encodings = face_recognition.face_encodings(frame, frame_locations)

        match = False
        for frame_encoding in frame_encodings:
            face_distances = face_recognition.face_distance(selected_face_encodings, frame_encoding)
            if len(face_distances) > 0 and min(face_distances) < score_detect_threshold:
                match = True
                logger.info(f"Person detected in frame {frame_num} with distance: {min(face_distances)}")
                break
        
        if match:
            start_time = frame_time * frame_num
            end_time = start_time + frame_time
            video_segments.append(video_clip.subclip(start_time, end_time))
        
        frame_num += 1

    if video_segments:
        logger.info("Concatenating video segments...")
        final_video_clip = concatenate_videoclips(video_segments, method="compose")
        video_output_path = "output_with_faces.mp4"
        logger.info(f"Writing final video to {video_output_path}...")
        final_video_clip.write_videofile(video_output_path, fps=fps, codec="libx264", audio_codec="aac")
        logger.info("Video created successfully.")
        return video_output_path, "Видео создано"
    else:
        logger.info("No person detected in the video.")
        return None, "Персоны не обнаружены"

def extract_frames_ffmpeg(uploaded_video_path, output_dir):

    ffmpeg_command = [
        'ffmpeg',
        '-hwaccel', 'auto',  
        '-i', uploaded_video_path,  
        os.path.join(output_dir, 'frame_%04d.jpg')  
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
        logger.info("Кадры успешно извлечены.")
    except subprocess.CalledProcessError as e:
        logger.error("Ошибка при извлечении кадров: {}".format(e))

def stop_processing():
    global is_processing_faces
    is_processing_faces = False  

  
def process_video(uploaded_video_path, score_face_threshold, duplicate_rate_threshold, score_detect_threshold, use_rem_bg):
    logger.info("Начало обработки видео.")

    frames_directory = "temp/frames"
    os.makedirs(frames_directory, exist_ok=True)
    
    faces_directory = "temp/faces"
    os.makedirs(faces_directory, exist_ok=True)

    extract_frames_ffmpeg(uploaded_video_path, frames_directory)


    logger.info("Начат процесс поиска персон")
    cap = cv2.VideoCapture(uploaded_video_path.name)
    padding = int(max(cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2)


    frame_count = 0
    existing_faces = []
    face_confidences = []
    face_images = []

    while cap.isOpened() and is_processing_faces:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        existing_faces, face_images_frame = process_frame(frame_count, frame, padding, existing_faces, duplicate_rate_threshold, faces_directory, use_rem_bg)
        face_images.extend(face_images_frame)
        avg_face_score = mean(face_confidences) if face_confidences else 0
        avg_face_score_text = f"AVG Face Score Detect: {avg_face_score:.2f}"
        yield face_images, avg_face_score_text
        
    cap.release()

def create_video(frames, uploaded_video_path, output_path="output.mp4", fps=30):  
    logger.info("Начало создания нового видео")
    frame_size = frames[0].shape[:2]
    frames = [frame for frame in frames if frame.shape[:2] == frame_size]
    
    if frames:
        temp_files = []
        for i, frame in enumerate(frames):
            temp_file_path = f"temp/temp_frame_{i}.jpeg"
            Image.fromarray(frame).save(temp_file_path)
            temp_files.append(temp_file_path)
        
        clips = ImageSequenceClip(temp_files, fps=fps)
        
        original_clip = VideoFileClip(uploaded_video_path)
        audio_clip = original_clip.audio
        final_clip = clips.set_audio(audio_clip)
        
        if final_clip.duration < audio_clip.duration:
            final_audio_clip = audio_clip.subclip(0, final_clip.duration)
            final_clip = final_clip.set_audio(final_audio_clip)
        
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
        logger.info(f"Новое видео успешно создано: {output_path}")
        
        for temp_file in temp_files:
            os.remove(temp_file)
        
        return output_path
    else:
        logger.warning("Нет кадров для создания видео.")
        return None


def gradio_ui():
    blocks = gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="pink"), title="PPersonCutter 🌊")
    with blocks as demo:
        gr.Markdown("# PersonPodcastCutter 🌊")
        main_info = gr.Markdown("""
                PersonPodcastCutter - это мощный инструмент, который позволяет с лёгкостью редактировать видео, фокусируясь на конкретных людях.\n
              """)
        with gr.Row():       
            file_input = gr.File(label="Загрузить видео")
            stop_button = gr.Button("Остановить")
            face_images = gr.Gallery(label="Найденные персоны", show_label=True)
            face_select = gr.Files(label="Перенесите нужные персоны")
            submit_btn = gr.Button("Создать видео")
            video_output = gr.Video(label="Ваше новое видео")

        with gr.Accordion(label="Настройки"):
            use_rem_bg_checkbox = gr.Checkbox(label="Use remBG", value=False)
            score_face_slider = gr.Slider(minimum=0.1, maximum=1000.0, value=0.3, step=0.1, label="Score Person Detect")
            duplicate_rate_slider = gr.Slider(minimum=0.1, maximum=100.0, value=0.9, step=0.1, label="Duplicate Rate")
            avg_face_score_text = gr.Textbox(label="AVG Person Score Detect:")
            
            settings_info = gr.Markdown("""
                **Use RemBG:** Удаляет Задний фон найденого изображения персоны.\n
                **Score Face:** Определяет  Score Face, необходимый для обнаружения персоны. Ореинтеровка - AVG.\n
                **Duplicate Rate:** Максимальная частота дублирования кадров с одним персом.Если много повторных персон, пробуйте эксперементировать с настройками. \n
                **AVG Person Score Detect:** Средний Score Face обнаруженных персон. \n
                
              """)
            
        with gr.Accordion(label="Настройки Видео"):
            fps_slider = gr.Slider(minimum=10.0, maximum=60.0, value=30.0, step=1.0, label="FPS")
            score_detect_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.5, step=0.1, label="Person Score Detect Video:")
            score_detect_text = gr.Textbox(label="АVG Person Score Detect Video:") 


            settings_info = gr.Markdown("""
                **FPS:** FPS вашего видео. Если исходник с 30 FPS, в программе больше ставить не рекомендуется. \n
                **Person Score Detect Video:** Определяет  Score Face, необходимый для обнаружения персоны в видео. Ореинтеровка - AVG. Чем меньше score, тем увереннее. \n
                **AVG Person Score Detect Video:** До этого значения Score Face Video обнаружает персоны при создании видео. \n
                
              """)

            file_input.change(
                fn=process_video,
                inputs=[file_input, score_face_slider, duplicate_rate_slider, score_detect_slider, use_rem_bg_checkbox], 
                outputs=[face_images, avg_face_score_text]
            )

            submit_btn.click(
                fn=track_faces_in_frames,
                inputs=[file_input, face_select, score_detect_slider, fps_slider, use_rem_bg_checkbox],
                outputs=[video_output, score_detect_text]
            )
            stop_button.click(fn=stop_processing, inputs=[], outputs=[])
            
        with gr.Accordion(label=""):
            big_block = gr.HTML("""
              <img src="https://i.postimg.cc/qMprpnPT/34.png" style='height: 50%;'>
            """)   

    demo.queue()
    demo.launch(favicon_path="images/icon.ico", inbrowser=True)
    
if __name__ == "__main__":
    gradio_ui()
