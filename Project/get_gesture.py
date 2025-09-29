import os
import subprocess
import cv2
import mediapipe as mp
from transformers import pipeline
from PIL import Image
import tempfile

def _ensure_opencv_can_open(path):
    cap = cv2.VideoCapture(path)
    ok = cap.isOpened()
    cap.release()
    return ok

def _ffmpeg_convert_to_mp4(src_path, dst_path):
    # requires ffmpeg installed in environment
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-c:v", "libx264", "-c:a", "aac",
        "-movflags", "+faststart",
        dst_path
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except Exception as e:
        print("ffmpeg conversion failed:", e)
        return False

def analyze_gestures(video_path, fps=1, device="cpu", model_cache_dir=None, min_detection_confidence=0.3):
    """
    Robust gesture analysis per (sampled) frame.
    Returns list of {"start":..,"end":..,"gestures":[{label,score},...]}.
    """
    # 0) check path exists
    if not os.path.exists(video_path):
        print("❌ Video file not found:", video_path)
        return []

    # 1) Attempt open
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("OpenCV could not open video. Attempting ffmpeg conversion to mp4...")
        tmp = os.path.join(tempfile.gettempdir(), "converted_for_cv2.mp4")
        ok = _ffmpeg_convert_to_mp4(video_path, tmp)
        if not ok or not _ensure_opencv_can_open(tmp):
            print("❌ Failed to convert/open with ffmpeg. Exiting.")
            return []
        video_path = tmp
        cap = cv2.VideoCapture(video_path)

    # 2) get properties
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0


    frame_interval = max(1, int(round(video_fps / fps)))

    # 3) prepare classifier (safely)
    device_param = device  # -1 CPU, 0 GPU
    
    classifier_kwargs = {"model": "prithivMLmods/Hand-Gesture-19", "device": device_param}

    gesture_classifier = pipeline("image-classification", **classifier_kwargs)

    # 4) mediapipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    gesture_data = []
    frame_count = 0
    processed_frames = 0

    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=min_detection_confidence) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                start_time = frame_count / video_fps
                end_time = (frame_count + frame_interval) / video_fps

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                has_hands = results.multi_hand_landmarks is not None

                gestures = []
                if has_hands:
                    # extract crops
                    h, w, _ = frame.shape
                    for hand_landmarks in results.multi_hand_landmarks:
                        x_coords = [lm.x for lm in hand_landmarks.landmark]
                        y_coords = [lm.y for lm in hand_landmarks.landmark]
                        x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                        y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h) # Removed extra parenthesis
                        margin = 20
                        x_min = max(x_min - margin, 0)
                        y_min = max(y_min - margin, 0)
                        x_max = min(x_max + margin, w)
                        y_max = min(y_max + margin, h)
                        crop = frame[y_min:y_max, x_min:x_max]
                        if crop.size == 0:
                            continue
                        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        try:
                            preds = gesture_classifier(pil_img)
                        except Exception as e:
                            print("Classifier error:", e)
                            preds = []
                        if preds:
                            top = preds[0]
                            gestures.append({"label": top["label"], "score": round(top["score"], 3)})

                if not gestures:
                    gestures = [{"label": "No hands", "score": 0.0}]

                gesture_data.append({
                    "start": round(start_time, 2),
                    "end": round(end_time, 2),
                    "gestures": gestures
                })
                processed_frames += 1

            frame_count += 1

    cap.release()
    return gesture_data