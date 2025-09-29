from faster_whisper import WhisperModel
import os

def transcribe_with_timestamps(video_path, model_size="small", device="cpu"):
    """
    Transcribe video/audio with Faster-Whisper and yield text + timestamps.
    """
    # Choose safe compute type
    compute_type = "float32" if device == "cpu" else "float16"

    # Load model
    model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root=r"/content/drive/MyDrive/Models")

    # Run transcription
    segments, info = model.transcribe(video_path)

    for segment in segments:
        # print(segment)

        yield {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        }
