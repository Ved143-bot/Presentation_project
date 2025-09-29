def merge_audio_gesture(audio_data, gesture_data):
    """
    Merge audio/text analysis with gesture detection data.
    For each transcript segment, attach all gestures that happened in its time window.
    """
    merged = []

    for seg in audio_data:
        seg_start, seg_end = seg["start"], seg["end"]

        # collect gestures within this segment
        gestures_in_range = [
            g for g in gesture_data
            if g["start"] >= seg_start and g["end"] <= seg_end
        ]

        # flatten gestures (optional: keep full timeline)
        gesture_summary = []
        for g in gestures_in_range:
            gesture_summary.extend(g["gestures"])

        merged.append({
            **seg,  # keep transcript + features
            "gesture_summary": gesture_summary      # flat list of all gestures
        })

    return merged