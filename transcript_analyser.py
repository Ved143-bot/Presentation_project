import language_tool_python
from transformers import pipeline
import textstat

# Load NLP tools once
tool = language_tool_python.LanguageTool('en-US')
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",device = "cpu")
emotion_analyzer = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1,device = "cpu")

# Define filler words list
FILLER_WORDS = {"um", "uh", "like", "you know", "basically", "actually", "so", "well"}

def analyze_transcript(transcript):
    """
    Takes transcript [{start, end, text}] and returns enriched analysis per sentence.
    """
    enriched = []
    prev_end = 0.0  # track pauses between sentences

    for seg in transcript:
        start, end, text = seg["start"], seg["end"], seg["text"]
        duration = max(end - start, 0.001)

        # Pace
        words = text.split()
        word_count = len(words)
        pace_wps = word_count / duration
        pace_wpm = pace_wps * 60

        # Pause before this segment
        pause_before = max(0, start - prev_end)

        # Grammar mistakes
        matches = tool.check(text)
        grammar_errors = len(matches)

        # Filler words count
        filler_count = sum(1 for w in words if w.lower() in FILLER_WORDS)

        # Sentiment & emotion
        sentiment_result = sentiment_analyzer(text[:512])[0]
        emotion_output = emotion_analyzer(text[:512])
        emotion_result = emotion_output[0][0]


        # Lexical richness
        unique_words = len(set(w.lower() for w in words))
        richness = round(unique_words / word_count, 3) if word_count > 0 else 0

        # Readability (Flesch Reading Ease)
        readability = textstat.flesch_reading_ease(text)

        enriched.append({
            "start": start,
            "end": end,
            "text": text,
            "features": {
                "duration_sec": round(duration, 2),
                "word_count": word_count,
                "pace_wpm": round(pace_wpm, 2),
                "pause_before_sec": round(pause_before, 2),
                "grammar_errors": grammar_errors,
                "filler_word_count": filler_count,
                "sentiment": sentiment_result["label"],
                "sentiment_score": round(sentiment_result["score"], 3),
                "emotion": emotion_result["label"],
                "emotion_score": round(emotion_result["score"], 3),
                "lexical_richness": richness,
                "readability_score": readability
            }
        })

        prev_end = end  # update for next sentence

    return enriched