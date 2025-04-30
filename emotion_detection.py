import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import json
import streamlit as st

class EmotionDetector:
    def __init__(self, language="en"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.language = language
        
        # Wybierz odpowiedni model w zależności od wybranego języka
        if language == "pl":
            # Dla języka polskiego użyjemy tego samego modelu co dla angielskiego
            # ponieważ nie ma dedykowanego modelu dla języka polskiego
            # W przyszłości można dodać fine-tuned model dla języka polskiego
            model_name = "j-hartmann/emotion-english-distilroberta-base"
            st.warning("⚠️ Dla języka polskiego używany jest model angielski. Wyniki mogą być mniej dokładne.")
        else:  # Domyślnie angielski
            model_name = "j-hartmann/emotion-english-distilroberta-base"
            
        # Załaduj model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, device=0 if self.device == "cuda" else -1)
        
    def detect_emotions(self, text):
        """
        Wykrywa emocje w tekście.
        Zwraca słownik z wynikami dla każdej emocji.
        """
        try:
            results = self.pipeline(text)
            
            # Dla modelu polskiego wyniki mogą być inne niż dla angielskiego
            if self.language == "pl":
                # Model sdadas/polish-roberta-large-emotions ma inny format wyników
                emotion_label = results[0]["label"]
                # Mapowanie polskich emocji na angielskie (dla spójności w wyświetlaniu)
                pl_to_en_emotions = {
                    "radość": "joy",
                    "smutek": "sadness",
                    "gniew": "anger",
                    "strach": "fear",
                    "zaskoczenie": "surprise",
                    "neutralny": "neutral",
                    "odraza": "disgust",
                    "wstręt": "disgust",
                    "zniesmaczenie": "disgust"
                }
                emotion_label = pl_to_en_emotions.get(emotion_label.lower(), emotion_label)
            else:
                # Model angielski
                emotion_label = results[0]["label"]
                
            return {
                "label": emotion_label,
                "score": results[0]["score"],
                "text": text
            }
        except Exception as e:
            st.error(f"Błąd podczas wykrywania emocji: {str(e)}")
            return None

    def process_transcription(self, transcription):
        """
        Przetwarza całą transkrypcję i wykrywa emocje w każdym segmencie.
        """
        if not transcription:
            return None
            
        emotion_results = []
        for segment in transcription.get("segments", []):
            segment_text = segment["text"]
            emotion = self.detect_emotions(segment_text)
            if emotion:
                emotion_results.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment_text,
                    "emotion": emotion
                })
        
        return emotion_results

def save_emotion_results(results, filename="emotion_results.json"):
    """
    Zapisuje wyniki wykrywania emocji do pliku JSON.
    """
    if not results:
        return None
        
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    return filename

# Testowanie
if __name__ == "__main__":
    # Przykład użycia
    detector = EmotionDetector()
    
    # Test na przykładowym tekście
    test_text = "I'm so happy today! Everything is going great!"
    result = detector.detect_emotions(test_text)
    print(f"Emotion detection result: {result}")
    
    # Test na pełnej transkrypcji
    test_transcription = {
        "text": "Hello, how are you? I'm feeling great today.",
        "segments": [
            {"start": 0, "end": 5, "text": "Hello, how are you?"},
            {"start": 5, "end": 10, "text": "I'm feeling great today."}
        ]
    }
    
    emotion_results = detector.process_transcription(test_transcription)
    if emotion_results:
        filename = save_emotion_results(emotion_results)
        print(f"Results saved to: {filename}")
