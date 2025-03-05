import whisper
import torch
import os
import json
import srt
import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import tempfile
from datetime import timedelta
from subprocess import run, CalledProcessError, DEVNULL

def transcribe_audio(audio_path, model_size="large-v3", language=None):
    """
    Transcribes an audio file using OpenAI's Whisper model and returns the result.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File {audio_path} not found.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("🔄 Ładowanie modelu...")
    progress_bar.progress(20)
    
    result = model.transcribe(audio_path, language=language)
    
    progress_bar.progress(100)
    status_text.text("✅ Transkrypcja zakończona!")
    
    return result

def save_transcription(result, output_text, output_srt, output_json):
    """Saves transcription as text, SRT, and JSON file."""
    text = result["text"]
    segments = result.get("segments", [])
    
    with open(output_text, "w", encoding="utf-8") as f:
        f.write(text)
    
    subtitles = []
    for i, segment in enumerate(segments):
        start = timedelta(seconds=segment["start"])
        end = timedelta(seconds=segment["end"])
        subtitles.append(srt.Subtitle(index=i+1, start=start, end=end, content=segment["text"]))
    
    with open(output_srt, "w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))
    
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    
    return text

def record_audio(duration=5, samplerate=44100):
    """Records live audio from microphone and saves it to a temporary file."""
    st.write("🎙️ Nagrywanie...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_audio.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(recording.tobytes())
    
    st.success("✅ Nagrywanie zakończone!")
    return temp_audio.name

def check_ffmpeg():
    """Ensures that FFmpeg is installed and accessible."""
    try:
        run(["ffmpeg", "-version"], stdout=DEVNULL, stderr=DEVNULL, check=True)
    except CalledProcessError:
        st.error("⚠️ FFmpeg nie jest zainstalowany lub nie jest w PATH. Zainstaluj FFmpeg i spróbuj ponownie.")
        return False
    return True

# Streamlit UI
st.set_page_config(page_title="Transkrypcja Audio", page_icon="🎙️", layout="wide")
st.title("🎙️ Transkrypcja Audio w Czasie Rzeczywistym")
st.markdown("**Konwertuj swoje nagrania audio na tekst za pomocą modelu Whisper!**")

if not check_ffmpeg():
    st.stop()

option = st.radio("🔹 Wybierz metodę transkrypcji:", ["📂 Z pliku audio", "🎤 Nagranie na żywo"])

if option == "📂 Z pliku audio":
    uploaded_file = st.file_uploader("📤 Wgraj plik audio", type=["mp3", "wav", "m4a", "ogg"])
    if uploaded_file is not None:
        with st.spinner("⏳ Przetwarzanie pliku..."):
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]).name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
            
            result = transcribe_audio(temp_path)
            text = save_transcription(result, "transcription.txt", "transcription.srt", "transcription.json")
        
        st.success("✅ Transkrypcja zakończona!")
        st.text_area("📄 Transkrypcja:", text, height=300)
        format_choice = st.radio("📥 Wybierz format pliku do pobrania:", ["TXT", "SRT", "JSON"])
        if format_choice == "TXT":
            st.download_button("⬇️ Pobierz transkrypcję TXT", text, file_name="transcription.txt")
        elif format_choice == "SRT":
            st.download_button("🎬 Pobierz napisy SRT", open("transcription.srt").read(), file_name="transcription.srt")
        else:
            st.download_button("📜 Pobierz JSON", open("transcription.json").read(), file_name="transcription.json")

elif option == "🎤 Nagranie na żywo":
    duration = st.slider("⏱️ Czas nagrania (sekundy)", 1, 10, 5)
    if st.button("🎙️ Rozpocznij nagrywanie"):
        with st.spinner("🎤 Nagrywanie w toku..."):
            audio_path = record_audio(duration)
            st.write("⏳ Transkrybuję nagranie...")
            result = transcribe_audio(audio_path)
            text = save_transcription(result, "live_transcription.txt", "live_transcription.srt", "live_transcription.json")
        
        st.success("✅ Transkrypcja zakończona!")
        st.text_area("📄 Transkrypcja:", text, height=300)
        format_choice = st.radio("📥 Wybierz format pliku do pobrania:", ["TXT", "SRT", "JSON"], key="live_format")
        if format_choice == "TXT":
            st.download_button("⬇️ Pobierz transkrypcję TXT", text, file_name="live_transcription.txt")
        elif format_choice == "SRT":
            st.download_button("🎬 Pobierz napisy SRT", open("live_transcription.srt").read(), file_name="live_transcription.srt")
        else:
            st.download_button("📜 Pobierz JSON", open("live_transcription.json").read(), file_name="live_transcription.json")
