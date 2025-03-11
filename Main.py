import whisper
import torch
import os
import json
import srt
import streamlit as st
import tempfile
from datetime import timedelta
from subprocess import run, CalledProcessError, DEVNULL


def transcribe_audio(audio_path, model_size="large-v3", language=None):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File {audio_path} not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the Whisper model
    model = whisper.load_model(model_size, device=device)

    # Create a progress bar
    progress_bar = st.progress(0)

    # Split the audio into smaller segments
    audio = whisper.load_audio(audio_path)
    duration = len(audio) / whisper.audio.SAMPLE_RATE
    segment_duration = 30  # seconds
    segments = [(i, min(i + segment_duration, duration))
                for i in range(0, int(duration), segment_duration)]

    results = []
    for i, (start, end) in enumerate(segments):
        segment_audio = audio[int(
            start * whisper.audio.SAMPLE_RATE):int(end * whisper.audio.SAMPLE_RATE)]
        result = model.transcribe(segment_audio, language=language)
        results.append(result)
        progress_bar.progress((i + 1) / len(segments))

    # Combine results
    combined_result = {
        "text": " ".join([result["text"] for result in results]),
        "segments": [segment for result in results for segment in result.get("segments", [])]
    }

    progress_bar.empty()  # Remove the progress bar when done
    return combined_result


def save_transcription(result):
    text = result["text"]
    segments = result.get("segments", [])

    subtitles = []
    for i, segment in enumerate(segments):
        start = timedelta(seconds=segment["start"])
        end = timedelta(seconds=segment["end"])
        subtitles.append(srt.Subtitle(index=i+1, start=start,
                         end=end, content=segment["text"]))

    return text, srt.compose(subtitles), json.dumps(result, indent=4, ensure_ascii=False)


def check_ffmpeg():
    try:
        run(["ffmpeg", "-version"], stdout=DEVNULL, stderr=DEVNULL, check=True)
    except CalledProcessError:
        st.error(
            "⚠️ FFmpeg nie jest zainstalowany lub nie jest w PATH. Zainstaluj FFmpeg i spróbuj ponownie.")
        return False
    return True


# Streamlit UI
st.set_page_config(page_title="Transkrypcja Audio",
                   page_icon="🎙️", layout="wide")
st.title("🎙️ Transkrypcja Audio")
st.markdown(
    "**Konwertuj swoje nagrania audio na tekst za pomocą modelu Whisper!**")

if not check_ffmpeg():
    st.stop()

languages = {
    "Auto": None, "Angielski": "en", "Polski": "pl", "Hiszpański": "es", "Francuski": "fr",
    "Niemiecki": "de", "Włoski": "it", "Rosyjski": "ru", "Chiński": "zh", "Japoński": "ja"
}

language_choice = st.selectbox(
    "🌍 Wybierz język transkrypcji:", list(languages.keys()))
selected_language = languages[language_choice]

uploaded_file = st.file_uploader("📤 Wgraj plik audio", type=[
                                 "mp3", "wav", "m4a", "ogg"])

if uploaded_file is not None:
    if 'transcription_result' not in st.session_state:
        with st.spinner("⏳ Przetwarzanie pliku..."):
            temp_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]).name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            result = transcribe_audio(temp_path, language=selected_language)
            st.session_state.transcription_result = save_transcription(result)

    text, srt_content, json_content = st.session_state.transcription_result

    st.success("✅ Transkrypcja zakończona!")
    st.text_area("📄 Transkrypcja:", text, height=300)

    col1, col2, col3 = st.columns(3)

    col1.download_button("📥 Pobierz TXT", text,
                         "transcription.txt", "text/plain")
    col2.download_button("📥 Pobierz SRT", srt_content,
                         "transcription.srt", "text/plain")
    col3.download_button("📥 Pobierz JSON", json_content,
                         "transcription.json", "application/json")
