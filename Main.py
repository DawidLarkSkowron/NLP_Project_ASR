import whisper
import torch
import os
import json
import srt
import streamlit as st
import tempfile
import language_tool_python
import requests
import difflib
from datetime import timedelta
from subprocess import run, CalledProcessError, DEVNULL
from emotion_detection import EmotionDetector

# === Pobieranie słowników ===
def download_if_needed(url, path):
    if not os.path.exists(path):
        r = requests.get(url)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                f.write(r.content)
        else:
            raise Exception(f"Nie udało się pobrać pliku: {url}")

# === Inicjalizacja LanguageTool ===
def get_language_tool(lang_code):
    try:
        tool = language_tool_python.LanguageTool(lang_code)
        return tool
    except Exception as e:
        st.error(f"❌ Błąd inicjalizacji LanguageTool ({lang_code}): {e}")
        return None

# === Korekta pisowni tekstu ===
def correct_spelling(text, tool):
    if not tool:
        return text
    
    matches = tool.check(text)
    corrected_text = language_tool_python.utils.correct(text, matches)
    return corrected_text

# === Korekta pisowni segmentów (dla SRT) ===
def correct_segments(segments, tool):
    if not tool:
        return segments
        
    corrected_segments = []
    for segment in segments:
        original = segment["text"]
        segment["text"] = correct_spelling(original, tool)
        corrected_segments.append(segment)
    return corrected_segments

# === Różnice tekstu ===
def show_diff(original, corrected):
    diff = difflib.ndiff(original.split(), corrected.split())
    return " ".join(
        [f"**{w[2:]}**" if w.startswith("+ ") else w[2:] for w in diff if not w.startswith("- ")]
    )

# === Transkrypcja audio ===
def transcribe_audio(audio_path, model_size="large-v3", language=None):
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File {audio_path} not found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model(model_size, device=device)

    progress_bar = st.progress(0)
    audio = whisper.load_audio(audio_path)
    duration = len(audio) / whisper.audio.SAMPLE_RATE
    segment_duration = 30
    segments = [(i, min(i + segment_duration, duration))
                for i in range(0, int(duration), segment_duration)]

    results = []
    for i, (start, end) in enumerate(segments):
        segment_audio = audio[int(start * whisper.audio.SAMPLE_RATE):int(end * whisper.audio.SAMPLE_RATE)]
        result = model.transcribe(segment_audio, language=language)
        results.append(result)
        progress_bar.progress((i + 1) / len(segments))

    combined_result = {
        "text": " ".join([result["text"] for result in results]),
        "segments": [segment for result in results for segment in result.get("segments", [])]
    }

    progress_bar.empty()
    return combined_result

# === Zapis transkrypcji ===
def save_transcription(text, segments):
    subtitles = []
    for i, segment in enumerate(segments):
        start = timedelta(seconds=segment["start"])
        end = timedelta(seconds=segment["end"])
        subtitles.append(srt.Subtitle(index=i+1, start=start, end=end, content=segment["text"]))

    return text, srt.compose(subtitles), json.dumps({
        "text": text,
        "segments": segments
    }, indent=4, ensure_ascii=False)

# === Sprawdzenie ffmpeg ===
def check_ffmpeg():
    try:
        run(["ffmpeg", "-version"], stdout=DEVNULL, stderr=DEVNULL, check=True)
    except CalledProcessError:
        st.error("⚠️ FFmpeg nie jest zainstalowany lub nie jest w PATH. Zainstaluj FFmpeg i spróbuj ponownie.")
        return False
    return True

# === UI ===
st.set_page_config(page_title="Transkrypcja Audio", page_icon="🎙️", layout="wide")
st.title("🎙️ Transkrypcja Audio")
st.markdown("**Konwertuj swoje nagrania audio na tekst za pomocą modelu Whisper i sprawdź pisownię!**")

if not check_ffmpeg():
    st.stop()

languages = {
    "Angielski": "en", "Polski": "pl"
}

language_choice = st.selectbox("🌍 Wybierz język transkrypcji:", list(languages.keys()), index=0)
selected_language = languages[language_choice]

uploaded_file = st.file_uploader("📤 Wgraj plik audio", type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file is not None:
    # Zapisz ścieżkę do tymczasowego pliku w sesji
    if 'temp_audio_path' not in st.session_state:
        temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]).name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        st.session_state.temp_audio_path = temp_path
        
        # Wykonaj transkrypcję
        with st.spinner("⏳ Przetwarzanie pliku..."):
            result = transcribe_audio(temp_path, language=selected_language)
            st.session_state.transcription_raw = result
            
    else:
        result = st.session_state.transcription_raw
    
    original_text = result["text"]
    segments = result["segments"]
    
    # Wyświetl wyniki transkrypcji
    # Wyświetl wyniki transkrypcji
    st.success("✅ Transkrypcja zakończona!")
    st.subheader("📄 Oryginalna transkrypcja:")
    st.text_area("Oryginalny tekst", original_text, height=250, key="original_text")
    
    # Formularz do korekty pisowni
    with st.form("spelling_correction_form"):
        corrected_text = original_text
        corrected_segments = segments
        
        if selected_language in ["pl", "en"]:
            if st.form_submit_button("🪄 Popraw pisownię (Hunspell)"):
                tool = get_language_tool(selected_language)
                if tool:
                    with st.spinner("🔍 Sprawdzanie pisowni..."):
                        corrected_segments = correct_segments(segments.copy(), tool)
                        corrected_text = correct_spelling(original_text, tool)
                        diff = show_diff(original_text, corrected_text)

                        st.subheader("✅ Po korekcie:")
                        st.text_area("Tekst po korekcie", corrected_text, height=250, key="corrected_text")
                        st.subheader("🧾 Różnice:")
                        st.markdown(diff)

    # Formularz do wykrywania emocji
    with st.form("emotion_detection_form"):
        if st.form_submit_button("🧠 Wykryj emocje"):
            with st.spinner("🔍 Wykrywanie emocji..."):
                # Przekazujemy wybrany język do detektora emocji
                language_code = selected_language if selected_language else "en"
                detector = EmotionDetector(language=language_code)
                emotion_results = detector.process_transcription({"text": corrected_text, "segments": corrected_segments})
                if emotion_results:
                    st.subheader("🧠 Emocje wykryte w transkrypcji:")
                    
                    # Definiuj emotki dla różnych emocji
                    emotion_emojis = {
                        "happiness": "😄",  # uśmiech
                        "sadness": "😢",    # płacz
                        "anger": "😡",      # złość
                        "fear": "😨",       # strach
                        "surprise": "😮",    # zaskoczenie
                        "neutral": "😐",     # neutralna
                        "disgust": "🤢",     # obrzydzenie
                        "joy": "😃",         # radość
                        "love": "😍",       # miłość
                        "admiration": "😊", # podziw
                        "amusement": "😂",  # rozbawienie
                        "annoyance": "😒",  # irytacja
                        "approval": "👍",    # aprobata
                        "caring": "🤗",      # troska
                        "confusion": "🤔",   # zmieszanie
                        "curiosity": "🤓",   # ciekawość
                        "desire": "😏",      # pożądanie
                        "disappointment": "😞", # rozczarowanie
                        "disapproval": "👎", # dezaprobata
                        "embarrassment": "😳", # zawstydzenie
                        "excitement": "😁",  # ekscytacja
                        "gratitude": "🙏",   # wdzięczność
                        "grief": "😔",      # żal
                        "nervousness": "😬", # zdenerwowanie
                        "optimism": "🙂",    # optymizm
                        "pride": "🤩",       # duma
                        "realization": "😯", # uświadomienie
                        "relief": "😌",      # ulga
                        "remorse": "😕",     # wyrzuty sumienia
                    }
                    
                    # Grupuj wyniki według emocji
                    emotions_grouped = {}
                    for result in emotion_results:
                        emotion_label = result['emotion']['label']
                        if emotion_label not in emotions_grouped:
                            emotions_grouped[emotion_label] = []
                        emotions_grouped[emotion_label].append(result)
                    
                    # Wyświetl pogrupowane emocje
                    for emotion, results in emotions_grouped.items():
                        emoji = emotion_emojis.get(emotion.lower(), "😶")  # Domyślna emotka jeśli nie znaleziono
                        avg_score = sum(r['emotion']['score'] for r in results) / len(results)
                        
                        # Nagłówek dla emocji
                        st.markdown(f"### {emoji} **{emotion}** (pewność: {avg_score:.2f})")
                        
                        # Fragment tekstu dla tej emocji
                        st.markdown("#### Fragmenty:")
                        for i, result in enumerate(results, 1):
                            start_time = result.get('start', 0)
                            end_time = result.get('end', 0)
                            text = result.get('text', 'Brak tekstu')
                            st.markdown(f"**{i}.** *{start_time:.1f}s - {end_time:.1f}s:* \"{text}\"")
                        
                        st.markdown("---")
                    
                    # Zapisz wyniki emocji do sesji
                    st.session_state.emotion_results = emotion_results

    # Przyciski do pobierania plików
    col1, col2, col3 = st.columns(3)
    
    text_output, srt_output, json_output = save_transcription(corrected_text, corrected_segments)
    
    if st.session_state.get('emotion_results'):
        emotion_json = json.dumps(st.session_state.emotion_results, indent=4, ensure_ascii=False)
        col1.download_button("📥 Pobierz wyniki emocji", emotion_json, "emotion_results.json", "application/json", key="download_emotions")
    
    col2.download_button("📥 Pobierz TXT", text_output, "transcription.txt", "text/plain", key="download_txt")
    col3.download_button("📥 Pobierz SRT", srt_output, "transcription.srt", "text/plain", key="download_srt")
    st.download_button("📥 Pobierz JSON", json_output, "transcription.json", "application/json", key="download_json")
