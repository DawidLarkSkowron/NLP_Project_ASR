import whisper
import torch
import os
import json
import srt
import streamlit as st
import tempfile
import hunspell
import requests
import difflib
from datetime import timedelta
from subprocess import run, CalledProcessError, DEVNULL

# === Pobieranie słowników ===
def download_if_needed(url, path):
    if not os.path.exists(path):
        r = requests.get(url)
        if r.status_code == 200:
            with open(path, 'wb') as f:
                f.write(r.content)
        else:
            raise Exception(f"Nie udało się pobrać pliku: {url}")

# === Inicjalizacja Hunspell z obsługą wielu języków ===
def get_hunspell_checker(lang_code):
    supported_langs = {
        "pl": "pl",
        "en": "en",
        "es": "es",
        "fr": "fr",
        "de": "de",
        "it": "it"
    }

    if lang_code not in supported_langs:
        return None

    base_url = "https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries"
    temp_dir = tempfile.gettempdir()
    lang_dir = supported_langs[lang_code]

    dic_path = os.path.join(temp_dir, f"{lang_code}.dic")
    aff_path = os.path.join(temp_dir, f"{lang_code}.aff")

    try:
        download_if_needed(f"{base_url}/{lang_dir}/index.dic", dic_path)
        download_if_needed(f"{base_url}/{lang_dir}/index.aff", aff_path)
        return hunspell.HunSpell(dic_path, aff_path)
    except Exception as e:
        st.error(f"❌ Błąd inicjalizacji Hunspell ({lang_code}): {e}")
        return None

# === Korekta pisowni tekstu ===
def correct_spelling(text, checker):
    corrected_words = []
    for word in text.split():
        if not checker.spell(word):
            suggestions = checker.suggest(word)
            corrected_words.append(suggestions[0] if suggestions else word)
        else:
            corrected_words.append(word)
    return " ".join(corrected_words)

# === Korekta pisowni segmentów (dla SRT) ===
def correct_segments(segments, checker):
    for segment in segments:
        original = segment["text"]
        segment["text"] = correct_spelling(original, checker)
    return segments

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
    "Auto": None, "Angielski": "en", "Polski": "pl", "Hiszpański": "es", "Francuski": "fr",
    "Niemiecki": "de", "Włoski": "it", "Rosyjski": "ru", "Chiński": "zh", "Japoński": "ja"
}

language_choice = st.selectbox("🌍 Wybierz język transkrypcji:", list(languages.keys()))
selected_language = languages[language_choice]

uploaded_file = st.file_uploader("📤 Wgraj plik audio", type=["mp3", "wav", "m4a", "ogg"])

if uploaded_file is not None:
    if 'transcription_result' not in st.session_state:
        with st.spinner("⏳ Przetwarzanie pliku..."):
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]).name
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())

            result = transcribe_audio(temp_path, language=selected_language)
            st.session_state.transcription_raw = result

    result = st.session_state.transcription_raw
    original_text = result["text"]
    segments = result["segments"]

    st.success("✅ Transkrypcja zakończona!")
    st.subheader("📄 Oryginalna transkrypcja:")
    st.text_area("", original_text, height=250)

    corrected_text = original_text
    corrected_segments = segments

    if selected_language in ["pl", "en", "es", "fr", "de", "it"]:
        if st.button("🪄 Popraw pisownię (Hunspell)"):
            checker = get_hunspell_checker(selected_language)
            if checker:
                with st.spinner("🔍 Sprawdzanie pisowni..."):
                    corrected_segments = correct_segments(segments.copy(), checker)
                    corrected_text = correct_spelling(original_text, checker)
                    diff = show_diff(original_text, corrected_text)

                    st.subheader("✅ Po korekcie:")
                    st.text_area("", corrected_text, height=250)
                    st.subheader("🧾 Różnice:")
                    st.markdown(diff)

    text_output, srt_output, json_output = save_transcription(corrected_text, corrected_segments)

    col1, col2, col3 = st.columns(3)
    col1.download_button("📥 Pobierz TXT", text_output, "transcription.txt", "text/plain")
    col2.download_button("📥 Pobierz SRT", srt_output, "transcription.srt", "text/plain")
    col3.download_button("📥 Pobierz JSON", json_output, "transcription.json", "application/json")
