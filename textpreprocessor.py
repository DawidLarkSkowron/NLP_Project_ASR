import os
import re
import nltk

# Wymuszony katalog dla danych NLTK
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.download("punkt", download_dir=nltk_data_path, quiet=True)
nltk.data.path.append(nltk_data_path)

from nltk.tokenize import sent_tokenize

class TextPreprocessor:
    def __init__(self, min_words_per_sentence=4):
        self.min_words = min_words_per_sentence

    def clean_text(self, text):
        text = re.sub(r"[^a-zA-Z0-9\s.,'’?!-]", "", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def filter_sentences(self, text):
        sentences = sent_tokenize(text)
        filtered = [s for s in sentences if len(s.split()) >= self.min_words]
        return " ".join(filtered)

    def prepare_for_summary(self, raw_text):
        cleaned = self.clean_text(raw_text)
        filtered = self.filter_sentences(cleaned)
        return filtered
