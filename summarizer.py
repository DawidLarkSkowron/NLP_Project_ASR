from transformers import pipeline

class LocalBartSummarizer:
    def __init__(self):
        # Tworzymy pipeline do streszczania z BART
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def summarize_text(self, text, max_length=150, min_length=40):
        """
        Streszcza podany tekst.
        """
        if not text or len(text.strip()) == 0:
            return "Brak tekstu do streszczenia."

        # BART ma ograniczenie ~1024 tokenów wejściowych, więc najlepiej dzielić długi tekst
        if len(text.split()) > 800:
            return self._chunk_and_summarize(text)
        else:
            result = self.summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return result[0]['summary_text']

    def _chunk_and_summarize(self, text, chunk_size=700):
        """
        Dzieli długi tekst na kawałki i streszcza każdy osobno.
        """
        words = text.split()
        chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        summaries = []

        for i, chunk in enumerate(chunks):
            result = self.summarizer(chunk, max_length=150, min_length=40, do_sample=False)
            summaries.append(f"[Fragment {i+1}]\n{result[0]['summary_text']}")

        return "\n\n".join(summaries)
