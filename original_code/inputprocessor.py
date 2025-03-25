from spellchecker import SpellChecker
from keybert import KeyBERT

class InputProcessor:
    def __init__(self):
        self.spell = SpellChecker()
        self.kw_model = KeyBERT()

    def processInput(self, question, topk=2):
        # Split the question into words and correct typos
        corrected_words = [self.spell.correction(word) if word in self.spell.unknown(word.split()) else word for word in
                           question.split()]
        corrected_question = " ".join(corrected_words)

        # Extract keywords from corrected text
        keywords = self.kw_model.extract_keywords(corrected_question, keyphrase_ngram_range=(1, 2), stop_words='english')
        if topk <= len(keywords):
            toplist = keywords[:topk]  # [('advancement quantum', 0.8026), ('quantum computing', 0.7876)]
        else:
            toplist = keywords
        # Extract single words from each phrase and flatten into a list
        words = []
        for phrase, _ in toplist:
            words.extend(phrase.split())

        # Deduplicate using a set and convert back to a list
        unique_words = list(dict.fromkeys(words))
        unique_words = " ".join(unique_words)
        print(f"Spell-checked question: {corrected_question}")
        print(f"Keywords: {unique_words}")
        return corrected_question, unique_words

if __name__ == "__main__":
    question = "what is advancements in robot surgeon?"
    processor = InputProcessor()
    question, keywords = processor.processInput(question)
