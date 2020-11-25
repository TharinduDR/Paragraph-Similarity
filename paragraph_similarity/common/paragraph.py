from nltk import tokenize


class Paragraph:
    def __init__(self, text):
        self.text = text
        self.words = text.split()
        self.sentences = tokenize.sent_tokenize(text)