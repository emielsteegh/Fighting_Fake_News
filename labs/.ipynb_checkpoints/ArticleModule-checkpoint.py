import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import os
import re
import pandas as pd

#Vars for manual sentence splitter
alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu|nl|be|de)"

#file for punkt splitter
nltk.download('punkt')

#file for vader sentiment
nltk.download('vader_lexicon')

#sia
sia = SIA()

class Article:
    def __init__(self, article_text_in, article_title_in, rating_in):
        self.article_text = article_text_in
        self.article_title = article_title_in
        self.rating = rating_in
        self.prediction = None

        self.data = pd.DataFrame()

        self.set_sentences_manual()
        self.set_sentiments()

        self.stats = self.data['score'].describe()

        self.set_markers()

    def set_sentences_manual(self):
        # splitting sentences more or less manually
        # https://stackoverflow.com/questions/4576077/how-can-i-split-a-text-into-sentences
        # -*- coding: utf-8 -*-
        text = self.article_text
        text = " " + text + "  "
        text = text.replace("\n", " ")
        text = re.sub(prefixes, "\\1<prd>", text)
        text = re.sub(websites, "<prd>\\1", text)
        if "Ph.D" in text: text = text.replace("Ph.D.", "Ph<prd>D<prd>")
        text = re.sub("\s" + alphabets + "[.] ", " \\1<prd> ", text)
        text = re.sub(acronyms + " " + starters, "\\1<stop> \\2", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>\\3<prd>", text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]", "\\1<prd>\\2<prd>", text)
        text = re.sub(" " + suffixes + "[.] " + starters, " \\1<stop> \\2", text)
        text = re.sub(" " + suffixes + "[.]", " \\1<prd>", text)
        text = re.sub(" " + alphabets + "[.]", " \\1<prd>", text)
        if "”" in text: text = text.replace(".”", "”.")
        if "\"" in text: text = text.replace(".\"", "\".")
        if "!" in text: text = text.replace("!\"", "\"!")
        if "?" in text: text = text.replace("?\"", "\"?")
        if "..." in text: text = text.replace("...", "<prd><prd><prd>")
        text = text.replace(".", ".<stop>")
        text = text.replace("?", "?<stop>")
        text = text.replace("!", "!<stop>")
        text = text.replace("<prd>", ".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]

        self.data = pd.DataFrame({'sentence' : sentences})

        return None

    def set_sentences_punkt(self):
        self.data = pd.DataFrame({'sentence' : nltk.tokenize.sent_tokenize(self.article_text)})
        return None

    def get_scores(self, content, method='VADER'):
        if method == 'VADER':
            scores = sia.polarity_scores(content)['compound'] #list of compound score per sentence
        else:
            scores = None

        return pd.Series({
            'score': scores
        })

    def set_sentiments(self):
        self.data['score'] = self.data.sentence.apply(self.get_scores)
        return None

    def set_markers(self):
        self.data['marker'] = (self.data.index+1)/self.stats['count']
        return None