import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA


class Article:

    def __init__(self, sentence_list: list = []):
        self.sentence_list = sentence_list
        self.sentiment_list = None

    def sentences_to_sentiment(self):
        sia = SIA()
        new_sentiment_list = []
        for sentence in self.sentences_list:
            sentiment_eval = sia.polarity_scores(sentence)['compound']
            new_sentiment_list.extend(sentiment_eval)
        self.sentiment_list = new_sentiment_list
