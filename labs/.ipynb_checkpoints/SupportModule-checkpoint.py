def split_sentences(article_text):
    """Takes a string, returns a list of its individual sentences ()"""
    return pd.Series(nltk.tokenize.sent_tokenize(article_text))

def get_scores(text: list, method='VADER'):
    if method == 'VADER':
        scores = text.apply(lambda s: sia.polarity_scores(s)['compound']) #list of compound score per sentence
    else:
        scores = None

    return scores

def get_stats(scores):
    stats = scores.describe()
    return stats

def text_to_sentiments(text):
    sentences = split_sentences(text)