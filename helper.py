import json, numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

punc = '''!()-[]{};:'",<>./?@#$%^&*_~'''
stop_words = json.load(open("stop_words.json")).get("stop_words")
stemmer = PorterStemmer()

stop_words_set = set()
for word in stop_words:
    stop_words_set.add(word)
    
nltk_stop_words_set = set(stopwords.words('english'))


def clean_text(txt: str):
    # Lowering the Words
    words = [w.lower().strip() for w in txt.split(" ")]
    words = [w for w in words if w != ""]
    
    # Removing Punctuations
    for i in range(len(words)):
        for p in punc:
            words[i] = words[i].replace(p, "")
    
    clean_words = [word for word in words if word not in nltk_stop_words_set]
    return " ".join(clean_words)

def get_softmax(data_array):
    # Subtract max for numerical stability
    exp_data = np.exp(data_array - np.max(data_array))
    return exp_data / np.sum(exp_data)
