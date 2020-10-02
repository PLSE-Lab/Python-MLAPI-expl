from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import nltk
#nltk.download('all')

text =open("../input/001.txt").read()
#print(text)

#membuat definisi untuk meringkas text
def get_summary(text, pct):
    summary = summarize(text, ratio=pct, split=True)
    return summary

def get_keyword(text):
    res = keywords(
        text, ratio=0.1,
        words=None,
        split=False,
        scores=False,
        pos_filter = ('NN','JJ'),
        lemmatize=False,
        deacc=False)
    
    res = res.split("\n")
    return res


print("Input data\n")
print(text)
print("Printing summary")
print("===========================================================")
print(get_summary(text, 0.3))
print("===========================================================")
print(get_keyword(text))

#definisi baru untuk memisahkan antara kata dengan kalimat
def sent_to_word(sentence):
    for st in sentence:
        yield(gensim.utils.simple_preprocess(
            str(sentence), deacc=True
        ))

print("\nRingkasan kosa kata yang telah terpotong: \n")
print("============================================================")
list(sent_to_word(get_summary(text, 0.3)))


    