import re
from nltk.stem import WordNetLemmatizer

def lemmatize(word):
    """
    WordNet lemmatizer, try verb, noun, adjective andd return the smallest.
    Always begin with verb.
    WARN: Slow code, we can for ex remove adjective from the list
    """
    lemmatizer = WordNetLemmatizer()
    
    word_v = lemmatizer.lemmatize(word, pos='v')
    word_n = lemmatizer.lemmatize(word, pos='n')
    word_a = lemmatizer.lemmatize(word, pos='a')
    
    return min([word_v, word_n, word_a], key=len)

"""
Commun useless words
"""
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) | ENGLISH_STOP_WORDS

# what is: ed
stop_words.update(["'s", "n't",
                   "e.g.", "e.g", "i.e.", "i.e", "etc.", "etc",
                   "co.", "ltd.",
                   "et", "al", "al.", "dr.",
                   "table", "figure", "fig", "fig.",
                   "url",
                   "http", "https"
                  ])


# BAD code rewrite
number = r'[\d\Wex]+' # fake number, but do the job
number_re = re.compile(f'^{number}s?$')
not_letters_digits_re = re.compile(r'^[^a-zA-Z0-9]+$')
single_re = re.compile(r'^[a-zA-Z]\.?$')

def valid_token(token):
    not_stop_words = not token in stop_words
    not_number = not number_re.match(token)
    not_punctuation = not not_letters_digits_re.match(token)
    not_single = not single_re.match(token)
    
    return (not_number and 
            not_stop_words and 
            not_punctuation and 
            not_single and 
            len(token)>1 and len(token)<70)

# Rewrite proprely, at the right place (tokenizer class)
a_reg = re.compile(fr"^[\W]+(.+)$")
reg = r'^(\d+[.e\d\W]+)((?!.*ncov$).+)'
b_reg = re.compile(reg)
c_reg = re.compile(r'^(.*)[-*]+$')

def post(token):
    # Drop non words characters at begining: "-+,=sometext" -> "sometext"
    token = a_reg.sub(r'\1',token)
    # "10km" -> "km", "10-years-old" -> "years-old" but keep "2019-ncov"
    token = b_reg.sub(r'\2',token)
    # "sometext-" -> "sometext", "word*" -> "word"
    token = c_reg.sub(r'\1',token)
    
    return token
