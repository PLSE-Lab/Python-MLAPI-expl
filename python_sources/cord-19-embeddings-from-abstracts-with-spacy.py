# %% [code]
"""
Output a Gensim Word2Vec model trained on CORD-19 abstract papers
"""

print("Loading NER model ... ")
import os
os.system('pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bc5cdr_md-0.2.4.tar.gz')
os.system('pip install glove_python')
print("loaded.")

import spacy
from spacy.pipeline import merge_entities
nlp = spacy.load('en_ner_bc5cdr_md')
nlp.add_pipe(merge_entities)

import re
import numpy as np
import pandas as pd

import string
import unidecode

# nltk
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# gensim
from gensim.models import Word2Vec

# progress bar
from tqdm import tqdm

tqdm.pandas()
np.random.seed(0)

from pathlib import Path

"""
SETTINGS
"""

WORD2VEC_SIZE = 200
MODEL_OUTPUT_PATH = Path("CORD19_word2vec_abstract_13032020_{}.model".format(WORD2VEC_SIZE))

COMMON_TERMS = ["-", "-", b"\xe2\x80\x93", b"'s", b"\xe2\x80\x99s", "from", "as", "at", "by", "of", "on", 
                "into", "to", "than", "over", "in", "the", "a", "an", "/", "under", ":"]

# Settings to compute phrases, i.e new_york
# More here: Distributed Representations of Words and Phrases and their Compositionality [https://arxiv.org/abs/1310.4546]
PHRASE_DEPTH = 2
PHRASE_COUNT = 10
PHRASE_THRESHOLD = 15
        
"""
LOAD DATA
"""

CLEAN_DATA_PATH = Path("../input/cord-19-eda-parse-json-and-generate-clean-csv/")

biorxiv = pd.read_csv(CLEAN_DATA_PATH / "biorxiv_clean.csv")
pmc = pd.read_csv(CLEAN_DATA_PATH / "clean_pmc.csv")
comm_use = pd.read_csv(CLEAN_DATA_PATH / "clean_comm_use.csv")
noncomm_use = pd.read_csv(CLEAN_DATA_PATH / "clean_noncomm_use.csv")

papers = pd.concat(
    [pmc, biorxiv, comm_use, noncomm_use], axis=0
).reset_index(drop=True)

abstracts_s = papers['abstract'].dropna()


"""
TOOLS
"""


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # NLTK default tag is NOUN


lemmatizer = WordNetLemmatizer()


def lemmatize_word(word, tag):
    return lemmatizer.lemmatize(word, get_wordnet_pos(tag))


def lemmatize(tokens):
    return list(map(lambda x: lemmatize_word(x[0], x[1]), nltk.pos_tag(tokens)))


def remove_numbers(input):
    """Remove numbers from input"""
    return re.sub(r"(^\d+\s+|\s+\d+\s+|\s+\d+$)", " ", input)


def remove_punctuations(input):
    """Remove punctuations from input"""
    return input.translate(str.maketrans("", "", '!"#$%&\'()*+,./:;<=>?@[\\]^`{|}~')) # all string punctuations except '_' and '-'

def remove_pharentesis(input):
    return re.sub(r"(\(|\)|\[|\])", " ", input)

def remove_diacritics(input):
    """Remove diacritics (as accent marks) from input"""
    return unidecode.unidecode(input)


def remove_white_space(input):
    """Remove all types of spaces from input"""
    input = input.replace(u"\xa0", u" ")  # remove space
    # remove white spaces, new lines and tabs
    return " ".join(input.split())


def remove_stop_words(input):
    """Remove stopwords from input"""
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(input)
    return [i for i in words if not (i in stop_words)]


# Remove empty brackets (that could happen if the contents have been removed already
# e.g. for citation ( [3] [4] ) -> ( ) -> nothing
# https://github.com/jakelever/bio2vec/blob/master/PubMed2Txt.py
def removeBracketsWithoutWords(text):
    fixed = re.sub(r'\([\W\s]*\)', ' ', text)
    fixed = re.sub(r'\[[\W\s]*\]', ' ', fixed)
    fixed = re.sub(r'\{[\W\s]*\}', ' ', fixed)
    return fixed

# Some older articles have titles like "[A study of ...]."
# This removes the brackets while retaining the full stop
# https://github.com/jakelever/bio2vec/blob/master/PubMed2Txt.py
def removeWeirdBracketsFromOldTitles(titleText):
    titleText = titleText.strip()
    if titleText[0] == '[' and titleText[-2:] == '].':
        titleText = titleText[1:-2] + '.'
    return titleText


def clean_text(text):
    #text = text.lower()
    
    text = removeBracketsWithoutWords(text)
    
    text = removeWeirdBracketsFromOldTitles(text)
    
    text = remove_pharentesis(text)
    
    text = remove_punctuations(text) # powerful
    
    text = remove_numbers(text) # only in case SPACE NUM SPACE
    #text = remove_punctuations(text)
    #text = remove_diacritics(text)
    text = remove_white_space(text)
    #tokens = remove_stop_words(text)  # return a list of token
    
    return text


# Generating word grams. adapted from https://github.com/materialsintelligence/mat2vec/blob/master/mat2vec/training/phrase2vec.py
def wordgrams(sentences, depth, count, threshold, common_terms, d=0):
    if depth == 0:
        return sent, None
    else:
        """Builds word grams according to the specification."""
        phrases = Phrases(
            sent,
            common_terms=common_terms,
            min_count=count,
            threshold=threshold)

        grams = Phraser(phrases)
        d += 1
        if d < depth:
            return wordgrams(grams[sent], depth, count, threshold, common_terms, d)
        else:
            return grams[sent], grams

        
"""
MAIN
"""

def remove_numbers(input):
    """Remove numbers that are not close to alphabetic character"""
    return re.sub(r"(\s+\d+\s+|^\d+\s+|\s+\d+$)", "", input)

def remove_punctuations(input):
    """Remove punctuations from input"""
    return input.translate(str.maketrans("", "", '!"#$%&\'()_-*+,.:;<=>?@[\\]^`{|}~')) # all string punctuations except '_' and '-'

def remove_citations(input):
    return re.sub(r"(\[\d+\])", "", input)

def tokenized_sentences(text):
    """Return a list of tokenized sentences"""
    doc = nlp(text)
    return [tokenize(s) for s in doc.sents]

def tokenize(s):
    tokenized = []
    for token in s:
        if not token.is_punct and not token.is_space and not token.is_bracket and not token.is_quote and not token.is_stop:
            tokenized.append(token.lemma_.replace(" ", "_"))
    return tokenized

print("Clean data ...")
# Remove numbers
abstracts = abstracts_s.apply(clean_text)

# Remove white spaces
#abstracts = abstracts.apply(remove_white_space)

# Remove punctuations
#abstracts = abstracts.progress_apply(remove_punctuations)

# Abstract series with each row a list of tokenized sentences
#abstracts = abstracts.progress_apply(tokenized_sentences)

# List of tokenized sentences [["hello", "world"], ["another", "sentence"]
#content = [sent for abstract in abstracts.values for sent in abstract] 

#tokenized_text = [[token for sentence in tokenized_text for token in sentence]]

abstract_tokenized = abstracts.progress_apply(lambda a: tokenize(nlp(a)))
content = [abstract for abstract in abstract_tokenized]

#tokenized_text, _ = wordgrams(tokenized_text, depth=2, count=10, threshold=15, common_terms=COMMON_TERMS)

#print(token_text)
 
# Generating Word Embeddings from Word2Vec

print("Train model ...")
model = Word2Vec(
    content,
    size=200,
    sg=1, # skip-gram
    min_count=5, # 
    window=8,
    hs=0,
    negative=15, # negative sampling loss
    workers=16,
)
print("w2v trained.")


print("Most similar word to 'coronavirus:'")
print(model.wv.most_similar("coronavirus"))

print("Vocabulary size: ", len(model.wv.vocab))

print("Saving model ...")
model.save(str(MODEL_OUTPUT_PATH))


# Generating word embeddings from FastText

from gensim.models import FastText
model_ft = FastText(content, 
                    size=200, 
                    window=8, 
                    min_count=5, 
                    workers=4,
                    sg=1,
                    negative=15)

print("FatstText trained.")


print("Most similar word to 'coronavirus:'")
print(model_ft.wv.most_similar("coronavirus"))

print("Vocabulary size: ", len(model_ft.wv.vocab))

print("Saving model ...")
model_ft.save(str(MODEL_OUTPUT_PATH)+"_FastText")

# Generating word embeddings from GloVe

from glove import Corpus, Glove

corpus = Corpus()
corpus.fit(content, window=8)

glove = Glove(no_components=5, learning_rate=0.001) 
glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
glove.add_dictionary(corpus.dictionary)

print(glove.most_similar("coronavirus"))
glove.save(str(MODEL_OUTPUT_PATH)+"_GloVe")