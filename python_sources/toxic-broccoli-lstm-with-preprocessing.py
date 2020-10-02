#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import nltk
from tqdm import tqdm_notebook
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
tqdm_notebook().pandas()

def echo(s):
    print(s)
    os.system('echo \"' + str(s) + '\"')

import os
echo(os.listdir("../input"))

import pickle
import psutil
import multiprocessing as mp
import itertools
cores = psutil.cpu_count()
echo(f"There are {cores} CPU cores.")
partitions = 4
echo(f"Using {partitions} partitions.")

import emoji
import unidecode
import re
import random
from string import punctuation
from sklearn import model_selection
from sklearn import metrics

from typing import List
from IPython.display import display, HTML

def iterate_counter(fn):
    global counter
    counter = 0
    def f(*args, **kwargs):
        global counter
        res = fn(*args, **kwargs)
        counter += 1
        if counter % 100000 == 0 and counter > 1:
            echo(f"Processed has iterated {counter} times.")
        return res
    return f


# In[ ]:


# disable progress bars when submitting
def is_interactive():
   return 'SHLVL' not in os.environ

if not is_interactive():
    def nop(it, *a, **k):
        return it
    tqdm = nop


# In[ ]:


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)


# In[ ]:


from typing import List
import numpy as np

"""
Interface definitions
"""
class BasePreprocess():
    def __init__(self):
        raise NotImplementedError("This is an interface, dont call me")
    def get_preprocess(self, sentence: str) -> List[str]:
        """
        @param sentence - a string of untokenized sentence.
        @return sentence - preprocessed sentence
        """
        raise NotImplementedError("This is an interface, dont call me")
    def get_batch_preprocess(self, text: List[str]) -> List[List[str]]:
        """
        @param text - a list of string of untokenized sentence.
        @return tokens - a list of preprocessed sentence
        """
        raise NotImplementedError("This is an interface, dont call me")

class BaseTokenizer():
    def __init__(self):
        raise NotImplementedError("This is an interface, dont call me")
    def get_tokens(self, sentence: str) -> List[str]:
        """
        @param sentence - a string of untokenized sentence.
        @return tokens - a list of string of tokens
        """
        raise NotImplementedError("This is an interface, dont call me")
    def get_batch_tokens(self, text: List[str]) -> List[List[str]]:
        """
        @param text - a list of string of untokenized sentence.
        @return tokens - a list of list of string of tokens
        """
        raise NotImplementedError("This is an interface, dont call me")

class BaseEmbedding():
    def __init__(self):
        raise NotImplementedError("This is an interface, dont call me")
    def get_embedding(self, tokens: List[str]) -> np.ndarray:
        """
        @param tokens - a list of string of tokens
        @return embedding - the embedding of the sentence.
        """
        raise NotImplementedError("This is an interface, dont call me")
    def get_batch_embedding(self, text_tokens: List[List[str]]) -> np.ndarray:
        """
        @param text_tokens - a list of tokens
        @return embedding - the embedding of the list of tokens.
        """
        raise NotImplementedError("This is an interface, dont call me")

class BaseModel():
    def __init__(self):
        raise NotImplementedError("This is an interface, dont call me")
    def predict(self, embed: np.ndarray) -> np.float32:
        """
        @param embed - the embedding of the sentence.
        @return prediction - the prediction of the embedding.
        """
        raise NotImplementedError("This is an interface, dont call me")
    def get_batch_predict(self, batch_embed: np.ndarray) -> np.ndarray:
        """
        @param batch_embed -  the embedding of the list of tokens.
        @return  prediction - the prediction of the list of embeddings.
        """
        raise NotImplementedError("This is an interface, dont call me")


# In[ ]:


class ToxicBroccoliPreprocess(BasePreprocess):
    def __init__(self):
        self.stemmer = nltk.stem.snowball.SnowballStemmer(language='english')
        self.stopwords = '((^|[\s]+)' + '([\s]+|$))|((^|[\s]+)'.join(nltk.corpus.stopwords.words('english')) + '([\s]+|$))'
    
    def urls(self, sentence):
        '''
        Keeps only alphanumerics in URLs of the sentence.
        - Related to spam or propaganda.
        - Negatively correlated with target.
        '''
        tokens = sentence.split()
        sentence, url_count = re.subn(
            r'[^\s]*((:\/\/)|(www.))[^\s]*',
            lambda x: re.sub(r'(%20)|([^A-Za-z0-9])', ' ', x.group(0)),
            sentence
        )
        return sentence, {
            'urls': url_count
        }
    
    def uncommon_chars(self, sentence, sentence_length):
        '''
        Uncommon characters are characters that are neither alphanumeric nor common punctuation from python string library.
        Replaces some common uncommon characters and replaces the rest with spaces.
        - Related to spam or propaganda.
        - Negatively correlated with target.
        '''
        uncommon_pattern = re.compile(r"([^a-zA-Z0-9\s\\" + r'\\'.join(punctuation) + r"]+)")
        char_count = 0
        for match in uncommon_pattern.finditer(sentence):
            char_count += len(match.group(0))
        pct_uncommon = char_count / sentence_length
        sentence = emoji.demojize(sentence)
        sentence = re.sub(r':[\w]+:', lambda x: x.group(0).replace('_', ' '), sentence)
        sentence = unidecode.unidecode(sentence)
        return sentence, {
            'uncommon_chars': char_count,
            'uncommon_pct': pct_uncommon
        }
    
    def uppercase(self, sentence):
        '''
        Convert sentence to lowercase and count the number of capital letters.
        - Highly correlated with toxicity subattributes.
        - Somewhat correlates with spam.
        '''
        caps_count = 0
        full_caps_tokens_count = 0
        tokens = sentence.split()
        for token in tokens:
            if token.isupper():
                full_caps_tokens_count += 1
            for c in token:
                if c.isupper():
                    caps_count += 1
        sentence = sentence.lower()
        return sentence, {
            'caps': caps_count,
            'full_caps': full_caps_tokens_count
        }

    def contractions(self, sentence, use_extra=True):
        '''
        Replaces some of the most common contractions with their expanded forms.
        - Usable only after converted to lowercase
        - Converts ` to ' because a lot of the contractions use `
        '''
        sentence = re.sub('`', '\'', sentence)
        contractions_mapping = {
            "don't": "do not", "i'm": "i am", "can't": "cannot", "doesn't": "does not", "didn't": "did not", "you're": "you are",
            "isn't": "is not", "won't": "will not", "i've": "i have", "they're": "they are", "aren't": "are not", "wouldn't": "would not",
            "wasn't": "was not", "i'd": "i would", "i'll": "i will", "we're": "we are", "couldn't": "could not", "haven't": "have not",
            "shouldn't": "should not", "you've": "you have", "hasn't": "has not", "you'll": "you will", "we've": "we have", "we'll": "we will",
            "they've": "they have", "weren't": "were not", "you'd": "you would", "they'll": "they will", "he'll": "he will", "ain't": "am not",
            "they'd": "they would", "he'd": "he would", "gov't": "government", "we'd": "we would", "it'll": "it will", "hadn't": "had not",
            "would've": "would have", "she'll": "she will", "that'll": "that will", "who've": "who have", "she'd": "she would", "it'd": "it would",
            "it'd": "it would", "should've": "should have", "y'all": "you all", "who'd": "who would", "could've": "could have",
            "there'd": "there would", "cont'd": "continued", "there'll": "there will", "ma'am": "madam", "how'd": "how did", "who'll": "who will",
            "needn't": "need not", "must've": "must have", "that'd": "that would", "y'know": "you know", "ya'll": "you all", "mustn't": "must not",
            "where'd": "where did", "might've": "might have", "who're": "who are", "that've": "that have", "ne'er": "never",
            "there're": "there are", "this'll": "this will", "what'll": "what will", "what'd": "what did", "what're": "what are",
            "there've": "there have", "where've": "where have", "what've": "what have", "that're": "that are"
        }
        if use_extra:
            extra = { # from https://gist.github.com/nealrs/96342d8231b75cf4bb82
              "ain't": "am not",
              "aren't": "are not",
              "can't": "cannot",
              "can't've": "cannot have",
              "'cause": "because",
              "could've": "could have",
              "couldn't": "could not",
              "couldn't've": "could not have",
              "didn't": "did not",
              "doesn't": "does not",
              "don't": "do not",
              "hadn't": "had not",
              "hadn't've": "had not have",
              "hasn't": "has not",
              "haven't": "have not",
              "he'd": "he would",
              "he'd've": "he would have",
              "he'll": "he will",
              "he'll've": "he will have",
              "he's": "he is",
              "how'd": "how did",
              "how'd'y": "how do you",
              "how'll": "how will",
              "how's": "how is",
              "i'd": "i would",
              "i'd've": "i would have",
              "i'll": "i will",
              "i'll've": "i will have",
              "i'm": "i am",
              "i've": "i have",
              "isn't": "is not",
              "it'd": "it had",
              "it'd've": "it would have",
              "it'll": "it will",
              "it'll've": "it will have",
              "it's": "it is",
              "let's": "let us",
              "ma'am": "madam",
              "mayn't": "may not",
              "might've": "might have",
              "mightn't": "might not",
              "mightn't've": "might not have",
              "must've": "must have",
              "mustn't": "must not",
              "mustn't've": "must not have",
              "needn't": "need not",
              "needn't've": "need not have",
              "o'clock": "of the clock",
              "oughtn't": "ought not",
              "oughtn't've": "ought not have",
              "shan't": "shall not",
              "sha'n't": "shall not",
              "shan't've": "shall not have",
              "she'd": "she would",
              "she'd've": "she would have",
              "she'll": "she will",
              "she'll've": "she will have",
              "she's": "she is",
              "should've": "should have",
              "shouldn't": "should not",
              "shouldn't've": "should not have",
              "so've": "so have",
              "so's": "so is",
              "that'd": "that would",
              "that'd've": "that would have",
              "that's": "that is",
              "there'd": "there had",
              "there'd've": "there would have",
              "there's": "there is",
              "they'd": "they would",
              "they'd've": "they would have",
              "they'll": "they will",
              "they'll've": "they will have",
              "they're": "they are",
              "they've": "they have",
              "to've": "to have",
              "wasn't": "was not",
              "we'd": "we had",
              "we'd've": "we would have",
              "we'll": "we will",
              "we'll've": "we will have",
              "we're": "we are",
              "we've": "we have",
              "weren't": "were not",
              "what'll": "what will",
              "what'll've": "what will have",
              "what're": "what are",
              "what's": "what is",
              "what've": "what have",
              "when's": "when is",
              "when've": "when have",
              "where'd": "where did",
              "where's": "where is",
              "where've": "where have",
              "who'll": "who will",
              "who'll've": "who will have",
              "who's": "who is",
              "who've": "who have",
              "why's": "why is",
              "why've": "why have",
              "will've": "will have",
              "won't": "will not",
              "won't've": "will not have",
              "would've": "would have",
              "wouldn't": "would not",
              "wouldn't've": "would not have",
              "y'all": "you all",
              "y'alls": "you alls",
              "y'all'd": "you all would",
              "y'all'd've": "you all would have",
              "y'all're": "you all are",
              "y'all've": "you all have",
              "you'd": "you had",
              "you'd've": "you would have",
              "you'll": "you you will",
              "you'll've": "you you will have",
              "you're": "you are",
              "you've": "you have"
            }
            contractions_mapping = {**contractions_mapping, **extra}
        contractions_pattern = re.compile('(%s)' % '|'.join(contractions_mapping.keys()))
        sentence = contractions_pattern.sub(lambda x: contractions_mapping[x.group(0)], sentence)
        return sentence
    
    def list_items(self, sentence):
        if re.match('\(1\)|\(1\.\)|1\)|1\.\)', sentence) and re.match('\(2\)|\(2\.\)|2\)|2\.\)', sentence):
            sentence = re.sub('\(1\)|\(1\.\)|1\)|1\.\)', ' @LIST ', sentence)
            sentence = re.sub('\(2\)|\(2\.\)|2\)|2\.\)', ' @LIST ', sentence)
            for i in range(3, 20):
                if re.match(f'\({i}\)|\({i}\.\)|{i}\)|{i}\.\)', sentence):
                    sentence = re.sub('\({i}\)|\({i}\.\)|{i}\)|{i}\.\)', ' @LIST ', sentence)
                else:
                    break
        if re.match('\(a\)|\(a\.\)|a\)|a\.\)', sentence) and re.match('\(b\)|\(b\.\)|b\)|b\.\)', sentence):
            sentence = re.sub('\(a\)|\(a\.\)|a\)|a\.\)', ' @LIST ', sentence)
            sentence = re.sub('\(b\)|\(b\.\)|b\)|b\.\)', ' @LIST ', sentence)
            for i in range(2, 26):
                c = chr(ord('a') + i)
                if re.match(f'\({c}\)|\({c}\.\)|{c}\)|{c}\.\)', sentence):
                    comment = re.sub('\({c}\)|\({c}\.\)|{c}\)|{c}\.\)', ' @LIST ', sentence)
                else:
                    break
        roman_numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi', 'xii', 'xiii', 'xiv', 'xv']
        if re.match('\(i\)|\(i\.\)|i\)|i\.\)', sentence) and re.match('\(ii\)|\(ii\.\)|ii\)|ii\.\)', sentence):
            sentence = re.sub('\(i\)|\(i\.\)|i\)|i\.\)', ' @LIST ', sentence)
            sentence = re.sub('\(ii\)|\(ii\.\)|ii\)|ii\.\)', ' @LIST ', sentence)
            for i in range(2, len(roman_numerals)):
                num = roman_numerals[i]
                if re.match(f'\({num}\)|\({num}\.\)|{num}\)|{num}\.\)', sentence):
                    sentence = re.sub('\({num}\)|\({num}\.\)|{num}\)|{num}\.\)', ' @LIST ', sentence)
                else:
                    break
        sentence = re.sub('#[0-9]+', ' # @LIST ', sentence)
        return sentence
    
    def text_emojis(self, sentence):
        emojis = {
            'happy': [':)', ':-)'],
            'frown': [':(', '):'],
            'wink': [';)', ';-)']
        }
        for key in emojis:
            pattern = re.compile(r'(' + r'|'.join(['\\' + '\\'.join(token) for token in emojis[key]]) + ')')
            sentence = pattern.sub(' ' + key + ' ', sentence)
        return sentence

    def closing_puncts(self, sentence):
        sentence = re.sub('[\[\{\<]', '(', sentence)
        sentence = re.sub('[\]\}\>]', ')', sentence)
        sentence = self.list_items(sentence)
        sentence = self.text_emojis(sentence)
        sentence = re.sub('[\(\)]', ' ', sentence)
        return sentence
    
    def exclamations_questions(self, sentence):
        sentence = re.sub('![\s]+!', '!!', sentence)
        sentence = re.sub('![\s]+!', '!!', sentence) # twice to substitute for all
        sentence = re.sub('\?[\s]+\?', '??', sentence)
        sentence = re.sub('\?[\s]+\?', '??', sentence)
        sentence = re.sub('\?[\s]+!', '?!', sentence)
        sentence = re.sub('![\s]+\?', '!?', sentence)
        sentence = re.sub(r'[!\?]+', ' \g<0> ', sentence)
        sentence = re.sub(r'[!\?]+', lambda x: ''.join(sorted(x.group(0), reverse=True)), sentence)
        sentence = re.sub('(!+)|(\?+)', ' \g<0> ', sentence)
        return sentence
    
    def censored(self, sentence):
        puncts = '#$%&*+@\\^`|~'
        sentence = re.sub('[\\' + '\\'.join(puncts) + ']+', lambda x: ' @CENSOR ' if len(set(x.group(0))) >= 3 else x.group(0), sentence)
        return sentence
    
    def asterisks(self, sentence):
        def process(token):
            token = token.group(0)
            if not (re.search('[a-zA-Z]', token) is None):
                has_bold = False
                if token[0] == '*' and token[-1] == '*':
                    has_bold = True
                    token = token[1:len(token)-1]
                token = re.sub('\*', '-', token)
                if has_bold:
                    token = '*' + token + '*'
            return token
        sentence = re.sub('[\S]*[\*]+[\S]*', process, sentence)
        return sentence
    
    def remove_puncts(self, sentence):
        puncts = "+=\\^`|~"
        sentence = re.sub('((sh^t)|(f^ck)|(f^^k)|(bi^ch)|(s^^t))', lambda x: x.group(0).replace('^', '-'), sentence)
        sentence = re.sub('[\\' + '\\'.join(puncts) + ']', ' ', sentence)
        return sentence
    
    def compress_puncts(self, sentence):
        puncts = '[\\' + '\\'.join(punctuation.replace("'", '').replace('"', '')) + ']'
        pattern = re.compile(puncts + '[\s]+' + puncts)
        sentence = pattern.sub(lambda x: re.sub('[\s]+', '', x.group(0)), sentence)
        sentence = pattern.sub(lambda x: re.sub('[\s]+', '', x.group(0)), sentence)
        pattern = re.compile(puncts + '+')
        sentence = pattern.sub(lambda x: ''.join(sorted(x.group(0))), sentence)
        return sentence
    
    def token_counts(self, sentence):
        '''
        Count the number of unique tokens of the regex form [\w'].
        - Note that we use token instead of word because \w has _ in it.
        '''
        tokens = {}
        for match in re.finditer(r'[\w\']+', sentence):
            token = match.group(0)
            if token[0] == '\'':
                token = token[1:]
            if len(token) > 0 and token[-1] == '\'':
                token = token[:-1]
            if len(token) > 0:
                tokens[token] = tokens.get(token, 0) + 1
        unique_count = len(tokens.keys())
        total_count = sum(tokens.values())
        return {
            'unique_tokens': unique_count,
            'tokens': total_count
        }
    
    def numbers(self, sentence):
        sentence = re.sub('[0-9]+', ' @NUMBER ', sentence)
        return sentence
    
    def process_punctuations(self, sentence, sentence_length):
        '''
        Process the punctuations to clean up a lot of dirty internet spams.
        '''
        count_puncts = '!?*/^+'
        puncts_stats = {}
        for punct in count_puncts:
            puncts_stats[punct] = 0
        for match in re.finditer('[\\' + '\\'.join(count_puncts) + ']', sentence):
            punct = match.group(0)
            puncts_stats[punct] += 1
        for punct in count_puncts:
            puncts_stats[punct + '_vs_count'] = puncts_stats[punct] / sentence_length
        sentence = self.closing_puncts(sentence)
        sentence = self.exclamations_questions(sentence)
        sentence = self.censored(sentence)
        sentence = re.sub('(^|[\s])#([\s]|$)', lambda x: x.group(0).replace('#', 'number'), sentence)
        sentence = re.sub('(^|[\s])\$[\S]*', ' $ @NUMBER ', sentence)
        sentence = re.sub('[\S]*%', ' @NUMBER % ', sentence)
        sentence = re.sub('[0-9 ]+/[0-9 ]+', ' @NUMBER ', sentence)
        sentence = re.sub('(^|[\s])&([\s,]|$)', lambda x: x.group(0).replace('&', 'and'), sentence)
        sentence = re.sub('&', 'n', sentence)
        sentence = self.asterisks(sentence)
        sentence = self.remove_puncts(sentence)
        sentence = self.compress_puncts(sentence)
        return sentence, puncts_stats
    
    def common(self, sentence):
        sentence = re.sub(self.stopwords, ' ', sentence)
        return sentence
    
    def stem(self, sentence):
        sentence = re.sub('[a-z]+', lambda x: self.stemmer.stem(x.group(0)), sentence)
        return sentence
    
    def possessives(self, sentence):
        sentence = re.sub(r"[\w]+'s[\W]+", lambda x: x.group(0).replace("'s", ''), sentence)
        return sentence
    
    def manual(self, sentence):
        words = {
            'trolly': 'troll'
        }
        manual_corrections = re.compile('(%s)' % '|'.join(words.keys()))
        sentence = manual_corrections.sub(lambda x: words[x.group(0)], sentence)
        return sentence
    
    def get_preprocess(self, sentence: str) -> (str, dict):
        sentence_length = len(sentence)
        token_count = self.token_counts(sentence)
        sentence, new_lines_count = re.subn('\n', ' ', sentence)
        sentence, url_count = self.urls(sentence)
        sentence, uncommon_count = self.uncommon_chars(sentence, sentence_length)
        sentence, caps_count = self.uppercase(sentence)
        sentence = self.contractions(sentence)
        sentence = self.possessives(sentence)
        sentence, punct_stats = self.process_punctuations(sentence, sentence_length)
        sentence = self.numbers(sentence)
        sentence = self.manual(sentence)
        #sentence = self.common(sentence)
        #sentence = self.stem(sentence)
        sentence = re.sub('[\s]+', ' ', sentence)
        return sentence, {
            'length': sentence_length,
            'newlines': new_lines_count,
            **token_count,
            **url_count,
            **uncommon_count,
            **caps_count,
            **punct_stats
        }
    
    def process_token(self, token):
        token = re.sub('&', 'n', token) # The new & symbol since casual tokenizer doesn't keep it together
        if not (re.search('[a-zA-Z]', token) is None):
            token = re.sub('\*', '-', token) # The new * censor since casual tokenizer doesn't keep it together
        token = self.compress_puncts(token)
        return token


# In[ ]:


# Simple tokenization using NLTK tokenize casual
class SimpleTweetTokenizer(BaseTokenizer):
    def __init__(self):
        self.tokenizer = nltk.tokenize.casual.TweetTokenizer(
            preserve_case=True,
            reduce_len=True,
            strip_handles=False
        )
    
    def get_tokens(self, sentence: str) -> List[str]:
        tokens = self.tokenizer.tokenize(sentence)
        return tokens

    def tokens_parallelize(self, part):
        return [self.get_tokens(sent) for sent in part]
    
    def get_batch_tokens(self, text: List[str]) -> List[List[str]]:
        '''
        part_size = len(text) // partitions
        parts = [text[i:i + part_size] for i in range(0, len(text), part_size)]
        with mp.Pool(processes=cores) as pool:
            pooled = pool.map(self.tokens_parallelize, parts)
            batch_tokens = list(itertools.chain.from_iterable(pooled))
            return batch_tokens
        '''
        return [self.get_tokens(sent) for sent in text]
    
    def get_segmentized_tokens(self, sentence: str, size: int) -> List[List[str]]:
        tokens = self.tokenizer.tokenize(sentence)
        segmentized_tokens = []
        loc = len(tokens) - size
        while loc >= 0:
            segmentized_tokens.append(tokens[loc:loc + size])
            loc -= size
        if loc > -size:
            segmentized_tokens.append(tokens[:loc + size])
        segmentized_tokens.reverse()
        return segmentized_tokens


# In[ ]:


class GloveFastTextEmbedding(BaseEmbedding):
    def __init__(self, preprocessor, unk_freq=None):
        '''
        @param unk_freq - change a word to token <UNK> if its frequency in data
                          is less than unk_freq times. Use None or 0 if want all tokens.
        '''
        self.vocab = {}
        self.preprocessor = preprocessor
        if not unk_freq:
            unk_freq = 0
        self.unk_freq = unk_freq
        self.finalized = False
    
    def __len__(self):
        return len(self.vocab)
    
    def _load_from_pickle(self):
        with open('embedding_vocab_weights.pickle', 'rb') as f:
            self.vocab = pickle.load(f)
        with open('embedding_vocab.pickle', 'rb') as f:
            self.indexer = pickle.load(f)
        self.finalized = True
        print(f"Loaded {len(self.vocab)} embeddings.")
        
    def _parse(self, line):
        '''
        @param line - a line of text from glove or fasttext pretrained embedding file
        @return (token, np.ndarray) - the token of the line and its corresponding embedding weights
        '''
        mapping = line.strip().split(" ")
        token = mapping[0]
        weights = np.array(mapping[1:]).astype(np.float32)
        return (token, weights)
    
    def _add_token(self, token, weight_type, weights=None):
        if weight_type != "glove" and weight_type != "fasttext" and weight_type != "torch":
            raise ValueError("weight_type should be one of the three values: glove, fasttext, or torch")
        if weights is None and (weight_type == "glove" or weight_type == "fasttext"):
            raise ValueError("weights should not be None for a glove or fasttext embedding")
        if not token in self.vocab:
            self.vocab[token] = {
                "count": 0,
                "glove": None,
                "fasttext": None
            }
        if weight_type != "torch":
            self.vocab[token][weight_type] = weights
        self.vocab[token]["count"] += 1
        
    '''
    Add all the tokens from train and test first, then add existing
    pretrained word embeddings from GLoVe and FastText to reduce
    the amount of space used.
    
    Procedure:
    add_token (for all vocabulary words)
    build_vocab
    finalize
    '''

    def add_token(self, token):
        if self.finalized:
            raise RuntimeError("Vocabulary has been finalized, no new token may be added")
        if self.unk_freq > 0 and token == '@UNK':
            raise ValueError("Detected \"@UNK\" token in the dataset")
        if token == '@START':
            raise ValueError('Detected "@START" token in the dataset')
        if token == '@PAD':
            raise ValueError('Detected "@PAD" token in the dataset')
        if token == '@END':
            raise ValueError('Detected "@END" token in the dataset')
        self._add_token(token, weight_type="torch")

    def build_vocab(self):
        echo("Adding FastText vocabulary...")
        with open('../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec') as f:
            tokens_count = int(f.readline().split()[0])
            actual_count = 0
            for i in range(tokens_count):
                line = f.readline()
                token, weights = self._parse(line)
                processed_token = self.preprocessor.process_token(token)
                if token in self.vocab:
                    if self.vocab[token]['fasttext'] is None or token == processed_token:
                        self._add_token(token, weight_type='fasttext', weights=weights)
                        actual_count += 1
            echo(f"Added {actual_count} tokens from FastText")
        echo("Adding GLoVe vocabulary...")
        with open('../input/glove840b300dtxt/glove.840B.300d.txt') as f:
            tokens_count = 0
            for line in f:
                token, weights = self._parse(line)
                processed_token = self.preprocessor.process_token(token)
                if token in self.vocab:
                    if self.vocab[token]['glove'] is None or token == processed_token:
                        self._add_token(token, weight_type='glove', weights=weights)
                        tokens_count += 1
            echo(f"Added {tokens_count} tokens from GLoVe")
    
    def finalize(self):
        unk_tokens = []
        self.no_embed_tokens = []
        for token in self.vocab:
            if self.vocab[token]['glove'] is None                     and self.vocab[token]['fasttext'] is None:
                if self.vocab[token]['count'] >= self.unk_freq:
                    self.no_embed_tokens.append(token)
                else:
                    unk_tokens.append(token)
        self._add_token('@UNK', weight_type="torch")
        self._add_token('@START', weight_type='torch')
        self._add_token('@PAD', weight_type='torch')
        self._add_token('@END', weight_type='torch')
        for token in unk_tokens:
            self.vocab["@UNK"]["count"] += self.vocab[token]["count"]
            self.vocab.pop(token)
        
        self.indexer = []
        for token in self.vocab:
            self.vocab[token]["index"] = torch.LongTensor([len(self.indexer)])
            self.indexer.append(token)
        '''
        self.torch_embedding_layer = nn.Embedding(
            num_embeddings=len(self),
            embedding_dim=100
        )
        self.torch_embedding_layer.weight.requires_grad = False
        '''
        
        echo(f"Finalizing {len(self.vocab)} tokens for vocabulary")
        for token in self.vocab:
            token_dict = self.vocab[token]
            token_glove = token_dict['glove']
            token_fasttest = token_dict['fasttext']
            #token_norm = self.torch_embedding_layer(token_dict['index']).view(-1).data.numpy().astype(np.float32)
            self.vocab[token]["embedding"] = np.concatenate([
                (token_glove if token_glove is not None else np.zeros(300, dtype=np.float32)),
                (token_fasttest if token_fasttest is not None else np.zeros(300, dtype=np.float32))
            ])
            token_dict.pop('glove')
            token_dict.pop('fasttext')
        self.finalized = True
        echo(f"Finalized {len(self.vocab)} tokens for vocabulary")
        echo(f"{len(self.no_embed_tokens)} tokens do not have pretrained embeddings.")
    
    def get_token_embedding(self, token) -> (np.ndarray, torch.LongTensor):
        if not self.finalized:
            raise RuntimeError("Vocabulary must be finalized to get its embeddings")
        if not token in self.vocab:
            token = '@UNK'
        token_embedding = self.vocab[token]['embedding']
        token_index = self.vocab[token]['index']
        return token_embedding, token_index
    
    def get_embedding(self, tokens: List[str], size=185, pad=True) -> (np.ndarray, torch.Tensor):
        '''
        Note: resulting size of the embeddings list will be size + 2 due to appended @START and @END tokens
        '''
        if len(tokens) > size:
            # raise ValueError(f"Given more than {size} tokens when the amount of tokens were expected to be less than {size}")
            tokens = tokens[:size]
        start_embedding, start_idx = self.get_token_embedding('@START')
        start_embedding = np.expand_dims(start_embedding, axis=0)
        pad_embedding, pad_idx = self.get_token_embedding('@PAD')
        pad_embeddings = np.array([pad_embedding.copy() for i in range(size - len(tokens))])
        pad_idxs = torch.LongTensor([pad_idx.clone() for i in range(size - len(tokens))])
        end_embedding, end_idx = self.get_token_embedding('@END')
        end_embedding = np.expand_dims(end_embedding, axis=0)
        if pad_embeddings.size > 0:
            start_embedding = np.concatenate([start_embedding, pad_embeddings])
            start_idx = torch.cat([start_idx, pad_idxs])
        if len(tokens) == 0:
            all_embeddings = np.concatenate([start_embedding, end_embedding])
            all_idxs = torch.cat([start_idx, end_idx])
            return all_embeddings, all_idxs
        embeddings, embeddings_idxs = zip(*(self.get_token_embedding(token) for token in tokens))
        embeddings = np.array(list(embeddings))
        embeddings_idxs = torch.LongTensor(list(embeddings_idxs))
        all_embeddings = np.concatenate([start_embedding, embeddings, end_embedding])
        all_idxs = torch.cat([start_idx, embeddings_idxs, end_idx])
        return all_embeddings, all_idxs
    
    def embeddings_parallelize(self, part, args):
        size, pad = args
        return np.array([self.get_embedding(tokens, size, pad) for tokens in part])
    
    def get_batch_embedding(self, text_tokens: List[List[str]], size=185, pad=True) -> (np.ndarray, torch.Tensor):
        '''
        part_size = len(text_tokens) // partitions
        parts = [text_tokens[i:i + part_size] for i in range(0, len(text_tokens), part_size)]
        with mp.Pool(processes=cores) as pool:
            pooled = pool.starmap(self.embeddings_parallelize, zip(parts, itertools.repeat((size, pad))))
            embeddings = np.concatenate(pooled)
            return embeddings
        '''
        better_size = min(max([len(tokens) for tokens in text_tokens]), size)
        batch_embeddings, batch_idxs = zip(*(self.get_embedding(tokens, better_size, pad) for tokens in text_tokens))
        return np.stack(list(batch_embeddings)), torch.stack(list(batch_idxs))


# In[ ]:


#import types
#embedding.get_embedding = types.MethodType(GloveFastTextEmbedding.get_embedding, embedding)
#embedding.embeddings_parallelize = types.MethodType(GloveFastTextEmbedding.embeddings_parallelize, embedding)


# In[ ]:


class ToxicBroccoliModel(nn.Module, BaseModel):
    def __init__(self, embeddings_size, stats_features_dim):
        super(ToxicBroccoliModel, self).__init__()
    
        # Add trainable embeddings
        self.embed = nn.Embedding(embeddings_size, 100)
    
        # Add entire channel dropout for embeddings
        self.dropout = nn.Dropout2d(p=0.2)
        
        feature_hidden_size = 64
        
        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=700,
            hidden_size=feature_hidden_size * 2,
            bidirectional=True,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=feature_hidden_size * 4,
            hidden_size=feature_hidden_size,
            bidirectional=True,
            batch_first=True
        )
        
        # Simple Convolution of the first Bi-LSTM layer
        conv_channels = 64
        self.conv = nn.Conv1d(
            in_channels=feature_hidden_size * 4,
            out_channels=conv_channels,
            kernel_size=5
        )
        self.convmaxpool = nn.MaxPool1d(kernel_size=187) # Sentence length of 200 with start and end tags
        self.convavgpool = nn.AvgPool1d(kernel_size=187) # Same as above
        
        # Standard Attention Layer
        # As found on https://www.aclweb.org/anthology/S18-1040
        self.attention = self.Attention(feature_hidden_size * 4)
        
        # Max pool and average tool for fine tuning
        # As found on https://arxiv.org/pdf/1801.06146.pdf, section 3.3
        self.maxpool = nn.MaxPool1d(kernel_size=187) # Sentence length of 200 with start and end tags
        self.avgpool = nn.AvgPool1d(kernel_size=187) # Same as above
        
        # Dense/Fully-connected/Feed-forward layer for statistical features
        statistical_features = stats_features_dim
        statistical_features_output = 64
        self.sff = nn.Linear(
            in_features=statistical_features,
            out_features=statistical_features_output,
            bias=False
        )
        nn.init.xavier_uniform_(self.sff.weight)
        self.sfbn = nn.BatchNorm1d(
            num_features=statistical_features_output
        )
        self.sfact = nn.Tanh()
        
        # Dense/Fully-connected/Feed-forward layer
        firstout = 128
        lastout = 1
        self.ff1 = nn.Linear(
            #in_features=feature_hidden_size * 2,
            #in_features=feature_hidden_size * 2 + statistical_features_output,
            #in_features=feature_hidden_size * 2 + conv_channels * 2, # If using CNN in combination with LSTM
            #in_features=feature_hidden_size * 6, # Both attention and lstm output, or max + avg pool + lstm output
            in_features=feature_hidden_size * 10 + conv_channels * 2 + statistical_features_output, # Everything
            out_features=firstout,
            bias=False
        )
        self.ff1act = nn.PReLU()
        self.ffdropout = nn.Dropout(p=0.5)
        self.ffbn = nn.BatchNorm1d(
            num_features=firstout
        )
        self.ff2 = nn.Linear(
            in_features=firstout,
            out_features=lastout
        )
        nn.init.xavier_uniform_(self.ff2.weight)
        self.act = nn.Sigmoid()
        
        # Custom differentiable ROCAUC Loss
        # self.loss = self.ROCAUC()
        
        # BCELoss
        self.loss = self.BCELoss()
        
        # Optimizer
        self.optim = optim.Adam(
            params=self.parameters(),
            lr=1e-3
        )
        
    def Attention(self, hidden_size):
        self.attention_weights = nn.Linear(
            in_features=hidden_size,
            out_features=1
        )
        nn.init.xavier_uniform_(self.attention_weights.weight)
        self.attention_softmax = nn.Softmax(dim=1)
        
        # hidden_states - [batch_size, sequence_length, hidden_size * 2] for Bi-LSTM
        def attention(hidden_states, get_layers=False):
            e = self.attention_weights(hidden_states) # [batch_size, sequence_length, 1]
            w = self.attention_softmax(e) # [batch_size, sequence_length, 1]
            r = torch.squeeze(torch.matmul(torch.unsqueeze(w, dim=2), torch.unsqueeze(hidden_states, dim=2)), dim=2) # [batch_size, sequence_length, hidden_size * 2]
            r = torch.transpose(r, 1, 2) # [batch_size, hidden_size * 2, sequence_length]
            s = torch.sum(r, dim=2) # [batch_size, hidden_size]
            if get_layers:
                return s, r
            return s, None
        return attention
    
    def ROCAUC(self, gamma=0.2, p=3):
        '''
        Differentiable ROCAUC as a loss function (minimize the loss)
        Copied from tflearn.objectives.roc_auc_score
        
        Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
        Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
        Refer to equations (7) and (8)
        '''
        def loss(pred: torch.Tensor, gold: torch.Tensor):
            gold = gold.byte()
            pos = torch.masked_select(pred, gold)
            neg = torch.masked_select(pred, ~gold)
            
            pos = torch.unsqueeze(pos, 0)
            neg = torch.unsqueeze(neg, 1)
            
            diff = torch.zeros((pos * neg).shape).cuda() + pos - neg - gamma
            diff = diff[diff < 0]
            
            return torch.sum(torch.pow(torch.neg(diff), p))
        return loss
    
    def BCELoss(self):
        loss_fn = nn.BCELoss()
        def loss(pred: torch.Tensor, gold: torch.Tensor):
            gold = gold.float()
            return loss_fn(pred, gold)
        return loss

    def spatial_dropout(self, x):
        v = torch.transpose(x, 1, 2)
        v = torch.unsqueeze(v, dim=2)
        v = self.dropout(v)
        v = torch.squeeze(v, dim=2)
        v = torch.transpose(v, 1, 2)
        return v
    
    def sf_layer(self, sf, get_layers=False):
        sfo = self.sff(sf)
        #sfo = self.sfbn(sfo)
        sf = self.sfact(sfo)
        if get_layers:
            return sf, sfo
        return sf, None
    
    def forward(self, x, idxs, sf, get_layers=False):
        '''
        @param x - features of dimension [batch_size, tokens_length, embedding_size]
        @param idxs - indices for tokens in the sentence to get their trainable embeddings [batch_size, tokens_length, 1]
        @param sf - statistical features of dimension [batch_size, statistical_features]
        @return prediction - probability of positive target
        '''
        emb = self.embed(idxs) # [batch_size, sequence_length, trainable_embedding_size]
        x = torch.cat([x, emb], 2) # [batch_size, sequence_length, embedding_size]
        
        v = self.spatial_dropout(x) # [batch_size, sequence_length, embedding_size]
        v1, _ = self.lstm(v) # v1 - [batch_size, sequence_length, 2 * hidden_size]
        
        v2, (hn2, cn2) = self.lstm2(v1) # v2 - [batch_size, sequence_length, 4 * hidden_size], hn - [2, batch_size, 2 * hidden_size]
        hnf = torch.transpose(hn2, 0, 1).flatten(start_dim=1) # [batch_size, 4 * hidden_size]
        sfo, sflayer = self.sf_layer(sf, get_layers) # [batch_size, statistical_features_output]
        
        # full = torch.cat([hnf, sfo], dim=1) # [batch_size, 2 * hidden_sizes + statistical_features_output]
        
        conv = self.conv(torch.transpose(v1, 1, 2))
        cmaxv = torch.squeeze(self.convmaxpool(conv), dim=2)
        cavgv = torch.squeeze(self.convavgpool(conv), dim=2)
        # full = torch.cat([hnf, cmaxv, cavgv], dim=1) # [batch_size, 2 * hidden_sizes + 2 * conv_channels]
        
        attn, attn_layer = self.attention(v1, get_layers) # [batch_size, 4 * hidden_size]
        # full = torch.cat([hnf, attn], dim=1) # [batch_size, 6 * hidden_size]
        
        maxv = torch.squeeze(self.maxpool(torch.transpose(v2, 1, 2)), dim=2) # [batch_size, 2 * hidden_state]
        avgv = torch.squeeze(self.avgpool(torch.transpose(v2, 1, 2)), dim=2) # [batch_size, 2 * hidden_state]
        # full = torch.cat([hnf, maxv, avgv], dim=1) # [batch_size, 6 * hidden_size]
        
        full = torch.cat([hnf, cmaxv, cavgv, attn, maxv, avgv, sfo], dim=1) # [batch_size, 10 * hidden_size + 2 * conv_channels + statistical_features_output]
        
        ffo1 = self.ff1(full) # [batch_size, 128]
        ffo1 = self.ffdropout(ffo1) # [batch_size, 128]
        ffo1 = self.ffbn(ffo1) # [batch_size, 128]
        ffo2 = self.ff2(ffo1) # [batch_size, 1]
        p = self.act(ffo2) # [batch_size, 1]
        p = torch.squeeze(p.transpose(0, 1)) # [batch_size]
        
        if get_layers:
            layers = []
            layers.append(x)
            layers.append(v1)
            layers.append(v2)
            layers.append(conv)
            layers.append(cmaxv)
            layers.append(cavgv)
            layers.append(attn_layer)
            layers.append(maxv)
            layers.append(avgv)
            layers.append(sflayer)
            layers.append(ffo1)
            return p, layers
        return p, None
    
    def _predict(self, embed: np.ndarray, idxs: torch.Tensor, stats_feat: np.ndarray, get_layers=False) -> torch.Tensor:
        self.zero_grad()
        x = torch.unsqueeze(torch.FloatTensor(embed), 0).cuda()
        sf = torch.unsqueeze(torch.FloatTensor(stats_feat), 0).cuda()
        idxs = torch.unsqueeze(idxs, 0).cuda()
        p, layers = self.forward(x, idxs, sf, get_layers)
        if get_layers:
            for idx in range(len(layers)):
                layers[idx] = torch.squeeze(layers[idx], 0)
            return p, layers
        return p
    
    def _get_batch_predict(self, batch_embed: np.ndarray, batch_idxs: torch.Tensor, batch_stats_feat: np.ndarray) -> torch.Tensor:
        self.zero_grad()
        x = torch.FloatTensor(batch_embed).cuda()
        sf = torch.FloatTensor(batch_stats_feat).cuda()
        idxs = batch_idxs.cuda()
        p, _ = self.forward(x, idxs, sf)
        return p
        
    def predict(self, embed: np.ndarray, idxs: torch.Tensor, stats_feat: np.ndarray, get_layers=False) -> np.float32:
        p, layers = self._predict(embed, idxs, stats_feat, get_layers)
        p = p.cpu().detach().numpy()
        if get_layers:
            for idx in range(len(layers)):
                layers[idx] = layers[idx].cpu().detach().numpy()
            return p, layers
        return p
    
    def get_batch_predict(self, batch_embed: np.ndarray, batch_idxs: torch.Tensor, batch_stats_feat: np.ndarray) -> np.ndarray:
        p = self._get_batch_predict(batch_embed, batch_idxs, batch_stats_feat)
        p = p.cpu().detach().numpy()
        return p
    
    def train_model(self, batch: np.ndarray, batch_idxs: torch.Tensor, batch_stats_feat, gold: np.ndarray) -> np.float32:
        self.train()
        p = self._get_batch_predict(batch, batch_idxs, batch_stats_feat)
        gold = torch.tensor(gold).cuda()
        loss = self.loss(p, gold).cuda()
        loss.backward()
        self.optim.step()
        return loss.float().cpu().detach().numpy()


# In[ ]:


#import types
#model.forward = types.MethodType(ToxicBroccoliModel.forward, model)
#embedding.embeddings_parallelize = types.MethodType(GloveFastTextEmbedding.embeddings_parallelize, embedding)


# Data Preprocessing

# In[ ]:


train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')


# In[ ]:


train['label'] = train['target'].apply(lambda x: 1 if x >= 0.5 else 0)


# In[ ]:


preprocessed_values = []
# Parallelize this
@iterate_counter
def preprocess_row(row):
    preprocessor = ToxicBroccoliPreprocess()
    text, values = preprocessor.get_preprocess(row['comment_text'])
    preprocessed_values.append({'id': row['id'], **values})
    return text


# In[ ]:


get_ipython().run_cell_magic('time', '', 'echo("Preprocessing training data...")\ntrain[\'processed_text\'] = train.apply(preprocess_row, axis=1)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'echo("Preprocessing testing data...")\ntest[\'processed_text\'] = test.apply(preprocess_row, axis=1)')


# In[ ]:


preprocessor = ToxicBroccoliPreprocess()


# In[ ]:


preprocessed_values = pd.DataFrame(preprocessed_values, dtype=np.float32).set_index('id')
preprocessed_values.index = preprocessed_values.index.astype('int64')


# In[ ]:


preprocessed_values.to_csv('preprocessed.csv')


# In[ ]:


stats_features_dim = preprocessed_values.shape[1]
echo(f"Number of statistical features: {stats_features_dim}")


# Word Embeddings

# In[ ]:


tokenizer = SimpleTweetTokenizer()


# In[ ]:


embedding = GloveFastTextEmbedding(ToxicBroccoliPreprocess(), 5)


# In[ ]:


possible_token_lengths = []


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for idx, row in train.iterrows():\n    comment = row[\'processed_text\']\n    tokens = tokenizer.get_tokens(comment)\n    if len(tokens) == 0:\n        echo(f"Found empty sentence: {comment}")\n        echo(f"Original sentence: {row[\'comment_text\']}")\n    possible_token_lengths.append(len(tokens))\n    for token in tokens:\n        embedding.add_token(token)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'for idx, row in test.iterrows():\n    comment = row[\'processed_text\']\n    tokens = tokenizer.get_tokens(comment)\n    if len(tokens) == 0:\n        echo(f"Found empty sentence: {comment}")\n        echo(f"Original sentence: {row[\'comment_text\']}")\n    possible_token_lengths.append(len(tokens))\n    for token in tokens:\n        embedding.add_token(token)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'embedding.build_vocab()')


# In[ ]:


word_occurrences = [v['count'] for v in embedding.vocab.values()]


# In[ ]:


np.percentile(word_occurrences, [1, 5, 40, 44, 45, 55, 75])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'embedding.finalize()')


# In[ ]:


echo(f"Has no pretrained embeddings: {embedding.no_embed_tokens}")


# In[ ]:


np.percentile(possible_token_lengths, [50, 90, 95, 99, 100])


# In[ ]:


echo(f"Percent of coverage: {1 - len(embedding.no_embed_tokens) / len(embedding)}")


# Model Validation

# In[ ]:


pos_train = train[train['label'] == 1]
neg_train = train[train['label'] == 0]


# Validation ROC AUC (thanks to Google's Benchmark model at https://www.kaggle.com/dborkan/benchmark-kernel/notebook)

# In[ ]:


SUBGROUP_AUC = 'subgroup_auc'
BPSN_AUC = 'bpsn_auc'  # stands for background positive, subgroup negative
BNSP_AUC = 'bnsp_auc'  # stands for background negative, subgroup positive
MODEL_NAME = 'prediction'
TOXICITY_COLUMN = 'label'

def compute_auc(y_true, y_pred):
    try:
        return metrics.roc_auc_score(y_true, y_pred)
    except ValueError:
        return np.nan

def compute_subgroup_auc(df, subgroup, label, model_name):
    subgroup_examples = df[df[subgroup]]
    return compute_auc(subgroup_examples[label], subgroup_examples[model_name])

def compute_bpsn_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
    subgroup_negative_examples = df[df[subgroup] & ~df[label]]
    non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
    examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
    return compute_auc(examples[label], examples[model_name])

def compute_bnsp_auc(df, subgroup, label, model_name):
    """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
    subgroup_positive_examples = df[df[subgroup] & df[label]]
    non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
    examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
    return compute_auc(examples[label], examples[model_name])

def compute_bias_metrics_for_model(dataset,
                                   subgroups,
                                   model,
                                   label_col,
                                   include_asegs=False):
    """Computes per-subgroup metrics for all subgroups and one model."""
    records = []
    for subgroup in subgroups:
        record = {
            'subgroup': subgroup,
            'subgroup_size': len(dataset[dataset[subgroup]])
        }
        record[SUBGROUP_AUC] = compute_subgroup_auc(dataset, subgroup, label_col, model)
        record[BPSN_AUC] = compute_bpsn_auc(dataset, subgroup, label_col, model)
        record[BNSP_AUC] = compute_bnsp_auc(dataset, subgroup, label_col, model)
        records.append(record)
    return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


# In[ ]:


def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

def power_mean(series, p):
    total = sum(np.power(series, p))
    return np.power(total / len(series), 1 / p)

def get_final_metric(bias_df, overall_auc, POWER=-5, OVERALL_MODEL_WEIGHT=0.25):
    bias_score = np.average([
        power_mean(bias_df[SUBGROUP_AUC], POWER),
        power_mean(bias_df[BPSN_AUC], POWER),
        power_mean(bias_df[BNSP_AUC], POWER)
    ])
    return (OVERALL_MODEL_WEIGHT * overall_auc) + ((1 - OVERALL_MODEL_WEIGHT) * bias_score)


# In[ ]:


def compute_metrics(validate_df):
    # List all identities
    identity_columns = [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
    for col in identity_columns:
        validate_df[col] = np.where(validate_df[col] >= 0.5, True, False)
    
    bias_metrics_df = compute_bias_metrics_for_model(validate_df, identity_columns, MODEL_NAME, TOXICITY_COLUMN)
    display(HTML(bias_metrics_df.to_html()))
    echo(f"Current final ROCAUC: {get_final_metric(bias_metrics_df, calculate_overall_auc(validate_df, MODEL_NAME))}")


# In[ ]:


def predict_model(model, df, get_loss=False):
    model.eval()
    batch_size = 512
    preds = []
    loss = 0
    echo(f"Predicting...")
    for i in range(0, df.shape[0], batch_size):
        batch = df.iloc[i:i + batch_size]
        
        '''
        batch_indices = {}
        batch_tokens = []
        for index, row in batch.iterrows():
            segmented_tokens = tokenizer.get_segmentized_tokens(row['processed_text'], 65)
            batch_indices[index] = [len(batch_indices) + j for j in range(len(segmented_tokens))]
            batch_tokens.extend(segmented_tokens)
        batch_embeddings = embedding.get_batch_embedding(batch_tokens, size=65)
        '''

        batch_tokens = tokenizer.get_batch_tokens(batch['processed_text'].values)
        batch_embeddings, batch_idxs = embedding.get_batch_embedding(batch_tokens)
        batch_statistical_features = preprocessed_values.loc[batch['id'].values].values

        p = model.get_batch_predict(batch_embeddings, batch_idxs, batch_statistical_features)
        if get_loss:
            if not 'label' in batch.columns:
                raise RuntimeError("Cannot get loss without 'label' gold column")
            loss += model.loss(torch.tensor(p).cuda(), torch.tensor(batch['label'].values).cuda()).cpu().detach().numpy()
        #for index, row in batch.iterrows():
        for j in range(batch.shape[0]):
            '''
            p_row = p[batch_indices[index]]
            p_row = np.sum(p_row) / p_row.shape[0]
            '''
            preds.append({
                'id': batch.iloc[j]['id'], #row['id'],
                'prediction': p[j] # p_row
            })
        if i % 40960 == 0 and i > 0:
            echo(f"Predicted {i} samples...")
    preds = pd.DataFrame(preds).set_index("id")
    return preds, loss


# In[ ]:


def eval_model(model, pos, neg):
    eval_df = pd.concat([pos, neg])
    eval_df = eval_df.reindex(np.random.permutation(eval_df.index))
    
    pred_df, loss = predict_model(model, eval_df, get_loss=True)
    if pred_df.shape[0] != eval_df.shape[0]:
        raise RuntimeError('eval_df and pred_df has different number of rows, indicating data has been lost')
    df = eval_df.join(pred_df, on="id", sort=True)
    
    true_pos = 0
    true_neg = 0
    false_neg = 0
    false_pos = 0
    for index, row in df.iterrows():
        if row['label'] == 1:
            if row['prediction'] >= 0.5:
                true_pos += 1
            else:
                false_neg += 1
        else:
            if row['prediction'] >= 0.5:
                false_pos += 1
            else:
                true_neg += 1
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    accuracy = (true_pos + true_neg) / df.shape[0]
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    compute_metrics(df)
    echo(f"--- Evaluation on {eval_df.shape[0]} samples ---")
    echo(f"Accuracy: {accuracy}")
    echo(f"Precision: {precision}")
    echo(f"Recall: {recall}")
    echo(f"F1: {f1}")
    echo(f"Loss: {loss}")
    echo("----------------------------------")
    return (accuracy, f1, loss, df)


# In[ ]:


def train_model(model, pos, neg, eval_per=None, eval_pos=None, eval_neg=None):
    batch_size = 512
    df = pd.concat([neg, pos])
    df = df.reindex(np.random.permutation(df.index))
    total_loss = 0
    prev_loss = 0
    for i in range(0, df.shape[0], batch_size):
        batch = df.iloc[i:i + batch_size]
        
        '''
        gold = []
        batch_tokens = []
        for index, row in batch.iterrows():
            segmentized_tokens = tokenizer.get_segmentized_tokens(row['processed_text'], 65)
            gold.extend([row['label'] for k in range(len(segmentized_tokens))])
            batch_tokens.extend(segmentized_tokens)
            
        gold = np.asarray(gold, dtype=np.float32)
        batch_embeddings = embedding.get_batch_embedding(batch_tokens, size=65)
        loss = model.train(batch_embeddings, gold) / gold.shape[0]
        '''
        
        gold = batch['label'].values
        batch_tokens = tokenizer.get_batch_tokens(batch['processed_text'].values)
        batch_embeddings, batch_idxs = embedding.get_batch_embedding(batch_tokens)
        batch_statistical_features = preprocessed_values.loc[batch['id'].values].values
        loss = model.train_model(batch_embeddings, batch_idxs, batch_statistical_features, gold)

        total_loss += loss
        if i % 163840 == 0 and i > 0:
            echo(f"Loss on {i} samples: {total_loss}")
            echo(f"Change on loss: {total_loss - prev_loss}")
            prev_loss = float(total_loss)
        if eval_per is not None and i % eval_per == 0 and i > 0:
            eval_model(model, eval_pos, eval_neg)
    return total_loss


# In[ ]:


def validate_model(pos, neg):
    pos_train, pos_test = model_selection.train_test_split(pos, test_size=0.2)
    neg_train, neg_test = model_selection.train_test_split(neg, test_size=0.2)
    
    epochs = 2
    model = ToxicBroccoliModel(len(embedding), stats_features_dim).cuda()
    for i in range(epochs):
        echo(f"--- Epoch {i} ---")
        train_model(model, pos_train, neg_train, eval_per=None, eval_pos=pos_test, eval_neg=neg_test)
        accuracy, f1, loss, df = eval_model(model, pos_test, neg_test)
        df.set_index("id").to_csv(f"epoch_{i}_validation_output.csv")
        torch.save(model.state_dict(), f'./toxic-broccoli-lstm-validation-epoch-{i}.pt')
        echo(f"-----------------")
    '''
    from sklearn.model_selection import KFold
    folds = 3
    epochs = 3
    negkf = KFold(n_splits=folds).split(neg)
    poskf = KFold(n_splits=folds).split(pos)
    losses = [[] for i in range(epochs)]
    accuracies = [[] for i in range(epochs)]
    f1s = [[] for i in range(epochs)]
    for f in range(folds):
        echo(f"--- Fold {f} ---")
        pos_train, pos_test = next(poskf)
        neg_train, neg_test = next(negkf)
        pos_train = pos.iloc[pos_train]
        pos_test = pos.iloc[pos_test]
        neg_train = neg.iloc[neg_train]
        neg_test = neg.iloc[neg_test]
        model = ToxicBroccoliModel().cuda()
        for i in range(epochs):
            loss = train_model(model, pos_train, neg_train, eval_per=100000, eval_pos=pos_test.sample, eval_neg=neg_test.sample)
            echo(f"--- Epoch {i} ---")
            accuracy, f1 = eval_model(model, pos_test, neg_test)
            losses[i].append(loss)
            accuracies[i].append(accuracy)
            f1s[i].append(f1)
            echo(f"-----------------")
        echo(f"----------------")
    for i in range(epochs):
        echo(f"Epoch {i} loss: {sum(losses[i]) / folds}")
        echo(f"Epoch {i} accuracy: {sum(accuracies[i]) / folds}")
        echo(f"Epoch {i} f1: {sum(f1s[i]) / folds}")
    return (losses, accuracies, f1s)
    '''


# In[ ]:


get_ipython().run_cell_magic('time', '', 'validate_model(pos_train, neg_train)')


# Model Predictions

# In[ ]:


model = ToxicBroccoliModel(len(embedding), stats_features_dim).cuda()


# In[ ]:


get_ipython().run_cell_magic('time', '', 'epoch = 2\nfor i in range(epoch):\n    echo(f"------ EPOCH {i} ------")\n    train_model(model, pos_train, neg_train)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'traindf, _ = predict_model(model, train)')


# In[ ]:


traindf.to_csv('train_submission.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df, _ = predict_model(model, test)')


# In[ ]:


df.to_csv('submission.csv')


# In[ ]:


df.head()


# In[ ]:


echo("Model's state_dict:")
echo(model)
echo("")
echo("Optimizer's state_dict:")
echo(model.optim)


# In[ ]:


torch.save(model.state_dict(), './toxic-broccoli-lstm-preprocessed-all.pt')
torch.save(model.optim.state_dict(), './toxic-broccoli-lstm-preprocessed-all-optim.pt')
echo(os.listdir("./"))


# In[ ]:


with open('embedding_vocab_weights.pickle', 'wb') as f:
    pickle.dump(embedding.vocab, f)


# In[ ]:


with open('embedding_vocab.pickle', 'wb') as f:
    pickle.dump(embedding.indexer, f)


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
def attn_transfer(example_text):
    model.eval()
    preprocessed_text, pv = preprocessor.get_preprocess(example_text)
    example_tokens = tokenizer.get_tokens(preprocessed_text)
    example_embed, example_embed_idx = embedding.get_embedding(example_tokens)
    plt.rcParams['figure.figsize'] = (20, len(example_tokens))
    echo(preprocessed_text)
    echo(pv)
    p, layers = model.predict(example_embed, example_embed_idx, np.array([*pv.values()]), get_layers=True)
    echo('')
    echo("Embedding layer")
    pad_length = min(5, 187 - len(example_tokens) + 1)
    tokens = ["@PAD" for i in range(pad_length - 1)] + example_tokens + ['@END']
    plt.imshow(np.power(layers[0][-len(example_tokens) - pad_length:], 2), interpolation='bicubic', aspect=10)
    plt.yticks([*range(len(tokens))], tokens)
    plt.ylabel("Tokens")
    plt.xlabel("Embedding Features")
    plt.show()
    echo("First LSTM layer")
    plt.imshow(np.power(layers[1][-len(example_tokens) - pad_length:], 2), interpolation='bicubic', aspect=2.5)
    plt.yticks([*range(len(tokens))], tokens)
    plt.ylabel("Tokens")
    plt.xlabel("Hidden State Features")
    plt.show()
    sns.heatmap(layers[1][-len(example_tokens) - pad_length:] * np.abs(layers[1][-len(example_tokens) - pad_length:]),
                vmin=-0.5, vmax=0.5, center=0, yticklabels=tokens)
    plt.ylabel("Tokens")
    plt.xlabel("Hidden State Features")
    plt.show()
    echo("Attention layer summed")
    plt.imshow(np.power(np.expand_dims(np.sum(layers[6], axis=1), 0), 2), interpolation='bicubic', aspect=5)
    plt.ylabel("Totaled Weight")
    plt.xlabel("Hidden State Features")
    plt.show()
    echo("Attention layer")
    plt.imshow(np.power(layers[6].transpose()[-len(example_tokens) - pad_length:], 2), interpolation='bicubic', aspect=5)
    plt.yticks([*range(len(tokens))], tokens)
    plt.ylabel("Tokens")
    plt.xlabel("Hidden State Features")
    plt.show()
    attn_norm = layers[6].transpose()[-len(example_tokens) - pad_length:] * np.abs(layers[6].transpose()[-len(example_tokens) - pad_length:])
    sns.heatmap(attn_norm, vmin=np.amin(attn_norm), vmax=np.amax(attn_norm), center=0, yticklabels=tokens)
    plt.ylabel("Tokens")
    plt.xlabel("Hidden State Features")
    plt.show()
    echo("Second LSTM layer")
    plt.imshow(np.power(layers[2][-len(example_tokens) - pad_length:], 2), interpolation='bicubic', aspect=2.5)
    plt.yticks([*range(len(tokens))], tokens)
    plt.ylabel("Tokens")
    plt.xlabel("Hidden State Features")
    plt.show()
    sns.heatmap(layers[2][-len(example_tokens) - pad_length:] * np.abs(layers[2][-len(example_tokens) - pad_length:]),
                vmin=-0.5, vmax=0.5, center=0, yticklabels=tokens)
    plt.ylabel("Tokens")
    plt.xlabel("Hidden State Features")
    plt.show()
    echo("Second LSTM maxpool and avgpool layer")
    plt.imshow(np.power(np.expand_dims(layers[7], 0), 2), interpolation='bicubic', aspect=2.5)
    plt.ylabel("Pooled Max")
    plt.xlabel("Hidden State Features")
    plt.show()
    plt.imshow(np.power(np.expand_dims(layers[8], 0), 2), interpolation='bicubic', aspect=2.5)
    plt.ylabel("Pooled Average")
    plt.xlabel("Hidden State Features")
    plt.show()
    echo("Convolution layer (for the first LSTM)")
    plt.imshow(np.power(layers[3].transpose()[-len(example_tokens) - pad_length:], 2), interpolation='bicubic', aspect=1)
    plt.yticks([*range(len(tokens))], tokens)
    plt.ylabel("Tokens")
    plt.xlabel("Channels")
    plt.show()
    sns.heatmap(layers[3].transpose()[-len(example_tokens) - pad_length:] * np.abs(layers[3].transpose()[-len(example_tokens) - pad_length:]),
                vmin=-5, vmax=5, center=0, yticklabels=tokens)
    plt.ylabel("Tokens")
    plt.xlabel("Channels")
    plt.show()
    echo("Convolution maxpool and avgpool layer")
    plt.imshow(np.power(np.expand_dims(layers[4], 0), 2), interpolation='bicubic', aspect=1)
    plt.ylabel("Pooled Max")
    plt.xlabel("Channels Features")
    plt.show()
    plt.imshow(np.power(np.expand_dims(layers[5], 0), 2), interpolation='bicubic', aspect=1)
    plt.ylabel("Pooled Average")
    plt.xlabel("Channels")
    plt.show()
    echo("Statistical features layer")
    plt.imshow(np.power(np.expand_dims(layers[9], 0), 2), interpolation='bicubic', aspect=1)
    plt.ylabel("Activation Weights")
    plt.xlabel("Hidden Features")
    plt.show()
    echo("Feed forward layer")
    plt.imshow(np.power(np.expand_dims(layers[10], 0), 2), interpolation='bicubic', aspect=2.5)
    plt.ylabel("Activation Weights")
    plt.xlabel("Hidden Features")
    plt.show()
    print(f"Toxicity Probability: {p}")


# The following examples display the attention weights outputs of the model. Warning: the sentences can be considered vulgar, profane, and offensive.

# In[ ]:


# Correct Positive
attn_transfer('i wish i was funny like a black person.')


# In[ ]:


# False Negative
attn_transfer('The Troll is strong with this one.')


# In[ ]:


# False Positive
attn_transfer('Gary:  is it sexist to ask people to vote for you because you are a woman?   Is it racist to ask for special treatment because you are black?')


# In[ ]:


# Correct negative
attn_transfer('Nice! Loved your "How to Win the Lottery" talk at XOXO a few years ago.')


# In[ ]:


# Very borderline examples
attn_transfer('Be nice to see the harassment comments, or just plain hateful ones, go away')


# In[ ]:


attn_transfer('Wouldn\'t be the first criminal elected or re-elected in the US.')

