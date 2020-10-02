#!/usr/bin/env python
# coding: utf-8

# # Language model

# In[ ]:


import string

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# turn a doc into clean tokens
def clean_doc(doc):
	# replace '--' with a space ' '
	doc = doc.replace('--', ' ')
	# split into tokens by white space
	tokens = doc.split()
	# remove punctuation from each token
	table = str.maketrans('', '', string.punctuation)
	tokens = [w.translate(table) for w in tokens]
	# remove remaining tokens that are not alphabetic
	tokens = [word for word in tokens if word.isalpha()]
	# make lower case
	tokens = [word.lower() for word in tokens]
	return tokens

# save tokens to file, one dialog per line
def save_doc(lines, filename):
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# load document
in_filename = '../input/republic_sequences.txt'
doc = load_doc(in_filename)
print(doc[:200])

# clean document
tokens = clean_doc(doc)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))

# organize into sequences of tokens
length = 50 + 1
sequences = list()
for i in range(length, len(tokens)):
	# select sequence of tokens
	seq = tokens[i-length:i]
	# convert into a line
	line = ' '.join(seq)
	# store
	sequences.append(line)
print('Total Sequences: %d' % len(sequences))

# save sequences to file
out_filename = 'republic_seq.txt'
save_doc(sequences, out_filename)



from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# load
in_filename = 'republic_seq.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
lines = lines[:int(len(lines)/2)]

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# define model
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=128, epochs=100)

# save the model to file
model.save('model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))


from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# load doc into memory
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

# generate a sequence from a language model
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
	result = list()
	in_text = seed_text
	# generate a fixed number of words
	for _ in range(n_words):
		# encode the text as integer
		encoded = tokenizer.texts_to_sequences([in_text])[0]
		# truncate sequences to a fixed length
		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		# predict probabilities for each word
		yhat = model.predict_classes(encoded, verbose=0)
		# map predicted word index to word
		out_word = ''
		for word, index in tokenizer.word_index.items():
			if index == yhat:
				out_word = word
				break
		# append to input
		in_text += ' ' + out_word
		result.append(out_word)
	return ' '.join(result)

# load cleaned text sequences
in_filename = 'republic_seq.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')
#lines = lines[:int(len(lines)/5)]

seq_length = len(lines[0].split()) - 1

# load the model
model = load_model('model.h5')

# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))


# # Spell check

# In[1]:


from __future__ import absolute_import, division, unicode_literals

import os
import json
import string
from collections import Counter

""" Additional utility functions """
import sys
import re
import gzip
import contextlib

if sys.version_info < (3, 0):
    import io  # python 2 text file encoding support

    OPEN = io.open  # hijack this
else:
    OPEN = open
    


@contextlib.contextmanager
def load_file(filename, encoding):
    try:
        with gzip.open(filename, mode="rt") as fobj:
            yield fobj.read()
    except (OSError, IOError):
        with OPEN(filename, mode="r", encoding=encoding) as fobj:
            yield fobj.read()


def write_file(filepath, encoding, gzipped, data): 
    if gzipped:
        with gzip.open(filepath, "wt") as fobj:
            fobj.write(data)
    else:
        with OPEN(filepath, "w", encoding=encoding) as fobj:
            if sys.version_info < (3, 0):
                data = data.decode(encoding)
            fobj.write(data)


def _parse_into_words(text):
    # Parse the text into words; currently removes punctuation
    return re.findall(r"\w+", text.lower())








class SpellChecker(object):
    # The SpellChecker class encapsulates the basics needed to accomplish a simple spell checking algorithm.
    __slots__ = ["_distance", "_word_frequency"]

    def __init__(self, language="en", local_dictionary=None, distance=2):
        self._distance = None
        self.distance = distance  # use the setter value check
        self._word_frequency = WordFrequency()
        if local_dictionary:
            self._word_frequency.load_dictionary(local_dictionary)
        elif language:
            filename = "../input/{}3.json".format(language.lower())
            print(filename)
            #here = os.path.dirname(__file__)
            full_filename = filename
            if not os.path.exists(full_filename):
                msg = (
                    "The provided dictionary language ({}) does not " "exist!"
                ).format(language.lower())
                raise ValueError(msg)
            self._word_frequency.load_dictionary(full_filename)

    def __contains__(self, key):
        """ setup easier known checks """
        return key in self._word_frequency

    def __getitem__(self, key):
        """ setup easier frequency checks """
        return self._word_frequency[key]

    @property
    def word_frequency(self):
        # WordFrequency: An encapsulation of the word frequency `dictionary`
        return self._word_frequency

    @property
    def distance(self):
        # int: The maximum edit distance to calculate
        # valid values : 1 or 2; invalid => default value =2
        return self._distance

    @distance.setter
    def distance(self, val):
        # set the distance parameter 
        tmp = 2
        try:
            int(val)
            if val > 0 and val <= 2:
                tmp = val
        except (ValueError, TypeError):
            pass
        self._distance = tmp

    @staticmethod
    def split_words(text):
        # split the text into words
        return _parse_into_words(text)

    def export(self, filepath, encoding="utf-8", gzipped=True):
        # Export the word frequency list for import in the future
        data = json.dumps(self.word_frequency.dictionary, sort_keys=True)
        write_file(filepath, encoding, gzipped, data)

    def word_probability(self, word, total_words=None):
        # Calculate the probability of the `word` being the desired, correct word
        # returns float: The probability that the word is the correct word 
        if total_words is None:
            total_words = self._word_frequency.total_words
        return self._word_frequency.dictionary[word] / total_words

    def correction(self, word):
        # The most probable correct spelling for the word
        return max(self.candidates(word), key=self.word_probability)

    def candidates(self, word):
        # Generate possible spelling corrections for the provided word up to an edit distance of two, if and only when needed
        if self.known([word]):  # short-cut if word is correct already
            return {word}
        # get edit distance 1...
        res = [x for x in self.edit_distance_1(word)]
        tmp = self.known(res)
        if tmp:
            return tmp
        # if still not found, use the edit distance 1 to calc edit distance 2
        if self._distance == 2:
            tmp = self.known([x for x in self.__edit_distance_alt(res)])
            if tmp:
                return tmp
        return {word}

    def known(self, words):
        # The subset of `words` that appear in the dictionary of words
        tmp = [w.lower() for w in words]
        return set(
            w
            for w in tmp
            if w in self._word_frequency.dictionary
            or not self._check_if_should_check(w)
        )

    def unknown(self, words):
        # The subset of `words` that do not appear in the dictionary
        tmp = [w.lower() for w in words if self._check_if_should_check(w)]
        return set(w for w in tmp if w not in self._word_frequency.dictionary)

    def edit_distance_1(self, word):
        # Compute all strings that are one edit away from `word` using only the letters in the corpus
        word = word.lower()
        if self._check_if_should_check(word) is False:
            return {word}
        letters = self._word_frequency.letters
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edit_distance_2(self, word):
        # Compute all strings that are two edits away from `word` using only the letters in the corpus
        word = word.lower()
        return [
            e2 for e1 in self.edit_distance_1(word) for e2 in self.edit_distance_1(e1)
        ]

    def __edit_distance_alt(self, words):
        # Compute all strings that are 1 edits away from all the words using only the letters in the corpus
        words = [x.lower() for x in words]
        return [e2 for e1 in words for e2 in self.edit_distance_1(e1)]

    @staticmethod
    def _check_if_should_check(word):
        one_letter_words=['I','i','a']
        if ( (len(word) == 1) and (word in string.punctuation) and (word not in one_letter_words) ):
            return False
        try:  # check if it is a number (int, float, etc)
            float(word)
            return False
        except ValueError:
            pass

        return True


class WordFrequency(object):
    """ Store the `dictionary` as a word frequency list while allowing for
        different methods to load the data and update over time """

    __slots__ = ["_dictionary", "_total_words", "_unique_words", "_letters"]

    def __init__(self):
        self._dictionary = Counter()
        self._total_words = 0
        self._unique_words = 0
        self._letters = set()

    def __contains__(self, key):
        """ turn on contains """
        return key.lower() in self._dictionary

    def __getitem__(self, key):
        """ turn on getitem """
        return self._dictionary[key.lower()]

    def pop(self, key, default=None):
        # Remove the key and return the associated value or default if not found
        return self._dictionary.pop(key.lower(), default)

    @property
    def dictionary(self):
        # Counter: A counting dictionary of all words in the corpus and the  number of times each has been seen
        return self._dictionary

    @property
    def total_words(self):
        # int: The sum of all word occurances in the word frequency dictionary
        return self._total_words

    @property
    def unique_words(self):
        # int: The total number of unique words in the word frequency list
        return self._unique_words

    @property
    def letters(self):
        # str: The listing of all letters found within the corpus
        return self._letters

    def keys(self):
        # Iterator over the key of the dictionary
        # Yields:  str: The next key in the dictionary
        for key in self._dictionary.keys():
            yield key

    def words(self):
        # Iterator over the words in the dictionary
        # Yields:  str: The next word in the dictionary

        for word in self._dictionary.keys():
            yield word

    def load_dictionary(self, filename, encoding="utf-8"):
        # Load in a pre-built word frequency list
        with load_file(filename, encoding) as data:
            self._dictionary.update(json.loads(data.lower(), encoding=encoding))
            self._update_dictionary()

    def load_text_file(self, filename, encoding="utf-8", tokenizer=None):
        # Load in a text file from which to generate a word frequency list
        with load_file(filename, encoding=encoding) as data:
            self.load_text(data, tokenizer)

    def load_text(self, text, tokenizer=None):
        # Load text from which to generate a word frequency list 
        if tokenizer:
            words = [x.lower() for x in tokenizer(text)]
        else:
            words = _parse_into_words(text)

        self._dictionary.update(words)
        self._update_dictionary()

    def load_words(self, words):
        # Load a list of words from which to generate a word frequency list
        self._dictionary.update([word.lower() for word in words])
        self._update_dictionary()

    def add(self, word):
        # Add a word to the word frequency list 
        self.load_words([word])

    def remove_words(self, words):
        # Remove a list of words from the word frequency list 
        for word in words:
            self._dictionary.pop(word.lower())
        self._update_dictionary()

    def remove(self, word):
        # Remove a word from the word frequency list 
        self._dictionary.pop(word.lower())
        self._update_dictionary()

    def remove_by_threshold(self, threshold=5):
        # Remove all words at, or below, the provided threshold 
        keys = [x for x in self._dictionary.keys()]
        for key in keys:
            if self._dictionary[key] <= threshold:
                self._dictionary.pop(key)
        self._update_dictionary()

    def _update_dictionary(self):
        """ Update the word frequency object """
        self._total_words = sum(self._dictionary.values())
        self._unique_words = len(self._dictionary.keys())
        self._letters = set()
        for key in self._dictionary:
          self._letters.update(key)
          
          
def get_misspelt_index(list,misspelt,string_split):
  #print(misspelt)
  ret_index=[]
  for word in misspelt:
    ret_index.append(string_split.index(word))
  #ret_index.sort()
  #print(ret_index)
  return ret_index

def get_correct_spelling(list,misspelt):
  spell = SpellChecker()
  ret_val=[]
  for word in misspelt:
      ret_val.append(spell.correction(word))
  return ret_val


#Testing 

def test_string(inpstring):
  string_split = inpstring[0].split(' ')
  spell = SpellChecker()
  text = inpstring[0].split(' ')
  l = [words for words in text]
  misspelt = spell.unknown(l)
  
  
  #print(string_split)
  print("Input string :",inpstring[0])
  mis_index = get_misspelt_index(inpstring,misspelt,string_split)
  cor_set=get_correct_spelling(inpstring,misspelt)
  #print("Dfdf",cor_set)
  correct_words=[None] * len(string_split)
  j=0
  for i in cor_set:
    correct_words[mis_index[j]] = i
    j=j+1
  #correct_words = get_correct_spelling(string)
  #print(correct_words)
  i=0
  j=0
  output_list=[]
  for w in string_split:
    if(i not in mis_index):
      output_list.append(w)
    elif j<len(mis_index):
      #print(correct_words[j])
      #print(i)
      output_list.append(correct_words[i])
      j=j+1
    i=i+1
  op_string=""

  for i in output_list:
    op_string+=i+" "
  #print("Output string :",op_string)
  return op_string


# In[2]:


def enter_input(inpstring):
 inpstring = [inpstring]
 op_string = test_string(inpstring)
 print("Correct string :",op_string,"\n\n")

 # select a seed text
 #seed_text = lines[randint(0,len(lines))]
 seed_text = op_string
 print("Seed Text :")
 print(seed_text + '\n')

 # generate new text
 print("Generated Text :")
 generated = generate_seq(model, tokenizer, seq_length, seed_text, 50)
 print(generated)


# In[3]:


enter_input('comfinations in blife and esperyence theref is a muzic of the coul chich awswers to the barmony of the worl and the fairezt ovject of a misycal soul is the fxair mind in the fair body some defxect in the lattter may be excuzed but not in the formir frue bove')

inp = input()
enter_input(inp)


# In[ ]:


from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))
language="en"
filename = "{}.json.gz".format(language.lower())
print(filename)


# In[ ]:




