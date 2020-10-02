# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.

filename = "../input/nlp-beginners/book.txt"
file = open(filename,'rt')
text = file.read()
file.close()

words = text.split()
words

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import *

tokens = word_tokenize(text)
len(tokens)
print(tokens)
########### TEXT CLEANING #############

#removing stop words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
cleaned = []
for w in tokens:
    if w not in stop_words:
        cleaned.append(w)
print(cleaned)
len(tokens)
len(cleaned)
cleaned

#Remove Punctuations
from nltk.tokenize import RegexpTokenizer
import re
import string
string.punctuation

for  i in cleaned:
    if i not in string.punctuation:
        cleaned.append(i)
        
print(cleaned)


#stemming
len(tokens)
porter = nltk.PorterStemmer()
tokens = [porter.stem(t) for t in tokens]
len(tokens)
#lemmatization

wnl = nltk.WordNetLemmatizer()
tokens = [wnl.lemmatize(t) for t in tokens]
len(tokens)
#Segmentation
sents = nltk.sent_tokenize(text)
sents

final = []
for sentenc
e in sents:
    final.append(word_tokenize(sentence))
print (final)


########### BAG-OF-WORDS ###############

#### Preparing Text with MachineLearning using scikit ###

#######CountVectorizer
from sklearn.feature_extraction.text import *

#create transform
vectorizer = CountVectorizer()

#tokenize and build vocab
vectorizer.fit([text])

#summarize
print(vectorizer.vocabulary_)

#encode document
vector = vectorizer.transform([text])

#summarize
print(vector.shape)
print(type(vector))
print(vector.toarray())

########TF_IDF

#create transfom
vectorizer = TfidfVectorizer()

#tokenize and build vocab
vectorizer.fit([text])

#summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)

#encode doc
vector = vectorizer.transform([text])

#summarize encoded vector
print(vector.shape)
print(vector.toarray())

########HashingVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

# No Building Vocabulary here

#create transform
vectorizer = HashingVectorizer(n_features = 20)

#encode documents
vector = vectorizer.transform([text])

#summarize
print(vector.shape)
print(vector.toarray())

###Preparing Text for DeepLearning using Keras###

######Splitting words
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
print(words)
######Encoding with one-hot

result = one_hot(text, round(vocab_size*1.3))
print(result)

####Hash Encoding with hashing_trick
from keras.preprocessing.text import hashing_trick

result = hashing_trick(text, round(vocab_size*1.3), hash_function = 'md5')
print(result)

#######Tokenizer API
from keras.preprocessing.text import Tokenizer

#Create Tokenizer
t = Tokenizer()

#fit tokenizer
t.fit_on_texts([text])

#summarize
print(t.word_counts)
print(t.word_index)
print(t.document_count)
print(t.word_docs)

#encode
encoded_docs = t.texts_to_matrix(text, mode = 'count')
print(encoded_docs)

############### Word Embeddings ################
from gensim.models import Word2Vec
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'], ['this', 'is', 'the', 'second', 'sentence'], ['yet', 'another', 'sentence'], ['one', 'more', 'sentence'], ['and', 'the', 'final', 'sentence']]

model = Word2Vec(tokens, min_count = 1)

#summarize
print(model)
words = list(model.wv.vocab)
print(words)

#access vector for one word
print(model['sentence'])

######## Use Embeddings #######
from sklearn.decomposition import PCA
from matplotlib import pyplot

X = model[model.wv.vocab]
pca = PCA(n_components = 2)
result = pca.fit_transform(X)

#create a scatter plot
pyplot.scatter(result[:,0], result[:,1])
words = list(model.wv.vocab)

print(enumerate(words))
for i, word in enumerate(words):
    pyplot.annotate(word, xy = (result[i,0], result[i,1]) )

pyplot.show()
