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

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, GlobalMaxPool1D
from keras.models import Model

maxSentenceLength = 100
maxVocabNumber = 100000

# Clean the text.
def cleanText(text, removeStopwords=True, performStemming=True):
    
    #regex for removing non-alphanumeric characters and spaces
    remove_special_char = re.compile('r[^a-z\d]', re.IGNORECASE)
    #regex to replace all numerics
    replace_numerics = re.compile(r'\d+', re.IGNORECASE)
    text = remove_special_char.sub('', text)
    text = replace_numerics.sub('', text)

    stop_words = set(stopwords.words('english')) 
    stemmer = SnowballStemmer('english')
    lemmatizer = WordNetLemmatizer()
    
    #convert text to lowercase.
    text = text.lower().split()

    
    processedText = list()
    for word in text:        
        if removeStopwords:
            if word in stop_words:
                continue
        if performStemming:
            word = stemmer.stem(word)
            
        word = lemmatizer.lemmatize(word)
        word = lemmatizer.lemmatize(word, 'v')
            
        processedText.append(word)

    text = ' '.join(processedText)

    return text

# Load file.
imdb = pd.read_csv('../input/IMDB Dataset.csv')

# Prepare X
x = [cleanText(text) for text in list(imdb['review'])]
# Prepare Y
y = [1 if sentiment=='positive' else 0 for sentiment in list(imdb['sentiment'])]

# Make a tokenizer
tokenizer = Tokenizer(num_words=maxVocabNumber)
tokenizer.fit_on_texts(x)

# Padding.
x_tokenized = pad_sequences(tokenizer.texts_to_sequences(x), maxlen=maxSentenceLength)


# Build a model.
inp = Input(shape=(maxSentenceLength,))
emb = Embedding(maxVocabNumber, 100)(inp)
bilstm = Bidirectional(LSTM(60, return_sequences = True))(emb)
maxp = GlobalMaxPool1D()(bilstm)
out = Dense(60, activation='selu')(maxp)
out = Dense(1, activation='sigmoid')(out)
model = Model(inputs=inp, outputs=out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

batchSize = 1000
epochs = 10
hist = model.fit(x_tokenized, y, batch_size=batchSize, epochs=epochs, verbose=1, shuffle=True, validation_split=0.5)





