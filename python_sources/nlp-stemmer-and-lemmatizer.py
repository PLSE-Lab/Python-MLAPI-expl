# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 13:33:20 2018

@author: uknemani
"""
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
print (ps.stem('playing'))
print (ps.stem('increased'))

#SNowBall Stemmer for non english stemming
from nltk.stem import SnowballStemmer
print(SnowballStemmer.languages)
sbs = SnowballStemmer('german')
print(sbs.stem('bon jour'))

#Lemmatizing words
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize('increases'))
print(lemmatizer.lemmatize('increasing', pos ='v'))
# similarly v ,n, a,r
