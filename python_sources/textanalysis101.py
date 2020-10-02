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

# from textblob lib. import TextBlob method 
from textblob import TextBlob 
  
text = ("Natural language processing (NLP) is a field " + 
       "of computer science, artificial intelligence " + 
       "and computational linguistics concerned with " +  
       "the interactions between computers and human " +  
       "(natural) languages, and, in particular, " +  
       "concerned with programming computers to " + 
       "fruitfully process large natural language " +  
       "corpora. Challenges in natural language " +  
       "processing frequently involve natural " + 
       "language understanding, natural language" +  
       "generation frequently from formal, machine" +  
       "-readable logical forms), connecting language " +  
       "and machine perception, managing human-" + 
       "computer dialog systems, or some combination " +  
       "thereof.") 
    
# create a TextBlob object 
blob_object = TextBlob(text) 
  
# tokenize paragraph into words. 
print(" Word Tokenize :\n", blob_object.words) 
  
# tokenize paragraph into sentences. 
print("\n Sentence Tokenize :\n", blob_object.sentences) 