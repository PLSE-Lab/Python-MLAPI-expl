#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import string

import nltk
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from wordcloud import WordCloud

import matplotlib.style as style
style.use('dark_background')


# In[ ]:


data = pd.read_csv("../input/covid19-containment-and-mitigation-measures/COVID 19 Containment measures data.csv")


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.isna().sum()/len(data)*100


# In[ ]:


data['Date Start'] = pd.to_datetime(data['Date Start'])


# In[ ]:


amount = data.groupby('Country')['Description of measure implemented'].nunique().sort_values(ascending=False).head(20)
amount = amount.to_frame().reset_index()


# In[ ]:


co = sns.cubehelix_palette(n_colors=20,
                           start=0,
                           rot=0.4,
                           gamma=1.0,
                           hue=0.8,
                           light=0.85,
                           dark=0.15,
                           reverse=True,
                           as_cmap=False)


# In[ ]:


plt.figure(figsize=(15,10))

sns.barplot(data = amount, y = 'Country', x = 'Description of measure implemented', palette=co)
plt.title('Number of implemented measures and restrictions by country', fontsize=24)
plt.xlabel('Measures implemented', fontsize=18)
plt.ylabel('Country', fontsize=18)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)


# # Quick text mining

# In[ ]:


def remove_line_breaks(text):
    text = text.replace('\r', ' ').replace('\n', ' ')
    return text

#remove punctuation
def remove_punctuation(text):
    re_replacements = re.compile("__[A-Z]+__")  # such as __NAME__, __LINK__
    re_punctuation = re.compile("[%s]" % re.escape(string.punctuation))
    '''Escape all the characters in pattern except ASCII letters and numbers: word_tokenize('ebrahim^hazrati')'''
    tokens = word_tokenize(text)
    tokens_zero_punctuation = []
    for token in tokens:
        if not re_replacements.match(token):
            token = re_punctuation.sub(" ", token)
        tokens_zero_punctuation.append(token)
    return ' '.join(tokens_zero_punctuation)

def remove_special_characters(text):
    text = re.sub('[^a-zA-z0-9\s]', '', text)
    return text

def lowercase(text):
    text_low = [token.lower() for token in word_tokenize(text)]
    return ' '.join(text_low)

def remove_stopwords(text):
    stop = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    text = " ".join([word for word in word_tokens if word not in stop])
    return text

#remove punctuation
def remove_punctuation(text):
    re_replacements = re.compile("__[A-Z]+__")  # such as __NAME__, __LINK__
    re_punctuation = re.compile("[%s]" % re.escape(string.punctuation))
    '''Escape all the characters in pattern except ASCII letters and numbers: word_tokenize('ebrahim^hazrati')'''
    tokens = word_tokenize(text)
    tokens_zero_punctuation = []
    for token in tokens:
        if not re_replacements.match(token):
            token = re_punctuation.sub(" ", token)
        tokens_zero_punctuation.append(token)
    return ' '.join(tokens_zero_punctuation)

#remobe one character words
def remove_one_character_words(text):
    '''Remove words from dataset that contain only 1 character'''
    text_high_use = [token for token in word_tokenize(text) if len(token)>1]      
    return ' '.join(text_high_use)   

##remove specific word list
def remove_special_words(text):
    '''Remove the User predefine useless words from the text. The list should be in the lowercase.'''
    special_words_list=['af', 'iv', 'ivm', 'mg', 'dd', 'vrijdag','afspraak','over','met', 'van', 'patient', 'dr', 'geyik','heyman','bekker','dries','om', 'sel', 'stipdonk', 'eurling', 'knackstedt'
                        'lencer','volder','schalla']# list : words
    querywords=text.split()
    textwords = [word for word in querywords if word.lower() not in special_words_list]
    text=' '.join(textwords)
    return text
    
#%%
# Stemming with 'Snowball Dutch stemmer" package
def stem(text):
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    text_stemmed = [stemmer.stem(token) for token in word_tokenize(text)]        
    return ' '.join(text_stemmed)

def lemma(text):
    wordnet_lemmatizer = WordNetLemmatizer()
    word_tokens = nltk.word_tokenize(text)
    text_lemma = " ".join([wordnet_lemmatizer.lemmatize(word) for word in word_tokens])       
    return ' '.join(text_lemma)


#break sentences to individual word list
def sentence_word(text):
    word_tokens = nltk.word_tokenize(text)
    return word_tokens
#break paragraphs to sentence token 
def paragraph_sentence(text):
    sent_token = nltk.sent_tokenize(text)
    return sent_token    


def tokenize(text):
    """Return a list of words in a text."""
    return re.findall(r'\w+', text)


#%% make a text c' '.join(data['le'][6])learning function specific for pitch decks so far 
def normalization_pitchdecks(text):
    _steps = [
    remove_line_breaks,
    remove_one_character_words,
    remove_special_characters,
    lowercase,
    remove_punctuation,
    remove_stopwords,
    remove_special_words,
    stem
]
    for step in _steps:
        text=step(text)
    return text   
#%%


# In[ ]:


text = data['Description of measure implemented']
exceptions = data['Exceptions']


# In[ ]:


data = data.replace(np.nan, '', regex=True) 


# In[ ]:


common_words = []

for i in data['Description of measure implemented']:
    common_words.append(normalization_pitchdecks(i))
    
    
    
common_exceptions = []

for i in data['Exceptions']:
    common_exceptions.append(normalization_pitchdecks(i))


# In[ ]:


data['text_clean'] = pd.Series(common_words)
data['exceptions_clean'] = pd.Series(common_exceptions)


# In[ ]:


text_clean = data['text_clean'].dropna().to_list()
exceptions_clean = data['exceptions_clean'].dropna().to_list()


# In[ ]:





# In[ ]:


plt.figure(figsize=(16,13))
wc = WordCloud(background_color="black", max_words=1000, max_font_size= 200,  width=1600, height=800)
g = wc.generate(" ".join(exceptions_clean))
plt.title("Most discussed Exceptions", fontsize=27)
plt.imshow(wc.recolor( colormap= 'gist_rainbow' , random_state=17), alpha=0.98, interpolation="bilinear", )
plt.axis('off')

plt.figure(figsize=(16,13))
wc = WordCloud(background_color="black", max_words=1000, max_font_size= 200,  width=1600, height=800)
q = wc.generate(" ".join(text_clean))
plt.title("Most discussed terms", fontsize=27)
plt.imshow(wc.recolor( colormap= 'gist_rainbow' , random_state=17), alpha=0.98, interpolation="bilinear", )
plt.axis('off')


# In[ ]:


ls= []

for i in text_clean:
    ls.append(str(i).split())


# In[ ]:


fdist = FreqDist()

for sentence in ls:
    for token in sentence:
        fdist[token] +=1


# In[ ]:


top_title = fdist.most_common(20)


# In[ ]:


ls = []
for i in top_title:
    ls.append({'Word': i[0], 'Num': i[1]})

df = pd.DataFrame(ls)


# In[ ]:


co = sns.cubehelix_palette(n_colors=20,
                           start=0,
                           rot=-0.4,
                           gamma=1.0,
                           hue=0.8,
                           light=0.85,
                           dark=0.15,
                           reverse=True,
                           as_cmap=False)


# In[ ]:


plt.figure(figsize=(15,10))

sns.barplot(data = df, y = 'Word', x = 'Num', palette=co)
plt.title('Most discussed words', fontsize = 24)
plt.xlabel('Occurrences', fontsize=18)
plt.ylabel('Word', fontsize=18)
ax = plt.gca()
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)


# In[ ]:





# In[ ]:




