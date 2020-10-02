#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 

# Dataset from Kaggle 
# source https://www.kaggle.com/gunnvant/game-of-thrones-srt
# Json to CSV transform completed with Trifacta

data = pd.read_csv('../input/got-subtitles/Scripts.csv')


# In[2]:


data.head(5)


# In[3]:


data = pd.DataFrame(data)


# In[4]:


data.iloc[:1]


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer


# In[6]:


def seasonLine(season, line, df=data):
       num = season - 1
       x = df.iloc[num, line]
       return x 


# In[7]:


seasonLine(1, 2, df=data)


# In[8]:


def seasonTokens(season, df=data):
    '''Input - row from csv. Subtitles for entire season
    
       Output - Token list by word (tokens), and by line (lineList) 
    '''
    
    x = 1
    
    lineList = []
    
    tokens = []
    
    wordline = ''
    
    while isinstance(wordline, str):
    
        wordline = seasonLine(season, x, df=data)
               
        lineList.append(wordline)
        
        x += 1 
        
    for y in lineList:
        
        bite = str(y).split()
        
        tokens = tokens + bite 
    
    return tokens, lineList   


# In[9]:


tokenList, SeasonList = seasonTokens(1, df=data)


# In[10]:


SeasonList[:5]


# In[11]:


tokenList[:35]


# In[12]:


def TopList(num):
    
    '''Input - num - integer, list of tokens  
       Ouput count of the the most popular terms by frequency '''
    
    vector = CountVectorizer(tokenList, stop_words = 'english')
    
    word_matrix = vector.fit_transform(tokenList).toarray()
    
    frequency = pd.DataFrame(word_matrix, columns = vector.get_feature_names())
    
    total = frequency.sum(axis = 0).sort_values(ascending = False)
    
    return total[:num], total.index[:num]


# In[13]:


topWordcount, topWord = TopList(25)


# In[14]:


TopList(5)


# In[15]:


# Spacy Natural Language Processing package 
#small model 
import spacy
nlp =spacy.load('en_core_web_sm')


# In[16]:


# accounts for non-string terms in list of subtitles 

seasonText = []
for x in SeasonList:
    seasonText.append(str(x))
    
# Creates a string of subtitles for each episode     
episodeText = ','.join(seasonText)


# In[17]:


episodeText[:1000]


# In[18]:


# textacy is a Python library for performing higher-level natural language processing (NLP) tasks, 
#built on the high-performance Spacy library
import textacy
import textacy.keyterms
import textacy.extract


# In[19]:


# preprocess text
cleanText = textacy.preprocess.remove_punct(episodeText)


# In[20]:


cleanText[:1000]


# In[21]:


normalizedText = textacy.preprocess.normalize_whitespace(cleanText)


# In[22]:


normalizedText[:1000]


# In[23]:


# Create SpaCy Text object 
SpacyTextObject = nlp(normalizedText)


# In[24]:


for entity in SpacyTextObject.ents:
    if str(entity).lower() in topWord:
        print(entity.text, entity.label_)


# In[25]:


for entity in SpacyTextObject.noun_chunks:
    if str(entity).lower() in topWord:
        print(entity.text, entity.label_)


# In[26]:


# Extract key terms from a document using the [SGRank] algorithm.
key = textacy.keyterms.sgrank(SpacyTextObject, ngrams=(1, 2, 3, 4, 5, 6), normalize='lemma', window_width=1500, n_keyterms=10, idf=None)
key


# In[27]:


# Extract an ordered sequence of named entities (PERSON, ORG, LOC, etc.) 
#from a spacy-parsed doc, optionally filtering by entity types and frequencies.
word1 = textacy.extract.named_entities(SpacyTextObject, include_types=None, exclude_types=None, drop_determiners=True, min_freq=4)
for x in word1:
    print (x)


# In[28]:


#Top Phrases by word count
word2 = textacy.extract.ngrams(SpacyTextObject, 3, filter_stops=True, filter_punct=True, filter_nums=False, include_pos=None, exclude_pos=None, min_freq=3)
for y in word2:
    print(y)


# In[29]:


text = '''Ned is a fictional character. Ned is the lord of Winterfell, an ancient fortress in the North of the fictional continent of Westeros. 
Though the character is established as the main character in the novel and the first season of the TV adaptation,
Martin's plot twist at the end involving Ned shocked both readers of the book and viewers of the TV series. 
Ned is the leader of the Stark Family. Ned is a father of six children.'''


# In[30]:


def summary(sentence, matchWord):
    summary = ''
    sentobj = nlp(sentence)
    sentences = textacy.extract.semistructured_statements(sentobj, matchWord, cue = 'be')
    for i, x in enumerate(sentences):
        subject, verb, fact = x
        
        summary += 'Fact '+str(i+1) +': '+(str(fact))+" "
    return summary   


# In[31]:


summary(text, 'Ned')


# In[32]:


doc = nlp(text)
verbTriples = textacy.extract.subject_verb_object_triples(doc)
for x in verbTriples:
    print(x)


# In[ ]:




