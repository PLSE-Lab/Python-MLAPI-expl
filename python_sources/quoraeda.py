#!/usr/bin/env python
# coding: utf-8

# ## Quora Dataset Sincere vs InSincere classification
# 
# An existential problem for any major website today is how to handle toxic and divisive content. Quora wants to tackle this problem head-on to keep their platform a place where users can feel safe sharing their knowledge with the world.
# 
# #### About the Data 
# Train and test data set are provided . Test Data set doesn't have the target column.
# 
# Expectation is to reuse the embeddings provided 
# 
# #### About the Notebook
# This notebook covers the basic EDA on this data set The following are covered .
# 1. Word count  Sincere vs Insincere
# 2. Sentence length Sincere vs Insincere 
# 3. Vocabulary based Sentiment Sincere vs InSincere 
# 4. Unigram Sincere vs InSincere
# 5. NER analysis of InSincere PERSON/Organisation

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


traindata=pd.read_csv('../input/train.csv')


# In[ ]:


traindata.shape


# In[ ]:


traindata.info()


# In[ ]:


traindata.target.value_counts().plot(kind='bar')


# In[ ]:


import nltk
from nltk.tokenize import word_tokenize
traindata['wordlen']=traindata['question_text'].apply(lambda x: len(word_tokenize(x)))


# In[ ]:


traindata.head()


# In[ ]:


sns.boxplot(x='target',y='wordlen',data=traindata)


# In[ ]:


print("Max,Mean and Min of word count for Sincere questions")
print(traindata[traindata.target==0]['wordlen'].max())
print(traindata[traindata.target==0]['wordlen'].mean())
print(traindata[traindata.target==0]['wordlen'].min())


# In[ ]:


print("Query with maximum word count", traindata[(traindata.target==0) & (traindata.wordlen==traindata[traindata.target==0]['wordlen'].max())]['question_text'])
                                                 
print("Query with Minimum word count", traindata[(traindata.target==0) & (traindata.wordlen==traindata[traindata.target==0]['wordlen'].min())]['question_text'])
                                                 
                                                 


# #### Let's look at the query with max and min words

# In[ ]:


print("Max,Mean and Min of word count for Sincere questions")
print(traindata[traindata.target==1]['wordlen'].max())
print(traindata[traindata.target==1]['wordlen'].mean())
print(traindata[traindata.target==1]['wordlen'].min())


# In[ ]:


print("Query with maximum word count", traindata[(traindata.target==1) & (traindata.wordlen==traindata[traindata.target==1]['wordlen'].max())]['question_text'])
                                                 
print("Query with Minimum word count", traindata[(traindata.target==1) & (traindata.wordlen==traindata[traindata.target==1]['wordlen'].min())]['question_text'])
                                                 
                                                 


# In[ ]:


traindata['sentencelen']=traindata['question_text'].apply(lambda x: len(x))


# In[ ]:


print("Max,Mean and Min of word count for Sincere questions")
print(traindata[traindata.target==0]['sentencelen'].max())
print(traindata[traindata.target==0]['sentencelen'].mean())
print(traindata[traindata.target==0]['sentencelen'].min())


# In[ ]:


print("Max,Mean and Min of word count for Sincere questions")
print(traindata[traindata.target==1]['sentencelen'].max())
print(traindata[traindata.target==1]['sentencelen'].mean())
print(traindata[traindata.target==1]['sentencelen'].min())


# In[ ]:


traindata.head()


# In[ ]:


sns.boxplot(x='target',y='sentencelen',data=traindata)


# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# ### Vocabulary based sentiment Analysis

# In[ ]:


traindata['sentiment']=traindata['question_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])


# In[ ]:


sns.boxplot(x='target',y='sentiment',data=traindata)


# In[ ]:


sns.distplot(traindata[traindata.target==0]['sentiment'])


# In[ ]:


sns.distplot(traindata[traindata.target==1]['sentiment'])


# ### Clear indication of the sentiment is negative for target==1 and outliers for negative sentiment when target==0
# 
# ### This could be a good feature to feed in

# ### Word Cloud

# In[ ]:


from wordcloud import WordCloud
from nltk.corpus import stopwords
import re


# In[ ]:


from tqdm import tqdm
def preprocess_narrative( questions ):
    final=""
    for text in tqdm(questions):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        words = text.lower().split()  
        stops = set(stopwords.words("english")) 
        for w in words:
            if w not in stops:
                final=final+" "+w
    #print(final)
    return final


# In[ ]:


x=preprocess_narrative(traindata[traindata.target==1]['question_text'])


# In[ ]:


wc = WordCloud(background_color="white", max_words=1000,width=1000, height=500)# mask=alice_mask)
wc.generate(x)


# In[ ]:


fig = plt.figure(figsize = (10, 10))
plt.imshow(wc)


# In[ ]:


# sampling the data set for 0 as the data set is huge 
tempdata=traindata[traindata.target==0]
y=preprocess_narrative(tempdata.sample(frac=0.1)['question_text'])


# In[ ]:


wc.generate(y)


# In[ ]:


fig = plt.figure(figsize = (10, 10))
plt.imshow(wc)


# ### unigram

# In[ ]:


from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from string import punctuation, ascii_lowercase
import regex as re
from tqdm import tqdm
# setup tokenizer
tokenizer = WordPunctTokenizer()

stops = set(stopwords.words("english"))
def text_to_wordlist(text, lower=False):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
    # Tokenize
    text = tokenizer.tokenize(text)
    
    # optional: lower case
    if lower:
        text = [t.lower() for t in text]
    
    
    text = [t if t not in stops else None for t in text]
    
    
    
    # Return a list of words
    vocab.update(text)
    #return text

def process_comments(list_sentences, lower=False):
    comments = []
    for text in tqdm(list_sentences):
        text_to_wordlist(text, lower=lower)
        


# In[ ]:


vocab=Counter()
process_comments(traindata[traindata.target==0]['question_text'],True)


# In[ ]:


vocab.pop(None)
since_most_common=vocab.most_common(20)


# In[ ]:


vocab=Counter()
process_comments(traindata[traindata.target==1]['question_text'],True)


# In[ ]:


vocab.pop(None)
insincere_most_common=vocab.most_common(20)


# In[ ]:


sincere_mc=pd.DataFrame(since_most_common)
insincere_mc=pd.DataFrame(insincere_most_common)
sincere_mc.columns=['word','count']
insincere_mc.columns=['word','count']


# In[ ]:


sincere_mc.plot(x='word',kind='bar')
insincere_mc.plot(x='word',kind='bar')


# ### Check the most frequent words of target==0 and sentiment is negative 

# In[ ]:


vocab=Counter()
process_comments(traindata[(traindata.target==0) & (traindata.sentiment <0)]['question_text'],True)


# In[ ]:


vocab.pop(None)
negative_sincere_most_common=vocab.most_common(20)


# In[ ]:


negative_sincere_most_common=pd.DataFrame(negative_sincere_most_common)
negative_sincere_most_common.columns=['word','count']
negative_sincere_most_common.plot(x='word',kind='bar')


# ### Checking Which PERSON has been referred more in the insincere questions
# #### Trump should definitely come up , Let's see who else comes up 

# In[ ]:





# In[ ]:


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()


# In[ ]:


from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from string import punctuation, ascii_lowercase
import regex as re
from tqdm import tqdm
# setup tokenizer
tokenizer = WordPunctTokenizer()
vocab=Counter()
org=Counter()
stops = set(stopwords.words("english"))
labels=[]
def process_ner(list_sentences):
    for text in tqdm(list_sentences):
        
        doc = nlp(text)
        for x in doc.ents:
            if(x.label_=='PERSON'):
                vocab.update([x.text.lower()])
            if(x.label_=='ORG'):
                org.update([x.text.lower()])


# #### Sampling the data  50 % 

# In[ ]:


process_ner(traindata[traindata.target==1].sample(frac=0.5)['question_text'])


# In[ ]:


plt.figure(figsize=(20,20))
person_most_common=pd.DataFrame(vocab.most_common(50))
person_most_common.columns=['Name','count']
personplot=sns.barplot(y="count",x="Name",data=person_most_common)
loc, labels = plt.xticks(rotation='vertical')


# In[ ]:


plt.figure(figsize=(20,20))
org_most_common=pd.DataFrame(org.most_common(50))
org_most_common.columns=['Name','count']
orgplot=sns.barplot(y="count",x="Name",data=org_most_common)
loc, labels = plt.xticks(rotation='vertical')


# #### Insincere Questions seem to have religious orientation

# In[ ]:




