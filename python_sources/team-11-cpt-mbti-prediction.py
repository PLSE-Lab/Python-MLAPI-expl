#!/usr/bin/env python
# coding: utf-8

# # <center> Personality Profile Predictions </center>

# ___________________________________________________________________________________________________________________________________________________

# # Table of contents

# 1. [Introduction](#1)
# 2. [Importing libraries](#2)
# 3. [Importing data](#3)
# 3. [Creating required functions](#4)
# 4. [EDA and feature engineering](#5)
# 5. [Train preprocessing](#6)
# 6. [Test preprocessing](#7)
# 7. [Vectorization](#8)
# 8. [Model fitting and predicting](#9)
#     * [Mind](#10)
#     * [Energy](#11)
#     * [Nature](#12)
#     * [Tactics](#13)
#     
# 9. [Prepared submission](#14)
# 10. [Still to do](#15) 
# 11. [Acknowledgements](#16)
# 

# 
# <a id='1'></a>
# ___________________________________________________________________________________________________________________________________________________

# # Introduction

# <div style="text-align: justify">The Myers-Briggs Type Indicator (mbti) categories individuals into 16 different personality types using four opposite pairs of variables represented by a letter or word. These letters each represent a characteristic that groups interests, needs and values together. The MBTI personality type binary variables are: Mind: Introverted (I) or Extraverted (E) Energy: Sensing (S) or Intuitive (N) Nature: Feeling (F) or Thinking (T) Tactics: Perceiving (P) or Judging (J) An individual's final type is made up of one of the variables combined. For example an individual with INFP type would have a combination of Introverted(I), Intuitive (N), Feeling (F) and Perceiving (P). </div>
# 
# <div style="text-align: justify"> The common way of finding out your persoanlity type is to take a personality type test on a websites, where they would determine your personality type from the different questions you have to answer about yourself.
# In this notebook, I will build a model that will predict the personality of a person from their twitter post. We will predict four separate labels for each person which, when combined, results in that person's personality type just like the example. The data is available on kaggle competition.</div>
# 
# For more info about the MBTI personality types click [here](https://www.16personalities.com/personality-types) OR the test click [here to take the test](https://www.16personalities.com/free-personality-test)

# [Return to index](#index)
# <a id='2'></a>
# ___________________________________________________________________________________________________________________________________________________

# # Importing libraries

# In[ ]:


#Standard Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#Natural language processing libraries
import nltk
import re

import time

#Interactive computing
from IPython.core.magics.execution import _format_time
from IPython.display import display as d
from IPython.display import Audio
from IPython.core.display import HTML

#Accountability
import logging as log

import os
print(os.listdir("../input"))


# [Return to index](#index)
# ___________________________________________________________________________________________________________________________________________________
# <a id='3'></a>

# # Importing data

# We loaded our data (train.csv and test.csv) and inspected. This helped to see where we can start with feature engineering. 

# In[ ]:


train_df = pd.read_csv('../input/train.csv')
train_df.head()


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


#Checking if we have all sixteen personality types represented
Personalities = train_df.groupby("type").count()
Personalities.sort_values("posts", ascending=False, inplace=True)
Personalities.index.values


# Above output shows the 'type' column contains 16 unique codes, representing the 16 different personality types.

# ### Distribution of Myers-Briggs Personality Types in the Dataset

# In[ ]:


#Visualizing the distribution of the personality types
count_types = train_df['type'].value_counts()
plt.figure(figsize=(10,5))
sns.barplot(count_types.index, count_types.values, alpha=1, palette="winter")
plt.ylabel('No of persons', fontsize=12)
plt.xlabel('personality Types', fontsize=12)
plt.title('Distribution of personality types')
plt.show()


# The bar chart above shows that INFP (Introversion - Intuition - Feeling - Perceiving) is the most frequently appearing type in the dataset, followed by INFJ (Introversion - Intuition - Feeling - Judging). Overall, the dataset contains many more Intuitive-Intuition (IN-) groupings than any other type. Conversely, the dataset contains very few Extroversion-Sensing (ES-) types.

# [Return to index](#index)
# ___________________________________________________________________________________________________________________________________________________
# <a id='4'></a>

# # Function creation

# In cleaning and exploring our data we built some functions that will help remove and transform our features that make machine learning algorithms work. We created the 'alert' function for the long running time cells, this will alert us when they are done running. The 'link transformer' function opens the link address in the data frame to find title names of the url. It then replaces the url with the title. The 'remove urls' function removes urls and replace it with web-link. The function 'no_punc_num' removes numbers and puntuation. The 'lemmatized' function lemmatizes our words using WordNetLemmatizer. Lastly, we created 'remove_stop_words' function which removed stop words that we will decide not to use.

# In[ ]:


def alert():
    """ makes sound on client using javascript"""  
    
    framerate = 44100
    duration=0.5
    freq=340
    t = np.linspace(0,duration,framerate*duration)
    data = np.sin(2*np.pi*freq*t)
    d(Audio(data,rate=framerate, autoplay=True))


# In[ ]:


def link_transformer(df, column, reports=True):
    """Search over a column in a pandas dataframe for urls.
    
    extract the title related to the url then replace the url with the title.
    
    
    df : pandas Dataframe object
    
    column: string type object equal to the exact name of the colum you want to replace the urls
    
    reports: Boolean Value (default=True)
        If true give active report updates on the last index completed and the ammount of reported fail title extractions
   
    """
    
    total_errors = 0
    count = 0
    from mechanize import Browser
    br = Browser()
    
    while count != len(df):
        errors = 0
        
        url = re.findall(r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+', df.loc[count, column])
        
        for link in url:
            try: 
                br.open(link)
                df.loc[count, column] = df.loc[count, column].replace(link, br.title())
            except:
                
                if reports == True:
                    print(f'failed--- {link}')
                elif reports == False:
                    pass
                else:
                    raise ValueError('reports expected a boolean value')
                
                total_errors += 1
                errors += 1
                
                continue
                
        if reports == True:
            if errors == 0:
                report = 'no errors'
                errors = ''
            elif errors == 1:
                report = 'error'
            else:
                report = 'errors'
            print(f'\nIndex {count + 1} completed. {errors} {report} reported\n______________________\n\n')
    
        elif reports == False:
            pass
        
        else:
            raise ValueError('reports expected a boolean value')
                
        
        count += 1
    print(f'{total_errors} total errors throughout full runtime')
    
#example

#sample = pd.read_csv('train.csv').sample(3, random_state=20).reset_index(drop=True)
#sample

#link_transformer(sample, 'posts')

#sample


# In[ ]:


def remove_links(df, column):
    """Replace urls by searching for the characters normally found in urls 
    and replace the string found with the string web-link
    
    df : pandas Dataframe object
    
    column: string type object equal to the exact name of the colum you want to replace the urls
    """
    
    return df[column].replace(r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+', 
                           r'web-link', regex=True, inplace=True)


# In[ ]:


def no_punc_num(post):
    """The function imports punctuation and define numbers then removes them from a column in a dataframe 
    using a for loop.
    
    to use, use pd.DataFrame.apply() on a Dataframe object"""
    
    from string import punctuation
    pun_num = punctuation + '1234567890'
    return ''.join([letter for letter in post if letter not in pun_num])


# In[ ]:


tokens = nltk.tokenize.TreebankWordTokenizer()


# In[ ]:


lem = nltk.stem.WordNetLemmatizer()
def lemmatized(words, lemmatizer):
    """Transform a list of words into base forms 
    
    example: hippopotami = hippopotamus


    Required imports  
   ------------------
    nltk.stem.WordNetLemmatizer()
    
    
    Parameters 
   ------------
   lemmatizer: nltk.stem.WordNetLemmatizer() object
   
   
   to use, use pd.DataFrame.apply() on a Dataframe object 
    """
    return [lemmatizer.lemmatize(word) for word in words]


# In[ ]:


def remove_stop_words(tokens):
    """Removes a list of words from a Dataframe
    
    Required imports  
   ------------------
    A list of stopwords to remove
    
    
    to use, use pd.DataFrame.apply() on a Dataframe object  
    """
    return [t for t in tokens if t not in stopwords]


# # Feature engineering

# To start with our feature engineering we started by spitting the sentences. (|||) indicated where a sentece started or ended so we split the sentence were we find (|||). We tokenized and lammatized our sentences. Then we created a bag of words. We grouped the lemmatized words that were used per type, this we can see which words were mostly used by certain personality types and create stopwords for each variable.   

# In[ ]:


#Splitting sentences
train = []
for types, posts in train_df.iterrows():
    for split_at in posts['posts'].split('|||'):
        train.append([posts['type'], split_at])
train = pd.DataFrame(train, columns=['type', 'post'])


# In[ ]:


train.head()


# In[ ]:


#making all the words lowwer case
train.post = train.post.str.lower()


# In[ ]:


#removing punctuation and numbers
train.post = train.post.apply(no_punc_num)


# In[ ]:


#tokenizing words
train['tokenized'] = train.post.apply(tokens.tokenize)


# In[ ]:


#lemmatizing words
train['lemmatized'] = train.tokenized.apply(lemmatized, args=(lem,))


# In[ ]:


def bag_count(word, bag={}):
    '''text vectorize by representing every word as a integer and counting the frequency of appearence'''
    for w in word:
        if w not in bag.keys():
            bag[w] = 1
        else:
            bag[w] += 1
    return bag

per_type = {}
for pt in list(train.type.unique()):
    df = train.groupby('type')
    per_type[pt] = {}
    for row in df.get_group(pt)['lemmatized']:
        per_type[pt] = bag_count(row, per_type[pt])

len(per_type.keys())


# In[ ]:


#creating a list of unique words
unique_words = set()
for pt in list(train.type.unique()):
    for word in per_type[pt]:
        unique_words.add(word)


# In[ ]:


unique_words


# In[ ]:


personality_stop_words = list(per_type.keys())


# In[ ]:


#finding the frequency of words
per_type['all'] = {}
for tp in list(train.type.unique()):
    for word in unique_words:
        if word in per_type[tp].keys():
            if word not in per_type['all']:
                per_type['all'][word] = per_type[tp][word]
            else:
                per_type['all'][word] += per_type[tp][word] 


# In[ ]:


per_type['all']


# In[ ]:


print(len(per_type['all']))


# In[ ]:


#Appearence of a word longer that 2 standard deviations in percentage
(sum([v for v in per_type['all'].values() if v >= 43]))/sum([v for v in per_type['all'].values()])


# In[ ]:


#Checking the words
word_index = [k for k, v in per_type['all'].items() if v > 43]


# In[ ]:


#using for loop to find word usage per type
per_type_words = []
for pt, p_word in per_type.items():
    word_useage = pd.DataFrame([(k, v) for k, v in p_word.items() if k in word_index], columns=['Word', pt])
    word_useage.set_index('Word', inplace=True)
    per_type_words.append(word_useage)


# In[ ]:


word_useage = pd.concat(per_type_words, axis=1)
word_useage.fillna(0, inplace=True)


# In[ ]:


word_useage.sample(10)


# In[ ]:


personality_stop_words


# In[ ]:


#Finding sum of the word usage and identifying them to each variable

I = [x for x in personality_stop_words if x[0] == 'I']
E = [x for x in personality_stop_words if x[0] == 'E']
word_useage['I'] = word_useage[I].sum(axis=1)
word_useage['E'] = word_useage[E].sum(axis=1)

S = [x for x in personality_stop_words if x[1] == 'S']
N = [x for x in personality_stop_words if x[1] == 'N']
word_useage['S'] = word_useage[S].sum(axis=1)
word_useage['N'] = word_useage[N].sum(axis=1)

F = [x for x in personality_stop_words if x[2] == 'F']
T = [x for x in personality_stop_words if x[2] == 'T']
word_useage['F'] = word_useage[F].sum(axis=1)
word_useage['T'] = word_useage[T].sum(axis=1)

P = [x for x in personality_stop_words if x[3] == 'P']
J = [x for x in personality_stop_words if x[3] == 'J']
word_useage['P'] = word_useage[P].sum(axis=1)
word_useage['J'] = word_useage[J].sum(axis=1)


# In[ ]:


word_useage.sample(10)


# In[ ]:


#Word usage in percentage form
for col in ['I', 'all']:
    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()
for col in ['E', 'all']:
    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()

for col in ['S', 'all']:
    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()
for col in ['N', 'all']:
    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()

for col in ['F', 'all']:
    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()
for col in ['T', 'all']:
    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()

for col in ['P', 'all']:
    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()
for col in ['J', 'all']:
    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()


# In[ ]:


word_useage.sample(1)


# In[ ]:


stopwords = nltk.corpus.stopwords.words('english')


# In[ ]:


#Word usage in percentage form for each variable 
word_useage['I chi2'] = np.power((word_useage['I_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)
word_useage['E chi2'] = np.power((word_useage['E_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)

word_useage['S chi2'] = np.power((word_useage['S_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)
word_useage['N chi2'] = np.power((word_useage['N_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)

word_useage['F chi2'] = np.power((word_useage['F_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)
word_useage['T chi2'] = np.power((word_useage['T_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)

word_useage['P chi2'] = np.power((word_useage['P_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)
word_useage['J chi2'] = np.power((word_useage['J_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)


# In[ ]:


I_words = word_useage[['I_perc', 'all_perc', 'I chi2']][(word_useage['I_perc'] > word_useage['all_perc'])].sort_values(by='I chi2', ascending=False)
E_words = word_useage[['E_perc', 'all_perc', 'E chi2']][word_useage['E_perc'] > word_useage['all_perc']].sort_values(by='E chi2', ascending=False)

S_words = word_useage[['S_perc', 'all_perc', 'S chi2']][(word_useage['S_perc'] > word_useage['all_perc'])].sort_values(by='S chi2', ascending=False)
N_words = word_useage[['N_perc', 'all_perc', 'N chi2']][word_useage['N_perc'] > word_useage['all_perc']].sort_values(by='N chi2', ascending=False)

F_words = word_useage[['F_perc', 'all_perc', 'F chi2']][(word_useage['F_perc'] > word_useage['all_perc'])].sort_values(by='F chi2', ascending=False)
T_words = word_useage[['T_perc', 'all_perc', 'T chi2']][word_useage['T_perc'] > word_useage['all_perc']].sort_values(by='T chi2', ascending=False)

P_words = word_useage[['P_perc', 'all_perc', 'P chi2']][(word_useage['P_perc'] > word_useage['all_perc'])].sort_values(by='P chi2', ascending=False)
J_words = word_useage[['J_perc', 'all_perc', 'J chi2']][word_useage['J_perc'] > word_useage['all_perc']].sort_values(by='J chi2', ascending=False)


# In[ ]:


I_keep = I_words[I_words.index.isin(list(stopwords))].head(5)
E_keep = E_words[E_words.index.isin(list(stopwords))].head(5)

S_keep = S_words[S_words.index.isin(list(stopwords))].head(5)
N_keep = N_words[N_words.index.isin(list(stopwords))].head(5)

F_keep = F_words[F_words.index.isin(list(stopwords))].head(5)
T_keep = T_words[T_words.index.isin(list(stopwords))].head(5)

P_keep = P_words[P_words.index.isin(list(stopwords))].head(5)
J_keep = J_words[J_words.index.isin(list(stopwords))].head(5)


# In[ ]:


I_keep = list(I_keep.index)
E_keep = list(E_keep.index)

S_keep = list(S_keep.index)
N_keep = list(N_keep.index)

F_keep = list(F_keep.index)
T_keep = list(T_keep.index)

P_keep = list(P_keep.index)
J_keep = list(J_keep.index)


# In[ ]:


keep = I_keep+E_keep+S_keep+N_keep+F_keep+T_keep+P_keep+J_keep

keep = set(keep)


# In[ ]:


len(keep)


# In[ ]:


stop = nltk.corpus.stopwords.words('english')


# In[ ]:


len(stop)


# In[ ]:


stopwords = []
for i in stop:
    if i in keep:
        pass
    else:
        stopwords.append(i)


# In[ ]:


len(stopwords)


# [Return to index](#index)
# ___________________________________________________________________________________________________________________________________________________
# <a id='5'></a>

# # train

# To check if we still have any unwatned characters, we look at our training data again and remove any unwanted characters. We also created our four columns for each variable using the type column. 

# In[ ]:


train = train_df
sample = train.sample(3).reset_index(drop=True)
train.shape


# In[ ]:


#Removes links
remove_links(train, 'posts')
train.head(1)


# In[ ]:


train['posts'].replace(r'\|\|\|', r' ', regex=True, inplace=True)
train['posts'].head(1)


# In[ ]:


#Removes punchuations and set text to lowercase
train['posts'] = train['posts'].str.lower()
train['posts'] = train['posts'].apply(no_punc_num)
train['posts'].head(1)


# In[ ]:


#Tokenize the posts
train['posts'] = train['posts'].apply(tokens.tokenize)
train['posts'].head(1)


# In[ ]:


#Lemmatize the posts
train['posts'] = train['posts'].apply(lemmatized, args=(lem,))
train.head(1)


# In[ ]:


#Removes stopwords from the posts
train['posts'] = train['posts'].apply(remove_stop_words)
train.head(1)


# In[ ]:


train['posts'] = [' '.join(map(str, l)) for l in train['posts']]
train.head(1)


# In[ ]:


train['Mind']   = train['type'].apply(lambda x: x[0] == 'E').astype('int')
train['Energy'] = train['type'].apply(lambda x: x[1] == 'N').astype('int')
train['Nature'] = train['type'].apply(lambda x: x[2] == 'T').astype('int')
train['Tactics']= train['type'].apply(lambda x: x[3] == 'J').astype('int')
train = train[['Mind','Energy','Nature','Tactics','posts', 'type']]


# Column names for the new columns and their binary codes(1s and 0s):
# 
# - Mind: Introversion(I = 0) - Extroversion(E = 1)<br/>
# - Energy: Sensing(S = 0) - Intuition(N = 1)<br/>
# - Nature: Feeling(F = 0) - Thinking(T = 1)<br/>
# - Tactics: Perceiving(P = 0) - Judging(J = 1)

# In[ ]:


train.head(1)


# [Return to index](#index)
# ___________________________________________________________________________________________________________________________________________________
# <a id='6'></a>

# # test

# We did the same for our test data. We removed links, punctuation, numbers and stop words. We lammatized our words. We did all this by using the functions we built. We will call our functions and check everytime to see if the function is working by using .head()

# In[ ]:


#Removes links
remove_links(test, 'posts')
test.head(1)


# In[ ]:


test['posts'].replace(r'\|\|\|', r' ', regex=True, inplace=True)
test['posts'].head(1)


# In[ ]:


#Removes punchuations and set text to lowercase
test['posts'] = test['posts'].str.lower()
test['posts'] = test['posts'].apply(no_punc_num)
test['posts'].head(1)


# In[ ]:


#Tokenize the posts
test['posts'] = test['posts'].apply(tokens.tokenize)
test['posts'].head(1)


# In[ ]:


#Lemmatize the posts
test['posts'] = test['posts'].apply(lemmatized, args=(lem,))
test.head(1)


# In[ ]:


#Removes the stopwords from the posts
test['posts'] = test['posts'].apply(remove_stop_words)
test.head(1)


# In[ ]:


test['posts'] = [' '.join(map(str, l)) for l in test['posts']]
test.head(1)


# [Return to index](#index)
# ___________________________________________________________________________________________________________________________________________________
# <a id='7'></a>

# # Vectorization

# We created a bag of words above but we will also be using CountVectorizer and TfidfVectorizer below. These methods work differently to do the same work. Although CountVectorizer is traditionally the main vectorizer these methods are found on the same module and not superior to each other. But since they work differently and have different parameters, we will test them both.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# CountVectorizer encodes text by splitting a set of words into one column per word, with (by default) the count of the word for that row in that column.

# In[ ]:


#Vectorising using CountVectorizer
Count_vect = CountVectorizer(max_df=0.8, min_df=43,  lowercase=False)
Count_train = Count_vect.fit_transform(train['posts'])
Count_test = Count_vect.transform(test['posts'])


# TfidfVectorizer convert a collection of raw documents to a matrix of TF-IDF features.

# In[ ]:


#Vectorising using TfidfVectorizer
Tfidf_vect =TfidfVectorizer(max_df=0.8, min_df=43, lowercase=False)
Tfidf_train = Tfidf_vect.fit_transform(train['posts'])
Tfidf_test = Tfidf_vect.transform(test['posts'])


# In[ ]:


#It seems they have exactly the same result in this case according to the results printed out.
print(f'count: {Count_train.shape}\nCount_test: {Count_test.shape}\n\nTfidf: {Tfidf_train.shape}\nTfidf_test: {Tfidf_test.shape}')


# [Return to index](#index)
# ___________________________________________________________________________________________________________________________________________________
# <a id='8'></a>

# # Model fitting and predicting

# We are ready to fit our model. First we import our models. There are different machine learning models which we first tried (Logistic Regression, Naive Bayes(the three known), Extra-trees Classifier and Random Forest) among which the code for logistic regression is shown below. We decided to use logistic regression, because it seemed to fit our data and predict better from our kaggle submissions.

# In[ ]:


#Import libraries
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import log_loss


# In[ ]:


#We saved our id for submission purposes
subm = {}
subm['id'] = test['id'].values


# [Return to index](#index)
# ___________________________________________________________________________________________________________________________________________________
# <a id='9'></a>

# ### Model fitting and predicting for each class (Mind, Energy, Nature, and Tactics) using Logistic Regression

# # Mind

# First is mind which compares Introverted (I) and Extraverted (E). We fitted the model with out 'y' as train['Mind'] and our 'x' as Tfidf_train. We then we predicted our X_test that we also vectorised using TfidfVectorizer.

# In[ ]:


np.mean(train['Mind'] == 1)


# In[ ]:


mind = LogisticRegression(C=1, solver='lbfgs')
mind.fit(Tfidf_train, train['Mind'].values)
alert()


# In[ ]:


y_probs = mind.predict_proba(Tfidf_train)
mind_pred = mind.predict(Tfidf_train)

for thresh in np.arange(0.1, 1, 0.1).round(1):
    fiddled_y = np.where(y_probs[:,1] > thresh, 1, 0)
    print(f'the loss for {thresh} is:  {log_loss(fiddled_y, train["Mind"])}')


# In[ ]:


true_mind = np.where(mind.predict_proba(Tfidf_test)[:,1] > 0.3, 1, 0)


# In[ ]:


true_mind


# In[ ]:


subm['mind'] = true_mind


# [Return to index](#index)
# ___________________________________________________________________________________________________________________________________________________
# <a id='10'></a>

# # Energy

# Second is Energy which compares  Sensing (S) and Intuitive (N). We fitted the model with out 'y' as train['Energy'] and our 'x' as Tfidf_train. We then we predicted our X_test that we also vectorised using TfidfVectorizer.

# In[ ]:


np.mean(train['Energy'] == 1)
Energy  = LogisticRegression(C=1, solver='lbfgs')
Energy.fit(Tfidf_train,  train['Energy'])
alert()


# In[ ]:


y_probs = Energy.predict_proba(Tfidf_train)
Energy_pred = Energy.predict(Tfidf_train)

for thresh in np.arange(0.1, 1, 0.1).round(1):
    fiddled_y = np.where(y_probs[:,1] > thresh, 1, 0)
    print(f'the loss for {thresh} is:  {log_loss(fiddled_y, train["Energy"])}')


# In[ ]:


true_energy = np.where(Energy.predict_proba(Tfidf_test)[:,1] > 0.7, 1, 0)


# In[ ]:


true_energy


# In[ ]:


subm['energy'] = true_energy


# [Return to index](#index)
# ___________________________________________________________________________________________________________________________________________________
# <a id='11'></a>

# # Nature

# Third is Nature which compares Feeling (F) and Thinking (T). We fitted the model with out 'y' as train['Nature'] and our 'x' as Tfidf_train. We then we predicted our X_test that we also vectorised using TfidfVectorizer.

# In[ ]:


np.mean(train['Nature'] == 1)
Nature  = LogisticRegression(C=1, solver='lbfgs')
Nature.fit(Tfidf_train,  train['Nature'])
alert()


# In[ ]:


y_probs = Nature.predict_proba(Tfidf_train)
Nature_pred = Nature.predict(Tfidf_train)

for thresh in np.arange(0.1, 1, 0.1).round(1):
    fiddled_y = np.where(y_probs[:,1] > thresh, 1, 0)
    print(f'the loss for {thresh} is:  {log_loss(fiddled_y, train["Nature"])}')


# In[ ]:


true_nature = np.where(Nature.predict_proba(Tfidf_test)[:,1] > 0.5, 1, 0)


# In[ ]:


true_nature


# In[ ]:


subm['nature'] = true_nature


# [Return to index](#index)
# ___________________________________________________________________________________________________________________________________________________
# <a id='12'></a>

# # Tactics

# Last one is Tactics which compares Perceiving (P) and Judging (J). We fitted the model with out 'y' as train['Tactics'] and our 'x' as Tfidf_train. We then we predicted our X_test that we also vectorised using TfidfVectorizer.

# In[ ]:


np.mean(train['Tactics'] == 1)
Tactics = LogisticRegression(C=1, solver='lbfgs')
Tactics.fit(Tfidf_train, train['Tactics'])
alert()


# In[ ]:


y_probs = Tactics.predict_proba(Tfidf_train)
Tactics_pred = Tactics.predict(Tfidf_train)

for thresh in np.arange(0.1, 1, 0.1).round(1):
    fiddled_y = np.where(y_probs[:,1] > thresh, 1, 0)
    print(f'the loss for {thresh} is:  {log_loss(fiddled_y, train["Tactics"])}')


# In[ ]:


true_tactics = np.where(Tactics.predict_proba(Tfidf_test)[:,1] > 0.4, 1, 0)


# In[ ]:


true_tactics


# In[ ]:


subm['tactics'] = true_tactics


# [Return to index](#index)
# ___________________________________________________________________________________________________________________________________________________
# <a id='13'></a>

# # Prepared submission

# In[ ]:


submit = pd.DataFrame(subm)


# In[ ]:


submit.sample(10)


# In[ ]:


submit.to_csv('kaggle submit.csv', index=False)


# [Return to index](#index)
# ___________________________________________________________________________________________________________________________________________________
# <a id='14'></a>

# # Conclusion
# Text data comes in many different forms depending on the source. The data we used are twitter posts which do not come with translation of any videos or images shared. We had to clean our data in a way that saves some of those messages we can't see just from the text. We saw that in our model improved a lot after we have replaced the urls with title of their videos from youtube.
# 
# This model present an alternative way of getting mbti personality type. Also with the method described in this notebook, one does not need to wait for somebody to take the test but need their social media posts or words they normally say to people. It makes it easier to get someone else's personality type. This can be very useful for companies that would like to pick people according to their personality type, people looking to interact with certain personality type and more.
# 
# There are different machine learning models which we first tried (Logistic Regression, Naive Bayes, Extra-trees Classifier and Random Forest) among which the code for logistic regression is shown above. We decided to use logistic regression, because it seemed to fit our data and predict better score from our kaggle submissions.The binary classification exercise to predict each of the classes in the four axes (mind, energy, nature, and tactics) was somewhat more successful. A Logistic Regression was used in each of the classes.
# 
# We still have challenges of reading people's tone and level of english and their writting skills. This factors can play a big role in our model predicting the exact personality type. The fact that another forum or platform can have a different way of writting can also mean that the model will have to be trained and fitted again for that specific platform. 

# # Still to do

# * Attempt upscaling to help with the skewness of the data
# * Weighting the response of personality traits based on other traits
# * Apply GridsearchCV to my models

# [Return to index](#index)
# ___________________________________________________________________________________________________________________________________________________
# <a id='15'></a>

# # Acknowledgements

# 1. A large part of the code and idea to fiddle with the logistic regression thresholds was inspired by the **advanced logistic regression** train and **Nicholas Meyers** 
# 2. EDA/Feature engineering (to keep certain stopwords) was largely inspired by the **How do machines understand language** train
# 3. Most preprocessing functions were largely inspired by the **How do machines understand language** train
# 4. EDSA supervisors **Bryan Davies, Tristan Naidoo**
# 5. Kaggle dataset from [Personality Cafe website forums](https://www.personalitycafe.com/forum/)

# [Return to index](#index)
# ___________________________________________________________________________________________________________________________________________________
