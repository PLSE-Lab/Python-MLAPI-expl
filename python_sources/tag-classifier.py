#!/usr/bin/env python
# coding: utf-8

# # Stack Exchange - Tag Classifier

# > **Note:** Main focus of this kernel is not objective. Instead of focusing on getting as high accuracy score as possible, we will focus on real-world scenarios, ML/DS concepts

# ## 01 - Data - High Level Overview
# 
# Dataset contains content from **disparate(different) stack exchange sites**, containing a **mix of both technical and non-technical** questions.
# 
# | Feature | Desc |
# | --- | --- |
# | Id | Unique identifier for each question |
# | Title | The question's title |
# | Body | The body of the question |
# | Tags | The tags associated with the question (all lowercase, should not contain tabs '\t' or ampersands '&') |
# 
# 
# <br><br>
# The questions are **randomized** and contains a mix of verbose **text sites as well as sites related to math and programming**. The number of **questions from each site may vary**, and **no filtering** has been performed on the questions (such as closed questions).
# 
# **SIZE:** ~3GB
# 
# 
# | Source | Inference |
# | --- | --- |
# | "disparate(different) stack exchange sites" | Question are from different domains |
# |"questions from each site may vary"| Chance of underfitting/overfitting |
# | "both technical and non-technical" "text sites as well as sites related to math and programming" | Question text may be numbers/words/symbols |
# |"questions are **randomized**"| End to end solution is needed |
# | "no filtering" | -- |

# ## 02 - Objective / Problem Statement
# 
# Predict **tags** based on input question's **Title** and **Body**
# 
# Problem Type: `Classification`

# ## 03 - Real world use case constraints
# 
# | Constraint | Required / Not Required | Comment |
# | --- | --- | --- |
# | High Precision | required | Impact on UX | 
# | High Recall | required | Impact on UX | 
# | Low Latency | not requred | No significant impact on UX |
# | High Interpretability | required | Impact on UX |
# 
# 
# <br><br><br>
# If you forgot about precision and recall: 
# 
# > When a search engine returns 30 pages only 20 of which were relevant while failing to return 40 additional relevant pages, its precision is 20/30 = 2/3 while its recall is 20/60 = 1/3. So, in this case, precision is **"how useful the search results are"**, and recall is **"how complete the results are".** [wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall)

# ## 04 - Mapping Real World Problem to Machine Learning Problem
# 
# 1. Type of ML Problem
# 2. Key Performance Indicatiors
# 
# ### Type of ML problem
# 
# - Classification Problem (Not multi-class)
# - Multi-Label Classification [sci-kit docs](https://scikit-learn.org/stable/modules/multiclass.html)
# 
# ### Key Performance Indicatiors
# 
# Not binary classification.
# 
# - F1 Score
# - Micro F1 Score
# - Macro F1 Score
# - Hamming Loss
# 
# ### i. F1-Score
# 
# Wighted average of precision and recall. When all weights are 1 $\implies$ Harmonic mean $\implies \frac{\sum{w_i}}{\frac{w_i}{x_i}} $
# 
# $$F1 = \frac{2PR}{P + R}$$
# 
# where  $p=\frac{tp}{tp+fp}$,  $r=\frac{tp}{tp+fn}$
# 
# > It combines power of both P and R
# 
# ### ii. Macro F1 Score 
# 
# Simple average of F1 Score of each class
# 
# $$ F1_{macro} = \frac{1}{n_{classes}} \sum_{k=1}^{n_{classes}} \text{F1}_{k} $$ 
# 
# > Doesn't take class imbalance into account
# 
# 
# ### iii. Micro F1 Score
# 
# $$ F1_{micro} = \frac{2 P_{micro} R_{micro}} {P_{micro} + R_{micro}} $$
# 
# where  $p_{micro}= \sum_{k \in C} \frac{tp_{k}}{tp_{k}+fp_{k}}$ and $r_{micro}= \sum_{k \in C} \frac{tp_{k}}{tp_{k}+fn_{k}}$ 
# 
# - We are calculating $P$ and $R$ from all classes and using it in $F1-Score$ formula
# 
# - **Note:** F1-Score can be high even when $P$ and $R$ of minority class is very small **WHEN** $P$ and $R$ of majority class is very high 
# 
# > It is sort of weighted average so, takes class imbalance into account (both numerator and denominator of $P_{micro}$ $R_{micro}$ will incr/decr with imbalance)
# 
# ### iv. Hamming Loss
# 
# $$ Hamming Loss(\hat{y_{i}}, y_{i}) = \frac{1}{N_{samples}} \sum_{i=1}^{N_{samples}} \frac{xor(\hat{y_{i}}, y_{i})}{N_{labels}} $$
# 
# where $\hat{y_{i}}, y_{i}$ are encoded vectors vectors

# ## A. EDA - Preliminary Analysis

# In[ ]:


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("/kaggle/input/facebook-recruiting-iii-keyword-extraction/Train.zip")
df.head()


# In[ ]:


df.info()


# **FEATURES DESC**
# 
# |Name|Desc|Data Type|
# | --- | --- | --- |
# | Id | Unique |Continous, Numerical |
# | Title | ascii | Mixed |
# | Body  | ascii | Mixed |
# | Tags | Target | Categorical, Multi-Label |
# 
# **Number of samples:** 6,034,195 (6M) <bR>
# **Number of feats:** 4

# ### Data Cleaning - Removing Duplicates

# In[ ]:


# get duplicates
df_dups = df[df.duplicated(['Title', 'Body', 'Tags'])]
print('Total Duplicates: ', len(df_dups))
print('ratio: ', len(df_dups)/len(df))


# In[ ]:


# remove duplicates
df = df.drop_duplicates(['Title', 'Body', 'Tags'])
print('After removing dups: ', len(df))
print('ratio: ', len(df)/6034194)


# ### Analysis of Target Varible - Tags

# **a. distribution of number of tags per qn**

# In[ ]:


x = df["Tags"].apply(lambda x: type(x)==float)
x[x==True]


# In[ ]:


# removing `Tags` which are float instead of str
df.drop([err_idx for err_idx in x[x==True].index], inplace=True)


# In[ ]:


df["num_of_tags"] = df["Tags"].apply(lambda x: len(x.split(" ")))
df['num_of_tags'].value_counts()


# In[ ]:


plt.close()

plt.bar(
    df.num_of_tags.value_counts().index,
    df.num_of_tags.value_counts()
)

plt.xlabel('Number of tags')
plt.ylabel('Freq (x10^6)')
plt.show()


# **Observation and Inference:**
#     
# - Maximum 5 tags per question averaging 3 tags per question

# **b. unique tags**

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

# get unique tags w/ help of BoW. Tags are space separated
vectorizer = CountVectorizer(tokenizer = lambda x: x.split())

# fit_transform
# - learn the vocabulary and store in `vectorizer`
# - convert training data into feature vectors
#    - converts each input (tag) into one hot encoded based on vocab
tag_vecs = vectorizer.fit_transform(df['Tags'])


# In[ ]:


# learnt vocabulary
vocab = vectorizer.get_feature_names()
print(vocab[:5])

# total vocabulary
print('Total vocabulary: ', len(vocab))


# In[ ]:


# one hot encoded training data
print('Num of samples: ', tag_vecs.shape[0])
print('Size of one hot encoded vec (each val represents a tag): ', tag_vecs.shape[1])


# In[ ]:


# distribution of unique tags
freq_of_tags = tag_vecs.sum(axis=0).getA1() # (1, vocab_size) -> (vocab_size) i.e flatten it
tags = vocab

tag_freq = zip(tags[:5], freq_of_tags[:5])

for tag, freq in tag_freq:
    print(tag, ':', freq)


# In[ ]:


sorted_idxs = np.argsort(- freq_of_tags) # -1: descending

sorted_freqs = freq_of_tags[sorted_idxs] 
sorted_tags  = np.array(tags)[sorted_idxs]

for tag, freq in zip(sorted_tags[:5], sorted_freqs[:5]):
    print(tag, ':', freq)


# In[ ]:


# distribution of occurances
plt.close()

plt.plot(sorted_freqs)

plt.title("Distribution of number of times tag appeared questions\n")
plt.grid()
plt.xlabel("Tag idx in vocabulary")
plt.ylabel("Number of times tag appeared")
plt.show()


# In[ ]:


# zoom in first 1k
plt.close()

plt.plot(sorted_freqs[:1000])

plt.title("Distribution of number of times tag appeared questions\n")
plt.grid()
plt.xlabel("Tag idx in vocabulary")
plt.ylabel("Number of times tag appeared")
plt.show()


# In[ ]:


# zoom in first 200
plt.close()

plt.plot(sorted_freqs[:200])

plt.title("Distribution of number of times tag appeared questions\n")
plt.grid()
plt.xlabel("Tag idx in vocabulary")
plt.ylabel("Number of times tag appeared")
plt.show()


# In[ ]:


# zoom in first 100
plt.close()

# quantiles with 0.05 difference
plt.scatter(x=list(range(0,100,5)), y=sorted_freqs[0:100:5], c='orange', label="quantiles with 0.05 intervals")
# quantiles with 0.25 difference
plt.scatter(x=list(range(0,100,25)), y=sorted_freqs[0:100:25], c='m', label = "quantiles with 0.25 intervals")

for x,y in zip(list(range(0,100,25)), sorted_freqs[0:100:25]):
    plt.annotate(s="({} , {})".format(x,y), xy=(x,y), xytext=(x-0.05, y+500))
    
#for x,y in zip(list(range(0,100,5)), sorted_freqs[0:100:5]):
#    plt.annotate(s="({} , {})".format(x,y), xy=(x,y), xytext=(x-0.05, y+500))

x=100
y=sorted_freqs[100]
plt.annotate(s="({} , {})".format(x,y), xy=(x,y), xytext=(x-0.05, y+500))


plt.plot(sorted_freqs[:100])

plt.legend()
plt.grid()

plt.title("Distribution of top 100 tags\n")
plt.xlabel("Tag idx in vocabulary")
plt.ylabel("Number of times tag appeared")
plt.show()


# ---------------------------------------------------------------------
# PDF AND CDF
plt.close()
plt.figure(figsize=(10,10))

plt.subplot(211)
counts, bin_edges = np.histogram(sorted_freqs, bins=100, 
                                 density = True)
pdf = counts/(sum(counts))
#print(pdf);
#print(bin_edges)
cdf = np.cumsum(pdf)

plt.title("CDF all tags\n")
plt.xlabel("Freq of tag occurances")
plt.ylabel("Percent of Tags out of all tags")
plt.grid()

plt.plot(bin_edges[1:], cdf)

# -------------
plt.subplot(212)
counts, bin_edges = np.histogram(sorted_freqs[:100], bins=100, 
                                 density = True)
pdf = counts/(sum(counts))
#print(pdf);
#print(bin_edges)
cdf = np.cumsum(pdf)

#plt.title("CDF top 100 tags\n")
plt.xlabel("Freq of to 100 tag occurances")
plt.ylabel("Percent of Tags ut of 100 tags")
plt.grid()

plt.plot(bin_edges[1:], cdf)

plt.show()


# | | **0BSERVATION** | **INFERENCE** |
# |---|---|---|
# |1| Top 100 tags appear atleast 13k times. Of which top 15 tags appear 100k times | Most of our tags i.e 420,6207 tags occur less than 13k(max value) and only 15 tags occur more than 100k times. (Huge difference). <br> We may easily overfit on top 15 tags(Analyze these tags) |
# |2| 98.8% of all tags' frequencies occur insignificantly  | Only 2.2% of tags occur more frequently <br> Highly imbalanced. Micro F1 might be good choice but if model predicts 2.2% with high precision and high recall, we wont be able to handle class imbalance |

# In[ ]:


# visulaize all tags wrt their frequencies
from wordcloud import WordCloud

# input is (tag, fre) tuple
tup = dict(zip(sorted_tags, sorted_freqs))

#Initializing WordCloud using frequencies of tags.
wordcloud = WordCloud(    background_color='black',
                          width=1600,
                          height=800,
                    ).generate_from_frequencies(tup)

fig = plt.figure(figsize=(30,20))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
fig.savefig("tag.png")
plt.show()


# **Observation:**
# 
# - Most of questions are from CS/IT domain
# - Common tags include
#     - Java
#     - C#
#     - PHP
#     - Javascript
#     - Android
#     - python
#     - JQuery
#     - ASP.NET
# 
# 
# **Inference:**
# 
# - Most of frequent tags are *progamming languages* 

# **Analyze Top 20 Tags (Occur more than 50k times each)**

# In[ ]:


plt.close()
plt.figure(figsize=(20, 5))

plt.bar(sorted_tags[:20], sorted_freqs[:20])

plt.xlabel('Top 20 Tags')
plt.ylabel('Counts')
plt.show()


# - Most common tags are either
#     1. Programming langs: c#, java, php, js etc.
#     2. OS: android, iphone, ios, linux (windows exists in word cloud)
# 
# - No significance shown for other non-tech domains
# - With time, popularity of programming langs may change

# ### Analyzing Title/Body

# In[ ]:


# random 10 titles
for i in np.random.choice(len(df), 10):
    print(df['Title'].iloc[i])


# - As we have already removed duplicates,looks good. Nothing much to preprocess
# - Simple stopword removal, stemming etc. will suffice

# In[ ]:


# random body
NUM = 5
for i in np.random.choice(len(df), NUM):
    print(df['Body'].iloc[i])
    print("="*100)


# |OSERVATION|INFERENCE|
# |---|---|
# |Bodies are html based | Need to remove tags, anchor links etc. |
# |Code is present in `<code>` tag | Create new col for code for two reasons - <br> 1. Obviously to featurize code/desc better way. For example we can remove special chars in desc without impacting code input<br> 2. Importantly to distinguish between technical/non-tech |

# In[ ]:


# let's see how many questions have code and will it be useful for distinguishing tech/non-tech
cntr = 0
for body in df['Body']:
    if '<code>' in body:
        cntr += 1
print(f'Total entries with code: {cntr} (ratio: {cntr/len(df)})')


# |OBSERVATION | INFERENCE|
# |---|---|
# |58% of bodies have code| Doesn't help much for differentiating non-tech/tech as even tech questions can sometime not have code |

# In[ ]:


df.drop(['Id'], axis=1, inplace=True)
df.head()


# In[ ]:


# create empty cols for new feats
df['question(title+body)'] = df['code'] = df['len_question_before_processing'] = df['len_after_before_processing'] = df['is_code_present'] = df['num_of_tags'].apply(lambda x: '')


# In[ ]:


df.head(3)


# **Remove obsolete feats and add new preprocessed ones**

# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import re

def striphtml(data):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(data))
    return cleantext

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

preprocessed_data_list=[]
questions_with_code = 0
len_pre = 0
len_post = 0
questions_proccesed = 0

# iterate through each row and
# remove obsolete feats and add new
for i in range(len(df)):
    
    title, question, tags = df['Title'].iloc[i], df['Body'].iloc[i], df['Tags'].iloc[i]
    # remove obsolete feats (replace with empty space for now, drop whole col later)
    # replacing with empty '' to save memory
    df['Title'].iloc[i] = df['Body'].iloc[i] = df['Tags'].iloc[i] = ''
    
    is_code = 0
    if '<code>' in question:
        questions_with_code+=1
        is_code = 1
    
    x = len(question)+len(title)
    len_pre+=x
    
    # all code separated
    code = str(re.findall(r'<code>(.*?)</code>', question, flags=re.DOTALL))
    
    # code, html removed
    question=re.sub('<code>(.*?)</code>', '', question, flags=re.MULTILINE|re.DOTALL)
    question=striphtml(question.encode('utf-8'))
    
    # filter unwanted (recommended for safety)
    title=title.encode('utf-8')

    question=str(title)+" "+str(question) # combine title and question
    question=re.sub(r'[^A-Za-z]+',' ',question) # remove spl. chars
    words=word_tokenize(str(question.lower())) # toenize
    
    # Removing all single letter and and stopwords from question except for the letter 'c'
    question=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))

    len_post+=len(question)
    
    # feats: 
    # question(title+body), 
    # code, 
    # tags, 
    # len_question_before_processing, 
    # len_after_before_processing,
    # is_code_present
    df['question(title+body)'].iloc[i]               = question
    df['code'].iloc[i]                               = code
    df['len_question_before_processing'].iloc[i]     = tags
    df['len_after_before_processing'].iloc[i]        = x 
    df['is_code_present'].iloc[i]                    = is_code
        
    questions_proccesed += 1
    
no_dup_avg_len_pre=(len_pre*1.0)/questions_proccesed
no_dup_avg_len_post=(len_post*1.0)/questions_proccesed

print( "Avg. length of questions(Title+Body) before processing: %d"%no_dup_avg_len_pre)
print( "Avg. length of questions(Title+Body) after processing: %d"%no_dup_avg_len_post)
print ("Percent of questions containing code: %d"%((questions_with_code*100.0)/questions_proccesed))


# In[ ]:


# remove old obsolete feats
df.drop(['Title', 'Body', 'Tags'], axis=1, inplace=True)
df.head()

