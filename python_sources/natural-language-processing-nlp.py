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


# From Scratch


# In[ ]:


import nltk


# In[ ]:


dir(nltk)


# In[ ]:


from nltk.corpus import stopwords
stopwords.words('english')[0:500:25]


# ### Reading in text data and why we need to clean it

# In[ ]:


rawData=open('/kaggle/input/nlp-data-set/SMSSpamCollection.tsv').read() 
rawData[0:500]


# In[ ]:


parsedData=rawData.replace("\t", "\n").split("\n")
parsedData[0:5]


# In[ ]:


label_list=parsedData[0::2]
text_list=parsedData[1::2]


# In[ ]:


print(label_list[0:5])
print(text_list[0:5])


# In[ ]:


print(len(label_list))
print(len(text_list))


# In[ ]:


print(label_list[-5:])


# In[ ]:


fullCorpus=pd.DataFrame({'label' : label_list[:-1], 'body_list' : text_list})
fullCorpus.head()


# # Short cut we can use instead of the above method

# In[ ]:


dataset=pd.read_csv('/kaggle/input/nlp-data-set/SMSSpamCollection.tsv' , sep='\t', header = None)
dataset.head()


# # Exploring Dataset

# In[ ]:


dataset.columns=['label', 'body_list']
dataset.head()


# In[ ]:


# shape #5568
print("Input data has {} rows and {} columns".format(len(fullCorpus), len(fullCorpus.columns)))


# In[ ]:


# how many ham and spam
print("Out of the {} rows, {} are spam, {} are ham". format(len(fullCorpus), 
                                                            len(fullCorpus[fullCorpus['label']=='spam']),
                                                            len(fullCorpus[fullCorpus['label']=='ham'])))


# In[ ]:


# missing data
print("Number of missing label {}".format(fullCorpus['label'].isnull().sum()))
print("Number of missing text {}".format(fullCorpus['body_list'].isnull().sum()))


# # Regular Expression
# * Text string for describing a search pattern
# 
# ## Uses
# * Identifying white spaces between words and token
# * Identifying /creating delimiters or end-of-line escape characters
# * removing punctuations or numbers from your text
# * cleaning html tags from your text
# * Identifying some textual patterns you are interested in
# 
# ## Application
# * Confirming password meets criteria
# * searching url for substring 
# * searching for files on your computer
# * Document scrapping

# # Learnig how to use regular Expression

# In[ ]:


import re


# In[ ]:


re_test = 'This is a made up string to test 2 different regex method'
re_test_messy =  'This is     a made up      string to test 2       different regex method'
re_test_messy1 = 'This-is-a-made/up.string*to>>>>test----2""""""different-regex-method'


# In[ ]:


# splitting a sentence into a list of words
# 1st method
re.split('\s', re_test)


# In[ ]:


re.split('\s', re_test_messy)


# In[ ]:


re.split('\s+', re_test_messy)


# In[ ]:


re.split('\s+', re_test_messy1)


# In[ ]:


re.split('\W+', re_test_messy1)


# In[ ]:


# Second method
re.findall('\S+', re_test_messy)


# In[ ]:


re.findall('\S+', re_test)


# In[ ]:


re.findall('\S+', re_test_messy1)


# In[ ]:


re.findall('\w+', re_test_messy1)


# # Replacing a specific string

# In[ ]:


pep8_test = 'I try to follow PEP8 guidelines'
pep7_test = 'I try to follow PEP7 guidelines'
peep8_test = 'I try to follow PEEP8 guidelines'


# In[ ]:


re.findall('[a-z]+' , pep8_test )


# In[ ]:


re.findall('[A-Z]+' , pep8_test )


# In[ ]:


re.findall('[A-Z0-9]+' , pep8_test )


# In[ ]:


re.findall('[A-Z]+[0-9]+' , pep8_test )


# In[ ]:


re.findall('[A-Z]+[0-9]+' , pep7_test )


# In[ ]:


re.findall('[A-Z]+[0-9]+' , peep8_test )


# In[ ]:


re.sub('[A-Z]+[0-9]+', 'PEP8 Python Styleguide',pep8_test )


# In[ ]:


re.sub('[A-Z]+[0-9]+', 'PEP8 Python Styleguide',pep7_test )


# In[ ]:


re.sub('[A-Z]+[0-9]+', 'PEP8 Python Styleguide',peep8_test )


# # Other Regex methods
# * re.search()
# * re.match()
# * re.fullmatch()
# * re.finditer()
# * re.escape()

# # Machine Learning Pipeline
# 
# 1. Raw Text - model can't distingush words
# 2. Tokenize - tell the model what to look at
# 3. clean text - remove stop words / punctuation, stemming, etc
# 4. vectorize - convert to numeric form
# 5. Spam filter - system to filter emails

# # Implementation: Removing Punctuation
# 
# ** preprocessing data
# 1. Remove Punctuation
# 2. Tokenization
# 3. Removing stopwords
# 4. Lemmatize/Stem

# In[ ]:


pd.set_option('display.max_colwidth', 100)
data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)
data.columns = ['label', 'body_text']
data.head()


# In[ ]:


# How does cleaned up version look like
data_cleaned=pd.read_csv("/kaggle/input/cleaned-data/SMSSpamCollection_cleaned.tsv" , sep='\t')
data_cleaned


# # Remove Punctuation

# In[ ]:


import string
string.punctuation


# In[ ]:


"I like NLP." == "I like NLP"


# In[ ]:


def remove_punc(text):
    text_no_punc = [char for char in text if char not in string.punctuation]
    return text_no_punc
data['body_text_clean']=data['body_text'].apply(lambda x : remove_punc(x))
data.head()


# In[ ]:


def remove_punc(text):
    text_no_punc = "".join([char for char in text if char not in string.punctuation])
    return text_no_punc
data['body_text_clean']=data['body_text'].apply(lambda x : remove_punc(x))
data.head()


# # Implementation of Tokenization

# In[ ]:


def tokenize(text):
    tokens = re.split("\W+", text)
    return tokens
data['body_text_tokenize']=data['body_text_clean'].apply(lambda x : tokenize(x.lower()))
data.head()


# In[ ]:


'NLP'== 'nlp'


# # Remove StopWords

# In[ ]:


stopword = nltk.corpus.stopwords.words('english')


# In[ ]:


def remove_stopwords(tokenized_list):
    text = [ word for word in tokenized_list if word not in stopword]
    return text
data['body_text_nostop']=data['body_text_tokenize'].apply(lambda x : remove_stopwords(x))
data.head()


# # Supplemental Data cleaning
# ### Stemming - process of removing inflected(or sometimes derived) words to their word stem or root 
# ### Crudely chopping off the end of the word to leave only the base
# * example - Berries/berry = Berri
# 
# ## Types
# 
# * Porter Stemmer 
# * Snowball Stemmer 
# * Lancaster Stemmer
# * Regex-Based Stemmer

# In[ ]:


ps=nltk.PorterStemmer()


# In[ ]:


dir(ps)


# In[ ]:


print(ps.stem('grows'))
print(ps.stem('growing'))
print(ps.stem('grow'))


# In[ ]:


print(ps.stem('run'))
print(ps.stem('running'))
print(ps.stem('runner'))


# In[ ]:


#import re
#import string
pd.set_option('display.max_colwidth', 100)

stopwords=nltk.corpus.stopwords.words('english')
data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)
data.columns = ['label', 'body_text']
data.head()


# In[ ]:


def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    tokens=re.split('\W+', text)
    text = [word for word in tokens if word not in stopwords]
    return text
data['body_text_nostop']=data['body_text'].apply(lambda x : clean_text(x.lower()))
data.head()


# In[ ]:


def stemming(tokenized_text):
    text = [ps.stem(word) for word in tokenized_text]
    return text
data['body_text_stemmed']=data['body_text_nostop'].apply(lambda x : stemming(x))
data.head()


# # Lemmatizing 
# ### Process of grouping together the inflected forms of a word so they can be analyzed as a single term, identified by the word's lemma
# 
# ### Using vocabulary analysis of words aiming to remove inflectional ending to return the dictionary for of  a word  

# In[ ]:


#ps=nltk.PorterStemmer()
wn=nltk.WordNetLemmatizer()
#import re
#import string
#stopwords=nltk.corpus.stopwords.words('english')


# In[ ]:


dir(wn)


# In[ ]:


print(ps.stem('meanness'))
print(ps.stem('meaning'))


# In[ ]:


print(wn.lemmatize('meanness'))
print(wn.lemmatize('meaning'))


# In[ ]:


print(ps.stem('goose'))
print(ps.stem('geese'))


# In[ ]:


print(wn.lemmatize('goose'))
print(wn.lemmatize('geese'))


# # Raw Data

# In[ ]:


data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)
data.columns = ['label', 'body_text']
data.head()


# In[ ]:


def clean_text(text):
    text = "".join([word for word in text if word not in string.punctuation])
    tokens=re.split('\W+', text)
    text = [word for word in tokens if word not in stopwords]
    return text
data['body_text_nostop']=data['body_text'].apply(lambda x : clean_text(x.lower()))
data.head()


# # Lemmatize

# In[ ]:


def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text
data['body_text_lemmatized']=data['body_text_nostop'].apply(lambda x : lemmatizing(x))
data.head()


# # Vectorizing Raw Data - process of encoding text as integers to create feature vectors

# # Process
# ### Raw Text - model can't distingush words
# ### Tokenize - tell the model what to look for
# ### Clean text - remove stop words, punctuations, stemming, etc
# ### vectorize - convert to numeric form
# ### machine learning algorithm - fit/train model
# ### spam filter - systems to filter mail

# Vector Types = 1] Count vectorization, 2] N-grams, 3] Term frequency - numeric document frequency (TF-IDF)

# In[ ]:


# import string
# import re
#import nltk
pd.set_option('display.max_colwidth', 100)

stopwords=nltk.corpus.stopwords.words('english')
ps=nltk.PorterStemmer()
    
data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)
data.columns = ['label', 'body_text']
data.head()


#  ### create function to remove punctuation, tokenize, remove stopwords and stem

# In[ ]:


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens=re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text
#data['body_text_nostop']=data['body_text'].apply(lambda x : clean_text(x.lower()))
#data.head()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(analyzer = clean_text)
x_counts=count_vect.fit_transform(data['body_text'])
print(x_counts.shape)
print(count_vect.get_feature_names())


# ### The above data is quite vast so we will apply it on smaller sample

# In[ ]:


data_sample=data[0:20]
count_vect_sample = CountVectorizer(analyzer = clean_text)
x_counts_sample=count_vect_sample.fit_transform(data_sample['body_text'])
print(x_counts_sample.shape)
print(count_vect_sample.get_feature_names())


# # Sparse Matrix
# #### A matrix in which most entries are zero. in the interest of efficient storage, a sparse matrix will be stored by only storing locations of the non-zero elements.

# In[ ]:


x_counts_sample


# In[ ]:


df=pd.DataFrame(x_counts_sample.toarray())
df.head()


# In[ ]:


df.columns = count_vect_sample.get_feature_names()
df.head()


# # N-Gram Vectorizing
# ### creates a document-term matrix where counts still occupy the cell but instead of the columns representing single terms, they represent all combination of adjacent words of length n in your text.

# In[ ]:


# import string
# import re
#import nltk
pd.set_option('display.max_colwidth', 100)

stopwords=nltk.corpus.stopwords.words('english')
ps=nltk.PorterStemmer()
    
data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)
data.columns = ['label', 'body_text']
data.head()


# In[ ]:


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens=re.split('\W+', text)
    text = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
    return text
data['cleaned_text']=data['body_text'].apply(lambda x : clean_text(x))
data.head()


# # Apply CounterVectorizer

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
ngram_vect = CountVectorizer(ngram_range=(2,2)) # 1,2,3 = unigram, bigram, trigram
x_counts=ngram_vect.fit_transform(data['cleaned_text'])
print(x_counts.shape)
print(ngram_vect.get_feature_names())


# # Apply to small sample

# In[ ]:


data_sample= data[0:20]
ngram_vect_sample = CountVectorizer(ngram_range=(2,2)) # 1,2,3 = unigram, bigram, trigram
x_counts_sample=ngram_vect_sample.fit_transform(data_sample['cleaned_text'])
print(x_counts_sample.shape)
print(ngram_vect_sample.get_feature_names())


# In[ ]:


df=pd.DataFrame(x_counts_sample.toarray())
df.columns = ngram_vect_sample.get_feature_names()
df.head()


# # Inverse Document Frequency Weighting (TF-IDF Equation)

# In[ ]:


# import string
# import re
#import nltk
pd.set_option('display.max_colwidth', 100)

stopwords=nltk.corpus.stopwords.words('english')
ps=nltk.PorterStemmer()
    
data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)
data.columns = ['label', 'body_text']
data.head() 


# In[ ]:


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens=re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text


# # Apply Tfidfvectorizer

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect=TfidfVectorizer(analyzer=clean_text)
x_tfidf=tfidf_vect.fit_transform(data['body_text'])
print(x_tfidf.shape)
print(tfidf_vect.get_feature_names())


# In[ ]:


data_sample= data[0:20]
tfidf_vect_sample=TfidfVectorizer(analyzer=clean_text)
x_tfidf_sample=tfidf_vect_sample.fit_transform(data_sample['body_text'])
print(x_tfidf_sample.shape)
print(tfidf_vect_sample.get_feature_names())


# In[ ]:


df=pd.DataFrame(x_tfidf_sample.toarray())
df.columns = tfidf_vect_sample.get_feature_names()
df.head()


# # Feature Engineering
# 
# ### Feature Creation

# In[ ]:


data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)
data.columns = ['label', 'body_text']
data.head()


# In[ ]:


# Create feature for text message length
data['body_len'] = data['body_text'].apply(lambda x : len(x) - x.count(" "))
data.head()


# In[ ]:


# Create feature for % of text that is  punctuation
import string
def count_punc(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100
data['punc%']= data['body_text'].apply(lambda x : count_punc(x))
data.head()


# In[ ]:


# Evalute new features
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


bins= np.linspace(0,200,40)
plt.hist(data[data['label'] == 'spam']['body_len'], bins, alpha=0.5, normed=True, label='spam')
plt.hist(data[data['label'] == 'ham']['body_len'], bins, alpha=0.5, normed=True, label='ham')
plt.legend(loc='best')
plt.show()


# In[ ]:


bins= np.linspace(0,50,40)
plt.hist(data[data['label'] == 'spam']['punc%'], bins, alpha=0.5, normed=True, label='spam')
plt.hist(data[data['label'] == 'ham']['punc%'], bins, alpha=0.5, normed=True, label='ham')
plt.legend(loc='best')
plt.show()


# # Identifying features for transformation

# In[ ]:


bins= np.linspace(0,200,40)
plt.hist(data['body_len'], bins)
plt.title('Body Length Distribution')
plt.show()


# In[ ]:


bins= np.linspace(0,50,40)
plt.hist(data['punc%'], bins)
plt.title('Punctuation Length Distribution')
plt.show()


# # Box Cox Power Transformation

# In[ ]:


for i in [1,2,3,4,5]:
    plt.hist((data['punc%']) ** (1/i), bins=40)
    plt.title('Transformation : 1/{}'.format(str(i)))
    plt.show()


#  # Machine Learning
#  
#  ## Random Forest Model

# In[ ]:


import nltk
import re 
import string
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords= nltk.corpus.stopwords.words('english')
data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)
data.columns = ['label', 'body_text']

def count_punc(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100

data['body_len']= data['body_text'].apply(lambda x : len(x) - x.count(" "))
data['punc%']= data['body_text'].apply(lambda x : count_punc(x))


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens= re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
x_tfidf = tfidf_vect.fit_transform(data['body_text'])

x_features = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_tfidf.toarray())], axis =1)
x_features.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


print(dir(RandomForestClassifier))
print(RandomForestClassifier())


# In[ ]:


from sklearn.model_selection import KFold, cross_val_score


# In[ ]:


rf = RandomForestClassifier(n_jobs=-1) #n_jobs will execute all the decesion tree parallel
K_Fold = KFold(n_splits=5)
cross_val_score(rf, x_features, data['label'], cv=K_Fold, scoring = 'accuracy', n_jobs=-1)


# # Explore Random Forest Hold Out Test Set

# In[ ]:


import nltk
import re 
import string
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords= nltk.corpus.stopwords.words('english')
data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)
data.columns = ['label', 'body_text']

def count_punc(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100

data['body_len']= data['body_text'].apply(lambda x : len(x) - x.count(" "))
data['punc%']= data['body_text'].apply(lambda x : count_punc(x))


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens= re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
x_tfidf = tfidf_vect.fit_transform(data['body_text'])

x_features = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_tfidf.toarray())], axis =1)
x_features.head()


# In[ ]:


from sklearn.metrics import precision_recall_fscore_support as score 
from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_features, data['label'], test_size = 0.2)


# In[ ]:


rf=RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1)
rf_model = rf.fit(x_train, y_train)


# In[ ]:


sorted(zip(rf_model.feature_importances_, x_train.columns), reverse=True)[0:10]


# In[ ]:


y_pred=rf_model.predict(x_test)
precision, recall, fscore, support = score(y_test, y_pred, pos_label = 'spam', average='binary')


# In[ ]:


print('precision: {} / recall: {} / accuracy: {}'. format(round(precision, 3), round(recall, 3), 
                                                         round((y_pred==y_test).sum() / len(y_pred),3)))


# # Grid Search

# In[ ]:


import nltk
import re 
import string
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords= nltk.corpus.stopwords.words('english')
data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)
data.columns = ['label', 'body_text']

def count_punc(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100

data['body_len']= data['body_text'].apply(lambda x : len(x) - x.count(" "))
data['punc%']= data['body_text'].apply(lambda x : count_punc(x))


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens= re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
x_tfidf = tfidf_vect.fit_transform(data['body_text'])

x_features = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_tfidf.toarray())], axis =1)
x_features.head()


# In[ ]:


from sklearn.metrics import precision_recall_fscore_support as score 
from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x_features, data['label'], test_size = 0.2)


# In[ ]:


def train_RF(n_est, depth):
    rf=RandomForestClassifier(n_estimators=n_est, max_depth=depth, n_jobs=-1)
    rf_model=rf.fit(x_train, y_train)
    y_pred=rf_model.predict(x_test)
    precision, recall, fscore, support = score(y_test, y_pred, pos_label='spam', average='binary')
    print('Est: {} / Depth: {} ---- Precision : {} / Recall : {} / Accuracy : {}'.format(
         n_est, depth, round(precision,3), round(recall, 3), round((y_pred==y_test).sum() / len(y_pred),3)))


# In[ ]:


for n_est in [10, 50, 100]:
    for depth in [10 , 20 , 30, None]:
        train_RF(n_est, depth)


# # GridSearch and cross-validation

# In[ ]:


import nltk
import re 
import string
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords= nltk.corpus.stopwords.words('english')
data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)
data.columns = ['label', 'body_text']

def count_punc(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100

data['body_len']= data['body_text'].apply(lambda x : len(x) - x.count(" "))
data['punc%']= data['body_text'].apply(lambda x : count_punc(x))


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens= re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
x_tfidf = tfidf_vect.fit_transform(data['body_text'])
x_tfidf_feat = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_tfidf.toarray())], axis =1)

count_vect = CountVectorizer(analyzer=clean_text)
x_count = count_vect.fit_transform(data['body_text'])
x_count_feat = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_count.toarray())], axis =1)

x_count_feat.head()


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


rf=RandomForestClassifier()
param = {'n_estimators' : [10, 150, 130],
        'max_depth' : [30, 60, 90 , None ] }
gs =GridSearchCV(rf, param, cv=5, n_jobs = -1 )
gs_fit=gs.fit(x_tfidf_feat, data['label'])
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]
    


# In[ ]:


rf=RandomForestClassifier()
param = {'n_estimators' : [10, 150, 130],
        'max_depth' : [30, 60, 90 , None ] }
gs =GridSearchCV(rf, param, cv=5, n_jobs = -1 )
gs_fit=gs.fit(x_count_feat, data['label'])
pd.DataFrame(gs_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]


# # Gradient boosting Grid Search

# In[ ]:


import nltk
import re 
import string
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords= nltk.corpus.stopwords.words('english')
data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)
data.columns = ['label', 'body_text']

def count_punc(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100

data['body_len']= data['body_text'].apply(lambda x : len(x) - x.count(" "))
data['punc%']= data['body_text'].apply(lambda x : count_punc(x))


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens= re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
x_tfidf = tfidf_vect.fit_transform(data['body_text'])

x_features = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_tfidf.toarray())], axis =1)
x_features.head()


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


print(dir(GradientBoostingClassifier))
print(GradientBoostingClassifier())


# # Build your own Grid-search

# In[ ]:


def train_GB(est, max_depth, lr):
    GB = GradientBoostingClassifier(n_estimators=est, max_depth=max_depth, learning_rate=lr)
    GB_model = GB.fit(x_train, y_train)
    y_pred = GB_model.predict(x_test)
    precision, recall, fscore, support = score(y_test, y_pred, pos_label = 'spam', average = 'binary')
    print('Est : {} / Max_Depth : {} / LR : {} ----- Precision : {} / Recall : {} /  Accuracy : {}'.format(est, max_depth, lr, 
                                                                     round(precision,3),round(recall,3),round((y_pred==y_test).sum() / len(y_pred),3)))
    


# In[ ]:


for n_est in [50,100,150]:
    for max_depth in [3, 7 ,11, 15]:
        for lr in [0.01, 0.1, 1]:
            train_GB(n_est,max_depth,lr)


# # Evaluate GB with GridSearchCV

# In[ ]:


import nltk
import re 
import string
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords= nltk.corpus.stopwords.words('english')
data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)
data.columns = ['label', 'body_text']

def count_punc(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100

data['body_len']= data['body_text'].apply(lambda x : len(x) - x.count(" "))
data['punc%']= data['body_text'].apply(lambda x : count_punc(x))


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens= re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
x_tfidf = tfidf_vect.fit_transform(data['body_text'])
x_tfidf_feat = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_tfidf.toarray())], axis =1)

count_vect = CountVectorizer(analyzer=clean_text)
x_count = count_vect.fit_transform(data['body_text'])
x_count_feat = pd.concat([data['body_len'], data['punc%'], pd.DataFrame(x_count.toarray())], axis =1)

x_count_feat.head()


# In[ ]:


gb = GradientBoostingClassifier()
param = {'n_estimators' : [100, 150], 'max_depth' : [7, 11, 15], 'learning_rate' : [0.1]}

gs= GridSearchCV(gb. param, cv=5, n_jobs= -1)
cv_fit = gs.fit(x_tfidf_feat , data['label'])
pd.DataFrame(cv_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]


# In[ ]:


gb = GradientBoostingClassifier()
param = {'n_estimators' : [100, 150], 'max_depth' : [7, 11, 15], 'learning_rate' : [0.1]}

gs= GridSearchCV(gb. param, cv=5, n_jobs= -1)
cv_fit = gs.fit(x_count_feat , data['label'])
pd.DataFrame(cv_fit.cv_results_).sort_values('mean_test_score', ascending=False)[0:5]


# # Final Model Selection

# In[ ]:


import nltk
import re 
import string
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords= nltk.corpus.stopwords.words('english')
data=pd.read_csv("/kaggle/input/nlp-data-set/SMSSpamCollection.tsv" , sep='\t', header = None)
data.columns = ['label', 'body_text']

def count_punc(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")),3)*100

data['body_len']= data['body_text'].apply(lambda x : len(x) - x.count(" "))
data['punc%']= data['body_text'].apply(lambda x : count_punc(x))


def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens= re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(data[['body_text','body_len','punc%']], data['label'] ,test_size = 0.2)


# In[ ]:


tfidf_vect = TfidfVectorizer(analyzer=clean_text)
tfidf_vect_fit = tfidf_vect.fit(x_train['body_text'])

tfidf_train = tfidf_vect_fit.transform(x_train['body_text'])
tfidf_test = tfidf_vect_fit.transform(x_test['body_text'])

x_train_vect = pd.concat([x_train[['body_len','punc%']].reset_index(drop=True),
         pd.DataFrame(tfidf_train.toarray())], axis = 1)

x_test_vect = pd.concat([x_test[['body_len','punc%']].reset_index(drop=True),
         pd.DataFrame(tfidf_test.toarray())], axis = 1)
x_train_vect.head()


# # Model Selection : Results

# In[ ]:


import time


# In[ ]:


rf=RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)

start=time.time()
rf_model=rf.fit(x_train_vect, y_train)
end=time.time()
fit_time = (end-start)

start=time.time()
y_pred = rf_model.predict(x_test_vect)
end=time.time()
pred_time = (end-start)

precision, recall, fscore, support = score(y_test, y_pred, pos_label='spam', average='binary')
print('Fit_time: {} / Predict_time : {} / Precision : {} / Recall : {} / Accuracy : {}'.format(round(fit_time,3),round(pred_time,3),round(precision,3), round(recall, 3), 
                                                                round((y_pred==y_test).sum() / len(y_pred),3)))


# In[ ]:



gb = GradientBoostingClassifier(n_estimators=150, max_depth=11)
start=time.time()
gb_model=gb.fit(x_train_vect, y_train)
end=time.time()
fit_time = (end-start)

start=time.time()
y_pred = gb_model.predict(x_test_vect)
end=time.time()
pred_time = (end-start)

precision, recall, fscore, support = score(y_test, y_pred, pos_label='spam', average='binary')
print('Fit_time: {} / Predict_time : {} / Precision : {} / Recall : {} / Accuracy : {}'.format(round(fit_time,3),round(pred_time,3),round(precision,3), round(recall, 3), 
                                                                round((y_pred==y_test).sum() / len(y_pred),3)))


# In[ ]:




