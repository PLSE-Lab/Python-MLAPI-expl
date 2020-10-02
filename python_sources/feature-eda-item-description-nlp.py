#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

df_train = pd.read_csv('../input/train.tsv', sep='\t')
df_test = pd.read_csv('../input/test.tsv', sep='\t')
# Any results you write to the current directory are saved as output.


# In[ ]:


dataset = [df_train,df_test]


# In[ ]:


df_train['category_name'].fillna(value="unknown",inplace=True)
df_test['category_name'].fillna(value="unknown",inplace=True)


# In[ ]:


def split_categories(my_string) : 
    
    my_string = str(my_string)
    return [x.strip() for x in my_string.split('/')]


# In[ ]:


for c in dataset : 
    c['categories'] = c['category_name'].apply(split_categories)
    c['category_count'] = c['categories'].apply(len)


# In[ ]:


for data in dataset : 
    data.loc[data['category_name']=='unknown','category_count'] = 0


# In[ ]:


def get_cat1(x) :
    
    if x['category_count'] != 0:
        return x["categories"][0]
    else : 
        return np.nan

def get_cat2(x) :
    
    if x['category_count'] != 0:
        return x["categories"][1]
    else : 
        return np.nan

def get_cat3(x) :
    
    if x['category_count'] != 0:
        return x["categories"][2]
    else : 
        return np.nan

def get_cat4(x) :
    
    if x['category_count'] >=4 :
        return x["categories"][3]
    else : 
        return np.nan

def get_cat5(x) :
    
    if x['category_count'] >=5:
        return x["categories"][4]
    else : 
        return np.nan


# In[ ]:


for data in dataset : 
    data['cat1'] = np.nan
    data['cat2'] = np.nan
    data['cat3'] = np.nan
    data['cat4'] = np.nan
    data['cat5'] = np.nan
    
# for data in dataset : 
#     data['cat1'] = data.apply(get_cat1,axis=1)
#     data['cat2'] = data.apply(get_cat2,axis=1)
#     data['cat3'] = data.apply(get_cat3,axis=1)
#     data['cat4'] = data.apply(get_cat4,axis=1)
#     data['cat5'] = data.apply(get_cat5,axis=1)


# In[ ]:


df_train['cat1'] = df_train.apply(get_cat1,axis=1)
df_train['cat2'] = df_train.apply(get_cat2,axis=1)
df_train['cat3'] = df_train.apply(get_cat3,axis=1)
df_train['cat4'] = df_train.apply(get_cat4,axis=1)
df_train['cat5'] = df_train.apply(get_cat5,axis=1)


# In[ ]:


df_test['cat1'] = df_test.apply(get_cat1,axis=1)
df_test['cat2'] = df_test.apply(get_cat2,axis=1)
df_test['cat3'] = df_test.apply(get_cat3,axis=1)
df_test['cat4'] = df_test.apply(get_cat4,axis=1)
df_test['cat5'] = df_test.apply(get_cat5,axis=1)


# In[ ]:


plt.figure(figsize=(15,12))
sns.boxplot(df_train['category_count'],np.log(df_train['price']),hue=df_train['shipping'])
plt.ylabel("Log(Price)")
plt.xlabel("No of Categories")


# In[ ]:


df_train['brand_name'].fillna(value="unknown",inplace=True)
df_test['brand_name'].fillna(value="unknown",inplace=True)


# In[ ]:


df_train.isnull().sum()


# In[ ]:


df_test.isnull().sum()


# In[ ]:


df_train.dropna(subset=["item_description"],inplace=True)


# In[ ]:


df_train.drop("categories",axis=1,inplace=True)
df_test.drop("categories",axis=1,inplace=True)


# In[ ]:


df_train.head()


# In[ ]:


df_train['cat1'].fillna(value="unknown",inplace=True)
df_test['cat1'].fillna(value="unknown",inplace=True)

df_train['cat2'].fillna(value="unknown",inplace=True)
df_test['cat2'].fillna(value="unknown",inplace=True)

df_train['cat3'].fillna(value="unknown",inplace=True)
df_test['cat3'].fillna(value="unknown",inplace=True)

df_train['cat4'].fillna(value="unknown",inplace=True)
df_test['cat4'].fillna(value="unknown",inplace=True)

df_train['cat5'].fillna(value="unknown",inplace=True)
df_test['cat5'].fillna(value="unknown",inplace=True)


# **lets plot the main (First) categories for every item which is under cat1 column **

# In[ ]:


plt.figure(figsize=(12,10))

sns.countplot(df_train['cat1'])
plt.xlabel("Primary Categories")
plt.xticks(rotation=90)
plt.title("Count by Primary Categories")


# In[ ]:


plt.figure(figsize=(12,10))
sns.boxplot(df_train['cat1'],y=np.log(df_train['price']),hue=df_train['shipping'])
plt.xticks(rotation=90)
plt.xlabel("Categories")
plt.ylabel("Log(Price)")
plt.title("Price Distribution by Primary Categories")


# In[ ]:


plt.figure(figsize=(15,10))
sns.distplot(df_train['price'],kde=False)


# In[ ]:


df_train.head()


# **Text Cleaning **

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer
import re 

count = CountVectorizer()
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)

def text_processor(text) : 
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                           text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


# In[ ]:


df_train['item_description'] = df_train['item_description'].apply(text_processor)


# **Stemming and removing Stop Words**

# In[ ]:


from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords 

stop_words = stopwords.words("english")
porter = PorterStemmer()

def tokenizer(text):
    return text.split()

def tokenizer_porter(text) : 
    return [porter.stem(word) for word in text.split() if word not in stop_words]


# In[ ]:


df_train['item_description'][:10].apply(tokenizer_porter)


# In[ ]:


vectorizer = TfidfVectorizer(min_df=20,
                             max_features=180000,
                             tokenizer=tokenizer_porter,
                             ngram_range=(1, 1))


# In[ ]:


all_desc = np.append(df_train['item_description'].values, df_test['item_description'].values)
vz = vectorizer.fit_transform(list(all_desc))


# In[ ]:


tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
tfidf = pd.DataFrame(columns=['tfidf']).from_dict(
                    dict(tfidf), orient='index')
tfidf.columns = ['tfidf']


# In[ ]:


tfidf.sort_values(by=['tfidf'], ascending=True).head(10)


# In[ ]:


tfidf.sort_values(by=['tfidf'], ascending=False).head(10)


# In[ ]:




