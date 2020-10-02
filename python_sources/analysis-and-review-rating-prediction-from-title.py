#!/usr/bin/env python
# coding: utf-8

# <h1> Grammar and Online Product Reviews
# 

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("../input/GrammarandProductReviews.csv")

data.head()


# In[119]:


data.shape


# ### There are about 392 unique brands whose products are reviewed

# In[11]:


len(data['brand'].unique())


# ### There are 581 unique product types 

# In[12]:


len(data['categories'].unique())


# ### 1-5 review ratings

# In[135]:


sorted(data['reviews.rating'].unique())


# ### Lets Analyse Universal Music's rating  ### 
# Universal Music Group is an American global music corporation that is a subsidiary of the French media conglomerate Vivendi. UMG's global corporate headquarters are located in Santa Monica, California

# In[152]:


Universalmusic = data.loc[data['brand'] == 'Universal Music',:]
ratings = list(Universalmusic['reviews.rating'])
print("Average Rating of UniversalMusic is:",sum(ratings)/len(ratings))


# In[161]:


top = Universalmusic['reviews.rating'].value_counts().index.tolist()
value = Universalmusic['reviews.rating'].value_counts().values.tolist()
sns.barplot(top, value, alpha=0.8)
plt.xlabel('Rating of the product', fontsize=14)
plt.ylabel('Number of reviews with that given', fontsize=14)
plt.title("Rating for Universal Music", fontsize=16)
plt.show()


# ### Univeral Music has a great average rating and is a good brand

# ### Lets analyse Lundbergs rating
# The Lundberg Family Farms rice products are farmed and produced in an eco-friendly, sustainable manner. Featuring rice recipes and a full listing of our product.

# In[151]:


Lundberg = data.loc[data['brand'] == 'Lundberg',:]
ratings = list(Lundberg['reviews.rating'])
print("Average Rating of Lundberg is:",sum(ratings)/len(ratings))


# ### Lundberg also seems to have a higher average rating, it's a good brand

# ### Let us check out the reviews for The Jungle Book by Disney

# In[27]:


junglebook = data[(data['brand'] == 'Disney') & (data['name'] == "The Jungle Book (blu-Ray/dvd + Digital)")]


# In[31]:


top = junglebook['reviews.rating'].value_counts().index.tolist()
value = junglebook['reviews.rating'].value_counts().values.tolist()
sns.barplot(top, value, alpha=0.8)
plt.xlabel('Rating given for Jungle Book', fontsize=14)
plt.ylabel('Number of that rating given', fontsize=14)
plt.title("Rating for Jungle Book by disney", fontsize=16)
plt.show()


# ### Jungle Book is a Disney Classic and no wonder has garnered more positive reviews

# In[47]:


data['reviews.title'].unique()[:5]


# ### 40 words is the average number of words per review

# In[4]:


totalreviews = list(data['reviews.text'])
length = []
for i in range(0,len(totalreviews)):
        totalreviews[i] = str(totalreviews[i])
        a = len(totalreviews[i].split(' '))
        length.append(a)

    
print("On average a review has about:", sum(length)/len(length),"words in them")


# In[118]:


len(length)


# In[6]:


ratings = list(data['reviews.rating'])
len(ratings)


# ### Average Number of words per rating

# In[7]:


dt = pd.DataFrame()
dt['length'] =  length
dt['ratings'] =  ratings
five_star = dt.loc[dt['ratings'] == 5,:]
five = sum(five_star['length'])/len(five_star['length'])
four_star = dt.loc[dt['ratings'] == 4,:]
four = sum(four_star['length'])/len(four_star['length'])
three_star = dt.loc[dt['ratings'] == 3,:]
three = sum(three_star['length'])/len(three_star['length'])
to_star = dt.loc[dt['ratings'] == 2,:]
to = sum(to_star['length'])/len(to_star['length'])
on_star = dt.loc[dt['ratings'] == 1,:]
on = sum(on_star['length'])/len(on_star['length'])


# ### Five star ratings are shortest whereas two star ratings tend to be the longest

# In[146]:


colors = ['gold', 'orange','yellowgreen', 'lightcoral', 'lightskyblue']
top = ['one','two','three','four','five']
value = [int(on), int(to),int(three),int(four),int(five)]
sns.barplot(top, value, alpha=0.8)
plt.xlabel('Rating of the product', fontsize=14)
plt.ylabel('Average number of words in the review', fontsize=14)
plt.title("Rating given vs Number of words used in review", fontsize=16)
plt.show()


# In[7]:


f = data.loc[data['reviews.rating'] == 5,:]
ss = list(f['reviews.text'])
aa=[]
for i in range(0,len(ss)):
    ss[i] = str(ss[i])
    aa.append(ss[i].split(' '))


# ### Array of all words used in 5 star reviews

# In[8]:


all_words = [j for i in aa for j in i]
all_words[:5]


# In[9]:


import nltk
from nltk.corpus import stopwords
import string


# ### Remove Punctuations

# In[10]:


exclude = set(string.punctuation)
for i in range(0,len(all_words)):
    all_words[i] = all_words[i].lower()
    all_words[i] = ''.join(ch for ch in all_words[i] if ch not in exclude)


# ### Remove stopwords

# In[11]:


stop = set(stopwords.words('english'))
stopwordsfree_words = [word for word in all_words if word not in stop]


# In[12]:


from collections import Counter
counts = Counter(stopwordsfree_words)


# ### Most commonly used words in Positive reviews are GREAT & LOVE

# In[13]:


counts.most_common(5)


# In[14]:


f = data.loc[data['reviews.rating'] == 1,:]
ss = list(f['reviews.text'])
aa=[]
for i in range(0,len(ss)):
    ss[i] = str(ss[i])
    aa.append(ss[i].split(' '))


# ### Array of all words used in 1 star reviews
# 

# In[15]:


all_words = [j for i in aa for j in i]
all_words[:5]


# In[16]:


exclude = set(string.punctuation)
for i in range(0,len(all_words)):
    all_words[i] = all_words[i].lower()
    all_words[i] = ''.join(ch for ch in all_words[i] if ch not in exclude)


# In[17]:


stop = set(stopwords.words('english'))
stopwordsfree_words = [word for word in all_words if word not in stop]


# In[18]:


from collections import Counter
counts = Counter(stopwordsfree_words)


# In[19]:


counts.most_common(5)


# ### We can see that 1 star reviews refrain from using negative words but rather focus on the experience of the customer or describes the quality of the product

# # PREDICTING REVIEW RATING FROM REVIEW TITLE

# In[74]:


df1 = data.replace(np.nan, 'Not Filled', regex=True)


# In[76]:


X = list(df1['reviews.title'])
Y = list(df1['reviews.rating'])
(X[:5],Y[:5])


# In[77]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2, random_state=1)


# ### Using tfidf vectorizer

# In[51]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[78]:


def tfidf_features(X_train, X_val, X_test):
    tfidf_vectorizer = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(1, 2),token_pattern='(\S+)')
    X_train = tfidf_vectorizer.fit_transform(X_train)
    X_val = tfidf_vectorizer.transform(X_val)
    X_test = tfidf_vectorizer.transform(X_test)
    
    return X_train, X_val, X_test, tfidf_vectorizer.vocabulary_


# In[79]:


X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf_vocab = tfidf_features(X_train, X_val, X_test)


# ### Logisitic regression

# In[4]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression


# In[5]:


def train_classifier(X_train, y_train):    
    clf = OneVsRestClassifier(LogisticRegression())
    clf.fit(X_train, y_train)
    return clf


# In[6]:


classifier_tfidf = train_classifier(X_train_tfidf, y_train)


# In[93]:


y_val_predicted_labels_tfidf = classifier_tfidf.predict(X_val_tfidf)


# In[117]:


y_val_predicted_labels_tfidf  ==2


# ### For Positive review

# In[103]:


X_val[0]


# ### 5 stars has been correctly predicted

# In[104]:


y_val_predicted_labels_tfidf[0]


# ### For negative review

# In[113]:


X_val[15]


# ### 1 star has been correctly predicted

# In[114]:


y_val_predicted_labels_tfidf[15]


# ### 2 star review

# In[127]:


(X_val[101],y_val_predicted_labels_tfidf[101])


# ### 3 star review

# In[129]:


(X_val[68],y_val_predicted_labels_tfidf[68])


# ### 4 star review

# In[131]:


(X_val[42],y_val_predicted_labels_tfidf[42])


# In[122]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score


# In[3]:


def print_evaluation_scores(y_val, predicted):
    print(accuracy_score(y_val, predicted))
    print(f1_score(y_val, predicted, average='weighted'))
print_evaluation_scores(y_val, y_val_predicted_labels_tfidf)


# ### ABOUT 70% ACCURACY HAS BEEN REACHED
