#!/usr/bin/env python
# coding: utf-8

# In[652]:


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


# In[653]:


import csv
import pandas as pd
import collections
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer


# **Data Analysis:**

# In[654]:


df1 = pd.read_csv(r"../input/riverus-assignment/1.txt", sep="\t", header = None)
df2 = pd.read_csv(r"../input/riverus-assignment/2.txt", sep="\t", header = None)
test_df = pd.read_csv(r"../input/riverus-assignment/3.txt", sep=",", header = [0], index_col = 0)
testing_df = pd.read_csv(r"../input/riverus-assignment/3.txt", sep=",", header = [0], index_col = 0)
df2.columns = ['Target', 'Text'] 
df1.columns = ['Text', 'Target'] 
df2 = df2[['Text','Target']]
df = pd.concat([df1,df2], ignore_index = True)


# In[655]:


df.describe()


# In[656]:


df.groupby('Target').describe()


# In[657]:


df['length'] = df['Text'].apply(len)
df.head()


# In[658]:


df.length.describe()


# In[659]:


df.hist(column='length', by='Target', bins =110, figsize=(10,4))


# **Preprocessing and Analysis:**

# In[660]:


freq = pd.Series(' '.join(df['Text']).split()).value_counts()
freq = dict(freq)
len(freq)
# selectedKeys = list() 
# for (key, value) in freq.items() :
#     if value < 5:
#         selectedKeys.append(key)
word_counter = collections.Counter(freq)
# for word, count in word_counter.most_common():
#     print(word, ": ", count)

lst = word_counter.most_common(50)
df_bar = pd.DataFrame(lst, columns = ['Word', 'Count'])
df_bar.plot.bar(x='Word',y='Count', figsize = (15,10), title = 'Most frequently occuring words')


# In[661]:


sw = stopwords.words('english')
# sw.extend(selectedKeys)
print(sw)

tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9]+'
pat2 = r'https?://[A-Za-z0-9./]+'
combined_pat = r'|'.join((pat1, pat2))
def cleaner(text):
    soup = BeautifulSoup(text, 'lxml')
    souped = soup.get_text()
    stripped = re.sub(combined_pat, '', souped)
    try:
        clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
    except:
        clean = stripped
    letters_only = re.sub("[^a-zA-Z]", " ", clean)
    lower_case = letters_only.lower()
    words = tok.tokenize(lower_case)
    return (" ".join(words)).strip()

def remove_punctuation(text):
    import string
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_url(data):
#     emo = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
#     data = emo.sub(r'', data)
    if data.startswith('www.'):
        data = re.sub(r'www.', '', data)
    if data.startswith('http.'):
        data = re.sub(r'http.', '', data)
    domain = data.split("//")[-1].split("/")[0]
    return domain

def stopwords(text):
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    return " ".join(text)


# In[662]:


df['Text'] = df['Text'].apply(stopwords)
df['Text'] = df['Text'].apply(remove_url)
df['Text'] = df['Text'].apply(remove_punctuation)
df['Text'] = df['Text'].apply(cleaner)
df['Text'] = df['Text'].apply(stopwords)
test_df['Text'] = test_df['Text'].apply(remove_url)
test_df['Text'] = test_df['Text'].apply(remove_punctuation)
test_df['Text'] = test_df['Text'].apply(cleaner)
test_df['Text'] = test_df['Text'].apply(stopwords)


# In[663]:


df.groupby('Target').describe()


# In[664]:


df['cleaned_length'] = df['Text'].apply(len)
df.head()


# In[665]:


df.cleaned_length.describe()


# In[666]:


df.hist(column='cleaned_length', by='Target', bins =50, figsize=(10,4))


# In[667]:


cleaned_freq = pd.Series(' '.join(df['Text']).split()).value_counts()
cleaned_freq = dict(cleaned_freq)
len(cleaned_freq)
cleaned_word_counter = collections.Counter(cleaned_freq)
# for word, count in word_counter.most_common():
#     print(word, ": ", count)

cleaned_lst = cleaned_word_counter.most_common(50)
df_bar = pd.DataFrame(cleaned_lst, columns = ['Word', 'Count'])
df_bar.plot.bar(x='Word',y='Count', figsize = (15,10), title = 'Most frequently occuring words in cleaned dataset')


# In[668]:


df.drop(['length', 'cleaned_length'], axis = 1, inplace = True)


# In[669]:


df = df.reindex(np.random.permutation(df.index))
df.head()


# In[670]:


stemmer = SnowballStemmer("english")
def stemming(text):    
    text = [stemmer.stem(word) for word in text.split()]
    return " ".join(text) 
df['Text'] = df['Text'].apply(stemming)
test_df['Text'] = test_df['Text'].apply(stemming)


# **Building the Model:**

# In[671]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sn


# **TF-IDF Model:**

# In[672]:


cv = TfidfVectorizer(ngram_range = (1,2))
X_df = cv.fit_transform(df.Text).toarray()
test_df_cv = cv.transform(test_df.Text).toarray()
Y_df = df.iloc[:,1].values


# In[673]:


X_train, X_test, Y_train, Y_test = train_test_split(X_df, Y_df, test_size = 0.20, random_state = 10)


# **Logistic Regression:**

# In[674]:


lr = LogisticRegression()
lr.fit(X_train, Y_train)


# In[675]:


import pickle

with open('lr_model.pickle', 'wb') as f:
    pickle.dump(lr, f)


# In[676]:


Y_pred = lr.predict(X_test)
target_names = ['0', '1']
print(classification_report(Y_test, Y_pred, target_names=target_names))


# In[677]:


cm_lr = confusion_matrix(Y_test,Y_pred)
conf_matrix_lr = pd.DataFrame(data=cm_lr,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix_lr, annot=True,fmt='d',cmap="YlGnBu")


# **Random Forest:**

# In[678]:


rf = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 10)
rf.fit(X_train, Y_train)


# In[679]:


with open('rf_model.pickle', 'wb') as f:
    pickle.dump(rf, f)


# In[680]:


Y_pred_rf = rf.predict(X_test)
target_names = ['0', '1']
print(classification_report(Y_test, Y_pred_rf, target_names=target_names))


# In[681]:


cm_rf = confusion_matrix(Y_test,Y_pred_rf)
conf_matrix_rf = pd.DataFrame(data=cm_rf,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
sn.heatmap(conf_matrix_rf, annot=True,fmt='d',cmap="YlGnBu")


# **Testing:**

# In[682]:


test_pred_rf = lr.predict(test_df_cv)
print(test_pred_rf)


# In[683]:


testing_df.head()


# In[684]:


submission = pd.DataFrame({'index':test_df['index'],'Text':testing_df['Text'],'Prediction':test_pred_rf})
submission.head(10)


# In[685]:


submission.to_csv('Predictions1.csv', index = False)


# In[ ]:




