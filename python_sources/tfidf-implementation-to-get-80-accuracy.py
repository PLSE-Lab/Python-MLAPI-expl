#!/usr/bin/env python
# coding: utf-8

# **The Objective of this kernel is to get you on your feet. I would be using Logisitic Regression to create a basic baseline prediction model. Once we have that in place we would be doing some basic data cleaning and applying SVM,Naive Bayes to the data.**
# 
# Would be adding - data visualization , feature engineering steps in the future.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import warnings; warnings.simplefilter('ignore')
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

pd.set_option('display.width',1000000)
pd.set_option('display.max_columns', 500)

score_df = pd.DataFrame(columns={'Model Description','Score'})
# Any results you write to the current directory are saved as output.


# **1. Loading data set **

# In[ ]:


df_train= pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
df_test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')


# **2. Let's take initial look at the data **

# In[ ]:


print(df_train.head(5))


# In[ ]:


print(df_train.info())


# Check for Null/NAN Values

# In[ ]:


print(df_train.isnull().any())


# In[ ]:


print(df_test.isnull().any())


# In[ ]:


print(df_train.shape)


# Exploring the data distribution of tweets

# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt

fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)
plt.tight_layout()

labels=['Disaster Tweet','No Disaster']
size=  [df_train['target'].mean()*100,abs(1-df_train['target'].mean())*100]
explode = (0, 0.1)
#ig1,ax1 = plt.subplots()
axes[0].pie(size,labels=labels,explode=explode,shadow=True,
            startangle=90,autopct='%1.1f%%')
sns.countplot(x=df_train['target'], hue=df_train['target'], ax=axes[1])
plt.show()


# Before analyzing the data further would be nice to have a baseline model driven off just the tweet Text and TFIDF transformer. once we have some baseling we would look at some more visualization and feature engineering

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer

X_train,X_test,y_train,y_test = train_test_split(df_train['text'],df_train['target'])
vector = TfidfVectorizer().fit(X_train)

#print(vector.get_feature_names())
X_train_vector = vector.transform(X_train)
X_test_vector = vector.transform(X_test)

model = LogisticRegression().fit(X_train_vector,y_train)
print('Logistic Regression ROC Auc Score with TFIDF - %3f'%(roc_auc_score(y_test,model.predict(X_test_vector))))
print('F1Score - %3f'%(f1_score(y_test,model.predict(X_test_vector))))
score_df = score_df.append({'Model Description':'Basic LR Model - Basline - TFIDF',
                           'Score':roc_auc_score(y_test,model.predict(X_test_vector))}
                           ,ignore_index=True)

####### Now let's try with count vectorizer

cv_vector = CountVectorizer().fit(X_train)
X_train_vector = cv_vector.transform(X_train)
X_test_vector = cv_vector.transform(X_test)

model = LogisticRegression().fit(X_train_vector,y_train)
predict = model.predict(X_test_vector)
score = roc_auc_score(y_test,predict)
print('Logistic Regression Roc AUC Score with countvectorizer - %3f'%score)

score_df = score_df.append({'Model Description':'Basic LR Model - Basline - CV',
                          'Score':score}
                          ,ignore_index=True)


# **3. Clean Data**  - So we have a baseline score of 79% to work with , let's get to clean data and see if we can improve the score
# 
# As first step in cleaning - let us replace some commonly occuring shorthands 

# In[ ]:



def clean_text(text):
    import re
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"you'll", "you will", text)
    text = re.sub(r"i'll", "i will", text)
    text = re.sub(r"she'll", "she will", text)
    text = re.sub(r"he'll", "he will", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"here's", "here is", text)
    text = re.sub(r"who's", "who is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"shouldn't", "should not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"   ", " ", text) # Remove any extra spaces
    return text


df_train['clean_text'] = df_train['text'].apply(clean_text)
df_test['clean_text'] = df_test['text'].apply(clean_text)


# In the next step we are going to do some further massaging which would make Job of Prediction Algorithm easy
# 
# * Let us remove any characters other then alphabets
# * Convert all dictionary to lower case - for consistency 
# * Lemmatize - More details on Stemming and Lemmatization [here](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)
# 

# Also we are going to store this text in a seperate column as we want to keep the orignal text in case we want to do some feature engineering down the line.

# In[ ]:


def massage_text(text):
    import re
    from nltk.corpus import stopwords
    ## remove anything other then characters and put everything in lowercase
    tweet = re.sub("[^a-zA-Z]", ' ', text)
    tweet = tweet.lower()
    tweet = tweet.split()

    from nltk.stem import WordNetLemmatizer
    lem = WordNetLemmatizer()
    tweet = [lem.lemmatize(word) for word in tweet
             if word not in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    return tweet
    print('--here goes nothing')
    print(text)
    print(tweet)

df_train['clean_text'] = df_train['text'].apply(massage_text)
df_test['clean_text'] = df_test['text'].apply(massage_text)


# Let's take a look at the data now 

# In[ ]:


df_train.iloc[0:10][['text','clean_text']]


# **4. Creation of more Models**

# 4.1 Start by creating a Logistic Regression model again , this time we will use Grid Seach for hyper-parameter optimization

# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

vector = TfidfVectorizer().fit(df_train['clean_text'])
df_train_vector = vector.transform(df_train['clean_text'])
df_test_vector = vector.transform(df_test['clean_text'])
lr_model = LogisticRegression()
grid_values =  {'penalty':['l1', 'l2'],'C':[0.01, 0.1, 1, 10, 100]}
grid_search_model = GridSearchCV(lr_model,param_grid=grid_values,cv=3)
grid_search_model.fit(df_train_vector,df_train['target'])

print(grid_search_model.best_estimator_)
print(grid_search_model.best_score_)
print(grid_search_model.best_params_)

## dumping the output to a file 
predict_df = pd.DataFrame()
predict = grid_search_model.predict(df_test_vector)
predict_df['id'] = df_test['id']
predict_df['target'] = predict
predict_df.to_csv('sample_submission_2.csv', index=False)
score_df = score_df.append({'Model Description':'LR Model - with data cleaning and Grid Search',
                           'Score':grid_search_model.best_score_}
                           ,ignore_index=True)


### let's have another model with some ngram's though 
X_train,X_test,y_train,y_test = train_test_split(df_train['clean_text'],df_train['target'])
vector = TfidfVectorizer(ngram_range=(1,3)).fit(X_train)
X_train_vector = vector.transform(X_train)
X_test_vector = vector.transform(X_test)

lr_model = LogisticRegression(C=1,penalty='l2').fit(X_train_vector,y_train)
predict = lr_model.predict(X_test_vector)
score = roc_auc_score(y_test,predict)
print('Roc AUC curve for LR and TFIDF with ngrams  - %3f'%score)

score_df = score_df.append({'Model Description':'LR Model - with ngram range',
                           'Score':score}
                           ,ignore_index=True)


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

### let's have another model with some ngram's though 
X_train,X_test,y_train,y_test = train_test_split(df_train['clean_text'],df_train['target'])
vector = TfidfVectorizer(ngram_range=(1,3)).fit(X_train)
X_train_vector = vector.transform(X_train)
X_test_vector = vector.transform(X_test)

lr_model = LogisticRegression(C=1,penalty='l2').fit(X_train_vector,y_train)
predict = lr_model.predict(X_test_vector)
score = roc_auc_score(y_test,predict)
print('Roc AUC curve for LR and TFIDF with ngrams  - %3f'%score)

score_df = score_df.append({'Model Description':'LR Model - with ngram range',
                           'Score':grid_search_model.score}
                           ,ignore_index=True)

vector = TfidfVectorizer(ngram_range=(1,3)).fit(df_train['clean_text'])
X_train_vector = vector.transform(df_train['clean_text'])
X_test_vector = vector.transform(df_test['clean_text'])
lr_model = LogisticRegression(C=1,penalty='l2').fit(X_train_vector,df_train['target'])
predict = lr_model.predict(X_test_vector)


## dumping the output to a file 
predict_df = pd.DataFrame()
predict_df['id'] = df_test['id']
predict_df['target'] = predict
predict_df.to_csv('sample_submission_001.csv', index=False)


# In[ ]:


pd.concat([df_test,predict_df['target']],axis=1)

### you could dump this in a csv and do further analysis to check what
### misclassifications are there manually ,observations could then be used 
### to further tweak stuff


# 4.2 Let's apply Gaussian NB to the data 

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

X_train, X_test, y_train, y_test =         train_test_split(df_train['clean_text'], df_train['target'], random_state=20)
## Apply Tfidf tranformation
vector = TfidfVectorizer().fit(X_train)
X_train_vector = vector.transform(X_train)
X_test_vector  = vector.transform(X_test)
df_test_vector = vector.transform(df_test['clean_text'])

gb_model= GaussianNB().fit(X_train_vector.todense(),y_train)
predict = gb_model.predict(X_test_vector.todense())

print('Roc AUC score - %3f'%(roc_auc_score(y_test,predict)))
score_df = score_df.append({'Model Description':'Naive Bayes',
                           'Score':roc_auc_score(y_test,predict)}
                           ,ignore_index=True)


# 4.3 Support Vector Classifier - with Grid search to Optimize parameters

# In[ ]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

vector = TfidfVectorizer().fit(df_train['clean_text'])
df_train_vector = vector.transform(df_train['clean_text'])
df_test_vector = vector.transform(df_test['clean_text'])

svc_model = SVC()
grid_values={'kernel':['linear', 'poly', 'rbf'],'C':[0.001,0.01,1,10]}
grid_search_model= GridSearchCV(svc_model,param_grid=grid_values,cv=3)
grid_search_model.fit(df_train_vector,df_train['target'])

print(grid_search_model.best_estimator_)
print(grid_search_model.best_score_)
print(grid_search_model.best_params_)

score_df = score_df.append({'Model Description':'SVC - with Grid Search',
                           'Score':grid_search_model.best_score_}
                           ,ignore_index=True)

predict = grid_search_model.predict(df_test_vector)
predict_df = pd.DataFrame()
predict_df['id'] = df_test['id']
predict_df['target'] = predict

# # print(predict_df.head(5))
predict_df.to_csv('sample_submission_4.csv', index=False)


# Let's look at score_df which has scores of all models till now and let's sort the output in ascending based on the Score

# In[ ]:


score_df[['Model Description','Score']]


# **Please Upvote if you found the notebook usefull.Also please leave a comment if you think something could be improved/done in a better way. **
# 
# I have written another notebook to implement Word Embeddings using Word2Vec and then doing prediction implemention LR/ RF , it would save you a lot of time if you are new to Embeddings here is the [link](https://www.kaggle.com/slatawa/simple-implementation-of-word2vec)
