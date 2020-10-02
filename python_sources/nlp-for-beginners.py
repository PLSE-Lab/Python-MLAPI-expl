#!/usr/bin/env python
# coding: utf-8

# <h1><font size="+3.5" color="blue" style="font-family: Futura"><center>Natural Langauge Processing</center></font></h1>

# <font size='3' style="font-family: Futura">In current genreation NLP is going heigher and higher everyday. So many bigtech companies are using NLP im there product. like <a href="https://google.com">Google</a> in Google assistant, <a href="https://apple.com">Apple</a> in siri, <a href="https:www.amazon.com">Amazon</a> in Alexa, <a href='https://microsoft.com'>Microsoft</a> and so on.</font>

# <font size='3' style="font-family: Futura">Here I am going to explain Natural Langauge Processing. I have learn lot of thing from good Youtube channels and so many blogs which i sharing here which will help you also. <a href="https://www.youtube.com/playlist?list=PLZoTAELRMXVMdJ5sqbCK2LiM0HhQVWNzm">NLP Basic(Youtube)</a> and <a href="https://towardsdatascience.com/your-guide-to-natural-language-processing-nlp-48ea2511f6e1">Blog</a> this site and channle both helping me so much to learn Data Science and Machine learnig. Here i am using some diffrent approch for presentation which i learn from this <a href='https://www.kaggle.com/vishalvanpariya/house-pricing-ultimate-guide/edit'>kernel</a>.</font>

# <h2 style="font-family: Futura; color:blue;">Overview</h2>

# * [<font size='3' style="font-family: Futura">1.Importing Important Libraries</font>](#1)
# * [<font size='3' style="font-family: Futura">2.Importing Data</font>](#2)
# * [<font size='3' style="font-family: Futura">3.Data Cleaning</font>](#3)
# * [<font size='3' style="font-family: Futura">4.Feature Scalling</font>](#4)
# * [<font size='3' style="font-family: Futura">5.Modeling</font>](#5)
# * [<font size='3' style="font-family: Futura">6.Submission</font>](#6)

# <h2 style="font-family: Futura; color:green "><b>1. Libraries</b></h2><br><a id='1'></a>

# In[ ]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer


# <font size='3' style="font-family: Futura">
# <b style="color:blue">Use Of Liberaries:</b><br><br>
#     &emsp;&emsp;<b style="color:green">1. Numpy :</b> we will use it for maths opration<br>
#     &emsp;&emsp;<b style="color:green">2. Pandas :</b> Data Handling<br>
#     &emsp;&emsp;<b style="color:green">3. re :</b> Regular Expression<br>
#     &emsp;&emsp;<b style="color:green">4. Stopwords :</b> Remove Stopwords<br>
#     &emsp;&emsp;<b style="color:green">5. PorterStemmer :</b> For Stemming<br>
#     &emsp;&emsp;<b style="color:green">6. word_tokenize :</b> For work token<br>
#     &emsp;&emsp;<b style="color:green">7. defaultdict :</b> For dictionary<br>
#     &emsp;&emsp;<b style="color:green">8. WordNetLemmatizer :</b> For word lemmatize<br>
#     &emsp;&emsp;<b style="color:green">9. String :</b> For string oprations<br>
#     &emsp;&emsp;<b style="color:green">10. TfidfVectorizer :</b> Vecterization<br>
# </font>

# <h2 style="font-family: Futura; color:green "><b>2. Import Data</b></h2><br><a id='2'></a>

# In[ ]:


train=pd.read_csv('../input/nlp-getting-started/train.csv')
test=pd.read_csv('../input/nlp-getting-started/test.csv')


# <h2 style="font-family: Futura; color:green "><b>3. Data Cleaning</b></h2><br><a id='3'></a>

# In[ ]:


train.head()


# <font size='3' style="font-family: Futura">
# <b style="color:green">id</b> - a unique identifier for each tweet<br>
#     <b style="color:green">text</b> - the text of the tweet<br>
# <b style="color:green">location</b> - the location the tweet was sent from (may be blank)<br>
# <b style="color:green">keyword</b> - a particular keyword from the tweet (may be blank)<br>
# <b style="color:green">target</b> - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)<br>
# </font>

# <font size='3' style="font-family: Futura">Let's delete keyword and location</font>

# In[ ]:


train=train.drop('keyword',1)
train=train.drop('location',1)

test=test.drop('keyword',1)
test=test.drop('location',1)


# <font size='3' style="font-family: Futura">let's have important data from Dataframe</font>

# In[ ]:


y_train=train.iloc[:,-1]
x_train=train.iloc[:,:-1]


# In[ ]:


ps=PorterStemmer()
lemmatizer=WordNetLemmatizer()


# In[ ]:


def atcontain(text):
    ar=[]
    text=text.split()
    for t in text:
        if("@" in t):
            ar.append("TAGSOMEBODY")
        else:
            ar.append(t)
    return " ".join(ar)
    

def dataclean(data):
    corpus=[]
    for i in range(data.shape[0]):
        tweet=data.iloc[i,-1]
        tweet=atcontain(tweet)
        tweet=re.sub(r'http\S+', '', tweet)
        tweet=re.sub('[^a-zA-z]'," ",tweet)
        tweet=tweet.lower()
        tweet=word_tokenize(tweet)
#         tweet=[ps.stem(word) for word in tweet if word not in stopwords.words('english')]
        tweet=[lemmatizer.lemmatize(word) for word in tweet if word not in stopwords.words('english')]
        tweet=[word for word in tweet if word not in set(string.punctuation)]
        tweet=" ".join(tweet)
        corpus.append(tweet)
    return corpus


# <font size='3' style="font-family: Futura"><b style="color:green">Creating data clean function:</b><br>
#     in this function we are going remove puctuationa,stopword and changing to lower case<br>
#     i have commented one line in function which is for porter stemmer, i commented it because it is decresing model accuracy
# </font>

# In[ ]:


x_corpus_train=dataclean(x_train)
x_corpus_test=dataclean(test)


# <font size='3' style="font-family: Futura; color:green">Data Cleaned</font>

# In[ ]:


dic=defaultdict(int)
for text in x_corpus_train:
    text=text.split()
    for word in text:
        dic[word]=dic[word]+1


# <font size='3' style="font-family: Futura">Sorting values through frequency of word</font>

# In[ ]:


sorted_data=sorted(dic.items(), key=lambda x:x[1],reverse=True)
sorted_data[:20]


# <h2 style="font-family: Futura; color:green "><b>4. Feature Scalling</b></h2><br><a id='4'></a>

# In[ ]:


cv=TfidfVectorizer(max_features=8000)


# In[ ]:


x_train_vector=cv.fit_transform(x_corpus_train).toarray()
x_test_vector=cv.transform(x_corpus_test).toarray()


# <h2 style="font-family: Futura; color:green "><b>5. Modeling</b></h2><br><a id='5'></a>

# <h3 style="font-family: Futura; color:blue "><b>1. Naive Bayes</b></h3><br>

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB()
model.fit(x_train_vector,y_train)
print(model.score(x_train_vector,y_train))
y_pred=model.predict(x_test_vector)
y_pred


# <h3 style="font-family: Futura; color:blue "><b>2. Randomforest</b></h3><br>

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(random_state=4,n_estimators=500,warm_start=True,max_depth=6,min_samples_leaf=2,max_features='auto',min_samples_split=3)
rfc.fit(x_train_vector,y_train)
print(rfc.score(x_train_vector,y_train))
y_pred=rfc.predict(x_test_vector)


# <h3 style="font-family: Futura; color:blue "><b>3. XGBoost</b></h3><br>

# In[ ]:


from xgboost import XGBClassifier
xgb=XGBClassifier()
xgb.fit(x_train_vector,y_train,early_stopping_rounds=5, 
             eval_set=[(x_train_vector,y_train)], 
             verbose=False)
print(xgb.score(x_train_vector,y_train))
y_pred=xgb.predict(x_test_vector)


# <h3 style="font-family: Futura; color:blue "><b>4. Logistic Regression</b></h3><br>

# In[ ]:


from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()
reg.fit(x_train_vector,y_train)
print(reg.score(x_train_vector,y_train))
y_pred=reg.predict(x_test_vector)
y_pred


# <h3 style="font-family: Futura; color:blue "><b>5. PassiveAggressiveClassifier</b></h3><br>

# In[ ]:


from sklearn.linear_model import PassiveAggressiveClassifier
passive=PassiveAggressiveClassifier()
passive.fit(x_train_vector,y_train)
print(passive.score(x_train_vector,y_train))
y_pred=passive.predict(x_test_vector)
y_pred


# <h2 style="font-family: Futura; color:green "><b>6. Submission</b></h2><br><a id='6'></a>

# In[ ]:


submission=pd.read_csv('../input/nlp-getting-started/sample_submission.csv')
submission['target']=y_pred
submission.to_csv('submission.csv',index=False)


# <h1>Check out my other Notebooks</h1><font size='4'>
# <a href="https://www.kaggle.com/vishalvanpariya/top-5-on-leaderboard" target="_blank">House Price</a><br>
# <a href="https://www.kaggle.com/vishalvanpariya/data-explanation-titanic" target="_blank">Titanic EDA</a><br>
# <a href="https://www.kaggle.com/vishalvanpariya/titanic-top-6" target="_blank">Titanic Notebook</a><br>
# <a href="https://www.kaggle.com/vishalvanpariya/nlp-for-beginners" target="_blank">NLP</a><br><font>

# In[ ]:




