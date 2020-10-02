#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
data=pd.read_csv('../input/train.csv')

# Any results you write to the current directory are saved as output.


# In[ ]:


data.shape
from matplotlib import *
import matplotlib.pyplot as plt


# In[ ]:


data.head()


# In[ ]:


import nltk
import wordcloud


# In[ ]:


class_0=data[data['target']==0]
class_1=data[data['target']==1]
wc=wordcloud.WordCloud().generate(' '.join(class_0['question_text']))
plt.imshow(wc)


# In[ ]:


wc1=wordcloud.WordCloud().generate(' '.join(class_1['question_text']))
plt.imshow(wc1)


# In[ ]:


nltk.corpus.stopwords.words('english')
stop_words=nltk.corpus.stopwords.words('english')
junk_words=["amp",'rt','https','will']
len(stop_words)
stop_words.extend(junk_words)
len(stop_words)


# In[ ]:


## cleaning the data
docs=data['question_text'].str.lower()
docs.head()
docs=docs.str.replace('[^a-z #@]','') # retain all alphabets with #@
docs.head()


# In[ ]:


stemmer=nltk.PorterStemmer()
def clean_text(row_text):
    #print(type(row_text))
    row_words=row_text.split(' ')
    #print(row_words)
    row_words= [stemmer.stem(word) for word in row_words if word not in stop_words]
    #print(row_words)
    #print('----')
    return ' '.join(row_words)

docs_clean=docs.apply(lambda v: clean_text(v))


# In[ ]:


data["clean_one"]=docs_clean


# In[ ]:


data.head()


# In[ ]:


df=data[["clean_one","target"]]


# In[ ]:


df.head()


# In[ ]:



from sklearn.model_selection import train_test_split

train,validate= train_test_split(docs_clean,test_size=0.3, random_state=100)
train_y=data.loc[train.index]["target"]
validate_y=data.loc[validate.index]["target"]


# In[ ]:


train.shape, validate.shape,train_y.shape,validate_y.shape


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer


# In[ ]:


cv=CountVectorizer()
cv.fit(train)
train_x_sparse=cv.transform(train)
validate_x_sparse=cv.transform(validate)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
dt_model=DecisionTreeClassifier(max_depth=20,random_state=100)
dt_model.fit(train_x_sparse,train_y)


# In[ ]:


pred_class=dt_model.predict(validate_x_sparse)


# In[ ]:


from sklearn.metrics import accuracy_score,f1_score,roc_curve,auc


# In[ ]:


print(accuracy_score(validate_y,pred_class))


# In[ ]:


print(f1_score(validate_y,pred_class))


# In[ ]:


pred_probs=pd.DataFrame(dt_model.predict_proba(validate_x_sparse),columns=['Sincere','Insincere'])
pred_probs.head()


# In[ ]:


fpr,tpr,thresholds=roc_curve(validate_y,pred_probs["Insincere"])
auc_dt=auc(fpr,tpr)
plt.plot(fpr,tpr)
plt.legend(["Decision Tree -AUC: %2f" % auc_dt])


# In[ ]:


test=pd.read_csv('../input/test.csv')
test_docs=test['question_text'].fillna('').str.lower()
test_docs=test_docs.str.replace('[^a-z #@]','')
test_docs_clean=test_docs.apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split(' ') if word not in stop_words]))
test_docs_clean.shape


# In[ ]:


test_x=cv.transform(test_docs_clean)
test_pred_class=dt_model.predict(test_x)
test_pred_class.shape


# In[ ]:


submission=pd.DataFrame({'qid':test['qid'],'prediction':test_pred_class})
submission.to_csv("submission.csv",index=False)
submission.head() 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




