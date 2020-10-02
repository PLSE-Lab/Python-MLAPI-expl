#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


# **Loading Data**

# In[ ]:


train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")
sub_df = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# **Analysis**

# In[ ]:


train_df.isnull()


# **Target Distribution**

# In[ ]:


x=train_df.target.value_counts()
sns.barplot(x.index,x)
plt.gca().set_ylabel('samples')


# *Seperating target and other data*

# In[ ]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
train_df_len1 = train_df[train_df['target']==1]['text'].str.len()
ax1.hist(train_df_len1,color='red')
ax1.set_title('disaster tweets')
train_Df_len2 = train_df[train_df['target']==0]['text'].str.len()
ax2.hist(train_Df_len2,color = 'green')
ax2.set_title('Non Disaster Tweets')


# In[ ]:


def create_corpus(target):
    corpus=[]
    
    for x in train_df[train_df['target']==target]['text'].str.split():
        for i in x:
            corpus.append(i)
    return corpus


# In[ ]:


corpus=create_corpus(0)

dic=defaultdict(int)
for word in corpus:
    if word in stop:
        dic[word]+=1
        
top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10] 


# In[ ]:


for k in range(0,len(top)):
    print(top[k][0])


# In[ ]:


x,y = zip(*top)
print(x)
plt.bar(x,y)


# In[ ]:


x = train_df["text"]
y = train_df["target"]


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[ ]:


X_train


# In[ ]:


vect = CountVectorizer(stop_words = 'english')

x_train_cv = vect.fit_transform(X_train)
x_test_cv = vect.transform(X_test)


# In[ ]:


x_train_cv


# In[ ]:


clf = MultinomialNB()
clf.fit(x_train_cv, y_train)


# In[ ]:


pred = clf.predict(x_test_cv)


# In[ ]:


pred


# In[ ]:


y_test


# In[ ]:


# y_pred = model.predict(X_test)

f1score = f1_score(y_test,pred)
print(f"Model Score: {f1score * 100} %")


# In[ ]:


confusion_matrix(y_test, pred)


# In[ ]:


accuracy_score(y_test,pred)


# In[ ]:


y_test = test_df["text"]
y_test_cv = vect.transform(y_test)
preds = clf.predict(y_test_cv)


# In[ ]:


sub_df["target"] = preds
sub_df.to_csv("submission.csv",index=False)


# In[ ]:


sub_df.describe()


# In[ ]:





# In[ ]:





# In[ ]:


predicted = rf.predict(x_test_cv)
print(predicted)


# In[ ]:


y_test1 = test_df["text"]
y_test_cv = vect.transform(y_test1)
preds = rf.predict(y_test_cv)


# In[ ]:


sub_df["target"] = preds
sub_df.to_csv("submission_final.csv",index=False)


# In[ ]:




