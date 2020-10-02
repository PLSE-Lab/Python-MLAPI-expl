#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import nltk
#read csv file
df=pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv",encoding="latin")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df=df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)
df.head()


# In[ ]:


df_copy=df['v2'].copy()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
vectorizer.fit(df_copy)
vector = vectorizer.transform(df_copy)


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[ ]:


v2_train,v2_test,v1_train,v1_test=train_test_split(vector,df['v1'],test_size=0.5,random_state=20)


# In[ ]:


Spam_model=LogisticRegression(solver='liblinear')
Spam_model.fit(v2_train,v1_train)
v1_pred=Spam_model.predict(v2_test)
m=accuracy_score(v1_test,v1_pred)
print(accuracy_score)
print(m)


# In[ ]:


df1=pd.DataFrame{{'Actual':v1_test,'Predicted':v1_pred}}
print(df1.head(40))


# In[ ]:


from matplotlib import pyplot as plt
plt.plot(v1_test,v1_pred,colors='red',linewidth=2)
plt.show()


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
results = confusion_matrix(v1_test, v1_pred)
print(results)
confusion_matrix = pd.crosstab(v1_test, v1_pred, rownames=['Actual'], colnames=['Predicted'])
sns.heatmap(confusion_matrix, annot =True ,fmt='g')
plt.title("confusion_matrix")
plt.show()


# In[ ]:




