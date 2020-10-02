#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting
import seaborn as sns
sns.set_style('whitegrid')

# Input data files are available in the "../input/" directory.
# Any results you write to the current directory are saved as output.


# In[2]:


data = pd.read_csv("../input/train.csv")
data.shape


# In[3]:


data.info()


# In[4]:


data.head(10)


# In[8]:


data['Sex'].value_counts(normalize = True)


# In[9]:


# compare the counts of the two classes
_ = sns.countplot(data["Survived"])


# In[10]:


data.groupby(['Pclass', 'Sex']).mean()['Survived'].unstack()


# In[11]:


_ = data.groupby(['Pclass', 'Sex']).mean()['Survived'].unstack().plot.bar()


# In[12]:


# Select only the numeric and categorical columns
X=data[["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]]
X.head(6)


# In[13]:


# convert categorical to "dummy" and then fill NaN
X["Pclass"] = X["Pclass"].astype("category")
X=pd.get_dummies(X)
X = X.fillna(X.mean())
X.head(6)


# In[14]:


y=data["Survived"]
y[0:6]


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train.shape


# In[16]:


# Let's try a  decision tree
from sklearn.tree import DecisionTreeClassifier

DTClf = DecisionTreeClassifier(max_depth=3)
DTClf = DTClf.fit(X_train,y_train)


# In[17]:


X_test.head()


# In[18]:


y_test[0:5]


# In[19]:


# here's how you'd actually use it when "new" data comes in!
predicted = DTClf.predict(X_test)
predicted[0:5]


# In[20]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predicted)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax, fmt='g'); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['Died', 'Survived']); ax.yaxis.set_ticklabels(['Died', 'Survived']);


# In[21]:


# precision = tp / (tp + fp)
# recall = tp / (tp + fn) 
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test,predicted, average='binary')


# In[ ]:




