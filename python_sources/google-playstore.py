#!/usr/bin/env python
# coding: utf-8

# **IMPORTING LIBRARIES**

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')


# In[ ]:


data.head()


# In[ ]:


data.shape


# **Thus there are 10841 rows and 13 columns**

# In[ ]:


data.nunique()


# **Let us check if any null values are present.**

# In[ ]:


data.isnull().sum()


# **Let us have a look at the outliers.**

# **We know that the maximum rating that is possible is 5. Let us look if every app meet this criteria.**

# In[ ]:


outlier=data[data['Rating']>5]
outlier


# **Let us drop this row.**

# In[ ]:


data.drop(10472,inplace=True)


# In[ ]:


data.hist(bins=40,color='fuchsia')
plt.show()


# **Let us compare the Free and Paid Apps.**

# In[ ]:


plt.figure(figsize=(5,5))
data['Type'].value_counts().plot.bar(color='black')
plt.show()


# **Let us look at the category wise distribution.**

# In[ ]:


plt.rcParams['figure.figsize']=(12,6)


# In[ ]:


data['Category'].value_counts().plot.bar()
plt.show()


# **Content Rating**

# In[ ]:


data['Content Rating'].value_counts().plot.bar(color='gray')


# **Rating vs Installs**

# In[ ]:


plt.figure(figsize=(20,20))
sns.barplot(x='Installs',y='Rating',data=data)
plt.show()


# **Rating vs Android Version**

# In[ ]:


plt.rcParams['figure.figsize']=(25,12)
sns.barplot(x='Android Ver',y='Rating',data=data)
plt.show()


# **Let us create dummy variables for Category,Type and Content Rating.**

# In[ ]:


category=pd.get_dummies(data['Category'],drop_first=True)
types=pd.get_dummies(data['Type'],drop_first=True)
content=pd.get_dummies(data['Content Rating'],drop_first=True)
new=[data,category,types]
data=pd.concat(new,axis=1)
data.drop(['Category','Installs','Type','Content Rating'],axis=1,inplace=True)


# In[ ]:


data.head()


# **Let us drop the columns which are not required.**

# In[ ]:


data.drop(['App','Size','Price','Genres','Last Updated','Current Ver','Android Ver'],axis=1,inplace=True)


# In[ ]:


data.head()


# **Model Building**

# In[ ]:


X=data.drop('Rating',axis=1)
y=data['Rating'].values
y=y.astype('int')


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=102)


# In[ ]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logr=LogisticRegression()


# In[ ]:


model=logr.fit(X_train,y_train)


# In[ ]:


prediction=model.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score


# In[ ]:


accuracy_score(y_test,prediction)


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_test,prediction)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtc=DecisionTreeClassifier(random_state=123,criterion='entropy')


# In[ ]:


model=dtc.fit(X_train,y_train)


# In[ ]:


prediction=model.predict(X_test)


# In[ ]:


accuracy_score(y_test,prediction)


# In[ ]:


confusion_matrix(y_test,prediction)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc=RandomForestClassifier(random_state=456)


# In[ ]:


model=rfc.fit(X_train,y_train)


# In[ ]:


prediction=model.predict(X_test)


# In[ ]:


accuracy_score(y_test,prediction)


# In[ ]:


confusion_matrix(y_test,prediction)

