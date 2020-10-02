#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re

np.random.seed(42)


# In[ ]:


train = pd.read_excel('/kaggle/input/fooddeliverytime/Data_Train.xlsx')
train.head()


# In[ ]:


train.info()


# All the features are with datatype object

# # Data Preprocessing
# 
# ### Changing object to numeric

# In[ ]:


train['Average_Cost'] = pd.to_numeric(train['Average_Cost'].str.replace('[^0-9]',''))
train['Minimum_Order'] = pd.to_numeric(train['Minimum_Order'].str.replace('[^0-9]',''))
train['Delivery_Time'] = pd.to_numeric(train['Delivery_Time'].str.replace('minutes',''))

train.info()


# In[ ]:



train['Rating'] = pd.to_numeric(train['Rating'].apply(lambda x : np.NaN if x in ['-','NEW','Opening Soon', 'Temporarily Closed'] else x))
train.isna().mean()*100


# In[ ]:


train['Votes'] = pd.to_numeric(train['Votes'].apply(lambda x : np.NaN if x == '-' else x))
train['Reviews'] = pd.to_numeric(train['Reviews'].apply(lambda x : np.NaN if x == '-' else x))


# In[ ]:


train.info()


# In[ ]:


g = train.groupby('Restaurant')['Average_Cost'].mean().sort_values(ascending=False)
g= g.reset_index().sort_values(by='Average_Cost',ascending=False).head(10)
plt.figure(figsize=(6,5))
ax = sns.barplot(g['Restaurant'],g['Average_Cost'],palette='Blues_r')
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
plt.show()


# **Highest 'Average cost' of 10 restaurants**

# In[ ]:


g = train.groupby('Restaurant')['Reviews'].mean().sort_values(ascending=False)
g= g.reset_index().sort_values(by='Reviews',ascending=False).head(5)
ax = sns.barplot(g['Restaurant'],g['Reviews'])
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
plt.show()


# **Top 5 restaurants with highest reviews**

# In[ ]:


plt.figure(figsize=(8,6))
sns.heatmap(train.corr(),annot=True,cmap='Blues')
plt.show()


# **The heatmap shows that Votes,Reviews highly correlated with the target variable.**

# In[ ]:


sns.scatterplot(train['Reviews'],train['Delivery_Time'])
plt.show()


# In[ ]:


sns.scatterplot(train['Votes'],train['Delivery_Time'])
plt.show()


# In[ ]:


sns.countplot(train['Delivery_Time'])
plt.show()


# **Label encoding**

# In[ ]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

train['Restaurant'] = le.fit_transform(train['Restaurant'])
train['Location'] = le.fit_transform(train['Location'])
train['Rating'].fillna(round(train['Rating'].mean()),inplace=True)
train['Votes'].fillna(train['Votes'].mode()[0],inplace=True)
train['Reviews'].fillna(train['Reviews'].median(),inplace=True)
train['Average_Cost'].fillna(train['Average_Cost'].mean(),inplace=True)

train1 = train.copy()

train['Cuisines'] = le.fit_transform(train['Cuisines'])

train.info()


# # Standardization & Train-Test Split

# In[ ]:


from scipy.stats import zscore
from sklearn.model_selection import train_test_split

X = train.drop('Delivery_Time',axis=1)
y = train['Delivery_Time']

X = X.apply(zscore)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=50)


# # Model Building

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
ypred = rf.predict(X_test)

accuracy_score(y_test,ypred)


# With Random Forest Classifier, the accuracy score is 0.78

# In[ ]:


impf = pd.DataFrame({'Feature_importance':rf.feature_importances_,'Features':X.columns}).sort_values(by='Feature_importance',
                                                                                                            ascending=False)
ax = sns.barplot(impf['Features'],impf['Feature_importance'],palette="rocket")
ax.set_xticklabels(ax.get_xticklabels(),rotation=30)
sns.despine(bottom=True)
plt.show()


# A bar plot showing feature importance

# ### Trying to increase the accuracy score..
# 
# #### Using Countvectorizer on Cuisines

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()
vec = cv.fit_transform(train1['Cuisines']).toarray()
vec.shape


# In[ ]:


from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=5, random_state=42)
data = svd.fit_transform(vec) 
data.shape


# In[ ]:


train1['Cuisines'] = data

X = train1.drop('Delivery_Time',axis=1)
y = train1['Delivery_Time']

X = X.apply(zscore)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3,random_state=42)


# In[ ]:


rf1 = RandomForestClassifier()
rf1.fit(X_train,y_train)
ypred = rf1.predict(X_test)

accuracy_score(y_test,ypred)


# In[ ]:


from sklearn.metrics import confusion_matrix

sns.heatmap(confusion_matrix(y_test,ypred),annot=True,cmap='rainbow',fmt='d')


# In[ ]:




