#!/usr/bin/env python
# coding: utf-8

# # Fun with Pokemon Dataset
# 
# ## Thanks for checking it out, leave a comment.
# ------

# ## Notebook Preparation
# 
# Importing Dependencies

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('max.columns', None)
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
from sklearn import metrics
from sklearn.metrics import roc_curve, f1_score, accuracy_score, precision_recall_curve, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Data Importing

# In[ ]:


df = pd.read_csv('../input/Pokemon.csv', low_memory=False)


# ## Basic EDA

# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


## Percentage of Legendaries in the dataset
print('Legendary:',str(len(df[df['Legendary'] == True]) / len(df) * 100) + '%')


# ------

# ## Data Visualization

# ### Pokemon Type 1 count plot

# In[ ]:


plt.title('Count Plot')
plt.xticks(rotation = 45)
sns.countplot(df['Type 1'])

# Expected Fire type to be the highest


# ### Pokemon Type 2 count plot

# In[ ]:


plt.title('Count Plot')
plt.xticks(rotation = 45)
sns.countplot(df['Type 2'])


# ### Distribution plot of Pokemon Total

# In[ ]:


sns.distplot(df['Total'])


# In[ ]:


## Break down of the Generations
df['Generation'].value_counts()


# ## Pair-plot to understand linear relationships

# In[ ]:


sns.pairplot(df[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']])


# -------

# ## Correlation Matrix

# In[ ]:


corr = df.corr()


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(corr, cmap='coolwarm', annot=True)


# -------

# ## More EDA

# In[ ]:


df.describe()


# In[ ]:


df[df['Name'].duplicated()] # no dupliactes


# In[ ]:


pd.crosstab(df['Type 1'] , df['Legendary'])


# In[ ]:


for i in df.columns:
    print(i, len(df[i].unique()))


# -------

# ## Data Transformation

# In[ ]:


df['Legendary'] = df['Legendary'].apply(lambda x: 1 if x == True else 0)


# In[ ]:


dataset = df.iloc[:, 2:]


# In[ ]:


dataset.head()


# In[ ]:


dataset = pd.get_dummies(dataset, dummy_na=True,drop_first=True)
dataset['Target'] = dataset['Legendary']
dataset.drop(['Legendary', 'Total'], inplace=True, axis=1)


# ------

# ## Machine Learning

# In[ ]:


X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


# In[ ]:


y.head(2)


# In[ ]:


X.head()


# In[ ]:


X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# In[ ]:


y_train.shape


# In[ ]:


y_test.shape


# In[ ]:


clr = LogisticRegression()


# In[ ]:


clr.fit(X_train, y_train)


# In[ ]:


y_pred = clr.predict(X_test)


# In[ ]:


accuracy_score(y_test, y_pred)


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:


cm = confusion_matrix(y_test, y_pred)


# In[ ]:


cm


# In[ ]:


probs = clr.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)


# In[ ]:


## SVC


# In[ ]:


svc = SVC(probability=True)


# In[ ]:


svc.fit(X_train, y_train)


# In[ ]:


svc_probs = svc.predict_proba(X_test)
svc_preds = svc_probs[:,1]
svc_fpr, svc_tpr, svc_threshold = metrics.roc_curve(y_test, svc_preds)
svc_roc_auc = metrics.auc(svc_fpr, svc_tpr)


# In[ ]:


svc_y_pred = svc.predict(X_test)


# In[ ]:


accuracy_score(y_test, svc_y_pred)


# In[ ]:


tpr


# In[ ]:


svc_tpr


# In[ ]:


cm


# In[ ]:


svc_cm = confusion_matrix(y_test, svc_y_pred)


# In[ ]:


svc_cm


# In[ ]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'LR AUC = %0.2f' % roc_auc)
plt.plot(svc_fpr, svc_tpr, 'g', label = 'SVC AUC = %0.2f' % svc_roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# #### Thank You!

# In[ ]:




