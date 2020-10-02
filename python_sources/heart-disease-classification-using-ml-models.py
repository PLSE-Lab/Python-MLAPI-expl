#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[ ]:


import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# plt.style.use('default')
color_pallete = ['#2a2a2a', '#ff0000']
sns.set_palette(color_pallete, 2)
sns.set_style("whitegrid")


# # Data

# In[ ]:


df = pd.read_csv('../input/heart.csv')
df.head()


# * AGE (age in years)  
# * SEX (1 = male; 0 = female)  
# * CP (chest pain type)  
# * TRESTBPS (resting blood pressure (in mm Hg on admission to the hospital))  
# * CHOL (serum cholestoral in mg/dl)  
# * FPS (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)  
# * RESTECH (resting electrocardiographic results)  
# * THALACH (maximum heart rate achieved)  
# * EXANG (exercise induced angina (1 = yes; 0 = no))  
# * OLDPEAK (ST depression induced by exercise relative to rest)  
# * SLOPE (the slope of the peak exercise ST segment)  
# * CA (number of major vessels (0-3) colored by flourosopy)  
# * THAL (3 = normal; 6 = fixed defect; 7 = reversable defect)  
# * TARGET (1 = presense or 0 = absence)

# In[ ]:


# data.describe()


# In[ ]:


# df.info()


# In[ ]:


# df.sample(5)


# # EDA

# In[ ]:


df.isna().sum()


# In[ ]:


sns.countplot(x='target', data=df)
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(), annot=True, fmt='.1f', cmap='RdBu', vmax=0.8, vmin=-0.8)
plt.show()


# In[ ]:


# plt.figure(figsize=(8, 8))
# sns.pairplot(df, hue="target")
# plt.plot()


# In[ ]:


cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
dis_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

for i in cat_cols:
    df[i] = df[i].astype('category')


# In[ ]:


for i in dis_cols:
    ax = sns.kdeplot(df[df['target']==1][i], shade=True, )
    ax = sns.kdeplot(df[df['target']==0][i], shade=True)
    ax.set_xlabel(i)
    plt.legend(['Absent', 'Present'])
    plt.show()


# In[ ]:


for i in cat_cols:
    sns.countplot(x=i, hue='target', data=df)
    ax.set_xlabel(i)
    plt.legend(['Absent', 'Present'])
    plt.show()


# In[ ]:


df['age_cat'] = pd.cut(df['age'], 
                       bins = [0, 40, 55, 90],
                       labels = ['young', 'mid', 'old'],
                       include_lowest=True)

ax = sns.countplot(x="age_cat", hue='target', data=df)
plt.plot()


# # Preprocessing

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

for i in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach' , 'exang', 'oldpeak', 'slope', 'ca', 'thal']:
    df[i] = df[i].astype('float64')
    df[i] =  sc.fit_transform(df[i].values.reshape(-1,1))
    
df.head()


# In[ ]:


df = pd.get_dummies(df, drop_first=True)
df.head()


# In[ ]:


df.columns


# # Model

# In[ ]:


X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal', 'age_cat_mid',
       'age_cat_old']]
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:


nbc = GaussianNB()
nbc.fit(X_train, y_train)
pred = nbc.predict(X_test)

print(accuracy_score(pred, y_test))
print(confusion_matrix(pred, y_test))


# In[ ]:


import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(nbc, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist(), )


# In[ ]:


# decision tree
# tree visualization
# auc, roc plot
# heatmap - confusion matrix
# neural network

