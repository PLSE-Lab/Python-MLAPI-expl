#!/usr/bin/env python
# coding: utf-8

# ## Context
# This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to
# this date. The "goal" field refers to the presence of heart disease in the patient. It is integer valued from 0 (no presence) to 4.

# ## Content
# 
# ### Attribute Information:
# 
# 1. Age
# 2. Sex
# 3. Chest pain type (4 values)
# 4. Resting blood pressure
# 5. Serum cholestoral in mg/dl
# 6. Fasting blood sugar > 120 mg/dl
# 7. Resting electrocardiographic results (values 0,1,2)
# 8. Maximum heart rate achieved
# 9. Exercise induced angina
# 10. Oldpeak = ST depression induced by exercise relative to rest
# 11. The slope of the peak exercise ST segment
# 12. Number of major vessels (0-3) colored by flourosopy
# 13. thal: 3 = normal; 6 = fixed defect; 7 = reversable defect

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


path = '/kaggle/input/heart-disease-uci/heart.csv'


# In[ ]:


#load the dataset
data = pd.read_csv(path)
data.head()


# In[ ]:


data.tail()


# In[ ]:


data.columns


# In[ ]:


data.describe()


# In[ ]:


data.describe().T


# In[ ]:


data.info()


# # EDA

# In[ ]:


data['age'].value_counts()


# In[ ]:


plt.figure(figsize=(15,12))
sns.countplot(x=data['age'])


# In[ ]:


data['sex'].value_counts()


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(x=data['sex'])


# In[ ]:


data['cp'].value_counts()


# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(x=data['cp'])


# In[ ]:


data['trestbps'].value_counts()


# In[ ]:


plt.figure(figsize=(20,8))
sns.countplot(x=data['trestbps'])


# In[ ]:


data['fbs'].value_counts()


# In[ ]:


# plt.figure(figsize=(4,8))
sns.countplot(x=data['fbs'])


# In[ ]:


data['restecg'].value_counts()


# In[ ]:


sns.countplot(x=data['restecg'])


# In[ ]:


plt.figure(figsize=(20,8))
sns.countplot(x=data['thalach'])


# In[ ]:



data['thal'].value_counts()


# In[ ]:


plt.figure(figsize=(12,8))
sns.countplot(x=data['thal'])


# # Feature Selection

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
#get correlations of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[ ]:


data.hist(figsize=(20,10))


# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='target',data=data,palette='RdBu_r')


# # Data Processing

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
data[columns_to_scale] = standardScaler.fit_transform(data[columns_to_scale])


# In[ ]:


y = data['target']
X = data.drop(['target'], axis = 1)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(X, y,  test_size=0.20)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())


# In[ ]:


plt.figure(figsize=(20,8))
plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
for i in range(1,21):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1, 21)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')


# In[ ]:


knn_classifier = KNeighborsClassifier(n_neighbors = 12)
score=cross_val_score(knn_classifier,X,y,cv=10)


# In[ ]:


score.mean()


# # Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


randomforest_classifier= RandomForestClassifier(n_estimators=10)

score=cross_val_score(randomforest_classifier,X,y,cv=10)


# In[ ]:


score.mean()


# # Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
score = cross_val_score(tree, X, y, cv=40)


# In[ ]:


score.mean()


# # SVC

# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report
from sklearn.svm import SVC


# In[ ]:


clf = SVC()
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
confusion_matrix(y_pred,y_test)
print(classification_report(y_pred,y_test))


# # I am working on this dataset so I will keep updating it.
# 
# # <font color='red'> Consider an Upvote if you like it !</font>

# In[ ]:





# In[ ]:




