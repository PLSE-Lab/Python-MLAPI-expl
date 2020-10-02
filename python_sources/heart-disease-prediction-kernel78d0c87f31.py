#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction
# 
# 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


df = pd.read_csv('../input/heart-disease-dataset/Heart Disease Dataset.csv')
df.head(5)


# In[ ]:


df.info()


# In[ ]:


df.describe()


# ## Feature Selection

# In[ ]:


import seaborn as sns
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[ ]:


#plt.figure(figsize=(10,10))
df.hist(figsize=(10,10))


# It's always a good practice to work with a dataset where the target classes are of approximately equal size. Thus, let's check for the same.

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')


# ### Data Processing
# 
# After exploring the dataset, I observed that I need to convert some categorical variables into dummy variables and scale all the values before training the Machine Learning models.
# First, I'll use the `get_dummies` method to create dummy columns for categorical variables.

# In[ ]:


dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()

columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


# In[ ]:


dataset.head()


# In[ ]:


y = dataset['target']
X = dataset.drop(['target'], axis = 1)


# In[ ]:


from sklearn.model_selection import cross_val_score
knn_scores = []

for k in range(1,21):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(score.mean())


# In[ ]:


plt.figure(figsize=(15,15))
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


# ## Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


randomforest_scores=[]
#estimators= (i for i in range(1,20))
for n_est in range(1,20):
    randomforest_classifier= RandomForestClassifier(n_estimators= n_est)
    
    score=cross_val_score(randomforest_classifier,X,y,cv=10)


# In[ ]:


score.mean()


# In[ ]:


randomforest_scores.append(score.mean())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




