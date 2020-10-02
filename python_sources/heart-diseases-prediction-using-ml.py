#!/usr/bin/env python
# coding: utf-8

# ### Heart Diseases Prediction 
# I have collected this dataset from Kaggale ('https://www.kaggle.com/ronitf/heart-disease-uci')
# I analyzed the data in two Machine Learning Algorithosms & also derived which model gives the best prediction about the probabilty of heart diseases. 
# Two ML Algorithms:
# 1. KNeighboursClassifier
# 2. Random Forest Classifier
# 
# I'm also trying to apply rest of the ML algorithm for this data set.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


df = pd.read_csv('C:/Users/USER/Downloads/Predicting-Heart-Disease-master/Predicting-Heart-Disease-master/dataset.csv')


# df.info()

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


df.hist()


# It's always a good practice to work with a dataset where the target classes are of approximately equal size. Thus, let's check for the same.

# In[ ]:


sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')


# ### Data Processing
# 
# Categorical variables are converted into dummy variable & as these variables have different different values they are scaled down.

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


randomforest_classifier= RandomForestClassifier(n_estimators=10)

score=cross_val_score(randomforest_classifier,X,y,cv=10)


# In[ ]:


score.mean()


# So, for this data set KNeighbour Classifier is more efficient than Random Forest Classifier. 
# 
