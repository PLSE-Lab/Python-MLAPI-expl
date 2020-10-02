#!/usr/bin/env python
# coding: utf-8

# # Welcome! Let's Discuss Dimensionality Reduction Techniques using Python & Sklearn

# In this notebook, I'll walk through the dimensionality reduction techniques namely PCA, t-SNE and LDA. <br>
# The dataset used is **Breast Cancer Winconsin Dataset**. Python Library Used -> **Sklearn**. <br>
# ## If you liked my work, please support my efforts my giving an UPVOTE & please add your valuable comments in the comment section.

# ### Setting up Dataset for the notebook

# In[ ]:


data_url = 'https://raw.githubusercontent.com/pkmklong/Breast-Cancer-Wisconsin-Diagnostic-DataSet/master/data.csv'


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv(data_url)
df.head()


# In[ ]:


df.columns


# In[ ]:


sns.countplot(df['diagnosis'])
plt.show()


# In[ ]:


df.drop(['Unnamed: 32'], axis = 1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


df.drop(['id'], axis = 1, inplace = True)


# In[ ]:


df.head()


# In[ ]:


df.columns


# In[ ]:


X = df.iloc[:, 1:].values
y = df['diagnosis'].values


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 0)


# ### Feature Scaling

# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# We are ready to dive in to perform dimensionality reduction as our dataset is in the format we wanted.

# # Let's Begin -> PCA, t-SNE & LDA

# ## PCA (Principal Component Analysis)

# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(n_components = 1)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)


# #### Training and Making Predictions

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth = 2, random_state = 0)
clf.fit(X_train, y_train)

# Predicting the Test set results
y_pred = clf.predict(X_test)


# #### Evaluation

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy -> '+ str(accuracy_score(y_test, y_pred)))


# ### Conclusion
# 

# PCA has no concern with the class labels. It summarizes the feature set without considering the output. PCA tries to find the directions of the maximum variance in the dataset. In a high cardinality feature set, there are possibilities of duplicate features which would add redundancy to the dataset, increase the computation cost and add unneccessary model complexity. **The role of PCA is to find such highly correlated or duplicate features and to come up with a new feature set where there is minimum correlation between the features or in other words feature set with maximum variance between the features.**

# ## t-SNE (t-Distributed Stochastic Neighbor)

# **NOTE->** Make sure you don't reuse the X_train and X_test as they were transformed. We'll resume from X and y.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 0)


# In[ ]:


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from sklearn.manifold import TSNE


# In[ ]:


tsne = TSNE(n_components = 2, random_state = 0)


# In[ ]:


tsne_obj = tsne.fit_transform(X_train)


# In[ ]:


tsne_df = pd.DataFrame({'X' : tsne_obj[:,0],
                       'Y' : tsne_obj[:,1],
                        'classification' : y_train
                       })


# In[ ]:


tsne_df.head()


# In[ ]:


tsne_df['classification'].value_counts()


# In[ ]:


plt.figure(figsize = (10,10))
sns.scatterplot(x = 'X', y = 'Y', data = tsne_df)
plt.show()


# We have obtained the plot of the data points but are unable to segregate. Let's introduce hue

# In[ ]:


plt.figure(figsize = (10,10))
sns.scatterplot(x = "X", y = 'Y', hue = 'classification', legend = 'full', data = tsne_df)
plt.show()


# ## LDA (Linear Discriminant Analysis)

# ### Performing LDA (Linear Discriminant Analysis)

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state= 0)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components = 1)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)


# #### Let's perform classification using RandomForestClassifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(max_depth = 2, random_state = 0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


# #### Performance Evaluation

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)

print('Accuracy -> ' + str(accuracy_score(y_test, y_pred)))


# # What's the difference from PCA?
# LDA tries to reduce the dimensionality by taking into consideration the information that discriminates the output classes. LDA tries to find the decision boundary around each cluster of class. <br><br>
# It projects the data points to new dimension in a way that the clusters are as seperate from each other as possible and individual elements within a class are as close to the centroid as possible. <br><br>
# **In other words, the inter-class seperability is increased in LDA. Intra-class seperability is reduced.** <br>
# The new dimensions are the linear discriminants of the feature set.

# # If you liked the Notebook, please support my work by giving an UPVOTE. And please add your valuable comments in the comment section.
