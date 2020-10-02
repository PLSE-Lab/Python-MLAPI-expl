#!/usr/bin/env python
# coding: utf-8

# # Dimensionality Reduction and Classification Analysis

# ### Importing Library

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB


# ### Reading CSV

# In[ ]:


df=pd.read_csv('../input/Wine.csv')


# ### EDA

# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.info()


# #### no null values

# In[ ]:


correlation=df.corr()
plt.figure(figsize=(25,25))
sns.heatmap(correlation,annot=True,cmap='coolwarm')


# In[ ]:


X=df.drop('Customer_Segment',axis=1)
y=df['Customer_Segment']


# ### Scaling the features:

# In[ ]:


sc=StandardScaler()
X=sc.fit_transform(X)


# #### Train test split

# In[ ]:


(X_train,X_test,Y_train,Y_test)=train_test_split(X,y,test_size=0.30)


# # Principal Component Analysis

# In[ ]:


pca=PCA(0.95)
pca.fit(X_train)
pca.explained_variance_ratio_


# ## taking 3 principal components because explained variance not good enough with just first two components , to get ~0.95 variance required components=10 but then visualization gets impossible and contribution from later components is insignificant thus, we can drop those:

# In[ ]:


pca=PCA(3)
pca.fit(X_train)
pca.explained_variance_ratio_


# In[ ]:


pca_train=pca.transform(X_train)
pca_test=pca.transform(X_test)


# ### Training set scatter plot:

# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pca_train[Y_train==1,0],pca_train[Y_train==1,1],pca_train[Y_train==1,2], c='red', marker='x')
ax.scatter(pca_train[Y_train==2,0],pca_train[Y_train==2,1],pca_train[Y_train==2,2], c='blue', marker='o')
ax.scatter(pca_train[Y_train==3,0],pca_train[Y_train==3,1],pca_train[Y_train==3,2], c='green', marker='^')

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')

plt.show()


# ### Test set scatter plot

# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(pca_test[Y_test==1,0],pca_test[Y_test==1,1],pca_test[Y_test==1,2], c='red', marker='x')
ax.scatter(pca_test[Y_test==2,0],pca_test[Y_test==2,1],pca_test[Y_test==2,2], c='blue', marker='o')
ax.scatter(pca_test[Y_test==3,0],pca_test[Y_test==3,1],pca_test[Y_test==3,2], c='green', marker='^')

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')

plt.show()


# ## Since principal components are orthogonal thus, best approach should be Naive Bayes classification with it's independence assumption:

# In[ ]:


gnb = GaussianNB()
gnb.fit(pca_train,Y_train)


# In[ ]:


Ypreds=gnb.predict(pca_test)


# In[ ]:


gnb.score(pca_test,Y_test)


# ### 10-fold cross validated score :

# In[ ]:


scores = cross_val_score(gnb, pca_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())


# ## Accuracy with PCA and GNB

# In[ ]:


cm = confusion_matrix(Y_test,Ypreds)
xy=np.array([1,2,3])
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True,square=True,cmap='coolwarm',xticklabels=xy,yticklabels=xy)


# # Linear Discriminant Analysis

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis()
lda_train = lda.fit(X_train, Y_train)
lda.explained_variance_ratio_


# ### when n_components is left to default it chooses components which sum up to a 100% variance , and it's just 2 components with LDA approoach and still it captures 100% variance of data

# In[ ]:


lda_train=lda.transform(X_train)
lda_test=lda.transform(X_test)


# ### Scatter plot to visualize training set:

# In[ ]:


plt.scatter(lda_train[Y_train==1,0],lda_train[Y_train==1,1], c='red', marker='x')
plt.scatter(lda_train[Y_train==2,0],lda_train[Y_train==2,1], c='blue', marker='o')
plt.scatter(lda_train[Y_train==3,0],lda_train[Y_train==3,1], c='green', marker='^')

plt.xlabel('LD1')
plt.ylabel('LD2')

plt.show()


# ### Scatter plot to visualize test set:

# In[ ]:


plt.scatter(lda_test[Y_test==1,0],lda_test[Y_test==1,1], c='red', marker='x')
plt.scatter(lda_test[Y_test==2,0],lda_test[Y_test==2,1], c='blue', marker='o')
plt.scatter(lda_test[Y_test==3,0],lda_test[Y_test==3,1], c='green', marker='^')

plt.xlabel('LD1')
plt.ylabel('LD2')

plt.show()


# ## As we can see LDA has done a great job of maximizing separation between classes:
# #### On a data like this with so well separated classes any multi class classifier should do a great job
# ## Now to check classification score with the discriminate function:
# #### which is just a modified form of bayes theorem with gaussian distribution function for estimated probability of x belonging to the class

# In[ ]:


Ypreds=lda.predict(X_test)
lda.score(X_test,Y_test)


# In[ ]:


scores = cross_val_score(lda, lda_train, Y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())


# ## 100% accuracy , much better than score from PCA 

# In[ ]:


cm = confusion_matrix(Y_test,Ypreds)
xy=np.array([1,2,3])
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True,square=True,cmap='coolwarm',xticklabels=xy,yticklabels=xy)

