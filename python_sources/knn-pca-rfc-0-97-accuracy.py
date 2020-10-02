#!/usr/bin/env python
# coding: utf-8

# # 1. Data preparation

# ## 1.1 Load data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score


# In[ ]:


train=pd.read_csv("../input/digit-recognizer/train.csv")
test=pd.read_csv("../input/digit-recognizer/test.csv")


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


train.head()


# In[ ]:


#check labels
train.label.unique()


# ## 1.2 Check missing values

# In[ ]:


y_train=train["label"]
train.drop("label",axis=1,inplace=True)
sns.countplot(y_train)


# In[ ]:


#check missing values
train.isnull().values.any()


# # 2. Visulization

# In[ ]:


#plot first 20 digits
fig,axes=plt.subplots(4,5,figsize=(6,4),subplot_kw={"xticks":[],"yticks":[]})
for i,ax in enumerate(axes.flat):
    ax.imshow(train.values[i,:].reshape(28,28))
plt.show()


# # 3. PCA

# * Use cumulative explained variance ratio to check the range of components we should use for dimension reduction.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'X=train.values\ny=y_train.values\npca_line=PCA().fit(X)\nplt.figure(figsize=[20,5])\nplt.plot(np.cumsum(pca_line.explained_variance_ratio_))\nplt.xlabel("number of components after dimension reduction")\nplt.ylabel("cumulative explained variance ratio")\nplt.show()')


# * It is obvious that the trend increases dramatically when the number of components range from 0 to 100 and the trend flattens after the number of components beyond 100.
# * Therefore, we need to draw a similar plot to narrow the range of components from 1 to 100 in order to find the most suitable component with the highest cumulative explained variance ratio.

# In[ ]:


score=[]
for i in range(1,101,10):
    X_dr=PCA(i).fit_transform(X)
    once=cross_val_score(RFC(n_estimators=20,random_state=0),X_dr,y,cv=5).mean()
    score.append(once)
plt.figure(figsize=(10,5))
plt.plot(range(1,101,10),score)
plt.show()


# * From the above plot,we can notice that RandomForestClassifier preforms great when the n_components of PCA ranges from 20 to 100.
# * We should not ignore the turning point which is around 20 to 30.Thus we narrow the n_components again to find the best one.

# In[ ]:


score=[]
for i in range(20,30):
    X_dr=PCA(i).fit_transform(X)
    once=cross_val_score(RFC(n_estimators=20,random_state=0),X_dr,y,cv=5).mean()
    score.append(once)
plt.figure(figsize=(10,5))
plt.plot(range(20,30),score)
plt.show()


# * It is so great that we can use just 30 features but get almost 0.94 accuracy,which means that the dimensionality reduction is successful.I think if we try to use grid search to adjust parameters for RandomForestClassifier,we can get a better score.But now I want to change another model,KNN.

# # 4. KNN

# In[ ]:


#PCA+KNN
score=[]
for i in range(10):
    X_dr=PCA(30).fit_transform(X)
    once=cross_val_score(KNN(i+1),X_dr,y,cv=5).mean()
    score.append(once)
plt.figure(figsize=(10,5))
plt.plot(range(10),score)
plt.show()


# * We can notice that when K is 3, the accuracy for cross_val_score beyond 0.97.

# In[ ]:


pca=PCA(n_components=30)
pca.fit(X)
train_dr=pca.transform(X)
test_dr=pca.transform(test)
clf=KNN(3)
clf.fit(train_dr,y)
results=clf.predict(test_dr)
results=pd.Series(results,name="Label")


# In[ ]:


#accuracy 0.97
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submission.to_csv("knn_digit_recognizer.csv",index=False)


# Summary:
# * PCA performs very well on this dataset,we can use few features to get a very high accuracy,even though I just used very basic models.
# * Cumulative explained variance ratio is very useful and important,we can use it to decide the most suitable n_components for PCA.
# * I just used very basic models this time, and I think with more complex models the accuracy can be improved.

#  **If you think this notebook is helpful,some upvotes would be very much appreciated.**
