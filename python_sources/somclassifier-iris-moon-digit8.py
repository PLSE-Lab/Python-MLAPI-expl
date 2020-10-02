#!/usr/bin/env python
# coding: utf-8

# * [Iris](#Iris)
# * [Moon Data](#Moon-Data)
# * [Digit 8x8](#Digit-8x8)
# 
# 
# see also [som and scikit-learn api](https://www.kaggle.com/wordroid/som-and-scikit-learn-api)

# In[1]:


get_ipython().system('pip install git+https://github.com/darecophoenixx/wordroid.sblo.jp')


# In[ ]:


from som import som


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import SVG


# In[ ]:


import random
import os

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score

from keras.utils import to_categorical

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:





# # Iris

# In[ ]:


iris_src = '../input/'


# In[ ]:


iris = pd.read_csv(os.path.join(iris_src, "Iris.csv"))
iris.head()


# In[ ]:


X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
print(X.shape)
X_sc = preprocessing.scale(X)
X_sc[:3]


# In[ ]:


target = [ee.replace('Iris-', '') for ee in iris['Species']]
target


# In[ ]:


le = preprocessing.LabelEncoder()
le.fit(target)
y = le.transform(target)
y


# ## no "rand_stat", use linear init

# In[ ]:


'''
no "rand_stat", use linear init
'''
sobj = som.SomClassifier((20, 30), it=(50,750), verbose=2)
sobj


# In[ ]:


y_cat = to_categorical(y)
y_cat[:5]


# In[ ]:


sobj.fit(X_sc, y_cat)


# In[ ]:


sobj._estimator_type


# In[ ]:


sobj.score(X_sc, y)


# In[ ]:


lw = 2
plt.plot(np.arange(len(sobj.sksom.meanDist)), sobj.sksom.meanDist, label="mean distance to closest landmark",
             color="darkorange", lw=lw)
plt.legend(loc="best")


# In[ ]:


'''
initial landmarks
'''
img = som.conv2img(sobj.init_lm, (20, 30), target=(4,5,6))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.predict(X_sc)


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.knn.kneighbors(X_sc, 1, return_distance=False).flatten()):
    b, a = divmod(m, sobj.sksom.kshape[1])
    if target[i] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif target[i] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=[3,4,5])
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=[3,4,5])
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.knn.kneighbors(X_sc, 1, return_distance=False).flatten()):
    b, a = divmod(m, sobj.sksom.kshape[1])
    if target[i] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif target[i] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


print(classification_report(y, sobj.predict(X_sc)))
confusion_matrix(y, sobj.predict(X_sc))


# In[ ]:





# ## iris 2 features and sphere
# * r1=None, r2=(1.0, 0.3)
# * alpha=1.5 : use learning rate bigger than one

# In[ ]:


k_shape = (50, 30)
sobj = som.SomClassifier(k_shape, it=(20,1500), alpha=1.5, r2=(1.0, 0.3), verbose=2, form='sphere')
sobj


# In[ ]:


y_cat = to_categorical(y)
y_cat[:5]


# In[ ]:


sobj.fit(X_sc[:,:2], y_cat)


# In[ ]:


lw = 2
plt.plot(np.arange(len(sobj.sksom.meanDist)), sobj.sksom.meanDist, label="mean distance to closest landmark",
             color="darkorange", lw=lw)
plt.legend(loc="best")


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, k_shape)
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.predict(X_sc[:,:2])


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, k_shape)
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.knn.kneighbors(X_sc[:,:2], 1, return_distance=False).flatten()):
    b, a = divmod(m, sobj.sksom.kshape[1])
    if target[i] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif target[i] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, k_shape, target=[2,3,4])
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, k_shape, target=[2,3,4])
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.knn.kneighbors(X_sc[:,:2], 1, return_distance=False).flatten()):
    b, a = divmod(m, sobj.sksom.kshape[1])
    if target[i] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif target[i] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


df1= pd.DataFrame(sobj.sksom.landmarks_[:,3:])
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X_sc[:,:2])
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
df.head()
df.shape
sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, diag_kind='hist')


# In[ ]:


print(classification_report(y, sobj.predict(X_sc[:,:2])))
confusion_matrix(y, sobj.predict(X_sc[:,:2]))


# In[ ]:





# # Moon Data

# In[ ]:


n_samples = 1500

X, y = datasets.make_moons(n_samples=n_samples, noise=.15, random_state=0)
df = pd.DataFrame(X)
df.columns = ["col1", "col2"]
df['cls'] = y

sns.lmplot("col1", "col2", hue="cls", data=df, fit_reg=False, height=8)


# In[ ]:


sobj = som.SomClassifier((20, 30), it=(20,750), alpha=1.5, r2=1.0, verbose=2)
sobj


# In[ ]:


sobj.fit(X, y)


# In[ ]:


'''
initial landmarks
'''
img = som.conv2img(sobj.init_lm, (20, 30), target=[0,1])
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,0])


# In[ ]:


df1= pd.DataFrame(sobj.init_lm[:,1:])
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
print(df.shape)
df.head()
sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, height=5, diag_kind='hist')


# In[ ]:


lw = 2
plt.plot(np.arange(len(sobj.sksom.meanDist)), sobj.sksom.meanDist, label="mean distance to closest landmark",
             color="darkorange", lw=lw)
plt.legend(loc="best")


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=[0,1])
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,0])


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=[0,1])
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,1])


# In[ ]:


sobj.predict(X) # must "predict()" first
sobj.knn.kneighbors(X, 1, return_distance=False).flatten()


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,(0,1,1)])

for i, m in enumerate(sobj.knn.kneighbors(X, 1, return_distance=False).flatten()):
    b, a = divmod(m, sobj.kshape[1])
    plt.text(a, b, str(y[i]), ha='center', va='center',
           bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))


# In[ ]:


print(classification_report(y, sobj.predict(X)))
confusion_matrix(y, sobj.predict(X))


# In[ ]:


df1= pd.DataFrame(sobj.sksom.landmarks_[:,1:])
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
df.columns = ['col1', 'col2', 'cls']
df.head()
#df.shape

sns.lmplot("col1", "col2", hue="cls", data=df, fit_reg=False, height=8, scatter_kws={'alpha': 0.5})


# In[ ]:


'''
default n_neighbors = 5
'''
from matplotlib.colors import ListedColormap

h = .01
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
y_pred = sobj.predict_proba(np.c_[xx.ravel(), yy.ravel()])
y_pred

cm = plt.cm.coolwarm
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
y_pred = y_pred.reshape(xx.shape)
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, y_pred, 100, cmap=cm, alpha=1)
plt.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, edgecolors='k')


# In[ ]:


sksom02 = sobj.sksom


# ### n_neighbors = 1

# In[ ]:


from sklearn.neighbors import NearestNeighbors
sobj = som.SomClassifier((20, 30), it=(20,750), sksom=sksom02, knn=NearestNeighbors(n_neighbors=1), verbose=1)
sobj


# In[ ]:


sobj.fit(X, y)


# In[ ]:


print(classification_report(y, sobj.predict(X)))
confusion_matrix(y, sobj.predict(X))


# In[ ]:


from matplotlib.colors import ListedColormap

h = .01
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
y_pred = sobj.predict_proba(np.c_[xx.ravel(), yy.ravel()])
y_pred

cm = plt.cm.coolwarm
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
y_pred = y_pred.reshape(xx.shape)
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, y_pred, 100, cmap=cm, alpha=1)
plt.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, edgecolors='k')


# ### n_neighbors = 15

# In[ ]:


sobj = som.SomClassifier((20, 30), it=(20,750), sksom=sksom02, knn=NearestNeighbors(n_neighbors=15), verbose=2)
sobj


# In[ ]:


sobj.fit(X, y)


# In[ ]:


print(classification_report(y, sobj.predict(X)))
confusion_matrix(y, sobj.predict(X))


# In[ ]:


h = .01
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
y_pred = sobj.predict_proba(np.c_[xx.ravel(), yy.ravel()])
y_pred

cm = plt.cm.coolwarm
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
y_pred = y_pred.reshape(xx.shape)
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, y_pred, 100, cmap=cm, alpha=1)
plt.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, edgecolors='k')


# ### RandomForestClassifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf


# In[ ]:


clf.fit(X, y)


# In[ ]:


print(classification_report(y, clf.predict(X)))
confusion_matrix(y, clf.predict(X))


# In[ ]:


from matplotlib.colors import ListedColormap

h = .01
x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
y_pred = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
y_pred

cm = plt.cm.coolwarm
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
y_pred = y_pred[:,1].reshape(xx.shape)
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, y_pred, 100, cmap=cm, alpha=1)
plt.scatter(X[:,0], X[:,1], c=y, cmap=cm_bright, edgecolors='k')


# In[ ]:





# # Digit 8x8

# In[ ]:


digits = datasets.load_digits()
X, y = digits.data, digits.target
X[:5]


# In[ ]:


X = X.reshape((X.shape[0], -1))
X[:5]


# In[ ]:


X_sc = X / 16.0
X_sc.shape


# In[ ]:


y_cat = to_categorical(y)
y_cat.shape


# In[ ]:


sobj = som.SomClassifier((20, 30), it=(15,750), verbose=2, alpha=1.5)
sobj


# In[ ]:


y_cat = to_categorical(y)
y_cat[:5]


# In[ ]:


sobj.fit(X_sc, y_cat)


# In[ ]:


lw = 2
plt.plot(np.arange(len(sobj.sksom.meanDist)), sobj.sksom.meanDist, label="mean distance to closest landmark",
             color="darkorange", lw=lw)
plt.legend(loc="best")


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)
sobj.predict(X_sc)

for i, m in enumerate(sobj.knn.kneighbors(X_sc, 1, return_distance=False).flatten()):
    b, a = divmod(m, sobj.sksom.kshape[1])
    plt.text(a, b, y[i], ha='center', va='center',
           bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=(3,4,5))
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.knn.kneighbors(X_sc, 1, return_distance=False).flatten()):
    b, a = divmod(m, sobj.sksom.kshape[1])
    plt.text(a, b, y[i], ha='center', va='center',
           bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=(6,7,8))
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.knn.kneighbors(X_sc, 1, return_distance=False).flatten()):
    b, a = divmod(m, sobj.sksom.kshape[1])
    plt.text(a, b, y[i], ha='center', va='center',
           bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=[11,12,13])
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=[11,12,13])
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.knn.kneighbors(X_sc, 1, return_distance=False).flatten()):
    b, a = divmod(m, sobj.sksom.kshape[1])
    plt.text(a, b, str(y[i]), ha='center', va='center',
           bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))


# In[ ]:


print(classification_report(y, sobj.predict(X_sc)))
confusion_matrix(y, sobj.predict(X_sc))


# In[ ]:





# In[ ]:





# In[ ]:




