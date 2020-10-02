#!/usr/bin/env python
# coding: utf-8

# sksom is scikit-learn api for SimpleSOM. This class has method below
# * fit
# * predict
# * predict_proba
# 
# And attributes  labels_ too.  
# 
# see also  
# * [iris demo](https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/demo/som/demo_iris_001.ipynb)  
# * [som with scikit-learn manner](https://github.com/darecophoenixx/wordroid.sblo.jp/blob/master/demo/som/demo_iris_002.ipynb)
# 

# # Contents
# * [Iris](#Iris)
# * [Moon Data](#Moon-Data)
# * [Digit (8x8)](#Digit-8x8)
# * [Don't Overfit! II data](#Don't-Overfit!-II-data)

# In[ ]:


get_ipython().system('pip install git+https://github.com/darecophoenixx/wordroid.sblo.jp')


# In[ ]:


from som import som


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import SVG


# # Iris

# In[ ]:


import random

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import preprocessing
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


iris = datasets.load_iris()
print(dir(iris))
X = iris.data
print(iris.target_names)
X_sc = preprocessing.scale(X)


# In[ ]:


X_sc[:3]


# In[ ]:


iris.target


# ### prepare initial landmarks (random)
# We can obtain initial landmarks using SimpleSOM.

# In[ ]:


sobj_makeK = som.SimpleSOM((20, 30))
sobj_makeK


# In[ ]:


sobj_makeK._initialize(X_sc)
sobj_makeK.K.shape


# In[ ]:


img = som.conv2img(sobj_makeK.K, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


df = pd.DataFrame(sobj_makeK.K)
sns.pairplot(df, markers='.')


# ### prepare initial landmarks (linear)
# We can also obtain initial landmarks using SimpleSOM.

# In[ ]:


sobj_makeK = som.SimpleSOM((20, 30), initialization_func='linear')
#sobj_makeK = som.SimpleSOM((20, 30))
sobj_makeK


# In[ ]:


sobj_makeK._initialize(X_sc)
sobj_makeK.K.shape


# In[ ]:


img = som.conv2img(sobj_makeK.K, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


df = pd.DataFrame(sobj_makeK.K)
sns.pairplot(df, markers='.')


# In[ ]:


df1= pd.DataFrame(sobj_makeK.K)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X_sc)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
df.head()
df.shape
sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, diag_kind='hist')


# ### sksom

# In[ ]:


'''
Argument init_K must be needed.
We can get init_K (initial landmarks) by the way above
'''
#sobj = som.sksom((20, 30), init_K=sobj_makeK.K, it=50)
sobj = som.sksom((20, 30), init_K=sobj_makeK.K, it=50, verbose=1, early_stopping=False)
sobj


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.fit(X_sc)


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.predict(X_sc)


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


df = pd.DataFrame(sobj.landmarks_)
sns.pairplot(df, markers='.')


# In[ ]:


df = pd.DataFrame(X_sc)
sns.pairplot(df, markers='.')


# In[ ]:


df1= pd.DataFrame(sobj.landmarks_)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X_sc)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
df.head()
df.shape
sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, diag_kind='hist')


# In[ ]:


sobj.labels_


# In[ ]:


sobj.predict(sobj.landmarks_)


# In[ ]:


sobj.predict(X_sc)


# In[ ]:


'''
evaluate by gaussian kernel with gamma
as probability
'''
sobj.predict_proba(X_sc)[:10,:5]


# ### further train r: 1.5 -> 1.0

# In[ ]:


sobj.verbose = 2
sobj.r = (1.5, 1.0)
sobj.it = 1500
sobj.early_stopping = (5, 1.0e-5)
sobj.fit(X_sc)


# In[ ]:


lw = 2
plt.plot(np.arange(len(sobj.meanDist)), sobj.meanDist, label="mean distance to closest landmark",
             color="darkorange", lw=lw)
plt.legend(loc="best")


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


'''show sepal length'''
img = som.conv2img(sobj.landmarks_, (20, 30), target=[0,1,2,3])
#img.shape
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,0])

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


'''show sepal width'''
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,1])

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


'''show petal length'''
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,2])

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


'''show petal width'''
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,3])

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


df = pd.DataFrame(sobj.landmarks_)
sns.pairplot(df, markers='.')


# In[ ]:


df1= pd.DataFrame(sobj.landmarks_)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X_sc)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
df.head()
df.shape
sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, diag_kind='hist')


# ### iris (random init)

# In[ ]:


sobj_makeK = som.SimpleSOM((20, 30))
sobj_makeK
sobj_makeK._initialize(X_sc)
sobj_makeK.K.shape


# In[ ]:


sobj = som.sksom((20, 30), init_K=sobj_makeK.K, it=300, verbose=1, early_stopping=False)
sobj


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.fit(X_sc)


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:





# ### random init and fixed small r

# In[ ]:


sobj_makeK = som.SimpleSOM((20, 30))
sobj_makeK
sobj_makeK._initialize(X_sc)
sobj_makeK.K.shape


# In[ ]:


sobj = som.sksom((20, 30), init_K=sobj_makeK.K, r=1.5, it=300, verbose=1, early_stopping=False)
sobj


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.fit(X_sc)


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


df = pd.DataFrame(sobj.landmarks_)
sns.pairplot(df, markers='.')


# In[ ]:


df1= pd.DataFrame(sobj.landmarks_)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X_sc)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
df.head()
df.shape
sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, diag_kind='hist')


# In[ ]:


'''
further train
set r to (1.5, 1.0)
'''
sobj.r = (1.5, 1.0)
sobj.fit(X_sc)


# In[ ]:


lw = 2
plt.plot(np.arange(len(sobj.meanDist)), sobj.meanDist, label="mean distance to closest landmark",
         color="darkorange", lw=lw)
plt.legend(loc="best")


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


'''show sepal length'''
img = som.conv2img(sobj.landmarks_, (20, 30))
#img.shape
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,0])

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


'''show sepal width'''
img = som.conv2img(sobj.landmarks_, (20, 30))
#img.shape
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,1])

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


'''show petal length'''
img = som.conv2img(sobj.landmarks_, (20, 30))
#img.shape
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,2])

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


'''show petal width'''
img = som.conv2img(sobj.landmarks_, (20, 30), target=[0,1,3])
#img.shape
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,2])

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


df = pd.DataFrame(sobj.landmarks_)
sns.pairplot(df, markers='.')


# In[ ]:


df1= pd.DataFrame(sobj.landmarks_)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X_sc)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
df.head()
df.shape
sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, diag_kind='hist')


# ### sphere

# In[ ]:


k_shape = (20, 30)

'''get initial landmarks'''
#sobj_makeK = som.SphereSOM(k_shape, initialization_func='linear')
sobj_makeK = som.SphereSOM(k_shape, initialization_func=None)
sobj_makeK
sobj_makeK._initialize(X_sc)
sobj_makeK.K.shape


# In[ ]:


sobj = som.sksom(k_shape, init_K=sobj_makeK.K, it=300, verbose=2, early_stopping=False, alpha=1, form='sphere')
sobj


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape)
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.fit(X_sc)


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape)
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape)
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


df = pd.DataFrame(sobj.landmarks_)
sns.pairplot(df, markers='.')


# In[ ]:


df1= pd.DataFrame(sobj.landmarks_)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X_sc)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
df.head()
df.shape
sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, diag_kind='hist')


# In[ ]:


'''further train'''
sobj.r = (1.0, 0.8)
sobj.it = 1500
sobj.early_stopping = (5, 1.0e-7)
#sobj.early_stopping = False
sobj.fit(X_sc)


# In[ ]:


lw = 2
plt.plot(np.arange(len(sobj.meanDist)), sobj.meanDist, label="mean distance to closest landmark",
             color="darkorange", lw=lw)
plt.legend(loc="best")


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape)
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape)
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


'''show sepal length'''
img = som.conv2img(sobj.landmarks_, k_shape, target=[0,1,2])
#img.shape
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,0])

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


'''show sepal width'''
img = som.conv2img(sobj.landmarks_, k_shape, target=[0,1,2])
#img.shape
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,1])

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


'''show petal length'''
img = som.conv2img(sobj.landmarks_, k_shape, target=[0,1,2])
#img.shape
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,2])

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


'''show petal width'''
img = som.conv2img(sobj.landmarks_, k_shape, target=[0,1,3])
#img.shape
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,2])

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    if iris.target_names[iris.target[i]] == 'versicolor':
        plt.text(a, b, 'versicolor', ha='center', va='center',
               bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))
    elif iris.target_names[iris.target[i]] == 'virginica':
        plt.text(a, b, 'virginica', ha='center', va='center',
               bbox=dict(facecolor='pink', alpha=0.5, lw=0))
    else:
        plt.text(a, b, 'setosa', ha='center', va='center',
               bbox=dict(facecolor='white', alpha=0.5, lw=0))


# In[ ]:


df = pd.DataFrame(sobj.landmarks_)
sns.pairplot(df, markers='.')


# In[ ]:


df1= pd.DataFrame(sobj.landmarks_)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X_sc)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
df.head()
df.shape
sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, diag_kind='hist')


# In[ ]:





# # Moon Data

# In[ ]:


n_samples = 1500

X, y = datasets.make_moons(n_samples=n_samples, noise=.15, random_state=0)
df = pd.DataFrame(X)
df.columns = ["col1", "col2"]
df['cls'] = y

sns.lmplot("col1", "col2", hue="cls", data=df, fit_reg=False, size=8)


# ### prepare linear initialization

# In[ ]:


sobj_init = som.SimpleSOM((20, 30), initialization_func='linear')
sobj_init


# In[ ]:


sobj_init._initialize(X)
sobj_init.K
img = som.conv2img(sobj_init.K, (20, 30), target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,0])


# In[ ]:


df = pd.DataFrame(sobj_init.K)
df.columns = ['col1', 'col2']
sns.lmplot("col1", "col2", data=df, fit_reg=False, size=8)


# In[ ]:


df1= pd.DataFrame(sobj_init.K)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
df.head()
df.shape
df.columns = ['col1', 'col2', 'cls']
sns.lmplot("col1", "col2", hue='cls', data=df, fit_reg=False, size=8, scatter_kws={'alpha': 0.5})


# ## train from linear init and small r
# r = 1.5  
# and early_stopping

# In[ ]:


sobj = som.sksom((20, 30), init_K=sobj_init.K.copy(), r=1.5, it=1500, verbose=2)
#sobj = som.sksom((20, 30), init_K=sobj_init.K.copy(), r=1.5, it=1500, verbose=0, early_stopping=None)
sobj


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30), target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,0])


# In[ ]:


sobj.fit(X)


# In[ ]:


lw = 2
plt.plot(np.arange(len(sobj.meanDist)), sobj.meanDist, label="mean distance to closest landmark",
             color="darkorange", lw=lw)
plt.legend(loc="best")


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30), target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,0])


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30), target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,1])


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30), target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,(0,1,1)])

for i, m in enumerate(sobj.predict(X)):
    b, a = divmod(m, sobj.kshape[1])
    plt.text(a, b, str(y[i]), ha='center', va='center',
           bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))


# In[ ]:


df = pd.DataFrame(sobj.landmarks_)
df.columns = ['col1', 'col2']
sns.lmplot("col1", "col2", data=df, fit_reg=False, height=8)


# In[ ]:


df1= pd.DataFrame(sobj.landmarks_)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
print(df.shape)
df.columns = ['col1', 'col2', 'cls']
df.head()
#sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, size=5)
sns.lmplot("col1", "col2", hue="cls", data=df, fit_reg=False, height=8, scatter_kws={'alpha': 0.5})


# ## train from linear init

# In[ ]:


sobj = som.sksom((20, 30), init_K=sobj_init.K.copy(), it=150, verbose=2, early_stopping=False)
#sobj = som.sksom((20, 30), init_K=sobj_init.K.copy(), r=1.5, it=1500, verbose=0, early_stopping=None)
sobj


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30), target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,0])


# In[ ]:


sobj.fit(X)


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30), target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,0])


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30), target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,1])


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30), target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,(0,1,1)])

for i, m in enumerate(sobj.predict(X)):
    b, a = divmod(m, sobj.kshape[1])
    plt.text(a, b, str(y[i]), ha='center', va='center',
           bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))


# In[ ]:


df = pd.DataFrame(sobj.landmarks_)
df.columns = ['col1', 'col2']
sns.lmplot("col1", "col2", data=df, fit_reg=False, height=8)


# In[ ]:


df1= pd.DataFrame(sobj.landmarks_)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
print(df.shape)
df.columns = ['col1', 'col2', 'cls']
df.head()
#sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, size=5)
sns.lmplot("col1", "col2", hue="cls", data=df, fit_reg=False, height=8, scatter_kws={'alpha': 0.5})


# ### further train r=1.0 fixed gamma

# In[ ]:


sobj.verbose = 2
sobj.r = 1.0
sobj.it = 1500
sobj.early_stopping = (5, 1.0e-6)

sobj.fit(X)


# In[ ]:


lw = 2
plt.plot(np.arange(len(sobj.meanDist)), sobj.meanDist, label="mean distance to closest landmark",
             color="darkorange", lw=lw)
plt.legend(loc="best")


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30), target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,0])


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30), target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,1])


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30), target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,(0,1,1)])

for i, m in enumerate(sobj.predict(X)):
    b, a = divmod(m, sobj.kshape[1])
    plt.text(a, b, str(y[i]), ha='center', va='center',
           bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))


# In[ ]:


df = pd.DataFrame(sobj.landmarks_)
df.columns = ['col1', 'col2']
sns.lmplot("col1", "col2", data=df, fit_reg=False, size=8)


# In[ ]:


df1= pd.DataFrame(sobj.landmarks_)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
print(df.shape)
df.columns = ['col1', 'col2', 'cls']
df.head()
#sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, size=5)
sns.lmplot("col1", "col2", hue="cls", data=df, fit_reg=False, size=8, scatter_kws={'alpha': 0.5})


# ## sphere

# In[ ]:


k_shape = (50, 30)
sobj_init = som.SimpleSOM(k_shape, initialization_func='linear')
sobj_init


# In[ ]:


sobj_init._initialize(X)
sobj_init.K

df = pd.DataFrame(sobj_init.K)
df.columns = ['col1', 'col2']
sns.lmplot("col1", "col2", data=df, fit_reg=False, height=5)


# In[ ]:


df1= pd.DataFrame(sobj_init.K)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
df.head()
df.shape
df.columns = ['col1', 'col2', 'cls']
sns.lmplot("col1", "col2", hue='cls', data=df, fit_reg=False, size=5, scatter_kws={'alpha': 0.5})


# In[ ]:


sobj = som.sksom(k_shape, init_K=sobj_init.K, it=150, verbose=2, early_stopping=False, alpha=1, form='sphere')
sobj


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape, target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,0])


# In[ ]:


sobj.fit(X)


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape, target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,0])


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape, target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,1])


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape, target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,(0,1,0)])

for i, m in enumerate(sobj.predict(X)):
    b, a = divmod(m, sobj.kshape[1])
    plt.text(a, b, str(y[i]), ha='center', va='center',
           bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))


# In[ ]:


df = pd.DataFrame(sobj.landmarks_)
df.columns = ['col1', 'col2']
sns.lmplot("col1", "col2", data=df, fit_reg=False, height=8)


# In[ ]:


df1= pd.DataFrame(sobj.landmarks_)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
print(df.shape)
df.columns = ['col1', 'col2', 'cls']
df.head()
#sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, size=5)
sns.lmplot("col1", "col2", hue="cls", data=df, fit_reg=False, height=8, scatter_kws={'alpha': 0.5})


# ### further train

# In[ ]:


'''
further train
'''
sobj.r = (1.0, 0.5)
sobj.it = 1500
sobj.early_stopping = (5, 1.0e-7)
#sobj.early_stopping = False
sobj.fit(X)


# In[ ]:


lw = 2
plt.plot(np.arange(len(sobj.meanDist)), sobj.meanDist, label="mean distance to closest landmark",
             color="darkorange", lw=lw)
plt.legend(loc="best")


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape, target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,0])


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape, target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,1])


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape, target=(0,1))
plt.figure(figsize=(10, 10))
plt.imshow(img[:,:,(0,1,0)])

for i, m in enumerate(sobj.predict(X)):
    b, a = divmod(m, sobj.kshape[1])
    plt.text(a, b, str(y[i]), ha='center', va='center',
           bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))


# In[ ]:


df = pd.DataFrame(sobj.landmarks_)
df.columns = ['col1', 'col2']
sns.lmplot("col1", "col2", data=df, fit_reg=False, height=8)


# In[ ]:


df1= pd.DataFrame(sobj.landmarks_)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
print(df.shape)
df.columns = ['col1', 'col2', 'cls']
df.head()
#sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, size=5)
sns.lmplot("col1", "col2", hue="cls", data=df, fit_reg=False, height=8, scatter_kws={'alpha': 0.5})


# In[ ]:





# # Digit 8x8

# In[ ]:


digits = datasets.load_digits()
X, y = digits.data, digits.target
X[:5]


# In[ ]:


X = X.reshape((X.shape[0], -1))


# In[ ]:


X_sc = X / 16.0
X_sc.shape


# ### prepare linear initialization

# In[ ]:


sobj_init = som.SimpleSOM((20, 30), initialization_func='linear')
sobj_init


# In[ ]:


sobj_init._initialize(X_sc)
sobj_init.K
img = som.conv2img(sobj_init.K, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


df = pd.DataFrame(sobj_init.K[:,:5])
df.columns = ['col1', 'col2', 'col3', 'col4', 'col5']
sns.pairplot(df, markers=['.'])


# In[ ]:


df1= pd.DataFrame(sobj_init.K[:,:5])
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X_sc[:,:5])
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
df.columns = ['col1', 'col2', 'col3', 'col4', 'col5', 'cls']
df.head()
print(df.shape)
sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, diag_kind='hist')


# ## train from linear init and small r
# and early_stopping

# In[ ]:


sobj = som.sksom((20, 30), init_K=sobj_init.K.copy(), r=1.5, it=150, verbose=2, alpha=10)
#sobj = som.sksom((20, 30), init_K=sobj_init.K.copy(), r=1.5, it=1500, verbose=0, early_stopping=None)
sobj


# In[ ]:


sobj.fit(X_sc)


# In[ ]:


lw = 2
plt.plot(np.arange(len(sobj.meanDist)), sobj.meanDist, label="mean distance to closest landmark",
             color="darkorange", lw=lw)
plt.legend(loc="best")


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30), target=(3,4,5))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30), target=(3,4,5))
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    plt.text(a, b, str(y[i]), ha='center', va='center',
           bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))


# In[ ]:


df = pd.DataFrame(sobj.landmarks_[:,:5])
df.columns = ['col1', 'col2', 'col3', 'col4', 'col5']
sns.pairplot(df, markers=['.'])


# In[ ]:


df1= pd.DataFrame(sobj.landmarks_[:,:5])
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(X_sc[:,:5])
df2['cls'] = 'X'
df2.head()
df = pd.concat([df1, df2], axis=0)
print(df.shape)
df.columns = ['col1', 'col2', 'col3', 'col4', 'col5', 'cls']
df.head()
sns.pairplot(df, markers=['.', 's'], hue='cls', plot_kws={'alpha': 0.5}, diag_kind='hist')


# ## sphere (50x30)

# In[ ]:


k_shape = (50, 30)

sobj_init = som.SimpleSOM(k_shape, initialization_func='linear')
sobj_init

sobj_init._initialize(X_sc)
sobj_init.K
img = som.conv2img(sobj_init.K, k_shape)
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj = som.sksom(k_shape, init_K=sobj_init.K.copy(), it=30, verbose=2, early_stopping=False, alpha=1, form='sphere')
#sobj = som.sksom((20, 30), init_K=sobj_init.K.copy(), r=1.5, it=1500, verbose=0, early_stopping=None)
sobj


# In[ ]:


sobj.fit(X_sc)


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape)
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape, target=(3,4,5))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape, target=(3,4,5))
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    plt.text(a, b, str(y[i]), ha='center', va='center',
           bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))


# ### further train r=1.0 -> 0.5

# In[ ]:


'''
r: 1.0 -> 0.5
'''
sobj.verbose = 2
sobj.r = (1.0, 0.5)
sobj.it = 1500
sobj.early_stopping = (5, 1.0e-5)

sobj.fit(X_sc)


# In[ ]:


lw = 2
plt.plot(np.arange(len(sobj.meanDist)), sobj.meanDist, label="mean distance to closest landmark",
             color="darkorange", lw=lw)
plt.legend(loc="best")


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape)
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape, target=(3,4,5))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape, target=(3,4,5))
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    plt.text(a, b, str(y[i]), ha='center', va='center',
           bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))


# In[ ]:





# In[ ]:


'''
r: 0.5
'''
sobj.verbose = 2
sobj.r = 0.5
sobj.it = 1500
sobj.early_stopping = (5, 1.0e-5)

sobj.fit(X_sc)


# In[ ]:


lw = 2
plt.plot(np.arange(len(sobj.meanDist)), sobj.meanDist, label="mean distance to closest landmark",
             color="darkorange", lw=lw)
plt.legend(loc="best")


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape)
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape, target=(3,4,5))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, k_shape, target=(3,4,5))
plt.figure(figsize=(10, 10))
plt.imshow(img)

for i, m in enumerate(sobj.predict(X_sc)):
    b, a = divmod(m, sobj.kshape[1])
    plt.text(a, b, str(y[i]), ha='center', va='center',
           bbox=dict(facecolor='lightblue', alpha=0.5, lw=0))


# In[ ]:





# # Don't Overfit! II data

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


#src_dir = '../input/dont-overfit-ii'
src_dir = '../input'
train_csv = pd.read_csv(os.path.join(src_dir, 'train.csv'))
print(train_csv.shape)
train_csv.head()


# In[ ]:


x_train0 = train_csv.iloc[:,2:].values
print(x_train0.shape)
x_train0


# In[ ]:


y_train0 = train_csv.target.values
print(y_train0.shape)
y_train0


# In[ ]:


from keras.utils import to_categorical, plot_model

y_cat_train0 = to_categorical(y_train0)
print(y_cat_train0.shape)
y_cat_train0[:5]


# In[ ]:


test_csv = pd.read_csv(os.path.join(src_dir, 'test.csv'))
print(test_csv.shape)
test_csv.head()


# In[ ]:


x_test = test_csv.iloc[:,1:].values
print(x_test.shape)
x_test


# In[ ]:


sample_submission_csv = pd.read_csv(os.path.join(src_dir, 'sample_submission.csv'))
print(sample_submission_csv.shape)
sample_submission_csv.head()


# In[ ]:


X = np.vstack([x_train0, x_test])
X.shape


# In[ ]:


df = pd.DataFrame(X[:,:5])
df.columns = ["col1", "col2", "col3", "col4", "col5"]
#df['cls'] = ['c'+str(ee) for ee in y]
df.head()

fig = sns.pairplot(df, markers='.', height=2.2, diag_kind='hist')


# ### get initial landmarks

# In[ ]:


sinit_obj = som.SimpleSOM((20, 30))
sinit_obj


# In[ ]:


sinit_obj._initialize(X)
sinit_obj.K


# In[ ]:


img = som.conv2img(sinit_obj.K, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# ### create instance

# In[ ]:


'''
when "r" is set, r is NOT updated during fit
'''
sobj = som.sksom((20, 30), init_K=sinit_obj.K.copy(), r=1.5, it=5, alpha=20, verbose=1, early_stopping=False)
#sobj = som.sksom((20, 30), init_K=sinit_obj.K.copy(), r=1.5, it=5, verbose=1)
sobj


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.fit(X)


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.fit(X)

img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.fit(X)

img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.fit(X)

img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.fit(X)

img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.verbose = 2

sobj.fit(X)

img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.fit(X)

img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


sobj.fit(X)

img = som.conv2img(sobj.landmarks_, (20, 30))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.landmarks_, (20, 30), target=(297, 298, 299))
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:




