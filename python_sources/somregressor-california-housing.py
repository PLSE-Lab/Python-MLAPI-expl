#!/usr/bin/env python
# coding: utf-8

# [SomClassifier (Iris Moon Digit8)](https://www.kaggle.com/wordroid/somclassifier-iris-moon-digit8)

# In[ ]:


get_ipython().system('pip install git+https://github.com/darecophoenixx/wordroid.sblo.jp')


# In[ ]:


from som import som


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[ ]:


import random

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_digits
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

import matplotlib.pyplot as plt
import seaborn as sns


# ## california_housing

# In[ ]:


from sklearn.datasets.california_housing import fetch_california_housing


# In[ ]:


cal_housing = fetch_california_housing()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(cal_housing.data,
                                                    cal_housing.target,
                                                    test_size=0.2,
                                                    random_state=1)


# In[ ]:


names = cal_housing.feature_names


# In[ ]:


X_train[:3]


# In[ ]:


sns.distplot(X_train[:,0])


# In[ ]:


sns.distplot(np.log(X_train[:,0]))


# In[ ]:


sns.distplot(X_train[:,1])


# In[ ]:


sns.distplot(X_train[:,2])


# In[ ]:


sns.distplot(np.log(X_train[:,2]))


# In[ ]:


sns.distplot(X_train[:,3])


# In[ ]:


sns.distplot(np.log(X_train[:,3]))


# In[ ]:


sns.distplot(X_train[:,4])


# In[ ]:


sns.distplot(np.log(X_train[:,4]))


# In[ ]:


sns.distplot(X_train[:,5])


# In[ ]:


sns.distplot(np.log(X_train[:,5]))


# In[ ]:


sns.distplot(X_train[:,6])


# In[ ]:


sns.distplot(X_train[:,7])


# In[ ]:


sns.distplot(y_train)


# In[ ]:


sns.distplot(np.log(y_train))


# In[ ]:


y_train


# ## scale

# In[ ]:


import sklearn
sklearn.__version__


# In[ ]:


pt = preprocessing.PowerTransformer()


# In[ ]:


pt.fit(np.c_[cal_housing.target, cal_housing.data])


# In[ ]:


Xy_sc = pt.transform(np.c_[cal_housing.target, cal_housing.data])
Xy_sc.shape


# In[ ]:


pt.inverse_transform(Xy_sc)[:3]


# In[ ]:


df = pd.DataFrame(Xy_sc)
sns.pairplot(df)


# In[ ]:


y_sc = Xy_sc[:,0]
y_sc.shape


# In[ ]:


X_sc = Xy_sc[:,1:]
X_sc.shape


# ### train

# In[ ]:


sobj = som.SomRegressor((20, 30), it=(15,1500), r2=(1.5,0.5), verbose=2, alpha=1.5)
sobj


# In[ ]:


sobj.fit(X_sc, y_sc)


# In[ ]:


lw = 2
plt.plot(np.arange(len(sobj.sksom.meanDist)), sobj.sksom.meanDist, label="mean distance to closest landmark",
             color="darkorange", lw=lw)
plt.legend(loc="best")


# In[ ]:


sobj.predict(X_sc)


# ### y vs y_predicted

# In[ ]:


np.c_[y_sc, sobj.predict(X_sc)]


# In[ ]:


df = pd.DataFrame(np.c_[y_sc, sobj.predict(X_sc)])
df.columns = ['col1', 'col2']
df.head()
sns.lmplot("col1", "col2", data=df, fit_reg=False, size=8, scatter_kws={'alpha': 0.5})


# In[ ]:


df = pd.DataFrame(np.c_[cal_housing.target, pt.inverse_transform(np.c_[sobj.predict(X_sc), X_sc])[:,0]])
df.columns = ['col1', 'col2']
df.head()
sns.lmplot("col1", "col2", data=df, fit_reg=False, size=8, scatter_kws={'alpha': 0.5})


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=[0,1,2])
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=[3,4,5])
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


img = som.conv2img(sobj.sksom.landmarks_, (20, 30), target=[6,7,8])
plt.figure(figsize=(10, 10))
plt.imshow(img)


# In[ ]:


df1= pd.DataFrame(sobj.sksom.landmarks_)
df1['cls'] = 'K'
df1.head()
df2 = pd.DataFrame(Xy_sc)
df2['cls'] = 'X'
df2.head()
df = pd.concat([df2, df1], axis=0)
df.head()
df.shape
sns.pairplot(df, markers=['s', '.'], hue='cls', plot_kws={'alpha': 0.3}, diag_kind='hist')


# In[ ]:





# In[ ]:





# In[ ]:




