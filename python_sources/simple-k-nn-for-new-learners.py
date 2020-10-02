#!/usr/bin/env python
# coding: utf-8

# > # Simple K-NN

# ## The pakages I use

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.preprocessing import Normalizer
sb.set_style("dark")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import time

get_ipython().run_line_magic('pylab', 'inline')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df=pd.read_csv("../input/train.csv")\ntest_df=pd.read_csv("../input/test.csv")')


# In[ ]:


train_df.isnull().sum().describe()


# In[ ]:


train_df.describe()


# In[ ]:


train_df.shape


# In[ ]:


target=train_df['label']
train_df=train_df.drop('label',axis=1)


# In[ ]:


figure(figsize(5,5))
for digit_num in range(0,64):
    subplot(8,8,digit_num+1)
    grid_data = train_df.iloc[digit_num].as_matrix().reshape(28,28)  # reshape from 1d to 2d pixel array
    plt.imshow(grid_data, interpolation = "none", cmap = "bone_r")
    xticks([])
    yticks([])


# In[ ]:


target.hist()


# ## Scaling the data
#  
# As the data have very different range of value, we need to scal the data to make it easy to train.

# In[ ]:


norm = Normalizer().fit(train_df)
train_df = norm.transform(train_df)
test_df = norm.transform(test_df)


# In[ ]:


train_df = pd.DataFrame(train_df)
test_df= pd.DataFrame(test_df)


# In[ ]:


pca = PCA(n_components=784, random_state=0, svd_solver='randomized')
pca.fit(train_df)


# In[ ]:


plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.ylim(0.9, 1.0)
plt.grid()


# From the PCA , we can compress the data set as only 100 features can represent 92% of the data set.

# In[ ]:


def pca(X_tr, X_ts, test,n):
    pca = PCA(n)
    pca.fit(X_tr)
    X_tr_pca = pca.transform(X_tr)
    X_ts_pca = pca.transform(X_ts)
    test_pca = pca.transform(test)
    return X_tr_pca, X_ts_pca, test_pca


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_df, target,
    test_size=0.1, random_state=2)


# In[ ]:


test_num=y_test[(y_test==9)].index
num=len(test_num)


# In[ ]:


X_num9_test=X_test.loc[test_num]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train_pca, X_test_pca,test_num9_pca = pca(X_train, X_test, X_num9_test,100)')


# In[ ]:


get_ipython().run_cell_magic('time', '', "model = KNeighborsClassifier(n_neighbors = 4, weights='distance')\nmodel.fit(X_train_pca, y_train)\nscore = model.score(X_test_pca, y_test)\nprint ('KNN ', score)\n#pred_submit = model.predict(test_df_pca)\npred_homework=model.predict(X_test_pca)")


# The confusion matrix is :

# In[ ]:


confusion_matrix(y_test,pred_homework) 


# ## The number 9's accuracy change with the i

# In[ ]:


y_9=[9]*num


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in range(1,20):\n    model=KNeighborsClassifier(n_neighbors = i, weights='distance')\n    model.fit(X_train_pca, y_train)\n    score = model.score(test_num9_pca , y_9)\n    print ('The accuracy of number 9''s {}NN score is :{} '.format(i,score))")


# In[ ]:





# In[ ]:




