#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install imblearn

To enable plotting graphs in Jupyter notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


import pandas as pd
from sklearn.linear_model import LogisticRegression


import matplotlib.pyplot as plt   


import seaborn as sns



from sklearn.model_selection import train_test_split

import numpy as np


from sklearn import metrics


from sklearn.metrics import recall_score

from imblearn.over_sampling import SMOTE


# In[ ]:




colnames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']


pima_df = pd.read_csv("D:\ML_Data\pima-indians-diabetes.data", names= colnames)


# In[ ]:


pima_df.head(50)


# In[ ]:



pima_df[~pima_df.applymap(np.isreal).all(1)]


# In[ ]:





# In[ ]:



pima_df.describe().transpose()


# In[ ]:



pima_df.groupby(["class"]).count()


# In[ ]:





# In[ ]:



sns.pairplot(pima_df , hue='class' , diag_kind = 'kde')


# In[ ]:





# In[ ]:





# In[ ]:


array = pima_df.values
X = array[:,0:7] 
Y = array[:,8]   
test_size = 0.30
seed = 7  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
type(X_train)


# In[ ]:


print("Before UpSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before UpSampling, counts of label '0': {} \n".format(sum(y_train==0)))

sm = SMOTE(sampling_strategy = 1 ,k_neighbors = 5, random_state=1)   
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())


print("After UpSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After UpSampling, counts of label '0': {} \n".format(sum(y_train_res==0)))



print('After UpSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After UpSampling, the shape of train_y: {} \n'.format(y_train_res.shape))







# In[ ]:


# Fit the model on original data i.e. before upsampling
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print(model_score)
print(metrics.confusion_matrix(y_test, y_predict))


# In[ ]:


# fit model on upsampled data 

model.fit(X_train_res, y_train_res)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print(model_score)
print(metrics.confusion_matrix(y_test, y_predict))


# In[ ]:





# In[ ]:


non_diab_indices = pima_df[pima_df['class'] == 0].index  
no_diab = len(pima_df[pima_df['class'] == 0])            
print(no_diab)

diab_indices = pima_df[pima_df['class'] == 1].index       
diab = len(pima_df[pima_df['class'] == 1])                
print(diab)


# In[ ]:


random_indices = np.random.choice( non_diab_indices, no_diab - 200 , replace=False)    


# In[ ]:


down_sample_indices = np.concatenate([diab_indices,random_indices])  


# In[ ]:


pima_df_down_sample = pima_df.loc[down_sample_indices]  
pima_df_down_sample.shape
pima_df_down_sample.groupby(["class"]).count() 


# In[ ]:


array = pima_df_down_sample.values
X = array[:,0:7]
Y = array[:,8]   
test_size = 0.30 
seed = 7  
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
type(X_train)


# In[ ]:


print('After DownSampling, the shape of X_train: {}'.format(X_train.shape))
print('After DownSampling, the shape of X_test: {} \n'.format(X_test.shape))


# In[ ]:


# Fit the model on 30%
model = LogisticRegression()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print(model_score)
print(metrics.confusion_matrix(y_test, y_predict))


# In[ ]:


array = pima_df.values
X = array[:,0:7] # select all rows and first 8 columns which are the attributes
Y = array[:,8]   # select all rows and the 8th column which is the classification "Yes", "No" for diabeties
test_size = 0.30 # taking 70:30 training and test set
seed = 7  # Random numbmer seeding for reapeatability of the code
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
type(X_train)


# ## IMBLearn Random Under Sampling

# In[ ]:


from imblearn.under_sampling import RandomUnderSampler


# In[ ]:


rus = RandomUnderSampler(return_indices=True)


# In[ ]:


X_rus, y_rus, id_rus = rus.fit_sample(X_train, y_train)


# In[ ]:


y_rus


# In[ ]:


y_rus.shape


# ## IMBLearn Random Over Sampling

# In[ ]:


from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()

X_ros, y_ros = ros.fit_sample(X_train, y_train)


# In[ ]:


y_ros


# In[ ]:


y_ros.shape


# In[ ]:


X_ros.shape


# ##  Deleting nearest majority neighbors  TomekLinks
# 
# 

# In[ ]:


from imblearn.under_sampling import TomekLinks


# In[ ]:


tl = TomekLinks(return_indices=True, ratio='majority')


# In[ ]:


X_tl, y_tl, id_tl = tl.fit_sample(X_train, y_train)   # id_tl is removed instances of majority class


# In[ ]:


y_tl.shape


# In[ ]:


X_tl.shape


# ## Upsampling followed by downsampling

# In[ ]:


from imblearn.combine import SMOTETomek


# In[ ]:


smt = SMOTETomek(ratio='auto')


# In[ ]:


X_smt, y_smt = smt.fit_sample(X_train, y_train)


# In[ ]:


X_smt.shape


# ## Cluster based undersampling

# In[ ]:


from imblearn.under_sampling import ClusterCentroids


# In[ ]:


cc = ClusterCentroids()  
X_cc, y_cc = cc.fit_sample(X_train, y_train)


# In[ ]:


X_cc.shape


# In[ ]:


y_cc

