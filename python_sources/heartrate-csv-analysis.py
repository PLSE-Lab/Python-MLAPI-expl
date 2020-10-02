#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


import os


# In[ ]:


from sklearn.random_projection import SparseRandomProjection as sr  # Projection features
from sklearn.cluster import KMeans                    # Cluster features
from sklearn.preprocessing import PolynomialFeatures  # Interaction features


# In[ ]:


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif  # Selection criteria


# In[ ]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


from sklearn.tree import  DecisionTreeClassifier as dt
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation


# In[ ]:


from sklearn.ensemble import RandomForestClassifier as rf


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_graphviz #plot tree


# In[ ]:


import os, time, gc


# In[ ]:


data = pd.read_csv("../input/HeartRate.csv") #Loading of Data


# In[ ]:


data.head(2)


# In[ ]:


data.shape


# In[ ]:


data.dtypes.value_counts() 


# In[ ]:


data.isnull().sum().sum()  # 0


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data.drop('target', 1), data['target'], test_size = 0.3, random_state=10)


# In[ ]:


X_train.shape  


# In[ ]:


X_test.shape 


# In[ ]:


y_train.shape 


# In[ ]:


y_test.shape


# In[ ]:


X_train.isnull().sum().sum()  


# In[ ]:


X_test.isnull().sum().sum() 


# In[ ]:


X_train['sum'] = X_train.sum(numeric_only = True, axis=1) 
X_test['sum'] = X_test.sum(numeric_only = True,axis=1)
tmp_train = X_train.replace(0, np.nan)
tmp_test = X_test.replace(0,np.nan)


# In[ ]:


tmp_train is X_train


# In[ ]:


tmp_train._is_view  


# In[ ]:


tmp_train.head(2)


# In[ ]:


tmp_train.notna().head(1)


# In[ ]:


X_train["count_not0"] = tmp_train.notna().sum(axis = 1)
X_test['count_not0'] = tmp_test.notna().sum(axis = 1)


# In[ ]:


X_train.shape


# In[ ]:


feat = [ "var", "median", "mean", "std", "max", "min"]
for i in feat:
    X_train[i] = tmp_train.aggregate(i,  axis =1)
    X_test[i]  = tmp_test.aggregate(i,axis = 1)


# In[ ]:


del(tmp_train)
del(tmp_test)


# In[ ]:


gc.collect()


# In[ ]:


X_train.shape  


# In[ ]:


X_train.head(1)


# In[ ]:


colNames = X_train.columns.values


# In[ ]:


colNames


# In[ ]:


tmp = pd.concat([X_train,X_test],  axis = 0, ignore_index = True)


# In[ ]:


tmp.shape


# In[ ]:


tmp = tmp.values


# In[ ]:


tmp.shape 


# In[ ]:


NUM_OF_COM = 5
rp_instance = sr(n_components = NUM_OF_COM)
rp = rp_instance.fit_transform(tmp[:, :13])
rp[: 2, :  3]


# In[ ]:


rp_col_names = ["r" + str(i) for i in range(5)]


# In[ ]:


rp_col_names


# In[ ]:


centers = y_train.nunique()  


# In[ ]:


centers   


# In[ ]:


kmeans = KMeans(n_clusters=centers, n_jobs = 2)   
kmeans.fit(tmp[:, : 13])
kmeans.labels_


# In[ ]:


kmeans.labels_.size 


# In[ ]:


ohe = OneHotEncoder(sparse = False)
ohe.fit(kmeans.labels_.reshape(-1,1))     # reshape(-1,1) recommended by fit()


# In[ ]:


dummy_clusterlabels = ohe.transform(kmeans.labels_.reshape(-1,1))
dummy_clusterlabels


# In[ ]:


dummy_clusterlabels.shape    #(303,2)


# In[ ]:


k_means_names = ["k" + str(i) for i in range(2)]
k_means_names


# In[ ]:


degree = 2
poly = PolynomialFeatures(degree, interaction_only=True, include_bias = False)
df =  poly.fit_transform(tmp[:, : 5])
df.shape     # 303 X 15


# In[ ]:


poly_names = [ "poly" + str(i)  for i in range(15)]
poly_names


# In[ ]:


if ('dummy_clusterlabels' in vars()):               #
    tmp = np.hstack([tmp,rp,dummy_clusterlabels, df])
else:
    tmp = np.hstack([tmp,rp, df]) 
    


# In[ ]:


tmp.shape  


# In[ ]:


X = tmp[: X_train.shape[0], : ]
X.shape


# In[ ]:


test = tmp[X_train.shape[0] :, : ]
test.shape


# In[ ]:


del tmp
gc.collect()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X,y_train,test_size = 0.3)


# In[ ]:


X_train.shape


# In[ ]:


clf = rf(n_estimators=50)
clf = clf.fit(X_train, y_train)
classes = clf.predict(X_test)
(classes == y_test).sum()/y_test.size 


# In[ ]:


clf.feature_importances_ 


# In[ ]:


clf.feature_importances_.size


# In[ ]:


if ('dummy_clusterlabels' in vars()):       # If dummy_clusterlabels labels are defined
    colNames = list(colNames) + rp_col_names+ k_means_names + poly_names
else:
    colNames = colNames = list(colNames) + rp_col_names +  poly_names      # No kmeans      <==


# In[ ]:


len(colNames)


# In[ ]:


feat_imp = pd.DataFrame({ "importance": clf.feature_importances_ , "featureNames" : colNames } ).sort_values(by = "importance", ascending=False)


# In[ ]:


colNames


# In[ ]:


g = sns.barplot(x = feat_imp.iloc[  : 20 ,  1] , y = feat_imp.iloc[ : 20, 0])


# In[ ]:




