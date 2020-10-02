#!/usr/bin/env python
# coding: utf-8

# Building a Linear Regression and KNN models with and without using PCA

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


student_data_mat   = pd.read_csv("../input/student_math.csv",delimiter=";")
student_data_por   = pd.read_csv("../input/student_language.csv",delimiter=";")


# In[ ]:


student_data_mat.head()


# In[ ]:


student_data_mat.shape


# In[ ]:


student_data_por.head()


# In[ ]:


student_data_por.shape


# In[ ]:


student_data = pd.merge(student_data_mat,student_data_por,how="outer")
student_data.head()


# In[ ]:


student_data.shape


# In[ ]:


student_data.info()


# In[ ]:


columns_string = student_data.columns[student_data.dtypes == object]
columns_string


# In[ ]:


student_data = pd.get_dummies(student_data, columns = columns_string, drop_first = True)
student_data.info()


# In[ ]:


student_data.head()


# In[ ]:


student_data.shape


# In[ ]:


student_data[["G1","G2","G3"]].corr()


# In[ ]:


student_data.drop(axis = 1,labels= ["G1"],inplace=True)
student_data.head()


# In[ ]:


label = student_data["G3"].values
predictors = student_data.drop(axis = 1,labels= ["G3"]).values


# In[ ]:


predictors


# # Principal Component Analysis

# Now we perform Principal Component Analysis to identify which of the predictors are the most valuable. For that we first calculate the explained_variance_

# In[ ]:


pca = PCA(n_components=len(student_data.columns)-1)
pca.fit(predictors)
variance = pca.explained_variance_
variance


# In[ ]:


print(pca.explained_variance_ratio_)


# In[ ]:


variance_ratio_cum_sum=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(variance_ratio_cum_sum)
plt.plot(variance_ratio_cum_sum)


# In[ ]:


pca = PCA(n_components=9)
pca.fit(predictors)
Transformed_vector =pca.fit_transform(predictors)
print(Transformed_vector)


# In[ ]:


student_data_without_output=student_data.drop(axis = 1,labels= ["G3"],inplace=False)
features=student_data_without_output.columns
features


# In[ ]:


#Visualize coefficients using heat map

plt.figure(figsize=[25,5])
sns.heatmap(pca.components_[0:,:],annot=True,cmap='viridis')
plt.yticks([1,2,3,4,5,6,7,8,9],["First component","Second component","Third component","Fourth component","Fifth component","Sixth component","Seventh component","Eighth component"],rotation=360,ha="right")
plt.xticks(range(len(features)),features,rotation=90,ha="left")
plt.xlabel("Feature")
plt.ylabel("Principal components")


# Inference : In the heatmap, few features with different colors apart from common color indicates its importance played in each pricipal component

# In[ ]:


lr_pca = linear_model.LinearRegression()
lr_withoutpca = linear_model.LinearRegression()


# In[ ]:


score_lr_withoutpca = cross_val_score(lr_withoutpca, predictors, label, cv=5)
print("PCA Model Cross Validation score : " + str(score_lr_withoutpca))
print("PCA Model Cross Validation Mean score : " + str(score_lr_withoutpca.mean()))


# In[ ]:


score_lr_pca = cross_val_score(lr_pca, Transformed_vector, label, cv=5)
print("PCA Model Cross Validation score : " + str(score_lr_pca))
print("PCA Model Cross Validation Mean score : " + str(score_lr_pca.mean()))


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV


# In[ ]:


knn = KNeighborsRegressor()


# In[ ]:


n_neighbors=[5,7,9,10,11]
weights=['distance','uniform']
metric =['euclidean','manhattan','chebyshev']


# In[ ]:


grid = GridSearchCV(estimator=knn,param_grid=dict(n_neighbors=n_neighbors,weights=weights,metric=metric))
grid.fit(predictors,label)


# In[ ]:


grid.best_params_


# In[ ]:


knn_withoutpca = KNeighborsRegressor(n_neighbors=10,weights='uniform',metric='euclidean')
knn_withoutpca.fit(predictors,label)


# In[ ]:


grid = GridSearchCV(estimator=knn,param_grid=dict(n_neighbors=n_neighbors,weights=weights,metric=metric))
grid.fit(Transformed_vector,label)


# In[ ]:


grid.best_params_


# In[ ]:


knn_withpca = KNeighborsRegressor(n_neighbors=11,weights='distance',metric='euclidean')
knn_withpca.fit(Transformed_vector,label)


# In[ ]:


score_knn_withoutpca = cross_val_score(knn_withoutpca, predictors, label, cv=5)
print("Model Without Cross Validation score : " + str(score_knn_withoutpca))
print("Model Without Cross Validation Mean score : " + str(score_knn_withoutpca.mean()))


# In[ ]:


score_knn_withpca = cross_val_score(knn_withpca, Transformed_vector, label, cv=5)
print("PCA Model Cross Validation score : " + str(score_knn_withpca))
print("PCA Model Cross Validation Mean score : " + str(score_knn_withpca.mean()))

