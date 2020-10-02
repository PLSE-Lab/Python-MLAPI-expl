#!/usr/bin/env python
# coding: utf-8

# PCA vs TSNE 

# just importing the necessary libraries

# In[ ]:


from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.linalg import eigh
import seaborn as sn
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score,confusion_matrix,mean_squared_error,r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',10000)
np.random.seed(2)#This is important for scientific works, only with a seed you can replicate computations that uses random elements.


# from scikit-learn load the iris data

# In[ ]:


iris_data=load_iris()
print(iris_data.keys())


# setting the pandas display window in order to see all columns and rows

# In[ ]:


pd.set_option('display.max_columns',1000)
pd.set_option('display.max_rows',1000)
pd.set_option('display.width',1000)


# In[ ]:


df_data=pd.DataFrame(iris_data.data,columns=iris_data.feature_names,index=None)
df_target=pd.DataFrame(iris_data.target,columns=['class'])
df_target.loc[df_target['class'] ==0, 'Target_names'] = 'setosa'
df_target.loc[df_target['class'] ==1, 'Target_names'] = 'versicolor'
df_target.loc[df_target['class'] ==2, 'Target_names'] = 'virginica'
target_names=iris_data.target_names
print('features :\n',df_data.head(5))
print('labels :\n',df_target)


# In[ ]:


df_data


# In[ ]:


df_data['species']=df_target['Target_names']
sn.pairplot(data=df_data,kind='scatter', hue='species')


# In[ ]:


df_data=df_data.drop(['species'],axis=1)
standardized_data = StandardScaler().fit_transform(df_data)
print('standardized_data shape :\n',standardized_data.shape)
print('standardized_data data :\n',standardized_data)


# In[ ]:


sample_data=standardized_data
pca = decomposition.PCA()
pca.n_components = 4
pca_data = pca.fit_transform(sample_data)
# pca_reduced will contain the 2-d projects of simple data
print("shape of pca_reduced.shape = ", pca_data.shape)


# In[ ]:


pca.n_components = 4
pca_data = pca.fit_transform(sample_data)
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);
cum_var_explained = np.cumsum(percentage_var_explained)
# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))
plt.clf()
plt.plot(cum_var_explained, linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()


# now applying random forest classifier on pca tranformed data

# In[ ]:


y=df_target['Target_names']
x=pca_df.drop(['label'], axis=1)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.20,random_state=0)
model=RandomForestClassifier(max_depth=6)
model.fit(X_train,Y_train)
yp=model.predict(X_test)
print('accuracy ',accuracy_score(Y_test,yp))


# In[ ]:


model = TSNE(n_components=, random_state=0)
# configuring the parameteres
# the number of components = 2
# default perplexity = 30
# default learning rate = 200
# default Maximum number of iterations for the optimization = 1000
tsne_data = model.fit_transform(df_data)
# creating a new data frame which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, df_target['Target_names'])).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "Dim_3", "Dim_4", "label"))


# now applying random forest classifier on TSNE tranformed data

# In[ ]:


y=df_target['Target_names']
x=tsne_df.drop(['label'], axis=1)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.20,random_state=0)
print(X_train.shape)
model=RandomForestClassifier()
model.fit(X_train,Y_train)
yp=model.predict(X_test)
print('accuracy ',accuracy_score(Y_test,yp))


# TSNE with 40 perplexity and 4k iters

# In[ ]:


model = TSNE(n_components=2, random_state=0, perplexity=40,  n_iter=4000)
tsne_data = model.fit_transform(df_data)
# creating a new data fram which help us in ploting the result data
tsne_data = np.vstack((tsne_data.T, df_target['Target_names'])).T
tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
# Ploting the result of tsne
sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
plt.title('With perplexity = 40, n_iter=4000')
plt.show()


# aaplying random forest classifier on TSNE transformed data with 40 perplexity

# In[ ]:


y=df_target['Target_names']
x=tsne_df.drop(['label'], axis=1)
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.20,random_state=0)
model=RandomForestClassifier()
model.fit(X_train,Y_train)
yp=model.predict(X_test)
print('acc ',accuracy_score(Y_test,yp))

