#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


dat = pd.read_csv("/kaggle/input/wineuci/Wine.csv", names= [ 'Cultivator','Alcohol', 'Malic acid','Ash','Alcalinity_of_ash','Magnesium',
                                        'Total_phenols','Flavanoids','Nonflavanoid_phenols','Proanthocyanins'
                                        ,'Color_intensity','Hue','diluted_wines','Proline'])


# In[ ]:


dat.head()


# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(dat.corr(),annot=True)
plt.show()


# In[ ]:


cor= dat.corr()
#Correlation with output variable
cor_target = abs(cor["Cultivator"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features


# In[ ]:


i = sns.pairplot(dat, vars = ['Alcalinity_of_ash', 'Total_phenols', 'Flavanoids', 'Hue', 'diluted_wines', 'Proline'] ,hue='Cultivator', palette='husl')
plt.show()


# In[ ]:


# Wow, all the selected features shows distinctive varaition in the proprotion to estimate the cultivator


# ## Let's Remove the target and try to estimate the same with the clustering techniques
# ### K_Mean

# In[ ]:


# Removing the target
data= dat.iloc[:,1:]


# In[ ]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[ ]:


sc= StandardScaler()
data_scaled= sc.fit_transform(data)


# In[ ]:


num_cluster= range(1,15)
culster_error=[]

for cluster in num_cluster:
    model= KMeans(n_clusters=cluster,n_init=10)
    model.fit(data_scaled)
    label= model.labels_
    centroids=model.cluster_centers_
    culster_error.append(model.inertia_)


# In[ ]:


cluster_df= pd.DataFrame({"num_cluster": num_cluster, "error": culster_error})


# In[ ]:


cluster_df


# In[ ]:


# Elbow Curve
plt.plot(cluster_df["num_cluster"],cluster_df["error"],marker="*")
plt.show()


# In[ ]:


# Optimal Cluster point is 3
model= KMeans(n_clusters=3,n_init=10)
model.fit(data_scaled)


# In[ ]:


centroids= pd.DataFrame(model.cluster_centers_ , columns= list(data.columns))
centroids


# In[ ]:


data['Ktest_label']=model.labels_


# In[ ]:


data['Ktest_label'].value_counts()


# In[ ]:


dat['Cultivator'].value_counts()


# In[ ]:


# Lets Try a base prediction with the help of logistic regression as it binary classification


# In[ ]:


X= data.iloc[:,:-1]
y= data['Ktest_label']


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# In[ ]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[ ]:


y_test.value_counts()


# In[ ]:


y_train.value_counts()


# In[ ]:


lr=LogisticRegression(random_state=1)
lr.fit(X_train,y_train)


# In[ ]:


test_pred= lr.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


y_test.value_counts()


# In[ ]:


confusion_matrix(y_test,test_pred)


# In[ ]:


Precision = 23/25 *100  # True Positive/ True Positive and False Negative predicted
print(Precision)
Recall    =  23/24 *100
print(Recall)


# In[ ]:


Precision = 16/18 *100  # True Positive/ True Positive and False Negative predicted
print(Precision)
Recall    =  16/18 *100
print(Recall)


# In[ ]:


Precision = 11/11 *100  # True Positive/ True Positive and False Negative predicted
print(Precision)
Recall    =  11/12 *100
print(Recall)


# In[ ]:


print(classification_report(y_test,test_pred))


# In[ ]:


# Lets give a try with the Kmean again to see whether they are classifying correctly


# In[ ]:


kmeans=KMeans(n_clusters=3,n_init=10,random_state=1)
kmeans.fit(X_train,y_train)


# In[ ]:


pred_test=kmeans.predict(X_test)
pred_train=kmeans.predict(X_train)


# In[ ]:


print(classification_report(y_train,pred_train))


# In[ ]:


print(classification_report(y_test,pred_test))


# In[ ]:


y_test.value_counts()


# In[ ]:


confusion_matrix(y_test,pred_test)


# In[ ]:


Precision = 18/18 *100  # True Positive/ True Positive and False Negative predicted
print(Precision)
Recall    =  18/24 *100
print(Recall)


# In[ ]:


Precision = 13/19 *100  # True Positive/ True Positive and False Negative predicted
print(Precision)
Recall    =  13/18 *100
print(Recall)


# In[ ]:


Precision = 6/17 *100  # True Positive/ True Positive and False Negative predicted
print(Precision)
Recall    =  6/12 *100
print(Recall)


# In[ ]:


# K_Mean even though we performed Clustering the prediction is worst for both train and test data


# In[ ]:


# Hierarchical Clustering to classifiy the model better
from scipy.cluster.hierarchy import dendrogram,linkage
from sklearn.cluster.hierarchical import AgglomerativeClustering
from scipy.stats import zscore


# In[ ]:


X= X.apply(zscore)


# In[ ]:


plt.figure(figsize=(22,14))
Z= linkage(X,method='ward')
dendrogram(Z)
plt.show()


# In[ ]:


clus = AgglomerativeClustering(n_clusters=3)
hs=clus.fit_predict(X)


# In[ ]:


clus = KMeans(n_clusters=3)
cc=clus.fit_predict(X)


# In[ ]:


df_k=data.copy(deep=True)
df_k['label'] = cc


# In[ ]:


df_h=data.copy(deep=True)
df_h['label'] = hs


# In[ ]:


data


# In[ ]:


df_h['label'].value_counts() # Hirerchical Clustering


# In[ ]:


df_k['label'].value_counts() # Kmeans Clustering


# In[ ]:


dat['Cultivator'].value_counts() # Actual Value


# In[ ]:


pd.DataFrame(kmeans.cluster_centers_,columns=list(X.columns)) # Estimating the centroids of the cluster


# ### Selcting the correlated features for visualisation
# * > Cultivator           1.000000
# * > Alcalinity_of_ash    0.517859
# * > Total_phenols        0.719163
# * > Flavanoids           0.847498
# * > Hue                  0.617369
# * > diluted_wines        0.788230
# * > Proline              0.633717

# In[ ]:


dat['Cultivator'].value_counts()


# In[ ]:


plt.figure(figsize=(12,12))
plt.title('Original Classes')
sns.scatterplot(x='Flavanoids', y='diluted_wines', hue='Cultivator', data=dat,palette="rocket")
plt.show()
plt.figure(figsize=(12,12))
plt.title('K-Means Classes')
sns.scatterplot(x='Flavanoids', y='diluted_wines', hue='label', data=df_k,palette="rocket")
plt.show()
plt.figure(figsize=(12,12))
plt.title('Hierarchical Classes')
sns.scatterplot(x='Flavanoids', y='diluted_wines', hue='label', data=df_h,palette="rocket")
plt.show()


# In[ ]:


centroids= pd.DataFrame(kmeans.cluster_centers_, columns=list(X.columns))
centroids[['Flavanoids','diluted_wines']]


# In[ ]:


# Inference Hirerachical Clustering is performing the best compared to KMeans Clustering


# In[ ]:


# 3 plot
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


data


# In[ ]:


df_scaled= pd.DataFrame(data_scaled,columns=list(X.columns))


# In[ ]:


fig = plt.figure(figsize=(10,8))
ax= Axes3D(fig, rect=[0,0,1,1],elev=20,azim=150)
labels = df_k['label']
ax.scatter(df_scaled['Flavanoids'],df_scaled['diluted_wines'],df_scaled['Total_phenols'],
          c= labels.astype(np.float),edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

ax.set_xlabel("Flavanoids")
ax.set_ylabel("diluted_wines")
ax.set_zlabel("Total_phenols")
ax.set_title("3D plot of KMeansClustering")


# In[ ]:


fig = plt.figure(figsize=(10,8))
ax= Axes3D(fig, rect=[0,0,1,1],elev=20,azim=200)
labels = df_h['label']
ax.scatter(df_scaled['Flavanoids'],df_scaled['diluted_wines'],df_scaled['Total_phenols'],
          c= labels.astype(np.float),edgecolor='k')
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

ax.set_xlabel("Flavanoids")
ax.set_ylabel("diluted_wines")
ax.set_zlabel("Total_phenols")
ax.set_title("3D plot of Hirerachical Clustering")


# In[ ]:




