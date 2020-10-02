#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
from matplotlib import style
style.use('seaborn')


# In[ ]:


df = pd.read_csv('path_to_csv/pulsar_stars.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.columns = ['mean_profile', 'std_profile', 'kurtosis_profile', 'skewness_profile', 'mean_dmsnr',
               'std_dmsnr', 'kurtosis_dmsnr', 'skewness_dmsnr', 'target']


# In[ ]:


df.head()


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


sns.boxplot(data = df)


# In[ ]:


#sns.heatmap(df.isnull())


# In[ ]:


sns.jointplot( x= 'skewness_profile',y='kurtosis_profile',data=df)


# In[ ]:


fig, axs  = plt.subplots(2, 2, figsize=(20, 10))

sns.kdeplot(df['mean_profile'], shade=True, ax=axs[0][0])
sns.kdeplot(df['std_profile'], shade=True, color='green', ax=axs[0][1])
sns.kdeplot(df['kurtosis_dmsnr'], shade=True, ax=axs[1][0])
sns.kdeplot(df['kurtosis_profile'], shade=True, color='green', ax=axs[1][1])


# In[ ]:


fig,ax = plt.subplots(figsize=(15,7))
sns.scatterplot(x='kurtosis_profile',y='kurtosis_dmsnr',hue='target',data=df)
plt.show()


# In[ ]:


fig,ax = plt.subplots(figsize=(18,7))
sns.scatterplot(x='skewness_profile',y='skewness_dmsnr',hue='target',data=df)
plt.show()


# In[ ]:


fig,ax = plt.subplots(figsize=(15,7))
sns.scatterplot(x='std_profile',y='std_dmsnr',hue='target',data=df)
plt.show()


# In[ ]:


with sns.axes_style('white'):
    ax = sns.jointplot(x="skewness_profile", y="mean_profile", data=df, kind='hex', height=9)


# ## Machine Learning

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,log_loss,f1_score,confusion_matrix


# In[ ]:


X = df.drop(['target'],axis=1)
y = df.target


# # DecisionTreeClassifier

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X,y)
acc_rep = {}


# In[ ]:


model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print('Accuracy score : ',accuracy_score(y_test,y_pred)*100,'%')
acc_rep['DecisionTreeClassifier_log_loss'] = log_loss(y_test,y_pred)
acc_rep['DecisionTreeClassifier_f1_score'] = f1_score(y_test,y_pred)


# In[ ]:


plt.imshow(np.log(confusion_matrix(y_test,y_pred)),cmap = 'Blues',interpolation='nearest')
plt.show()


# # KnearestNeighbour

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


# In[ ]:


scores,score,hscore,bestk = 0,0,0,0
for k in range(3,20):
    knnmodel =KNeighborsClassifier(n_neighbors=k)
    scores  = cross_val_score(knnmodel,X,y,cv=10)
    score = scores.mean()
    if score>hscore:
        hscore=score
        bestk = k
print('Best k is {} with cross_val_score : {}'.format(bestk,hscore))


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=bestk)
knn.fit(X_train,y_train)


# In[ ]:


y_pred = knn.predict(X_test)


# In[ ]:


print('Accuracy score : ',accuracy_score(y_test,y_pred)*100,'%')
acc_rep['kNN_log_loss'] = log_loss(y_test,y_pred)
acc_rep['kNN_f1_score'] = f1_score(y_test,y_pred)
plt.imshow(np.log(confusion_matrix(y_test,y_pred)),cmap = 'Blues',interpolation='nearest')
plt.show()


# # kMeans Clustering

# In[ ]:


from sklearn.cluster import KMeans
model = KMeans()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# In[ ]:


print('Accuracy score : ',accuracy_score(y_test,y_pred)*100,'%')
acc_rep['kMeans_Clustering_log_loss'] = log_loss(y_test,y_pred)
acc_rep['kMeans_Clustering_f1_score'] = f1_score(y_test,y_pred,average='binary')
plt.imshow(np.log(confusion_matrix(y_test,y_pred)),cmap = 'Blues',interpolation='nearest')
plt.show()


# # SVM

# In[ ]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)


# In[ ]:


y_pred = model.predict(X_test)
print('Accuracy score : ',accuracy_score(y_test,y_pred)*100,'%')
acc_rep['SVM_log_loss'] = log_loss(y_test,y_pred)
acc_rep['SVM_f1_score'] = f1_score(y_test,y_pred)
plt.imshow(np.log(confusion_matrix(y_test,y_pred)),cmap = 'Blues',interpolation='nearest')
plt.show()


# # Accuracy Report

# ---

# In[ ]:


acc_rep


# | Algorithm | Log_Loss | F1_Score | Accuracy (%)
# |-----------|----------|----------|----------|
# |DecisionTreeClassifier | 1.17|0.82 | 97.6|
# |KNN | 0.93| 0.85| 96.2|
# |kMeans | 33.19|NA | 97.2|
# |SVM | 0.93|0.84 | 97.2|
#  
