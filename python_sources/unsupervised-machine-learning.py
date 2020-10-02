#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# #**Data** **Preprocessing**

# In[ ]:


#import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, KFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import random
from sklearn.svm import SVC
import sklearn.metrics as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score,auc, roc_auc_score, roc_curve,confusion_matrix,classification_report


# In[ ]:


#change the dataset location
df = pd.read_csv('/kaggle/input/bank-marketing/bank-additional-full.csv', sep = ';')
df.shape


# In[ ]:


#viewing data
df.head()


# In[ ]:


#checking descriptive stats
df.describe()


# In[ ]:


#data info
df.info()
#No null values in the data


# In[ ]:


#Removing non-relevant variables
df1=df.drop(columns=['day_of_week','month','contact','poutcome'],axis=1)
df1


# In[ ]:


#Replacing all the binary variables to 0 and 1
df1.y.replace(('yes', 'no'), (1, 0), inplace=True)
df1.default.replace(('yes', 'no'), (1, 0), inplace=True)
df1.housing.replace(('yes', 'no'), (1, 0), inplace=True)
df1.loan.replace(('yes', 'no'), (1, 0), inplace=True)
df1


# In[ ]:


#creating Dummies for categorical variables
df2 = pd.get_dummies(df1)
df2.head()


# In[ ]:


#Removing extra dummy variables & checking descriptive stats
df3=df2.drop(columns=['job_unknown','marital_divorced','education_unknown'],axis=1)
df3.describe().T


# In[ ]:


#Correlation plot
plt.figure(figsize=(14,8))
df3.corr()['y'].sort_values(ascending = False).plot(kind='bar')


# In[ ]:


#Creating binary classification target variable
df_target=df3[['y']].values
df_features=df3.drop(columns=['y'],axis=1).values
df_target1=df3[['y']]
df_features1=df3.drop(columns=['y'],axis=1)


# ##Feature Selection

# In[ ]:


##Feature Selection
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier  
feature_selector = SequentialFeatureSelector(RandomForestClassifier(n_jobs=-1),  
           k_features=12,
           forward=True,
           verbose=2,
           scoring='roc_auc',
           cv=2)
features = feature_selector.fit(df_features1,df_target1)
filtered_features= df_features1.columns[list(features.k_feature_idx_)] 
filtered_features


# ##PCA

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pca = PCA()
pca_X=pca.fit_transform(df_features)
pca.get_covariance()


# In[ ]:


explained_variance=pca.explained_variance_ratio_
explained_variance.shape


# In[ ]:


plt.figure(figsize=(6, 4))
plt.bar(range(40), explained_variance, alpha=0.5, align='center',label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()


# In[ ]:


pca = PCA(n_components=2)
pca_X=pca.fit_transform(df_features)
pca.get_covariance()


# In[ ]:


explained_variance=pca.explained_variance_ratio_
explained_variance


# In[ ]:


plt.figure(figsize=(8,4))
plt.title('PCA Components')
plt.scatter(pca_X[:,0], pca_X[:,1])


# ##ICA

# In[ ]:


from sklearn.decomposition import FastICA 
ica = FastICA(n_components=3, random_state=2) 
ica_X=ica.fit_transform(df_features)
ica_X.shape


# In[ ]:


plt.figure(figsize=(8,4))
plt.title('ICA Components')
plt.scatter(ica_X[:,0], ica_X[:,1])
plt.scatter(ica_X[:,1], ica_X[:,2])
plt.scatter(ica_X[:,2], ica_X[:,0])


# ##RCA

# In[ ]:


from sklearn.random_projection import GaussianRandomProjection
rca = GaussianRandomProjection(n_components=3, eps=0.1, random_state=2)
rca_X=rca.fit_transform(df_features)
rca_X.shape


# In[ ]:


plt.figure(figsize=(12,8))
plt.title('RCA Components')
plt.scatter(rca_X[:,0], rca_X[:,1])
plt.scatter(rca_X[:,1], rca_X[:,2])
plt.scatter(rca_X[:,2], rca_X[:,0])


# #K-Means

# In[ ]:


# plot data
plt.scatter(
   df_features[:, 0], df_features[:, 1],
   c='white', marker='o',
   edgecolor='black', s=50
)
plt.show()


# In[ ]:


##Determining number of clusters
from sklearn.cluster import KMeans 
Sum_of_squared_distances = []
K = range(1,16)
for k in K:
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, init = 'k-means++', random_state = 2)
    km=km.fit(df_features)
    Sum_of_squared_distances.append(km.inertia_)
##Checking out which SSE is low for different types of k means value
plt.plot(K,Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow method for optimal k')
plt.show()


# In[ ]:


# Fitting K-Means to the dataset
from scipy import stats

kmeans = KMeans(n_clusters = 2, n_init=10, max_iter=300, init = 'k-means++', random_state = 2)
prediction = kmeans.fit_predict(df_features)
print(prediction)

plt.scatter(df_features[:, 0], df_features[:, 1], c=prediction, s=50)
centers = kmeans.cluster_centers_


# In[ ]:


centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)


# ##With Feature Selection

# In[ ]:


df_features1


# In[ ]:


df_fs=df_features1[['age', 'housing_0','housing_1','housing_unknown', 'loan_0','loan_1','loan_unknown', 'duration', 'campaign',
       'job_admin.', 'job_self-employed', 'job_technician', 'marital_single',
       'education_university.degree']].values
df_fs


# In[ ]:


##Determining number of clusters
from sklearn.cluster import KMeans 
Sum_of_squared_distances = []
K = range(1,16)
for k in K:
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, init = 'k-means++', random_state = 2)
    km=km.fit(df_fs)
    Sum_of_squared_distances.append(km.inertia_)
##Checking out which SSE is low for different types of k means value
plt.plot(K,Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow method for optimal k')
plt.show()


# In[ ]:


# Fitting K-Means to the dataset
from scipy import stats

kmeans = KMeans(n_clusters = 2, n_init=10, max_iter=300, init = 'k-means++', random_state = 2)
prediction = kmeans.fit_predict(df_fs)
print(prediction)

plt.scatter(df_fs[:, 0], df_fs[:, 1], c=prediction, s=50, cmap='viridis_r')


# In[ ]:


centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)


# ##With PCA

# In[ ]:


##Determining number of clusters
from sklearn.cluster import KMeans 
Sum_of_squared_distances = []
K = range(1,16)
for k in K:
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, init = 'k-means++', random_state = 2)
    km=km.fit(pca_X)
    Sum_of_squared_distances.append(km.inertia_)
##Checking out which SSE is low for different types of k means value
plt.plot(K,Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow method for optimal k')
plt.show()


# In[ ]:


# Fitting K-Means to the dataset
from scipy import stats

kmeans = KMeans(n_clusters = 2, n_init=10, max_iter=300, init = 'k-means++', random_state = 2)
prediction = kmeans.fit_predict(pca_X)
print(prediction)

plt.scatter(pca_X[:, 0], pca_X[:, 1], c=prediction, s=50, cmap='viridis_r')
centers = kmeans.cluster_centers_
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)


# ##With ICA

# In[ ]:


##Determining number of clusters
from sklearn.cluster import KMeans 
Sum_of_squared_distances = []
K = range(1,16)
for k in K:
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, init = 'k-means++', random_state = 2)
    km=km.fit(ica_X)
    Sum_of_squared_distances.append(km.inertia_)
##Checking out which SSE is low for different types of k means value
plt.plot(K,Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow method for optimal k')
plt.show()


# In[ ]:


# Fitting K-Means to the dataset
from scipy import stats

kmeans = KMeans(n_clusters = 2, n_init=10, max_iter=300, init = 'k-means++', random_state = 2)
prediction = kmeans.fit_predict(ica_X)
print(prediction)

plt.scatter(ica_X[:, 0], ica_X[:, 1], c=prediction, s=50, cmap='viridis_r')
centers = kmeans.cluster_centers_
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)


# ##With RCA

# In[ ]:


##Determining number of clusters
from sklearn.cluster import KMeans 
Sum_of_squared_distances = []
K = range(1,16)
for k in K:
    km = KMeans(n_clusters=k, n_init=10, max_iter=300, init = 'k-means++', random_state = 2)
    km=km.fit(rca_X)
    Sum_of_squared_distances.append(km.inertia_)
##Checking out which SSE is low for different types of k means value
plt.plot(K,Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow method for optimal k')
plt.show()


# In[ ]:


# Fitting K-Means to the dataset
from scipy import stats

kmeans = KMeans(n_clusters = 2, n_init=10, max_iter=300, init = 'k-means++', random_state = 2)
prediction = kmeans.fit_predict(rca_X)
print(prediction)

plt.scatter(rca_X[:, 0], rca_X[:, 1], c=prediction, s=50, cmap='viridis_r')
centers = kmeans.cluster_centers_
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)


# #Expectation Maximization

# In[ ]:


## Expectation maximization analysis
from sklearn.mixture import GaussianMixture
em = GaussianMixture(n_components=2,random_state=2,covariance_type='tied')
em_pred = em.fit_predict(df_features)
em_probs = em.predict_proba(df_features)
#em.means_
#em.covariances_
plt.scatter(df_features[:, 0], df_features[:, 1], c=em_pred, s=50, cmap='viridis_r')


# ##With Feature Selection

# In[ ]:


## Expectation maximization analysis
from sklearn.mixture import GaussianMixture
em = GaussianMixture(n_components=2,random_state=2,covariance_type='tied')
em_pred = em.fit_predict(df_fs)
em_probs = em.predict_proba(df_fs)
#em.means_
#em.covariances_
plt.scatter(df_fs[:, 0], df_fs[:, 1], c=em_pred, s=50, cmap='viridis_r')


# ##With PCA

# In[ ]:


## Expectation maximization analysis
from sklearn.mixture import GaussianMixture
em = GaussianMixture(n_components=2,random_state=2,covariance_type='tied')
em_pred = em.fit_predict(pca_X)
em_probs = em.predict_proba(pca_X)
#em.means_
#em.covariances_
plt.scatter(pca_X[:, 0], pca_X[:, 1], c=em_pred, s=50, cmap='viridis_r')


# ##With ICA

# In[ ]:


## Expectation maximization analysis
from sklearn.mixture import GaussianMixture
em = GaussianMixture(n_components=2,random_state=2,covariance_type='tied')
em_pred = em.fit_predict(ica_X)
em_probs = em.predict_proba(ica_X)
#em.means_
#em.covariances_
plt.scatter(ica_X[:, 0], ica_X[:, 1], c=em_pred, s=50, cmap='viridis_r')


# ##With RCA

# In[ ]:


## Expectation maximization analysis
from sklearn.mixture import GaussianMixture
em = GaussianMixture(n_components=2,random_state=2,covariance_type='tied')
em_pred = em.fit_predict(rca_X)
em_probs = em.predict_proba(rca_X)
#em.means_
#em.covariances_
plt.scatter(rca_X[:, 0], rca_X[:, 1], c=em_pred, s=50, cmap='viridis_r')


# #ANN after PCA

# In[ ]:


x1_train, x1_test, y1_train, y1_test = train_test_split(pca_X, df_target, test_size = 0.3, random_state = 0)


# In[ ]:


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(32,activation="softmax"))

# Adding the second hidden layer
classifier.add(Dense(16,activation="softmax"))

# Adding the output layer
classifier.add(Dense(1,activation="sigmoid"))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history=classifier.fit(x1_train, y1_train, batch_size = 10, epochs=100,validation_split=0.3)


# In[ ]:


# Making the Confusion Matrix
def confusionmat(y,y_hat):
  from sklearn.metrics import confusion_matrix,accuracy_score
  cm = confusion_matrix(y, y_hat)
  accu=accuracy_score(y,y_hat)
  print(cm,"\n")
  print("The accuracy is",accu)

#Accuracy and Loss Curves
def learningcurve(history):
  # list all data in history
  print(history.history.keys())
  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict_classes(x1_test)
pre_score = sk.average_precision_score(y1_test, y_pred)
classifier.summary()
test_results = classifier.evaluate(x1_test, y1_test)
print("For epoch = {0}, the model test accuracy is {1}.".format(100,test_results[1]))
print("The model test average precision score is {}.".format(pre_score))
confusionmat(y1_test,y_pred)
learningcurve(history)


# #Task 5

# In[ ]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2, n_init=10, max_iter=300, init = 'k-means++', random_state = 2)
prediction = kmeans.fit_predict(x1_train)

em = GaussianMixture(n_components=2,random_state=2,covariance_type='tied')
em_pred = em.fit_predict(x1_train)
em_probs = em.predict_proba(x1_train)

train_df = pd.DataFrame()
train_df['KM_Pred']=prediction
train_df['EM_Prob']=em_probs[:,1]
train_df['y']=y1_train
train_df


# In[ ]:


from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2, n_init=10, max_iter=300, init = 'k-means++', random_state = 2)
prediction = kmeans.fit_predict(x1_test)

em = GaussianMixture(n_components=2,random_state=2,covariance_type='tied')
em_pred = em.fit_predict(x1_test)
em_probs = em.predict_proba(x1_test)

test_df = pd.DataFrame()
test_df['KM_Pred']=prediction
test_df['EM_Prob']=em_probs[:,1]
test_df['y']=y1_test
test_df


# In[ ]:


#Creating binary classification target variable
train_y=train_df[['y']].values
train_x=train_df.drop(columns=['y'],axis=1).values
test_y=test_df[['y']]
test_x=test_df.drop(columns=['y'],axis=1)


# In[ ]:


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(32,activation="softmax"))

# Adding the second hidden layer
classifier.add(Dense(16,activation="softmax"))

# Adding the output layer
classifier.add(Dense(1,activation="sigmoid"))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
history=classifier.fit(train_x, train_y, batch_size = 10, epochs=100,validation_split=0.3)


# In[ ]:


# Predicting the Test set results
y_pred = classifier.predict_classes(test_x)
pre_score = sk.average_precision_score(test_y, y_pred)
classifier.summary()
test_results = classifier.evaluate(test_x, test_y)
print("For epoch = {0}, the model test accuracy is {1}.".format(100,test_results[1]))
print("The model test average precision score is {}.".format(pre_score))
confusionmat(test_y,y_pred)
learningcurve(history)

