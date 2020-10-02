#!/usr/bin/env python
# coding: utf-8

# In[7]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[33]:


# read csv data
data = pd.read_csv("../input/column_2C_weka.csv")


# In[4]:


data.head()


# In[5]:


data.info()


# # A. SUPERVISED LEARNING

# * Supervised learning: It uses data that has labels. Example, there are orthopedic patients data that have labels normal and abnormal.
#     * There are features(predictor variable) and target variable. Features are like pelvic radius or sacral slope(If you have no idea what these are like me, you can look images in google like what I did :) )Target variables are labels normal and abnormal
#     * Aim is that as given features(input) predict whether target variable(output) is normal or abnormal
#     * Classification: target variable consists of categories like normal or abnormal
#     * Regression: target variable is continious like stock market
#     * If these explanations are not enough for you, just google them. However, be careful about terminology: features = predictor variable = independent variable = columns = inputs. target variable = responce variable = class = dependent variable = output = result

# **Data Analysis**

# In[6]:


data.describe()


# In[8]:


sns.countplot(x="class", data=data)
data.loc[:,'class'].value_counts()


# In[10]:


color_list = ['yellow' if i=='Abnormal' else 'red' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolor= "black")
plt.show()


# In[11]:


datax = data[data['class'] =='Abnormal']
x = np.array(datax.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(datax.loc[:,'sacral_slope']).reshape(-1,1)
# Scatter
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()


# **Linear Regression**

# * y = ax + b where y = target, x = feature and a = parameter of model
# * We choose parameter of model(a) according to minimum error function that is lost function
# * In linear regression we use Ordinary Least Square (OLS) as lost function.
# * OLS: sum all residuals but some positive and negative residuals can cancel each other so we sum of square of residuals. It is called OLS
# * Score: Score uses R^2 method that is ((y_pred - y_mean)^2 )/(y_actual - y_mean)^2

# In[15]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
# Predict space
predict_ = np.linspace(min(x), max(x)).reshape(-1,1)
# Fit
reg.fit(x,y)
# Predict
predicted = reg.predict(predict_)
# R^2 
print('R^2 score: ',reg.score(x, y))
# Plot regression line and scatter
plt.plot(predict_, predicted, color='red', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()


# **K-NEAREST NEIGHBORS (KNN)**

# * KNN: Look at the K closest labeled data points
# * Classification method.
# * First we need to train our data. Train = fit
# * fit(): fits the data, train the data.
# * predict(): predicts the data 
# * If you do not understand what is KNN, look at youtube there are videos like 4-5 minutes. You can understand better with it. 
# * Lets learn how to implement it with sklearn

# In[16]:


from sklearn.neighbors import KNeighborsClassifier
knnDt = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
knnDt.fit(x,y)
prediction = knnDt.predict(x)
print('Prediction: {}'.format(prediction))


# In[17]:


# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
knnDt = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
knnDt.fit(x_train,y_train)
prediction = knnDt.predict(x_test)
#print('Prediction: {}'.format(prediction))
print('With KNN (K=3) accuracy is: ',knnDt.score(x_test,y_test)) # accuracy


# In[19]:


neigBr = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neigBr):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neigBr, test_accuracy, label = 'Testing Accuracy')
plt.plot(neigBr, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neigBr)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# **UNSUPERVISED LEARNING**
# **KMEANS**

# * Unsupervised learning: It uses data that has unlabeled and uncover hidden patterns from unlabeled data. Example, there are orthopedic patients data that do not have labels. You do not know which orthopedic patient is normal or abnormal.
# 

# In[20]:


data2 = pd.read_csv('../input/column_2C_weka.csv')
plt.scatter(data2['pelvic_radius'],data2['degree_spondylolisthesis'])
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()


# # KMeans Clustering
# 

# * KMeans Cluster: The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity
# * KMeans(n_clusters = 2): n_clusters = 2 means that create 2 cluster

# In[22]:


data2_ = data2.loc[:,['degree_spondylolisthesis','pelvic_radius']]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(data2_)
labels = kmeans.predict(data2_)
plt.scatter(data2['pelvic_radius'],data['degree_spondylolisthesis'],c = labels)
plt.xlabel('pelvic_radius')
plt.xlabel('Degree Spondylolisthesis')
plt.show()


# **EVALUATING OF CLUSTERING**

# * There are two clusters that are 0 and 1
# * First class 0 includes 138 abnormal and 100 normal patients
# * Second class 1 includes 72 abnormal and 0 normal patiens *The majority of two clusters are abnormal patients.

# In[25]:


dataFrame = pd.DataFrame({'labels':labels,"class":data2['class']})
crossTab = pd.crosstab(dataFrame['labels'],dataFrame['class'])
print(crossTab)


# In[27]:


# inertia
inertia_list = np.empty(8)
for i in range(1,8):
    kMeans = KMeans(n_clusters=i)
    kMeans.fit(data2_)
    inertia_list[i] = kMeans.inertia_
plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.show()


# **STANDARDIZATION**

# * Standardizaton is important for both supervised and unsupervised learning
# * Do not forget standardization as pre-processing
# * As we already have visualized data so you got the idea. Now we can use all features for clustering.
# * We can use pipeline like supervised learning

# In[28]:


data_ = pd.read_csv('../input/column_2C_weka.csv')
data3_ = data.drop('class',axis = 1)


# In[30]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scalar = StandardScaler()
kMeans = KMeans(n_clusters = 2)
pipe = make_pipeline(scalar,kMeans)
pipe.fit(data3_)
labels = pipe.predict(data3_)
df = pd.DataFrame({'labels':labels,"class":data_['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)


# **HIERARCHY**

# * vertical lines are clusters
# * height on dendogram: distance between merging cluster
# * method= 'single' : closest points of clusters

# In[32]:


from scipy.cluster.hierarchy import linkage,dendrogram

merging = linkage(data3_.iloc[200:220,:],method = 'single')
dendrogram(merging, leaf_rotation = 90, leaf_font_size = 6)
plt.show()

