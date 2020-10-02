#!/usr/bin/env python
# coding: utf-8

# ### Classification using KNN <H3>
# 
# KNN is an algorithm used in machine learning unsupervised.
# 

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score  
from sklearn import neighbors
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# #### 1-Reading data <H4>
# 
# In this work, I used the database from kernel '[HCC survival (EDA + dataset cleaning)](http://https://www.kaggle.com/mirlei/hcc-survival-eda-dataset-cleaning)'. 
# 
# 

# In[ ]:


data = '../input/EDAdataHCC.csv'
dataHCC = pd.read_csv(data)
dataHCC.head(5)


# #### 2-Separating the data<h4>
# 
# It is necessary to separate the label from the attributes. 
# 

# In[ ]:


array = dataHCC.values
X = array[:,1:38]  # feature vector
y = array[:,40]  # class label 


# In[ ]:


#Counting the instances healthy and unhealthy
print ("Instances:", y.size)
print ("Class 0:", y[y==0].size)
print ("Class 1:", y[y==1].size)


# To use knn, it's necessary defining the test set and the training set. For the test set, thirty percent of the data set was defined.

# In[ ]:


X_train, X_test, y_train, y_test =    train_test_split(X, y, test_size=0.3, random_state=1)


# In[ ]:


print('Features - X_train: ', X_train.shape)
print('Class label - y_train: ', y_train.shape)
print('Features - X_test: ', X_test.shape)
print('Class label - y_test: ', y_test.shape)


# #### 3-Pipeline <h4>
# In this step, I used PCA and StandardScaler to improve the data. 

# In[ ]:


X_train= StandardScaler().fit_transform(X_train)
X_test= StandardScaler().fit_transform(X_test)


# In[ ]:


pca = PCA(n_components = 5)
X_train =pca.fit_transform(X_train)
X_test =pca.fit_transform(X_test)


# In[ ]:


#PLOT
plt.scatter(X_train[:,0],X_train[:,1],  c=y_train) 


# #### 4-Classification KNN <h4>
# 
# 4.1- Defining neighbors number. 

# In[ ]:


######Function plot with diferents k's
def plotvector(XTrain, yTrain, XTest, yTest, weights):
    results = []
    
    for n in range(1, 25, 2):
        clf = neighbors.KNeighborsClassifier(n_neighbors=n, weights=weights)
        clf = clf.fit(XTrain, yTrain)
        preds = clf.predict(XTest)
        accuracy = clf.score(XTest, yTest)
        results.append([n, accuracy])
 
    results = np.array(results)
    return(results)

###### Plot 
pltvector1 = plotvector(X_train, y_train, X_test, y_test, weights="uniform")
line1 = plt.plot(pltvector1[:,0], pltvector1[:,1], label="uniform")

plt.legend(loc=3)
plt.ylim(0.5, 0.8)
plt.title("Accuracy with different K's")
plt.grid(True)
plt.show()


# 4.2- modeling and training.

# In[ ]:


clfknn = KNeighborsClassifier(algorithm='auto', metric='euclidean', n_neighbors=3, weights='uniform')
clfknn.fit(X_train, y_train)
#To prediction use the line below -  To prediction you need the test sample. 
#y_pred = classifier.predict(X_test)  


# 4.3-Accuracy. 

# In[ ]:


accuracy_knn= clfknn.score(X_train, y_train)
print ("Accuracy - (train):", accuracy_knn)


# In[ ]:


accuracy_knn = clfknn.score(X_test, y_test)
print ("Accuracy - (test):", accuracy_knn)


# In[ ]:


# Predict using test sample
predict_test = clfknn.predict(X_test)


# In[ ]:


print("Confusion matrix - kNN (auto, euclidean, uniform)")
print("{0}".format(metrics.confusion_matrix(y_test, predict_test, labels=[1, 0])))
print(classification_report(y_test, predict_test))  


# 4.4-Cross validation. 

# In[ ]:


all_accuracies = cross_val_score(estimator=clfknn, X=X_train, y=y_train, cv=10)  
print('All accuracies:',all_accuracies)  
print('Mean accuracy:',all_accuracies.mean())  

