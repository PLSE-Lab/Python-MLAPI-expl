#!/usr/bin/env python
# coding: utf-8

# **PLEASE UP VOTE ME!!!!**
# 
# In this Kernel, I will take up a use case for identifying anamoly in the data, and thereby being able to predict anamoly in data. I will take a use case of credit card fraud data.
# 
# Firstly, we will have a look at our provided data, and glean insights by using data exploratory and visualization tools.
# 
# Then we will explore different ways to look for anamolies, and compare & contrast between them. We will start with Unsupervised learning, in that we will develop KMeans Cluster & IsolationForest Model, and use them to predict anamolies. After that, we will use Supervised learning methods, in that we will use gradient boost & Logistic regression to predict anamolies.
# 
# Finally, we will dive into deep learning methods. We will build a deep sequential model, and predict anamolies.
# 
# This Kernel is for a beginner to understand how to work through different steps in building a model, and selecting the right one.
# 
# **Have Fun!!! - Lijesh Shetty..**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))


# Any results you write to the current directory are saved as output.


# Lets read data from the credit card csv file. 
# You can read more about this data, but essentially the dimension of the data has been reduced by applying PCA (Principal Component Analysis). 

# In[ ]:


# Lets read the data into a dataframe
df = pd.read_csv('../input/creditcard.csv')
df.head()


# Its good to always do df.info() & df.describe()
# These methods help you to find out whether there are any invalid/empty data, and how the data is spread as well.

# In[ ]:


# split into label and features
target_label = df['Class']
target_label.value_counts()
X = df.iloc[:,:-1]
df.info() # there are 284806 rows with 31 columns including the target label 'Class' in the provided data.


# Lets look at correlation of features. This will tell us how the features are correlated with each other. 
# Correlation gives us a intution on which variables are important, and have impact on the predicted class.

# In[ ]:


# Now lets look at the correlationin the data. 
# Prior to that we will split out data into training and testing set.
from sklearn.model_selection import train_test_split 
train_X, test_X, train_y, test_y = train_test_split(X,target_label,test_size=0.3,random_state=42)
# find the correlation between the different variables.
corr_mtx = df.corr()
corr_mtx
print(corr_mtx['Class'].sort_values(ascending = False)) 
# V11 thru V27 are the features which have the most correlation impact on Class, and there are attributes which are negatively 
# correlated as well

    


# Here I am plotting feature data, and seeing their spread. Histogram will help us to do that. :)

# In[ ]:


# Here we will plot a hist of all features to see the spread of data.
# doing a visual on data helps to further understand the data.
import matplotlib.pyplot as plt
X.hist(figsize=(20,21))
plt.show()


# Lets try our first model, and see where it gets us..
# 
# **KMEANS**
# 
# We will try with KMeans Clustering algorithm. We will seggregate the data in different clusters, and see whether the fraud gets segregated differently...Also, I will train the model using the non-fraud data (meaning, I will remove the fraud data out, and train the model with non fraud data only). Then we will feed in fraud data, and see how the prediction is..
# 
# You can feed in the entire data set, including Fraud, and see how the model behaves. 
# 
# In the below step, I am separating fraud from non-fraud, and creating train and set set for non fraud data.

# In[ ]:


# Separate out Fraud & Non-Fraud Data, and split to get train & test set.
non_fraud_data = df[df.Class == 0]
fraud_data = df[df.Class == 1]
np.unique(non_fraud_data.Class)


non_fraud_label = non_fraud_data['Class']
non_fraud_X = non_fraud_data.iloc[:,:-1]
non_fraud_X.head()

fraud_label = fraud_data['Class']
fraud_X = fraud_data.iloc[:,:-1]

non_train_X, non_test_X, non_train_y, non_test_y = train_test_split(non_fraud_X,non_fraud_label,test_size=0.3,random_state=42)


# In[ ]:


# Lets look at Kmeans...
# Lets see whether we can segregate data in Clusters.
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

ks = range(1, 6)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
   # Fit model to samples
    model.fit(non_train_X)
   # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


# From the above plot of inertia with number of clusters, we can see that the inertial change is smaller post the 3 clusters. We will now build our model for three clusters, and use it for predictions.

# In[ ]:


# lets use three clusters..

from sklearn.cluster import KMeans

model = KMeans(n_clusters=3, random_state=42)
k_labels = model.fit_predict(non_fraud_X,non_fraud_label)


# In[ ]:


print('len of fraud', len(fraud_X))
print('len of non_fraud_X', len(non_fraud_X))
print('len of df', len(df))


# KMeans was not really helpful here. 
# Ofcourse, you can try including fraud data when you are training the model, and see how it works.
# The predictions has classified them in one of the existing clusters. Not much helpful.
# So lets move on to other predictor models.

# In[ ]:


fraud_predict_labels = model.predict(fraud_X)
np.unique(fraud_predict_labels)

len(fraud_predict_labels[fraud_predict_labels < 0])


# **Lets do an Isolation Forest. Isolation Forest is used to find anamoly.******

# In[ ]:


# compute outlier_fraction for the model here
outlier_fraction = len(fraud_X)/len(df)
print(outlier_fraction)

from sklearn.ensemble import IsolationForest
clf = IsolationForest(n_estimators=10, max_samples= len(train_X),contamination = outlier_fraction,n_jobs=5, random_state=42, behaviour ='new')
clf.fit(train_X)
score = clf.decision_function(train_X)

y_pred_train = clf.predict(train_X)
y_pred_test = clf.predict(test_X)


# Lets try to find accuracy and precision of our model for the test data.
# 
# First we have to align both the prediction and the observation. To match with Observation categorical values, we change predictions values to '1', when it is fraud, and '0' when it is not a fraud.
# We will build a confusion matrix, and then calculate our precision and accuracy values.

# In[ ]:


# lets try to build confusion matrix and get precision and accuracy scores.
y_pred_test[y_pred_test == 1] = 0
y_pred_test[y_pred_test == -1] = 1
np.unique(y_pred_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

cnf_mtrx = confusion_matrix(test_y,y_pred_test)
cnf_mtrx


# Only 44 fraud's have been detected by our model, and 93 have been missed whereas 92 have been incorrectly classified as fraud by our model.

# In[ ]:


accuracy_score(train_y,y_pred_train)
############################################################################################################################
# Calculate precison & recall. 
# Precision is actual positive prediction/ total positive prediction
# Recall is actual positive prediction/ total actual positives
############################################################################################################################
precision = cnf_mtrx[1,1]/(cnf_mtrx[1,1]+cnf_mtrx[0,1])
recall = cnf_mtrx[1,1]/(cnf_mtrx[1,1]+cnf_mtrx[1,0])
print("precision is {0}, and recall is {1}".format(precision,recall))


# Precision and recall both are around 32%. Although, this may seem low, but the results are fantastic! 
# Model is now able to detect 32% of fraud cases.

# Lets change our gear to using Supervised Learning. We will use ADABOOST & Logistic Regression, and see where it gets us...
# 
# **ADABOOST**

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier

clf_ada = AdaBoostClassifier(n_estimators=100, random_state=42)
clf_ada.fit(train_X,train_y) 


# In[ ]:


test_y_ada_predict = clf_ada.predict(test_X)

# lets build confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

ada_test_cnf_mtrx = confusion_matrix(test_y,test_y_ada_predict)
ada_test_cnf_mtrx


# In[ ]:


##Find precision and recall for AdaBoost Model
precision = ada_test_cnf_mtrx[1,1]/(ada_test_cnf_mtrx[1,1]+ada_test_cnf_mtrx[0,1])
recall = ada_test_cnf_mtrx[1,1]/(ada_test_cnf_mtrx[1,1]+ada_test_cnf_mtrx[1,0])
print("precision is {0}, and recall is {1}".format(precision,recall))


# **WOW!!** If you were excited about 32% accuracy with IsolationForest, AdaBoost has been able to give us 87% accuracy. We will able to predict 87 out of 100 Fraud cases, and can potentially stop them before happening :)

# Lets use a simple Logistic Regression Model and check our precision.
# 
# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression

lg_clf = LogisticRegression(penalty='l2',tol=0.0001,random_state=42)
lg_clf.fit(train_X,train_y)


# In[ ]:


test_y_lg_predict = lg_clf.predict(test_X)

# lets build confusion matrix
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

lg_test_cnf_mtrx = confusion_matrix(test_y,test_y_lg_predict)
lg_test_cnf_mtrx


# In[ ]:


##Find precision and recall for Logistic Regression Model
precision = lg_test_cnf_mtrx[1,1]/(lg_test_cnf_mtrx[1,1]+lg_test_cnf_mtrx[0,1])
recall = lg_test_cnf_mtrx[1,1]/(lg_test_cnf_mtrx[1,1]+lg_test_cnf_mtrx[1,0])
print("precision is {0}, and recall is {1}".format(precision,recall))


# A simple Log Regression model has given us a precision of 69% and recall of 60%.
# Now lets dive into Deep Learning Model, and see how further it can take us......

# Now..Lets do DeepLearning using Keras and TensorFlow as our backend.
# I have built a very simple model with no hidden layers.
# I have 128 neuron for the input layer, and output layer has 2 neurons (0 or 1) with softmax activation..
# I have used to_categorical method to change to binary output for train_y & test_y label array.
# I have used sgd optimizer, and then I run the model...

# In[ ]:


from keras import  backend as K

from keras.models import Sequential
from keras.layers.core import Dense,Activation, Flatten, Dropout
from keras.optimizers import Adam,RMSprop, SGD

from keras.utils import np_utils
import numpy as np

n_cols = train_X.shape[1]
y_train = np_utils.to_categorical(train_y,2)
y_test = np_utils.to_categorical(test_y,2)
print('shape is',n_cols)

model = Sequential()
model.add(Dense(128,activation='relu',input_shape=(n_cols,)))
model.add(Dense(2,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
history = model.fit(train_X,y_train,batch_size=1000,epochs=200,verbose='VERBOSE',validation_split = 0.2)


# **WOW!!! Blown away....**
# train accuracy is 99.82% whereas test accuracy is 99.84%.
# 
# I am satisfied with this outcome. Question is, **Are You?**

# In[ ]:


score1 = model.evaluate(train_X,y_train)
print('train score',score1[0])
print('train accuracy',score1[1])

score = model.evaluate(test_X,y_test)
print('test score',score[0])
print('test accuracy',score[1])

