#!/usr/bin/env python
# coding: utf-8

# The subject of this work is to create a **binary classification model** to know whether a client would subscribe to a term deposit.
# 
# This information would be based on phone calls made by marketing campaigns. 
# 
# One of the issues that make this dataset interesting is that the dataset is unbalanced, which means we have a frequency bigger in class 0 than in class 1.
# 
# To achieve this model we will explore different algorithms like **RF, NB, KNN SVM , Neural Networks,** and techniques like **"undersampling"**.
# 
# We will also have to pay attention to the way we measure the model.
# 

# The dataset has information about customers and telemarketing campaigns made by a bank. There is a column named "y"  in the dataset which tells us whether or not the customer opted for term deposit. This is divided into 17 attributes, being 8 categorical and 9 of factors.
# 
# What's more exciting is the distribution of "y". We have 39922 "no" vs 5289 "yes".
# 
# That means we have 95% of class 0 and just 5% of class 1.
# 
#  Finally, the size of the dataset (45211 rows) is big enough for working with some algorithms.
#  
# Below you can see the meaning of each attribute:
# 
# 
# Age :numeric
# 
# Job :Type of job (categorical)
# 
# Marital : Marital status
# 
# Education : Level of education of the customer
# 
# default : Has credit in default? (binary: "yes","no")
# 
# Balance : average yearly balance, in euros (numeric)
# 
# Housing : has housing loan? (binary: "yes","no")
# 
# Loan : has personal loan? (binary: "yes","no")
# 
# Contact: contact communication type (categorical: "unknown","telephone","cellular")
# 
# Day : last contact day of the month (numeric)
# 
# Month : last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
# 
# Duration : last contact duration, in seconds (numeric), important
# 
# Campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
# 
# Pdays : number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
# 
# Previous :  number of contacts performed before this campaign and for this client (numeric)
# 
# Poutcome : outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
# 
# Y : has the client subscribed a term deposit? (binary : 0, 1)
# 
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.model_selection import KFold

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/bank-marketing-term-deposit/bank_customer_survey.csv")


# In[ ]:


data.head()


# First .We check for missing values.

# In[ ]:


data.isnull().sum()


# We have 7 numeric attributes. "Describe()" give us important information like the mean, min , max , percentiles ,and standard deviation.

# In[ ]:


data.describe()


# **but what happen when  we separete by class "y" ?**

# In[ ]:


data[data['y']==0].describe()


# In[ ]:


data[data['y']==1].describe()


# now we can said sometime more about our data:
# * We have 39922 rows of class 0 and 5289 rows of class 1
# * We have an imbalance dataset (95% vs 5%)
# * Duration looks quite different in  the last 2 tables (mean:221 vs 537 )
# 

# In[ ]:


data[ data['age'] > 90 ].head()


# Calls with a duration less than 8 are class 0. it means they don't subscribe to a term deposit. It makes sense.

# In[ ]:


data[ data['duration'] < 8 ]['y'].count()


#  Next code we'll transform raw data into an understandable format

# In[ ]:


from sklearn.preprocessing import LabelEncoder

newdata = data
le = LabelEncoder()
for col in newdata.columns:
    if(newdata[col].dtype == 'object'):
        newdata.loc[:,col] = le.fit_transform(newdata.loc[:,col])

newdata.head()


# In[ ]:


corr = newdata.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between the two variables.
# 
#   If we look at the last column we can look at the correlation of all variables with "y". Duration has the biggest correlation with "y".Also, the matrix shows us a bigger correction. between attributes related to time.
# 

# Now, We are going to try some algorithms with different hyper-parameters.These are :
# * Random Forest
# * k-nearest neighbors 
# * Naive Bayes
# * Support vector machine

# In[ ]:


X = newdata.iloc[:,:-1].values
y = newdata.iloc[:,-1].values


# In[ ]:


#If you want to try KNeighborsClassifier uncomment lines 13,21,and comment  14 and 22

scoresAc = []
scoresF1 = []

preds = []
actual_labels = []
# Initialise the 5-fold cross-validation
kf = KFold(n_splits=10,shuffle=True)


for i in range(1,15):
  #model1= neighbors.KNeighborsClassifier(n_neighbors = i)
  model2= RandomForestClassifier(max_depth=i ,n_estimators = 200,n_jobs= 5) 
  aux1 =[]
  aux2 = []
  for train_index,test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #estimator = model1.fit(X_train,y_train)
    estimator = model2.fit(X_train,y_train)
    
    predictions = estimator.predict(X_test)
    scoreF1 = metrics.f1_score(y_test,predictions)
    accuracy = metrics.accuracy_score(y_test,predictions)
    aux1.append(accuracy)
    aux2.append(scoreF1)
  
  scoresAc.append(np.average(aux1))
  scoresF1.append(np.average(aux2))

#print("F1 score: {0}".format((scoresF1)))

#print("accuracy score: {0}".format((scoresAc)))
report = classification_report(y_test, predictions)
print(report)

plt.plot(range(1,15), scoresAc, label="training accuracy")
plt.plot(range(1,15), scoresF1, label="F1 accuracy") 
plt.ylabel("score")
plt.xlabel("max_depth")
plt.legend()


# ![](http://)

# In[ ]:


#If you want to try SVC uncomment lines 21,22,and comment  18 and 19



scoresAc = []
scoresF1 = []

preds = []

# Initialise the 5-fold cross-validation
kf = KFold(n_splits=10,shuffle=True)

for train_index,test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

  
  model3 = GaussianNB()
  estimator = model3.fit(X_train, y_train)

  #model4 = SVC(C=1000, kernel = "rbf")
  #estimator = model4.fit(X_train, y_train)

  predictions = estimator.predict(X_test)
  scoreF1 = metrics.f1_score(y_test,predictions)
  accuracy = metrics.accuracy_score(y_test,predictions)

  
scoresAc.append(accuracy)
scoresF1.append(scoreF1)

print("F1 score: {0}".format((np.average(scoreF1))))

print("accuracy score: {0}".format(np.average(scoresAc)))

report = classification_report(y_test, predictions)
print(report)


# tabla con resultados .
# ![allALG.png](attachment:allALG.png)

# Now, we are gonna do the same, but this time we are using  Undersampling techniques. Undersampling can be defined as removing some observations of the majority class. So we are going to remove a bunch of files (to train our model).

# In[ ]:


sizeY = data['y'].count()
print ("number of observations:",sizeY)

#using undersampling
sizeClass0=data[data['y']==0]['y'].count()
print ("number of observations with class 0:",sizeClass0)

sizeClass1=data[data['y']==1]['y'].count()
print ("number of observations with class 1:",sizeClass1)


#preprocesing
newdata= data
le = LabelEncoder()
for col in newdata.columns:
    if(newdata[col].dtype == 'object'):
        newdata.loc[:,col] = le.fit_transform(newdata.loc[:,col])


# We are going to create dt with class1 and 5000 elements of class0
# in testclass0 We save the remaining elements of class0
dataClass0 = newdata[newdata['y']==0]
dataClass1 = newdata[newdata['y']==1]

perm = np.random.permutation(sizeClass0)

# underSampling  with sizeClass1 

split_point = sizeClass1
#split_point = int(np.ceil(sizeClass0*0.5))

dataClass0 = dataClass0.values
dataClass1 = dataClass1.values


class0ForTrain = dataClass0[perm[:split_point].ravel(),:] 

testClass0 = dataClass0[perm[split_point:].ravel(),:] 

dt = np.concatenate((class0ForTrain,dataClass1))


print('length of dt contains class1 and 5000 elements of class0', len(dt))
print('length of remaining elements of class0 :', len(testClass0))

X = dt[:,:-1]
y = dt[:,-1]

XTestClass0 = testClass0[:,:-1]
yTestClass0 = testClass0[:,-1]


# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
scoresAc = []
scoresF1 = []

preds = []

kf = KFold(n_splits=10,shuffle=True)

for train_index,test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

  model1 = GaussianNB()
  estimator = model1.fit(X_train, y_train)

  #model1 = SVC(C=30, kernel = "rbf")
  #estimator = model1.fit(X_train, y_train)

  #model1= RandomForestClassifier(max_depth=16 ,n_estimators = 200) 
  #estimator = model1.fit(X_train, y_train)

  #model1 = neighbors.KNeighborsClassifier(n_neighbors = 2)
  #estimator = model1.fit(X_train, y_train)


  XfinalTest = np.concatenate((X_test,XTestClass0))
  yfinalTest = np.concatenate((y_test,yTestClass0))

  predictions = model1.predict(XfinalTest)

  scoreF1 = metrics.f1_score(yfinalTest,predictions)
  accuracy = metrics.accuracy_score(yfinalTest,predictions)

  #predictions = estimator.predict(X_test)
  #scoreF1 = metrics.f1_score(y_test,predictions)
  #accuracy = metrics.accuracy_score(y_test,predictions)

  
scoresAc.append(accuracy)
scoresF1.append(scoreF1)

print("F1 score: {0}".format((np.average(scoreF1))))

print("accuracy score: {0}".format(np.average(scoresAc)))

report = classification_report(yfinalTest, predictions)
print(report)


# tabla![US.png](attachment:US.png)

# We didn't improve the performance of F1-score.Although we improved the sensitivity (recall of NB: 78%).
# 
# So instead of reducing the observations of class 0 to the size of class 1. We are going to removing samples from the majority class to half. 

# In[ ]:


sizeY = data['y'].count()
print ("number of observations:",sizeY)

#undersampling
sizeClass0=data[data['y']==0]['y'].count()
print ("number of observations with class 0:",sizeClass0)

sizeClass1=data[data['y']==1]['y'].count()
print ("number of observations with class 1:",sizeClass1)


#preprocesing
newdata= data
le = LabelEncoder()
for col in newdata.columns:
    if(newdata[col].dtype == 'object'):
        newdata.loc[:,col] = le.fit_transform(newdata.loc[:,col])


# We are going to create dt with class1 and 19961 elements of class0
# in testclass0 We save the remaining elements of class0
dataClass0 = newdata[newdata['y']==0]
dataClass1 = newdata[newdata['y']==1]

perm = np.random.permutation(sizeClass0)

# underSampling  with class0ForTrain = sizeClass0 / 2


split_point = int(np.ceil(sizeClass0*0.5))

dataClass0 = dataClass0.values
dataClass1 = dataClass1.values


class0ForTrain = dataClass0[perm[:split_point].ravel(),:] 

testClass0 = dataClass0[perm[split_point:].ravel(),:] 

dt = np.concatenate((class0ForTrain,dataClass1))


print('length of dt contains class1 and 19961 elements of class0', len(dt))
print('length of remaining elements of class0 :', len(testClass0))

X = dt[:,:-1]
y = dt[:,-1]

XTestClass0 = testClass0[:,:-1]
yTestClass0 = testClass0[:,-1]


# In[ ]:


scoresAc = []
scoresF1 = []

preds = []

kf = KFold(n_splits=10,shuffle=True)

for train_index,test_index in kf.split(X):
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

  #model1 = GaussianNB()
  #estimator = model1.fit(X_train, y_train)

  #model1 = SVC(C=30, kernel = "rbf")
  #estimator = model1.fit(X_train, y_train)

  model1= RandomForestClassifier(max_depth=16 ,n_estimators = 200,n_jobs= 5) 
  estimator = model1.fit(X_train, y_train)

  #model1 = neighbors.KNeighborsClassifier(n_neighbors = 2)
  #estimator = model1.fit(X_train, y_train)


  XfinalTest = np.concatenate((X_test,XTestClass0))
  yfinalTest = np.concatenate((y_test,yTestClass0))

  predictions = model1.predict(XfinalTest)

  scoreF1 = metrics.f1_score(yfinalTest,predictions)
  accuracy = metrics.accuracy_score(yfinalTest,predictions)

  #predictions = estimator.predict(X_test)
  #scoreF1 = metrics.f1_score(y_test,predictions)
  #accuracy = metrics.accuracy_score(y_test,predictions)

  
scoresAc.append(accuracy)
scoresF1.append(scoreF1)

print("F1 score: {0}".format((np.average(scoreF1))))

print("accuracy score: {0}".format(np.average(scoresAc)))

report = classification_report(yfinalTest, predictions)
print(report)


# ![US05.png](attachment:US05.png)

# Ultimately we are going to try  using a neural network.

# In[ ]:



from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

X = newdata.iloc[:,:-1].values
y = newdata.iloc[:,-1].values

perm = np.random.permutation(y.size)

PRC = 0.80
split_point = int(np.ceil(y.shape[0]*PRC))
X_train = X[perm[:split_point].ravel(),:] 
y_train = y[perm[:split_point].ravel()]

X_test = X[perm[split_point:].ravel(),:]

y_test = y[perm[split_point:].ravel()]


y1 = to_categorical(y)

y_train1 = y1[perm[:split_point].ravel()]
y_test1 = y1[perm[split_point:].ravel()]

#We standardize features by removing the mean and scaling to unit variance
sc = StandardScaler()
X_trainS = sc.fit_transform(X_train)
X_testS = sc.fit_transform(X_test)


# In[ ]:



model = Sequential()

#get number of columns in training data
n_cols_2 = X.shape[1]

#add layers to model
model.add(Dense(250, activation='relu', input_shape=(n_cols_2,)))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(2, activation='softmax'))


#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


# In[ ]:


model.fit(X_train, y_train1, epochs=20, batch_size=32)

predictions = model.predict(X_test)


y_pred1 = predictions[:,0] < 0.5

k = classification_report(y_test, y_pred1)
print (k)


# In[ ]:


#Same neural network but after standardize their features.
model.fit(X_trainS, y_train1, epochs=20, batch_size=32)

predictions = model.predict(X_testS)


y_pred1 = predictions[:,0] < 0.5

k = classification_report(y_test, y_pred1)
print (k)


# 
# ![NN.png](attachment:NN.png)
# 

# After all these experiments, we learned some lessons.
# 
# 
# Firstly, as we discussed in class, we should avoid  measuring accuracy for unbalanced data, instead, we should use other metrics such F1, recall, and precision,  since these metrics give us more information about the behavior of our model.
# In our first try, our  4 models achieved had an accuracy of up to 85%. (Random Forest had  an accuracy of 91%) but F1 was not very well. So, we tried some techniques of undersampling and we got better results.
# 
# 
# Secondly, we have many tools that choose the best hyperparameters for us
# (like grid-search) . But it is still good, making a chart to show different scores with different hyperparameters.
# .For example in random Forest, we learned that increasing the depth of the tree gives us a better F1 up until  it  starts to converge.
# 
# Finally, it is important to understand our dataset, but also what is the interest or  the purpose for our dataset?. Are we more concerned about precision? Or instead, we want to be sure of having the bigger amount of buyers. In particular, we have to think about the cost. For example, The cost of lending to a defaulter is far greater than the loss-business cost of refusing a loan to a non-defaulter. By the other side in promotional mailing: the cost of sending junk mail to a household that doesn't respond is far less than the lost-business cost of not sending it to a household that would have responded.
# 
#  We should ask these questions. I think we are particularly interested in class 1 (the buyers). Next, we can do other analyzes to arrive at a better model like precision or recall).
# 
# If we are interested in maximizing the rate of calls to buyers; We will choose a model that pays more attention to the sensitivity of class 1. It could be NB or Randomn Forest
#  using undersampling .these have a sensitivity (recall) of 79% and 68% respectively.
# 
# But suppose we decided to hire a specialist for the calls that we predicted as buyers. In this case, we are interested in precision. Consequently, Randon Forest and  3-layer neural network have the best accuracy (69% and 61% respectively)
# 
# If I have to choose, I'll say Random forest. Although RF using undersampling could be a good option too. If we observe our different models we have a trade-off between precision and recall.
# 
#  In conclusion, some of the steps to follow could be:
# * Keep experimenting with neural networks, the ones we used gave us good results. But other variants are worth trying.
# * explore other techniques of undersampling. We applied one of these techniques and they got good results 
# * get more information from the dataset and the client.
# * we could think of making a  priority queue with the elements that we predict as class 1. The criteria to choose  elements( that we predicted as class one) would be the probability that they are 1
# 
