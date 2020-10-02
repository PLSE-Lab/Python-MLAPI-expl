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


# ## K Nearest Neighbour Classification

# 1. Read the data
# 2. Split the x and y variables from train and test.
# 3. Try to see how the data looks like by writing one record to csv
# 4. Looking at 5 samples from train and test.
# 5. Build knn model with k=3. Check the model performance.
# 6. look at the sample misclassified points.
# 7. Use grid search for model building and check the model performance.

# In[ ]:


#import the required packages for solving 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,mean_squared_error,accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# 1. **Read the data(both from train and test)**

# In[ ]:


train = pd.read_csv('../input/Train_sample.csv')
test = pd.read_csv('../input/Test_sample.csv')
train.head()


# In[ ]:


train.tail()


# In[ ]:


test.head()


# In[ ]:


test.tail()


# 2. **Split the x and y from train and test**

# In[ ]:


X_train = train.iloc[:,1:]
y_train = train.iloc[:,0]

print(X_train.shape)
print(y_train.shape)


# In[ ]:


X_test = test.iloc[:,1:]
y_test = test.iloc[:,0]

print(X_test.shape)
print(y_test.shape)


# 3. **Try to visually look at the data by writing one of the records to csv**

# In[ ]:


x1 = X_train.iloc[0,:].values.reshape(28,28)
x1[x1> 0] =1
x1 = pd.DataFrame(x1)
x1.to_csv("one.csv")


# 4. **Look at 5 sample records from train and test**

# In[ ]:


train_sample = np.random.choice(range(0,X_train.shape[0]),replace=False, size=5)
test_sample = np.random.choice(range(0,X_test.shape[0]),replace = False, size =5)


# In[ ]:


train_sample


# In[ ]:


test_sample


# In[ ]:


plt.figure(figsize=(10,5))
for i,j in enumerate(train_sample):
    plt.subplot(2,5,i+1)
    plt.imshow(X_train.iloc[j,:].values.reshape(28,28))
    plt.title("Digit:" +str(y_train[j]))
    plt.gray()


# In[ ]:


plt.figure(figsize=(10,5))
for i,j in enumerate(test_sample):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test.iloc[j,:].values.reshape(28,28))
    plt.title("Digit:"+str(y_test[j]))
    plt.gray()


# 5. **Build knn model with k =3. Check the model performace**

# In[ ]:


knn_classifier = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='brute')
knn_classifier.fit(X_train, y_train)


# In[ ]:


pred_train = knn_classifier.predict(X_train)
pred_test = knn_classifier.predict(X_test)


# Build confusion matrix and find the accuracy of the model

# In[ ]:


cm_test = confusion_matrix(y_pred=pred_test, y_true=y_test)

print(cm_test)


# In[ ]:


#Accuracy:
sum(np.diag(cm_test))/np.sum(cm_test)

#np.trace(cm_test)/np.sum(cm_test)


# In[ ]:


print("Accuracy on train is:" , accuracy_score(y_train,pred_train))
print("Accuracy on test is:", accuracy_score(y_test,pred_test))


# 6. **Look at the sample misclassified points**

# In[ ]:


misclassified = y_test[pred_test != y_test]


# In[ ]:


##First 5 misclassified ponts
misclassified.index[:5]


# In[ ]:


plt.figure(figsize=(10,5))
for i,j in enumerate(misclassified.index[:5]):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test.iloc[j,:].values.reshape(28,28))
    plt.title("Digit:"+str(y_test[j])+" "+"Pred:"+str(pred_test[j]))
    plt.gray()


# 7. **Use grid search for model building and check the model performance**

# In[ ]:


knn_classifier = KNeighborsClassifier(algorithm= 'brute', weights='distance')


# In[ ]:


params = {"n_neighbors": [1,3,5],"metric": ["euclidean", "cityblock"]}
#params = {"n_neighbors": [1],"metric": ["euclidean", "cityblock"]}

grid = GridSearchCV(knn_classifier,param_grid=params,scoring="accuracy",cv=10)


# In[ ]:


grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_params_)


# In[ ]:


best_knn = grid.best_estimator_
pred_train = best_knn.predict(X_train)
pred_test = best_knn.predict(X_test)
print("Accuracy o train is:", accuracy_score(y_train,pred_train))
print("Accuracy on test is:", accuracy_score(y_test,pred_test))


# ## K Nearest Neighbor Regression

# 1. Create random data with 3 variables and 1 target
# 2. Split the data into train and test
# 3. Scale the variables using minmax scaler
# 4. Build knn model with k = 3. Check the model performance
# 5. Use grid search for model building and check the model performance
# 
# 

# 1. **Craete random dat with 3 variables and 1 target**

# In[ ]:


##Randomly generate some data

data  = pd.DataFrame(np.random.randint(low = 2,high = 100,size = (1000, 4)),
                     columns=["Target","A","B","C"])
data.head()


# 2. **Split the data into train and Test**

# In[ ]:


train_x,test_x,train_y,test_y = train_test_split(data.iloc[:,1:],data.Target,test_size = 0.2)
print(train_x.shape, test_x.shape)


# 3. **Scale the variables using minmax scaler**

# In[ ]:


scaler = MinMaxScaler(feature_range=(0,1))

scaler.fit(train_x)


# In[ ]:


scaled_train_x = pd.DataFrame(scaler.transform(train_x),columns=['A','B','C'])
scaled_test_x = pd.DataFrame(scaler.transform(test_x),columns=["A","B","C"])


# 4. **Build knn model with k = 3 . Checck the model performance**

# In[ ]:


knn_regressor = KNeighborsRegressor(n_neighbors=3,algorithm="brute",weights="distance")
knn_regressor.fit(scaled_train_x, train_y)


# In[ ]:


train_pred = knn_regressor.predict(scaled_train_x)
test_pred = knn_regressor.predict(scaled_test_x)


# In[ ]:


print(mean_squared_error(train_y,train_pred))
print(mean_squared_error(test_y,test_pred))


# 5. **Use grid search for model building and check the model performance**

# In[ ]:


knn_regressor = KNeighborsRegressor(algorithm="brute",weights="distance")
params = {"n_neighbors": [1,3,5],"metric": ["euclidean", "cityblock"]}
grid = GridSearchCV(knn_regressor,param_grid=params,scoring="neg_mean_squared_error",cv=5)


# In[ ]:


grid.fit(scaled_train_x, train_y)
print(grid.best_params_)
print(grid.best_score_)


# In[ ]:


best_knn = grid.best_estimator_
train_pred = best_knn.predict(scaled_train_x)
test_pred = best_knn.predict(scaled_test_x)


# In[ ]:


print(mean_squared_error(train_y,train_pred))
print(mean_squared_error(test_y,test_pred))

