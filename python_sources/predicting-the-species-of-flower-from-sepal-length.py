#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Important libraries are loaded along with the Iris Species dataset

# In[ ]:


import pandas as pd
import numpy as np
import math
import operator
df = pd.read_csv("../input/Iris.csv")
print(df.head()) 
df.shape
from collections import Counter


# ## Importing the model for training and testing splitting

# In[ ]:


from sklearn.model_selection import train_test_split
x=df[["SepalLengthCm"]]
y=df["Species"]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.33)
len(y_train)


# # Developing KNN from scratch

# In[ ]:


def predict(x_train, y_train, x_test, k):
    # create list for distances and targets
    distances = []
    targets = []
    
    for i in range(len(x_train)):
        # first we compute the euclidean distance
        distance = np.sqrt(np.sum(np.square(x_test - x_train.values[i, :])))
        # add it to list of distances
        distances.append([distance, i])

	# sort the list
    distances = sorted(distances)
# make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train.values[index])

# return most common target
    return Counter(targets).most_common(1)[0][0]      
        


# In[ ]:


def train(x_train,y_train):
    return
def kNearestNeighbor(x_train, y_train, x_test, predictions, k):
	# train on the input data
	train(x_train, y_train)

	# loop over all observations
	for i in range(len(x_test)):
		predictions.append(predict(x_train, y_train, x_test.values[i, :], k))


# ## Now, we will predict the species using the x_test data

# # Here, the value of K=9, we obtained from the graph between error and K values

# In[ ]:


predictions =[]

from sklearn.metrics import accuracy_score
kNearestNeighbor(x_train, y_train, x_test, predictions, 9)

# transform the list into an array
predictions = np.asarray(predictions)

# evaluating accuracy
accuracy = accuracy_score(y_test, predictions)
for i in range(len(x_test)):
    print("Flower with sepal length",x_test.iloc[i],":")
    print("belongs to the kingdom",predictions[i])




# # Checking for the accuracy score of our model

# In[ ]:


print("accuracy score is :",accuracy*100,"%")


# In[ ]:


from sklearn import neighbors
from sklearn.metrics import mean_squared_error 
from math import sqrt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from sklearn.model_selection import train_test_split
x=df[["SepalLengthCm"]]
y=df["PetalLengthCm"]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.33)
len(y_train)


# In[ ]:


rmse_values = [] #to store rmse values for different k
for k in range(20):
    k = k+1
    model = neighbors.KNeighborsRegressor(n_neighbors = k)

    model.fit(x_train, y_train)  #fit the model
    pred=model.predict(x_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_values.append(error) #store rmse values
    print('RMSE value for k= ' , k , 'is:', error)


# In[ ]:


min(rmse_values)


# # Minimum value of RMSE is 0.7899 corresponding to K=11
# # Also, it is verified that both the KNN models that is KNN from scratch and KNN from inbuilt model,predicts the same value of K=11

# # Plot between RMSE and K

# In[ ]:


import matplotlib.pyplot as plt
plt.xlabel('Value of K')
plt.ylabel('RMSE')
plt.plot(range(20),rmse_values)
plt.show()


# # It is clear that the min value of RMSE is coressponding to approx 10.

# ## Predicting the Petal length from Sepal Length

# In[ ]:


predict=model.predict(x_test)


# In[ ]:


len(predict)


# In[ ]:





# In[ ]:





# # Predictions

# In[ ]:


for i in range(len(predict)):
    print("For sepal length:",x_test.values[i])
    print("The coressponding petal length in centimeters is:",predict[i])


# In[ ]:


import seaborn as sns
plt.scatter(x_test,predict)
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")


# In[ ]:


K=[]
for x in range(1,21):
    j=1/x
    K.append(j)


# # A plot between RMSE and 1/K

# In[ ]:


plt.plot(rmse_values,K)
plt.xlabel("1/K")
plt.ylabel("RMSE Values")


# In[ ]:




