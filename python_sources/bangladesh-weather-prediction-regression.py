#!/usr/bin/env python
# coding: utf-8

# # <center> Rain prediction </center>

# ## Recurrent Nural Net Approach

# In[ ]:


## importing basic modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#try:
#    !pip install tensorflow-gpu
#except:
#!pip install tensorflow
import tensorflow as tf

import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,20)


# # importing dataset

# In[ ]:



dataset = pd.read_csv('../input/Temp_and_rain.csv')


# In[ ]:


dataset.head()


# ## checking missing data

# ## No missing value

# In[ ]:


dataset.isnull().sum()


# In[ ]:


dataset[['rain']].plot()


# ## histogram of the rain 

# In[ ]:


dataset.rain.hist()


# ## histogram of the temp

# In[ ]:


dataset.tem.hist()


# ## Rain in different year

# In[ ]:


plt.bar(dataset['Year'],dataset['rain'])
plt.xlabel("Year")
plt.ylabel("Rain")
plt.legend()


# ## Temp to Rain plot

# In[ ]:


plt.bar(dataset['tem'],dataset['rain'])
plt.xlabel("TEMP")
plt.ylabel("Rain")
plt.legend()


# In[ ]:


import seaborn as sns


# In[ ]:


correlation = dataset.corr()


# ## Correlation matrix

# In[ ]:


correlation


# In[ ]:


sns.heatmap(correlation,cmap='coolwarm',annot=True)


# In[ ]:


## setting the style first
sns.set(style="whitegrid",color_codes=True) ## change style


# In[ ]:


sns.distplot(dataset['rain'], kde=False, bins=100);


# In[ ]:


sns.distplot(dataset['tem'],kde=False, bins=100);


# ## relational scatter plot of differnt rain quantity in different Year

# In[ ]:


sns.relplot(x="Year", y="rain", data=dataset);


# In[ ]:


sns.relplot(x="Year", y="tem", data=dataset);


# ## relation betwen temp and rain in different year

# In[ ]:


sns.relplot(x="Year", y="tem", hue="rain", data=dataset);


# ## box plot

# In[ ]:


sns.boxplot(data=dataset,orient='h')


# # model selection with preprocessing

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# ## splitting the feature matrix and target

# In[ ]:


dataset.head()
X = dataset.drop('rain',axis=1)
X = X.drop('tem',axis=1)
y = dataset[['rain','tem']]


# In[ ]:


X.head()


# In[ ]:


y.head()


# ###  to use RNN you have to maintain the value between a limit hence transforming it
# 
# # we transform the y so after predict we have to inverse transeform it

# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.2)


# ## importing RNN module

# In[ ]:


from keras import Sequential
from keras.layers import Dense,Dropout,LSTM,Flatten


# In[ ]:


print (x_train.shape)
print (x_test.shape)


# In[ ]:


x_train = np.array(x_train)
x_test = np.array(x_test)


# In[ ]:


x_train


# ## this reshaping is very important before feeding to RNN

# In[ ]:


print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape)


# In[ ]:


n_col = x_train.shape[1]


# ## CREATE A BASIC RNN MODEL

# In[ ]:


from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
visible = Input(shape=(n_col,))
hidden1 = Dense(100, activation='relu')(visible)
hidden2 = Dense(200, activation='relu')(hidden1)
hidden3 = Dense(100, activation='relu')(hidden2)
hidden4 = Dense(100, activation='relu')(hidden3)
hidden5 = Dense(100, activation='relu')(hidden4)
hidden6 = Dense(100, activation='relu')(hidden5)
hidden7 = Dense(100, activation='relu')(hidden6)
output = Dense(2)(hidden7)
model = Model(inputs=visible, outputs=output)
model.compile(optimizer='adam',loss='mean_absolute_error')


# In[ ]:


model.fit(x_train,y_train,epochs = 100)


# In[ ]:


y_pred = model.predict(x_test)


# In[ ]:


y_pred


# In[ ]:


model.evaluate(x_test,y_test)


# ## KNN APPROACH

# In[ ]:


dataset = pd.read_csv('../input/Temp_and_rain.csv')
X = dataset.drop('rain',axis=1)
X = X.drop('tem',axis=1)
y = dataset[['rain','tem']]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=.2)


# In[ ]:


from sklearn.neighbors import KNeighborsRegressor


# In[ ]:


knn = KNeighborsRegressor(n_neighbors=5)


# In[ ]:


knn.fit(x_train,y_train)
predicted=knn.predict(x_test)


# In[ ]:


predicted


# In[ ]:


model.evaluate(x_test,y_test)


# In[ ]:


accuracy=[]
for k in range(1,50):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(x_train,y_train)
    accuracy.append(knn.score(x_test,y_test))


    


# In[ ]:


plt.plot(range(1,50),accuracy)


# In[ ]:


training_accuracy=[]
testing_accuracy=[]

neighbors = list(range(1,50))


for k in neighbors:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(x_train,y_train)
    training_accuracy.append(knn.score(x_train,y_train))
    testing_accuracy.append(knn.score(x_test,y_test))    

plt.plot(neighbors,training_accuracy,label='training accuracy')
plt.plot(neighbors,testing_accuracy,label='testing accuracy')
plt.ylabel("Accuracy")
plt.xlabel("K value")
plt.legend()


# ## CROSS VAL SCORE

# In[ ]:


from sklearn.model_selection import cross_val_score

knn = KNeighborsRegressor(n_neighbors=5)

scores = cross_val_score(knn,X,y,cv=10)

print (scores)
print (scores.mean())

print ("Mean Accuracy "+str(scores.mean()))



# # now we find cross val score for different K value

# In[ ]:





k_range = range(1,50)
k_scores = []

for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    scores = cross_val_score(knn,X,y,cv=10)
    k_scores.append(scores.mean())
    
print (k_scores)


plt.plot(k_range,k_scores)
plt.xlabel("k range")
plt.ylabel("scores")


# ## Decision Tree Regresson

# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


tree_clf = DecisionTreeRegressor(max_depth=2,random_state=42)


# In[ ]:


tree_clf.fit(X,y)


# In[ ]:


tree_clf.score(X,y)


# ## For different Depth

# In[ ]:


accuracy=[]
for depth in range(1,50):
    dt = DecisionTreeRegressor(max_depth=depth,random_state=42)
    dt.fit(x_train,y_train)
    accuracy.append(dt.score(x_test,y_test))


# In[ ]:


plt.plot(range(1,50),accuracy)


# ## RANDOM FOREST

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


rnd = RandomForestRegressor(max_depth=10)


# In[ ]:


rnd.fit(x_train,y_train)


# In[ ]:


rnd.score(x_test,y_test)


# In[ ]:


accuracy=[]
for depth in range(1,50):
    dt = RandomForestRegressor(max_depth=depth,random_state=42)
    dt.fit(x_train,y_train)
    accuracy.append(dt.score(x_test,y_test))


# In[ ]:


plt.plot(range(1,50),accuracy)


# ## Combining all the algorithm

# ![](http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor_files/stackingregression_overview.png)

# In[ ]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from mlxtend.regressor import StackingRegressor
dtc=DecisionTreeRegressor()
knnc=KNeighborsRegressor()
gbc=GradientBoostingRegressor()
rfc=RandomForestRegressor()


# In[ ]:


stregr = StackingRegressor(regressors=[dtc,knnc,gbc,rfc], 
                           meta_regressor=knnc)


# In[ ]:


y_train


# In[ ]:


stregr.fit(x_train, y_train['tem'])


# In[ ]:


prediction = stregr.predict(x_test)


# In[ ]:


stregr.score(x_test,y_test['tem'])


# In[ ]:


stregr.fit(x_train, y_train['rain'])
prediction = stregr.predict(x_test)
stregr.score(x_test,y_test['rain'])


# In[ ]:




