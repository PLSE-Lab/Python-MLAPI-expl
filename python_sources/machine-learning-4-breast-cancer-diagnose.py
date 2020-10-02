#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# import warnings
import warnings
# ignore warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/data.csv')


# In[ ]:


# information of our data (lenght, how many NaN objects we have etc.)
data.info()


# In[ ]:


# to see features and target variable of the first 5 rows
data.head()


# In[ ]:


# to see features and target variable of the last 5 rows
data.tail()


# * **drop unnecessary columns**

# In[ ]:


data = data.drop(['Unnamed: 32', 'id'], axis=1)
data.head()


# * **how many types of diagnosis do we have?**

# In[ ]:


data['diagnosis'].value_counts()


# In[ ]:


import seaborn as sns
sns.countplot(data['diagnosis'])


# *** change diagnosis column to integers**

# In[ ]:


data['diagnosis'] = [1 if x=='M' else 0 for x in  data['diagnosis']]


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


color_list = ['red' if i==1.0 else 'green' for i in data.loc[:,'diagnosis']]
pd.plotting.scatter_matrix(data.iloc[:, 7:13],
                                       c=color_list,
                                       figsize= [10,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolor= "black")
plt.show()


# **Choosing x and y values**
# 1. x is our features except diagnosis (classification columns)
# 2. y is diagnosis

# In[ ]:


x = data.iloc[:,1:]
y = data['diagnosis']


# **Split our data into train data and test data**

# In[ ]:


# train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1) # 0.3 means 30% of data is splitted for testing. Remaining 70% is used to train our data


# **K-NEAREST NEIGHBORS (KNN)**
# 1. k is a selected number (for example k=3)
# 2. we select k nearest neighbors of a particle that we want to predict
# 3. in this selected set, if we have more M diagnosis than B diagnosis, we predict this particle as M. Or vise versa  

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train) # to train our data
predicted_values = knn.predict(x_test)
correct_values = np.array(y_test) # just to make them array


# In[ ]:


predicted_values


# In[ ]:


correct_values


# **KNN Prediction Score**

# In[ ]:


print('KNN (with k=3) accuracy is: ',knn.score(x_test,y_test)) # accuracy


# **k value**
# 1. k value is as hyperparameter
# 2. but is this selected k value (k=3) optimum one?

# In[ ]:


neig = np.arange(1,25)
test_accuracy = []
# Loop over different values of k
for k in neig:
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.legend()
plt.title('k-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))


# **REGRESSION**

# **Linear Regression**
# 1. y = ax + b where y = target, x = feature and a = parameter of model
# 2. We choose parameter of model(a) according to minimum error function that is lost function

# In[ ]:


data1 = data[data['diagnosis'] == 1.0]
x = np.array(data1.loc[:,'fractal_dimension_mean']).reshape(-1,1)
y = np.array(data1.loc[:,'symmetry_mean']).reshape(-1,1)
# Scatter
plt.figure(figsize=[10,10])
plt.scatter(x=x,y=y)
plt.xlabel('fractal_dimension_mean')
plt.ylabel('symmetry_mean')
plt.show()


# In[ ]:


# LinearRegression
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
# data to be predicted
test_data = np.linspace(min(x), max(x)).reshape(-1,1)
# Fit
LR.fit(x,y)
# Predict
predicted = LR.predict(test_data)
# score
print('score: ',LR.score(x, y))
# Plot regression line and scatter
plt.figure(figsize=(10,10))
plt.plot(test_data, predicted, color='black', linewidth=3)
plt.scatter(x=x,y=y)
plt.xlabel('fractal_dimension_mean')
plt.ylabel('symmetry_mean')
plt.show()


# **CROSS VALIDATION**
# * In KNN method if we change train-test data,  the score can change. But if the score changes according to our data to which score values should I rely on?
# * the answer is I dont know. But if I use cross validation, I can find a reasonable accuracy.

# In[ ]:


# CV
from sklearn.model_selection import cross_val_score
reg = LinearRegression()
k = 5 # 5 times split train and predict 
cv_result = cross_val_score(reg,x,y,cv=k) # uses R^2 as score 
print('CV Scores: ',cv_result)
print('CV scores average: ',np.sum(cv_result)/k)


# **Random Forest Regression**

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
x = data.iloc[:,1:]
y = data['diagnosis']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1) 
RF = RandomForestClassifier(random_state = 4)
RF.fit(x_train,y_train)
y_predicted = RF.predict(x_test)


# **Confusion matrix with random forest**

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_predicted)
print('Confusion matrix: \n',cm)


# **Classification Report**

# In[ ]:


from sklearn.metrics import classification_report
print('Classification report: \n',classification_report(y_test,y_predicted))


# In[ ]:


# visualize with seaborn library
sns.heatmap(cm,annot=True,fmt="d") 
plt.show()


# we see that for the diagnosis 0 we have predicted 106 of them correctly (2 false prediction) (98% accuracy for diagnosis 0)
# 
# on the other hand, 
# 
# we see that for the diagnosis 1 we have predicted 56 of them correctly (7 false prediction) (89% accuracy for diagnosis 1)

# **Logistic Regression**

# In[ ]:


x = data.iloc[:,1:]
y = data['diagnosis']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 1)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train,y_train)
y_pred = logreg.predict(x_test)


# In[ ]:


print('logistic regression score: ', logreg.score(x_test, y_test))


# **Cross Validation for logistic regression**

# In[ ]:


# CV
from sklearn.model_selection import cross_val_score
logreg = LogisticRegression()
k = 5 # 5 times split train and predict 
cv_result = cross_val_score(logreg,x,y,cv=k) # uses R^2 as score 
print('CV Scores: ',cv_result)
print('CV scores average: ',np.sum(cv_result)/k)


# **HYPERPARAMETER TUNING**
# 
# GridSearchCV is utilized for finding the hyperparameters (KNN)

# In[ ]:


# grid search cross validation with 1 hyperparameter (KNN)
from sklearn.model_selection import GridSearchCV
grid = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=3) # GridSearchCV
knn_cv.fit(x_train,y_train)# Fit

# Print hyperparameter
print("Tuned hyperparameter k: {}".format(knn_cv.best_params_)) 
print("Best score: {}".format(knn_cv.best_score_))


# In[ ]:




