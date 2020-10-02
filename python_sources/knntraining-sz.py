#!/usr/bin/env python
# coding: utf-8

# # KNN Algorithms Tutorial
# 1. Importing Libraries
# 1. Importing the Dataset
# 1. Plotting Data
# 1. Preprocessing
# 1. Train Test Split
# 1. Feature Scaling
# 1. Training and Predictions
# 1. Evaluating the Algorithm
# 1. Comparing Error Rate with the K Value

# ## Importing Libraries
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import warnings
warnings.filterwarnings('ignore') 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## Importing the Dataset

# In[ ]:


data = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_2C_weka.csv')
data3 = pd.read_csv('/kaggle/input/biomechanical-features-of-orthopedic-patients/column_3C_weka.csv')


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


type(data)


# In[ ]:


data.isnull().sum()


# * There is no missing value ,so we dont need to perform on it .

# In[ ]:


def discrete_univariate(dataset, discrete_feature):
    fig, axarr=plt.subplots(nrows=1,ncols=2, figsize=(8,5))
      
    dataset[discrete_feature].value_counts().plot(kind="bar",ax=axarr[0])
    dataset[discrete_feature].value_counts().plot.pie(autopct="%1.1f%%",ax=axarr[1])
        
    plt.tight_layout()
    plt.show()


# ## Plotting Data
# 

# In[ ]:


discrete_univariate(dataset=data , discrete_feature="class")


# In[ ]:


discrete_univariate(dataset=data3, discrete_feature="class")


# In[ ]:


sns.pairplot(data ,hue ="class",palette="husl")
plt.show()


# ## Preprocessing

# In[ ]:


#%%  Normal =1  Abnormal =0
data['class'] = [1 if each == "Normal" else 0 for each in data['class']]


# In[ ]:



y = data.loc[:,'class']

x1 = data.loc[:,data.columns != 'class']


# Normalization for better understand 

# In[ ]:


x = (x1 - np.min(x1))/(np.max(x1)-np.min(x1))


# ## Train Test Split

# In[ ]:



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2 ,random_state=1)


# ## Feature Scaling
# Before making any actual predictions, it is always a good practice to scale the features so that all of them can be uniformly evaluated

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)

X_train = scaler.transform(x_train)
X_test = scaler.transform(x_test)


# ## Training and Predictions

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 23) 
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print(" {} nn score: {} ".format(23,knn.score(x_test,y_test)))


# ## Evaluating the Algorithm
# For evaluating an algorithm, confusion matrix, precision, recall and f1 score are the most commonly used metrics. The confusion_matrix and classification_report methods of the sklearn.metrics can be used to calculate these metrics.

# In[ ]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, prediction))


# The results show that our KNN algorithm was able to classify all the 62 records in the test set with 81% accuracy, which is well enough. 

# ## Comparing Error Rate with the K Value
# * One way to help you find the best value of K is to plot the graph of K value and the corresponding error rate for the dataset.
# * In this section, we will plot the mean error for the predicted values of test set for all the K values between 1 and 40.

# In[ ]:


error=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test)) 
    
    
    


# The above script executes a loop from 1 to 40. In each iteration the mean error for predicted values of test set is calculated and the result is appended to the error list.
# 
# The next step is to plot the error values against K values. Execute the following script to create the plot:

# The output graph looks like this:

# In[ ]:


plt.figure(figsize=(20, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()


# From the output we can see that the mean error is closest to zero when the value of the K is 20 ,21 .
