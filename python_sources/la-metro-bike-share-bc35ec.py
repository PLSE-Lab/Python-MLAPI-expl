#!/usr/bin/env python
# coding: utf-8

# ### Modification of Kernel by another Kaggle User ### 
# I looked through some datasets and kernels and found this one interesting. For some predictions near the end, the score achieved was always 100%, which seemed unusually high. I noticed that the data used for the predictions was 100% correlated due to some generated features being mutually exclusive. Here is a different take on the predictions, where the correlations are removed.
# 

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


# # IMPORTING LIBRARIES

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# # LOADING LA METRO-BIKE-SHARE-TRIP-DATASET

# In[ ]:


train = pd.read_csv("../input/metro-bike-share-trip-data.csv")


# # DATA EXPLORATION

# In[ ]:


print ('There are',len(train.columns),'columns:')
for x in train.columns:
    print(x+' ',end=',')


# In[ ]:


train.head()


# In[ ]:


train.tail()


# In[ ]:


print("Number of columns (features) in the given dataset is :",train.shape[1])
print("Number of rows (entries) in the given dataset is :",train.shape[0])


# In[ ]:


train.info()


# In[ ]:


train_na = (train.isnull().sum()*100)/len(train)
print("Percentage of Missing Data in each feature:")
train_na.sort_values(ascending=False)


# In[ ]:


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# # DROPPING NULL VALUES.
# Since the dataset features such as Starting Station ID,Starting Station Latitude,Starting Station Longitude,Ending Station ID,Ending Station Latitude ,Ending Station Longitude,Bike ID ,Plan Duration,Starting Lat-Long,Ending Lat-Long contains some null values. So before moving on further with the dataset we need to drop null values so that these null values can create problems while moving forward.

# In[ ]:


train = train.dropna()


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(train['Passholder Type'], train['Duration'])
plt.ylabel('Duration (in seconds)', fontsize=13)
plt.xlabel('PassHolder Type', fontsize=13)
plt.show()


# # DISTANCE CALCULATION
# Using the Starting Station Latitude and Longitude and Ending Station Latitude and Longitude we can calculate the distance travelled by the user. This extra feature calculated can be used to fit into the model as it can help in improving the predicting ability of the model.

# In[ ]:


l = []
import math 
degrees_to_radians = math.pi/180.0
def distance_on_unit_sphere(lat1, long1, lat2, long2):
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians
    
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    
    a = ((math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2)) +(math.cos(phi1)*math.cos(phi2)))
    if a>1:
        a=0.999999
    dis = math.acos( a )
    return dis*6373
for i in range(97825):
    l.append(distance_on_unit_sphere(train['Starting Station Latitude'].iloc[i],
                                     train['Starting Station Longitude'].iloc[i],
                                     train['Ending Station Latitude'].iloc[i],
                                     train['Ending Station Longitude'].iloc[i]))


# Adding all the important features to a temporary dataframe temp 

# In[ ]:


temp = pd.DataFrame(data=[train['Duration'],
                               train['Starting Station Latitude'],
                               train['Starting Station Longitude'],
                               train['Ending Station Latitude'],
                               train['Ending Station Longitude'],
                               train['Plan Duration']],
                               index=['Duration',
                                      'Starting Station Latitude',
                                      'Starting Station Longitude',
                                      'Ending Station Latitude',
                                      'Ending Station Longitude',
                                      'Plan Duration'])


# Creating a new dataframe distance having the newly calculated distances under the Distance column

# In[ ]:


distance = pd.DataFrame({'Distance':l})


# In[ ]:


new_train = temp.T


# In[ ]:


print("Shape of new train ",new_train.shape)
print ("Shape of distance ",distance.shape)


# In[ ]:


new_train = new_train.reset_index(drop=True)


# In[ ]:


new_train = pd.concat([distance,
                       new_train,
                       pd.get_dummies(data=train['Passholder Type']).reset_index(),
                       pd.get_dummies(data=train['Trip Route Category'],drop_first=True).reset_index()],
                       axis=1)


# In[ ]:


new_train = new_train.drop('index',axis=1)
new_train.info()


# In[ ]:


print("There are 3 different types of Passholder : ")
train['Passholder Type'].value_counts()


# # From below here, I modified the training data to remove obious correlations between Flex Pass, Monthly Pass, Walk-up and Plan Duration
# 
# ## USING LOGISTIC REGRESSION TO PREDICT WHETHER PASSHOLDER TYPE IS "Walk-up" OR "Not"

# In[ ]:


X1 = new_train.drop(columns=['Flex Pass','Monthly Pass','Walk-up','Plan Duration'])
y1 = new_train['Walk-up']


# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X1,y1,test_size=0.33)


# In[ ]:


lr = LogisticRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


pred1 = lr.predict(X_test)


# In[ ]:


print(classification_report(y_test,pred1))


# In[ ]:


print(confusion_matrix(y_test,pred1))


# # USING RANDOM FOREST CLASSIFIER TO PREDICT WHETHER THE PASSHOLDER TYPE IS "Monthly Pass" OR "Not"

# In[ ]:


X2 = new_train.drop(columns=['Flex Pass','Monthly Pass','Walk-up','Plan Duration'])
y2 = new_train['Monthly Pass']
X_train,X_test,y_train,y_test = train_test_split(X2,y2,test_size=0.33)


# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf = RandomForestClassifier()


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


pred2 = clf.predict(X_test)
# pred2 = clf2.predict(X_test)


# In[ ]:


print(classification_report(y_test,pred2))


# In[ ]:


print(confusion_matrix(y_test,pred2))


# # USING DECISION TREE CLASSIFIER TO PREDICT WHETHER THE PASSHOLDER TYPE IS "Flex Pass" OR "Not"

# In[ ]:


X3 = new_train.drop(columns=['Flex Pass','Monthly Pass','Walk-up','Plan Duration'])
y3 = new_train['Flex Pass']
X_train,X_test,y_train,y_test = train_test_split(X3,y3,test_size=0.33)


# In[ ]:


print(X3.head())


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


clf2 = DecisionTreeClassifier()


# In[ ]:


clf2.fit(X_train,y_train)


# In[ ]:


pred3 = clf2.predict(X_test)


# In[ ]:


print(classification_report(y_test,pred3))


# In[ ]:


print(confusion_matrix(y_test,pred3))


# ### Conclusion
# We find that the decision tree classifier is fairly accurate for deciding whether a user holds a flex pass or not. Now, it would be interesting to examine which features of the data set have the strongest effect on the classification.

# In[ ]:





# In[ ]:




