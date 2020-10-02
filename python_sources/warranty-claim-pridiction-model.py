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


# In[ ]:


import sklearn.model_selection


# In[ ]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import os


# In[ ]:


sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=False, random_state=None)


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


# Loading data set
claims=pd.read_csv("..//input/warranty-claims//train.csv")
claims.head()


# In[ ]:


#### DATA CLEANSING ####

## Region correction according to states
claims.loc[(claims.State == "Delhi") | (claims.State == "Uttar Pradesh") |(claims.State == "UP") |
        (claims.State == "Haryana") | (claims.State == "HP") | (claims.State == "J&K"), "Region"] = "North"

claims.loc[(claims.State == "Andhra Pradesh") | (claims.State == "Karnataka") |
        (claims.State == "Kerala") | (claims.State == "MP") | (claims.State == "Tamilnadu") | 
        (claims.State == "Telengana"), "Region"] = "South"

claims.loc[(claims.State == "Assam") | (claims.State == "Jharkhand") |
        (claims.State == "Tripura") | (claims.State == "West Bengal"), "Region"] = "East"

claims.loc[(claims.State == "Gujarat"), "Region"] = "West"
        
claims.loc[(claims.State == "Bihar") | (claims.State == "UP") | (claims.State == "Uttar Pradesh"), "Region"] = "North East"

claims.loc[(claims.State == "Goa") | (claims.State == "Maharashtra"), "Region"] = "South West"

claims.loc[(claims.State == "Odisha"), "Region"] = "South East"
        
claims.loc[(claims.State == "Rajasthan"), "Region"] = "North West"     


# In[ ]:


## Replacing UP with Uttar Pradesh 
claims.loc[(claims.State == "UP"), "State"] = "Uttar Pradesh"


# In[ ]:


## Replacing claim with Claim
claims.loc[(claims.Purpose == "claim"), "Purpose"] = "Claim"


# In[ ]:


## Separating hyderbad among two states. like Andhra Pradesh = Hyderbad, Telengana = Hyderabad 1
claims.loc[(claims.State == "Telengana"), "City"] = "Hyderabad 1"


# In[ ]:


# Deleting first column
claims.drop(["Unnamed: 0"],inplace=True,axis=1) 


# In[ ]:


#### EXPLORATORY DATA ANALYSIS ####
list(claims.columns)


# In[ ]:


claims.shape[False]


# In[ ]:


claims.duplicated(subset=None, keep='first').sum()


# In[ ]:


claims1=claims.drop_duplicates(keep="first")


# In[ ]:


claims1.shape


# In[ ]:


## filling NA values
claims1.isnull().sum()


# In[ ]:


claims1["Claim_Value"].fillna(7370,inplace=True)  ##median of claim value is 7370 


# In[ ]:


claims1.isnull().sum()


# In[ ]:


## creating dummies for categorical variables
dummies = pd.get_dummies(claims1[['Region','State','Area','City','Consumer_profile','Product_category','Product_type',
                                  'Purchased_from','Purpose']])


# In[ ]:


# Dropping the columns for which we have created dummies
claims1.drop(['Region','State','Area','City','Consumer_profile','Product_category','Product_type',
             'Purchased_from','Purpose'],inplace=True,axis = 1)


# In[ ]:


# adding the columns to the salary data frame 
claims2 = pd.concat([claims1,dummies],axis=1)


# In[ ]:


claims2.head(3)


# In[ ]:


claims2.shape


# In[ ]:


claims2['Fraud'].value_counts()


# In[ ]:


# Separate majority and minority classes
claims2_majority = claims2[claims2.Fraud==0]
claims2_minority = claims2[claims2.Fraud==1]
 


# In[ ]:


from sklearn.utils import resample


# In[ ]:


# Upsample minority class
claims2_minority_upsampled = resample(claims2_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=323,    # to match majority class
                                 random_state=123) # reproducible results
 


# In[ ]:


# Combine majority class with upsampled minority class
claims2_upsampled = pd.concat([claims2_majority, claims2_minority_upsampled])
 


# In[ ]:


# Display new class counts
claims2_upsampled.Fraud.value_counts()


# In[ ]:


X = claims2_upsampled.iloc[:,[0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81]]
y = claims2_upsampled.iloc[:,10]


# In[ ]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test


# In[ ]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()


# In[ ]:


# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)


# In[ ]:


#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[ ]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[ ]:


y_pred1 = clf.predict(X_train)


# In[ ]:


print("Accuracy:",metrics.accuracy_score(y_train,y_pred1))


# In[ ]:


print ('Recall:', metrics.recall_score(y_test, y_pred))


# In[ ]:


print ('Precision:', metrics.precision_score(y_test,y_pred))
                                    


# In[ ]:


print ('\n clasification report:\n', metrics.classification_report(y_test,y_pred))


# In[ ]:


print ('\n confussion matrix:\n',metrics.confusion_matrix(y_test,y_pred))


# In[ ]:




