#!/usr/bin/env python
# coding: utf-8

# # Importing the libraries

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 


# # Importing the dataset

# In[ ]:


dataset = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
dataset.head()


# # Checking Null values in dataset
# **First of all we will check is there any null or nan value in our dataset.For This I will use two methods.**
# 
# 1. By using inbuilt method of our data ,i.e., isnull() method
# 2. By using heatmap function of seaborn library

# In[ ]:


dataset.isnull().sum()
sns.heatmap(dataset.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# **As we can see here that there is no null value. But if we see our data there is one problem some the columns like Glucose, BloodPressure, SkinThickness, Insulin and BMI.So actually these column have nan values which are represented by 0.So we will treat it after making our independent and dependent variables.**

# In[ ]:


dataset.info()


# **As there is no categorical data so we need not to take care of categorical or string data**

# # Making our dependent and independent features
# Now making our dependent and independent features to test our model and predict for future values.

# In[ ]:


X = dataset.iloc[:, 0:8].values
y = dataset.iloc[:, 8].values


# # Taking care of missing data

# In[ ]:


imputer = SimpleImputer(missing_values=0, strategy='mean')
imputer.fit(X[:, 1:6])
X[:, 1:6] = imputer.transform(X[:, 1:6])


# # Splitting the dataset into the Training set and Test set
# **Here I had given a 20% of my whole dataset to the test data as we want to feed the maximum of our data to our training set so that our model can predict with a very high accuracy**

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# # Applying feature scaling on test and train dataset

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # Training the train dataset 

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy')
classifier.fit(X_train, y_train)


# # Predicting the Test set results

# In[ ]:


y_pred_test = classifier.predict(X_test)
print('Confusion Matrix :')
print(confusion_matrix(y_test, y_pred_test)) 
print('Accuracy Score :',accuracy_score(y_test, y_pred_test))
print('Report : ')
print(classification_report(y_test, y_pred_test))


# **And here we have achieved an accuracy of around 78% on our test dataset by using random forest classifier.**
