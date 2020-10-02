#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Importing our libraries that We need
import numpy as np
import pandas as pd
import matplotlib as plt


# In[ ]:


#reading our rain data using the pandas funtion read_csv()
df=pd.read_csv("../input/weatherAUS.csv")
df.head()


# In[ ]:


#Dropping the columns that are not necessary

data=df.drop(['Evaporation','Sunshine'], axis=1)


# In[ ]:


data.head()


# In[ ]:


# Dropping the NaN values from the data as they can be problematic 
# the dropna function of pandas removes the entire row in the Nan is present in any of the column
data.dropna(inplace=True)


# In[ ]:


data.head()


# In[ ]:


#Using labelEncoder to assign numeric values to the string data, axxording to the label.
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Location']=le.fit_transform(data['Location'])
data.head()


# In[ ]:


data['WindGustDir']=le.fit_transform(data['WindGustDir'])
data.head()


# In[ ]:


data['WindDir9am']=le.fit_transform(data['WindDir9am'])
data.head()


# In[ ]:


data['WindDir3pm']=le.fit_transform(data['WindDir3pm'])
data.head()


# In[ ]:


data['RainToday']=le.fit_transform(data['RainToday'])
data.head()


# In[ ]:


data['RainTomorrow']=le.fit_transform(data['RainTomorrow'])
data.head()


# In[ ]:


#The describe function helpls describe our data
data.describe()


# In[ ]:


#The info function gives information about the data
data.info()


# In[ ]:


# Getting our inputs for the classifier storing it in variable X
# the .values function gets the values from the dataframe and converts it into a numpy array

X=data.iloc[:,1:22].values
print(X[0:5,:])


# In[ ]:


#Getting the output for the classifier and storing it into variable y

y=data['RainTomorrow'].values
print(y[0:5])


# In[ ]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#printing the first five instances of the X_train
print(X_train[0:5,:])


# In[ ]:


#Printing the first five instances of the y_train
print(y_train[0:5])


# In[ ]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# In[ ]:


# Fitting SVM classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)


# In[ ]:


#Getting the accuracy score on our testing data 
classifier.score(X_test,y_test)


# In[ ]:


#Using the cross-val_score to divide our data into multiple splits and check for accuracy
#One way to stop overfitting
from sklearn.model_selection import cross_val_score
results=cross_val_score(classifier,X,y,cv=3)


# In[ ]:


#Printing the results
print(results)


# In[ ]:




