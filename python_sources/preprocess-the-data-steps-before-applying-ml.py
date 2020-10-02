#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#OS is use for access a data from your storage and set a working path
import os
print(os.listdir("../input"))


# # ** Important Libraries for Data preprocessing **

# In[ ]:


import pandas as pd #For data handling and manipulating
from sklearn.preprocessing import Imputer #For fill a missing values
from sklearn.preprocessing import LabelEncoder #Encode a categorcal values
from sklearn.preprocessing import OneHotEncoder #Dummy variables
from sklearn.model_selection import train_test_split


# # ** Now we need to take a data in our program , so we will use Panda**

# In[ ]:


dataset = pd.read_csv('../input/Data.csv') 


# # **Let's see the data first **

# In[ ]:


dataset.head() #It displays first five rows


# # ** Now we taken a data in our program **
# 
# ** Machine learning Requirements **
# 
# ** (1) No missing values ** 
# 
# ** (2) No characters values ** 
# 
# ** (3) Correlations between variables ** 
# 
# ** (4) Scaling between variable's data **

# # Now first we will find & handle missing values 

# In[ ]:


dataset.isnull().sum()


# # We see there is missing values in Age and Salary column 

# # Now we split the data into independent and dependent variables (x and y)

# In[ ]:


x = dataset.iloc[:, :-1].values
y = dataset.iloc[:,3:].values


# # Now we use a Imputer for handle missing values 

# In[ ]:


#[:, <- it is for select all rows of data ] 
#[:,1:3] <- it is basic slicing concept for fille the missing values in column 1 and 2

#Strategy means we will fill missing values with mean
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
x[:,1:3]=imp.fit_transform(x[:,1:3])


# In[ ]:


x


# # We habdle the missng values now you can see there is categorical values in our column 0 ,so we need to encode this in numbers, we will use Labelencoder for this task

# # Let's see how it works
# ** There are three countries in column 0 France ,Spain ,Germany **
# 
# ** So it will do encoding like this ** 
# 
# ** France - 0 ** 
# 
# ** Germany - 1 ** 
# 
# ** Spain - 2 ** 

# In[ ]:


LE=LabelEncoder()
x[:,0]=LE.fit_transform(x[:,0])


# In[ ]:


x


# # Now we need a OneHotEncoder
# # (1)When we will use OneHotEncoder?
# 
# # - > When our catogorical values are more than 2 (means not only 'Yes and NO' or 'Win or Lose' .
# 
# # (2) Why we are using OneHotEncoder?
# 
# # -> Imagine if you have a 3 categorical values like in our dataset so we take France as 1 , Germany as 2 and Spain as 3 , so the problem is when model calculate the average ,it will something like 1+3 = 4/2 = 2, means accoring to your model average of France and Germany together is Spain
# 

# In[ ]:


OE=OneHotEncoder(categorical_features=[0])
x=OE.fit_transform(x).toarray()


# In[ ]:


x


# # Now we encode Y (Yes and No values into 1 and 0)

# In[ ]:


y=LE.fit_transform(y)


# In[ ]:


y


# # Now we split the data into training and testing

# In[ ]:


##Train test split data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# # Now we need to Scale the data we use StandardScaler
# 
# # When we use StandardScaler?
# 
# # -> (Example) When our 1 column data range is between (50- 100) and 2nd column range is (1000+), so the data is not in scale, so we need a StandardScaler.
# 
# # What is StandardScaler?
# 
# # -> The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1. Given the distribution of the data, each value in the dataset will have the sample mean value subtracted, and then divided by the standard deviation of the whole dataset.

# In[ ]:


###Stndard scalar for scaling data into one scale
from sklearn.preprocessing import StandardScaler
SC=StandardScaler()
x_train=SC.fit_transform(x_train)
x_test=SC.transform(x_test)


# # So finally we preprocess the data now this data is perfect for implement Machine Learning 

# # Upvote my kernel if you Like this , and comment down your questions and opinion
