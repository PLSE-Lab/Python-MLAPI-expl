#!/usr/bin/env python
# coding: utf-8

# # Linear Regressional Prediction on Health Insurance

# First we import all the necessary libraries and functions.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error


# Here we load the data from the Google Data Library.

# In[ ]:


data_set = pd.read_csv('../input/insurance/insurance.csv')
print(data_set.head(10))
print(data_set.shape)


# Here, I try to see the total number of people in each region according to the sex.

# In[ ]:


sns.countplot(x = 'region', hue = 'sex' , data = data_set)


# Here, I try to see the total number of smokers in each category of 'sex'.

# In[ ]:


sns.countplot(x = 'smoker', hue = 'sex' , data = data_set)


# Now, to see the BMI graph.

# In[ ]:


data_set['bmi'].plot.hist(bins=50,figsize=(15,5))


# As we need numberic values instead of 'str', therefore we replace all the 'str' value columns to binary inputs '1' or '0'.

# In[ ]:


data_set.sex.replace(['male', 'female'], [1, 0], inplace=True) # if male , replace value to 1 else to 0.
data_set.smoker.replace(['yes', 'no'], [1,0], inplace=True) # if smoker, replace value to 1 else to 0.
data_set.drop(['region'], axis=1, inplace=True)


# We now need to check if there are any 'NaN' values in our data.

# In[ ]:


data_set.isnull().sum()


# Now we need to make the data into our standard Linear Regression data set, i.e. splitthe data into 'input' and 'output'.

# In[ ]:


X = data_set.drop('charges', axis=1) # input column
y = data_set['charges'] # output column
print(X.head(5))
print()
print(y.head(5))


# Now we split the data into traingset and testing set.

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)


# Here we introduce our Linear Regression Model.

# In[ ]:


linear_model = LinearRegression()


# Now we fit our training data into the model and predict the output using the testing data.

# In[ ]:


linear_model.fit(X_train,y_train)
predictions = linear_model.predict(X_test)


# In[ ]:


print(mean_squared_error(y_test, predictions))


# In[ ]:


linear_model.score(X_test, predictions)


# If the user want to enter any custom input and check for the most probable output, the next column can be executed

# In[ ]:


val = input('Do you want to predict a value : ( "Y" for yes "N" for no ) : ')
temp_val = []
if val == 'Y' or 'y':
    age = int(input('Enter the age of candidate : '))
    temp_val.append(age)
    sex = int(input('Enter the sex of the candidates : '))
    temp_val.append(sex)
    bmi = float(input('Enter the BMI of the candidate : '))
    temp_val.append(bmi)
    children = int(input('Enter the value of the children : '))
    temp_val.append(children)
    smoker = int(input('Enter the value for smoking : '))
    temp_val.append(smoker)
else :
    print('Have fun !!')
new_val = [temp_val]
predict = linear_model.predict(new_val)
print()
print("Input Data : %s, Predicted Amount : $ %s " % (new_val[0], predict[0]))
#print(data_set.head(10))

