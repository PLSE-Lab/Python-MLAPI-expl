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


# In[ ]:


#Importing Seaborn

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


#Reading from the CSV File

df = pd.read_csv("../input/insurance.csv")


# In[ ]:


#Check the header to get a rough idea about the columns and rows

df.head()


# In[ ]:


#Describe the dataframe to know about it's statistics 

df.describe().T


# We can see that the data given to us does not contain any significant outliers

# In[ ]:


#Obtaining the total rows and columns (Entries)

df.shape


# In[ ]:


#We need to get to know about the data types of columns

df.info()


# We can infer that there are three categorical columns 

# In[ ]:


#We are checking for Null values

df.isna().sum()


# There are no null values present in the dataframe

# In[ ]:


#We are looking for duplicated rows 

df.duplicated().sum()


# * There is one duplicated row.
# * We need to remove it from the dataframe

# In[ ]:


#Removing duplicated rows

df = df.drop_duplicates()
df.duplicated().sum()


# In[ ]:


#Checking the dataframe

df


# Now, we need to preprocess the categorical data and convert it to useful numbers for our model 

# In[ ]:


#Checking the 'Sex' column for any ambiguous/incorrect entries

df.sex.unique()


# There are no incorrect/ambiguous entry in this column

# In[ ]:


#Checking the 'Region' column for any ambiguous/incorrect entries

df.region.unique()


# There are no incorrect/ambiguous entry in this column

# In[ ]:


#Checking the 'Smoker' column for any ambiguous/incorrect entries

df.smoker.value_counts()


# There are no incorrect/ambiguous entry in this column

# Since the smoker column has only two distinct categorical entries, we can simply replace them with 0/1

# In[ ]:


#Replacing 'no' with 0 and 'yes' with 1 in the dataframe

df.smoker.replace({"no":0,"yes":1}, inplace=True)


# In[ ]:


df


# We can see that the smoker column has been changed to a numerical one

# In[ ]:


#Checking the Correlation between each columns

df.corr()


# Our target data is expenses and we can see that it has high correlation with the smoker column.

# We will see the whole distribution of data using different plots

# In[ ]:


#We are using Pairplot to get a visual representation of complete data distribution

sns.pairplot(data=df)


# We need to try to draw a linear line between our target data and the columns which have considerable impact on our target data.

# In[ ]:


#We use regplot to draw a linear regression line between two columns, namely, age & expenses here.

sns.regplot(x=df["age"],y=df["expenses"])


# This does not give a clear idea as the data is distributed evenly. So, we need to change the parameters.

# We need to change the plot and try to include one more parameter which might help us understand the data distribution better.

# In[ ]:


#We use scatterplot so that we can include the smoker column, which has high correlation with expenses 
#and see whether it gives any knowledge on the dataframe 

sns.scatterplot(x=df["bmi"],y=df["expenses"],hue=df["smoker"])


# As expected, smoker column gives us a rough idea on what is influencing the target data.

# In[ ]:


#We can alter the parameters to get a better visual representation

sns.scatterplot(x=df["age"],y=df["expenses"],hue=df["smoker"])


# Again, we need to refine the parameters

# In[ ]:


#We will make use of the lmplot to draw two regression lines for the parameters.

sns.lmplot(x="bmi", y="expenses", hue="smoker", data=df)


# In[ ]:


#We can make use of swarmplot to better understand the relationship between expenses and smoker columns

sns.swarmplot(x=df["smoker"],y=df["expenses"])


# We can finally infer that the target data of our dataset is strongly depending on smoker column. Apart from this, the bmi and age column has considerable significance as well.

# Now, we need to convert the rest of the categorical columns to numbers.

# In[ ]:


#Selecting the categorical columns

df_categorical_col = df.select_dtypes(exclude=np.number).columns
df_categorical_col


# In[ ]:


#Selecting the numerical columns

df_numeric_col = df.select_dtypes(include=np.number).columns
df_numeric_col


# We can perform One hot encoding to convert these categorical data to numerical ones as they are nominal data.

# In[ ]:


#Get the truth table of each row for the categorical columns

df_onehot = pd.get_dummies(df[df_categorical_col])


# In[ ]:


#Viewing the obatined truth table

df_onehot


# We have obtained an encoded truth table for all the 1337 entries. Now we need to concatenate it with other numerical columns. 

# In[ ]:


#Concatenation of encoded data and existing numerical columns we obtained earlier.

df_after_encoding = pd.concat([df[df_numeric_col],df_onehot], axis = 1)
df_after_encoding


# The above dataframe does not have any categorical columns and can be further used for developing a model

# We will be implementing the Linear Regression model

# In[ ]:


#Importing necessary libraries for Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[ ]:


#Selecting the 'y' value (Target Data)

y = df_after_encoding["expenses"]


# In[ ]:


#Selecting the 'x' value (Coefficients array)

x = df_after_encoding.drop(columns = "expenses")


# We need to split the data into Train and Test datasets for training the model and then testing it to check the accuracy.

# In[ ]:


#Splitting the dataframe into train & test datasets in 70:30 ratio

train_x, test_x, train_y, test_y = train_test_split(x,y,test_size=0.3,random_state=1)


# In[ ]:


#Selecting the model

model = LinearRegression()


# In[ ]:


#We need to draw a best-fit line for our model

model.fit(train_x,train_y)


# In[ ]:


#Print the obtained 'c' value

print(model.intercept_)


# In[ ]:


#Print the obtained 'x' coefficients value

print(model.coef_)


# We have obtained the model formula using which we can predict the target data. 

# We will check the accuracy of the model by implementing the following tests.

# In[ ]:


#We are Predicting the target data for our dataset

print("Predicting train data")
train_predict = model.predict(train_x)
print("Predicting test data")
test_predict = model.predict(test_x)

#Test using MAE, MSE, RMSE, R^2 Error
print(" ")
print("MAE")
print("Train data: ",mean_absolute_error(train_y,train_predict))
print("Test data: ",mean_absolute_error(test_y,test_predict))
print(" ")
print("MSE")
print("Train data: ",mean_squared_error(train_y,train_predict))
print("Test data: ",mean_squared_error(test_y,test_predict))
print(" ")
print("RMSE")
print("Train data: ",np.sqrt(mean_squared_error(train_y,train_predict)))
print("Test data: ",np.sqrt(mean_squared_error(test_y,test_predict)))
print(" ")
print("R^2")
print("Train data: ",r2_score(train_y,train_predict))
print("Test data: ",r2_score(test_y,test_predict))


# We can see that there is a slight difference in the train and test data for every test. We can say that our model is fairly accurate and can be improved upon performing further changes in the preprocessing of data.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




