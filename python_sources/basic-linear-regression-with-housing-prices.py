#!/usr/bin/env python
# coding: utf-8

# ### In statistical modeling, regression analysis is a set of  processes for estimating the relationships among variables.

# So here we are going to learn regression with actual examples by writing some ML code!

# We are going to use a library known as scikit-learn. It is a high level library where many popular machine learning algorithms are available as off the shelf functions. This makes it more easier to use than standard frameworks like TensorFlow or Keras. 
# 
# For example if you want to use a Linear Regression model with scikit learn, all you have to do is to call LinearRegression( ). Then you can train the model by just calling the model.fit( ) function. 
# 
# All of this might not make sense right now, but I will walk you through the whole process with an example in this notebook.

# ### Suppose we want to create a machine learning model to predict the prices of houses in say California.
# 
# As the california housing prices dataset are already availlable in colab as default, we can simply use them

# First we import the basic matheatical and statistical packages

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib as plt

from sklearn.model_selection import train_test_split


# Now we import the datasets which are given in the sample_data folder. (Check the left menu for folders)

# In[ ]:



housing = pd.read_csv('../input/housing.csv')

train,test = train_test_split(housing,test_size=0.33,random_state=42)



# Now we have two dataset, train and test. We always need to split our data into train and test. We will train the model with the train dataset and then test its accuracy with the test dataset.

# In[ ]:


train.head()


# Here you can see the first 5 samples of the data. We need to predict the 'mean house price' with the help of all other attributes. So here, our x will be all the attributes and y will be the price. So we are now gonna split the train dataset into x_train and y_train

# In[ ]:


train = train.drop('ocean_proximity',axis=1)
test = test.drop('ocean_proximity',axis=1)
train = train.drop('total_bedrooms',axis=1)
test = test.drop('total_bedrooms',axis=1)

train.head()


# In[ ]:


x_train = train.drop('median_house_value',axis=1)


# In[ ]:


y_train = train.median_house_value


# In[ ]:


x_train.head()


# In[ ]:





# In[ ]:


y_train.head()


# As you might have noticed, the .head( ) function displays the first 5 elements in a DataFrame. A DataFrame is a table in pandas. Here as we read train and test with **pd.read_csv( )** function, they are pandas DataFrames. Its much similar to saying that they are Excel spreadsheets.

# ### Now we can go on to train our model with x_train and y_train
# 
# 
# 

# In[ ]:


# first import the function from scikit-learn
from sklearn.linear_model import LinearRegression


# In[ ]:


# create a new object of Linear Regression class
model = LinearRegression()


# In[ ]:


# fitting the model = finding the perfect line with minimum error
model.fit(x_train,y_train)


# In[ ]:


model.score(x_train,y_train)


# And with that, we trained out first machine learning model!

# But it has only an accuracy of 64% But what does this really mean?
# If we inspect the test dataset, we can see that it includes the median_house_value that we need to predict. So we can split the dataset into only its attributes, put it into our model to predict the values and then compare the original median house price with what our model predicted. This will be 64% accurate.

# In[ ]:


x_test = test.drop('median_house_value',axis=1)


# In[ ]:


model.predict(x_test)


# These are the house prices that our model predicted. Now lets look at the actual prices which were given in the dataset.

# In[ ]:


test.median_house_value


# The above array contains the actual values of the prices given in the dataset.

# ### But how can we increase the accuracy of the model? There are several methods, known as Hyperparameter Tuning, which can go deep into later on. For now, we can try using another technique than just simple Linear Regression. Let's try using an algorithm called 'Random Forest Regression'

# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


my_new_model = RandomForestRegressor()


# In[ ]:


my_new_model.fit(x_train,y_train)


# In[ ]:


my_new_model.score(x_train,y_train)


# And that's 96% accuracy of the model's values with the actual values. This means that the model passes through 96% of the actual data points!!

# In[ ]:


output = my_new_model.predict(x_test)


# In[ ]:


my_new_model.score(x_test,test.median_house_value)


# In[ ]:


output_csv = pd.DataFrame({'Label':output})

output_csv.to_csv('output.csv',index=False)

