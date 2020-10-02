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


# # About Turi Create

# Turi Create simplifies the development of custom machine learning models. You don't have to be a machine learning expert to add recommendations, object detection, image classification, image similarity or activity classification to your app.
# 
# * **Easy-to-use**: Focus on tasks instead of algorithms
# * **Visual**: Built-in, streaming visualizations to explore your data
# * **Flexible**: Supports text, images, audio, video and sensor data
# * **Fast and Scalable**: Work with large datasets on a single machine
# * **Ready To Deploy**: Export models to Core ML for use in iOS, macOS, watchOS, and tvOS apps
# 

# ## Install TuriCreate Library 

# In[ ]:


get_ipython().system('pip install turicreate')
import turicreate as tc


# # Load house sales data
# 
# I am working on the dataset from the King County region in the Seattle city of the United States of America. This is a public record data. 

# In[ ]:


sales = tc.SFrame.read_csv('/kaggle/input/home-prices-dataset/home_data.csv')


# In[ ]:


# Have a first look at our data 
sales


# # Exploring the data

# In[ ]:


# Used for quick visualisation and data exploration
sales.show()


# Inferences Drawn 

# In[ ]:


# Plot a scatter plot to see the relationship between plot size of living space and the price
tc.visualization.set_target('auto') # to display the graph in the desired location 
tc.visualization.scatter(x=sales["sqft_living"], y=sales["price"], xlabel="Living Area", ylabel="Price", title="Scatter Plot")


# In[ ]:


tc.show(sales[1:5000]['sqft_living'],sales[1:5000]['price'])


# ### More visualizations 
# 
# There are other visualizations that can be explored in this library which is incredibly scalable 
# 

# # Simple regression model that predicts price from square feet

# In[ ]:


# Splitting the data into training and testing data
training_set, test_set = sales.random_split(.8,seed=0)


# ## Build and Train simple regression model

# In[ ]:


# Building our linear regression model
sqft_model = turicreate.linear_regression.create(training_set,target='price',features=['sqft_living'])


# # Evaluate the quality of our model

# In[ ]:


# Printing the mean value of the test data
print(test_set['price'].mean())


# In[ ]:


# Evaluate the model prediction against the actual data for predictions
print(sqft_model.evaluate(test_set))


# Inference : The RMSE is really high and there is one value which is a huge outlier that should be dealt with. The model is very simple though and more features have to be added. 

# # Explore model a little further
# 
# By plotting our regression line which we fit to our data. 

# In[ ]:


# Let's look at model weights ( slope and intercept ) that we fit
sqft_model.coefficients


# In[ ]:


# Visualizing the regression line that was fit in this case 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(test_set['sqft_living'],test_set['price'],'.',
        test_set['sqft_living'],sqft_model.predict(test_set),'-')


# # Explore other features of the data
# 
# Model with just one feature wasn't the best so it is important to explore other features in the data

# In[ ]:


my_features = ['bedrooms','bathrooms','sqft_living','sqft_basement','floors']


# In[ ]:


sales[my_features].show()


# # Build a model with these additional features

# In[ ]:


my_features_model = turicreate.linear_regression.create(training_set,target='price',features=my_features)


# # Compare simple model with more complex one

# In[ ]:


print (my_features)


# In[ ]:


print (sqft_model.evaluate(test_set))
print (my_features_model.evaluate(test_set))


# In[ ]:


my_features_model.coefficients


# # Apply learned models to make predictions
# 
# Working on one example

# In[ ]:


# Extracting a particular feature
house1 = sales[sales['id']== 5309101200]


# In[ ]:


house1


# <img src="http://blue.kingcounty.com/Assessor/eRealProperty/MediaHandler.aspx?Media=2916871">

# In[ ]:


print (house1['price'])


# In[ ]:


print (sqft_model.predict(house1))


# In[ ]:


print (my_features_model.predict(house1))


# ## Prediction for a second house, a fancier one

# In[ ]:


house2 = sales[sales['id']==1925069082]


# In[ ]:


house2


# <img src="https://ssl.cdn-redfin.com/photo/1/bigphoto/302/734302_0.jpg">

# In[ ]:


print(house2['price'])


# In[ ]:


print (sqft_model.predict(house2))


# In[ ]:


print (my_features_model.predict(house2))


# ## Prediction for a super fancy home
# 
# Let's assume what Bil Gates house would be worth.

# In[ ]:


bill_gates = {'bedrooms':[8], 
              'bathrooms':[25], 
              'sqft_living':[50000], 
              'sqft_lot':[225000],
              'floors':[4], 
              'zipcode':['98039'], 
              'condition':[10], 
              'grade':[10],
              'waterfront':[1],
              'view':[4],
              'sqft_above':[37500],
              'sqft_basement':[12500],
              'yr_built':[1994],
              'yr_renovated':[2010],
              'lat':[47.627606],
              'long':[-122.242054],
              'sqft_living15':[5000],
              'sqft_lot15':[40000]}


# <img src="https://upload.wikimedia.org/wikipedia/commons/2/26/Residence_of_Bill_Gates.jpg">

# In[ ]:


print (my_features_model.predict(turicreate.SFrame(bill_gates)))


# In[ ]:




