#!/usr/bin/env python
# coding: utf-8

# ## Boston Housing Exercise

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets
seed = 0
np.random.seed(seed)


# In[ ]:


from sklearn.datasets import load_boston


# In[ ]:


# Load the Boston Housing dataset from sklearn
boston = load_boston()
bos = pd.DataFrame(boston.data)
# give our dataframe the appropriate feature names
bos.columns = boston.feature_names
# Add the target variable to the dataframe
bos['Price'] = boston.target


# ### Our goal will be to predict the price of housing based on the feaures in this data set

# In[ ]:


# For student reference, the descriptions of the features in the Boston housing data set
# are listed below
boston.DESCR


# In[ ]:


bos.head()


# In[ ]:


# Select target (y) and features (X)
X = bos.iloc[:,:-1]
y = bos.iloc[:,-1]


# In[ ]:


# Split the data into a train test split
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=seed, shuffle=True)


# ### Exercise:  Use sklearn to fit a multiple linear regression model.  How will you decide which features to include?

# In[ ]:





# In[ ]:





# In[ ]:





# ### What is the coefficient of determination (r-squared) for your model?  What about the mean squared error?

# In[ ]:





# In[ ]:





# ### Can you improve upon your origninal model? 

# #### Hint 1:  Look at the correlations of your features to your target - are there features you think are more important than others?  This is exploratory - just play with buildind different models
# 
# #### Hint 2:  Are there features you can engineer (categorical features based on binning the numeric features in the dataset) that may be useful?  How do you handle categorical features in MLR?

# In[ ]:





# In[ ]:





# ### Make a scatterplot of the observations in the test data, where the x-axis is the actual price and the y axis is the predicted price from your favorite model.  What does this plot tell you about your regression model?

# In[ ]:




