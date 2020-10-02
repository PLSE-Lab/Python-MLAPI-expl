#!/usr/bin/env python
# coding: utf-8

# # Predicting House Prices using Multivariate Linear Regression

# At the time of writing this I am working my way through Stanfords's 11-week Machine Learning course offered on Coursera (link below). The course is taught using matlab, but most anaylsis in industry is done using either Python or R. So, I've decided to give linear regression a shot using some of pythons many data-science libraries.

# https://www.coursera.org/learn/machine-learning

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split


# sklearn provides some datasets as well as functions to import them in order to test out their machine-learning tools, so I'll be using one here (for convenience)

# In[ ]:


from sklearn.datasets import load_boston
boston = load_boston()


# the "boston" variable contains a dictionary provided by sklearn that contains all the data required to try out linear regression in one place, as well as a description of the dataset itself. Below, you'll see that this dataset contains housing prices in Boston as well as other useful features to aid in training an algorithm.

# In[ ]:


print(boston.get('DESCR'))


# Now, I'll grab the features i'd like to use to train my algorithm and import them into a dataframe. I'll also do the same for my training set (house prices).

# In[ ]:


df_x = pd.DataFrame(boston.get('data'),columns=boston.get('feature_names'))
df_y = pd.DataFrame(boston.get('target'))
df_y = np.array(df_y)
df_x.head()


# Here, I split my data into a training set and a testing set. This way, I can test my algorithm on data points that it has not seen before.

# In[ ]:


X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(df_x, df_y, test_size = 0.33, random_state = 5)


# Here, I create my model and then plot my data to better understand how my model is performing.

# In[ ]:


reg = linear_model.LinearRegression()
reg.fit(X_train,Y_train)
Y_predict = reg.predict(X_test)


# In the classic problem of predicting house prices based on square footage, linear regression is visualized by fitting a straight line through a plot where the x-axis represents square footage and the y axis represents house price. In this example, I use 12 features to predict house prices so a different visualization is required.

# Below is a "Prices vs. Predicted Prices" plot. If my model predicted price perfectly for all data points, we would see a straight line. However, my model is not perfect.

# In[ ]:


plt.scatter(Y_test, Y_predict, color='blue')
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()


# The squared mean error equation is one way of quantifying how accurate a linear regression model is. Put simply, it measures the average difference between the predicted value and the value in the training set. The higher the output, the less accurate the model is.

# In[ ]:


error = np.mean((Y_predict - Y_test)**2)
print('Mean Squared Error: ' + str(error))


# To verify I have calculated the error correctly, I use sklearn's mean_squared_error function to see if I get the same result.

# In[ ]:


error = mean_squared_error(Y_test, Y_predict)
print('Mean Squared Error: ' + str(error))


# I've noticed that my model is especially inaccurate when it comes to predicting the price of the most expensive houses in the dataset. As an experiment I'll try to feed the model the entire dataset to see if this trend lessens, or if more data will at least lower my mean squared error.

# In[ ]:


reg.fit(df_x,df_y)
Y_predict2 = reg.predict(df_x)
plt.scatter(df_y, Y_predict2, color='green')
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices (More data)")
plt.show()
error = np.mean((Y_predict2 - df_y)**2)
print('Mean Squared Error when model is trained using entire dataset: ' + str(error))


# Judging by the graph above, adding more data has not made my model any better at predicting the prices of the most expensive houses in the data set. However, it has made it a little more accurate (judging by my reduced mean squared error)

# Update: 
# Above, I have made a beginners mistake! While training the algorithm on the entire dataset (rather then just the test set) will yield a lower error, it does not make for a better algorithm. The point of checking the error on a test set is to see how well the algorithm can predict values it has not seen before. By testing my algorithm on the entire dataset (training + test set), I defeated the purpose of splitting the dataset in the first place. You live and you learn I guess. Let me know if you notice any other mistakes, I love feedback!
