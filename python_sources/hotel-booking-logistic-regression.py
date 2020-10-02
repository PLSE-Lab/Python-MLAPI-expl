#!/usr/bin/env python
# coding: utf-8

# In[ ]:


***Hotel Booking Prediction - with Data Analysis and Logistic Regression. ***
***To check if a booking is canceled or not ***

Importing the Libraries:
To access the data, which is available in CSV, and further manipulate it, we'll use **pandas**. To do operations on the data, we'll use **numpy**.


# In[ ]:


import numpy as np
import pandas as pd


# Importing the dataset.

# In[ ]:


data = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')


# **Analysing the Data**
# 
# Let's first have a look at the dataset and let's try to get an essence of the information it contains.

# In[ ]:


data.head(10)


# In[ ]:


data.shape


# We can see that there are 32 features (columns) and 119390 records (rows) in our dataset.
# 
# Our main objective with this data is to predict if the booking would be made by a customer, provided if they make a reservation within the constraints of out data.
# 
# Since, we have defined our objective, let's see which all features (columns) won't be any use to us for finding the objective.
# 
# Upon inspecting, we can see that the following features won't be useful for our objective:
# 1. hotel - It doesn't matter which type of hotel they make a reservation, the main objective is to see if they make ANY type of reservation at all or not
# 2. agent - The agent that got the reservation for us won't matter
# 3. company - Same logic goes for company as for the agent
# 4. reservation_status_date - We have other features (like: arrival_date_week_number, arrival_date_day_of_month etc) that gives us the same information
# 
# Hence all these 4 columns need to be dropped from the data.

# In[ ]:


data.drop(inplace=True, axis=1, labels=['agent', 'company','hotel','reservation_status_date'])


# Note:
# * inplace = True - The changes will be reflected in the original dataframe
# * axis = 1 - inferring that the columns are to be dropped
# * labels = [...] - The names of the columns that need to be dropped
# 
# *P.S. Dropping of these columns is just based on my intution and hence you can probably use all of these columns and decide to drop some other, or maybe none. Therefore, it's recommended to play with the data and have an iterative approach to solving the problem*

# It will be interesting to have a look at all the unique values that every column contains

# In[ ]:


cols = data.columns
for i in cols:
    print('\n',i,'\n',data[i].unique(),'\n','-'*80)


# Let's check for any null values, if there are any, in the remaining dataset.

# In[ ]:


data.isnull().sum()


# As it can bee seen, only 'country' column has null values. We can deal with this by choosing one of the following methods:
# 1. Replacing the null values with the most frequent value in the column (In this case, it would be the most frequent country).
# 2. Deeting the records (rows) which contains the null values
# 3. Developing a model to predict the null values from existing data.
# 
# All the above mentioned solutions are good solutions for a dataset of this many records. I decided to choose the 1st solution as it is the most easily implemented solution.

# In[ ]:


data.fillna(data.mode().iloc[0], inplace=True)


# Note: 'mode()', will replace the 'NaN's with most frequent value in the column.
# 
# Let's check again for the null values and have a look at how our data looks now

# In[ ]:


data.isnull().sum()


# In[ ]:


data.head()


# As we can see, there are only 28 features left, after we removed 4 columns from our data and there are no null values left in our data.
# 
# Let's now seperate the dependant and independant variables from each other. The independant variable, which we eventually need to predict, would be the 'is_cancelled' column as it tells us if the that particular reservation was cancelled or not. If the reservations was cancelled, the 'is_cancelled' column would hold the value '1' for that particular record, otherwise it would hold the value '0'.

# In[ ]:


X = data.iloc[:,1:]
y = data.iloc[:,0]


# Now, we can see that our data doesn't only have numerical values but it also has strings as values. Machine Learning models, since they work with distances such as Euclidean, Manhattan, Minkowski etc, which all require nuumeric values to be accessed, required all the values to be numerics. Hence, we convert all the categorical variables (columns with string values) to numeric representations. And to do that, we will be using One Hot Encoder.

# In[ ]:


# Importing relevant libraries
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


# In[ ]:


#Implementing Column Transformer
ct = make_column_transformer(
    (OneHotEncoder(),['meal','distribution_channel','reservation_status','country','arrival_date_month','market_segment','deposit_type','customer_type', 'reserved_room_type','assigned_room_type' ]), remainder = 'passthrough'
    )


# Here, the Column Transformer is given the One Hot Encoder and the list of all categorical columns. Now, we simply need to apply fit and transform to our independant variables.

# In[ ]:


X = ct.fit_transform(X).toarray()


# Please note that 'X' is no longer a dataframe, it has been changed to numpy array and the number of columns has also been increased from 28 to 256. This is because the One Hot Encoder has converted each unique value of every categorical variable to its dedicated column.

# In[ ]:


X


# In[ ]:


y


# Perfect.
# 
# Now, we need to split our data into training and test sets.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Note: We are spliting the training and test set with 20% records in the test set and remaining 80% in the training. You can play with this number if you think it will have serious impact on the prediction rate.
# 
# Another important note to make here is that we just saw the number of features exploding from just 28 to 256. That's a huge number. Generally, more number of features in any dataset leads to the [curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality). It simply means that our model will have too many unncessary information to process, which will eventually hamper its processing time and efficiency.
# 
# To avoid the curse of dimensionality, we use something known as [Dimensionality Reduction](https://en.wikipedia.org/wiki/Dimensionality_reduction) algorithms. One of the most used one is known as [PCA - Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis). We are going to use the same. However, one small requirement of PCA is that the data it is applied on should have a sandar scale. Which can be achieved by sklearn's [Standard Scalar](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) function as follows

# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("X_train ---------->\n", X_train, "\nX_test -------->\n", X_test)


# As you can see, now all the values are in a standardised scale.
# 
# Now, we can safely implement PCA.

# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 100)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_


# Please note that upon running the PCA for the first time, set 'n_components' to 'None' and then evaluate the 'explained_variance' variable for choosing the optimal number of n_components. In this case, 100 should be fine.
# 
# Now, we are finally done with everything else except fitting the Logistic Regression model on our data. Let's do that now.
# 
# **Logistic Regression**

# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, max_iter=1000)
classifier.fit(X_train, y_train)


# Now, let's see how our model performs on the test data

# In[ ]:


y_pred = classifier.predict(X_test)


# To calculate the accuracy of our model, the simplest way is to construct a confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# Accuracy can be calculated as:
# 
# 14908 + 8922 (Total number of correct predictions) / 14908 + 8922 + 26 + 22 (Total number of predictions) 
# 
# = 23830 / 23878 * 100 
# 
# = 99.79%
# 
# That's a GREAT accuracy rate.
# 
# 
# **Hope this was useful. Please leave suggestions, mistakes or any other tips in the comments. 
# **
# 
