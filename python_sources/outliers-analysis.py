#!/usr/bin/env python
# coding: utf-8

# # Outliers Analysis   
# Outliers could introduce biased analysis, and also our final models. So, it is important to solve this issue using appropriate techniques. In fact, these outliers often carry some interesting insights, so removing them might not the right solution.  
# 
# Here we are going to look at the following techniques to solve the  problem for outliers:   
# -  Log transformation  
# - Binning    
# For demonstration purpose, we are going to do the whole preprocessing for only one feature (**fare**)

# In[ ]:


# Useful libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# In[ ]:


titanic = pd.read_csv('../input/titanic-data/titanic3.csv')


# In[ ]:


titanic.head()


# ### Use histogram to understand the distributions  
# 
# ##### Age feature

# In[ ]:


titanic.age.plot(kind='hist', title='Histogram for Age', 
                 bins=30, color='c')


# We can see that most of the **age** values are in the [20, 30s] and a very few are in [70, 80s]. 
# Let's have have a look a those passengers where their age is more than 70. 

# In[ ]:


titanic.loc[titanic.age > 70]


# We have 6 passengers in the results, with 2 survivors (survived=1) and 4 who did not survive (survived=0)

# ##### Fare 

# In[ ]:


titanic.fare.plot(kind='hist', title='Histogram for Fare', 
                 bins=30, color='c')


# We can see that there are few passengers who have paid exceptional high fare (more than 400) than the rest of the others!  
# Let's also create a Boxplot for passenger fare.  

# In[ ]:


titanic.fare.plot(kind='box', title='Boxplot for Fare')


# We can identify many outliers, but there is one (more than 500) which is significantly far from the others.   
# Let's try to find interesting information about those outliers.

# In[ ]:


titanic.loc[titanic.fare == titanic.fare.max()]


# There are 4 passengers who have paid a very high fare of 512, moreover, they have the same ticket number, and were also in the **First class**. This might a family that booked in last moment. Then, all of them survived to the tragedy.   
# 
# 
# ### Transformation of outliers   
# By looking at the fare histogram, we can see that we have a very right skewed distribution, so we can reduce the skeweness by applying some kind of transformation. As we know that the passenger fare will never be negative, then, we can apply the **Log** transformation technique.  
# 
# #### Log transformation

# In[ ]:


# LOG Transformation
'''
Adding 1.0 to accomodate zero fares, because log(0) is not defined
'''
Log_fare = np.log(titanic.fare + 1.0)  


# In[ ]:


# Let's look at the histogram of the Log transformation
Log_fare.plot(kind='hist', title='Histogram for Log_Fare', 
                  bins=30, color='c')


# We can see that our resultant histogram is less skewed than the original one (before transformation)   

# #### Binning   
# This is also a very useful technique for outliers treatment. We can the **binning** by using the pandas **qcut** function, that performs quantile-based binning. 
# Here we are splitting the passenger fare in 4 bins, where each bin contains almost equal number of observations. 

# In[ ]:


pd.qcut(titanic.fare, 4)


# We can see that pandas has created 4 bins, and each observation has been added into one bin. 
# This technique also allows us to specify a name for each bin.  
# To do so, we are going to create the following ones which are self explanatory: very_low_fare, low_fare, high_fare, vey_high_fare. Such technique is known as **discretization**. 

# In[ ]:


pd.qcut(titanic.fare, 4, 
        labels = ['very_low_fare', 'low_fare', 
                  'high_fare', 'vey_high_fare'])


# Here, we can see that pandas has assigned each observation, one label. So, we have created a categorical feature, where each fare value corresponds to a categorical level (ordinal).  
# Now, we can visualize the number of fare value for each category.

# In[ ]:


fare_labels = ['very_low_fare', 'low_fare','high_fare', 'vey_high_fare'] 

pd.qcut(titanic.fare, 4, labels=fare_labels).         value_counts().plot(kind='bar', color='c')


# We can see that we have almost the same number of observations in each bin.  
# Now that we have analysed and preprocessed the fare feature, we can now create a new feature corresponding to that preprocessed value, and remove the old one.  

# In[ ]:


titanic['fare_bin'] = pd.qcut(titanic.fare, 4, fare_labels)


# In[ ]:


titanic.drop('fare', axis=1, inplace=True)


# In[ ]:


titanic.head()


# We can see the change in the **fare_bin** column
