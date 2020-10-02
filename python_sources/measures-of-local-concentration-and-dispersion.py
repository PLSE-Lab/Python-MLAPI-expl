#!/usr/bin/env python
# coding: utf-8

# # Measures of Central tendency and dispersion  
# These are the measures that can help us answer some questions about our dataset, like its local concentration. For that, we have measures like the mean, the median, the mode and the quantiles, standard deviation, variance, Skewness and Kurtosis. These are the concepts we are going to cover in this notebook.  
# For the understanding purpose of the previous concepts, we are going to take in consideration only the **temperature feature**.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# ## 1. Looking at the features

# In[ ]:


data = pd.read_csv('../input/forestfires/forestfires.csv')
data.head()


# ## 2. Local Concentration

# We are going to determine the mean, the mode, and the median of the temperature parameter. 

# In[ ]:


from statistics import mean 
from scipy import stats 


# Determined by the mean, median, mode and quantiles. 

# In[ ]:


mean_value = data['temp'].mean()
mode_value = data['temp'].mode()
median_value = np.median(data['temp'])  

print('Mean : {}'.format(mean_value))
print('Mode : {}'.format(mode_value))
print('Median : {}'.format(median_value))


# ## 3. Dispersion 
# Determined by the standard deviation, and the variance. They help us identify how dispersed our data is. First of all, the standard deviation gives us how our data is far away from the mean value. Then the variance is the squared value of the standard deviation.  
# 
# 

# ![alt text](https://outlier.ai/wp-content/uploads/2016/09/Dispersion.png "Dispersion")

# 
# For example, by looking at the two distributions, we can see that the yellow one has a lower variance than the red one.  

# In[ ]:


from statistics import stdev


# In[ ]:


std_value = data['temp'].std()
var_value = data['temp'].var()
print('Standard deviation: {}'.format(std_value))
print('Variance : {}'.format(var_value))


# ## 4. Shape 
# Determined by the skewness and the kurtosis.  
# a- The skewness determines the degree of symetry of our distribution. The skewness can be skewed:  
# * to the left: negatively skewed. 
# * to the right: positively skewed.   
# Here is an example 

# ![alt text](https://upload.wikimedia.org/wikipedia/commons/f/f8/Negative_and_positive_skew_diagrams_%28English%29.svg "Dispersion")

# b- The kurtosis tells us the degree of peakness of a distribution.  
# Here is an example for three distributions.   
# ![alt text](https://tekmarathon.files.wordpress.com/2015/11/kurtosis_graph.png "Kurtosis")
# 

# In[ ]:


from scipy.stats import skew


# In[ ]:


skew_value = data['temp'].skew()
kurt_value = data['temp'].kurt()

print('Skeewness: {}'.format(skew_value))
print('Kurtosis: {}'.format(kurt_value))


# Our distribution is negatively skewed, which means that it looks like the distribution of the left in our previous example.  

# ## 5. Describe method 
# This function can summarize all the previous steps. 

# In[ ]:


data.describe()


# ## 6. Some visualizations  
# We are going to look at the distribution of the temperature value. 

# In[ ]:


sns.set(color_codes=True)
sns.kdeplot(data['temp'], shade=True)

# Mean and median values. 
plt.axvline(data['temp'].mean(), 0, 1, color = 'b') # shows the mean value
plt.axvline(data['temp'].median(), 0, 1, color = 'g') # shows the median value
#plt.xlabel()
#plt.ylabel()


# We can see the mean value at the intersection of the blue line and the X-axis. Then the median at the intersection of the green line and the X-axis. 

# We can show more details by getting the describe value from the describe method.

# In[ ]:


description = data.describe()
description['temp']


# In[ ]:


sns.set(color_codes=True)
sns.kdeplot(data['temp'], shade=True)

plt.axvline(description['temp']['25%'], 0, 1, color='g') # First percentile
plt.axvline(description['temp']['75%'], 0, 1, color='g') # Third percentile

IQR = description['temp']['75%'] - description['temp']['25%']

# Show the lower and upper outlier limits 
lower_outliers = description['temp']['75%'] - 1.5 * IQR
upper_outliers = description['temp']['25%'] + 1.5 * IQR 

plt.axvline(lower_outliers, 0, 1, color='r')
plt.axvline(upper_outliers, 0, 1, color='r')


# * The limit for the lower outliers is given by the first red vertical line (correspond to the first quartile Q1).   
# * The limit for the upper outliers limit is given by the second red vertical line (correspond to the third quartile Q3).
# 
# Outliers are the values that are very far away from the rest our our observations. Those values can affect our meaures of central tendency. 
# 
# The distance between the quartile values (Q1 and Q3) is the interquartile range (IQR). 
# An outlier is defined as those observations that are at least 1.5 times the interquartile range above from either Q1 or Q3. If an observation is 3.5 or more times far from this distance, it can be considered as an extreme outlier. These outliers can be good or bad depending on the problem we are solving. 
# For example: 
# * For the manufactoring analysis in a factory, we can have some broken machines that are also makes good with anomalies. So when studying the production standard, maybe we should get rid of these anomalies while making analysis.   
# 
# * In performing fraud detection for a financial problem, abnormal behaviors are important to take into consideration, so this time we should keep outiers in the analysis. 
