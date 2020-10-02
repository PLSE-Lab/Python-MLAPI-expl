#!/usr/bin/env python
# coding: utf-8

# # Problem - Predict FIFA 2018 Man of the Match

# ***If you find the content informative to any extend, kindly encourge me by upvoting. ***

# ## Load Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


data = pd.read_csv('../input/FIFA 2018 Statistics.csv')


# In[ ]:


data.shape


# In[ ]:


numerical_features   = data.select_dtypes(include = [np.number]).columns
categorical_features = data.select_dtypes(include= [np.object]).columns


# In[ ]:


numerical_features


# In[ ]:


categorical_features


# ## Univariate Analysis

# In[ ]:


data.describe()


# In[ ]:


data.hist(figsize=(30,30))
plt.plot()


# In[ ]:


skew_values = skew(data[numerical_features], nan_policy = 'omit')
pd.concat([pd.DataFrame(list(numerical_features), columns=['Features']), 
           pd.DataFrame(list(skew_values), columns=['Skewness degree'])], axis = 1)


# For normally distributed data, the skewness should be about 0. For unimodal continuous distributions, a skewness value > 0 means that there is more weight in the right tail of the distribution. The function skewtest can be used to determine if the skewness value is close enough to 0, statistically speaking.
# 
# Although data is not normally distribute, there are positive as well have negative skewedness
# - 'Yello & Red', 'Red' and 'Goals in PSO' are highly positively skewed.
# 

# In[ ]:


# Missing values
missing_values = data.isnull().sum().sort_values(ascending = False)
percentage_missing_values = (missing_values/len(data))*100
pd.concat([missing_values, percentage_missing_values], axis = 1, keys= ['Missing values', '% Missing'])


# In[ ]:


# encode target variable 'Man of the match' into binary format
data['Man of the Match'] = data['Man of the Match'].map({'Yes': 1, 'No': 0})


# In[ ]:


sns.countplot(x = 'Man of the Match', data = data)


# # Bivariate analysis
#     - Understanding how statistics of one feature is impacted in presence of other features
#     - Commonly used tools are:
#         - Pearson Correlation Coefficient (or) scatter plots
#         - Pairplots

#  ##### Correlation Coefficient
#  - The Pearson product-moment correlation coefficient, also known as r, R, or Pearson's r, is a measure of the strength and direction of the linear relationship between two variables that is defined as the covariance of the variables divided by the product of their standard deviations.
#  - It is of two type: Positive correlation and Negative correlation
#      - positive correlation if the values of two variables changing with same direction
#      - negative correlation when the values ofvariables change with opposite direction
#  - r values always lie between -1 to + 1
#  - Interpretation:
#         Exactly -1. A perfect downhill (negative) linear relationship
#         0.70. A strong downhill (negative) linear relationship
#         0.50. A moderate downhill (negative) relationship
#         0.30. A weak downhill (negative) linear relationship
#         0. No linear relationship
#         +0.30. A weak uphill (positive) linear relationship
#         +0.50. A moderate uphill (positive) relationship
#         +0.70. A strong uphill (positive) linear relationship
#         Exactly +1. A perfect uphill (positive) linear relationship
# 

# In[ ]:


plt.figure(figsize=(30,10))
sns.heatmap(data[numerical_features].corr(), square=True, annot=True,robust=True, yticklabels=1)


# In[ ]:


var = ['Man of the Match','Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 
       'Fouls Committed', 'Own goal Time']
corr = data.corr()
corr = corr.filter(items = ['Man of the Match'])
plt.figure(figsize=(15,8))
sns.heatmap(corr, annot=True)


# - 'Man of the Match' is highly correlated with 'Goal Scored', 'On-Target', 'Corners', 'Attempts', 'free Kicks', 'Yellow Card', 'red', 'Fouls Committed', 'Own goal Time'
# - OWn goal time is twins of 'Ball possession %', and Passes, pass Accuracy
# - Pass HCW Pass Accuracy %
# - passes SCW 'Ball possession %
# - Passes HCW 'Attempts'
# - Goals in PSO SCW DIstance Covered (Kms)
# 
# - Correlated columns needs to be removed to avoid multicollinearity. Let's use multicollinearity check

# - These features have least or no correlation with 'Man of the Match'
#     - ['Blocked', 'OffSides', 'Saves','Distance Covered (Kms)', 'Yellow & Red', '1st Goal', 'Goals in PSO']
#     - These features will not have impact on aur analysis and thus, holding them or retaining them is our choice
#     - We will see what to do with these later

# In[ ]:


var = ['Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 
       'Fouls Committed', 'Own goal Time']
plt.figure(figsize=(15,10))
sns.heatmap((data[var].corr()), annot=True)


# #### Features contributing values in 'Yello' , 'Ligh Yellow', 'Black' and 'Dark black' boxes can lead to multi-collinearity.
#     Colinearity is the state where two variables are highly correlated and contain similiar information about the variance within a given dataset. 
#     
#     To detect colinearity among variables, simply create a correlation matrix and find variables with large absolute values. In R use the corr function and in python this can by accomplished by using numpy's corrcoef function.
#     
#     Multicolinearity on the other hand is more troublesome to detect because it emerges when three or more variables, which are highly correlated, are included within a model. To make matters worst multicolinearity can emerge even when isolated pairs of variables are not colinear. Multi-collinearity is an important pipeline steps fpr 
# 
# - Steps for Implementing VIF
#     - Run a multiple regression.
#     - Calculate the VIF factors.
#     - Inspect the factors for each predictor variable, if the VIF is between 5-10, multicolinearity is likely present and you should consider dropping the variable.

# Let's understand relations of each of above 9 features with respect to 'Man of the match' closely using scatter plot, box plot etc.
# 
# Scatter plot is another great tool to see correlation degree and direction among features.  Using seaborn pairplot makes this task easy for us by plotting all possible combinations.

# In[ ]:


var1 = ['Goal Scored', 'On-Target', 'Corners', 'Attempts', 'Free Kicks', 'Yellow Card', 'Red', 'Fouls Committed']
var1.append('Man of the Match')
sns.pairplot(data[var1], hue = 'Man of the Match', palette="husl")
plt.show()


# - As I can notice 'Attempts' is linearly proportional to 'On-Target' and 'Corners'
# - 'Corners' and 'On-Targets' are also linearly positively proportional

# ## Outliers detection and removal

# In[ ]:


dummy_data = data[var1]
plt.figure(figsize=(20,10))
sns.boxplot(data = dummy_data)
plt.show()


# - As per boxplot there are :
#     - 1 outlier in Goal scored
#     - 2 in On-Target
#     - 1 in corners
#     - 2 in Attempts
#     - 3 in Yellow Card
#     - 1 in Red
# - In statistics, an outlier is an observation point that is distant from other observations. An outlier may be due to   variability in the measurement or it may indicate experimental error; the latter are sometimes excluded from the data set.
# - In simple words, for a normally distributed data any value that lies beyond range of 1.5 times IQR (Inter quartile range) is considered to be an outliers.
# - However, 'outliers = anything > 1.5*IQR' does not word practically, as real data are not normally distributed.
# - Pragmatic approach: plot scatter visualisation or boxplot and identify abnormally distant points
# 
# -  The quantity of outliers present in this problem is not too huge and will not have gravity impact if left untreated. They are only few and within range.

# ## Missing values treatment
# ![image.png](attachment:image.png)
# 
#     - As 'own Goal Time' and 'Own goals' are having > 90% missing values, filling them with any combination will lead predictive model to false direction. So, dropping them is the best option
#     - '1st Goal' represents 'When did the team score the 1st goal?'
#     - As per discription 1st Goal should provide date information but the data is a numeric value.
#     - It's possible that these numerical values is nothing but number of days between two dates [Today's date - Date when team was formed']
#     - As Date when a team was formed is not given, missing values can be filled with some stats values.
#     - But, filling number of days information with mean, median, mode, etc. does not seem to be informative to me, and thus, I will drop this field to rather than using it by filling with uninformative data.
#     - Note: '1st Goal' is negligebly correlated with 'Man of the Match', hence, dropping this should not have any impact
#    

# In[ ]:


data.drop(['Own goal Time', 'Own goals', '1st Goal'], axis = 1, inplace= True)


# ## Categorical features encoding
#     - As machine laearning models understand only numbers data in different formats including text and dates needs to be mapped into numbers prior to feeding to the model
#     - The process of changing non-numerical data into numerical is called 'Encoding'
#     - Before encoding let's understand how many categories or levels are present in each categorical features

# In[ ]:


categorical_features


# In[ ]:


def uniqueCategories(x):
    columns = list(x.columns).copy()
    for col in columns:
        print('Feature {} has {} unique values: {}'.format(col, len(x[col].unique()), x[col].unique()))
        print('\n')
uniqueCategories(data[categorical_features].drop('Date', axis = 1))


# - Categorical -['Date', 'Team', 'Opponent','Round', 'PSO']
#     - Nominal - Team, Opponent
#     - Ordinal - Round
#     - Interval - Date, PSO is binary
# - Including nominal data is of no use, however, I am guessing combination of Team and Opponent should be useful. If not, we will drop them. Also, 'Man of the Match' depends, as per data, on goal scored. A player from good team should be capable enough to score high. So, team branding turns out to be an important factor. In our data, there is no way to identify each team brand value.
# - I believe 'Round' should also not have any impact on 'Man of the Match' because, a player performance should be consistent over all matches to become man of the match than just in a particular round. Thus, let's give equal weitage to each round.
# - PSO is binary 
# - I am not going to include 'Match date' as it should definately not impact a player formance.

# In[ ]:


data.drop('Date', axis = 1, inplace=True)


#  - Dropping "Corners', 'Fouls Committed' and 'On-Targets' will remove high correlated elements and remove chances of multi-collinearity. these features are selected based on their low collinearity with 'Man of the Match' and high collinearity with other features.

# In[ ]:


data.drop(['Corners', 'Fouls Committed', 'On-Target'], axis = 1, inplace=True)


# In[ ]:


print(data.shape)
data.head()


# In[ ]:


cleaned_data  = pd.get_dummies(data)


# In[ ]:


print(cleaned_data.shape)
cleaned_data.head()


#     - The data has been cleaned and is ready for further steps in data pipeling
#         - Pre-processing
#         - Modeling
#         - Evaluation
#         - Prediction

# ***If you found the content informative to any extend, kindly encourge me by upvoting. :)***
