#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# This is my first project after almost three years away from anything related to Machine Learning. When I started it, the very first thing that came on my mind was: "WHY DID YOU TAKE SO LONG TO RETURN?". I could say there were a lot of things that kept me away from ML but... there are no enough excuses. So, no... I'm not going to argue with myself... it's better to explain how was this returning experience. 
# 
# The first thing that I did was to read the [Ames, Iowa: Alternative to the Boston Housing Data as anEnd of Semester Regression Project](https://ww2.amstat.org/publications/jse/v19n3/decock.pdf) document, so I could understand better the data set of this competition, its main features, how they can be related and any other detail that could help my progess.
# 
# Okay, at this point I was able to start a simple linear regression implementation but then I realized that I did not have enough ML Python skills to perform it, so I found the [Dan Becker's free course](https://www.kaggle.com/learn/machine-learning) and it helped me a lot not only to learn ML Python libraries usage but also to review/learn some pre-processing techniques, modeling and evaluation methods. Thank you Dan!
# 
# Then I was ready... was I? Actually, I wasn't sure about my data analysis skills, so I started to dive into some interesting notebooks, such as:
# * [Comprehensive data exploration with Python](https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python/notebook) (Pedro Marcelino)
# * [Stacked Regressions : Top 4% on LeaderBoard](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook) (Serigne)
# * [Regularized Linear Models](https://www.kaggle.com/apapiu/regularized-linear-models/notebook) (Alexandru Papiu)
# * [A study on Regression applied to the Ames dataset ](https://www.kaggle.com/juliencs/a-study-on-regression-applied-to-the-ames-dataset)(juliencs)
# 
# These notebooks are great and I could learn so much about data analysis in Python, Feature Engineering and modeling. Thanks to all authors. Thus, it was time to start.
# 
# 

# **Data Analysis and Cleaning**
# On this section, it's possible to find two main approaches for data analysis and cleaning: Univariate and Multivariate Linear Regression.
# 
# * **Univariate Linear Regression**: once this is a univariate linear regression, the first thing to do is check all the variables correlations to the target (the Sale Price) in order to choose the variable that would make the best predictions. We can generated the correlation matrix  using the seaborn heatmap method and/or use the *pandas.DataFrame.corr()* method.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from scipy.stats import norm
from scipy.stats import skew
pd.options.mode.chained_assignment = None  # default='warn'

file_path = '../input/train.csv' 
train_data = pd.read_csv(file_path)

correlation = train_data.corr()
correlation.sort_values(["SalePrice"], ascending = False, inplace = True)
print(correlation.SalePrice)
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(correlation, vmax=.8, square=True);
plt.show()


# Space and quality... It's not a surprise that the overall quality and the above grade living area are the most correlated variables to the Sale Price of a house, but which one is the most suitable to make better predictions? We can decide it later. For now, let's check if any of these variables has missing values.

# In[ ]:


print("Missing values count of OverallQual = ", train_data.OverallQual.isnull().sum())
print("Missing values count of GrLivArea = ", train_data.GrLivArea.isnull().sum())


# Good! No missing values. At this point we can try to run a[ simple test ](https://www.kaggle.com/brianviegas/univariate-linear-regression-house-prices)to see if our data analysis is going well.

# The initial results for both variables were really bad. We need more information about the correlation between each one of them and the Sale Price variable. One way to see that is generating some scatter plots.

# In[ ]:


key = 'GrLivArea'
data = pd.concat([train_data['SalePrice'], train_data[key]], axis=1)
data.plot.scatter(x=key, y='SalePrice', ylim=(0,800000))

key = 'OverallQual'
data = pd.concat([train_data['SalePrice'], train_data[key]], axis=1)
data.plot.scatter(x=key, y='SalePrice', ylim=(0,800000))


# It's possible to see that both variables have linear relationships, but they also have some values that are not following the pattern. In the first scatter plot, two entries have the biggest GrLivArea values and their SalePrice is too low, and this might indicate that they are outliers. Since outliers can jeopardize the overall performance, these values shoulde be removed. Actually, based on the suggestion of data set author *"I would recommend removing any houses with more than 4000 square feet from the data set"* it's better to remove all the houses with more than 4000 square feet from the data set.

# In[ ]:


train_data = train_data[train_data.GrLivArea < 4000]


# Now it's time to check the variables distribution. We can do this by plotting their histograms.

# In[ ]:


print("Skew = ", train_data['OverallQual'].skew())
sns.distplot(train_data['OverallQual'], fit=norm)


# In[ ]:


print("Skew = ", train_data['GrLivArea'].skew())
sns.distplot(train_data['GrLivArea'], fit=norm)


# As we can see, both GrLivArea and OverallQual graphs are skewed. In this case, since the skewness of OverallQual is low, we will apply the log transformation only on GrLivArea variable in attempt to get it more normal distributed.

# In[ ]:


train_data['GrLivArea'] = np.log(train_data['GrLivArea'])
print("Skew = ", train_data['GrLivArea'].skew())
sns.distplot(train_data['GrLivArea'], fit=norm)


# Now the features are ready to be used, it's time to check the target normality.

# In[ ]:


print("Skew = ", train_data['SalePrice'].skew())
sns.distplot(train_data['SalePrice'], fit=norm)


# Skewed to the right. Following the same concept, we apply the log transformation on SalePrice variable.

# In[ ]:


train_data['SalePrice'] = np.log(train_data['SalePrice'])
print("Skew =  ", train_data['SalePrice'].skew())
sns.distplot(train_data['SalePrice'], fit=norm)


# That's it. Let's [test](https://www.kaggle.com/brianviegas/univariate-linear-regression-house-prices) to see how it goes. 

# * **Multivariate Linear Regression**: the first step on this approach could be the same of the previous one:  check all the variables and see how they are correlated. But, for this case, we will work with all variables available on data set. Thus, we start by checking the missing values of all variables.

# In[ ]:


missing_values_count = train_data.isnull().sum().sort_values(ascending=False)
missing_values_percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([missing_values_count, missing_values_percent], axis=1, keys=['Total', 'Percent'])
print(missing_data[missing_data['Total'] > 0])


# We can notice that for some variables, such as PoolQC, MiscFeature, Alley, Fence, FireplaceQu, more than 40% of their data are missing. This tell us a lot about them and after some analysis we can see that none of them has great impact on the target. Actually, if we analyze carefuly none of the missing values variables have relevant correlation to the target. In addition to that, they may also contain outliers. So we could remove most of them, but since we are working with all variables, we will filling up all the missing values. We can classify the those variables types as Categorical or Numerical. Based on the data description, for both categorical and numerical variables we will assume that the input with missing values does not have the attribute. For categorical we set 'None' and for numerical variables we set 0. The only exceptions is the LotFrontage variable, since we may assume the area of each street connected to the house probably have a similar area to other houses in its neighborhood. For the Electrical variable, we can just remove the unique input that has missing value.

# In[ ]:


print(train_data.isnull().sum().max())
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'):
    train_data[col] = train_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    train_data[col] = train_data[col].fillna(0)
train_data["LotFrontage"] = train_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
train_data = train_data.drop(train_data.loc[train_data['Electrical'].isnull()].index)
print(train_data.isnull().sum().max())


# At this point, we could analyze the numerical variables in order to find possible outliers. Since we have a large set of variables, we can use the skewness analysis and the consequent transformation to minimize the impact of outliers on the model. As general rule, we will transform variables with skewness greater than 0.5.

# In[ ]:


numerical_features = train_data.select_dtypes(exclude = ["object"]).columns
train_data_numerical = train_data[numerical_features]
skewness = train_data_numerical.apply(lambda x: skew(x))
skewness = skewness[abs(skewness) > 0.5]
skewed_features = skewness.index
train_data_numerical[skewed_features] = np.log1p(train_data_numerical[skewed_features])


# For the categorical variables we just need to convert them to dummy variables via one-hot enconding.

# In[ ]:


categorical_features = train_data.select_dtypes(include = ["object"]).columns
train_data_categorical = train_data[categorical_features]
train_data_categorical = pd.get_dummies(train_data_categorical)


# Missing values: Check!
# Transformation: Check!
# One-hot enconding: Check!
# 
# Now we just put our features together again and [go train our model](https://www.kaggle.com/brianviegas/multivariate-linear-regression-house-prices). 
#     

# In[ ]:


# Join categorical and numerical variables
train_data = pd.concat([train_data_numerical, train_data_categorical], axis = 1)
print("NAs count of features in data set : ", train_data.isnull().values.sum())


# **Conclusion**
# * As a home buyer, when someone tell me the price of some house the first thing that comes in my mind is the area. So I think the Univariate approach, considering the basic ouliers cleaning, the log transformation of GrLivArea and SalePrice did perfom well, even with the Gradient Descent algorithm. No heavy processing, no subjectivity. 
# The multivariate approach did perform better but I believe that some variables could be removed from the modeling process, most because they don't have relevant correlation to the target. Who is going to ask about the masonry veneer type?
# Thus, a multivariate approach with the most relevant variables, a better algorithm for modeling and more feature engineering techniques would be the best approach. As soon as possible I will try it and I will update this notebook.
# **Future Improvements**
# * **Feature Engineering**
# For the multivariate approach, we can perform some feature engineering such as simplify some variables by reducing the range of the discrete values on subjective fields. For example, the OverallQual and OverallCond are good candidates for that, so their values (1 to 10) could be reduced (1 to 3). Another strategy would be combine correlated variables or create polynomial variables based on their correlation to the target.
# * **Modeling **
# Scikit-learn has a variety of machine learning algorithms that could improve the prediction performance (not only regression algorithms). For me, it's just a matter of time to explore and use as many of them as possible to improve my results. In particular, the Gradient Boosted Decision Trees algorithm that works in cycles with an incremental model training approach building an esemble model.
