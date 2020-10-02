#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# * > ### Objective: To perform Statistical Analysis on the data set by implementing various stats modules (on New York AirBnb data) such as Hypothesis Testing,  Tests of Mean (Kruskal Wallis Test, ANOVA - one way and two way), Tests of Proportion (z test and chi-squared test) and Tests of Variance (F-test, Levene test), after checking for the three assumptions of (i) Normality of target variable (ii) Randomness of Sampling (iii) Equal variance across categories. The level of significance is assumed to be 5 percent (i.e. alpha = 0.05) If assumptions are satisfied, parametric tests can be performed, else non-parametric tests have to be performed. The results of the tests performed will enable us to find the associativity and dependability of different features on one-another. We shall use data visualization techniques to confirm our findings.

# In[ ]:


df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()


# * ### Cleaning the Data:

# #### Checking Null Values

# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


df['reviews_per_month'].head(10)


# In[ ]:


df['reviews_per_month'].mean()


# > #### Since the number of null values in columns 'last_review' and 'reviews_per_month' is large (10052), we can either perform forward or backward filling to impute the values or we can drop those rows. The problem with forward or backward fill is that (say for column 'reviews_per_month') if the next value after the null value is large (like 4.64 as shown above) but the actual number of reviews per month for that apartment were close to zero, then we are making a mistake by performing forward fill in that null value. Similar reasoning applies for both the columns. Thus, it makes more sense to drop the rows containing null values since it will not result in the Curse of Dimensionality for the data.

# In[ ]:


df = df.dropna()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# #### Outlier Treatment

# In[ ]:


df['price'].describe()


# > #### The variable 'price' contains large outliers. Therefore, to improve the normality of data, we will take the data between 25th percentile and 75th percentile, thereby eliminating the effects of outliers.

# In[ ]:


Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR=Q3-Q1
df = df[~((df['price']<(Q1-1.5*IQR))|(df['price']>(Q3+1.5-IQR)))]


# In[ ]:


df.info()


# ## Statistical Analysis

# ### Testing the assumptions:
# * Randomness of Data
# * Normality Test
# * Variance Test
# 
# The target variable being the price.

# In[ ]:


plt.figure(figsize=(10,7))
sns.distplot(df.price,color='r')
plt.xlabel("Price")
plt.title("Distribution of Price of Apartments")
plt.show()


# ### Shapiro Test (for checking Normality)
# 
# >H0 (Null Hypothesis) : Distribution is normal
# 
# >H1 (Alternate Hypothesis): Distribution is not normal

# In[ ]:


import scipy.stats as st
st.shapiro(df.price)


# > #### Since p value (0.0) is less than alpha (5% - assumed), the null hypothesis is rejected. Therefore, the distribution is not normal (as cound be seen from the Distribution Plot above). Since the distribution not normal, theoretically only non-parametric tests can be performed on the data.

# ## Price vs Room Type

# In[ ]:


df.room_type.unique()


# In[ ]:


pvt = df[df['room_type'] == 'Private room']
share = df[df['room_type'] == 'Shared room']
apt = df[df['room_type'] == 'Entire home/apt']


# ### Levene Test (for testing of variance)
# 
# > H0 (null hypothesis): variance(private_room) = variance(shared_room) = variance(entire_home)
# 
# > H1 (alternate hypothesis): variance(private_room) != variance(shared_room) != variance(entire_home)

# In[ ]:


st.levene(pvt.price, share.price, apt.price)


# > #### Since p value is approximately zero (and thus less than alpha = 0.05), we reject H0. Therefore, the variance of the price for different categories of rooms is not the same. This can be confirmed by observing the boxplot below.

# In[ ]:


plt.figure(figsize=(10,6))
sns.boxplot(y='price',x='room_type',data=df)
plt.show()


# ## Kruskal Wallis Test
# 
# > H0 (null hypothesis): mean_price(private_room) = mean_price(shared_room) = mean_price(entire_home/apt)
# 
# > H1 (alternate hypothesis): mean_price(private_room) != mean_price(shared_room) != mean_price(entire_home/apt)
# [](http://)

# In[ ]:


st.kruskal(pvt.price,share.price,apt.price)


# > #### In the above test result, p value < alpha (0.05). Therefore, null hypothesis is rejected. This implies that the mean price across different types of apartments is not the same. We can confirm that from the barplot of the mean prices shown below.
# 

# In[ ]:


ind = ['Private Rooms','Apartments','Shared Rooms']
x = pd.DataFrame([pvt.price.mean(),apt.price.mean(),share.price.mean()], index=ind)
x


# In[ ]:


x.plot.bar(color='g')
plt.title("Barplot of Mean Price Across Different Categories of Rooms")
plt.show()


# > #### Conclusion: There is association between Price and Room Type. The price is dependent on the type of room that a person chooses since the mean price across all types is not equal.

# #### Since the assumptions of Normality and Variance are violated, theoretically parametric tests cannot be performed but still it can be checked if the parametric test (one way ANOVA) gives the same result in this case.

# ## One Way ANOVA
# 
# 
# > H0 (null hypothesis): mean_price(private_room) = mean_price(shared_room) = mean_price(entire_home/apt)
# 
# > H1 (alternate hypothesis): mean_price(private_room) != mean_price(shared_room) != mean_price(entire_home/apt)

# In[ ]:


st.f_oneway(pvt.price,share.price,apt.price)


# In[ ]:


x.plot.bar(color='r')
plt.title("Barplot of Mean Price Across Different Categories of Rooms")
plt.show()


# #### Therefore, in one way ANOVA also, p value is less than alpha and null hypthesis is rejected implying that the mean prices across different types of rooms are not the same. The barplot is drawn again for reference.

# ## Price vs Neighbourhood

# In[ ]:


df.neighbourhood_group.unique()


# #### The 'neighbourhood_group' is a categorical variable having more than two categories, so we can perform either Kruskal Wallis test or One Way ANOVA test.

# ## Kruskal Wallis Test
# 
# > H0 (null hypothesis): mean_price(Brooklyn) = mean_price(Manhattan) = ..... = mean_price(Bronx)  (see categories above)
# 
# 
# > H1 (null hypothesis): mean_price(Brooklyn) != mean_price(Manhattan) != ..... != mean_price(Bronx)
# 

# In[ ]:


a = df[df['neighbourhood_group'] == 'Brooklyn']['price']
b = df[df['neighbourhood_group'] == 'Manhattan']['price']
c = df[df['neighbourhood_group'] == 'Queens']['price']
d = df[df['neighbourhood_group'] == 'Staten Island']['price']
e = df[df['neighbourhood_group'] == 'Bronx']['price']

st.kruskal(a,b,c,d,e)


# > #### Since p value is close to zero and less than alpha (0.05), we reject H0. Thus, the mean price across different neighbourhood groups is not the same.
# 
# > #### Conclusion: There is an association between price and neighbourhood group, i.e. the price is dependent on the neighbourhood group that the house is available in.

# ## One Way ANOVA
# 
# > H0 (null hypothesis): mean_price(Brooklyn) = mean_price(Manhattan) = ..... = mean_price(Bronx)
# 
# 
# > H1 (null hypothesis): mean_price(Brooklyn) != mean_price(Manhattan) != ..... != mean_price(Bronx)

# In[ ]:


st.f_oneway(a,b,c,d,e)


# > #### Since p value is close to zero and less than alpha (similar to Kruskal test), we reject H0. Thus, the mean price across different neighbourhood groups is not the same. This can be seen in the barplot below.

# In[ ]:


ind = ['Brooklyn','Manhattan','Queens','Staten Island','Bronx']
x = pd.DataFrame([a.mean(),b.mean(),c.mean(),d.mean(),e.mean()], index=ind)
x.plot.bar(color='m')
plt.show()


# ## Room Type vs Neighbourhood Group

# #### Since both the variables Room Type and Neighbourhood Group are categorical having more than two categories, we can peform Chi-squared test.

# ## Chi Squared Test
# 
# > #### H0 (null hypothesis): There is no association between Room Type and Neighbourhood Group.
# 
# > #### H1 (alternate hypothesis): There is an association between Room Type and Neighbourhood Group.

# In[ ]:


tab = pd.crosstab(df['room_type'],df['neighbourhood_group'])


# In[ ]:


st.chi2_contingency(tab)


# > #### The p value obtained (2.899e-23) is less than alpha (0.05) and thus the null hypothesis is rejected. 
# > #### Conclusion: There is association between Room Type and Neighbourhood Group implying that the proportion of availability of a particular type of room (private, shared, apartment) is dependent on the neighbourhood group in which we are searching. This can be verified from the stacked bar graph drawn below.

# In[ ]:


ct = pd.crosstab(df['room_type'],df['neighbourhood_group'])
ct.plot.bar(stacked=True)
plt.show()


# ## Price vs Neighbourhood Group & Host Name

# #### The continuous variable Price is to be compared with categorical variables Neighbourhood and Host Name having more than two categories. Thus, two way ANOVA test has to be used.

# 
# ## Two Way ANOVA

# ### Price vs Neighbourhood Group
# > #### H0: Mean Price (neighbourhood_group 1) = Mean Price (neighbourhood_group 2) = .... = Mean Price (neighbourhood_group n)
# > #### H1: Mean Price (neighbourhood_group 1) != Mean Price (neighbourhood_group 2) != .... != Mean Price (neighbourhood_group n)
# 
# ### Price vs Host Name
# > #### H0: Mean Price (host_name 1) = Mean Price (host_name 2) = .... = Mean Price (host_name n)
# > #### H1: Mean Price (host_name 1) != Mean Price (host_name 2) != .... != Mean Price (host_name n)

# In[ ]:


from statsmodels.formula.api import ols


# In[ ]:


model = ols("price~neighbourhood_group+host_name",data=df).fit()


# In[ ]:


# model.summary()


# In[ ]:


from statsmodels.stats.anova import anova_lm
anova_lm(model,typ=2)


# > #### The p values of host name and neighbourhood group are less than alpha (0.05). This implies that the mean price of rooms with different neighbourhood groups and host names are not equal.
# 
# > #### Conclusion: The mean price of a room has an association with the name of its host and the neighbourhood group it belongs to. The price for a room with one host name in one neighbourhood group will be different from the price of a room with a different host name in a different neighbourhood group.

# ## Correlation Between Continuous Variables

# #### A heatmap can be used to check the correaltion between continuous variables in the dataset.

# In[ ]:


plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')
plt.show()


# > #### Observations: The highest correlation exists between host_id and id (0.6) closely followed number_of_reviews and reviews_per_month, while the lowest correlation exits between longitude and price (-.092).

# ##  Neighbourhood vs Neighbourhood Group

# #### Both variables are categorical having more than two categories => perform chi-squared test.

# ## Chi-Squared Test
# 
# > #### H0: There is no association between neigbourhood and neighbourhood groups.
# > #### H1: There is an association between neighbourhood and neighbourhood groups.

# In[ ]:


tab = pd.crosstab(df['neighbourhood'],df['neighbourhood_group'])
st.chi2_contingency(tab)


# > #### Conclusion: The null hypothesis is rejected since the p value (0.0) is less than alpha (0.05). Thus, there is association between neighbourhood and neighbourhood group (as can be expected). This simply implies that the neighbourhood group is dependent upon the neighbourhood in which the house is present.

# ## Concluding Remarks
# * Performed different hypothesis tests on the data.
# * Found the association and dependability of different variables on one-another.
# * Performed both parametric (one way ANOVA, two way ANOVA) and non-parametric (z proportions test, Mann Whitney test, Kruskal Wallis test, Chi-squared test) tests on variables and compared the observations from both types.
# * Visualized and verified the conclusions drawn from the different tests using boxplots, distributions plots, bar graphs and stacked bar charts.
# 

# In[ ]:




