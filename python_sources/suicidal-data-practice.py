#!/usr/bin/env python
# coding: utf-8

# In this project, I will try to visulize the global suicidal datasets and find meaningful correlation between suicides and other socio-geological factors.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Importing datasets**

# In[ ]:


total_suicidal=pd.read_csv("../input/master.csv")


# **Showing top 5 rows of datasets**

# In[ ]:


total_suicidal.head()


# This indicates that there are multiple rows for each country for different conditions such as "Male vs Female" or "Generation X vs Boomers", and there are 12 attributes in total. 
# 
# The "Country" coulmn's repetitiveness (and some other columns' too) means that it is probably a categorical attribute. Let's see what categories exist and how many of conditions belong to each category by using the value_counts() method:

# In[ ]:


total_suicidal["country"].value_counts()


# **Getting information about datasets**

# Let's get a quick description of the data, in particular the total number of rows, and each attribute's type and number of non-null values

# In[ ]:


total_suicidal.info()


# Because there are only 8364 non-null instances for "HDI for year" attribute while there are 27,820 instances in total, we can expect about ~19,000 null instances in "HDI for year".
# Also, since this data came from a CSV file, the "object" type must indicate a text attribute.

# **Using DESCRIBE method for a summary**

# The describe() method shows a summary of the numerical attributes:

# In[ ]:


total_suicidal.describe()


# Note that the null values are ignored as you can see in the 'count' of "HDI for year". The std row shows the standard deviation. The 25%,50% and 75% rows show the corresponding percentiles below which a given percentage of observations in a group of observations falls.

# **Plotting histogram**

# Let's plot a histogram for each numerical attribute to better understand the type of data we are dealing with although this does not directly show any correlation yet.

# In[ ]:


import matplotlib.pyplot as plt
total_suicidal.hist(bins=50, figsize=(20,15))
plt.show()


# **Data visualization**

# * Global trend of suicides as a function of year

# We can take a quick look at the data by visualizing some of the features.
# Let's first take a look at the global trend of suicides. For this we first need to sum of the number of suicides for each year:

# In[ ]:


year_grouped=total_suicidal.groupby("year")


# We will now sum them up for each year:

# In[ ]:


year_grouped_sum=year_grouped.sum()


# In[ ]:


import seaborn as sns


# In[ ]:


plt.rcParams["axes.labelsize"] = 20
ax = sns.lineplot(x='year', y="suicides/100k pop", data=year_grouped_sum.reset_index())
ax.set(xlabel='Year', ylabel='Suicides / 100k population')


# We can see that there is a dramatic drop in the total number of suicides past year 2015.

# * Suicides by gender

# Let's see how gender has affected suicides.

# In[ ]:


gender_grouped=total_suicidal.groupby("sex")


# In[ ]:


gender_grouped_sum=gender_grouped.sum()


# In[ ]:


gender_grouped_sum=gender_grouped_sum.reset_index()


# In[ ]:


plt.rcParams["axes.labelsize"] = 20
ax=sns.catplot(x='sex', y='suicides/100k pop', 
            data=gender_grouped_sum, kind="bar", height=10, aspect=0.6)
ax.set(xlabel='Gender', ylabel='Suicides / 100k population')
plt.show()


# We can clearly see that men have committed more suicides than women

# * Suicides by country

# Same thing as above but now for each country

# In[ ]:


country_grouped=total_suicidal.groupby("country")


# In[ ]:


country_grouped_sum=country_grouped.sum()


# In[ ]:


country_grouped_sum=country_grouped_sum.reset_index()


# In[ ]:


country_grouped_sum_sorted=country_grouped_sum.sort_values('suicides/100k pop')


# In[ ]:


plt.rcParams["axes.labelsize"] = 30
ax=sns.catplot(x='suicides/100k pop', y='country', 
            data=country_grouped_sum_sorted, kind="bar", height=20, aspect=0.6)
ax.set(xlabel='Total Suicides', ylabel='Country')
plt.show()


# From the figure above, we can see that Russian Federation was the country with the highest number of suicides over the years between 1985 and 2016.

# * Suicides by generation

# We can also compare generation by generation.

# In[ ]:


gen_grouped=total_suicidal.groupby("generation")


# In[ ]:


gen_grouped_sum=gen_grouped.sum()


# In[ ]:


gen_grouped_sum=gen_grouped_sum.reset_index()


# In[ ]:


plt.rcParams["axes.labelsize"] = 40
sns.set(font_scale=2)
ax=sns.catplot(x='suicides/100k pop', y='generation', 
            data=gen_grouped_sum, kind="bar", height=15, aspect=0.8)
ax.set(xlabel='Total suicides/100K population', ylabel='Generation')
plt.show()


# This hows that the "Silent" generation has committed the most number of suicides/100K population across all years and countries.

# * GDP effect on suicides

# Now, let's take a deeper look into the GDP effect on suicides. I will choose "GDP per capita" for every conntry across all the available years for my analysis. Because GDP per capita is changing over the years, I will choose an average for this. For the same reason, I will take an average for suicides/100k population data.

# In[ ]:


se=total_suicidal.groupby(['country','year'])['suicides/100k pop'].mean().reset_index().groupby('country').mean()


# In[ ]:


se=se.reset_index()


# In[ ]:


se2=total_suicidal.groupby(['country','year'])['gdp_per_capita ($)'].mean().reset_index().groupby('country').mean()


# In[ ]:


se2=se2.reset_index()


# In[ ]:


se['100k_avg']=se["suicides/100k pop"]


# In[ ]:


se['gdp_per_capita_average']=se2["gdp_per_capita ($)"]


# In[ ]:


se.head()


# In[ ]:


sns.set(rc={'figure.figsize':(15,10)})
plt.rcParams["axes.labelsize"] = 20
ax=sns.scatterplot(x='gdp_per_capita_average', y='100k_avg', s=100, hue="100k_avg",palette="Set1",data=se)
ax.set(xlabel='GDP per capita ($)', ylabel='Suicides / 100K')
plt.show()


# Let's try to fit a linear model to this data just for our initial try. First, I will create numpy arrays for "GDP per capita" and "Suicides/100K pop" attributes.

# In[ ]:


se_num_X=se["gdp_per_capita_average"]


# In[ ]:


se_num_Y=se["100k_avg"]


# In[ ]:


se_num=pd.concat([se_num_X,se_num_Y],axis=1)


# In[ ]:


se_num_Xarray=se_num_X.values.reshape(-1,1)


# In[ ]:


se_num_Yarray=se_num_Y.values.reshape(-1,1)


# Loading linear regression model

# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lin_reg=LinearRegression()


# In[ ]:


lin_reg.fit(se_num_Xarray,se_num_Yarray)


# In[ ]:


print("Y intercept value is", lin_reg.intercept_)


# In[ ]:


print("Slope is", lin_reg.coef_)


# Let's plot the predicted data (red) along with the original data (gray).

# In[ ]:


plt.scatter(se_num_X, se_num_Y, label='Data (WITH outliers)', color='green', marker='^', alpha=.5)
ax=sns.regplot(x='gdp_per_capita_average', y='100k_avg', data=se_num, scatter=None, color="g",label="Linear_outlier")
ax.set(xlabel='GDP per capita ($)', ylabel='Suicides / 100K')
plt.legend(loc="best")
plt.show()


# It does not look super bad, but not very convincing either. Let's remove some of the outliers based on the Zscore values.

# In[ ]:


from scipy.stats import zscore


# In[ ]:


se_Y_zscore=zscore(se_num_Y)


# Let's use a threshold of 3 for our Zscore filtering

# In[ ]:


se_bool_Y=np.absolute(se_Y_zscore)<3


# In[ ]:


se_bool_series=pd.Series(se_bool_Y)


# In[ ]:


new_se=pd.concat([se_num_X,se_num_Y,se_bool_series],axis=1)


# In[ ]:


new_se=new_se[new_se[0]]


# Let's rerun linear regression without the outlier point.

# In[ ]:


new_se_num_Xarray=new_se["gdp_per_capita_average"].values.reshape(-1,1)


# In[ ]:


new_se_num_Yarray=new_se["100k_avg"].values.reshape(-1,1)


# In[ ]:


lin_reg.fit(new_se_num_Xarray,new_se_num_Yarray)


# In[ ]:


print("Y intercept value is", lin_reg.intercept_)


# In[ ]:


print("Slope is", lin_reg.coef_)


# In[ ]:


plt.scatter(se_num_X, se_num_Y, label='Data (WITH outliers)', color='green', marker='^', alpha=.5)
sns.regplot(x='gdp_per_capita_average', y='100k_avg', data=se_num, scatter=None, color="g",label="Linear_outlier")
plt.scatter(new_se["gdp_per_capita_average"], new_se["100k_avg"], label='Data (WITHOUT outliers)', color='red', marker='o', alpha=.5)
ax=sns.regplot(x='gdp_per_capita_average', y='100k_avg', data=new_se, scatter=None, marker="^",color="r",label="Linear_no_outlier")
ax.set(xlabel='GDP per capita ($)', ylabel='Suicides / 100K')
plt.legend(loc="best")
plt.show()


# We can see that there was only one outlier (Lithuania in the actual dataset), and there is also a slight linearly increasing trend even without the outlier. However, the overdata is pretty scattered, and thus it should not be concluded that GDP per capita directly and linearly affects suicides.

# * Linearity test

# Before we move onto some prediction work, let's find which country has a linearly increasing suicidal trend **over YEARS** for easiness.
# We will test this through hypothesis test for regression slope.

# In[ ]:


se3=total_suicidal.groupby(['country','year'])['suicides/100k pop'].sum().unstack()


# In[ ]:


se3=se3.reset_index()


# In[ ]:


se3.head()


# Transposing datasets for easiness

# In[ ]:


se3=se3.T.reset_index()


# In[ ]:


se3.columns = se3.iloc[0]


# In[ ]:


se3=se3.rename(columns = {'country':'year'})


# In[ ]:


se3=se3.drop([0])


# In[ ]:


se3=se3.replace('NaN', np.NaN)


# In[ ]:


se3.head()


# In[ ]:


column_length=len(se3.columns)


# Data needs to be processed a bit to exclude the rows with "NaN" values. Also, we will choose those datasets that show R^2 values higher than 0.8 as the datasets that match very well with linear regression.

# In[ ]:


from sklearn.metrics import r2_score


# In[ ]:


index_append=[]
r2_append=[]
coeff_append=[]
for x in range(1,column_length):
    new_series=pd.concat([se3['year'],se3.iloc[:,x]], axis=1)
    new_series2=new_series.dropna()
    new_data=pd.DataFrame(new_series2).reset_index()
    year_array=new_data['year'].values.reshape(-1,1)
    sui_array=new_data.iloc[:,2].values.reshape(-1,1)
    lin_reg.fit(year_array,sui_array)
    coeff=lin_reg.coef_
    y_pred = lin_reg.predict(year_array)
    r2_sklearn = r2_score(sui_array,y_pred) 
    if 0.8<r2_sklearn<1:
        index_append.append(x)
        r2_append.append(r2_sklearn)
        coeff_append.append(coeff)


# In[ ]:


scal_coeff=[]
for x in coeff_append:
    co=np.asscalar(x)
    scal_coeff.append(co)
print(scal_coeff)


# In[ ]:


cols=se3.columns


# In[ ]:


L=[]
L2=[]
for x in range(0,len(index_append)):
    new_country=cols[index_append[x]]
    L.append(new_country)
se4=pd.DataFrame(L, columns=['country'])
se5=pd.DataFrame(r2_append, columns=['R^2'])
se_co=pd.DataFrame(scal_coeff, columns=['coeff'])
se6=pd.concat([se4,se5,se_co], axis=1)
#se_sign=se6['coeff'].apply(lambda x: x<0)
#se6['negative']=se_sign
se6.loc[se6.coeff<0, 'sign']='Negative trend'
se6.loc[se6.coeff>0, 'sign']='Positive trend'


# In[ ]:


ax = sns.catplot(x="R^2", y="country", data=se6, saturation=0.5, kind="bar", hue="sign",ci=None, height=10,aspect=1,palette="Set2",legend=False)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ax.set(xlabel='R^2 value for linear regression', ylabel='Country')


# Now let's see how these countries look. For negatively trending countries, I will look at Finland, France and Hungary while I will look at Mexico, Phillippines and Republic of Korea for the postively trending countries.

# In[ ]:


L=['Finland','France','Hungary','year']


# In[ ]:


se_N=se3[L]


# In[ ]:


se_N=se_N.reset_index()


# In[ ]:


plt.scatter(se_N['year'], se_N['Finland'], label='Finland', color='blue', marker='^', alpha=.5)
ax=sns.regplot(x='year',y='Finland',data=se_N, scatter=None, color='blue',label="Finland")
plt.scatter(se_N['year'], se_N['France'], label='France', color='red', marker='^', alpha=.5)
ax2=sns.regplot(x='year', y='France', data=se_N, marker="^",color="red",scatter=None,label="France")
plt.scatter(se_N['year'], se_N['Hungary'], label='Hungary', color='green',marker='^', alpha=.5)
ax3=sns.regplot(x='year', y='Hungary', data=se_N, marker="x",color="green", scatter=None, label="Hungary")
ax3.set(xlabel='Year', ylabel='Suicides / 100K')
plt.legend(loc="best")
plt.show()


# From this we can see that Hundary's suicidal rate has been decreasing the fastest.

# Now let's take a look at the positively trending countries.

# In[ ]:


L=['Republic of Korea','Mexico','Philippines','year']


# In[ ]:


se_P=se3[L]


# In[ ]:


se_P=se_P.reset_index()


# In[ ]:


plt.scatter(se_P['year'], se_P['Republic of Korea'], label='Republic of Korea', color='blue', marker='^', alpha=.5)
ax=sns.regplot(x='year',y='Republic of Korea',data=se_P, scatter=None, color='blue',label="Republic of Korea")
plt.scatter(se_P['year'], se_P['Mexico'], label='Mexico', color='red', marker='^', alpha=.5)
ax2=sns.regplot(x='year', y='Mexico', data=se_P, marker="^",color="red",scatter=None,label="Mexico")
plt.scatter(se_P['year'], se_P['Philippines'], label='Hungary', color='green',marker='^', alpha=.5)
ax3=sns.regplot(x='year', y='Philippines', data=se_P, marker="x",color="green", scatter=None, label="Philippines")
ax3.set(xlabel='Year', ylabel='Suicides / 100K')
plt.legend(loc="best")
plt.show()


# You can see the Republic of Korea (South Korea)'s suicidal trend has been increasing very rapidly, which is concerning.

# **Conclusion**

# - Overall, suicides have been decreasing since early 2000.
# - Men have committed more sucides than women.
# - "Silent" generation has committed the most suicides.
# - There is a positive correlation between suicides and GDP per capita.
# - Of those countries that have a linear relationship between Suicides/100K and Year, South Korea has the steepest increasing slope.
