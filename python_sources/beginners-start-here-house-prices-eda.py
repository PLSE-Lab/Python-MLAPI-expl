#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ![](https://cdn.geekwire.com/wp-content/uploads/2020/03/bigstock-Residential-Neighborhood-And-F-345221452-630x420.jpg)

# Hi and welcome to this notebook on predicting house prices! If you are new to data analysis, Python and/or machine learning, this is the perfect place to start!
# 
# This notebook will guide you through this dataset - it is composed of more than 4,000 houses and the price they sold for, as well as some of their features, like number of bedrooms, bathrooms, etc...
# 
# The goal of this notebook is to answer the following question: *what are the most important features that influence the price of a house?*
# 
# Here is what we're going to do:
# 
# 1. Exploratory Data Analysis (EDA) - let's find out what the features are and how they individually affect house prices
# 2. Recoding - fixing the problems we've identified in EDA
# 3. Feature selection - we have to be picky about which features we include in the model - which ones have to be in and which ones are optional?
# 4. Model building - let's see what is the impact of our features on house prices!
# 
# So if you haven't already, grab yourself a nice cuppa and let's dig in!
# 
# 

# In[ ]:


# here are the modules we'll be using throughout this notebook
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import math


# In[ ]:


from sklearn.metrics import mean_absolute_error


# In[ ]:


from scipy import stats


# In[ ]:


# Load our data from the csv file
houses = pd.read_csv('../input/housedata/data.csv') 


# So, how many houses do we have in our dataset?

# In[ ]:


houses.shape


# OK, so the first number tells us the number of rows (the number of houses) and the second one is the number of columns (the number of features).
# 
# We have 4600 houses in the dataset and 18 features, including price. Therefore, we can choose between 17 different features that influence the price of a house. Let's check these out.

# In[ ]:


houses.dtypes


# We have a nice list of all the features, some being *categorical variables* (object types), like the country of the house and some being *measures* (float64 or int64 types) like the surface of the basement.
# 
# Just curious, what country are these houses in?

# In[ ]:


houses.country.value_counts()


# Interesting. What state ?

# In[ ]:


houses.statezip.value_counts()


# Alright, so all of the houses are located in the state of Washington (Pacific Northwest, where Seattle is). This might serve us for later.
# 
# Let's check out the average price of a house in that area.

# In[ ]:


"The average price of a house is ${:,.0f}".format(houses.price.mean())


# I live in New Zealand and **550,000 USD** is roughly **900,000 NZD**, which gets you quite a nice house down here!

# To me the first feature of a house that stands out is the number of bedrooms. Bigger houses have more bedrooms and thus command a higher price. Let's look at that relationship.

# In[ ]:


#get the average price for houses along their number of bedrooms:
plt.figure(figsize=(10,6))
sns.barplot(x=houses.bedrooms, y=houses['price'])


# OK something strange here.  There is clearly a relationship between the number of bedrooms and the average price of a house. However, seems that a house with 9 bedrooms (!) sells for less than a house with 4 bedrooms... 
# 
# Also, some houses don't have any rooms?
# 
# Let's look at this in more detail.

# In[ ]:


# get a price breakdown for each bedroom group
bybedroom = houses.groupby(['bedrooms']).price.agg([len, min, max])


# In[ ]:


#problem #1 and #2 - 2 houses with 0 bedrooms, giant outlier at 3 bedrooms
bybedroom


# The table above provides an explanation for the price discrepancy we have seen. There is only one house with 9 bedrooms! This may be a house located far from the city, or the owner might have needed to sell it in a hurry. Whatever the circumstances, 1 house is not big enough a sample. We'll need to do something if we want to use the number of bedrooms as a predictor in our model.
# 
# The table above also highlighted 2 other problems with the data. 
# 1. Two houses have no bedroom!
# 2. Some houses have a price of zero
# 
# Let's look at this last problem in more detail.

# In[ ]:


# problem #3 - houses with null prices
houses_zero= houses[houses.price==0]
print('There are '+str(len(houses_zero))+' houses without a price')


# Out of 4600 houses in the sample, 49 don't have a price. It's not a lot, but this might confuse the model.
# 
# We're almost done with exploring the features. Let's look at the price distribution.

# In[ ]:


# problem #4 - house prices are not normal
sns.distplot(houses['price'], fit=norm)


# The price distribution is in blue, while the normal distribution is in black. Clearly, houses prices are not normal. This is not a problem per se, rather something to keep in mind.
# 
# So, to recap, we have 3 problems :
# 1. Houses with 0 bedroom
# 2. Giant outlier at almost $27M - 50 times the price of a normal house
# 3. 49 houses without a price
# 
# 
# We'll take the easy way out - remove them from our analysis.

# In[ ]:


# new dataframe without problem #1 #2 #3
houses_o = houses[(houses.price<2.5*10**7) & (houses.bedrooms>0) & (houses.price>0)].copy()


# Now, there is one other potential problem with our data. There are too few houses with more than 6 bedrooms. This is a problem if we want to use the number of bedrooms as a predictor of house price.
# 
# To fix this, we can simply group the houses with 7, 8 and 9 bedrooms with the houses featuring 6 bedrooms.

# In[ ]:


#recode houses with more than 6 bedrooms as 6 bedrooms
houses_o['bedrooms_recoded'] = houses_o['bedrooms'].replace([7,8,9],6)


# In[ ]:


houses_o['renovated_0_1'] = houses_o['yr_renovated']/houses_o['yr_renovated']
houses_o['renovated_0_1'] = houses_o['renovated_0_1'].fillna(0)


# OK we're done with the recoding.
# Let's get a nice Pearson correlation matrix going on

# In[ ]:


features = ['price','bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
       'floors', 'waterfront', 'view', 'condition', 'sqft_above',
       'sqft_basement', 'yr_built', 'yr_renovated']
mask = np.zeros_like(houses_o[features].corr(), dtype=np.bool) 
mask[np.triu_indices_from(mask)] = True 

f, ax = plt.subplots(figsize=(16, 12))
plt.title('Pearson Correlation Matrix',fontsize=25)

sns.heatmap(houses_o[features].corr(),linewidths=0.25,vmax=0.7,square=True,cmap="BuGn", #"BuGn_r" to reverse 
            linecolor='w',annot=True,annot_kws={"size":8},mask=mask,cbar_kws={"shrink": .9});


# OK, this can be a bit overwhelming, so let's focus on one element at a time. 
# 
# The first thing to look at is the first column. It tells us how correlated the features in the houses are to the house price.
# We can see that the most correlated feature is sqft_living, with a coefficent of 0.62. This makes sense - the higher the surface of the house, the higher the price.
# 
# The second most correlated feature is sqft_above, with 0.53. However, in our model **we cannot use both sqft_living and sqft_above**, and that's because these features are highly correlated - 0.88.
# 
# If we do use both of these features, our model won't be able to properly estimate the coefficients - it won't know whether the price is high because sqft_living is high or because sqft_above is high.
# 
# So, best practice is to select features that are highly correlated with house prices, but not correlated with each other.
# For now, we'll pick the following:
# 
# 1. bedrooms_recoded
# 2. floors
# 3. view
# 4. condition
# 5. renovated_0_1

# Let's go ahead and separate the price from our features:

# In[ ]:


# Move our features into the X DataFrame
X = houses_o.loc[:,['bedrooms_recoded', 'floors','view','condition','renovated_0_1']]

# Move our labels into the y DataFrame
y = houses_o.loc[:,['price']] 


# Next we need to separate our houses into train and test set

# In[ ]:


# separate y and X into train and test
X_train, X_test, y_train, y_test = train_test_split(
                                                    X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=42
                                                   )


# Finally, we use a multiple regression model on the train set to find out what is the impact of our predictor variables on price:

# In[ ]:


#train a basic multiple regression model and print out the coefficients
mod = sm.OLS(y_train, X_train)
res = mod.fit()
print(res.summary())


# We get a nice little novel above. I know this is a lot but it's necessary. 
# 
# We can see for each of our predictor variables, (*bedrooms_recoded, floors, view,condition, renovated_0_1*) there are several columns - *coef, std err, t,  P>|t|, [0.025 and 0.975]*. The column we need to check first is *P>|t|*. It tells us what is the probability that our coefficients are equal to zero, meaning our predictor variables do not have an impact on price.
# 
# Here we're lucky - most of these probabilities are zero, except for the condition of the house and whether it has been renovated or not.
# 
# The probability that *condition* and *renovated_0_1* are zero is 12.1% and 59.9%, respectively.
# 
# The next thing we can check is the coefficients themselves - what is the most important predictor variable? Here it seems that it's *view*, followed closely by *floors*. According to the model, if the house has a view, it will gain about $ 170k in value (1.712e+05 = 1,712*(10^5))
# 
# So this is obviously not a great model, but let's see what it does right and what it does wrong.

# In[ ]:


# Ask the model to predict prices in the train and test set based just on our predictor variables
lr = LinearRegression()
lr.fit(X_train,y_train)
test_pre = lr.predict(X_test)
train_pre = lr.predict(X_train)


# In[ ]:


# Now let's plot our predicted values on one axis and the real values on the other axis
plt.scatter(train_pre, y_train, c = "blue",  label = "Training data")
plt.scatter(test_pre, y_test, c = "black",  label = "Validation data")
plt.title("Linear regression")
plt.xlabel("Predicted values")
plt.ylabel("Real values")
plt.legend(loc = "upper right")
plt.plot([0.2*10**6, 0.25*10**7], [0.2*10**6, 0.25*10**7], c = "red")
plt.show()


# The x-axis represents the prices predicted by the model, while the y-axis shows the true price of these houses. Ideally, we would want houses to be grouped on the red line, meaning the estimated value and the true value of a house are very close. 
# 
# We are not there yet, but it isn't too bad! Our predictions follow the red line and there is no obvious train/test bias. Also, it seems that our predictions don't get worse when house prices increase, which is a good sign.
# 
# It seems our model has trouble with high-value houses though. It accurately forecasts the price of cheap and moderately priced properties, but when prices rise on the y axis, this is where we start to deviate from the ideal red line. The poster child for this prediction error is the obvious outlier sitting at 1.2x10^7 = a 12 million dollar house, where our model predicted less than $ 500 000!
# 
# Now let's compute the mean error.

# In[ ]:


#get the results from the regression in dataframe format
res = pd.DataFrame(data=train_pre, columns=['predicted values'])
#join with the actual prices
res = y_train.reset_index().join(res)
#join with the training dataset
resfin = res.join(X_train, on='index',lsuffix='_y')
# compute the actual prices, predicted prices and error
resfin['predprice']=res['predicted values']
resfin['actprice']=res['price']
resfin['error']=resfin['predprice']-resfin['actprice']


# In[ ]:


#get the results from the regression in dataframe format
res_test = pd.DataFrame(data=test_pre, columns=['predicted values'])
#join with the actual prices
res_test = y_test.reset_index().join(res_test)
#join with the training dataset
resfin_test = res_test.join(X_test, on='index',lsuffix='_y')
# compute the actual prices, predicted prices and error
resfin_test['predprice']=resfin_test['predicted values']
resfin_test['actprice']=resfin_test['price']
resfin_test['error']=resfin_test['predprice']-resfin_test['actprice']
resdf = pd.concat([resfin,resfin_test])


# In[ ]:


"The mean error of our model is ${:,.0f}".format(resfin_test['error'].mean())


# Mean error on the test set is close to  $ 14k, which means that the model tends to overestimate the value of houses.
# 
# Let's see what the shape of the errors looks like.
# 
# 

# In[ ]:


#plot the error
plt.figure(figsize=(15,8))
sns.distplot(resfin_test['error'], fit=norm)


# Next let's isolate the model's biggest mistakes to see if there is a pattern.

# In[ ]:


#standardize the errors
x_array = np.array(resfin_test['error'])
normalized_X = stats.zscore(x_array)


# In[ ]:


#let's get the normalized error back into our dataset
error_df = pd.DataFrame(data=normalized_X.T, columns=['normalized error'])
resfin2 = resfin_test.join(error_df)
resfin2['abs_norm_error'] = abs(resfin2['normalized error'])
#now let's select only the errors that are 2 standard deviations away from the mean
resfin2['massive underestimation'] = resfin2['normalized error']<-2 


# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(error_df, fit=norm)


# Great! Now that we have flagged our biggest mistakes, it will be easier to find out whether there is a pattern to them.

# In[ ]:


#how many big mistakes in our test dataset?
resfin2['massive underestimation'].value_counts()


# 50 houses are underestimated in our test data! How much is that in %?

# In[ ]:


"approximately {:.1%} of the test houses are massively underestimated".format(resfin2['massive underestimation'].values.sum()/len(resfin2))


# In[ ]:


plt.figure(figsize=(12,8))
plt.scatter(resfin2['predprice'], resfin2['actprice'], c = resfin2['massive underestimation'])
plt.plot([0.2*10**6, 1.75*10**6], [0.2*10**6, 1.75*10**6], c = "red")
plt.legend(loc = "upper left")


# Great! We've highlighted the underestimated houses in yellow. 

# In[ ]:


#Now let's explore - what kind of houses is the model particularly bad at estimating the price of?
pd.crosstab(resfin2['bedrooms_recoded'],resfin2['massive underestimation']).apply(lambda r: r/r.sum(), axis=1)


# OK so the model apparently has trouble with houses that feature 4 rooms or more. There may be something we're missing about these houses. Could be the location? After all, according to real estate agents, it's all about location!

# Let's visualise the price of a house and the zip code!

# In[ ]:


result = houses_o.groupby(["statezip"])['price'].aggregate(np.median).reset_index().sort_values('price', ascending=False)
plt.figure(figsize=(15,8))
chart = sns.barplot(
    x='statezip',
    y='price',
    data=houses_o,
    order = result['statezip'],
    estimator=np.median
    
    
)
chart.set_xticklabels(chart.get_xticklabels(), rotation=90)


# * We can see there are a handful of zipcodes that are way more expensive than the rest. Otherwise, statezip doesn't seem to play a huge role in the price of a house.
# 
# Let's select the 5 most expensive zips and create a new binary variable called 'posh zip':

# In[ ]:


houses_o['posh_zip'] = houses_o['statezip'].isin(['WA 98039','WA 98004','WA 98040','WA 98109']).astype(int)


# In[ ]:


# Move our features into the X DataFrame
X = houses_o.loc[:,['bedrooms_recoded', 'floors', 'condition','view','renovated_0_1', 'posh_zip']]

# Move our labels into the y DataFrame
y = houses_o.loc[:,['price']] 


# In[ ]:


# separate y and X into train and test
X_train, X_test, y_train, y_test = train_test_split(
                                                    X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=42
                                                   )


# In[ ]:


#train a basic multiple regression model and print out the coefficients
mod = sm.OLS(y_train, X_train)
res = mod.fit()
print(res.summary())


# OK, let's unpack this shall we?
# 
# What's reassuring is that condition and renovated_0_1 stay non significant, meaning they don't seem to contribute too much to the price of a house. However posh_zip does contribute and not in a small way! It is the most significant variable by a long shot AND it contributes the most to the price of a house! If you could move a house from an average neighborhood to a good neighborhood, it increases the house value by $ 700k, everything else being constant!
# 

# In[ ]:


# Ask the model to predict prices in the train and test set based just on our predictor variables
lr = LinearRegression()
lr.fit(X_train,y_train)
test_pre = lr.predict(X_test)
train_pre = lr.predict(X_train)


# In[ ]:


#get the results from the regression in dataframe format
res_test = pd.DataFrame(data=test_pre, columns=['predicted values'])
#join with the actual prices
res_test = y_test.reset_index().join(res_test)
#join with the training dataset
resfin_test = res_test.join(X_test, on='index',lsuffix='_y')
# compute the actual prices, predicted prices and error
resfin_test['predprice']=resfin_test['predicted values']
resfin_test['actprice']=resfin_test['price']
resfin_test['error']=resfin_test['predprice']-resfin_test['actprice']
resdf = pd.concat([resfin,resfin_test])


# In[ ]:


#standardize the errors
x_array = np.array(resfin_test['error'])
normalized_X = stats.zscore(x_array)


# In[ ]:


#let's get the normalized error back into our dataset
error_df = pd.DataFrame(data=normalized_X.T, columns=['normalized error'])
resfin2 = resfin_test.join(error_df)
resfin2['abs_norm_error'] = abs(resfin2['normalized error'])
#now let's select only the errors that are 2 standard deviations away from the mean
resfin2['massive underestimation'] = resfin2['normalized error']<-2 


# In[ ]:


plt.figure(figsize=(12,8))
plt.scatter(resfin2['predprice'], resfin2['actprice'], c = resfin2['massive underestimation'])
plt.plot([0.2*10**6, 1.75*10**6], [0.2*10**6, 1.75*10**6], c = "red")
plt.legend(loc = "upper left")


# Our model has one too many variables, but it could me missing something too. The Omnibus test tells us about the probability that the residuals are not normally distributed. The result of this test was zero which means they're not. If they're not normal, it means there is a pattern to the residuals that we're missing, and maybe this pattern could be explained by another variable in our dataset.
# 
# For now, let's just check the shape of our residuals:

# In[ ]:


#plot the residuals
plt.figure(figsize=(15,8))
sns.distplot(res.resid, fit=norm)


# Welp, something is definitely off here. there is a first peak around -1, and then we get the main peak around 1! That is surely not normal.

# In[ ]:


# Move our features into the X DataFrame
X = houses_o.loc[:,['sqft_living','condition', 'yr_built']]

# Move our labels into the y DataFrame
y = houses_o.loc[:,['price']] 


# In[ ]:


# separate y and X into train and test
X_train, X_test, y_train, y_test = train_test_split(
                                                    X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=42
                                                   )


# In[ ]:


#train a basic multiple regression model and print out the coefficients
mod = sm.OLS(y_train, X_train)
res = mod.fit()
print(res.summary())


# In[ ]:


#plot the residuals
plt.figure(figsize=(15,8))
sns.distplot(res.resid, fit=norm)


# 

# In[ ]:


#partial regression plots
fig = plt.figure(figsize=(12,8))
fig = sm.graphics.plot_partregress_grid(res, fig=fig)

