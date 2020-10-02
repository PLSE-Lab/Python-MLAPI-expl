#!/usr/bin/env python
# coding: utf-8

# # About the Notebook

# In this kernel, I will present a simple linear regression model combined with more advanced concepts, such as *data manipulation, data distribution, multicollinearity, polynomial relationships* etc., and present ways to deal with these problems.
# 
# I chose a simple data set to allow a better understanding of these concepts. Notice how applying these tools will improve the fitting  test results and the quality of your predictions.
# 
# I am sometimes using different terms that might imply the same thing. For example, features and variables, or target and response. It is important to know all of the terms that are being used by the community.
# 
# If you have any question or suggestions, please don't hesitate to comment!
# 
# **If you like this kernel, please don't hesitate to UPVOTE :)**

# ## The Importance of Predicting House Prices

# Are you planning on buying a house for investment, and wondering which house would be the best buy? Are you planning to renovate your house to increase its value on the market, but don't know where to invest the most to get the best results? Do you have a real-estate company that wants to give the best machine-learning based solutions to its customers? You are in the right place!
# 
# In this notebook, I will analize house sales in King County, WA, USA between 2014 to 2015.  
# 
# Let's start by looking the maps below: the top image is the King County region; the bottom image is downtown Seattle, the capital of Washington.

# <img align="center" src="https://imgur.com/O1ImtR8.png" width="700" hight="550" title="King County Region, WA, USA" />
# <img align="center" src="https://imgur.com/culbAe4.png" width="700" hight="250" title="Downtown Seattle" />

# # Explore the Data

# In[ ]:


import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from geopy import distance
from scipy.stats import skew
from statsmodels.stats.outliers_influence import variance_inflation_factor

plt.style.use('seaborn-whitegrid')
sns.set_style('whitegrid')
plt.rcParams["axes.labelsize"] = 15

from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.transform import log_cmap
from bokeh.models import ColumnDataSource, LogTicker, ColorBar #, HoverTool, CategoricalColorMapper, LogColorMapper
from bokeh.models.formatters import BasicTickFormatter, NumeralTickFormatter
import bokeh.palettes as bp

output_notebook()
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's read the data, look at the head of the table and the information about the features.

# In[ ]:


df = pd.read_csv('../input/kc_house_data.csv')
df.head()


# In[ ]:


df.info()


# The dataframe contains 19 house features, plus the price and the ID columns, along with 21613 observations.
# 
# The ID column doesn't contribute any insight into the data, and neither does the date, as all of the data is from 2014 to 2015. Let's drop those columns.

# In[ ]:


df.drop(['id', 'date'], axis=1, inplace=True)
df.columns


# # Data Visualization

# Let's have a look at the different features and their relation to the price of the house, starting with the discrete variables.

# In[ ]:


fig, axes = plt.subplots(3,2, figsize=(18,16))
for xcol, ax in zip(['floors', 'waterfront', 'view', 'condition',
                     'grade', 'bedrooms'], axes.flatten()):
    sns.boxplot(xcol, 'price', data=df, ax=ax)
    

fig = plt.figure(figsize=(16, 8))
sns.boxplot('bathrooms', 'price', data=df)

plt.tight_layout()


# We can see that some of the features have a linear relation to the target (*price*), but some of them have a non-linear relation, such as the *grade*, which looks more like an exponential relation to price. This seems a bit problematic, considering we are going to use a *linear* regression model. We will learn how to deal with this problem later in this kernel.
# 
# Let's have a look at the continuous variables and their relation to the target.

# In[ ]:


features_cont = ['sqft_living', 'sqft_lot', 'sqft_above','sqft_basement', 
                'sqft_living15', 'sqft_lot15']

fig, axes = plt.subplots(3,2, figsize=(14,14))

for xcol, ax in zip(features_cont, axes.flatten()):
    sns.scatterplot(xcol, 'price', data=df, ax=ax)

plt.tight_layout()


# Once again, we can see some features with a more linear relation to *price*, but some of the features reveal more complex relationships, like polynomial, exponential or even a square-root relation. We can see the complex relations clearly with the features such as *sqft_lot* and *sqft_lot15*.
# 
# Let's have a look at the price of houses according to their location in Seattle.

# In[ ]:


def lgn2x(a):
    return a * (np.pi/180) * 6378137

def lat2y(a):
    return np.log(np.tan(a * (np.pi/180)/2 + np.pi/4)) * 6378137


# project coordinates
df['x_coor'] = df['long'].apply(lambda row: lgn2x(row))
df['y_coor'] = df['lat'].apply(lambda row: lat2y(row))

# creating the map
output_file("tile.html")
xmin, xmax =  df['x_coor'].min(), df['x_coor'].max() 
ymin, ymax =  df['y_coor'].min(), df['y_coor'].max() 

# range bounds supplied in web mercator coordinates
map_kc = figure(x_range=(xmin, xmax), y_range=(ymin, ymax),
           x_axis_type="mercator", y_axis_type="mercator", title="House Price on King County, USA",
           plot_width=700, plot_height=500,)

map_kc.title.text_font_size = '16pt'
map_kc.add_tile(CARTODBPOSITRON)

source = ColumnDataSource({'x':df['x_coor'], 'y':df['y_coor'], 'z':df['price']})
colormapper = log_cmap('z', palette=bp.Inferno256, low=df['price'].min(), high=df['price'].max())

map_kc.circle(x ='x', y='y', source=source, color=colormapper)

color_bar = ColorBar(color_mapper=colormapper['transform'], width=18, location=(0,0), 
                     ticker=LogTicker(), label_standoff=12)

color_bar.formatter = NumeralTickFormatter(format='0,0')
# color_bar.formatter = BasicTickFormatter(precision=3)

map_kc.add_layout(color_bar, 'right')

show(map_kc)


# We can make two observations from this plot:
# 1. The northern part of King County region has a higher house prices.
# 2. The closer the house is to downtown Seattle, the price of the houses increases.
# 
# As for the first observation, we can assume that the *lat* feature is important in our model. We can check our assumption by using feature engineering methods, but since we don't have many features compared to the number of data points, we see better results using the entire data set, which negates the need for feature engineering. 
# 
# As for the second observation, we can create a new feature that measures the *distance* from each house to downtown Seattle. This feature is a *nonlinear combination* of the *lat* and *long* features, so it doesn't increase the multicollinearity* (see next session).
# 
# Let's add the *distance* feature (in km) and have a look at the head of the table.

# In[ ]:


location = tuple(map(tuple, df[['lat', 'long']].values))
# the distance of every house from downtowm seattle
seattle_dt = (47.6050, -122.3344)

df['distance'] = [distance.distance(seattle_dt, loc).km for loc in location]

# df.drop(['lat', 'long', 'x_coor', 'y_coor'], axis=1, inplace=True)
df.drop(['x_coor', 'y_coor'], axis=1, inplace=True)

df.head()


# # Multicollinearity and Data Manipulation

# Collinearity between variables can produce misleading results. While the \\(R^2\\) might not be affected by collinearity, the interpretation of the results are highly affected by it. The presence of collinearity can pose problems in
# the regression context, since it can be difficult to separate out the individual effects of collinear variables on the response. In other words, since two correlated variables tend to increase or decrease together, it can be difficult to determine how each one separately is associated with the response (in our case, the price of the house). This phenomena can completely change the coefficient values (and therefore the interpretation of their importance), and in some cases it can even change the sign of the coefficient value. 
# 
# Unfortunately, it is not enough to check the correlation matrix, as multicollinearity can occur between three or more variables, even when there is no indication of collinearity between two variables. A better way to assess multicollinearity is by computing the *[Variance Inflation Factor](http://www.statisticshowto.datasciencecentral.com/variance-inflation-factor/)* (\\(VIF\\)). As a rule of thumb, we would like to keep the \\(VIF\\) under 5, as  \\(VIF > 5\\) suggests medium multicollinearity, while \\(VIF > 10\\) suggests high multicollinearity.
# 
# Let's have a look at the \\(VIF\\) value of our variables.

# In[ ]:


def get_vif(data):
    
    X = data.iloc[:,1:]
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns

    return vif.round(1)

get_vif(df)


# You'll see that most of the variables suggest strong multicollinearity; some of the values even go to infinity. To solve this problem, we can create another variable, which will be a linear combination of two highly correlated variables. 
# 
# Computing a correlation heatmap will help us choose the features we would like to combine together. 

# In[ ]:


plt.figure(figsize=(16,12))
sns.heatmap(df.corr(), annot=True)


# As we can see above, we have high correlation between *sqft_living* and *sqft_above*. We can also see high correlation between *sqft_living15* and both *sqft_living* and *sqft_above*. 
# 
# Let's create a new variable called *sqft* that will be a linear combination of the three predictors, and will replace these predictors. We can do the same with *sqft_lot* and *sqft_lot15*, creating a new variable called *sqft_lot_comb*. 
# 
# Let's have a look at the new table's head.

# In[ ]:


df['sqft'] = df['sqft_living'] + df['sqft_above'] + df['sqft_living15']
df.drop(['sqft_living', 'sqft_above', 'sqft_living15'], axis=1, inplace=True)

df['sqft_lot_comb'] = df['sqft_lot'] + df['sqft_lot15']
df.drop(['sqft_lot', 'sqft_lot15'], axis=1, inplace=True)

df.head()


# Let's see the affect on the \\(VIF\\) table.

# In[ ]:


get_vif(df)


# We can see a big improvement: we no longer have infinite \\(VIF\\) values!  However, we still have some more data manipulation to do for better results. For example, *lat, long, zipcode* and *yr_built* have high \\(VIF\\) values. 
# 
# Let's have a look at the data distribution and deal with the polynomial relationships we obsereved earlier between the variables and the target.

# # Data Distributuion and Polynomial Relationships

# In the plot below, you can see the data distribution of the target. Notice that the target has a right-skewed distribution, meaning, it has a "tail" on the right side. Skewing the data by log-transformation will transform the right-skewed distribution to a normal distribution.
# 
# To be clear, there is no need for the target to be normally distributed in order to fit a linear regression model. The normallity assumption in linear regression refers to the error terms between the target and the predicted values (\\(\epsilon\\)), and not the distribution of the target itself. 
# 
# However, a log transformation can sometimes solve multiple problems simultaneously: it will linearize some of the polynomial relationships, and help us create a more flexible model that allows [non-linear relashionship with the target](http://stats.stackexchange.com/questions/107610/what-is-the-reason-the-log-transformation-is-used-with-right-skewed-distribution) (see Bill's answer, poin 2). It will also [reduce outlier influences](https://heartbeat.fritz.ai/how-to-make-your-machine-learning-models-robust-to-outliers-44d404067d07). 
# 
# Plus, we get normally distributed data, which is always nice to have, and it's important for some statistical hypothesis tests. 
# 
# We will use the normality assumption in the next section, when we will carry on with the high multicollinearity issue.

# In[ ]:


fig = plt.figure(figsize=(11,5))
fig = sns.distplot(df['price'])
fig.set(yticks=[]);


# Now, let's look at the skewness factor of the target and the features. 
# 
# In order to be normally distributed, the skewness should be zero. A positive skewness is a right-skewed data. 

# In[ ]:


# computing skewness factor
skewness = df.apply(lambda x: skew(x))
skewness


# Let's do a log transformation on the data where *skewness*  \\(> 0.75\\), and have a look on the affect on the *price* distribution.

# In[ ]:


# converting longtitude to positive values to enable us using the log function on all data
# this operation doesn't affect results, as all the whole 'long' column is negative
df['long'] = abs(df['long'])

skewed = skewness[skewness > 0.75].index

df[skewed] = np.log1p(df[skewed])

# plot the new target ditribution
fig = plt.figure(figsize=(11,5))
fig = sns.distplot(df['price'])
fig.set(yticks=[]);


# We can see that the *price* is now normally distributed.
# 
# Let's have a look at some of the predictors and their relation to the target.

# In[ ]:


fig, axes = plt.subplots(2,2, figsize=(16,10))

sns.scatterplot('sqft', 'price', data=df, ax=axes[0,0])
sns.scatterplot('sqft_lot_comb', 'price', data=df, ax=axes[0,1])
sns.boxplot('bedrooms', 'price', data=df, ax=axes[1,0])
sns.boxplot('grade', 'price', data=df, ax=axes[1,1])
axes[1,0].set_xticks([])
axes[1,1].set_xticks([])

plt.tight_layout()


# Above we can see the change in the relationship between some of the features and the target. We also see that what once was a nonlinear relation between the feature to the *price* now have a more linear relationship to it. Moreover, we've reduced the outliers' affect. 
# 
# This dramatically increases the accuracy of our model. 
# 
# Notice that it has no affect on the multicollinearity.

# In[ ]:


get_vif(df)


# # Data Standardization 

# Going back dealing with the high  \\(VIF\\), we can use [data standardization to reduce multicollinearity](https://statisticsbyjim.com/regression/multicollinearity-in-regression-analysis/). In fact, only by centering the data, we will lower the \\(VIF\\) values (the scaling part is a matter of preference). 
# 
# Under the assumption that the data has a symetric distribution (like normal distribution), the correlation created by interaction terms will be zero. For the full mathematical explanation, see [this link](https://psychometroscar.com/why-does-centering-in-linear-regression-reduces-multicollinearity/).

# Let's see the affect of standardizing the data on the \\(VIF\\).

# In[ ]:


# Standardizing the data
df = (df - df.mean()) / df.std()

get_vif(df)


# As you can see, standardizing the data has a huge affect on the \\(VIF\\) values. 
# 
# Now we are ready to fit the model and get predictions!

# # House Price Prediction

# Now, we will predict house prices using a simple linear regression and a k-fold cross-validation. The fitted model will be tested with the R-squared adjusted test, so it will not be affected by the number of features I chose to use in the model (like in the R-squared test).

# In[ ]:


def split_kfold(folds, i):    
    train = folds.copy() 
    test = folds[i]
    del train[i]
    train = np.concatenate(train, axis=0)
    d = train.shape[1]-1
    x_train, y_train = train[:, :d], train[:, d]
    x_test, y_test = test[:, :d], test[:, d]
    
    return x_train, x_test, y_train, y_test


def get_error(Y, Yhat):
    N = len(Y)   
    d1 = Y - Yhat
    d2 = Y - Y.mean()
    r2 = 1 - (d1.dot(d1) / d2.dot(d2))
    r2_adj = 1 - (1 - r2)*((N - 1) / (N - D - 1))
    mse = d1.dot(d1) / N
    return r2_adj, mse


def fit_kfold(X, Y, X_test, Y_test):
    w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
    Yhat = X.dot(w)
    Yhat_test =  X_test.dot(w)
    r2_test, mse_test = get_error(Y_test, Yhat_test)
    
    return r2_test, w


# In[ ]:


# df_array = df[features].values
X = df.iloc[:,1:]
Y = df.iloc[:,0]

df_array = np.c_[X.values, Y.values]
k = 7
D = X.shape[1]
folds = np.array_split(df_array, k)

r2_test = []
coef = []

for i in range(k):
    x_train, x_test, y_train, y_test = split_kfold(folds, i)
    # prepare the array
    x_train = np.c_[np.ones(x_train.shape[0]), x_train]
    x_test = np.c_[np.ones(x_test.shape[0]), x_test]
    
    r2_test_temp, w = fit_kfold(x_train, y_train, x_test, y_test)
    r2_test.append(r2_test_temp)
    coef.append(w)
    
r2_test_kfold = sum(r2_test) / len(r2_test)
coef = np.sum(coef, axis=0) / len(coef)

indx = list(df.columns)
indx[0] = 'bias'
coef = pd.DataFrame(coef, index=indx, columns=['coef'])

print('Using  k-fold cross-validation where k = ', k,':')
print('R2_adjusted of the test data, using a simple linear regression, is: ', r2_test_kfold)


# Notice that the \\(R^2\\) is higher than the everage \\(R^2\\) from other notebooks that used a linear regression model. 
# 
# Let's have a look at the coefficients.

# In[ ]:


coef.reindex(coef['coef'].abs().sort_values(ascending=False).index)


# Above you can see the coefficients sorted by their importance. We can see that the features *sqft, distance, grade* and *lat* have the greatest affect on the *price*. The negative coefficients imply that the increase of these features lowers the price of the house. For example, as the houses' distance from downtwon Seattle increases, the price of the house decreases.
# 
# Because we used log-transformation on the data and standardized it, it cannot be interpreted using traditional or straightforward methods. More information on how to interpret the coefficients will be added soon.

# # Summary

# ### We fit a simple linear model and got a great \\(R^2\\) result! 
# **How did we do it?**
# 1. Data manipulation: we created the *distance* feature.
# 2. Log-transformation of the data: created a more flexible model that allows non-linear relationships with the target.
# 
# ### Lowered multicollinearity and got statistically significant results! 
# **How did we do it?**
# 1. Created new features to replace highly correlated ones with their linear combination. 
# 2. Standardizing the data to reduce correlations caused by interaction terms.

# **I hope you enjoy my notebook! If you did, please UPVOTE!**
# 
# If you have a comment or ideas for improvement, please leave a comment below.
# 
# See you on my next kernel :)
