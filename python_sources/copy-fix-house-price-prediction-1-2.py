#!/usr/bin/env python
# coding: utf-8

# This kernel is copy of below kernels. And I have fixed some parts which didn't operate.

# https://www.kaggle.com/harlfoxem/house-price-prediction-part-1  
# https://www.kaggle.com/kabure/predicting-house-prices-xgb-rf-bagging-reg-pipe

# # House price prediction using multiple regression analysis
#   This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015. It's great dataset for evaluating simple regression models.

# # Part 1: Exploratory Data Analysis
# The following notebook presents a thought process of predicting a **continuous variable** through Machine Learning methods. More specifically, we want to predict house price based on multiple features using regression analysis.
# 
# As a example, we will use a dataset of house sales in King County, where Seattle is located.
# 
# In this notebook, we will first apply some Exploratory Data Analysis (EDA) techniques to summarize the main characteristic of the dataset.

# ## 1. Preparation

# ### 1.1 Load the libraries

# In[ ]:


import numpy as np # NumPy is the fundamental package for scientific computing

import pandas as pd # Pandas is an easy-to-use data structures and data analysis tools
pd.set_option('display.max_columns', None) # To display all columns

import matplotlib.pyplot as plt # Matplotlib is a python 2D plotting library
get_ipython().run_line_magic('matplotlib', 'inline')
# A magic command that tells matplotlib to render figures as static images in the Notebook.

import seaborn as sns # Seaborn is a visualization library based on matplotlib (attractive statistical graphics).
sns.set_style('whitegrid') # One of the five seaborn themes
import warnings
warnings.filterwarnings('ignore') # To ignore some of seaborn warning msg

from scipy import stats, linalg

from matplotlib import rcParams
import scipy.stats as st

import folium # for map visualization
from folium import plugins


# ### 1.2 Load the dataset
# Let's load the data from CSV file using pandas and convert some columns to category type(for better summarization)

# In[ ]:


data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv", parse_dates=['date'])


# In[ ]:


# Why are the following features converted to category type?
data['waterfront'] = data['waterfront'].astype('category',ordered=True)
data['view'] = data['view'].astype('category',ordered=True)
data['condition'] = data['condition'].astype('category',ordered=True)
data['grade'] = data['grade'].astype('category',ordered=False) # Why are these ordered 'False'?
data['zipcode'] = data['zipcode'].astype(str)
data.head(2) # Show the first 2 lines


# In[ ]:


data = data.sort_index()
data.head(2)


# In[ ]:


print(data.shape)
print(data.nunique())


# In[ ]:


print(data.info())


# In[ ]:


data.head()


# In[ ]:


# Knowing the Price variable
f, ax = plt.subplots(1, 2, figsize = (12, 6))
sns.distplot(data['price'], ax=ax[0])
ax[0].set_title('Price Distribution')
plt.xlim()

sns.scatterplot(range(data.shape[0]), data['price'].sort_values(), ax=ax[1], marker="x")
ax[1].set_title('Price Curve Distribution', fontsize=15)

plt.show()


# ## 2. Descriptive statistics.
# The initial dimension of the dataset.

# In[ ]:


data.shape


# Let's summarize the main statistics of each parameters.

# In[ ]:


data.describe(include='all')


# ## 3. Setting the context (map visualization)
# Before we dive into exploring the data, we'll want to set the context of the analysis. One good way to do this is with exploratory charts or maps. In this case, we'll map out the positions of the houes, which will help us understand the problem we're exploring.
# 
# In the below code, we:
# * Setup a map centered on king County.
# * Add a marker to the map for each house sold in the area.
# * Display the map.

# In[ ]:


houses_map = folium.Map(location = [data['lat'].mean(), data['long'].mean()], zoom_start = 10)
lat_long_data = data[['lat', 'long']].values.tolist()
h_cluster = folium.plugins.FastMarkerCluster(lat_long_data).add_to(houses_map)

houses_map


# The map is helpful but it's hard to see where the houses our dataset are located. Instead, we could make a heatmap:

# In[ ]:


houses_heatmap = folium.Map(location = [data['lat'].mean(), data['long'].mean()], zoom_start=9)
houses_heatmap.add_children(plugins.HeatMap([[row['lat'], row['long']] for name, row in data.iterrows()]))
houses_heatmap


# Heatmaps are good for mapping out gradients, but we'll want something with more structure to plot out differences in house sale accross the county. Zip codes are a good way to visualize this information.
# 
# We could, for example, compute the mean house price by zip code, then plot this out on a maps. In the below code, we 'll
# * Group the dataframe by zipcode.
# * Compute the average price of each column
# * add a column with the total number of observations ( i.e., house sales ) per zipcode.

# In[ ]:


zipcode_data = data.groupby('zipcode').aggregate(np.mean)


# In[ ]:


zipcode_data.reset_index(inplace=True)


# In[ ]:


data['count'] = 1
count_house_zipcode = data.groupby('zipcode').sum()
count_house_zipcode.reset_index(inplace=True)
count_house_zipcode = count_house_zipcode[['zipcode', 'count']]
data.drop(['count'], axis = 1, inplace=True)


# In[ ]:


zipcode_data = zipcode_data.join(count_house_zipcode.set_index('zipcode'), on='zipcode')


# In[ ]:


zipcode_data.head()


# We'll now be able to plot the average value of a specific attribute for each zip code. In order to do this, we'll read data in GeoJSON format to get the shape of each zip code, then match each zip code shpe with the attribuge score. Let's first create a function.

# In[ ]:


def show_zipcode_map(col):
    geo_path = '../input/house-prices-data/zipcode_king_county.geojson'
    zipcode = folium.Map(location=[data['lat'].mean(), data['long'].mean()], zoom_start=9)
    zipcode.choropleth(
        geo_data=geo_path, 
        data=zipcode_data, 
        columns=['zipcode', col],
        key_on='feature.properties.ZCTA5CE10',
        fill_color='OrRd', 
        fill_opacity=0.6, 
        line_opacity=0.2)
#     zipcode.save(col + '.html')
    return zipcode


# In[ ]:


show_zipcode_map('count')


# The map helps us understand a few things about the dataset.
# First, we can see that we don't have data for every zip code in the county. This is especially true for the inner suburbs of Seattle.
# Second, some zipcodes have a lot more house sales recorded than others. The number of observations range from ~50 to ~ 600. Let's show a few more maps:

# In[ ]:


show_zipcode_map('price')


# In[ ]:


show_zipcode_map('sqft_living')


# In[ ]:


show_zipcode_map('yr_built')


# We can see that on average, the houses on the eastern suburbs of Seattle are more expensive. They are also bigger in sqft. The houses close to the metropolitan of Seattle are relatively old compare to the houses in the rural area.

# ## 3. The Output Variable
# Now that we've set the context by plotting out where the houses in our dataset are located, we can move into exploring different angles for our regression analysis.
# 
# Let's first display the distribution of the target column 'price' using a boxplot.

# In[ ]:


plt.figure(figsize=(12,4))
sns.boxplot(x = 'price', data=data, orient='h', width=0.8, fliersize=3, showmeans=True)

plt.show()


# There seems to be a lot of outliers at the top of the distribution, with a few houses above the 5,000,000$ value.
# 
# If we ignore outliers, the range is illustrated by the distance between the opposite ends of the whiskers(1.5 IQR) - about 1,000,000$ here. 
# Also, we can see that the right whisker is slightly longer than the left whisker and that the median line is gravitating towords the left of the box. 
# 
# **The distribution is therefore slightly skewed to the right.**

# ## 4. Associations and Correlations between Variables
# 
# Let's analyze now the relationship between the independent variables available in the dataset and the dependent variable that we are trying to predict (i.e. price). These analysis should provide smoe interesting insights for our regression models.
# 
# We'll be using scatterplots and correlations coefficents (e.g Pearson, Spearman) to explore potential associations between the variables.

# ### 4.1 Continuous Variables.
# For example, let's analyze the relationship between the square footage of a house (sqft_living) and it selling price. Since the two variables are measured on a continuous scale, we can use Pearson's coefficient 'r' to measures the strength and direction of the relationship.

# In[ ]:


# A joint plot is used to visualize the bivariate distribution.
sns.jointplot(x='sqft_living', y='price', data=data, kind='reg', size=7)
plt.show()


# In[ ]:


print("PearsonR : ", data.corr(method='pearson')['sqft_living']['price'])


# There is a clear linear association between the variable (r = 0.7), indicating a strong positive relatioinship.   
# sqft_living shoud be a good predicator of house price, (note: sqft_living distribution is also skewed to the right)

# Let's do the same with the 7 remaining continuous variables:
# * sqft_lot
# * sqft_above( i.e., sqft_above = sqft_living - sqft_basement)
# * sqft_basement
# * sqft_living15, the average house square footage of the 15 closest neighbors
# * sqft_lot15, the average lot square footage of the 15 closest neighbors
# * yr_built
# * yr_renovated
# * lat
# * long

# In[ ]:


corr_price = data.corr(method='pearson')['price']


# In[ ]:


sns.jointplot(x='sqft_lot', y='price', data=data, kind='reg', size=5)
sns.jointplot(x='sqft_above', y='price', data=data, kind='reg', size=5)
sns.jointplot(x='sqft_basement', y='price', data=data, kind='reg', size=5)
plt.show()

print(corr_price['sqft_lot'])
print(corr_price['sqft_above'])
print(corr_price['sqft_basement'])


# In[ ]:


sns.jointplot(x='sqft_living15', y='price', data=data, kind='reg', size=5)
sns.jointplot(x='sqft_lot15', y='price', data=data, kind='reg', size=5)
sns.jointplot(x='yr_built', y='price', data=data, kind='reg', size=5)
plt.show()

print(corr_price['sqft_living15'])
print(corr_price['sqft_lot15'])
print(corr_price['yr_built'])


# In[ ]:


sns.jointplot(x='yr_renovated', y='price', data=data, kind='reg', size=5)
sns.jointplot(x='lat', y='price', data=data, kind='reg', size=5)
sns.jointplot(x='long', y='price', data=data, kind='reg', size=5)
plt.show()

print(corr_price['yr_renovated'])
print(corr_price['lat'])
print(corr_price['long'])


# sqft_lot, sqft_lot15 and yr_built seem to be poorly related to price.
# We can see that there is a lot of zeros in the sqft_basement distribution (i.e. No basement). Similarly, there is a lot of zeros in the yr_renovated variable.
# 
# Let's rerun the association tests for these two variables without the zeros.

# In[ ]:


# Create 2 new columns for the analysis
data['sqft_basement2'] = data['sqft_basement'].apply(lambda x : x if x > 0 else None)
data['yr_renovated2'] = data['yr_renovated'].apply(lambda x : x if x > 0 else None)

# Show the new plots with paerson correlation
sns.jointplot(x="sqft_basement2", y="price", data=data, kind = 'reg', dropna=True, size = 5)
sns.jointplot(x="yr_renovated2", y="price", data=data, kind = 'reg', dropna=True, size = 5)
plt.show()


# The house price is moderately correlated with the size of the basement ( if basement present ).  
# There is also a small correlation with the yaer of the renovation(if renovated).  
# 
# It might be more interesting for our analysis to classify basement and renovation as dichotomous variables(e.g. 0 for no basement, 1 for basement present). Let's create two new columns in our dataset.

# In[ ]:


data['basement_present'] = data['sqft_basement'].apply(lambda x: 1 if x > 0 else 0)
data['basement_present'] = data['basement_present'].astype('category', ordered = False)

data['renovated'] = data['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
data['renovated'] = data['renovated'].astype('category', ordered = False)


# We will analyse these new variable as categorical ( see in few cells below ).
# 
# But first, let's go back to the plots above and the two variables: sqft_above and sqft_livint15. They seem to be strongly related to price.
# Let's anlayze their associations (along with sqft_living) using the pairgrid() function from seagborn. This function creates a matrix of axes and shows the relationship for each par of the selected variables.
# 
# We will draw the univariate distribution of each variable on the diagonal Axes, and the bivariate distributions using scatterplots on the upper diagonal and kernel density estimation on the lower diagonal. We will create a function to display the pearson coefficeint of each pair.

# In[ ]:


# define a function to display pearson coeeficients on the lower graphs
def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("Pearson R = {:.2f}".format(r), xy=(.1, .9), xycoords=ax.transAxes)

g = sns.PairGrid(data=data, vars=['sqft_living', 'sqft_living15', 'sqft_above'], size=3.5) #define pairgrid
g.map_upper(plt.scatter)
g.map_diag(sns.distplot)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_lower(corrfunc)
plt.show()


# As envisaged, there is a strong positive relationship between the 3 variables( r> 0.7).   
# It was kind of obvious for sqft_above which is equal to sqft_living - sqft_baement. So  we know that they both have an impact on price.  
# 
# For sqft_living15 however, we are not sure if the relationship with house price is actually due to the average square footage of the 15th closest houses. This is because of the high correlation between sqft_living15 and sqft_living.  
# 
# To assess the true relationship between price and sqft_living15, we can use the Pearson Partial Correlation test. The correlation can assess the association between two continuous variables whilst controlling for the effect of other continuous variables called covariates. In our example, we will test the relationship between price and sqft_living15 using sqft_living as covariate.

# In[ ]:


# a Function to returns the sample linear partial correlation coefficient between pairs of variables in C controlling
# for the remaining variables in C (clone of Matlab's partial corr)
def partial_corr(C):
    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]
            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)            
            corr = stats.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr
    return P_corr

# Convert pandas dataframe to a numpy array using only three columns
partial_corr_array = data.as_matrix(columns = ['price', 'sqft_living', 'sqft_living15'])

# Calculate the partial correlation coefficients
partial_corr(partial_corr_array)


# We can see now that the average house size of the surrounding houses has no effect on the sell price when controlling for the size of the house( r = 0.06 )

# ### 4.2 Categorical Variables
# Let's now analyze the relationship between house price and the categorical variables.
# 
# As a first example, we will try to assess if having a waterfront is related to a higher house value. waterfront is a dichotomous variable with underlying continuous distribution (having a waterfront is better than not having a waterfront). We  can use a point-biserial correlation to highlight the relationship between the two variable.

# In[ ]:


# Let's show bosxplot first
plt.figure(figsize=(12,4))
sns.boxplot(y = 'waterfront', x = 'price', data=data, width = 0.8, orient='h'
           ,showmeans=True, fliersize=3)
plt.show()


# In[ ]:


# Calculate the correlation coefficient
r, p = stats.pointbiserialr(data['waterfront'], data['price'])
print('point biserial correlation r is %s with p = %s'%(r, p))


# Comments:  
# * The no waterfront boxplot is comparatively short. This suggests that overall, house prices in this group are very close to each other.
# * The waterfront boxplot is comparatively tall. This suggests that house prices differ greatly in this group.
# * There is obvious shape differences between the two distributions, Suggesting a higher sell price, in general, for houses with a waterfront. This is validated by a positive value of the point-biserial correlation.
# * The correlation if however small (r<0.3). Note that we haven't test here the 3 main assumptions of the point-biserial correlation and can't rely too much on the result.   
#     1. There should be no significant outliers in the two groups of the dichotomous variable in terms of the continuous variable
#     2. There should be homogeneity of variances.
#     3. The continuous variable should be approximately normally distributed for each group of the dichotomous variable.  
#     
# We can run the same test on the basement_present variable and whether or not the house had been renovated in the past.

# In[ ]:


# basement_present variable
plt.figure(figsize=(12,4))
sns.boxplot(y = 'basement_present', x='price', data=data, width=0.8, orient='h', 
            showmeans=True, fliersize=3)
plt.show()
r, p = stats.pointbiserialr(data['basement_present'], data['price'])
print ('point biserial correlation r between price and basement_present is %s with p = %s' %(r,p))


# In[ ]:


# renovatd variable
plt.figure(figsize=(12,4))
sns.boxplot(y = 'renovated', x = 'price', data = data,width = 0.8,orient = 'h', showmeans = True, fliersize = 3)
plt.show()
r, p = stats.pointbiserialr(data['renovated'], data['price'])
print('point biserial correlation r between price and renovated is %s with p = %s' %(r,p))


# Associations exist but they are fairly small (0.1 < r < 0.3)
# 
# Let's move on to our ordinal variables and asses their association with house price.
# We will show the distribution of the categories of each variable using boxplots.

# In[ ]:


fig, ax = plt.subplots(6, figsize=(12, 40))
sns.boxplot(y='bedrooms', x='price', data=data, width=0.8, orient='h',
           showmeans=True, fliersize=3, ax=ax[0])
sns.boxplot(y='bathrooms', x='price', data=data, width=0.8, orient='h',
           showmeans=True, fliersize=3, ax=ax[1])
sns.boxplot(y='floors', x='price', data=data, width=0.8, orient='h',
           showmeans=True, fliersize=3, ax=ax[2])
sns.boxplot(y='view', x='price', data=data, width=0.8, orient='h',
           showmeans=True, fliersize=3, ax=ax[3])
sns.boxplot(y='condition', x='price', data=data, width=0.8, orient='h',
           showmeans=True, fliersize=3, ax=ax[4])
sns.boxplot(y='grade', x='price', data=data, width=0.8, orient='h',
           showmeans=True, fliersize=3, ax=ax[5])
plt.show()


# As expeted, they all seem to be related to the house price.  
# We can use the Spearman's rank-order correlation to measure the strength and direction of the relationship between house price and these variables.

# In[ ]:


r, p = stats.spearmanr(data['bedrooms'], data['price'])
print ('spearman correlation r between price and bedrooms is %s with p = %s' %(r,p))
r, p = stats.spearmanr(data['bathrooms'], data['price'])
print ('spearman correlation r between price and bathrooms is %s with p = %s' %(r,p))
r, p = stats.spearmanr(data['floors'], data['price'])
print ('spearman correlation r between price and floors is %s with p = %s' %(r,p))
r, p = stats.spearmanr(data['view'], data['price'])
print ('spearman correlation r between price and view is %s with p = %s' %(r,p))
r, p = stats.spearmanr(data['condition'], data['price'])
print ('spearman correlation r between price and condition is %s with p = %s' %(r,p))
r, p = stats.spearmanr(data['grade'], data['price'])
print ('spearman correlation r between price and grade is %s with p = %s' %(r,p))


# There is indeed associations between these variables and the house price (except for condition).
# grade seems to be the best indicator.

# In this post, we analyzed the relationship between the output variable (house price) and the dependent variables in our dataset.
# More specifically, we highlighted that:
# * sqft_living, sqft_above and sqft_basement were moderately/strongly associated with price. Pearson r was equal to 0.70, 0.61 and 0.41, respectively. The 3 variables were also strongly related to each other as sqft_living = sqft_above and sqft_basement.
# * sqft_living15, the average house square footage of the 15 closet neighbors, was also strongly related to price(r=0.59). However, when controlling for sqft_living, the relationship disappeared(r = 0.06).
# * sqft_lot, sqft_lot15 (average lot size of the 15 closet houses) and yr_built were poorly related to price.
# * The three dichotomous variables ( waterfront, basement_present, renovated ) were associated with price. The association were small ( r < 0.3 )
# * Five of the ordinal parameters (bedrooms, bathrooms, floors, views, grade) were also moderately to strongly to strongly associated with price.
# 
# 
# 

# Our multiple regression analysis models in Part 2 will be built on these results.

# # Part 2: Regression Models

# The following notebook presents a thought process of predicting a continuous variable through Machine Learning methods. More specifically, we want to predict house price based on multiple featrues using regression analysis.
# 
# As an example, we will use a dataset of house sales in King County, where Seattle is located.
# 
# In the first part of the analysis, we set up the context using map visualization, and highlighted the association between the variables in our dataset.
# 
# This is, for example, a map of King County showing the average house price per zipcode. We can see the disparities between the different zipcodes. The location of houses should play an important role in our regression model.
# 
# In this second notebook we will apply multiple regression models. We will talk about model complexity and how we can select the best predictive model using a validation set or cross-validation techniques.

# ## 1. Preparation
# As in Part 1, Let's first load the libraries

# In[ ]:


# import numpy as np # NumPy is the fundamental package for scientific computing

# import pandas as pd # Pandas is an easy-to-use data structures and data analysis tools
# pd.set_option('display.max_columns', None) # To display all columns

# import matplotlib.pyplot as plt # Matplotlib is a python 2D plotting library
# %matplotlib inline 
# # A magic command that tells matplotlib to render figures as static images in the Notebook.

# import seaborn as sns # Seaborn is a visualization library based on matplotlib (attractive statistical graphics).
# sns.set_style('whitegrid') # One of the five seaborn themes
# import warnings
# warnings.filterwarnings('ignore') # To ignore some of seaborn warning msg

# from scipy import stats

from sklearn import linear_model # Scikit learn library that implements generalized linear models
from sklearn import neighbors # provides functionality for unsupervised and supervised neighbors-based learning methods
from sklearn.metrics import mean_squared_error # Mean squared error regression loss
from sklearn import preprocessing # provides functions and classes to change raw feature vectors

from math import log


# In[ ]:


data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv", parse_dates = ['date']) # load the data into a pandas dataframe
data.head(2) # Show the first 2 lines


# ### Data Cleaning
# Let's reduce the dataset by dropping colums that won't be used during the analysis.

# In[ ]:


data.drop(['id', 'date'], axis=1, inplace=True)


# ### Data Transformation
# Following the correlation analysis in Part 1, let's create some new variables in our dataset.

# In[ ]:


# Indicate whether there is a basement or not
data['basement_present'] = data['sqft_basement'].apply(lambda x : 1 if x > 0 else 0)
# 1 if the house has been renovated
data['renovated'] = data['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)


# ### Encode categorical variable using dummies
# A Dummy variable is an artificial variable created to represent an attribute with two or more distinct categories/levels. In this example, we will analyze bedrooms and bathrooms as continuous and therefore will encode following:
# * floors
# * view
# * condition
# * grade

# In[ ]:


categorical_cols = ['floors', 'view', 'condition', 'grade']

for col in categorical_cols:
    dummies = pd.get_dummies(data[col], drop_first=False)
    dummies = dummies.add_prefix("{}#".format(col))
    data.drop(col, axis=1, inplace=True)
    data = data.join(dummies)


# In[ ]:


data.head()


# We saw that zipcodes are also related to price. However, encoded all zipcodes will add 70 dummies variables.  
# Instead, we will only encode the 6 most expensive zipcodes as shown in the map.

# In[ ]:


dummies_zipcodes = pd.get_dummies(data['zipcode'], drop_first=False)
dummies_zipcodes.reset_index(inplace=True)
dummies_zipcodes = dummies_zipcodes.add_prefix("{}#".format('zipcode'))
dummies_zipcodes = dummies_zipcodes[['zipcode#98004','zipcode#98102','zipcode#98109','zipcode#98112','zipcode#98039','zipcode#98040']]
data.drop('zipcode', axis=1, inplace=True)
data = data.join(dummies_zipcodes)

data.dtypes


# ### Split the data
# We will split the dataframe into training and testing data using a 80% / 20% ratio.

# In[ ]:


from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(data, train_size=0.8, random_state=10)


# ## 2. Regression Models
# In this section, we will train numerous regerssion models on the train data (e.g., simple linear regression, lasso, nearest neighbor) and evaluate their performance using Root Mean Squared Error (RMSE) on the test data.

# ### 2.1 Simple Linear Regression
# Let's first predict house prices using simple (one input) linear regression.

# In[ ]:


# A function that take one input of the dataset and return the RMSE (of the test data), and the intercept and coefficent
def simple_linear_model(train, test, input_feature):
    regr = linear_model.LinearRegression() # Create a linear regression object
    # Train the model
    regr.fit(train.as_matrix(columns=[input_feature]), train.as_matrix(columns=['price']))
    
    RMSE = mean_squared_error(test.as_matrix(columns=['price']), 
                              regr.predict(test.as_matrix(columns=[input_feature])))**0.5
    
    return RMSE, regr.intercept_[0], regr.coef_[0][0]


# In[ ]:


RMSE, w0, w1 = simple_linear_model(train_data, test_data, 'sqft_living')
print ('RMSE for sqft_living is: %s ' %RMSE)
print ('intercept is: %s' %w0)
print ('coefficient is: %s' %w1)


# Similarly, we can run the same test on all the features in the dataset and assess which one would be the best estimator of house price using just a single linear regression model.

# In[ ]:


input_list = data.columns.values.tolist() # list of column name
input_list.remove('price')
simple_linear_result = pd.DataFrame(columns=['feature', 'RMSE', 'intercept', 'coefficient'])

# loop that calculate the RMSE of the test data for each input
for col in input_list:
    RMSE, w1, w0 = simple_linear_model(train_data, test_data, col)
    simple_linear_result = simple_linear_result.append({'feature':col, 'RMSE':RMSE, 'intercept':w0, 'coefficient': w1}, ignore_index=True)
    
simple_linear_result.sort_values('RMSE').head(10)
    


# When using simple linear regression, sqft_living provides the smallest test error estimate of house price for the dataset considered.

# ### 2.2 Multiple Regression  
# Now let's try to predict price using multiple features.  
# We can modify the simple linear regression function above to take multiple features as input.

# In[ ]:


# A function that take multiple features as input and return the RMSE (of the test data),
# and the intercept and coefficients.
def multiple_regression_model(train ,test, input_features):
    regr = linear_model.LinearRegression()
    regr.fit(train.as_matrix(columns=input_features), train.as_matrix(columns=['price']))
    RMSE = mean_squared_error(test.as_matrix(columns=['price']), regr.predict(test.as_matrix(columns=input_features))) ** 0.5
    
    return RMSE, regr.intercept_[0], regr.coef_


# In[ ]:


print ('RMSE: %s, intercept: %s, coefficients: %s'        %multiple_regression_model(train_data, test_data, ['sqft_living','bathrooms','bedrooms']))
print ('RMSE: %s, intercept: %s, coefficients: %s'        %multiple_regression_model(train_data, test_data, ['sqft_above','view#0','bathrooms']))
print ('RMSE: %s, intercept: %s, coefficients: %s'        %multiple_regression_model(train_data, test_data, ['bathrooms','bedrooms']))
print ('RMSE: %s, intercept: %s, coefficients: %s'        %multiple_regression_model(train_data, test_data, ['view#0','grade#12','bedrooms','sqft_basement']))
print ('RMSE: %s, intercept: %s, coefficients: %s'        %multiple_regression_model(train_data, test_data, ['sqft_living','bathrooms','view#0']))


# We can also try to fit a higher-order polynomial on the input.  
# For example, we can try to fit a qudratic function on sqft_living.

# In[ ]:


# create a new column in train_data
train_data['sqft_living_squared'] = train_data['sqft_living'].apply(lambda x: x**2)
# create a new column in test_data
test_data['sqft_living_squared'] = test_data['sqft_living'].apply(lambda x: x**2)
print('RMSE: %s, intercept: %s, coefficients: %s'       %multiple_regression_model(train_data, test_data, ['sqft_living', 'sqft_living_squared']))


# While we can get better performance than simple linear models, a few problems remain.
# * First, we don't know which feature to select.   
#     Obviously some combinations of features will yield smaller RMSE on the test set
# * Second, we don't know how many features to select.  
#     This is because the more features we incorporate in the train model, the more overfit we get on the train data, resulting in higher error on the test data.

# One solution would be to test multiple features combinations (all?) and keep the solution with the smallest error value calculated on the test data. However, this is overly optimistic approach, since the model complexity is selected to minimize the test error (error is biased). A more sophisticated approach is to use two sets for testing our models, a.k.a: a validation set and a test set. We select model complexity to minimize error on the validation set and approximate the generalization error based on the test set.

# Going through all subsets of features combinations is most often computationally infeasible. For example, having 30 features yield more than 1 billion combinations. Another approach is to use a greedy technique like a forward stepwise algorithm where the best estimator feature is added to the set of already selected features at each iteration. For example, let's pretend that the best single estimator is sqft_living. In the 2nd step of the greedy algorithm, we test all the remaining features one by one in combinations with sqft_living (e.g., sqft_living and bedrooms, sqft_living and waterfront, etc) and select the best combination using training error. At the end, we select the model complexity (number of features) using the validation error and estimate the generalization error using the test set.
# 
# Let's try this method.

# In[ ]:


# We're first going to add more features into the dataset.
# sqft_living cubed
train_data['sqft_living_cubed'] = train_data['sqft_living'].apply(lambda x: x**3) 
test_data['sqft_living_cubed'] = test_data['sqft_living'].apply(lambda x: x**3)

# bedrooms_squared: this feature will mostly affect houses with many bedrooms.
train_data['bedrooms_squared'] = train_data['bedrooms'].apply(lambda x: x**2) 
test_data['bedrooms_squared'] = test_data['bedrooms'].apply(lambda x: x**2)

# bedrooms times bathrooms gives what's called an "interaction" feature. It is large when both of them are large.
train_data['bed_bath_rooms'] = train_data['bedrooms']*train_data['bathrooms']
test_data['bed_bath_rooms'] = test_data['bedrooms']*test_data['bathrooms']

# Taking the log of squarefeet has the effect of bringing large values closer together and spreading out small values.
train_data['log_sqft_living'] = train_data['sqft_living'].apply(lambda x: log(x))
test_data['log_sqft_living'] = test_data['sqft_living'].apply(lambda x: log(x))


# In[ ]:


# split the train_data to include a validation set (train_data2 = 60%, validation_data = 20%, test_data = 20%)
train_data_2, validation_data = train_test_split(train_data, train_size = 0.75, random_state = 50)


# In[ ]:


print(data.shape)
# print(train_data.shape)
print(train_data_2.shape)
print(validation_data.shape)
print(test_data.shape)


# In[ ]:


# A function that take multiple features as input and return the RMSE (of the train and validation data)
def RMSE(train, validation, features, new_input):
    features_list = list(features)
    features_list.append(new_input)
    regr = linear_model.LinearRegression() # Create a linear regression object
    regr.fit(train.as_matrix(columns = features_list), train.as_matrix(columns = ['price'])) # Train the model
    RMSE_train = mean_squared_error(train.as_matrix(columns = ['price']), 
                              regr.predict(train.as_matrix(columns = features_list)))**0.5 # Calculate the RMSE on train data
    RMSE_validation = mean_squared_error(validation.as_matrix(columns = ['price']), 
                              regr.predict(validation.as_matrix(columns = features_list)))**0.5 # Calculate the RMSE on train data
    return RMSE_train, RMSE_validation 


# In[ ]:


input_list = train_data_2.columns.values.tolist() # list of column name
input_list.remove('price')

# list of features included in the regression model and the calculated train and validation errors (RMSE)
regression_greedy_algorithm = pd.DataFrame(columns = ['feature', 'train_error', 'validation_error'])  
i = 0
temp_list = []

# a while loop going through all the features in the dataframe
while i < len(train_data_2.columns)-1:
    
    # a temporary dataframe to select the best feature at each iteration
    temp = pd.DataFrame(columns = ['feature', 'train_error', 'validation_error'])
    
    # a for loop to test all the remaining features
    for p in input_list:
        RMSE_train, RMSE_validation = RMSE(train_data_2, validation_data, temp_list, p)
        temp = temp.append({'feature':p, 'train_error':RMSE_train, 'validation_error':RMSE_validation}, ignore_index=True)
        
    temp = temp.sort_values('train_error') # select the best feature using train error
    best = temp.iloc[0,0]
    temp_list.append(best)
    regression_greedy_algorithm = regression_greedy_algorithm.append({'feature': best, 'train_error': temp.iloc[0,1], 'validation_error': temp.iloc[0,2]}, ignore_index=True) # add the feature to the dataframe
    input_list.remove(best) # remove the best feature from the list of available features
    i += 1


# In[ ]:


regression_greedy_algorithm['index'] = regression_greedy_algorithm.index


# In[ ]:


regression_greedy_algorithm


# In[ ]:


plt.figure(figsize=(8,8))
sns.lineplot(data=regression_greedy_algorithm.loc[:, ['train_error', 'validation_error']])

plt.show()


# We can see that the validation error is minimum when we reach 25 features in the model (condition #4).  
# We stop the selection here even if the training error keeps getting smaller (overfitting).
# 
# Let's now calculate an estimation of the generalization error using test_data.

# In[ ]:


greedy_algo_features_list = regression_greedy_algorithm['feature'].tolist()[:] #select the first 30 features
test_error, _, _ = multiple_regression_model(train_data_2, test_data, greedy_algo_features_list)
print ('test error (RMSE) is: %s' %test_error)


# In[ ]:


test_temp = []
for cnt in range(regression_greedy_algorithm.shape[0]):
    greedy_algo_features_list = regression_greedy_algorithm['feature'].tolist()[:cnt+1] #select the first 30 features
    test_error, _, _ = multiple_regression_model(train_data_2, test_data, greedy_algo_features_list)
    test_temp.append(test_error)


# In[ ]:


regression_greedy_algorithm['test_error'] = test_temp


# In[ ]:


regression_greedy_algorithm


# In[ ]:


plt.figure(figsize=(8,8))
sns.lineplot(data=regression_greedy_algorithm.loc[:, ['train_error', 'validation_error', 'test_error']])

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




