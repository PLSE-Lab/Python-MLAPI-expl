#!/usr/bin/env python
# coding: utf-8

# # **CONTEXT**
# The objective of this project is to apply exploratory analysis and regression techniques to identify which features affect home prices the most in the Melbourne Housing Market.

# # **DATA PRE-PREPROCESSING**
# The first step is to load the data and gain a better understanding of the information each column contains.  

# In[ ]:


# Import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import sklearn 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Read data
dataset = pd.read_csv('../input/Melbourne_housing_extra_data-18-08-2017.csv')


# In[ ]:


# Number of rows and columns
print(dataset.shape)

# View first few records
dataset.head()


# ### **VARIABLE TYPES**

# 
# ### **Categorical Variables**  
# Based on the information below, the following variables: 'Suburb', 'Address', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname' will need to be specified as categories rather than general objects.   
# 
# In addition, the Date variable will need to be converted to a date object.

# In[ ]:


# View data types
dataset.info()


# In[ ]:


# Identify object columns
print(dataset.select_dtypes(['object']).columns)


# In[ ]:


# Convert objects to categorical variables
obj_cats = ['Suburb', 'Address', 'Type', 'Method', 'SellerG', 'CouncilArea','Regionname']

for colname in obj_cats:
    dataset[colname] = dataset[colname].astype('category')  


# In[ ]:


# Convert to date object
dataset['Date'] = pd.to_datetime(dataset['Date'])


# ### **Numeric Variables**
# A statistical summary of the numeric variables above indicates that Postcode is being treated as numeric when it should be identified as categorical.  This feature will need to be converted to the correct data type.

# In[ ]:


dataset.describe().transpose()


# In[ ]:


# Convert numeric variables to categorical
num_cats = ['Postcode']  

for colname in num_cats:
    dataset[colname] = dataset[colname].astype('category')   

# Confirm changes
dataset.info()


# ### Duplicate Variables
# According to dataset documentation, 'Rooms' and 'Bedroom2' both contain information on the number of rooms of a home has, but reported from different sources. I will investigate these columns further to determine if one should be removed from the dataset.  

# In[ ]:


# Examine Rooms v Bedroom2
dataset['Rooms v Bedroom2'] = dataset['Rooms'] - dataset['Bedroom2']
dataset


# The differences between these variables are minimal so keeping both would only be duplicating information.  Thus, the Bedroom2 feature will be removed from the data set altogether to allow for better analysis downstream.  

# In[ ]:


# Drop columns
dataset = dataset.drop(['Bedroom2','Rooms v Bedroom2'],1)


# ### **Feature Engineering**
# The dataset contains the year the home was built. Although this is being measured by the specific year, what this variable is really probing is the age of the home.  As such, home age can be expressed in terms of historic (greater than 50 years old) vs non-historic (less than 50 years old) to get the heart of this information in a more condensed way,  allowing for better analysis and visualization.  

# In[ ]:


# Add age variable
dataset['Age'] = 2017 - dataset['YearBuilt']

# Identify historic homes
dataset['Historic'] = np.where(dataset['Age']>=50,'Historic','Contemporary')

# Convert to Category
dataset['Historic'] = dataset['Historic'].astype('category')


# ### **MISSING DATA**
# 
# Based on a quick look at the number of entries for each variable, there appears to be missing information in the dataset. 
# I will explore which features are missing the most information.

# In[ ]:


# Number of entries
dataset.info()


# In[ ]:


# Visualize missing values
fig, ax = plt.subplots(figsize=(15,7))
sns.set(font_scale=1.2)
sns.heatmap(dataset.isnull(),yticklabels = False, cbar = False, cmap = 'Greys_r')
plt.show()


# In[ ]:


# Count of missing values
dataset.isnull().sum()


# In[ ]:


# Percentage of missing values
dataset.isnull().sum()/len(dataset)*100


# There are a significant amount of missing values in Price, Bathroom, Car, Landsize, Building Area, YearBuilt, Council Area, Lattitude, and Longitude. To allow for a more complete analysis, observations missing any data will be removed from the dataset.  

# In[ ]:


# View missing data
#dataset[dataset['Bedroom2'].isnull()]
#To remove rows missing data in a specific column 
# dataset =dataset[pd.notnull(dataset['Price'])]

# To remove an entire column
#dataset = dataset.drop('Bedroom2',axis = 1)

# Remove rows missing data
dataset = dataset.dropna()

# Confirm that observations missing data were removed  
dataset.info()


# ### **OUTLIERS**
# The statistical summary revealed minimum values of zero for Landsize and BuildingArea that seem odd.  Also, there is a max price of \$8.4 million in the dataset.  These observations will need to be investigated further to determine their validity and whether they should be included in the dataset for analysis.  

# In[ ]:


dataset.describe().transpose()


# In[ ]:


dataset[dataset['Age']>800]


# In[ ]:


dataset[dataset['BuildingArea']==0]


# In[ ]:


dataset[dataset['Landsize']==0]


# After additional research, I determined that a zero land size could be indicative of 'zero-lot-line' homes - residential real estate in which the structure comes up to or very near the edge of the property line.  Therefore,  these observations are valid and will remain the data set.  
# 
# However,  the observation with a 'zero'  BuildingArea will be removed because it is not possible for a home to have a size of zero.  Also, this observation is priced usually high at $8.4M (the outlier identified earlier), further confirming a possible error in the data point.  For these two reasons, this observation will be removed.

# In[ ]:


# Remove outlier
dataset = dataset[dataset['BuildingArea']!=0]

# Confirm removal
dataset.describe().transpose()


# # **EXPLORATORY ANALYSIS**
# 
# 
# ### **UNIVARIATE**  
# The dependent (or target) variable we are trying to predict in this analysis is Price.  This variable appears to be normally distributed and skewed to the right.  That is, the majority of homes around \$900k with some outliers around \$8M.

# In[ ]:


plt.figure(figsize=(16,7))
sns.distplot(dataset['Price'], kde = False,hist_kws=dict(edgecolor="k"))


# ### **BIVARIATE**
# ### **Categorical Features**
# Next, I'll take a look at the relationships between the target variable and the categorical features.   Suburb, Address, and Postcode are measures based on location.  Rather than using all of these features in the analysis, Regionname would be the best proxy of home location to use for analysis that gets to the heart of this information in a more condensed way.  
# 
# Based on domain knowledge, a home's real estate agent or council member has a minimal effect on a price relative to other features and will be excluded from further analysis.  

# In[ ]:


# Identify categorical features
dataset.select_dtypes(['category']).columns


# In[ ]:


# Abbreviate Regionname categories
dataset['Regionname'] = dataset['Regionname'].map({'Northern Metropolitan':'N Metro',
                                            'Western Metropolitan':'W Metro', 
                                            'Southern Metropolitan':'S Metro', 
                                            'Eastern Metropolitan':'E Metro', 
                                            'South-Eastern Metropolitan':'SE Metro', 
                                            'Northern Victoria':'N Vic',
                                            'Eastern Victoria':'E Vic',
                                            'Western Victoria':'W Vic'})


# In[ ]:


# Suplots of categorical features v price
sns.set_style('darkgrid')
f, axes = plt.subplots(2,2, figsize = (15,15))

# Plot [0,0]
sns.boxplot(data = dataset, x = 'Type', y = 'Price', ax = axes[0,0])
axes[0,0].set_xlabel('Type')
axes[0,0].set_ylabel('Price')
axes[0,0].set_title('Type v Price')

# Plot [0,1]
sns.boxplot(x = 'Method', y = 'Price', data = dataset, ax = axes[0,1])
axes[0,1].set_xlabel('Method')
#axes[0,1].set_ylabel('Price')
axes[0,1].set_title('Method v Price')

# Plot [1,0]
sns.boxplot(x = 'Regionname', y = 'Price', data = dataset, ax = axes[1,0])
axes[1,0].set_xlabel('Regionname')
#axes[1,0].set_ylabel('Price')
axes[1,0].set_title('Region Name v Price')

# Plot [1,1]
sns.boxplot(x = 'Historic', y = 'Price', data = dataset, ax = axes[1,1])
axes[1,1].set_xlabel('Historic')
axes[1,1].set_ylabel('Price')
axes[1,1].set_title('Historic v Price')

plt.show()


# ### **Insights**
# * Median prices for houses are over \$1M, townhomes are \$800k - \$900k and units are approx \$500k.    
# * Home prices with different selling methods are relatively the same across the board.    
# * Median prices in the Metropolitan Region are higher than than that of Victoria Region - with Southern Metro being the area with the highest median home price (~\$1.3M).  
# * With an average price of $1M, historic homes (older than 50 years old) are valued much higher than newer homes in the area, but have more variation in price.    

# ### **Numeric Features**
# Next, I visualize the relationships between numeric features in the dataset and price.  

# In[ ]:


# Identify numeric features
dataset.select_dtypes(['float64','int64']).columns


# In[ ]:


# Suplots of numeric features v price
sns.set_style('darkgrid')
f, axes = plt.subplots(4,2, figsize = (20,30))

# Plot [0,0]
axes[0,0].scatter(x = 'Rooms', y = 'Price', data = dataset, edgecolor = 'b')
axes[0,0].set_xlabel('Rooms')
axes[0,0].set_ylabel('Price')
axes[0,0].set_title('Rooms v Price')

# Plot [0,1]
axes[0,1].scatter(x = 'Distance', y = 'Price', data = dataset, edgecolor = 'b')
axes[0,1].set_xlabel('Distance')
# axes[0,1].set_ylabel('Price')
axes[0,1].set_title('Distance v Price')

# Plot [1,0]
axes[1,0].scatter(x = 'Bathroom', y = 'Price', data = dataset, edgecolor = 'b')
axes[1,0].set_xlabel('Bathroom')
axes[1,0].set_ylabel('Price')
axes[1,0].set_title('Bathroom v Price')

# Plot [1,1]
axes[1,1].scatter(x = 'Car', y = 'Price', data = dataset, edgecolor = 'b')
axes[1,0].set_xlabel('Car')
axes[1,1].set_ylabel('Price')
axes[1,1].set_title('Car v Price')

# Plot [2,0]
axes[2,0].scatter(x = 'Landsize', y = 'Price', data = dataset, edgecolor = 'b')
axes[2,0].set_xlabel('Landsize')
axes[2,0].set_ylabel('Price')
axes[2,0].set_title('Landsize v  Price')

# Plot [2,1]
axes[2,1].scatter(x = 'BuildingArea', y = 'Price', data = dataset, edgecolor = 'b')
axes[2,1].set_xlabel('BuildingArea')
axes[2,1].set_ylabel('BuildingArea')
axes[2,1].set_title('BuildingArea v Price')

# Plot [3,0]
axes[3,0].scatter(x = 'Age', y = 'Price', data = dataset, edgecolor = 'b')
axes[3,0].set_xlabel('Age')
axes[3,0].set_ylabel('Price')
axes[3,0].set_ylabel('Age v Price')

# Plot [3,1]
axes[3,1].scatter(x = 'Propertycount', y = 'Price', data = dataset, edgecolor = 'b')
axes[3,1].set_xlabel('Propertycount')
#axes[3,1].set_ylabel('Price')
axes[3,1].set_title('Property Count v Price')

plt.show()


# In[ ]:


# Pairplot
#sns.pairplot(dataset,vars= ['Rooms', 'Price', 'Distance', 'Bathroom', 'Car', 'Landsize','BuildingArea',  'Propertycount','Age'], palette = 'viridis')


# **Insights**
# 
# The majority of homes in the dataset have 4 or 5 rooms.  
# The most prominent trend is that there is a negative correlation between Distance from Melbourne's Central Business District (CBD) and Price.  The most expensive homes (\$2M or more) tend to be within 20km of the CBD.
# 

# # **CORRELATION**
# Next, I explore how all the variables are correlated with one another.  

# In[ ]:


plt.figure(figsize=(10,6))
sns.heatmap(dataset.corr(),cmap = 'coolwarm',linewidth = 1,annot= True, annot_kws={"size": 9})
plt.title('Variable Correlation')


# **Weak Positive Correlation  **  
# Age and Price  
# 
# **Moderate Positive Correlation**   
# Rooms and Price    
# Bathrooms and Price    
# Building Area and Price    
# 
# The Rooms, Bathroom, and BuildingArea features are also moderately correlated with one another as they are all measures of home size. 

# # **LINEAR REGRESSION**

# In[ ]:


# Identify numeric features
dataset.select_dtypes(['float64','int64']).columns


# In[ ]:


# Split
# Create features variable 
X =dataset[['Rooms', 'Distance', 'Bathroom', 'Car', 'Landsize', 
            'BuildingArea', 'Propertycount','Age']]

# Create target variable
y = dataset['Price']

# Train, test, split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .20, random_state= 0)


# In[ ]:


# Fit
# Import model
from sklearn.linear_model import LinearRegression

# Create linear regression object
regressor = LinearRegression()

# Fit model to training data
regressor.fit(X_train,y_train)


# In[ ]:


# Predict
# Predicting test set results
y_pred = regressor.predict(X_test)


# ### **Regression Evaluation Metrics**  
# Three common evaluation metrics for regresson problems:  
# 1. Mean Absolute Error (MAE)  
# 2. Mean Squared Error (MSE)  
# 3. Root Mean Squared Error (RMSE)  
# All basic variations on the difference between what you predicted and the true values.  
# 
# Comparing these metrics:  
# 
# **MAE** is the easiest to understand, because it's the average error.   
# **MSE**  more popular than MAE, because MSE "punishes" larger errors, tends to be useful in the real world.   
# **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units (target units) .   
# 
# All of these are loss functions, because we want to minimize them.    
# 

# In[ ]:


# Score It
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,y_pred))
print('MSE:',metrics.mean_squared_error(y_test,y_pred))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# RMSE tells us explicitly  how much our predictions deviate, on average, from the actual values in the dataset. In this case, our predicted values are \$508,212.42 away from the actual value.

# In[ ]:


# Calculated R Squared
print('R^2 =',metrics.explained_variance_score(y_test,y_pred))


# According to the R-squared,  47.6% of the variance in the dependent variable is explained by the model. 

# ### **Analyze the Residuals**

# In[ ]:


# Actual v predictions scatter
plt.scatter(y_test, y_pred)


# In[ ]:


# Histogram of the distribution of residuals
sns.distplot((y_test - y_pred))


# ### ** Interpreting the Cofficients**

# In[ ]:


cdf = pd.DataFrame(data = regressor.coef_, index = X.columns, columns = ['Coefficients'])
cdf


# # **CONCLUSION**

# Every one unit increase in:
# - **Rooms** is associated with an increase in Price by 	\$136,531.55  
# - **Distance** is associated with a *decrease* in Price by \$32,160.84  
# - **Bathroom** is associated with an increase in Price by \$236,639.21  
# - **Car** space is associated with an increase in Price by \$59,122.83  
# - **Landsize**  is associated with an increase in Price by \$35.75  
# - **BuildingArea**  is associated with an increase in Price by \$26,65.10  
# - **Propertycount ** is associated with a* decrease* in Price by \$0.05  
# - **Age** is associated with an increase in Price by \$4,729.73  
