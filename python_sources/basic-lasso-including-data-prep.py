#!/usr/bin/env python
# coding: utf-8

# # Machine Learning - Random Forest Regression for House Prices

# The purpose of this notebook is to expand on the basic training provided by Kaggle and add some more in depth data analyis and preparation activities, in an attempt to apply a "lite data science process" to the stated problem.

# In[ ]:


import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.preprocessing import LabelEncoder

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass 
warnings.warn = ignore_warn #ignore warning from sklearn and seaborn

#pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x)) #Limiting floats output to 2 decimal points


# In[ ]:


# Path of the file to read. 
train_set = '../input/train.csv'
train = pd.read_csv(train_set)

train.head()


# ## Understanding the dataset - Initial Data Analysis

# The ID column is useless for analysis, first we save the ID column from the data set - then we drop it so we are left with just the data we want to analyze.

# In[ ]:


#Save the Primary Key IDcolumn in case we need it later
train_ID = train['Id']

#Drop Primary Key - 
train.drop("Id", axis = 1, inplace = True)


# ## Correlation of Numerical Features

# One of the most useful ways in which we can visualize the correlation between the different numeric features of our dataset is by producing a correlation matrix, then overlaying it with a heatmap.  Generally lighter colours will signify a high correlation.   

# In[ ]:


#correlation matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True);


# We are mostly interested in the target variable 'SalePrice'.  Judging from the heatmap, there is a strong correlation between SalePrice and two other numeric variables.  These are: GrLivArea and OverallCond. 
# 
# ScatterPlot the different 'highly correlated' variables against eachother to investigate the linear relationship and identify any outliers. (Be patient, this may take some time to visualize. 

# In[ ]:


#scatterplot correlated variables.
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train[cols], size = 2.5)
plt.show()


# Are there any linear relationships to 'SalePrice' that we feel we need to dig into further?   Looks like we have linear relationships between SalesPrice and GrLivArea, and TotBsmtSF.   Let create a scattergraph we can use to zoom in on these and pick put the outliers.

# ## Outliers

# In[ ]:


#plot scatter to identify any outliers
var='GrLivArea'

fig, ax = plt.subplots()
ax.scatter(x = train[var], y = train['SalePrice'])
plt.ylabel('SalePrice', fontsize=10)
plt.xlabel(var, fontsize=10)
plt.show()


# Notice when we plot GrLivArea against Saleprice there are a couple of particularly significant outliers on the bottom right of the diagram.  Lets run some code to remove these outliers so they dont reduce the accuracy of our model.

# In[ ]:


#Delete outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Check the graph again
vari = 'GrLivArea'
fig, ax = plt.subplots()
ax.scatter(train[vari], train['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel(vari, fontsize=13)
plt.show()


# You can see from the second plot that the outliers are now removed.   Lets move forwards and take a look at the distribution of the SalesPrice values.

# ## Distribution / Skewedness of Values

# In[ ]:


#Plot the distribution of the Target column Y.
sns.distplot(train['SalePrice'])


# The distribution is left Skewed.  We can run Log1p to sort out the distribution of the data.   Log1p bascially applies Log(1+x) to the target column.

# In[ ]:


#Sort out the distribution by using Log1p
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'])


# Now we have a nice bell-shaped distribution of values for our data, let look for our other 

# ## Dealing with Missing Data

# First lets take a look at the percentage of nulls per column.   We sort this descending because we are only bothered about features with a high number of nulls

# In[ ]:


#Show % of nulls per column
train_na_percent = (train.isnull().sum() / len(train)).sort_values(ascending=False)
train_na_total = train.isnull().sum().sort_values(ascending=False)
missing_data = pd.concat([train_na_percent, train_na_total], axis=1, keys=['%', 'Total'])[:30]
missing_data.head()


# In[ ]:


#Graph it 
f, ax = plt.subplots(figsize=(6, 4))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['%'])
plt.xlabel('Feature')
plt.ylabel('Percent of missing values')


# Judging from the amount of missing data, its reasonable to remove PoolQC, MiscFeature, Alley, Fence and FireplaceQu.  Essentially I made the decision that these are not important because they are not things we consider when buying a house.  Ideally we would do more analysis on these features, but in the interest of keeping this simple I wont bother.  Lets just drop them altoghter. 
# 
# Note:  Model is actually more accurate with these in.  Will dig deeper and update further, for now the columns are included in the model.
# 
# 

# In[ ]:


#Drop Columns with poor data

#train = train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'], axis=1)



# ## Feature Engineering

# There are different methods we will use here to replace nulls with something dependent upon the type of feature we are dealing with.  We use the lines of code below to replace the nulls with either: 
# 1.  a string stating the word "None" - if the data descritpion defines this.
# 2.  the median value of the feature (if its a numberic variable)
# 3.  The modal class of the feature (if it is a categorical feature)
# 4.  A zero - if the data description defines this.
# 

# In[ ]:


##FillNA with String
for col in ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','MSSubClass', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    train[col] = train[col].fillna('None')
     
#FillNA with Median
train["LotFrontage"] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

#fillNA with Zero    
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea'):
    train[col] = train[col].fillna(0)

#fillNA with Mode    
for col in ('MSZoning', 'Electrical', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType'):
    train[col] = train[col].fillna(train[col].mode()[0])

#Drop
train = train.drop(['Utilities'], axis=1)

#Functional replace NA with Typ as per data descritpion - thanks @Sergine
train["Functional"] = train["Functional"].fillna("Typ")



# Once we have made the changes to our dataset, we run a check to see if there are any nulls left 

# In[ ]:


#Check remaining missing values if any 
train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame(train_na)
missing_data.head()


# Next we check the data description to see if we need to update the datatypes on any of our features.
# If a numeric variable is actually a category, we need to convert this to a string, so the RandomForest will treat is as a category and not a numeric variable.   A good example of this is years,  we dont want 2019 to be treated as a greater value than 2001, we want the algorithm to understand that the years are a category, not a number.

# In[ ]:


#MSSubClass needs to be str
train['MSSubClass'] = train['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
train['OverallCond'] = train['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
train['YrSold'] = train['YrSold'].astype(str)
train['MoSold'] = train['MoSold'].astype(str)


# We also should check if there are any "object" type categorical features left in our dataset.  Most ML algorithms need (or prefer) to accept encoded information.  Encoded inforomation is essentially changing values into numbers.  I.e. "Street" may contain values labelled "King Street" or "West 25th Street", an encoder will assign a number to the value and populate it throughout the dataset.  
# 
# The main two types of encoding you will come across will be Label and OneHot Encoders.  Both have separate use cases. [This link ](https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621) is useful if you want to learn more about encoding. 
# 
# 

# In[ ]:


#label encoding for all of the categorical varibles that are stored as object - also added in the remaining "object" type features.


cols = ('PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'MSZoning','LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 
        'Condition2',  'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 
        'MasVnrType', 'Foundation', 'Heating', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition')


# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))


# In[ ]:


#Check if there are any "object" types left. (used previously for randomForest)
train.info()


# Are there any additional features that we can create from our current dataset that we believe could be useful in identifying the value of the house?  These will generally always be a numeric.

# In[ ]:


# Adding total sqfootage feature 
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']


# Now we run train.head() to check that our data is encoded. i.e. check that there are no text values in there, and everything is represented by a number.

# In[ ]:


train.head()


# ## Split into X (Features) and Y (Target)

# Now we have a clean, encoded dataset that is ready to be used to train a model.  We need to split down our dataset into:
# 
# X  - This contains all of the features 
# 
# y  - This is the variable that we will use to train the model

# In[ ]:


# Split Data set into X and Y - X is the features, Y is the varible we want to predict.
X = train
X = train.drop(['SalePrice'], axis=1) #remove when using test data
y = train['SalePrice'] #remove when using test data


# In[ ]:


X.shape


# ## Train the Model (Train Data Only)

# First we need to pick a model, for this is have used a Lasso from SciKit.  Once we have selected the model we must fit the previously created X and y variables to it.   This is just for training data, you will not re-run this when it comes to submitting your test data. 

# In[ ]:


# Select an alpha - Trail and Error or Cross Validation - will expand on this.
best_alpha = 0.00099

#Train the Model
regr = Lasso(alpha=best_alpha, max_iter=50000)
regr.fit(X, y)


# ## Predict Using Trained Model

# Now we have a trained model, we need to use it to create some predictions on our training (or test) data.  We simply create a variable which references the model we trained and pass in X, which is our list of features.

# In[ ]:


#Predict using X as our parameters
lasso_pred = regr.predict(X)


# Now we should check that our list of predictions make sense, remember they wont look like house prices because we applied log1p to the values, we can undo this afterwards. 
# 

# In[ ]:


print(lasso_pred)


# Add the targets and predictions back into the data set also use expm1 to change them back to their original numbers

# In[ ]:


#Add the targets and predictiions back into the data set also use expm1 to change them back to their original numbers

#Add the score to X
X['lassoScore'] = lasso_pred

#Add the Actual Score to X
X['lassoScoreAct']= np.expm1(X['lassoScore'])

#Create a column named Targets containing the original SalesPrice feature 
X['Targets'] = y
X.head()


# ## Scoring the Results

# In[ ]:


#Graph Targets vs Prediction to see correllation
sns.scatterplot(x=X['lassoScore'], y=X['Targets'])
plt.xlabel('Target')
plt.ylabel('Prediction')
plt.show()


# As per Competition Rules, we validate the data using log mean squared error.  Which is the sqrt of mean_squared_error.

# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from math import sqrt


rf_val_sq_error2 = sqrt(mean_squared_error(X['Targets'], X['lassoScore'])) #correct
print(rf_val_sq_error2)




# ## Submitting to Kaggle

# This last box will save the columns required to submit to Kaggle and output a document named "submission.csv" to the root of your filesystem.

# In[ ]:


#for submission of test data set
#submission = X[['Id', 'SalePrice']]
#submission.to_csv('submission.csv', index=False)


