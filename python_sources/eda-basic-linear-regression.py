#!/usr/bin/env python
# coding: utf-8

# # Predicting Boston Housing Prices

# In this notebook we will take a look at Boston Housing Prices dataset on Kaggle. This dataset comes from the UCI Machine Learning Repository and contains 506 rows and 14 columns. Each row represents a home located in Boston, Massachusetts in 1978 and the 14 columns represent datapoints collected on each home. The purpose of this notebook is to use the data collected about each home to predict it's median home value. This is a supervised regression task meaning:
# 
# * **Supervised** - Target variable are included in the dataset.
# *  **Regression** - The target variable is continuous.

# # Dataset Feature Overview
# This list includes a description of all columns in the dataset.
# * CRIM   -   per capita crime rate by town
# * ZN   -   proportion of residential land zoned for lots over 25,000 sq.ft.
# * INDUS   -   proportion of non-retail business acres per town
# * CHAS   -   Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# * NOX   -   nitric oxides concentration (parts per 10 million)
# * RM   -   average number of rooms per dwelling
# * AGE   -   proportion of owner-occupied units built prior to 1940
# * DIS   -   weighted distances to five Boston employment centres
# * RAD   -   index of accessibility to radial highways
# * TAX   -   full-value property-tax rate per $10,000
# * PTRATIO - Pupil-techer ratio by town
# * B -  1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# * LSTAT - % lower status of the population
# * MEDV (TARGET) - Median value of owner-occupied homes in 1000's

# ## Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns

# Any results you write to the current directory are saved as output.


# ## Reading in the data
# Per the data and overview pages for this dataset and by the directory listing there is one file in the directory. One thing I noted when looking at this data in the data overview section was it does not include headers, so we will have to add headers to this file ourselves using pandas.

# In[ ]:


column_headers = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
boston = pd.read_csv('../input/housingdata.csv', names = column_headers)
print('Dataset Shape', boston.shape)
boston.head()


# This dataset has 506 rows (observations) and 14 columns (features) including our target variable MEDV. One thing to note right off the bat is the CHAS column is a binary variable and the RAD variable appears to be a categorical variable as well. 

# # Exploratory Data Analysis (EDA)
# EDA is the process of understanding what the data is telling us by calculating statistics and creating charts and figures. These statistics and charts can help find anomalies which could impact our analysis or find relationships and trends between the various features in our data. EDA starts off at a high level but narrows in scope as we find interesting patterns and relationships in our data.

# ### Examine the distribution and summary statistics of the MEDV (Target) column
# This notebook is focused on creating a model that uses the 13 features in our dataset to predict the MEDV column, which is the median home value of each home in the dataset (in thousands).

# In[ ]:


sns.set_style('whitegrid')
MEDV = sns.distplot(boston['MEDV'])
MEDV.set(title = "Distribution of MEDV column")
boston['MEDV'].describe()


# We can see the distribution of the target column is slightly right-skewed with mean of 22.53 and a standard deviation of 9.20. There do appear to be a few outliers on the higher end of the MEDV price distribution

# ### Check for missing values
# Next we need to check if the dataset has any missing values.

# In[ ]:


#Calculate all missing values in the dataset.
missing_values = boston.isnull().sum().sum()
print("Missing values in dataset: ", missing_values)


# This is a very nice and clean dataset with no missing values. In most datasets this is not the case and it's very important to check for missing values because those will need to be delt with through dropping columns or imputing values.

# ### Column Types
# It's important to understand the column types because a machine learning model can not use categorical variables. Categorical variables need to be encoded as numbers before being used by the model.

# In[ ]:


boston.info()


# All of the column types are either int64 or float64 which indicates they are all numeric columns. We know the CHAS feature is binary so it can only take values of 0 and 1. 

# ### Anomalies and Outliers
# One way we can check for anomalies and outliers in the data is to look at the distributions of the features in our dataset as well as summary statistics for each column. To do this I'll plot some histograms and use the .describe() method.

# In[ ]:


f, ax = plt.subplots(nrows = 7, ncols = 2, figsize=(16,16))
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B', 'LSTAT']
row = 0
col = 0
for i, column in enumerate(columns):
    g = sns.distplot(boston[column], ax=ax[row][col])
    col += 1
    if col == 2:
        col = 0
        row += 1

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.5)


# In[ ]:


boston.iloc[:,:-1].describe()


# Looking at these histograms and summary statistics, I noticed a couple interesting things:
# * TAX column has a majority of its values in the 200-400 range but there are a collection of homes that have a TAX value of above 600
# * RAD column has most of its values between 0-10 but there is a collection of values over 20.
# * Many of the columns are heavily skewed left or right. For example, the ZN column is heavily skewed right. I would like to understand that column in more detail and it's effect on the MEDV column.
# The code below will explore each observation.
# 

# In[ ]:


rad_out = boston.copy()
rad_out['OUTLIER'] = rad_out['RAD'].apply(lambda x: 1 if x > 15 else 0)


# In[ ]:


sns.boxplot(x='OUTLIER', y='MEDV', data=rad_out)


# In[ ]:


rad_out.groupby('OUTLIER').mean()['MEDV']


# This is interesting! We see that mean MEDV value for houses that have a RAD value > 15 is 16 where as the mean MEDV value for houses that have a RAD value of < 15 is roughly 24. This is pictorally depicted by the boxplot above as well. We will keep this in mind when we engineer features later in the notebook.

# Now we'll look at the TAX values

# In[ ]:


tax_out = boston.copy()
tax_out['OUTLIER'] = boston['TAX'].apply(lambda x: 1 if x > 600 else 0)
sns.boxplot(x='OUTLIER', y='MEDV', data=tax_out)
tax_out.groupby('OUTLIER').mean()['MEDV']


# Somewhat similar to the RAD situation, the TAX column also seems to have an impact on the MEDV column. We will also keep this in mind down the road.

# **ZN Column**
#  The ZN column represents the proportion of residential land zoned for lots over 25,000 sq.ft.

# In[ ]:


boston.groupby('ZN').count()


# 372 observations are 0 for the ZN column meaning that those lots are not in excess of 25000 square feet. The remaining observations are spread from numbers 12.5-100, indicating the percentage of land zoned for that lot. Let's try to vizualize the effect of these numebrs on the MEDV statistic.
# 
# * In order to vizualize this, I'm going to cut the ZN category into 4 bins (0-24, 25-50, 51-75, 76,-100) and chart the average MEDV value for each grouping. 

# In[ ]:


zn = boston.copy()
zn['BINNED'] = pd.cut(zn['ZN'], bins = 4)
zn.head()


# In[ ]:


zn_grouped = zn.groupby('BINNED').mean()['MEDV']
zn_grouped


# In[ ]:


plt.figure(figsize=(8,8))
plt.bar(zn_grouped.index.astype(str), zn_grouped)


# There is a clear positive trend between land zoned and MEDV home value. Intuitively this chart makes sense because the smaller the land size, the less value the home should be. I.E. homes with more land tend to be worth more. 

# ### Correlations

# One way to try and understand the data is by looking for correlations between the features and the target. We can calculate the Pearson correlation coefficient between every variable and the target using the .corr method.

# In[ ]:


#Sort correlations
correlations = boston.corr()['MEDV'].sort_values()
correlations


# Let's take a look at some of the more significant correlations. 
# * The most negative correlation is LSTAT which is % lower status of the population, so what this is saying is as the % of lower status of the population increases for a home, the MEDV value of the home tends to decrease.
# * The most positive correlation (Except for MEDV) is RM which is average number of rooms per home. This is saying that as the average number of rooms increase, the MEDV value of the home tends to increase.

# ### Effect of RM on MEDV

# In[ ]:


sns.lmplot(x='RM', y='MEDV', data=boston)


# The chart above helps confirm what was displayed by the correlation matrix. As the RM variable increases, the MEDV value also tends to increase.

# ### Effect of LSTAT on MEDV

# In[ ]:


sns.lmplot(x='LSTAT', y='MEDV', data=boston)


# The chart above helps confirm what was displayed by the correlation matrix. As the RM variable increases, the MEDV value also tends to increase.

# # Collinearity
# Collinearity is the term used to describe the event when features are highly correlated with each other. Correlated features can pose problems in a regression model by masing the true effect of significant features. This can impact the quality of fit of a regression model and should be taken into account when creating one.
# 
# ### Pairplot
# The pairplot is a great way to see relationships between pairs of variables. We can identify collinear variables as well as other interesting relationships between the predictors and the response.

# In[ ]:


sns.pairplot(data=boston)


# ### Correlation Matrix
# A great follow up to the pairplot is the correlation matrix. The correlation matrix displays the correlations for each pair of variables in the dataset and can make it easy to spot correlated features. 

# In[ ]:


plt.figure(figsize=(8,8))
sns.heatmap(boston.corr())


# We can see that there appear to be a strong relationship between the following features:
# * TAX and RAD
# * DIS and AGE
# * DIS and NOX
# * DIS and INDUS
# * TAX and INDUS
# * NOX and INDUS
# 
# 

# ### Multicollinearity
# Not all correlation problems can be detected by looking at the pairplot or correlatio matrix. It is possible for correlation to exist between 3 or more variables, which is called  multicollinearity. One way we can spot collinearity and multicollineary in the data is by calculating the [Variance Inflation Factor](https://en.wikipedia.org/wiki/Variance_inflation_factor) for each variable. a VIF 10 means that the variable could be problematic and impact the results of a regression model.

# In[ ]:


#Imports to calculate VIF's for each predictor
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from patsy import dmatrices


# In[ ]:


#gather features
features = "+".join(boston.columns[:-1])

# get y and X dataframes based on this regression:
y, X = dmatrices('MEDV ~' + features, boston, return_type='dataframe')


# In[ ]:


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.sort_values(by='VIF Factor', ascending=False).iloc[1:,:]


# We can see the VIF Factors for the TAX and RAD features are the highest among the features. Additionally, those two features are highly correlated with eachother as noted above. Let's try adding these two features together and see if the VIF factor of the new feature is lower than the previous two.

# In[ ]:


tax_rad = boston.copy()
tax_rad['taxrad'] = tax_rad['TAX'] + tax_rad['RAD']
tax_rad = tax_rad.drop(['TAX', 'RAD'], axis=1)


# In[ ]:


#gather features
features = "+".join(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'taxrad',
       'PTRATIO', 'B', 'LSTAT'])

# get y and X dataframes based on this regression:
y, X = dmatrices('MEDV ~' + features, tax_rad, return_type='dataframe')


# In[ ]:


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.sort_values(by='VIF Factor', ascending=False).iloc[1:,:]


# Here we can see the VIF factor of the new "taxrad" feature has a significantly lower VIF factor than both the TAX and RAD features. We will keep this in mind moving forward.

# # Feature Engineering

# Feature engineering is the process of constructing new features from features already in the dataset. There are many ways to perform feature engineering and one way is through constructing interaction terms between the features. These include current features raised to a power, current features multiplied by each other, etc. They are called interaction terms because they capture interactions within variables.

# In[ ]:


##Dataframe to capture polynomial features
poly_features = boston.copy()

#Capture target variable
poly_target = poly_features['MEDV']
poly_features = poly_features.drop(columns=['MEDV'])

#Import polynomial feature module
from sklearn.preprocessing import PolynomialFeatures

#Create polynomial object with degree of 2
poly_transformer = PolynomialFeatures(degree = 2)

#Train the polynomial features
poly_transformer.fit(poly_features)

#Transform the features
poly_features = poly_transformer.transform(poly_features)

print('Polynomial Features Shape: ', poly_features.shape)


# We see that creating polynomial features has increased our total number of features from 12 to 105.

# In[ ]:


#Create dataframe of features.
poly_features = pd.DataFrame(poly_features, columns = poly_transformer.get_feature_names(boston.columns[:-1]))

#Add target back in to poly_features
poly_features['MEDV'] = poly_target

#Find correlations within target
poly_corrs = poly_features.corr()['MEDV'].sort_values()

print(poly_corrs.head(10))
print(poly_corrs.tail(10))


# We can see that some of the highest magnitude correlated features with the target "MEDV" feature are ones we have created through polynomial feature engineering such as RM^2 and PRATIO * LSTAT. 

# ### Manual Feature Engineering
# As we saw in the EDA there were some relationships between MEDV column and the TAX and RAD columns. I'm going to add those features into a new dataframe called manual_features.

# In[ ]:


manual_features = boston.copy()
manual_features['TAX_OUT'] = manual_features['TAX'].apply(lambda x: 1 if x > 600 else 0)
manual_features['RAD_OUT'] = manual_features['RAD'].apply(lambda x: 1 if x > 15 else 0)
manual_features.head()


# # Model Fitting

# In machine learning we need to split our data into train dataset and a testing dataset. We fit our model on the training data to make predictions on the testing data. The function below implements this on our original boston dataset as well as the dataset incluing the polynomial features.

# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# In[ ]:


#Function to fit, train, and test linear regression model.
def basicLR(data):
    #X Set
    X = data.drop(columns='MEDV')
    
    #Y set
    y = data['MEDV']
    
    #Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    #Create linear model object
    lm = LinearRegression()
    
    #Fit linear object model to training data
    lm.fit(X_train, y_train)
    
    #Make predictions using lm.predict
    predictions = lm.predict(X_test)
    
    #Print model quality of fit scores.
    print('r^2: ', r2_score(y_test, predictions))
    print("MSE: ", mean_squared_error(y_test, predictions))
    
    return r2_score(y_test, predictions), mean_squared_error(y_test, predictions)


# In[ ]:


#original boston data results
bostonr2, bostonMSE = basicLR(boston)


# In[ ]:


#Polynomial feature results
polyr2, polyMSE = basicLR(poly_features)


# In[ ]:


basicLR_frame = pd.DataFrame(data=[[bostonr2, polyr2], [bostonMSE, polyMSE]], columns=['Baseline', 'Polynomial'], index=['r^2', 'MSE'])


# In[ ]:


basicLR_frame


# By engineering polynomial features we can see the r^2 score has increased by 12% and the MSE has decreased by 12. This is a solid improvement in model performance by just introducing interaction terms.

# # Ridge Regression
# Ridge regression is a form of linear regression that introduced regularization in the form of the L2 norm. This regularization aims to shrink the coefficients of the linear regression model therefore reducing the chance of overfitting. There are other types of regularization regression methods such as lasso and elastic net. Lasso is better suited for datasets that contain more features as it aims to reduce insignificant coefficients therefore reducing the dimensionality of the dataset.

# In[ ]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[ ]:


#Function to fit, train, and test linear regression model.
def RidgeLR(data):
    #X Set
    X = data.drop(columns='MEDV')
    
    #Y set
    y = data['MEDV']
    
    #Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    #Alphas to tune
    alphas = {'alpha':[.001, .01, .1, 10, 100]}
    
    #Create Ridge object
    ridge = Ridge(random_state = 101)
    
    #Create ridge model
    clf = GridSearchCV(ridge, alphas)
    
    #Fit linear object model to training data
    clf.fit(X_train, y_train)
    
    #Make predictions using lm.predict
    predictions = clf.predict(X_test)
    
    #Print model quality of fit scores.
    print('r^2: ', r2_score(y_test, predictions))
    print("MSE: ", mean_squared_error(y_test, predictions))
    
    return r2_score(y_test, predictions), mean_squared_error(y_test, predictions)


# In[ ]:


bridger2, bridgemse = RidgeLR(boston)


# In[ ]:


polyridger2, polyridgemse = RidgeLR(poly_features)


# In[ ]:


RidgeLR = pd.DataFrame(data=[[bridger2, polyridger2], [bridgemse, polyridgemse]], columns=['boston', 'poly features'], index=['r^2', 'MSE'])
RidgeLR


# We can see here that using ridge regression has increased the r^2 metric to 85% and decreased the MSE to 14.64. 

# # Conclusion
# This notebook encompased an end to end machine learning project. It followed a framework of Inspecting the data, performing some EDA, engineering features, training and testing our data on multiple machine learning models and interpreting the results. We saw an improvement in model performance by trying a regularized regression algorithm as well as introducing interaction variables to the model.
# 
# I'm relatively new to applied machine learning and am trying to get hands on experience using these algorithms through completing notebooks like this one. Some areas I didn't get a chance to tough on that I would have liked to in this notebook are:
# * Oulier detection and removal
# * Cross validation
# * Random Forest Regression
# As I continue to grow my knowledge base and understanding I will come back and update this notebook. I'm always trying to learn and improve myself in this area so please feel free to leave comments and questions about this notebook!
# 

# In[ ]:




