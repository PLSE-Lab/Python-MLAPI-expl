#!/usr/bin/env python
# coding: utf-8

# ![](https://wallpaper.wiki/wp-content/uploads/2017/04/wallpaper.wiki-Full-HD-Wallpapers-1080p-Cars-PIC-WPC002339-1.jpg)

# ## Problem Statement:
# 
# A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts. 
# 
#  
# 
# They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. The company wants to know:
# 
# * Which variables are significant in predicting the price of a car
# * How well those variables describe the price of a car

# ## Business Goal
# We are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics of a new market.

# #### This kernel is based on the assignment by IIITB collaborated with upgrad.

# #### If this Kernel helped you in any way, some <font color="red"><b>UPVOTES</b></font> would be very much appreciated

# #### Below are the steps which we will be basically following:
# 
# 1. [Step 1: Reading and Understanding the Data](#1)
# 1.  [Step 2: Cleaning the Data](#2)
#     - Missing Value check
#     - Data type check
#     - Duplicate check
# 1. [Step 3: Data Visualization](#3)
#     - Boxplot
#     - Pairplot
# 1. [Step 4: Data Preparation](#4) 
#    - Dummy Variable
# 1. [Step 5: Splitting the Data into Training and Testing Sets](#5)
#    - Rescaling
# 1. [Step 6: Building a Linear Model](#6)
#    - RFE
#    - VIF
# 1. [Step 7: Residual Analysis of the train data](#7)
# 1. [Step 8: Making Predictions Using the Final Model](#8)
# 1. [Step 9: Model Evaluation](#8)
#    - RMSE Score

# <a id="1"></a> <br>
# ## Step 1 : Reading and Understanding the Data

# In[ ]:


# import all libraries and dependencies for dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

# import all libraries and dependencies for data visualization
pd.options.display.float_format='{:.4f}'.format
plt.rcParams['figure.figsize'] = [8,8]
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', -1) 
sns.set(style='darkgrid')
import matplotlib.ticker as ticker
import matplotlib.ticker as plticker

# import all libraries and dependencies for machine learning
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score


# In[ ]:


# Local file path. Please change the file path accordingly

path = '../input/car-price-prediction/'
file = path + 'CarPrice_Assignment.csv'
file1 = path+ 'Data Dictionary - carprices.xlsx'


# In[ ]:


# Reading the automobile consulting company file on which analysis needs to be done

df_auto = pd.read_csv(file)

df_auto.head()


# In[ ]:


# Reading the data dictionary file

df_stru = pd.read_excel(file1)
df_stru.head(2)


# #### Understanding the dataframe

# In[ ]:


# shape of the data
df_auto.shape


# In[ ]:


# information of the data
df_auto.info()


# In[ ]:


# description of the data
df_auto.describe()


# <a id="2"></a> <br>
# ## Step 2: Cleaning the Data

# We need to do some basic cleansing activity in order to feed our model the correct data.

# In[ ]:


# dropping car_ID based on business knowledge

df_auto = df_auto.drop('car_ID',axis=1)


# In[ ]:


# Calculating the Missing Values % contribution in DF

df_null = df_auto.isna().mean().round(4) * 100

df_null.sort_values(ascending=False).head()


# In[ ]:


# Datatypes
df_auto.dtypes


# In[ ]:


# Outlier Analysis of target variable with maximum amount of Inconsistency

outliers = ['price']
plt.rcParams['figure.figsize'] = [8,8]
sns.boxplot(data=df_auto[outliers], orient="v", palette="Set1" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Price Range", fontweight = 'bold')
plt.xlabel("Continuous Variable", fontweight = 'bold')
df_auto.shape


# #### Insights: 
# - There are some price ranges above 36000 which can be termed as outliers but lets not remove it rather we will use standarization scaling.

# In[ ]:


# Extracting Car Company from the CarName as per direction in Problem 

df_auto['CarName'] = df_auto['CarName'].str.split(' ',expand=True)


# In[ ]:


# Unique Car company

df_auto['CarName'].unique()


# **Typo Error in Car Company name**
# - maxda = mazda
# - Nissan = nissan
# - porsche = porcshce
# - toyota = toyouta
# - vokswagen = volkswagen = vw

# In[ ]:


# Renaming the typo errors in Car Company names

df_auto['CarName'] = df_auto['CarName'].replace({'maxda': 'mazda', 'nissan': 'Nissan', 'porcshce': 'porsche', 'toyouta': 'toyota', 
                            'vokswagen': 'volkswagen', 'vw': 'volkswagen'})


# In[ ]:


# changing the datatype of symboling as it is categorical variable as per dictionary file

df_auto['symboling'] = df_auto['symboling'].astype(str)


# In[ ]:


df_auto['symboling'].head()


# In[ ]:


# checking for duplicates

df_auto.loc[df_auto.duplicated()]


# In[ ]:


# Segregation of Numerical and Categorical Variables/Columns

cat_col = df_auto.select_dtypes(include=['object']).columns
num_col = df_auto.select_dtypes(exclude=['object']).columns
df_cat = df_auto[cat_col]
df_num = df_auto[num_col]


# In[ ]:


print(df_cat.head(2))


# In[ ]:


print(df_num.head(2))


# <a id="3"></a> <br>
# ## Step 3: Visualising the Data
# 
# - Here we will identify if some predictors directly have a strong association with the outcome variable `price`

# In[ ]:


df_auto['CarName'].value_counts()


# In[ ]:


# Visualizing the different car names available

ax=df_auto['CarName'].value_counts().plot(kind='bar')
ax.title.set_text('CarName')
plt.xlabel("Names of the Car",fontweight = 'bold')
plt.ylabel("Count of Cars",fontweight = 'bold')


# #### Insights:
# - Toyota seems to be the most favoured cars.
# - Mercury seems to be the least favoured cars.

# #### Visualizing the distribution of car prices

# In[ ]:


#plt.figure(figsize=(8,8))

plt.title('Car Price Distribution Plot')
sns.distplot(df_auto['price'])


# - The plots seems to be right skewed, the prices of almost all cars looks like less than 18000.
# 

# #### Visualising Numeric Variables
# 
# Pairplot of all the numeric variables

# In[ ]:


ax = sns.pairplot(df_auto[num_col])


# #### Insights:
# - `carwidth` , `carlength`, `curbweight` ,`enginesize` ,`horsepower`seems to have a poitive correlation with price.
# - `carheight` doesn't show any significant trend with price.
# - `citympg` , `highwaympg` - seem to have a significant negative correlation with price.

# #### Visualising few more Categorical Variables
# 
# Boxplot of all the categorical variables

# In[ ]:


plt.figure(figsize=(20, 15))
plt.subplot(3,3,1)
sns.boxplot(x = 'doornumber', y = 'price', data = df_auto)
plt.subplot(3,3,2)
sns.boxplot(x = 'fueltype', y = 'price', data = df_auto)
plt.subplot(3,3,3)
sns.boxplot(x = 'aspiration', y = 'price', data = df_auto)
plt.subplot(3,3,4)
sns.boxplot(x = 'carbody', y = 'price', data = df_auto)
plt.subplot(3,3,5)
sns.boxplot(x = 'enginelocation', y = 'price', data = df_auto)
plt.subplot(3,3,6)
sns.boxplot(x = 'drivewheel', y = 'price', data = df_auto)
plt.subplot(3,3,7)
sns.boxplot(x = 'enginetype', y = 'price', data = df_auto)
plt.subplot(3,3,8)
sns.boxplot(x = 'cylindernumber', y = 'price', data = df_auto)
plt.subplot(3,3,9)
sns.boxplot(x = 'fuelsystem', y = 'price', data = df_auto)
plt.show()


# #### Insights
# - The cars with `fueltype` as `diesel` are comparatively expensive than the cars with `fueltype` as `gas`.
# - All the types of carbody is relatively cheaper as compared to `convertible` carbody.
# - The cars with `rear enginelocation` are way expensive than cars with `front enginelocation`.
# - The price of car is directly proportional to `no. of cylinders` in most cases.
# - Enginetype `ohcv` comes into higher price range cars.
# - `DoorNumber` isn't affecting the price much.
# - HigerEnd cars seems to have `rwd` drivewheel

# In[ ]:


plt.figure(figsize=(25, 6))

plt.subplot(1,3,1)
plt1 = df_auto['cylindernumber'].value_counts().plot('bar')
plt.title('Number of cylinders')
plt1.set(xlabel = 'Number of cylinders', ylabel='Frequency of Number of cylinders')

plt.subplot(1,3,2)
plt1 = df_auto['fueltype'].value_counts().plot('bar')
plt.title('Fuel Type')
plt1.set(xlabel = 'Fuel Type', ylabel='Frequency of Fuel type')

plt.subplot(1,3,3)
plt1 = df_auto['carbody'].value_counts().plot('bar')
plt.title('Car body')
plt1.set(xlabel = 'Car Body', ylabel='Frequency of Car Body')


# #### Insights:
# - The number of cylinders used in most cars is `four`.
# - Number of `Gas` fueled cars are way more than `diesel` fueled cars.
# - `Sedan` is the most prefered car type.

# #### Relationship between `fuelsystem` vs `price` with hue `fueltype`

# In[ ]:


plt.figure(figsize = (10, 6))
sns.boxplot(x = 'fuelsystem', y = 'price', hue = 'fueltype', data = df_auto)
plt.show()


# #### Relationship between `carbody` vs `price` with hue `enginelocation`

# In[ ]:


df_auto['carbody'].head()


# In[ ]:


plt.figure(figsize = (10, 6))
sns.boxplot(x = 'carbody', y = 'price', hue = 'enginelocation', data = df_auto)
plt.show()


# #### Relationship between `cylindernumber` vs `price` with hue `fueltype`

# In[ ]:


plt.figure(figsize = (10, 6))
sns.boxplot(x = 'cylindernumber', y = 'price', hue = 'fueltype', data = df_auto)
plt.show()


# #### Derived Metrices
# - Average Price

# In[ ]:


plt.figure(figsize=(20, 6))

df_autox = pd.DataFrame(df_auto.groupby(['CarName'])['price'].mean().sort_values(ascending = False))
df_autox.plot.bar()
plt.title('Car Company Name vs Average Price')
plt.show()


# #### Insights:
# - `Jaguar`,`Buick` and `porsche` seems to have the highest average price.

# In[ ]:


plt.figure(figsize=(20, 6))

df_autoy = pd.DataFrame(df_auto.groupby(['carbody'])['price'].mean().sort_values(ascending = False))
df_autoy.plot.bar()
plt.title('Car Company Name vs Average Price')
plt.show()


# #### Insights:
# - `hardtop` and `convertible` seems to have the highest average price.

# In[ ]:


#Binning the Car Companies based on avg prices of each car Company.

df_auto['price'] = df_auto['price'].astype('int')
df_auto_temp = df_auto.copy()
t = df_auto_temp.groupby(['CarName'])['price'].mean()
print(t)
df_auto_temp = df_auto_temp.merge(t.reset_index(), how='left', on='CarName')
bins = [0,10000,20000,40000]
label =['Budget_Friendly','Medium_Range','TopNotch_Cars']
df_auto['Cars_Category'] = pd.cut(df_auto_temp['price_y'], bins, right=False, labels=label)
df_auto.head()


# #### Significant variables after Visualization
# - Cars_Category , Engine Type, Fuel Type
# - Car Body , Aspiration , Cylinder Number 
# - Drivewheel , Curbweight , Car Length 
# - Car Length , Car width , Engine Size
# - Boreratio , Horse Power , Wheel base 
# - citympg , highwaympg , symboling

# In[ ]:


sig_col = ['price','Cars_Category','enginetype','fueltype', 'aspiration','carbody','cylindernumber', 'drivewheel',
            'wheelbase','curbweight', 'enginesize', 'boreratio','horsepower', 
                    'citympg','highwaympg', 'carlength','carwidth']


# In[ ]:


df_auto = df_auto[sig_col]


# <a id="4"></a> <br>
# ## Step 4: Data Preparation

# #### Dummy Variables

# The variable `carbody` has five levels. We need to convert these levels into integer. Similarly we need to convert the categorical variables to numeric.
# 
# For this, we will use something called `dummy variables`.

# In[ ]:


sig_cat_col = ['Cars_Category','fueltype','aspiration','carbody','drivewheel','enginetype','cylindernumber']


# In[ ]:


# Get the dummy variables for the categorical feature and store it in a new variable - 'dummies'

dummies = pd.get_dummies(df_auto[sig_cat_col])
dummies.shape


# In[ ]:


dummies = pd.get_dummies(df_auto[sig_cat_col], drop_first = True)
dummies.shape


# In[ ]:


# Add the results to the original dataframe

df_auto = pd.concat([df_auto, dummies], axis = 1)


# In[ ]:


# Drop the original cat variables as dummies are already created

df_auto.drop( sig_cat_col, axis = 1, inplace = True)
df_auto.shape


# <a id="5"></a> <br>
# ## Step 5: Splitting the Data into Training and Testing Sets
# 
# As we know, the first basic step for regression is performing a train-test split.

# In[ ]:


df_auto


# In[ ]:


# We specify this so that the train and test data set always have the same rows, respectively
# We divide the df into 70/30 ratio

np.random.seed(0)
df_train, df_test = train_test_split(df_auto, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[ ]:


df_train.head()


# ### Rescaling the Features 
# 
# For Simple Linear Regression, scaling doesn't impact model. So it is extremely important to rescale the variables so that they have a comparable scale. If we don't have comparable scales, then some of the coefficients as obtained by fitting the regression model might be very large or very small as compared to the other coefficients.
# There are two common ways of rescaling:
# 
# 1. Min-Max scaling 
# 2. Standardisation (mean-0, sigma-1) 
# 
# Here, we will use Standardisation Scaling.

# In[ ]:


scaler = preprocessing.StandardScaler()


# In[ ]:


sig_num_col = ['wheelbase','carlength','carwidth','curbweight','enginesize','boreratio','horsepower','citympg','highwaympg','price']


# In[ ]:


# Apply scaler() to all the columns except the 'dummy' variables
import warnings
warnings.filterwarnings("ignore")

df_train[sig_num_col] = scaler.fit_transform(df_train[sig_num_col])


# In[ ]:


df_train.head()


# In[ ]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (20, 20))
sns.heatmap(df_train.corr(), cmap="RdYlGn")
plt.show()


# Let's see scatterplot for few correlated variables  vs `price`.

# In[ ]:


col = ['highwaympg','citympg','horsepower','enginesize','curbweight','carwidth']


# In[ ]:


# Scatter Plot of independent variables vs dependent variables

fig,axes = plt.subplots(2,3,figsize=(18,15))
for seg,col in enumerate(col):
    x,y = seg//3,seg%3
    an=sns.scatterplot(x=col, y='price' ,data=df_auto, ax=axes[x,y])
    plt.setp(an.get_xticklabels(), rotation=45)
   
plt.subplots_adjust(hspace=0.5)


# - We can see there is a line we can fit in above plots

# ### Dividing into X and Y sets for the model building

# In[ ]:


y_train = df_train.pop('price')
X_train = df_train


# <a id="6"></a> <br>
# ## Step 6: Building a Linear Model

# In[ ]:


X_train_1 = X_train['horsepower']


# In[ ]:


# Add a constant
X_train_1c = sm.add_constant(X_train_1)

# Create a first fitted model
lr_1 = sm.OLS(y_train, X_train_1c).fit()


# In[ ]:


# Check parameters created

lr_1.params


# In[ ]:


# Let's visualise the data with a scatter plot and the fitted regression line

plt.scatter(X_train_1c.iloc[:, 1], y_train)
plt.plot(X_train_1c.iloc[:, 1], 0.8062*X_train_1c.iloc[:, 1], 'r')
plt.show()


# In[ ]:


# Print a summary of the linear regression model obtained
print(lr_1.summary())


# ### Adding another variable
# 
# The R-squared value obtained is `0.65`. Since we have so many variables, we can clearly do better than this. So let's go ahead and add the other highly correlated variable, i.e. `curbweight`.

# In[ ]:


X_train_2 = X_train[['horsepower', 'curbweight']]


# In[ ]:


# Add a constant
X_train_2c = sm.add_constant(X_train_2)

# Create a second fitted model
lr_2 = sm.OLS(y_train, X_train_2c).fit()


# In[ ]:


lr_2.params


# In[ ]:


print(lr_2.summary())


# * The R-squared incresed from 0.650 to 0.797 

# ### Adding another variable
# 
# The R-squared value obtained is `0.797`. Since we have so many variables, we can clearly do better than this. So lets add another correlated variable, i.e. `enginesize`.

# In[ ]:


X_train_3 = X_train[['horsepower', 'curbweight', 'enginesize']]


# In[ ]:


# Add a constant
X_train_3c = sm.add_constant(X_train_3)

# Create a third fitted model
lr_3 = sm.OLS(y_train, X_train_3c).fit()


# In[ ]:


lr_3.params


# In[ ]:


print(lr_3.summary())


# We have achieved a R-squared of `0.819` by manually picking the highly correlated variables.
# Now lets use RFE to select the independent variables which accurately predicts the dependent variable `price`.

# ### RFE
# Let's use Recursive feature elimination since we have too many independent variables

# In[ ]:


# Running RFE with the output number of the variable equal to 15
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 15)             
rfe = rfe.fit(X_train, y_train)


# In[ ]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[ ]:


# Selecting the variables which are in support

col_sup = X_train.columns[rfe.support_]
col_sup


# In[ ]:


# Creating X_train dataframe with RFE selected variables

X_train_rfe = X_train[col_sup]
X_train_rfe


# After passing the arbitary selected columns by RFE we will manually evaluate each models p-value and VIF value.
# Unless we find the acceptable range for p-values and VIF we keep dropping the variables one at a time based on below criteria.
# - High p-value High VIF : Drop the variable
# - High p-value Low VIF or Low p-value High VIF : Drop the variable with high p-value first
# - Low p-value Low VIF : accept the variable

# In[ ]:


# Adding a constant variable and Build a first fitted model
import statsmodels.api as sm  
X_train_rfec = sm.add_constant(X_train_rfe)
lm_rfe = sm.OLS(y_train,X_train_rfec).fit()

#Summary of linear model
print(lm_rfe.summary())


# - Looking at the p-values, it looks like some of the variables aren't really significant (in the presence of other variables)<br>
# and we need to drop it

# ### Checking VIF
# 
# Variance Inflation Factor or VIF, gives a basic quantitative idea about how much the feature variables are correlated with each other. It is an extremely important parameter to test our linear model. The formula for calculating `VIF` is:
# 
# ### $ VIF_i = \frac{1}{1 - {R_i}^2} $

# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values, i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# We generally want a VIF that is less than 5. So there are clearly some variables we need to drop.

# ### Dropping the variable and updating the model

# *Dropping `cylindernumber_twelve` beacuse its `p-value` is `0.393` and we want p-value less than 0.05 and hence rebuilding the model*

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_rfe1 = X_train_rfe.drop('cylindernumber_twelve', 1,)

# Adding a constant variable and Build a second fitted model

X_train_rfe1c = sm.add_constant(X_train_rfe1)
lm_rfe1 = sm.OLS(y_train, X_train_rfe1c).fit()

#Summary of linear model
print(lm_rfe1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe1.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe1.values, i) for i in range(X_train_rfe1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# *Dropping `cylindernumber_six` beacuse its `p-value` is `0.493` and we want p-value less than 0.05 and hence rebuilding the model*

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_rfe2 = X_train_rfe1.drop('cylindernumber_six', 1,)

# Adding a constant variable and Build a third fitted model

X_train_rfe2c = sm.add_constant(X_train_rfe2)
lm_rfe2 = sm.OLS(y_train, X_train_rfe2c).fit()

#Summary of linear model
print(lm_rfe2.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe2.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe2.values, i) for i in range(X_train_rfe2.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# *Dropping `carbody_hardtop` beacuse its `p-value` is `0.238` and we want p-value less than 0.05 and hence rebuilding the model*

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_rfe3 = X_train_rfe2.drop('carbody_hardtop', 1,)

# Adding a constant variable and Build a fourth fitted model
X_train_rfe3c = sm.add_constant(X_train_rfe3)
lm_rfe3 = sm.OLS(y_train, X_train_rfe3c).fit()

#Summary of linear model
print(lm_rfe3.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe3.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe3.values, i) for i in range(X_train_rfe3.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# *Dropping `enginetype_ohc` beacuse its `p-value` is `0.110` and we want p-value less than 0.05 and hence rebuilding the model*

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_rfe4 = X_train_rfe3.drop('enginetype_ohc', 1,)

# Adding a constant variable and Build a fifth fitted model
X_train_rfe4c = sm.add_constant(X_train_rfe4)
lm_rfe4 = sm.OLS(y_train, X_train_rfe4c).fit()

#Summary of linear model
print(lm_rfe4.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe4.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe4.values, i) for i in range(X_train_rfe4.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# *Dropping `cylindernumber_five` beacuse its `p-value` is `0.104` and we want p-value less than 0.05 and hence rebuilding the model*

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_rfe5 = X_train_rfe4.drop('cylindernumber_five', 1,)

# Adding a constant variable and Build a sixth fitted model
X_train_rfe5c = sm.add_constant(X_train_rfe5)
lm_rfe5 = sm.OLS(y_train, X_train_rfe5c).fit()

#Summary of linear model
print(lm_rfe5.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe5.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe5.values, i) for i in range(X_train_rfe5.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# *Dropping `enginetype_ohcv` beacuse its `p-value` is `0.180` and we want p-value less than 0.05 and hence rebuilding the model*

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_rfe6 = X_train_rfe5.drop('enginetype_ohcv', 1,)

# Adding a constant variable and Build a sixth fitted model
X_train_rfe6c = sm.add_constant(X_train_rfe6)
lm_rfe6 = sm.OLS(y_train, X_train_rfe6c).fit()

#Summary of linear model
print(lm_rfe6.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe6.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe6.values, i) for i in range(X_train_rfe6.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# *Dropping `curbweight` beacuse its `VIF` is `8.1` and we want VIF less than 5 and hence rebuilding the model*

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_rfe7 = X_train_rfe6.drop('curbweight', 1,)

# Adding a constant variable and Build a sixth fitted model
X_train_rfe7c = sm.add_constant(X_train_rfe7)
lm_rfe7 = sm.OLS(y_train, X_train_rfe7c).fit()

#Summary of linear model
print(lm_rfe7.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe7.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe7.values, i) for i in range(X_train_rfe7.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# *Dropping `cylindernumber_four` beacuse its `VIF` is `5.66` and we want VIF less than 5 and hence rebuilding the model*

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_rfe8 = X_train_rfe7.drop('cylindernumber_four', 1,)

# Adding a constant variable and Build a sixth fitted model
X_train_rfe8c = sm.add_constant(X_train_rfe8)
lm_rfe8 = sm.OLS(y_train, X_train_rfe8c).fit()

#Summary of linear model
print(lm_rfe8.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe8.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe8.values, i) for i in range(X_train_rfe8.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Lets drop `carbody_sedan` and see if there is any drastic fall in R squared.If not we can drop `carbody sedan`.
# Our aim is to explain the maximum variance with minimum variable.

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_rfe9 = X_train_rfe8.drop('carbody_sedan', 1,)

# Adding a constant variable and Build a sixth fitted model
X_train_rfe9c = sm.add_constant(X_train_rfe9)
lm_rfe9 = sm.OLS(y_train, X_train_rfe9c).fit()

#Summary of linear model
print(lm_rfe9.summary())


# The R squared value just dropped by `0.005`.Hence we can proceed with dropping `carbody_sedan`.

# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe9.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe9.values, i) for i in range(X_train_rfe9.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# *Dropping `carbody_wagon` beacuse its `p-value` is `0.315` and we want p-value less than 0.05 and hence rebuilding the model*

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_rfe10 = X_train_rfe9.drop('carbody_wagon', 1,)

# Adding a constant variable and Build a sixth fitted model
X_train_rfe10c = sm.add_constant(X_train_rfe10)
lm_rfe10 = sm.OLS(y_train, X_train_rfe10c).fit()

#Summary of linear model
print(lm_rfe10.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train_rfe10.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe10.values, i) for i in range(X_train_rfe10.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# Now the VIFs and p-values both are within an acceptable range. So we can go ahead and make our predictions using model `lm_rfe10` and `lm_rfe8`.

# ### Here, we are proposing Business 2 Models which can be used to predict the car prices.

# ## MODEL I
# - With `lm_rfe10` which has basically 5 predictor variables.

# <a id="7"></a> <br>
# ## Step 7: Residual Analysis of the train data
# 
# So, now to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of it.

# In[ ]:


# Predicting the price of training set.
y_train_price = lm_rfe10.predict(X_train_rfe10c)


# In[ ]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms Analysis', fontsize = 20)                   
plt.xlabel('Errors', fontsize = 18)


# <a id="8"></a> <br>
# ## Step 8: Making Predictions Using the Final Model
# 
# Now that we have fitted the model and checked the normality of error terms, it's time to go ahead and make predictions using the final model.

# #### Applying the scaling on the test sets

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

df_test[sig_num_col] = scaler.transform(df_test[sig_num_col])
df_test.shape


# #### Dividing test set into X_test and y_test

# In[ ]:


y_test = df_test.pop('price')
X_test = df_test


# In[ ]:


# Adding constant
X_test_1 = sm.add_constant(X_test)

X_test_new = X_test_1[X_train_rfe10c.columns]


# In[ ]:


# Making predictions using the final model
y_pred = lm_rfe10.predict(X_test_new)


# <a id="9"></a> <br>
# ## Step 9: Model Evaluation
# 
# Let's now plot the graph for actual versus predicted values.

# In[ ]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)   
plt.xlabel('y_test ', fontsize=18)                       
plt.ylabel('y_pred', fontsize=16)    


# ### RMSE Score

# In[ ]:


r2_score(y_test, y_pred)


# **The R2 score of Training set is 0.912 and Test set is 0.909 which is quite close.
# Hence, We can say that our model is good enough to predict the Car prices using below predictor variables**
# - horsepower
# - carwidth	
# - Cars_Category_TopNotch_Cars
# - carbody_hatchback
# - enginetype_dohcv

# #### Equation of Line to predict the Car prices values

# $ Carprice = -0.0925 +  0.3847  \times  horsepower  + 0.3381  \times  carwidth +  1.3179 \times Carscategorytopnotchcars  - 0.1565 \times carbodyhatchback  - 1.5033 \times enginetypedohcv $

# #### Model I Conclusions:
# - R-sqaured and Adjusted R-squared - 0.912 and 0.909 - 90% variance explained.
# - F-stats and Prob(F-stats) (overall model fit) - 284.8 and 1.57e-70(approx. 0.0) - Model fit is significant and explained 90%<br> variance is just not by chance.
# - p-values - p-values for all the coefficients seem to be less than the significance level of 0.05. - meaning that all the <br>predictors are statistically significant.

# ## MODEL II
# - With `lm_rfe8` which has basically 7 predictor variables.

# ## Step 7: Residual Analysis of the train data
# 
# So, now to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of it.

# In[ ]:


# Predicting the price of training set.
y_train_price2 = lm_rfe8.predict(X_train_rfe8c)


# In[ ]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price2), bins = 20)
fig.suptitle('Error Terms Analysis', fontsize = 20)                   
plt.xlabel('Errors', fontsize = 18)


# ## Step 8: Making Predictions Using the Final Model
# 
# Now that we have fitted the model and checked the normality of error terms, it's time to go ahead and make predictions using the model.

# In[ ]:


X_test_2 = X_test_1[X_train_rfe8c.columns]


# In[ ]:


# Making predictions using the final model
y_pred2 = lm_rfe8.predict(X_test_2)


# ## Step 9: Model Evaluation
# 
# Let's now plot the graph for actual versus predicted values.

# In[ ]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred2)
fig.suptitle('y_test vs y_pred2', fontsize=20)   
plt.xlabel('y_test ', fontsize=18)                       
plt.ylabel('y_pred2', fontsize=16)    


# ### RMSE Score

# In[ ]:


r2_score(y_test, y_pred2)


# **The R2 score of Training set is 0.918 and Test set is 0.915 which is quite close.
# Hence, We can say that our model is good enough to predict the Car prices using below predictor variables**
# - horsepower
# - carwidth	
# - Cars_Category_TopNotch_Cars
# - carbody_hatchback
# - enginetype_dohcv
# - carbody_sedan                  
# - carbody_wagon                  

# #### Equation of Line to predict the Car prices values

# $ Carprice = 0.2440 +  0.3599  \times  horsepower  + 0.3652  \times  carwidth +  1.2895 \times Carscategorytopnotchcars  - 0.4859 \times carbodyhatchback  - 1.4450 \times enginetypedohcv - 0.3518 \times carbodysedan - 0.4023 \times carbodywagon $

# #### Model II Conclusions:
# - R-sqaured and Adjusted R-squared - 0.918 and 0.915 - 90% variance explained.
# - F-stats and Prob(F-stats) (overall model fit) - 215.9 and 4.70e-70(approx. 0.0) - Model fit is significant and explained 90%<br> variance is just not by chance.
# - p-values - p-values for all the coefficients seem to be less than the significance level of 0.05. - meaning that all the <br>predictors are statistically significant.

# ### Closing Statement:
# - Both the models are good enough to predict the carprices which explains the variance of data upto 90% and the model is significant.

# ### If this Kernel helped you in any way, some <font color="red"><b>UPVOTES</b></font> would be very much appreciated
