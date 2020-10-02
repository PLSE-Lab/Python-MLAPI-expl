#!/usr/bin/env python
# coding: utf-8

# #                       House Prices: Advanced regression techniques

# This project aims to solve a Kaggle problem. We are given two datasets that contain house data from Ames, Iowa that can be used to predict a house price. With 79 house features, the Ultimate goal of this project is to train a model based on the given data to predict the house price as accurately as possible evaluated by the Kaggle score. Our aim is to get a score of 0.15 or less; in this case less is better.

# # Exploratory Data Analysis

# First we import all the libraries we wish to use for our EDA

# In[ ]:


#importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
#Blocking some nasty warnings
import warnings
warnings.filterwarnings('ignore', category=Warning)
print(os.listdir("../input"))


# Importing the dataset

# In[ ]:


df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

# A brief look into our training data

df_train.head()


# Now, let us set 'Id' as an index so that it is not used by our model but kept for later use.

# In[ ]:


df_train.set_index('Id', inplace=True)
df_test.set_index('Id', inplace=True)


# ### Correlation

# Now we need to see correlation within our features. We will do that by plotting a heatmap

# In[ ]:


plt.figure(figsize=[40,30])
_ = sns.heatmap(df_train.corr(),annot=True)


# Our heatmap tells us that we have multicollinearity in our data. Let us deal with it as best as we can. Between two highly correlated predictors, we will keep the one that is more correlated to the 'SalePrice' and dump the other.

# In[ ]:


df_train.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)
df_test.drop(['GarageArea','1stFlrSF','TotRmsAbvGrd','2ndFlrSF'], axis=1, inplace=True)


# Let us consider plotting the scatterplots of the dependant variable and all the features that correlate with it the most. This will help us detect outliers. On that note. It seems rather pointless to plot 'OverallQual' as it contains ordinal data. 'GrLivArea' makes the most sense and the other two highly correlated values ('GarageCars' and 'GarageArea') are about garages and not all houses have garages. Therefore we will plot 'SalePrice' vs 'GrLivArea'

# In[ ]:


plt.scatter(df_train[['GrLivArea']],df_train[['SalePrice']])
plt.xlabel('Total Living Area Excluding Basement(square foot)')
plt.ylabel('Sale Price')


# We can clearly see that the houses above 4500 square feet are outliers. They have a very high area and a very little value. Let us remove them.

# In[ ]:


df_train = df_train[df_train['GrLivArea']<4500]


# Let us see our plot again without the outliers

# In[ ]:


plt.scatter(df_train[['GrLivArea']],df_train[['SalePrice']])
plt.xlabel('Total Living Area Excluding Basement(square foot)')
plt.ylabel('Sale Price')


# *********************************************************************************************
# I see some "NaN" values. We cannot predict much with ""NaN's", can we?
# We need to see how much of our data is missing. To do that we create a function to help us.

# In[ ]:


# A function to percentage missing of overall
def check_nulls(df):
    percent_missing = (df.isnull().sum() * 100 / len(df)).sort_values()
    return round(percent_missing,2)


# Now we can happily apply our function to our df_train

# In[ ]:


check_nulls(df_train)


# Now we can happily apply our function to our df_test

# In[ ]:


check_nulls(df_test)


# We can see that both df_train and df_test have missing values. Therefore they should be dealt with.To do that we will create a function to help us fill the missing data.
# 
# Our friend "data_description.txt" has something to tell us about the missing data. The "missing data" is not actually missing, at least for the most part. The feature they are supposed to describe simply does not exist for that particular house. Therefore we will replace the "NaN" values with "Not present" in categorical data columns where "NaN" is said to mean the feature which does not exist and 0 for numerical ones. However, our Kaggle score gives us a pat in the back when we remove 'PoolQC'. It has more than 99% of missing values in both datasets. That means that less than 1% of the houses have pools. It is therefore not so relevant to include pool quality as a predictor. The rest we will replace with the most occuring value. In fact it would be good to see which of our data is numerical and which is categorical.

# To do so we create two lists:
# 1. categorical_list containing all column names of columns that contain strings.
# 2. numerical_list containing all column names of columns that do not contain strings.

# In[ ]:


categorical_list = [col for col in df_train.columns if df_train[col].dtypes == object]
numerical_list = [col for col in df_train.columns if df_train[col].dtypes != object]

print('Categories:', categorical_list)
print('Numbers:', numerical_list)


# Lets create a function that fills in the null values

# In[ ]:


def fill_missing_values(df):
    lst = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1",
             "BsmtFinType2","Fence","FireplaceQu","GarageType","GarageFinish",
             "GarageQual","GarageCond","Electrical","GarageFinish","MiscFeature","MasVnrType","PoolQC"]
    for col in lst:
        df[col] = df[col].fillna("Not present")
        
    lst = ['GarageYrBlt','MasVnrArea','BsmtFinSF1','BsmtFinSF2','TotalBsmtSF',
           'BsmtUnfSF','BsmtFullBath','BsmtHalfBath','MasVnrArea','GarageCars']
    for col in lst:
        df[col] = df[col].fillna(0)
        
    lst = ['Utilities','MSZoning','Exterior1st','Exterior2nd','Electrical','KitchenQual']
    for col in lst:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    df['Functional'] = df['Functional'].fillna('Typ')
    df['SaleType'] = df['SaleType'].fillna('Normal')
    df['LotFrontage'] = df['LotFrontage'].fillna(df.LotFrontage.mean())
    #removing 'PoolQC' as discussed
    df.drop('PoolQC', axis=1, inplace=True)
   


# Performing the function to both df_train and df_test

# In[ ]:


fill_missing_values(df_train)
fill_missing_values(df_test)


# Do we still have these nasty "NaN's" ?

# In[ ]:


# Checking null values
print(df_train.isnull().sum().sum())
print(df_test.isnull().sum().sum())


# Now that we have no more missing data we can carry on with the next step of our EDA.
# 
# Again after a careful look at data_description.txt we notice that some of our categorical data is infact ordinal. So let us encode them in order of importance. We can use the LabelEncoder class but it might not encode them in the order we want. Therefore we will use our own custom encoder function to do the job for us. Functions are awesome, aren't they?

# In[ ]:


# A function to label encode our ordinal data
def label_encode(df):
    df['MSSubClass'] = df['MSSubClass'].astype(object)
    df = df.replace({"Alley" : {"Not present" : 0, "Grvl" : 1, "Pave" : 2},
    "BsmtCond" : {"Not present" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    "BsmtExposure" : {"Not present" : 0, "No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
    "BsmtFinType1" : {"Not present" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
    "ALQ" : 5, "GLQ" : 6},
    "BsmtFinType2" : {"Not present" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
    "ALQ" : 5, "GLQ" : 6},
    "BsmtQual" : {"Not present" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
    "CentralAir" : {"N" : 0, "Y" : 1},
    "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
    "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
    "FireplaceQu" : {"Not present" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
    "Min2" : 6, "Min1" : 7, "Typ" : 8},
    "GarageCond" : {"Not present" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    "GarageQual" : {"Not present" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    "GarageFinish" :{"Not present" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},
    "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
    "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
    "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
    "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
    "PoolQC" : {"Not present" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
    "Street" : {"Grvl" : 1, "Pave" : 2},
    "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4},
    "Fence": {"Not present" : 0, "MnWw" : 1, "GdWo" : 2, "MnPrv" : 3, "GdPrv" : 4 }},
                       
                     )
    return df


# We will now apply the label_encode function to our dataframes

# In[ ]:


df_train = label_encode(df_train)
df_test = label_encode(df_test)


# Now we may want to look at the distribution of our independent variable.  This has been a useful step in reducing our RMSLE. This is the current distribution of our independent variable:

# In[ ]:


_ = sns.distplot(df_train["SalePrice"])


# We then apply a log transformation to our 'y' variable and run another distribution plot.

# In[ ]:


df_train['SalePrice']=np.log(df_train['SalePrice'])
_ = sns.distplot(df_train["SalePrice"])


# Now that we can proceed to data preprocessing.

# # Data Preprocessing

# This is the step where we  prepare our data for modelling. We will start by seperating our data into numerical and categorical just like we did earlier. Please note that now our numerical data also contains ordinal data. Remember the encoding we did?

# In[ ]:


cat_list = [col for col in df_train.columns if df_train[col].dtypes == object]
num_list = [col for col in df_train.columns if df_train[col].dtypes != object]


# Now we can use the lists we generated above to filter our dataframes into categorical data and numerical data. We will then concatinate the two mini-dataframes. This has now allowed us to have categorical data first and then numerical data. This step is just for convenience.

# In[ ]:


categorical_data = df_train[cat_list]
numerical_data = df_train[num_list]
df_train = categorical_data.join(numerical_data)


# Let us do the same to df_test. Since df_test has no 'SalePrice' we remove it from our list

# In[ ]:


num_list.remove('SalePrice')
cat_test = df_test[cat_list]
num_test = df_test[num_list]
df_test = cat_test.join(num_test)


# Now we can seperate our training dataset to 'X' and 'y' variables. We will also create an array of X test values from df_test

# In[ ]:


X = df_train.drop('SalePrice', axis=1).values
y = df_train['SalePrice'].values
df_test_values = df_test.values


# Now let us encode our categorical data.

# In[ ]:


#We import the LabelEncoder and OneHotEncoder classes
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Let us now define a function that will use the classes we imported to encode our data
def encode(X):
    # We create an object of LabelEncoder
    labelencoder = LabelEncoder()
    for i in range(len(cat_list)):
        
        #Using the fit_transform method we will convert each column of the categorical data into
        #numerical values
        
        X[:,i] = labelencoder.fit_transform(X[:,i])
    for i in range(len(cat_list)):
        # Now we will convert the values into dummy variables
        onehotencoder = OneHotEncoder(categorical_features=[i])
        X = onehotencoder.fit_transform(X).toarray()
        #For each column, we will remove the first dummy variable. This is done to avoid dummy variable trap
        X = X[:,i:]
    


# Using our function, we encode the X values of the training and testing datasets

# In[ ]:


encode(X)
encode(df_test_values)


# # Feature scaling

# Our data contains values that are on a different scale. We have little numbers such as our ordinal data that we have looked at as well as large numbers which are in tens of thousands and hundreds of thousands. We should, therefore, perform feature scaling. This will allow our data to be on the same scale and therefore predictions will be more accurate. We will do this for all 'X' values for both the train and test data. Our 'y' variable is log transformed and therefore we choose not to scale it. For this step we will use the StandardScaler class. Such scaling is also known as standardization. It is prefered when the data possibly contains outliers. 

# In[ ]:


#We need to import the StandardScaler class
from sklearn.preprocessing import StandardScaler

#We create an object of the class
sc = StandardScaler()

#We scale both X (training data) and the testing data
X = sc.fit_transform(X)
df_test_values = sc.transform(df_test_values)


# ## Splitting our test data into train and test

# We believe this step is necessary as testing our model will be handy in assessing it's performance before we submit to Kaggle. We therefore split our test data into 80% train and 20% test. The code below does exactly that. Now we are ready to build our model.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# # The Model

# Before showing the model we submitted to Kaggle, we will briefly show our other attempts.

# ### Multiple Linear Regression

# Using all the data we have prepared we build a multiple linear regression model

# We will now define a function to calculate RMSLE (Root Mean Square Logarithmic Error). This is the metric used by Kaggle. Your final score is calculated using this metric. We therefore, copied this function from Kaggle in order to assess our model's performance. By using this function on our train data, we will get an estimate of our final score.

# In[ ]:


import math
def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5


#  We import the LinearRegression class

# In[ ]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()

#We train our model using 80% of the training data and predict
lm.fit(X_train,y_train)
y_pred_reg = lm.predict(X_test)


# Let us see our model's y-intercept

# In[ ]:


lm.intercept_


# Now let us check the RMSLE

# In[ ]:


rmsle(np.exp(y_test), np.exp(y_pred_reg))


# Our score looks good on the training set. Our target was to get a score less than or equal to 0.15 and with this model on Kaggle we achieve a score of 0.13

# ### The Lasso

# Due to the presence of multicollinearity in our data. The Lasso regression might be a more favourable choice than multiple linear regression. Below we will implement the Lasso regression.

# In[ ]:


from sklearn.linear_model import Lasso
lasso_model = Lasso()
lasso_model.fit(X_train,y_train)


# Let us now predict using our model

# In[ ]:


y_pred_lasso = lasso_model.predict(X_test)


# Now let us check the RMSLE

# In[ ]:


rmsle(np.exp(y_test), np.exp(y_pred_lasso))


# Our first attempt at using lasso gave us a Kaggle score greater than 0.41. However we kept on playing with "alpha", giving it different values and our model started reaching our target as we went lower. We then tried to find the most optimal alpha using the LassoCV class. [Here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html) is the sklearn documentation about LassoCV

# ### LassoCV

# LassoCV is similar to Lasso. However LassoCV performs some cross validation to try and find the alpha that produces the optimal result. If you do not specify the type of cross validation to use, it will use the default 3-fold cross validation.

# In[ ]:


from sklearn.linear_model import LassoCV
lcv = LassoCV()
lcv.fit(X_train,y_train)


# Let us see the alpha our LassoCV has calculated below.

# In[ ]:


lcv.alpha_


# Let us see the intercept our LassoCV has calculated below.

# In[ ]:


lcv.intercept_


# Now we can predict the y-values by using our model

# In[ ]:


y_pred_lassocv = lcv.predict(X_test)


# Now we can calculate the RMSLE

# In[ ]:


rmsle(np.exp(y_test),np.exp(y_pred_lassocv))


# Much better isn't it? Now we can combine both the entire training data to train our model for the Kaggle submission. We will now use a Lasso with our calculated alpha.

# In[ ]:


model = Lasso(lcv.alpha_)
model.fit(X,y)
y_pred = model.predict(df_test_values)
predictions = np.exp(y_pred)


# Save the predictions into a csv with columns: 'Id' and 'SalePrice'. This shows the predicted SalePrice for each house (identified by Id).

# In[ ]:


result=pd.DataFrame({'Id':df_test.index, 'SalePrice':predictions})
result.to_csv('submission.csv', index=False)


# # Model Comparison

# ![](https://github.com/smtolo/EDSA/blob/master/comparison_table.PNG)

# The table below shows performances for the different models we implemented. "The Lasso" is our implementation of Lasso with the default alpha and "LassoCV" is the implementation of LassoCV.

# ![Picture](https://user-images.githubusercontent.com/31653400/58634113-c1639d80-82ea-11e9-80b4-30ee39d9667a.PNG)

# Now we plot the predictions of each model against the 20% we reserved for testing. This will show us our model's performance.

# In[ ]:


fig = plt.figure(figsize=(20,10))
plt.subplot(1, 3, 1)
plt.plot(np.arange(len(y_test)), y_test, label='Testing')
plt.plot(np.arange(len(y_pred_reg)), y_pred_reg, label='Regression')
plt.legend()
plt.subplot(1,3,2)
plt.plot(np.arange(len(y_test)), y_test, label='Testing')
plt.plot(np.arange(len(y_pred_lasso)), y_pred_lasso, label="Lasso")
plt.legend()
plt.subplot(1,3,3)
plt.plot(np.arange(len(y_test)), y_test, label='Testing')
plt.plot(np.arange(len(y_pred_lassocv)), y_pred_lassocv, label="Lasso CV")
plt.legend()
plt.show()


# Our models show some accuracy. Well that is except for the Lasso.

# Let us see the results.

# In[ ]:


result.head()


# # References

# 1. A detailed explaination of EDA: https://www.youtube.com/watch?v=zHcQPKP6NpM&t=2s
# 2. A typical solution using Multiple Regression: https://www.youtube.com/watch?v=eZgeYzf2QI4
# 3. Comprehensive data exploration with Python: https://www.kaggle.com/pmarcelino/comprehensive-data-exploration-with-python
# 4. Islr textbook on introduction to statistical learning: http://www-bcf.usc.edu/~gareth/ISL/
# 5. LassoCV: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html
