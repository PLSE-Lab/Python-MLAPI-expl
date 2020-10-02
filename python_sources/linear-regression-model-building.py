#!/usr/bin/env python
# coding: utf-8

# @Author: Tushar

# > Problem Statement
# A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and 
# producing cars locally to give competition to their US and European counterparts. 
# They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, 
# they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the
# Chinese market. The company wants to know:
# Which variables are significant in predicting the price of a car
# How well those variables describe the price of a car
# Based on various market surveys, the consulting firm has gathered a large dataset of different types of cars across the Americal 
# market.  
# 
# > Business Goal 
# You are required to model the price of cars with the available independent variables. It will be used by the management to understand
# how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business 
# strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics
# of a new market.

# In[ ]:


'''importing the required libraries
'''
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import seaborn as sns
import matplotlib.pyplot as plt

# Supress Warnings

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


'''
    Please update the location of the CSV file.
    reading the dataset from the required location
'''
df = pd.read_csv(r'../input/carprice-assignment/CarPrice_Assignment.csv')
df.describe()


# In[ ]:


df.info()


# > The df.info shows we have no nan/empty columns, now we will split the car name into two columns company_name and carname

# In[ ]:


'''
    splitting the column name CarName to both carname and company_name
'''
df[['company_name','CarName']] = df.CarName.apply(lambda x: pd.Series(str(x).split(" ",1)))


# > now, we have the data and it is divided into the minimum columns, which were expected till this point As a next step we will look for the data cleaning, and see if we have got the correct company/column names or not

# In[ ]:


df.info()


# > after splitting also we have non null values, we will check for carname and company name values now

# In[ ]:


'''
    checking for the data quality in the column CarName
'''
df.CarName.unique()


# In[ ]:


'''
    checking for the data quality in the company_name
'''

df.company_name.unique()


# 1. 'maxda', 'mazda', 'Nissan', 'nissan','porsche','porcshce','toyota', 'toyouta','vokswagen', 'volkswagen',
# 
# > From the above value pairs it is clear that there are some typos in this, hence we will correct all of this.
# Correction will include -:
# changing maxda to mazda changing Nissan to nissan --to bring similarity in the data changing porcshce to porsche changing toyouta
# to toyota changing vokswagen to volkswagen

# In[ ]:


'''
    replacing the column vaues to correct the typing mistakes to resolve the data quality issues
'''

df['company_name'] = df['company_name'].replace('maxda', 'mazda')
df['company_name'] = df['company_name'].replace('Nissan', 'nissan')
df['company_name'] = df['company_name'].replace('porcshce', 'porsche')
df['company_name'] = df['company_name'].replace('toyouta', 'toyota')
df['company_name'] = df['company_name'].replace('vokswagen', 'volkswagen')
df['company_name'] = df['company_name'].replace('vw', 'volkswagen')


# In[ ]:


'''verifying that the data quakity issues are no longer present in the data set'''
df.company_name.unique()


# > now all of the company names have been corrected, lets plot the graph and see if there is any relation between the columns or not

# In[ ]:


#start visualising
sns.pairplot(df)
plt.show()

# we should go with linear regresssion because for few variables we can see a 
#positive co-relation between the numerical variables


# 1. In most of the columns in the above graph we can see a positive co-reation, eg columns like -: wheelbase,carlength,carwidth,enginetype etc have positive or negative correlation and that too with all the columns linearly
# 
# 2. Hence we will go for multiple linear regression as we can see such linear patterns in almost all the columns and few of which seems to be in a straight line eg-: 
#   1. engine size and horse power 
#   2. wheelbase and carlength 
#   3. wheelbase and curbweight
#   4. curbweight and engine size 
#   5. curbweight and carlength

# In[ ]:


#in order to visualise a categorical variable we should use a box plot
plt.figure(figsize=(30, 18))

plt.subplot(3, 4, 1)
sns.boxplot(x = 'enginetype', y = 'price', data = df)

plt.subplot(3, 4, 2)
sns.boxplot(x = 'fueltype', y = 'price', data = df)

plt.subplot(3, 4, 3)
sns.boxplot(x = 'aspiration', y = 'price', data = df)

plt.subplot(3, 4, 4)
sns.boxplot(x = 'doornumber', y = 'price', data = df)

plt.subplot(3, 4, 5)
sns.boxplot(x = 'carbody', y = 'price', data = df)

plt.subplot(3, 4, 6)
sns.boxplot(x = 'drivewheel', y = 'price', data = df)

plt.subplot(3, 4, 7)
sns.boxplot(x = 'carbody', y = 'price', data = df)

plt.subplot(3, 4, 8)
sns.boxplot(x = 'cylindernumber', y = 'price', data = df)

plt.subplot(3, 4, 9)
sns.boxplot(x = 'fuelsystem', y = 'price', data = df)

plt.subplot(3, 4, 10)
sns.boxplot(x = 'symboling', y = 'price', data = df)
#boxplot boundaries represents - 25%, median, 75 %


# 1. here we have less amount of data, hence we can not remove outliers as they can significantly imapct the calculation of p-value, and slope of the line 
# 2. lesser variation can be observed with the columns like doornumber 
# 3. fuel type and aspiration have significance variation/impact where as drive wheel, carbody, fuelsystem, cylindernumber, drivewheel have significant impact

# In[ ]:


'''plotting the heatmap to find the correlation amongst the columns'''
plt.figure(figsize=(20,12))
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")
plt.show()


# > even from the heatmap it is visible that the correlation of price is high with the below -:
# 
# - curbweight and enginesize highest
# - horsepower,carwidth,highwaympg,carlength etc

# In[ ]:


# creating dummy variables for all the categorical columns

dummy_var = ['carbody','symboling','fuelsystem','cylindernumber','drivewheel','carbody','doornumber','aspiration',
              'fueltype','enginetype','company_name']
dummy_var_df = pd.get_dummies(df[dummy_var],drop_first=True)

dummy_var_df.head()


# In[ ]:


'''now concat the dummy data frame with a main dataframe'''
df = pd.concat([df,dummy_var_df],axis=1)
df.head()


# In[ ]:


'''drop the columns for which the dummy variables are already created'''
df = df.drop(dummy_var,axis=1)
df.head()


# > Since, we have high values, hence we will do the rescaling
# 
# > Rescaling
# After train and split we will do rescaling
# 
# > why we do rescaling
# eg here the values of price are very high comparing to the number of stroke hence the coefficeint of price will be very smaller in comparison to stroke 
# 
# > eg if coefficients are _ price = 0.0005 stroke = 400.
# - This does not mean that stroke is insignificant, this is one of the advantages of rescaling 
# - It gives the results in range of 0-1 3) minimization routine/ optimization increases
# 
# > How to rescale ?
# 1. Min-Max scaling or normalization between 0 and 1
# 2. Standardization (mean -0, sigma 1)
# 
# > Min-Max scaling or normalization ?
# - it says (x-xmin)/(xmax-xmin) - this is how the value is converted to a range from 0 to 1
# - Standardization -: (x - mu)/sigma - this gives a 0 mean and 1 sigma value

# In[ ]:


'''generating the train and test data set'''
df_train, df_test= train_test_split(df,train_size=0.7,random_state=100)
print(df_train.shape)
print(df_test.shape)


# In[ ]:


df_train.info()


# In[ ]:


'''convert the ctaegorical column enginelocation to a continuous variable'''
df_train.enginelocation.unique()


# In[ ]:


df_train.enginelocation.value_counts()


# In[ ]:


df['enginelocation'] = df['enginelocation'].replace('front', '1')
df['enginelocation'] = df['enginelocation'].replace('rear', '0')


# In[ ]:


# min -max scaling

# 1. Instantiate the objest of the imported class

scaler = MinMaxScaler()

num_variables =['wheelbase','carlength','carwidth','carheight','curbweight','enginesize','boreratio','stroke','compressionratio','horsepower','peakrpm','citympg','highwaympg','price']

#2. Fit on data
df_train[num_variables] = scaler.fit_transform(df_train[num_variables])
df_train.head()


# >fit()
# - it basically computes the xmax and xmin
# - we never do the fit() on the test set so that it does not learn anything
# 
# > transform()
# - It computes the (x-xmin)/(xmax-xmin)
# - you fit the scaler on the training data set and you transform the test data set
# 
# > fit_transform()
# - does both

# In[ ]:


#df_train = df_train[num_variables]
df_train.enginelocation.unique()


# > Step-3 Training the model

# In[ ]:


y_train = df_train.pop('price')
X_train = df_train


# In[ ]:


y_train.head()


# In[ ]:


X_train.pop('CarName')
X_train.pop('enginelocation')


# In[ ]:


# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 10)             # running RFE
rfe = rfe.fit(X_train, y_train)

X_train.info()


# In[ ]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[ ]:


col = X_train.columns[rfe.support_]
col


# In[ ]:


X_train.columns[~rfe.support_]


# > Step 4: Building the model now

# In[ ]:


# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]


# In[ ]:


X_train_rfe = sm.add_constant(X_train_rfe)


# In[ ]:


lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model


# In[ ]:


#Let's see the summary of our linear model
print(lm.summary())


# - cylindernumber_three is insignificant and hence it can be dropped, as it has a higher p-value

# In[ ]:


X_train_new = X_train_rfe.drop(["cylindernumber_three"], axis = 1)


# In[ ]:


X_train_lm = sm.add_constant(X_train_new)


# In[ ]:


lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model


# In[ ]:


print(lm.summary())


# - we will drop the constant to find out the variance now

# In[ ]:


X_train_new = X_train_new.drop(['const'], axis=1)


# In[ ]:


# Calculate the VIFs for the new model
vif = pd.DataFrame()
X = X_train_new
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# - Now, we will be dropping the boreratio as it has the highest p-value amongst all the features and VIF is also high

# In[ ]:


X_train_new = X_train_new.drop(['boreratio'], axis=1)


# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print(lm.summary())


# In[ ]:


'''Dropping the company name company_name_porsche feature as it has the highest p-value'''
X_train_new = X_train_new.drop(['company_name_porsche'], axis=1)


# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print(lm.summary())


# In[ ]:


'''removing the curbweight feature because of high p-value'''
X_train_new = X_train_new.drop(['curbweight'], axis=1)


# In[ ]:


vif = pd.DataFrame()
X = X_train_lm
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[ ]:


'''dropping the feature company_name_subaru as it has very high VIF and this shows multi collinearity'''
X_train_new = X_train_new.drop(['company_name_subaru'], axis=1)


# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print(lm.summary())


# In[ ]:


'''dropping the feature enginetype_ohcf as it has very high p-value'''
X_train_new = X_train_new.drop(['enginetype_ohcf'], axis=1)


# In[ ]:


X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
print(lm.summary())


# In[ ]:


vif = pd.DataFrame()
X = X_train_lm
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# > Residual Analysis of the train data
# - So, now to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression)
# -  let us plot the histogram of the error terms and see what it looks like.

# In[ ]:


y_train_price = lm.predict(X_train_lm)
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label


# ## Making Predictions
# Similarly Scaling the test data as well.

# In[ ]:


df_test[num_variables] = scaler.transform(df_test[num_variables])


# In[ ]:


df_test.describe()


# In[ ]:


y_test = df_test.pop('price')
X_test = df_test


# In[ ]:


# Now let's use our model to make predictions.

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_new.columns]

# Adding a constant variable 
X_test_new = sm.add_constant(X_test_new)


# In[ ]:


# Making predictions
y_pred = lm.predict(X_test_new)


# > Step 5: Model Evaluation

# In[ ]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label


# In[ ]:


# evaluation
r2_score(y_true=y_test, y_pred = y_pred)


# - y_test vs y_pred - seems to be in a linear fashion as it should be. Because if it all they were not similar the graph would not have come along y=x line
# 
# - r2_score for a prediction data set was found to be 0.790 vs (R2 and adj R2) for a test data set was 0.864 and 0.860 respectively This shows that the model has been trained properly as there is no significant drop in the R2 value
# 
# - final equation would look like -: y = ax1 + bx2 + cx3 + ....+ nxn
# 
# - y = 2.78*enginesize + 2.53*carwidth + 1.12*enginetype_rotor + company_name_bmw*1.08 + 7.58
