#!/usr/bin/env python
# coding: utf-8

# ## Simple Linear Regression
# 
# In this notebook, we'll build a linear regression model to predict `Healthcare charges billed by their Insurance provider` using an appropriate predictor variable.

# #### This kernel is based on the assignment by IIITB collaborated with Upgrad.

# #### If this Kernel helped you in any way, some <font color="red"><b>UPVOTES</b></font> would be very much appreciated

# ## Step 1: Reading and Understanding the Data

# In[ ]:


# import all libraries and dependencies for dataframe
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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


# In[ ]:


# Local file path. Please change the file path accordingly

# Read training data
train = pd.read_csv("../input/hospital-charges-simple-lr/insurance_training.csv")
# train.head()

# Read test data
test = pd.read_csv("../input/hospital-charges-simple-lr/insurance_test.csv")
test.head()


# #### Understanding the dataframe

# In[ ]:


train.shape


# In[ ]:


train.info()


# In[ ]:


train.describe()


# ## Step 2: Cleaning the Data

# We need to do some basic cleansing activity in order to feed our model the correct data.

# In[ ]:


# lets drop Unnamed: 0 and bmi
train = train.drop(['Unnamed: 0','bmi'],axis=1)


# In[ ]:


# Calculating the Missing Values % contribution in DF

df_null = train.isna().mean().round(4) * 100

df_null.sort_values(ascending=False).head()


# In[ ]:


train.dtypes


# In[ ]:


# Outlier Analysis of target variable with maximum amount of Inconsistency

outliers = ['charges']
plt.rcParams['figure.figsize'] = [8,8]
sns.boxplot(data=train[outliers], orient="v", palette="Set1" ,whis=1.5,saturation=1, width=0.7)
plt.title("Outliers Variable Distribution", fontsize = 14, fontweight = 'bold')
plt.ylabel("Charges Range", fontweight = 'bold')
plt.xlabel("Continuous Variable", fontweight = 'bold')


# #### Insights: 
# - There are some price ranges above 50000 which can be termed as outliers but lets not remove it rather we will use standarization scaling.

# In[ ]:


# checking for duplicates

train.loc[train.duplicated()]


# In[ ]:


train.head()


# ## Step 3: Visualising the Data
# 
# - Here we will identify if some predictors directly have a strong association with the outcome variable `Charges`

# In[ ]:


# Visualizing the smoker counts

plt.rcParams['figure.figsize'] = [6,6]
ax=train['smoker'].value_counts().plot(kind='bar',stacked=True, colormap = 'Set1')
ax.title.set_text('Smoker or Non-Smoker')
plt.xlabel("Smoker Status",fontweight = 'bold')
plt.ylabel("Count of Smoker/Non-Smoker",fontweight = 'bold')


# #### Insights:
# - More than 800 people are non-smoker.
# - Around 200 people are smoker.

# In[ ]:


# Visualizing the count of males/females 

plt.rcParams['figure.figsize'] = [6,6]
ax=train['sex'].value_counts().plot(kind='bar',stacked=True, colormap = 'Set1')
ax.title.set_text('Sex')
plt.xlabel("Sex",fontweight = 'bold')
plt.ylabel("Count of Males/Females",fontweight = 'bold')


# In[ ]:


# Visualizing the count of BMI groups available

plt.rcParams['figure.figsize'] = [6,6]
ax=train['BMI_group'].value_counts().plot(kind='bar',stacked=True, colormap = 'Set1')
ax.title.set_text('BMI group')
plt.xlabel("BMI group'",fontweight = 'bold')
plt.ylabel("Count of BMI group'",fontweight = 'bold')


# #### Visualizing the distribution of charges

# In[ ]:


plt.figure(figsize=(8,8))

plt.title('Charges Distribution Plot')
sns.distplot(train['charges'])


# - The plots seems to be right skewed, the patients were charged majorly less than 15000 .
# 

# #### Visualising Numeric Variables
# 
# Pairplot of all the numeric variables

# In[ ]:


plt.figure(figsize=(20,8))
warnings.filterwarnings("ignore")
sns.pairplot(train, x_vars=['age', 'sex','children','smoker','region','BMI_group'], y_vars='charges',size=6, aspect=1, kind='scatter')
plt.show()


# In[ ]:


# Let's check the correlation coefficients to see which variables are highly correlated

plt.figure(figsize = (8, 8))
df_corr = train.corr()
ax = sns.heatmap(df_corr, annot=True, cmap="RdYlGn") 
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)


# #### Insights:
# - `Smoker` seems to be high correlated with `charges`
# - `Age` and `bmi` seems to be moderately correlated with `charges`

# In[ ]:


train.head()


# Let's see scatterplot for few correlated variables  vs `Charges`.

# In[ ]:


col_rel = ['age','smoker','BMI_group']


# In[ ]:


# Scatter Plot of independent variables vs dependent variables

fig,axes = plt.subplots(1,3,figsize=(18,6))
for seg,col in enumerate(col_rel):
    x,y = seg//3,seg%3
    an=sns.scatterplot(x=col, y='charges' ,data=train, ax=axes[y])
    plt.setp(an.get_xticklabels(), rotation=45)
   
plt.subplots_adjust(hspace=0.5)


# ### Dividing into X and Y sets for the model building

# In[ ]:


y_train = train.pop('charges')
X_train = train


# ## Step 4: Building a linear model using Bottom down approach

# In[ ]:


# Adding a constant variable and Build a first fitted model
import statsmodels.api as sm  
X_train_c = sm.add_constant(X_train)
lm = sm.OLS(y_train,X_train_c).fit()

#Summary of linear model
print(lm.summary())


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
X = X_train
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# We generally want a VIF that is less than 5 and it look good.

# ### Dropping the variable and updating the model

# *Dropping `region` beacuse its `p-value` is `0.879` and we want p-value less than 0.05 and hence rebuilding the model*

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_1 = X_train.drop('region', 1,)

# Adding a constant variable and Build a second fitted model

X_train_1c = sm.add_constant(X_train_1)
lm1 = sm.OLS(y_train, X_train_1c).fit()

#Summary of linear model
print(lm1.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
X = X_train_1
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# *Dropping `sex` beacuse its `p-value` is `0.875` and we want p-value less than 0.05 and hence rebuilding the model*

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_2 = X_train_1.drop('sex', 1,)

# Adding a constant variable and Build a third fitted model

X_train_2c = sm.add_constant(X_train_2)
lm2 = sm.OLS(y_train, X_train_2c).fit()

#Summary of linear model
print(lm2.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
X = X_train_2
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# *Dropping `BMI_group` beacuse its `p-value` is `0.357` and we want p-value less than 0.05 and hence rebuilding the model*

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_3 = X_train_2.drop('BMI_group', 1,)

# Adding a constant variable and Build a fourth fitted model

X_train_3c = sm.add_constant(X_train_3)
lm3 = sm.OLS(y_train, X_train_3c).fit()

#Summary of linear model
print(lm3.summary())


# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
X = X_train_3
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# * Lets drop `children` and see if there is any significant drop in adj R squared. 

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_4 = X_train_3.drop('children', 1,)

# Adding a constant variable and Build a fifth fitted model

X_train_4c = sm.add_constant(X_train_4)
lm4 = sm.OLS(y_train, X_train_4c).fit()

#Summary of linear model
print(lm4.summary())


# - There is no significant drop in adj R squared.So, we will proceed with dropping `children`

# In[ ]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
X = X_train_4
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# * Lets drop `age` and see if there is any significant drop in adj R squared.*

# In[ ]:


# Dropping highly correlated variables and insignificant variables

X_train_5 = X_train_4.drop('age', 1,)

# Adding a constant variable and Build a sixth fitted model

X_train_5c = sm.add_constant(X_train_5)
lm5 = sm.OLS(y_train, X_train_5c).fit()

#Summary of linear model
print(lm5.summary())


# - There is a significant drop after dropping age. So, lets not drop `age`

# - Lets build model with `lm4` which has basically 2 predictor variables.

# ## Step 5: Residual Analysis of the train data
# 
# So, now to check if the error terms are also normally distributed (which is infact, one of the major assumptions of linear regression), let us plot the histogram of it.

# In[ ]:


# Predicting the price of training set.
y_train_charges = lm4.predict(X_train_4c)


# In[ ]:


# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_charges), bins = 20)
fig.suptitle('Error Terms Analysis', fontsize = 20)                   
plt.xlabel('Errors', fontsize = 18)


# - Here, residuals are following normal distribution with a mean 0. Looks good

# ## Step 6: Making Predictions Using the Final Model
# 
# Now that we have fitted the model and checked the normality of error terms, it's time to go ahead and make predictions using the `lm4` model.

# #### Dividing test set into X_test and y_test

# In[ ]:


X_test = test


# In[ ]:


# Adding constant
X_test_1 = sm.add_constant(X_test)

X_test_new = X_test_1[X_train_4c.columns]


# In[ ]:


# Making predictions using the final model

y_pred = lm4.predict(X_test_new)
y_pred = pd.DataFrame(y_pred)
y_pred = y_pred.rename(columns={0:'pred charges'})


# In[ ]:


merge = pd.concat([X_test,y_pred],axis=1)


# #### Equation of Line to predict the Hospital charges

# ### $ charges = -2387.3678 +  276  \times  age  + 2.369e+04  \times  smoker $

# #### Model Conclusions:
# - R-squared and Adjusted R-squared - 0.713 and 0.712 - 70% variance explained.
# - F-stats and Prob(F-stats) (overall model fit) - 1323 and 1.12e-289(approx. 0.0) - Model fit is significant and explained 70%<br> variance is just not by chance.
# - p-values - p-values for all the coefficients seem to be less than the significance level of 0.05. - meaning that all the <br>predictors are statistically significant.

# ### If this Kernel helped you in any way, some <font color="red"><b>UPVOTES</b></font> would be very much appreciated
