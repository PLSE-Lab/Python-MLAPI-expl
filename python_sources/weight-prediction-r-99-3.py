#!/usr/bin/env python
# coding: utf-8

# ### INTRODUCTION
# 
# #### Predicting the weight of fish through linear regression
# #### Steps taken in preprocessing includes Data cleaning, Assumption check, Outliers Removal, Standardization etc
# 
# ### SIDE NOTE
# #### You can leave your question about any unclear part in the comment section
# #### Any correction will be highly welcomed

# ### LOADING THE DATA

# In[ ]:


#importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[ ]:


path = '/kaggle/input/fish-market/Fish.csv'


# In[ ]:


df = pd.read_csv(path)


# In[ ]:


df.head(3)


# In[ ]:


df.describe(include = 'all')


# ### DEALING WITH MISSING VALUES

# In[ ]:


df.info()


# #### This dataset is clean

# ### DATASET ANALYSIS AND OUTLIERS REMOVAL

# #### we will plot the distribution of all  the numeric variables in other to be able to identify outliers and any other abnormalities
# #### Outliers will be dealt with by removing either top 1% or the bottom 1%

# In[ ]:


#plotting a distribution plot
sns.distplot(df['Weight']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['Weight'].quantile(0.99)
df = df[df['Weight']<q]

sns.distplot(df['Weight']) 


# In[ ]:


sns.distplot(df['Length1']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['Length1'].quantile(0.99)
df = df[df['Length1']<q]

sns.distplot(df['Length1'])


# In[ ]:


sns.distplot(df['Length2']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['Length2'].quantile(0.99)
df = df[df['Length2']<q]

sns.distplot(df['Length2'])


# In[ ]:


sns.distplot(df['Length3']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['Length3'].quantile(0.99)
df = df[df['Length3']<q]

sns.distplot(df['Length3'])


# In[ ]:


#We need to reset index of the dataframe after droppinh those observation
df.reset_index(drop = True, inplace = True)


# In[ ]:


df.describe()


# In[ ]:


sns.distplot(df['Height']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['Height'].quantile(0.99)
df = df[df['Height']<q]

sns.distplot(df['Height'])


# In[ ]:


sns.distplot(df['Width']) #we can see those few outliers shown by the longer right tail of the distribution


# In[ ]:


#Removing the top 1% of the observation will help us to deal with the outliers
q = df['Width'].quantile(0.99)
df = df[df['Width']<q]

sns.distplot(df['Width'])


# In[ ]:


df.describe()


# ### ASSUMPTION CHECK
# ### 1.The first assumption we will check is No multicollinearity
# #### The minimum variance inflation factor(vif) will be 10

# In[ ]:


df.columns.values


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# the target column (in this case 'weight') should not be included in variables
#Categorical variables already turned into dummy indicator may or maynot be added if any
variables = df[['Length1', 'Length2', 'Length3', 'Height','Width']]
X = add_constant(variables)
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range (X.shape[1]) ]
vif['features'] = X.columns
vif


# #### from the table above 'Length1' and  'Length2' are highly correlated with the rest of variables.
# #### Let's drop them and run the code again

# In[ ]:


df.drop(['Length1','Length2'], axis = 1, inplace = True)


# In[ ]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

variables = df[['Length3', 'Height','Width']]
X = add_constant(variables)
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range (X.shape[1]) ]
vif['features'] = X.columns
vif


# #### After dropping those variables and running the code again the result shows that Non of the variables are having a vif >= 10
# 
# ### 2.The next assumption we will like to check is LINEARITY
# #### Let's plot each numerical variable against our target variable 'Weight' and check to see if the resulting plot is linear

# In[ ]:


fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize =(15,3))
ax1.scatter(df['Length3'], df['Weight'])
ax1.set_title('Length3 and weight')

ax2.scatter(df['Height'], df['Weight'])
ax2.set_title('Height and weight')

ax3.scatter(df['Width'], df['Weight'])
ax3.set_title('Width and weight')


# ####  The resulting plots above gives us a rather curve line
# #### Let's try to correct this by taking the log of  our target variable 'Weight'

# In[ ]:


#Creating a new column in our dataset containing log-of-weight
df['log_weight'] = np.log(df['Weight'])


# In[ ]:


#RE plotting the graphs but this time using 'log_weight' as our target variable
fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize =(15,3))
ax1.scatter(df['Length3'], df['log_weight'])
ax1.set_title('Length3 and log_weight')

ax2.scatter(df['Height'], df['log_weight'])
ax2.set_title('Height and log_weight')

ax3.scatter(df['Width'], df['log_weight'])
ax3.set_title('Width and log_weight')


# #### The above plots till leads to a not so straight line, next we take the log of the numerical variables

# In[ ]:


#Creating new columns to hold the logs of the variables
df['log_length3'] = np.log(df['Length3'])

df['log_width'] = np.log(df['Width'])

df['log_height'] = np.log(df['Height'])


# In[ ]:


fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize =(15,3))
ax1.scatter(df['log_length3'], df['log_weight'])
ax1.set_title('log_weight and log_length 1')
ax2.scatter(df['log_width'], df['log_weight'])
ax2.set_title('log_width and log_weight')
ax3.scatter(df['log_height'], df['log_weight'])
ax3.set_title('log_height and log_weight')


# #### The above graph gives a straight line plot Therefore our Linearity assumption as been satified
# ### 3. The next assumption to check is Normality and Homoscedasticity
# #### Normality is assumed while from the last graphs we can see that homoscedasticity as been achieved
# 
# 
# 
# #### We are going to use the new log columns we created inplace of the former variables so let's drop those former variables

# In[ ]:


#Getting the present variables in our dataframe
df.columns.values


# In[ ]:


#Dropping those columns that as been logged
df = df.drop(['Length3', 'Height', 'Width', 'Weight'], axis = 1)


# In[ ]:


df.describe()


# #### From the the table above log_weight seems to have a minimum value of -inf(infinity) which is useless
# #### Let's locate and drop the row having the -inf value

# In[ ]:


df[df['log_weight'].apply(lambda x: x < 1000)].sort_values('log_weight', ascending = True)


# In[ ]:


#Drop a row with index number 40
df.drop([40], inplace = True)

#Resetting the index after dropping a row
df.reset_index(drop = True, inplace = True)


# ### Dummy Variables
# #### Species is a categorical variable so we need  to turn it into a dummy indicator before we can perform our regression

# In[ ]:


df = pd.get_dummies(df, drop_first = True)


# In[ ]:


df.head()


# ### Standardization
# 
# #### Standardizing helps to give our independent varibles a more standard and relatable numeric scale, it also helps in improving model accuracy

# In[ ]:


#Declaring independent variable i.e x
#Declaring Target variable i.e y
y = df['log_weight']
x = df.drop(['log_weight'], axis = 1)


# In[ ]:


scaler = StandardScaler() #Selecting the standardscaler
scaler.fit(x)#fitting our independent variables


# In[ ]:


scaled_x = scaler.transform(x)#scaling


# ### MULTIPLE LINEAR REGRESSION
# #### It is time to create our model
# #### We will split our dataframe into two, one part for training the other for testing

# In[ ]:


#Splitting our data into train and test dataframe
x_train,x_test, y_train, y_test = train_test_split(scaled_x, y , test_size = 0.2, random_state = 47)


# In[ ]:


reg = LinearRegression()#Selecting our model
reg.fit(x_train,y_train)


# In[ ]:


#predicting using x_train
y_hat = reg.predict(x_train)


# In[ ]:


#Plotting y_train vs our predicted value(y_hat)
fig, ax = plt.subplots()
ax.scatter(y_train, y_hat)


# #### From the plot above we can see that our model was pretty decent in predicting 
# ### 4. No Endogeneity assumption
# #### Lets plot the residual graph to check for No Endogeneity assumption

# In[ ]:


#Residual graph
sns.distplot(y_train - y_hat)
plt.title('Residual Graph')


# #### Our graph generally shows a normal distribution but with a  longer left tail and a sligthly longer right tail
# #### which means that our model tends to over estimate the target(a much higher value is predicted) a lot

# In[ ]:


#R2
reg.score(x_train, y_train)


# #### Our model is explaining 99.5% of the variabilty of the data which is quite excellent

# In[ ]:


#Intercept of the regression line
reg.intercept_


# In[ ]:


#Coefficient
reg.coef_


# In[ ]:


#Predicting with x_test
y_hat_test = reg.predict(x_test)


# In[ ]:


reg.score(x_test, y_test)


# In[ ]:


#Plotting predicted value against y_test
plt.scatter(y_test, y_hat_test, alpha=0.5)
plt.show()


# ####  This plot shows that our model prediction is quite close to the expected values with an R2 of 99.3%

# ### WEIGHT INTERPRETATION

# In[ ]:


#Creating a summary table containing coefficients for each variable
summary = pd.DataFrame( data = x.columns.values, columns = ['Features'] )
summary['Weight'] = reg.coef_
summary


# #### Considering the summary table the higher the weight the higher the impact that means log_length3 is the most impactful feature
# #### A positive weight shows an increase in log_weight and weight respectively
# #### A negative weight shows a decrease in log_weight and weight respectively

# ### CONCLUSION
# #### Let's take a closer look at the expected and predicted values

# In[ ]:


#Creating a new dataframe
df1 = pd.DataFrame( data = np.exp(y_hat_test), columns = ['Predictions'] )


# In[ ]:


#Resetting index to match the index of y_test with that of the dataframe
y_test = y_test.reset_index(drop = True)


# In[ ]:


#target column will hold our predicted values
df1['target'] = np.exp(y_test)


# In[ ]:


#Substrating predictions from target to get the difference in value
df1['Residual'] = df1['target'] - df1['Predictions']

#Difference in percentage
df1['Difference%'] = np.absolute(df1['Residual']/ df1['target'] * 100)


# In[ ]:


df1.describe()


# #### Our minimum Difference in % is 1.09 while our maximum is 22.88

# In[ ]:


df1.sort_values('Difference%')


# ####  We notice that  from the table above the values with the highest Diffence% have both negative and positive residual values
# #### That can also mean that our model tends both overestimate and underestimate its target
# #### Over all our model is a very good model

# In[ ]:





# #### If you find this notebook useful don't forget to upvote. #Happycoding

# In[ ]:




