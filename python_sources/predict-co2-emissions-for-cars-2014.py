#!/usr/bin/env python
# coding: utf-8

# ## <font color='r'> Regression Model to PREDICT CO2 Emissions for Cars </font>

# #### Import  Libraries

# In[ ]:


# For Data operations
import numpy as np
import pandas as pd

# For Viz's
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# For removing warnings (if any)
import warnings
warnings.filterwarnings('ignore')

# For basic statistics
from scipy import stats

# For Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# #### Read the data set

# In[ ]:


co2data = pd.read_csv("../input/FuelConsumption.csv",index_col='MODELYEAR')


# In[ ]:


co2data.head()


# #### Describe the data to check quick stats

# In[ ]:


co2data.describe(include='all')


# #### Looking at the data above the key features can be determined 
# #### 1) Categorical Variable and 2) Continous Variables
# #### Categorical Variables : Make, Vehicle Class , Cylinders, Transmission,  FuelType  (Make, Vehicle Class and Transmission will still need to analysed if these are required for regression analysis)
# #### Continous numerical variables : EngineSize, Fuel Consumption (City, Highway and Combined)

# In[ ]:


#co2data.MAKE.value_counts()  
## Quick look at MAKE variable. This doesn't sound a good variable for regression analysis


# In[ ]:


co2data.VEHICLECLASS.value_counts()  
## This could be potential variable.  Perhaps, we can group the data by Mid Size, Large Size etc


# In[ ]:


co2data.CYLINDERS.value_counts() ## Potential variable 


# In[ ]:


#co2data.TRANSMISSION.value_counts() 
## Not a potential variable


# In[ ]:


co2data.FUELTYPE.value_counts()  ## Potential variable


# In[ ]:


sns.swarmplot(x='FUELTYPE',y='CO2EMISSIONS',data=co2data)


# In[ ]:


sns.swarmplot(x='CYLINDERS',y='CO2EMISSIONS',data=co2data)


# ### Quick look at box plots to confirm IQR for Catergorical variables

# In[ ]:


sns.boxplot(x='MAKE',y='CO2EMISSIONS',data=co2data)


# In[ ]:


sns.boxplot(x='VEHICLECLASS',y='CO2EMISSIONS',data=co2data)


# In[ ]:


sns.boxplot(x='TRANSMISSION',y='CO2EMISSIONS',data=co2data)


# In[ ]:


sns.boxplot(x='FUELTYPE',y='CO2EMISSIONS',data=co2data)


# In[ ]:


sns.boxplot(x='CYLINDERS',y='CO2EMISSIONS',data=co2data)


# #### For cylinders there is no overlap on the IQR , hence this is potential variable that can be used. Remaining all are pretty much overalpping. There won't be a big difference if we add rest of tge varaibles for analysis

# #### Looking at value counts , we can determine the categorical variables along with numerical cont variables that are requred
# #### CAT VARIABLES > CYLINDERS, FUELTYPE
# ####  NUM CONT VARIABLES > ENGINESIZE, FUEL_CONSUMPTION_COMB
# #### All the variables above are independent variabled that can be used to predict CO2 Emissions of the given car
# 
# ** Because of large set of data points <b>FuelType</b> will add value in the anaylsis

# ### Subsetting the data frame

# In[ ]:


df = co2data[['FUELTYPE','CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB','CO2EMISSIONS']]


# In[ ]:


def categorise(a):
    if a == 'Z':
        return 1
    elif a == 'D':
        return 2
    elif a == 'X':
        return 3
    else:
        return 4


# In[ ]:


df['FUELTYPE_CAT'] = df['FUELTYPE'].apply(lambda x : categorise(x))


# In[ ]:


sns.pairplot(df)


# #### Converting the fuel types to numerical categorical variabled

# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(co2data.corr(),annot=True,cmap='coolwarm')


# #### Try looking at LM plots with HUE as Cylinders given that its a good categorical variable and would help further regression analysis

# In[ ]:


sns.lmplot('ENGINESIZE','CO2EMISSIONS',df,order=1)


# In[ ]:


sns.lmplot('FUELCONSUMPTION_COMB','CO2EMISSIONS',df)


# In[ ]:


sns.lmplot('ENGINESIZE','CO2EMISSIONS',hue='CYLINDERS',data=df)


# In[ ]:


sns.lmplot('FUELCONSUMPTION_COMB','CO2EMISSIONS',hue='CYLINDERS',data=df)


# #### A healthy correlation was noted on above graphs.  There is spread noted when Cylinder is added to the LMPLOT.  Given that Cylinder is impact the regression we should consider this for better prediction

# #### A quick look at residual plots

# In[ ]:


sns.residplot('ENGINESIZE','CO2EMISSIONS',df,color='g')


# In[ ]:


sns.residplot('FUELCONSUMPTION_COMB','CO2EMISSIONS',df,color='g')


# #### Since the points are sparsely dispersed , Enginesize is a good canditate for linear regression analysis

# ### Pearson's correlation (P-Values)

# In[ ]:


def thresh_pvalue(p_value):
    if p_value <= 0.001:
        print("The p_value is {:f} is less than threshold of 0.001 and is strong fit for regression analysis".format(float(p_value)))
    elif ((p_value > 0.001) & (p_value < 0.05)):
        print("The p_value is {:f} is less than threshold of 0.05 and greater than 0.001 and is moderate fit for regression analysis".format(float(p_value)))
    elif ((p_value > 0.05) & (p_value < 0.1)):
        print("The p_value is {:f} is less than threshold of 0.1  and greater than 0.05 abd is a weak fit for regression analysis".format(float(p_value)))
    else:
        print("The p_value is {:f} is greater than 0.1 and is not a good fit for regression analysis".format(float(p_value)))


# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['ENGINESIZE'], df['CO2EMISSIONS'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
thresh_pvalue(p_value)


# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['CYLINDERS'], df['CO2EMISSIONS'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
thresh_pvalue(p_value)


# In[ ]:


pearson_coef, p_value = stats.pearsonr(df['FUELCONSUMPTION_COMB'], df['CO2EMISSIONS'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value) 
thresh_pvalue(p_value)


# #### Hence, all three are good for regression analysis to get correct picture/model for prediction

# ## Linear Regression Model Prediction

# In[ ]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


## Instantaniate linear regression constructor
regr = linear_model.LinearRegression()

## Define Predictors
Predictors = ['FUELTYPE_CAT','ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']
x = df[['FUELTYPE_CAT','ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
y = df['CO2EMISSIONS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


## Fit the model
a = regr.fit (x_train, y_train)

# The coefficients and intercept
print("The Y-Intercept is",regr.intercept_, " with the slope value of ", regr.coef_)

## Run the prediction
y_hat= regr.predict(x_test[Predictors])


# Print residual errors
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - y_test) ** 2))
rmse = np.sqrt(mean_squared_error(y_test,y_hat))
print("Root Mean Squared Error: {}".format(rmse))
#print("R2-score: %.2f" % r2_score(y_hat , y_test) )
print("R^2-score: %.2f" % regr.score(x_test , y_test) )

#print(a.summary())


# ## Polynomial Regression Model Prediction

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

## Define Predictors
Predictors = ['FUELTYPE_CAT','ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']
x = df[['FUELTYPE_CAT','ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']]
y = df['CO2EMISSIONS']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

## Fit the model
poly = PolynomialFeatures(degree=5)
train_x_poly = poly.fit_transform(x_train)

clf = LinearRegression()
train_y_ = clf.fit(train_x_poly, y_train)

# The coefficients
print ('Coefficients: ', clf.coef_)
print ('Intercept: ',clf.intercept_)

## Run the prediction
test_x_poly = poly.fit_transform(x_test)
test_y_ = clf.predict(test_x_poly)

# Print residual errors
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - y_test) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , y_test) )
#print("R^2-score: %.2f" % regr.score(x_test , y_test) )

#rmse = np.sqrt(mean_squared_error(y_test,y_hat))
#print("Root Mean Squared Error: {}".format(rmse))
#print("R2-score: %.2f" % r2_score(y_hat , y_test) )


# #### Predicted values for test dataset

# In[ ]:


final_df = pd.DataFrame(x_test)
final_df.columns = Predictors
final_df['CO2EMISSIONS'] = y_test
final_df['CO2EMISSIONS_PRE'] = test_y_  ## Replace yHat with test_y_ to enable polynomial values
final_df.head()


# #### MODEL FITTING MULTI LINEAR REGRESSION

# In[ ]:


ax2 = sns.distplot(df['CO2EMISSIONS'],color='b',hist=True,label='Actual')
sns.distplot(y_hat,color='r',label='Predicted',hist=True,ax=ax2)


# In[ ]:


sns.distplot((y_hat-y_test), color='r')


# In[ ]:


sns.regplot(y_test, y_hat)


# #### MODEL FITTING POLYNOMIAL REGRESSION

# In[ ]:


ax2 = sns.distplot(df['CO2EMISSIONS'],color='b',hist=True,label='Actual')
sns.distplot(test_y_,color='r',label='Predicted',hist=True,ax=ax2)


# In[ ]:


sns.distplot((test_y_-y_test), color='r')


# In[ ]:


sns.regplot(y_test, test_y_)


#  # Model fits very well for Polynomial regression
