#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Linear Regression Problem
# 
# ## Car Production Trend in the US
# 
# #### Problem Statement
# Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts. 
# 
# They want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. The company wants to know:
# 
# * Which variables are significant in predicting the price of a car
# * How well those variables describe the price of a car
# 
# We will be modeling the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics of a new market.

# ## Step 1: Data Preparation
# Let's make the data more readable and understandable

# In[ ]:


#importing numpy and pandas libraries and also ignoring the warning if we may get
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd


# In[ ]:


cars = pd.read_csv("/kaggle/input/geely-auto/CarPriceAssignment.csv")  #Reading data from the csv file provided


# In[ ]:


#Check the head of the data
pd.set_option('display.max_columns', 500) #So that we can see all columns
cars.head()


# In[ ]:


#Car name includes company and model name. We are interested in company name only.
carName = cars.CarName.str.split(" ", n = 1, expand = True)
cars['CarCompany'] = carName[0]
cars.drop('CarName', axis = 1, inplace=True)
cars.CarCompany.unique()


# Clearly we can see there are some wrong values for car company names. Let's fix them.

# In[ ]:


cars['CarCompany'] = cars.CarCompany.str.upper()
cars['CarCompany'] = cars['CarCompany'].replace({'VW':'VOLKSWAGEN','VOKSWAGEN':'VOLKSWAGEN','TOYOUTA':'TOYOTA','PORCSHCE':'PORSCHE','MAXDA':'MAZDA'})
cars.CarCompany.unique()


# In[ ]:


#Getting the data type of each column
cars.info()


# In[ ]:


#Getting shape of the data
cars.shape


# We have 205 rows and 26 columns

# In[ ]:


#Removing duplicates if any
cars = cars.drop_duplicates(keep=False)


# In[ ]:


#Getting the description of the data
cars.describe()


# In[ ]:





# ## Step 2: Visualizing the data
# Here we will try to get a heads up for the multicolinearity and the proper predictors

# In[ ]:


#Importing apropriate libraries
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


sns.pairplot(cars)
plt.show()


# In[ ]:


#Heat map for the data
plt.figure(figsize=(12,10))
sns.heatmap(cars.corr(), annot=True)


# In[ ]:


#Since the pair plot and heatmap are very conjusted, let's try to get the correlation of each variable using .corr funtion
cars.corr()


# Clearly we can see the PRICE is highly correlated with curbweight, enginesize, horsepower and boreratio.
# And from heat map we can see the following sets are correlated
# * wheelbase carwidth carlength
# * carwidth carlength curbweight
# * curbweight enginesize
# * enginesize horsepower
# * citympg highwaympg
# 
# This data would be usefull in case of removing the multicolinearity

# Our data has some categorical columns as well. Let us visualize them as well

# #### Visualizing categorical variables

# In[ ]:


#From the data dictionary we can get the categorical varibles. Let's plot boxplot for a few of them
plt.figure(figsize=(20, 12))
plt.subplot(3,3,1)
sns.boxplot(x='fueltype',y='price',data = cars)
plt.subplot(3,3,2)
sns.boxplot(x='aspiration',y='price',data = cars)
plt.subplot(3,3,3)
sns.boxplot(x='doornumber',y='price',data = cars)
plt.subplot(3,3,4)
sns.boxplot(x='carbody',y='price',data = cars)
plt.subplot(3,3,5)
sns.boxplot(x='drivewheel',y='price',data = cars)
plt.subplot(3,3,6)
sns.boxplot(x='enginelocation',y='price',data = cars)
plt.subplot(3,3,7)
sns.boxplot(x='enginetype',y='price',data = cars)
plt.subplot(3,3,8)
sns.boxplot(x='cylindernumber',y='price',data = cars)
plt.subplot(3,3,9)
sns.boxplot(x='fuelsystem',y='price',data = cars)
plt.show()


# ### Inferences from the above boxplots
# * Diesel cars are relatively expensive compared to gas/petrol
# * Turbo cars are almost 5000 dollars expensive to the standard cars on an average
# * There is not much difference in price of the car on the number of doors.
# * Hartop and Convertible cars are in the expensive segment.
# * RearWheelDrive RWD cars cost more when compared to FWD and 4WD
# * Engine location makes a difference in price. Sports and luxury cars like Jaguar put the engine in the rare side
# * OHCV engine cars usually cost more than other type of engine cars.
# * The 6 and 8 cylinder cars are famous and hence expensive than others on an average.
# * MPFI and IDI fuelsystem increase the car cost.

# #### But does price is related to car company?

# In[ ]:


plt.figure(figsize=(16,9))
sns.boxplot(x='CarCompany',y='price',data = cars)
plt.xticks(rotation=90);


# Yes, we can see companies like BMW, AUDI, PORSCHE and BUICK prefer to sell their cars at the higher price when compared to others.

# ### Step 3: Data Prepartion for the ML Model
# * For the machine learning model we need to do some data preparation on the exsisting model.
# * We will be labelling and making dummies of the values.

# In[ ]:


cars.head()


# In[ ]:


# To get the list which are not of integer type
cars.select_dtypes(exclude=['int64', 'float64']).columns


# #### We need to deal with the following columns
# * doornumber
# * fueltype
# * carbody
# * drivewheel
# * enginelocation
# * enginetype
# * cylindernumber
# * fuelsystem
# * CarCompany

# #### Encoding variables
# Here we will try to map values to integer values

# In[ ]:


# Let change the doornumber from string to integer values
cars['doornumber'] = cars['doornumber'].apply(lambda x: 2 if x=='two' else (4 if x=='four' else 0))


# In[ ]:


# Now we will check for if the car is diesel or not
cars.fueltype = cars.fueltype.apply(lambda x: 0 if x=='gas' else 1)
cars['DieselCar'] = cars.fueltype
cars = cars.drop('fueltype', axis = 1)
cars.head()


# In[ ]:


# Let map the car has the engine in front side to 1 else 0
cars.enginelocation = cars.enginelocation.apply(lambda x: 0 if x=='rear' else 1)
cars['FrontEngine'] = cars.enginelocation
cars = cars.drop('enginelocation', axis = 1)
cars.head()


# In[ ]:


# We'll map 1 for the TURBO cars

cars.aspiration = cars.aspiration.apply(lambda x: 0 if x=='std' else 1)
cars['Turbo'] = cars.aspiration
cars = cars.drop('aspiration', axis = 1)
cars.head()


# In[ ]:


# For engine type
cars.enginetype.value_counts()


# If we combine engines having OHC (OverHead Camshaft) we can see it is in the more than 92% of the data. So we can make one column which tells if the car has OHC engine or not

# In[ ]:


cars['OHCEngine'] = cars['enginetype'].apply(lambda x: 1 if 'ohc' in x else 0)
cars = cars.drop('enginetype', axis = 1)
cars.head()


# The *RISK FACTOR* is dependent on the symboling and for better readablity we can change it to a new column. For the higher values will tell the car is more risky.

# In[ ]:


def risk_factor(x):
    m = {-3:0,-2:0,-1:1,0:2,1:3,2:4,3:5} # 5 means most risky car
    return m.get(x)

cars['RiskFactor'] = cars.symboling.apply(risk_factor)
cars = cars.drop('symboling', axis = 1)
cars.head()


# The `cylindernumber` can also be converted to number

# In[ ]:


def get_cyl_num(x):
    m = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12}
    return m.get(x)

cars.cylindernumber = cars.cylindernumber.apply(get_cyl_num)
cars.head()


# For `drivewheel` we can use *Dummy Variables*

# In[ ]:


status = pd.get_dummies(cars['drivewheel'], drop_first=True)  #It will create dummy variables.
status.head()


# Since the `drivewheel` can presented in 2 columns without lossing any information. For example if fwd = 0 and rwd =0 it is 4wd

# In[ ]:


#Concating status and cars dataframe
cars = pd.concat([cars,status], axis = 1)
cars = cars.drop('drivewheel',axis = 1)   #Dropping `drivewheel` as it not required
cars.head()


# In[ ]:


#Let's deal with carbody now.
# We can categories each carbody using category codes

#First let's check the current carbody values
print(cars.carbody.unique())


# In[ ]:


#Now we will convert each value to a code
cars.carbody = cars.carbody.astype('category')
cars.carbody = cars.carbody.cat.codes
print(cars.carbody.unique())


# So Python has assigned a code for each value of `carbody`.
# 
# The converstion reference is given below:
# * convertible = 0
# * hatchback = 2
# * sedan = 3
# * wagon = 4
# * hardtop = 1

# In[ ]:


cars.head()


# Similarly, we can assign codes for `fuelsystem`.

# In[ ]:


#Current fuelsystem values are
print(cars.fuelsystem.unique())


# In[ ]:


#Let encode the values
cars.fuelsystem = cars.fuelsystem.astype('category')
cars.fuelsystem = cars.fuelsystem.cat.codes
print(cars.fuelsystem.unique())


# The value before and after conversion is provided below:
# * mpfi=5
# * 2bbl=1
# * mfi=4
# * 1bbl=0
# * spfi=7
# * 4bbl=2
# * idi=3
# * spdi=6

# In[ ]:


cars.head()


# #### We need to find if the Company Name make any impact on the price. So instead of encoding it we will make it's dummy variables.

# In[ ]:


#The way we created the dummy variables for the previous columns, we will follow the same procedure
Car_Name_df = pd.get_dummies(cars.CarCompany)
Car_Name_df.head()


# Here we can delete one column as it will not add or loss any information to the dataset

# In[ ]:


#Let's check the counts for each brand
cars.CarCompany.value_counts()


# In[ ]:


#We can see, MERCURY car has only one value, we can delete it from Car_Name_df dataframe
Car_Name_df = Car_Name_df.drop('MERCURY', axis = 1)
#Then we can concate it with the main dataset i.e. cars
cars = pd.concat([cars,Car_Name_df], axis = 1)
cars.drop('CarCompany', axis=1, inplace=True)
cars.head()


# ### Till here we have converted all categorical values to a numeric value and now we can go ahead with the Machine Learning Modeling

# ## Step 4: Splitting the Data into Training and Testing Sets
# We will split the data into two sets, i.e. Training and Testing.

# In[ ]:


from sklearn.model_selection import train_test_split

# We specify this so that the train and test data set always have the same rows, respectively
df_train, df_test = train_test_split(cars, train_size = 0.7, test_size = 0.3, random_state = 100)


# We can see the values in the data set is not scaled as we have curbweight in 2000 range where as cylindernumber in 1-12 range.
# To make it in to a comparable scale we need to do *Scaling*. I have used Min Max scaling for this assignment.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler #Importing library
scaler = MinMaxScaler()   #Getting MinMaxScaler


# In[ ]:


num_vars = [ 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight','enginesize','boreratio', 'stroke','compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg','price']
#In num_vars we have columns that we need to scale.
df_train[num_vars] = scaler.fit_transform(df_train[num_vars])


# In[ ]:


df_test.head()


# In[ ]:


df_train.describe()


# #### Dividing into X and Y sets for the model building

# In[ ]:


y_train = df_train.pop('price')   #Dependent Variable
X_train = df_train


# ## Step 5: Linear model building using RFE
# We will use RFE feature from the SciKit learn library to which will help us to choose the columns in an automated fashion.
# But we will be using mannual elimination as well if required

# In[ ]:


# Importing RFE and LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[ ]:


# Running RFE with the output number of the variable equal to 10
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, 10)             # running RFE
rfe = rfe.fit(X_train, y_train)


# In[ ]:


# Checking which columns were selected by the RFE
col = X_train.columns[rfe.support_]
print(col)


# ### Building model using statsmodel, for the detailed statistics

# #### Model #1

# In[ ]:


# Creating X_test dataframe with RFE selected variables
X_train_rfe = X_train[col]


# In[ ]:


#Since statsmodels don't consider the constants or intercept by default, we will add constant
import statsmodels.api as sm  
X_train_rfe_lm = sm.add_constant(X_train_rfe)


# In[ ]:


lm1 = sm.OLS(y_train,X_train_rfe_lm).fit()    #Running Linear Model and filling the train data


# In[ ]:


print(lm1.summary())


# We can see here p values are pretty low and all columns suggest by RFE has significance.

# #### We should check for the _Multi-colinearity_ in our model, as it accfects the Interpretation and Inference of the model. In order to keep the _Multi-colinearity_ we should check *VIF* values and it should be low <5.

# In[ ]:


#VIF Calculation for Model #1
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending= False)
print(vif)


# Clearly the VIF value for `carwidth` is too high. We need to delete it

# ##### We will be following the above procedure till we get proper significance for each column and an appropriate VIF value.

# #### Model #2

# In[ ]:


X_train_rfe = X_train_rfe.drop('carwidth',axis=1)
X_train_rfe_lm = sm.add_constant(X_train_rfe)
lm2 = sm.OLS(y_train, X_train_rfe_lm).fit()
print(lm2.summary())
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending= False)
print(vif)


# Clearly the VIF value for `enginesize  ` is too high. We need to delete it

# #### Model #3

# In[ ]:


X_train_rfe = X_train_rfe.drop('enginesize',axis=1)
X_train_rfe_lm = sm.add_constant(X_train_rfe)
lm3 = sm.OLS(y_train, X_train_rfe_lm).fit()
print(lm3.summary())
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending= False)
print(vif)


# Still values are not < 5, We need to delete `FrontEngine`

# #### Model #4

# In[ ]:


X_train_rfe = X_train_rfe.drop('FrontEngine',axis=1)
X_train_rfe_lm = sm.add_constant(X_train_rfe)
lm4 = sm.OLS(y_train, X_train_rfe_lm).fit()
print(lm4.summary())
vif = pd.DataFrame()
X = X_train_rfe
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF', ascending= False)
print(vif)


# ### Finally, we can see the values of VIF is less than 5. We can make the Model #4 as our final model.

# In[ ]:


final_model = lm4


# Yes we are not done yet! Let's see how error terms are distributed

# ## Residual Analysis

# In[ ]:


y_train_price = lm4.predict(X_train_rfe_lm)   #getting the predicted values


# The error terms should be normally distributed. For that we need to plot distplot

# In[ ]:


# Importing the required libraries for plots.
import matplotlib.pyplot as plt
import seaborn as sns
# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((y_train - y_train_price), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label


# We can conclude our model is correct as the Error terms are distributed normally.

# ## Prediction time!!

# #### Scalling is again required for the test sets

# In[ ]:


num_vars = [ 'wheelbase', 'carlength', 'carwidth', 'carheight', 'curbweight','enginesize','boreratio', 'stroke','compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg','price']
df_test[num_vars] = scaler.transform(df_test[num_vars])


# #### Dividing into X_test and y_test

# In[ ]:


y_test = df_test.pop('price')
X_test = df_test


# In[ ]:


# Now let's use our model to make predictions.

# Creating X_test_new dataframe by dropping variables from X_test
X_test_new = X_test[X_train_rfe.columns]

# Adding a constant variable 
X_test_new_lm = sm.add_constant(X_test_new)


# In[ ]:


# Making predictions
y_pred = final_model.predict(X_test_new_lm)


# ## Model Evaluation

# In[ ]:


# Plotting y_test and y_pred to understand the spread.
fig = plt.figure()
plt.scatter(y_test,y_pred)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)                          # Y-label


# In[ ]:


from sklearn.metrics import r2_score, mean_squared_error
print("R-Square for test:")
print(round(r2_score(y_test, y_pred),4))


# In[ ]:


print("Mean Sqaure Error for test:")
print(round(mean_squared_error(y_test, y_pred),4))


# ### We can see the scatter plot is showing the showing a linear pattern and 86% percent for the values are accurately predicted by the model #4 and the mean square error is also very low.

# ## Final expression for the model

# In[ ]:


final_model.params.sort_values()


# ### Y = 0.67084xcurbweight+ 0.339065xPORSCHE+0.320060xBMW+0.264099xBUICK+0.260975xJAGUAR +0.131197xpeakrpm+0.085411xALFA-ROMERO - 0.143823 

# ## Suggestions for the business
# * High price means high profit margin. So concentrate on high price deals.
# * The Curbweight of the car contributes the most in the high price car. So you go for cars with high Curbweight.
# * Among the all brands available in the US concentrate only on PORSCHE, BMW, BUICK, ALFA-ROMERO and JAGUAR car manufacturing. And it is safe to ignore rest of the brands.
# * The high PeakRPM cars changes the price of the cars. So check if you can get high peak RPM cars for the manufacturing.
# * The negative intercept suggests if you don't follow this model at all, (all dependent variables zero) you may get negative price which mean loss to the business.
