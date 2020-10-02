#!/usr/bin/env python
# coding: utf-8

# 
# This dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass. It was obtained from the StatLib archive (http://lib.stat.cmu.edu/datasets/boston) The dataset is small in size with only 506 cases and 14 features. The details about this dataset available on -https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html The purpose of the database is to leverage the available data to predict the prices of houses in Boston using machine learning algorithms.
# This note book contains
# 1. Data Exploration 
# 2. Data visualization
# 3. Data Preprocessing
# 4. Hyper Tuning of the parameters
# 5. Model Builing using Various techniques
# 6 .Result Prediction
# 
# 

# In[ ]:


## Importing the basic libraries
import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[ ]:


## Importind the data into the system for futher analysis
print(os.listdir("../input"))
col = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing=pd.read_csv('../input/housing.csv',delim_whitespace=True, names=col)


# In[ ]:


housing.head()


# In[ ]:


## Check the spread of the data and identify potential outliers at glance
housing.describe()


# In[ ]:


# Verify the type and categories of data and is there any missing values present that we need to take care
housing.info()


# **Analysis From above step:**
# > This reflect  that data is quite clean and most of the variables are float and few of them are integers, no imputations or filling up values required.

# # 2 Exploratory Data analysis

# In[ ]:


housing.hist(bins=20,figsize=(15,15))


# It seems that many variables like Age,B,MEDV,RAD ..etc are caped, It seems this sample size might not be representative of sample space, 

# ##Let's explore each of the variables and see how are they affecting each i'e check correlation for each of the varibles by plotting box plot and analysing how does the shape of variables are and the test way to visualize is to look at the box plot for each of the variables

# In[ ]:


fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
index = 0
axs = axs.flatten()
for k,v in housing.items():
    sns.boxplot(y=k, data=housing, ax=axs[index])
    index += 1


# **Observations:**
# **CRIM Variable**
# 
# It has very short data bandwidth and correlation is not that segnificant being +0.39 which makes it an important parameter for prediction, but it have quite heavy outliers that needs to be handled carefully.

# In[ ]:


CRIM_data=pd.DataFrame(housing.iloc[:,0])
print(CRIM_data.describe())
CRIM_data.hist()
plt.title("Original CRIM Data")
min_cutoff=3.613524-3*8.601545
print("Minimum Cut off Value",min_cutoff)
max_cutoff=3.613524+3*8.601545
print("Maximum Cut Off Value",max_cutoff)
# Values beyond 29.4 must be dropped becasue they are potential outliers so dropping these 4 values
housing_df = housing.drop(housing[housing.CRIM>29.4].index)
print("Final shape of the dataset",housing_df.shape)


# In[ ]:


CRIM_data_new=pd.DataFrame(housing_df.iloc[:,0])
CRIM_data_new.hist()
plt.xlabel("Value of Criminal Rates")
plt.title("New CRIM Data")


# **Conclusion for CRIM Data variable:** Inspite of having a very small size data, it very important to remove the potential outlier from the data set, Incase it is not effecting to much target varibles, if it is further analysis is required, I will remove the excess data +/- 3 standard Deviation from the mean
#     

# **2. ZN Variable**** Observation

# In[ ]:


ZN_data=pd.DataFrame(housing_df.iloc[:,1])
print(ZN_data.describe())
ZN_data.hist()


# In[ ]:


min_cutoff=11.546185-3*23.464449
print("Minimum Cut off Value",min_cutoff)
max_cutoff=11.546185+3*23.464449
print("Maximum Cut Off Value",max_cutoff)
housing_df_N = housing_df.drop(housing_df[housing.ZN>=82].index)
CRIM_data_new=pd.DataFrame(housing_df_N.iloc[:,1])
CRIM_data_new.hist()


# 3. **CHAS Variable**
# This is a categorical variablle and has only 2 distinct values, and this column is already dummified so no need to take any action upon it

# 4. **INDUS Variable**
#      Since dataset is extremely small so thinking not to remove the outliers, but when dataset is large it is advisible to remove few outliers from the data to make the mode fit

# In[ ]:


housing_df_N['INDUS'].hist()
print(housing_df_N['INDUS'].describe())


# 

# 5. ** MDEV**
# Target variable

# In[ ]:


housing_df_N['MEDV'].hist()


# In[ ]:


housing_df_N['MEDV'].describe()


# In[ ]:


lower_limit=22.370248-3*8.802114
upper_limit=22.370248+3*8.802114
print(upper_limit,lower_limit) ## No action been been taken due to small data size
housing_df_M = housing_df_N.drop(housing_df_N[housing_df_N.MEDV>48.77659].index)


# **B variable**

# In[ ]:


housing_df_M['B'].hist()


# In[ ]:


housing_df_M['B'].describe()


# In[ ]:


upper_limit=358.160764+3*88.205368
lower_limit=358.160764-3*88.205368
print("Upper cut off value",upper_limit)
print("Lower Cut off Value",lower_limit) ## I will come back later on this based on model under fitting and over fitting


# In[ ]:


housing_df_O = housing_df_M.drop(housing_df_M[housing_df_M.B<92].index)


# In[ ]:


housing_df_O['B'].hist()


# In[ ]:


housing_df_O['NOX'].hist()


# In[ ]:


housing_df_O['NOX'].describe()


# In[ ]:


housing_df_O['RM'].describe()


# In[ ]:


housing_df_O['AGE'].hist()


# In[ ]:


housing_df_O['AGE'].unique()
housing_df_P = housing_df_O.drop(housing_df_O[housing_df_O.MEDV>48.77659].index)


# In[ ]:


housing_df_P.shape


# In[ ]:


housing_df_O['DIS'].describe()


# In[ ]:


housing_df_P['LSTAT'].describe()


# In[ ]:


lower_limit=12.522477-3*6.699438
upper_limit=12.522477+3*6.699438
print("Lower Limit",lower_limit)
print("Upper Limit",upper_limit)


# In[ ]:


housing_df_Q = housing_df_P.drop(housing_df_P[housing_df_P.MEDV>32.620791].index) ## rm the value>32.620791


# In[ ]:


housing_df_Q.shape


# In[ ]:


plt.figure(figsize = (15,12))
sns.heatmap(data=housing_df_Q.corr(), annot=True,linewidths=.8,cmap='Blues')


# From heat map we have observed that TAX ,RAD ,CRIM are highly correlated and CHAS , PTRATIO and B are quite less correlated
# wrt .to target variable RM positively and LSAT is negatively correlated and hence their impact
# ### In case prediction is not up to mark need to remove one of these highly correlated variable

# In[ ]:


sns.pairplot(housing_df_Q,x_vars=["CRIM","ZN","INDUS"],y_vars =["MEDV"], kind="reg",height=6)


# 1. ["CRIM","ZN","INDUS"] features are reflect  weak correlation with MDEV variable
# 2. CRIM has negative correlation with prices. The areas with lower rate of crime has high prices and vice versa.and there are few areas are with high crime rate else most of the areas has lower crime rate.
# 3. ZN - proportiion residential land zoned --this feature posses positive correlation with prices. More the residelntial land zone more higher the housing prices.And also it shows that threr are a large group of area has low prportion of residential land zone.
#  4. INDUS- proportion of non retail bussiness acre per town - it shows negative correlation. As the proportion decreases the housing prices lowers.

# In[ ]:


sns.pairplot(housing_df_Q,x_vars=["NOX","RM","AGE"],y_vars =["MEDV"], kind="reg",height=6)


#  1. NOX- Nitric oxide concentration - it shows negatve correaltion .We can see the areas with high cocentration has lower housing prices. lower the pollution of air , higher the housing prices.
#  2. RM- Avg number of room per house. -  it is obivious that as number of rooms increases the area of house increase and prices will be more. the sme trend we can see in the plot. Also it has strong correlation with prices as compared to other parameters we saw upto now.
#  3. AGE- proportion of owner-occupied units built prior to 1940 - this feature shows negative correlation . the older the property lower the housing prices . we can seee the dataset has slighlty more number of old houses.

# In[ ]:


sns.pairplot(housing_df_Q,x_vars=["DIS","RAD","TAX"],y_vars =["MEDV"], kind="reg",height=6)


#  1. DIS-    weighted distances to five Boston employment centres - This feature shows Positive correlation with housing prices. the areas near to the employment centres/ work places has high prices which is obivious trend we do see in housing prices.
#  2. RAD-index of accessibility to radial highways- there is negative correlation with prices.Alos we can see that a lot of areas has low index and few areas has high index for highway accessibility. Mthe plot shows people do not prefer houses near the highways.
#  3.TAX- full-value property-tax rate per USD 10,000 -Tax Rate shows negative correlation with housing prices. Higher the property tax lower the prices in that area. the housing prics are high where the propery taxes are low. people prefer areas with lower property tax.

# In[ ]:


sns.pairplot(housing_df_Q,x_vars=["PTRATIO","B","LSTAT"],y_vars =["MEDV"], kind="reg",height=6)


# 1. PTRATIO- pupil-teacher ratio by town - this feature has negative correltion with housing prices. 
# 2. B - proportion of blacks by town - It shows positive correlaton with prices.
# 3.LSTAT-LSTAT - % lower status of the population - This shows strong postive correation . it does not show direct correltion. the plot shows somewhat curvilinear nature.

# Till now we checked the individual relations with feature variable. And almost all variable shows linear relationship with target variable. Not every feature has strong correlation. Eventually we will find out which are not the good features for prediction. As the target variable has continous values we wil go for regression techniques. First lets try with Multiple linear regression.
# 
# The preassumption for Linear regression is that the features used for moel should not be correlated . From scatter plots above we checked linear relationship of individual feature with target variable.
# Lets check the correlation of all features with each other using corrrelaton matrix

# In[ ]:


plt.figure(figsize = (9,9))
sns.heatmap(data=housing_df_Q.corr(), annot=True,linewidths=.8,cmap='Blues')


# ** Considering the continous value of the housing price targer variable this above problem is considered to be a regression task**

# **Regression Model Using Linear Regression**

# In[ ]:


housing_df_Q.head()


# In[ ]:


import numpy as np
def split_train_test(data,test_ratio):
    shuffled_indicies=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indicies=shuffled_indicies[:test_set_size]
    train_indicies=shuffled_indicies[test_set_size:]
    return data.iloc[test_indicies],data.iloc[train_indicies]
test_set,train_set=split_train_test(housing_df_Q,0.2)


# In[ ]:


def training_and_testing_set(test_set,train_set):
    test_set_x=test_set.iloc[:,:-1]
    test_set_y=test_set.iloc[:,-1]
    train_set_x=train_set.iloc[:,:-1]
    train_set_y=train_set.iloc[:,-1]
    return test_set_x,test_set_y,train_set_x,train_set_y
test_set_x,test_set_y,train_set_x,train_set_y=training_and_testing_set(test_set,train_set)
print(test_set_x.shape,test_set_y.shape,train_set_x.shape,train_set_y.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lin_reg=LinearRegression()
lin_reg.fit(train_set_x,train_set_y)
y_pred=lin_reg.predict(test_set_x)
mserr=mean_squared_error(test_set_y,y_pred)
print("Mean Squared Error",mserr)
root_mean_squared_error=np.sqrt(mserr)
print("Root Mean Squared Error",root_mean_squared_error)


# ** Linear Regression Using Backward Elimination **

# In[ ]:


import statsmodels.formula.api as sm


# In[ ]:


X=housing_df_Q.iloc[:,:-1].values
y=housing_df_Q.iloc[:,-1].values


# In[ ]:


X.shape


# In[ ]:


X1=np.append(arr=np.ones((403, 1),int).astype(int),values = X,axis =1)


# In[ ]:


# soring the optimal dataset in X_opt
X_opt = X1[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# In[ ]:


X_opt = X1[:,[0,1,2,5,6,8,9,10,11,13]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# In[ ]:


X_opt = X1[:,[0,1,5,6,8,9,10,11,13]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_opt,y,test_size = 0.3,random_state = 1)
regressor.fit(X_train,y_train)


# In[ ]:


y_pred = regressor.predict(X_test)

print("RMSE: %.2f"% np.sqrt(((y_pred - y_test) ** 2).mean()))


# We can see there is improvment in regressor performance after elimination of features. there is improvment in RMSE values And there is still scope of improvment.
# 
# Lets try out the Random forest regression model on data. because the LSTAT feature shows some curvilinear nature and few shows direct correation. The dataset is combination of Linear and non linear Features.

# ** Linear Regression Using Random Forest**

# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=200,random_state=0)


# In[ ]:


rf.fit(train_set_x,train_set_y)


# In[ ]:


y_pred1=rf.predict(test_set_x)


# In[ ]:


mserr=mean_squared_error(test_set_y,y_pred1)
print("Mean Squared Error",mserr)
root_mean_squared_error=np.sqrt(mserr)
print("Root Mean Squared Error",root_mean_squared_error)


# In[ ]:


from sklearn.metrics import r2_score
r2 = r2_score(test_set_y,y_pred1)
r2


# In[ ]:




