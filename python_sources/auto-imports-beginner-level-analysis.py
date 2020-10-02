#!/usr/bin/env python
# coding: utf-8

# # Title : 1985 Auto Imports Database Analyses 

# <img src='Large10.jpg'>

# ## <font color='green'>Data Dictionary</font>
# ### Input variables
# 
#  01. **symboling**:            [its assigned insurance risk rating -> [-3, -2, -1, 0, 1, 2, 3]] 
#  02. **normalized-losses**:    [average loss payment per insured vehicle year -> continuous from 65 to 256.]  
#  03. make:                     [ Manufacturer name eg : alfa-romero, audi, bmw, chevrolet, dodge, honda,isuzu etc. ]
#  04. fuel-type:                [diesel, gas]
#  05. aspiration:               [std, turbo]
#  06. num-of-doors:             [four, two].
#  07. body-style:               [hardtop, wagon, sedan, hatchback, convertible]
#  08. drive-wheels:             [4wd, fwd, rwd]
#  09. engine-location:          [front, rear]
#  10. wheel-base:               [continuous from 86.6 120.9]
#  11. length:                   [continuous from 141.1 to 208.1]
#  12. width:                    [continuous from 60.3 to 72.3]
#  13. height:                   [continuous from 47.8 to 59.8]
#  14. curb-weight:              [continuous from 1488 to 4066]
#  15. engine-type:              [dohc, dohcv, l, ohc, ohcf, ohcv, rotor]
#  16. num-of-cylinders:         [eight, five, four, six, three, twelve, two]
#  17. engine-size:              [continuous from 61 to 326]
#  18. fuel-system:              [1bbl, 2bbl, 4bbl, idi, mfi, mpfi, spdi, spfi]
#  19. bore:                     [continuous from 2.54 to 3.94]
#  20. stroke:                   [continuous from 2.07 to 4.17]
#  21. compression-ratio:        [continuous from 7 to 23]
#  22. horsepower:               [continuous from 48 to 288]
#  23. peak-rpm:                 [continuous from 4150 to 6600]
#  24. city-mpg:                 [continuous from 13 to 49]
#  25. highway-mpg:              [continuous from 16 to 54]
#  
#  ## Output Variable
#   price:                    [continuous from 5118 to 45400]
# 

# ## Import libraries

# In[ ]:


# Numerical libraries
import numpy as np   

# Import Linear Regression machine learning library
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import Normalizer

# to handle data in form of rows and columns 
import pandas as pd    

# importing ploting libraries
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt   
get_ipython().run_line_magic('matplotlib', 'inline')
#importing seaborn for statistical plots
import seaborn as sns


# ##                                                Load data
# 

# In[ ]:


df = pd.read_csv("../input/Automobile_data.csv",na_values=['?'])


# In[ ]:


df.head()


# ##### This data set consists of three types of entities:
# ##### (a) the specification of an auto in terms of various characteristics 
# ##### (b)its assigned insurance risk rating
# ##### (c) its normalized losses in use as compared to other cars.  

# ##  Exploratory Data Analysis

# ### a. Analyse Data
# 

# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


na_cols = {}
for col in df.columns:
    missed = df.shape[0] - df[col].dropna().shape[0]
    if missed > 0:
        na_cols[col] = missed

na_cols


# In[ ]:


sum(df.isnull().any())
#sum(df.isnull().any())


# In[ ]:


df[np.any(df[df.columns[2:]].isnull(), axis=1)]


# #### This clearly shows the number of rows and columns having missing or NA values. 

# In[ ]:


df[['normalized-losses','bore','stroke','horsepower','peak-rpm']] = df[['normalized-losses','bore','stroke','horsepower','peak-rpm']].astype('float64')


# In[ ]:


df.info()


# In[ ]:


df_1 = df.copy()


# In[ ]:


df_1.head()


# ### b. Refine & Transform
# 

# In[ ]:


# Imputting Missing value
imp = Imputer(missing_values='NaN', strategy='mean' )
df_1[['normalized-losses','bore','stroke','horsepower','peak-rpm','price']] = imp.fit_transform(df_1[['normalized-losses','bore','stroke','horsepower','peak-rpm','price']])
df_1.head()
#########################################################################################################################


# In[ ]:


df_1['num-of-doors'] = df_1['num-of-doors'].fillna('four')


# In[ ]:


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
for i in ['make','fuel-type','aspiration', 'num-of-doors','body-style','drive-wheels','engine-location','engine-type','num-of-cylinders','fuel-system']:
    df_1[i] = labelencoder.fit_transform(df_1[i])
df_1.head()


# ### Analyse Dataset - 
# ##### 4 .How many records are available in the data set and how many attributes. Do you think the depth (number of records) is sufficient given the breadth? In other words, is the sample likely to be a good representative of the universe?

# In[ ]:


df_1.shape


# #### The above dataset has 205 rows and 26 columns which is not a good sample. We can say that it is not a good representative of the universe

# ### d. Visualize data
# ### <font color='red'> 5.Analyse the data distribution for the various attributes and share your observations. <\font>

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from matplotlib import pyplot as plt


# ###     *****Top Selling Car Manufacturer is **Toyota**
# 

# #### Categorical features distributions:

# In[ ]:


categorical = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'engine-location',
               'drive-wheels', 'engine-type', 'num-of-cylinders', 'fuel-system']
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
for col, ax in zip(categorical[1:], axs.ravel()):
    sns.countplot(x=col, data=df, ax=ax)


# #### Max Cars are running on Gas
# #### Max Cars have engine in front
# #### Max Cars have 4 cylinders
# #### Max Cars have mpfi as fuel system

# In[ ]:


df_1.corr()


# In[ ]:


from matplotlib import pyplot as plt
plt.figure(figsize=(15, 15))
ax = sns.heatmap(df_1.corr(), vmax=.8, square=True, fmt='.2f', annot=True, linecolor='white', linewidths=0.01)
plt.title('Cross correlation between numerical')
plt.show()


# In[ ]:


## Above graph shows Wheel base , Length , Width are highly correlated. 
## Highway mpg and city mpg is also highly correlated. 
## Compression ratio and fuel type is also correlated 
## Engine size and horse power is also correlated
df_2 = df_1.drop(['length','width','city-mpg','fuel-type','horsepower'],axis=1)
df_2.head()


# In[ ]:


from matplotlib import pyplot as plt
plt.figure(figsize=(15, 15))
ax = sns.heatmap(df_2.corr(), vmax=.8, square=True, fmt='.2f', annot=True, linecolor='white', linewidths=0.01)
plt.title('Cross correlation between numerical')
plt.show()


# ## Above graphs and HeatMap shows that - 
# ###  Wheel base , Length , Width are highly correlated. 
# ### Highway mpg and city mpg is also highly correlated. 
# ### Compression ratio and fuel type is also correlated 
# ### Engine size and horse power is also correlated

# ## Attributes which has stronger relationship with price - 
# 
# ## 1. Curb-Weight
# ## 2. Engine-Size
# ## 3. Horsepower
# ## 4. Mpg(City / Highway mpg)
# ## 5. Lenght/ Width 

# In[ ]:


sns.lmplot(x= 'curb-weight' , y='price', data=df_2)


# In[ ]:


sns.lmplot(x= 'engine-size' , y='price', hue = 'num-of-doors', data=df_2)


# In[ ]:


sns.lmplot(x= 'horsepower' , y='price',hue = 'fuel-system', data=df)


# In[ ]:


sns.lmplot(x= 'highway-mpg' , y='price', data=df)


# ## Split data into training and test data

# In[ ]:


X = df_2.drop('price',axis =1)
X.head()


# In[ ]:


# Lets use 80% of data for training and 20% for testing

import sklearn
Y = df_2['price']
X = df_2.drop('price',axis =1)

x_train, x_test, y_train,  y_test = sklearn.model_selection.train_test_split(X, Y,train_size=0.8, test_size=0.2, random_state=0)


# ### Linear Regression could be the best algorithm to solve such problem with better accuracy as most of the attributes (Independent Variables) follow Linear pattern with Dependent variable i.e. (Price)

# ## Training of the model

# In[ ]:


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
lm_1 = regressor.fit(x_train, y_train)


# In[ ]:


lm_1.score(x_train,y_train)


# In[ ]:


lm_1.score(x_test,y_test)


# In[ ]:


df_2.shape


# In[ ]:


df_3 = df_2.copy()


# In[ ]:


# Replace '-' in column names with '_'
names = []
for name in df_3.columns:
    names.append(name.replace('-', '_'))

df_3.columns = names


# In[ ]:


df_3.info()


# In[ ]:


import statsmodels.formula.api as smf

lm0 = smf.ols(formula= 'price ~ symboling+normalized_losses+make+aspiration+num_of_doors+body_style+drive_wheels+engine_location+wheel_base+height+curb_weight+engine_type+num_of_cylinders+engine_size+fuel_system+bore+stroke+compression_ratio+peak_rpm' , data =df_3).fit()


# In[ ]:


lm0.params


# In[ ]:


print(lm0.summary())


# ## Model Builduing Part -2 

# In[ ]:


from sklearn.preprocessing import Normalizer
# Normalizing Data
nor = Normalizer()
df_4 = nor.fit_transform(df_2)


# In[ ]:


col = []
for i in df_2.columns:
    col.append(i.replace('-', '_'))  


# In[ ]:


df_4 = pd.DataFrame(df_4 , columns  = col)
df_4.head()


# In[ ]:


# Lets use 80% of data for training and 20% for testing

import sklearn
Y_1 = df_4['price']
X_1 = df_4.drop('price',axis =1)

x_train_1, x_test_1, y_train_1,  y_test_1 = sklearn.model_selection.train_test_split(X_1, Y_1,train_size=0.8, test_size=0.2, random_state=0)


# In[ ]:


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
lm_2 = regressor.fit(x_train_1, y_train_1)


# In[ ]:


pred_train_y = regressor.predict(x_train_1)
pred_test_y = regressor.predict(x_test_1)


# In[ ]:


lm_2.score(x_train_1,y_train_1)


# ## R^2  = 0.98 for Train data

# In[ ]:


lm_2.score(x_test_1,y_test_1)


# ## R^2  = 0.96 for Test data

# In[ ]:


mse = np.mean((pred_test_y -y_test_1)**2)
mse


# In[ ]:


## Residual Vs fitted plot - 
x_plot = plt.scatter(pred_test_y,(pred_test_y - y_test_1),c='b')
plt.hlines(y=0,xmin = 0 , xmax = 1)
plt.title('Residual plot')


# ### There is no pattern so we can infer that data is linear and there is no Heteroskedasticity issue

# ## Linear model using OLS - 

# In[ ]:


import statsmodels.formula.api as smf

lm1 = smf.ols(formula= 'price ~ symboling+normalized_losses+make+aspiration+num_of_doors+body_style+drive_wheels+engine_location+wheel_base+height+curb_weight+engine_type+num_of_cylinders+engine_size+fuel_system+bore+stroke+compression_ratio+peak_rpm' , data =df_4).fit()


# In[ ]:


lm2 = smf.ols(formula= 'price ~ symboling+normalized_losses+make+aspiration+num_of_doors+drive_wheels+engine_location+wheel_base+height+curb_weight+engine_type+num_of_cylinders+engine_size+fuel_system+bore+stroke+compression_ratio+peak_rpm' , data =df_4).fit()


# In[ ]:


lm3 = smf.ols(formula= 'price ~ aspiration+num_of_doors+wheel_base+curb_weight+engine_size+fuel_system+bore+stroke+peak_rpm' , data =df_4).fit()


# In[ ]:


lm3.params


# In[ ]:


print(lm3.summary())


# ## The Above results shows Multi Linear Regression Model  with R^2  = 0.974 
