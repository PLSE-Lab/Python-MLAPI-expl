#!/usr/bin/env python
# coding: utf-8

# #Multiple Linear Regression using RFE, Statsmodels and Mean Target Encoding
# ## Problem Statement
# A Chinese automobile company Geely Auto aspires to enter the US market by setting up their manufacturing unit there and producing cars locally to give competition to their US and European counterparts. 
# <br>
# They have contracted an automobile consulting company to understand the factors on which the pricing of cars depends. Specifically, they want to understand the factors affecting the pricing of cars in the American market, since those may be very different from the Chinese market. The company wants to know:
# 
# - Which variables are significant in predicting the price of a car
# - How well those variables describe the price of a car
# <br>
# Based on various market surveys, the consulting firm has gathered a large dataset of different types of cars across the Americal market.

# In[ ]:


# Import necessary modules for data analysis and data visualization. 
import pandas as pd
import numpy as np

# Some visualization libraries
from matplotlib import pyplot as plt
import seaborn as sns

## Some other snippit of codes to get the setting right 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings ## importing warnings library. 
warnings.filterwarnings('ignore') ## Ignore warning


# ## Understanding the data

# In[ ]:


#Loading the data
car = pd.read_csv("../input/CarPrice_Assignment.csv")
car.head()


# In[ ]:


#checking basic details
car.shape


# In[ ]:


car.info()


#  **As we can see, this dataset does not contain any missing value**

# ## Data Cleaning

# In[ ]:


#checking for duplicated rows
car[car.duplicated()]


# As we can see we dont have any duplicate record.

# In[ ]:


#removing car_ID column as it is insignificant 
car.drop('car_ID', axis=1, inplace=True)


# In[ ]:


#Extracting car company name from CarName column
car['CarName'] = car['CarName'].apply(lambda x: x.split()[0])


# In[ ]:


#checking the slpit and data quality issues
car['CarName'].value_counts().index.sort_values()


# - Here we can see various data quality issues 
#   - mazda is written as `maxda`
#   - nissan is written as `Nissan`
#   - porsche is written as `porcshce`
#   - toyota is written as `toyouta`
#   - for volkswagen we have `vokswagen` and `vw`
# 
# **We will correct the names one by one**

# In[ ]:


#replacing car names with correct ones
car['CarName'] = car['CarName'].replace('maxda','mazda')
car['CarName'] = car['CarName'].replace('Nissan','nissan')
car['CarName'] = car['CarName'].replace('porcshce','porsche')
car['CarName'] = car['CarName'].replace('toyouta','toyota')
car['CarName'] = car['CarName'].replace('vokswagen','volkswagen')
car['CarName'] = car['CarName'].replace('vw','volkswagen')


# In[ ]:


#checking the car names again
car['CarName'].value_counts()


# Now we can see we have unique car names

# In[ ]:


#renaming Carname column to companyname
car = car.rename(columns={'CarName':'CompanyName'})


# In[ ]:


#checking distribution of price column
sns.distplot(car['price'], bins=50)
plt.show()


# ## Derived Metrics

# symboling : Its assigned insurance risk rating, A value of +3 indicates that the auto is risky, -3 that it is probably pretty safe.(Categorical) 		
# 

# In[ ]:


#creating symbol function and applying it of symboling column
def symbol(x):
    if x >= -3 & x <= -1:
        return 'No Risk'
    elif x>=0 and x <= 1:
        return 'Low Risk'
    else:
        return 'High Risk'
car['symboling'] = car['symboling'].apply(symbol)


# In[ ]:


car.symboling.value_counts()


# In[ ]:


#creating fuel economy metric
car['fueleconomy'] = (0.55 * car['citympg']) + (0.45 * car['highwaympg'])


# In[ ]:


#removing citympg and highwaympg cols as their effect is considered in fueleconomy
car.drop(['citympg','highwaympg'],axis=1, inplace=True)


# In[ ]:


car.head()


# In[ ]:


#recognising categorical and numerical features
cat_features = car.dtypes[car.dtypes == 'object'].index
print('No of categorical fetures:',len(cat_features),'\n')
print(cat_features)
print('*'*100)

num_features = car.dtypes[car.dtypes != 'object'].index
print('No of numerical fetures:',len(num_features),'\n')
print(num_features)


# In[ ]:


car[cat_features].head()


# In[ ]:


car[num_features].head()


# In[ ]:


#checking stats of numerical features
car[num_features].describe()


# ## EDA

# ### EDA on numerical variables

# In[ ]:


#checking correlation of all numerical variables
plt.figure(figsize=(10,8),dpi=100)
sns.heatmap(car[num_features].corr(), annot=True)
plt.show()


# As we can see from the above heatmap, **wheelbase, carlength, carwidth, curbweight, enginesize & horsepower** are highly positivily correlated with `price`

# **fuel economy** is highly negatively correlated

# In[ ]:


#checking the distribution of highly correlated numerical features with price variable
cols = ['wheelbase','carlength', 'carwidth', 'curbweight', 'enginesize','horsepower']
plt.figure(figsize=(20,4), dpi=100)
i = 1
for col in cols:
    plt.subplot(1,6,i)
    #sns.distplot(car['price'])
    sns.distplot(car[col])
    i = i+1
plt.tight_layout()
plt.show()


# All these features follows nearly normal distribution

# In[ ]:


num_features


# In[ ]:


#visualising all the numerical features against price column

nr_rows = 5
nr_cols = 3
from scipy import stats
fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*3.5,nr_rows*3),dpi=200)

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(num_features):
            sns.regplot(car[num_features[i]], car['price'], ax = axs[r][c])
            stp = stats.pearsonr(car[num_features[i]], car['price'])
            str_title = "r = " + "{0:.3f}".format(stp[0]) + "      " "p = " + "{0:.3f}".format(stp[1])
            axs[r][c].set_title(str_title,fontsize=11)
            
plt.tight_layout()    
plt.show()   


# From the above graph, we can make the following inferences:
# - **stroke, compressionratio and peakrpm** have very low correlation with price variable
# - All these variables have high p value (> 0.05), we can say that they will be insignificant in our model building process, so we will remove them

# In[ ]:


#removing these columns
car.drop(['carheight','stroke','compressionratio','peakrpm'],axis=1, inplace=True)


# ### EDA on Categorical Variables

# In[ ]:


#checking unique value counts of categorical features
car[cat_features].nunique().sort_values()


# In[ ]:


#eda on categorical columns
cols = ['fueltype','aspiration', 'doornumber','enginelocation','drivewheel']
i = 1
plt.figure(figsize=(17,8),dpi=100)
for col in cols:
    plt.subplot(1,len(cols),i)
    car[col].value_counts().plot.pie(autopct='%1.0f%%', startangle=90, shadow = True,colors = sns.color_palette('Paired'))
    i = i+1
plt.tight_layout()
plt.show()


# We can easily draw following points :
#  - Most number of cars runs on gasoline in America
#  - Standard aspiration is preferred over turbo aspiration
#  - There is only 1% cars which are having engine at rear position
#  - People prefer front wheel derive cars

# In[ ]:


#dropping engine location as it is highly imbalanced
car.drop('enginelocation',axis=1,inplace=True)


# In[ ]:


#making countplot for all below categorical variables
cols = ['symboling','carbody', 'enginetype', 'fuelsystem']
nr_rows = 2
nr_cols = 2
fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*6.5,nr_rows*3),dpi=100)

for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(cols):
            sns.countplot(car[cols[i]], ax = axs[r][c])
            
plt.tight_layout()    
plt.show()   


# Some useful points:
#  - Cars with Low risk rating are more in number
#  - people prefer Sedan and Hatchback cars
#  - 4 cylinder OHC(Over head cam) type engine is more poplular in America

# In[ ]:


#visualising carname feature
plt.figure(figsize=(10,4),dpi=100)
sns.countplot(car['CompanyName'])
plt.xticks(rotation=90)
plt.show()


# Lets visaualize effect of categorical variables on price

# In[ ]:


li_cat_feats = list(car.dtypes[car.dtypes=='object'].index)
nr_rows = 5
nr_cols = 2
fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(nr_cols*6,nr_rows*3),dpi=200)
for r in range(0,nr_rows):
    for c in range(0,nr_cols):  
        i = r*nr_cols+c
        if i < len(li_cat_feats):
            sns.boxplot(x=li_cat_feats[i], y='price', data=car, ax = axs[r][c])
plt.tight_layout()    
plt.show() 


# We can draw following points from the above graphs:
#  - Price of car does not depend much on Doornumber of car
#  - Price of car does depend on number brand name in the car <br>
# Except doornumber , price of car depends on every other feature

# In[ ]:


#removing doornumber from the dataset
car.drop('doornumber',axis=1, inplace=True)


# ## Data Preprocessing

# ### Encoding of categorical variables

# We will use one-hot code encoding for this assignment.

# In[ ]:


cat_features = car.dtypes[car.dtypes == 'object'].index


# In[ ]:


car[cat_features].nunique().sort_values()


# In[ ]:


#creating function for targe encoding
#credits : https://maxhalford.github.io/blog/target-encoding-done-the-right-way/
def calc_smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()

    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']

    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)

    # Replace each value by the according smoothed mean
    return df[by].map(smooth)


# In[ ]:


#performing target encoding with weight of 100
for col in cat_features:
    car[col] = calc_smooth_mean(car,by=col, on='price', m=100)


# ### Train-Test split

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


df_train, df_test = train_test_split(car, test_size=0.3, random_state=42)
print(df_train.shape)
print(df_test.shape)


# ### Feature scaling

# Here we will scale full dataframe using MinMax scaler

# In[ ]:


cols = df_train.columns


# In[ ]:


#importing minmax scaler from sklearn.preprocessing and scaling the training dataframe
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_train[cols] = scaler.fit_transform(df_train[cols])


# In[ ]:


#transforming the test data set
df_test[cols] = scaler.transform(df_test[cols])


# In[ ]:


#checking minmax scaling
df_train.describe()


# In[ ]:


#checking correlation of train dataframe 
plt.figure(figsize=(15,15),dpi=100)
sns.heatmap(df_train.corr(), cmap='RdYlBu')
plt.show()


# In[ ]:


#creating function for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
def vif(X_train):
    vif = pd.DataFrame()
    vif['Features'] = X_train.columns
    vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    return vif


# ## Model building

# In[ ]:


#creating X and y variables
y_train = df_train.pop('price')
X_train = df_train


# In[ ]:


print(X_train.shape)


# In[ ]:


#feature selection using RFE
#In this case we are have 57 features , lets select 20 features from the data using RFE and then we will 
# remove statistical insignificant variables one by one
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train,y_train)

rfe = RFE(lr,10)
rfe.fit(X_train,y_train)

print(list(zip(X_train.columns,rfe.support_,rfe.ranking_)))
print('*'*100)
cols_rfe = X_train.columns[rfe.support_]
print('Features with RFE support:')
print(cols_rfe)
print('*'*100)
print('Features without RFE support:')
cols_not_rfe = X_train.columns[~rfe.support_]
print(cols_not_rfe)


# In[ ]:


#taking cols with RFE support
X_train = X_train[cols_rfe]


# In[ ]:


#checking VIF
vif(X_train).head(10)


# In[ ]:


#removing carlength as it is having VIF
X_train.drop('carlength', axis=1, inplace=True)
vif(X_train).head()


# In[ ]:


#removing curbweight as it is having high VIF
X_train.drop('curbweight', axis=1, inplace=True)
vif(X_train).head()


# In[ ]:


#removing carwidth as it is having high VIF
X_train.drop('carwidth', axis=1, inplace=True)
vif(X_train).head()


# In[ ]:


#removing enginesize as it is having high VIF
X_train.drop('enginesize', axis=1, inplace=True)
vif(X_train).head()


# Lets start building model and we will remove other statistically insignificant variables based on p-value and vif

# In[ ]:


#importing statsmodel
import statsmodels.api as sm


# In[ ]:


#Building the first model
X_train_lr = sm.add_constant(X_train)
lr_1 = sm.OLS(y_train,X_train_lr).fit()
print(lr_1.summary())
print(vif(X_train))


# In[ ]:


#removing enginetype as it is having p-value  and building 2nd model
X_train.drop('enginetype', axis=1, inplace=True)
X_train_lr = sm.add_constant(X_train)
lr_2 = sm.OLS(y_train,X_train_lr).fit()
print(lr_2.summary())
print(vif(X_train))


# In[ ]:


#removing wheelbase as it is having high VIF building 3rd model
X_train.drop('wheelbase', axis=1, inplace=True)
X_train_lr = sm.add_constant(X_train)
lr_3 = sm.OLS(y_train,X_train_lr).fit()
print(lr_3.summary())
print(vif(X_train))


# Now we can see our model have all the variables having VIF less than 5 and all the p-values less than 0.05, so we can say that our model is good.

# ## Residual Analysis

# In[ ]:


#calculating residuals
y_train_pred = lr_3.predict(X_train_lr)
residuals = y_train-y_train_pred


# In[ ]:


#plotting residuals
plt.figure(dpi=100)
sns.distplot(residuals)
plt.xlabel('Residuals')
plt.show()


# In[ ]:


#checking mean of residuals
np.mean(residuals)


# As we can see , residuals are normally distributed and have a mean of zero

# In[ ]:


#scatterplot of resuduals v/s fitted values
plt.figure(figsize=(16,5),dpi=100)
plt.subplot(121)
plt.scatter(y_train_pred,residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')

plt.subplot(122)
plt.scatter(y_train,residuals)
plt.xlabel('Training Values')
plt.ylabel('Residuals')
plt.show()


# We are not having any pattern of residuals with either fitted value or training values 

# In[ ]:


plt.figure(dpi=100)
sns.regplot(y_train_pred,residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')


# As we can see from the above graphs residuals does not have any pattern with fitted and training values, so we can say that our model is good.

# ## Making predictions

# In[ ]:


#checking the test data
df_test.describe()


# In[ ]:


#creating X and y for test dataframe
y_test = df_test.pop('price')
X_test = df_test
X_test.head()


# In[ ]:


X_train.columns


# In[ ]:


#predicting test values
X_test = X_test[X_train.columns]
X_test = sm.add_constant(X_test)
y_test_pred = lr_3.predict(X_test)


# In[ ]:


#scatterplot of y_test and y_test_pred
plt.scatter(y_test_pred,y_test)


# We can see that we have almost linear relationship, so we can say that our model is good.

# ## Model Evaluation

# In[ ]:


#importing necessary libraries and methods
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#calculating r2_score 
r2_score(y_test,y_test_pred)


# As we can see r2-score of test set is `0.865` against r2-score of `0.884` and adjusted-r2 score of `0.881` of training data set

# In[ ]:


#calculating mean squared error for test set
mean_squared_error(y_test,y_test_pred)


# In[ ]:


#calculating mean squared error for traning set
mean_squared_error(y_train,y_train_pred)


# Thank You for visiting the kernel : )

# In[ ]:




