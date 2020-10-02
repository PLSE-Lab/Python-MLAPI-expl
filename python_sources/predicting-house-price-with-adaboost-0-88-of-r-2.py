#!/usr/bin/env python
# coding: utf-8

# # Pridicting House Price 

# ## Overview

#  Welcome to my Kernel! In this kernel, I use various regression methods and try to predict the house prices by using them. As you can guess, there are various methods to suceed this and each method has pros and cons. I think **regression is one of the most important methods when used with Adaboost** because it gives us more insight about the data. When we ask why, it is easier to interpret the relation between the response and explanatory variables.

# In[ ]:





# ## Importing tool and libraries

# Importing libraries and data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn


# In[ ]:


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, Normalizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


# In[ ]:


## Forcing pandas to display any number of elements
def set_pandas_options() -> None:
    pd.options.display.max_columns = 1000
    pd.options.display.max_rows = 1000
    pd.options.display.max_colwidth = 199
    pd.options.display.width = None
    pd.options.display.precision = 8
    pd.options.display.float_format = '{:,.3f}'.format
set_pandas_options()


# In[ ]:


# importing data
df = pd.read_csv('/kaggle/input/housesalesprediction/kc_house_data.csv')


# ## First Look of data

# let see how data is look.

# In[ ]:


df.head()


# In[ ]:


df.info()


# The data is pretty clean there is no null value in data-set

# ## EDA

# ### Price

# Lets explore distribution of Target variable

# In[ ]:


# distribtion of price variable
plt.figure(figsize=(15,5))
sns.distplot(df['price'], bins=20, kde=False)


# The distribtion is Unimodel right skewed and centered at around 500000.0 and seem there is outlier in right side of distribtion. So need some kind of transformation to make it normal.

# ### Categorial Variable

# Now we explore relation between target variable(price) and categorial variable.

# In[ ]:


# by looking the values count of these variable we see that these are categorial varable. 
cat_features = ['bedrooms', 'bathrooms', 'floors', 'waterfront', 'view', 'condition', 'grade']


# In[ ]:


for feature in cat_features:
    plt.figure(figsize=(15,8))
    sns.boxplot(y='price', x=feature, data=df)


# I found that **bathrooms**, **waterfront**, and **grade** are very good correlation with price. I also found that two of bedrooms *11* and *33* are very unusual that we might removed these point.I can see that bathrooms is floating number I can make these integer to reduce complexity.There are some outlier that furture away I also need to further look at these data point to confirm these are acutual outlier.

# ### Numerical Feature

# In[ ]:


num_feature = ['sqft_living', 'sqft_lot', 'sqft_above', 'sqft_basement', 'sqft_living15', 'sqft_lot15',
              'yr_built', 'yr_renovated', 'lat', 'long', 'price']


# In[ ]:


sns.pairplot(df[num_feature])


# There is very good correlation of **sqft_living**, **sqft_above**, **sqft_basement**,**sqft_living15**, **lat** these feature and **price**. There are some data point that stand way out from rest of data. I need to further look at these point to declare as outlier.
# I can represent as **yr_renovated** as category variable to reduce complexity.

# ## Preprocessing and Removing Outlier

# In[ ]:


# making target variable first column
def target_to_start(df, target):
    feature = list(df)
    feature.insert(0, feature.pop(feature.index(target)))
    df = df.loc[:, feature]
    return df
    
df = target_to_start(df, 'price')


# In[ ]:


# removing id column because it not relevant.
df.drop('id', axis=1, inplace=True)


# In[ ]:


# coverting columns into integer from float.
df.price = df.price.astype(int)
df.bathrooms = df.bathrooms.astype(int)
df.floors = df.floors.astype(int)


# ### Outlier

# By exploring further i find some potential outlier.

# In[ ]:


#I remove bedrooms above 11 because price is not as high as no.of bedroom e.g with 33 bedrooms price is less than 9 bed rooms
df = df[df['bedrooms']<11]


# In[ ]:


df = df[(df['bathrooms'] !=4) & (df['price'] != 7062500)]


# In[ ]:


df = df[(df['sqft_living']<13000) & (df['price']!=2280000)]


# ## Feature Engineering

# In[ ]:


# i think yr_built is not so useful feature so i convert this into age_of_house.
df['age_of_house'] = df['date'].apply(lambda x: int(x[:4])) - df['yr_built']
df.drop('yr_built', axis=1, inplace=True)


# In[ ]:


# droping data columns because we do not need anymore.
df.drop('date', axis=1, inplace=True)


# In[ ]:


# convert yr_renovated into categorical variable.
df['renovated'] = df['yr_renovated'].apply(lambda x: 0 if x == 0 else 1)
df.drop('yr_renovated', axis=1, inplace=True)


# In[ ]:


# Performing log transformation of numrical variable to get normal distribation.
df['price_log'] = np.log(df['price'])
df['sqft_living_log'] = np.log(df['sqft_living'])
df['sqft_lot_log'] = np.log(df['sqft_lot'])
df['sqft_above_log'] = np.log(df['sqft_above'])
df['sqft_living15_log'] = np.log(df['sqft_living15'])
df['sqft_lot15_log'] = np.log(df['sqft_lot15'])


# In[ ]:


plt.figure(figsize=(15,5))
sns.distplot(df['price_log'], bins=20, kde=False)


# Similarly all other feature get normalize.

# In[ ]:


df.to_pickle('clean_dataset')


# In[ ]:


df = pd.read_pickle('clean_dataset')


# ## Feature selection

# In[ ]:


# All features
feature1 = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','waterfront', 'view', 'condition',
            'grade', 'sqft_above', 'sqft_basement', 'zipcode', 'lat', 'long', 'sqft_living15','sqft_lot15',
            'age_of_house', 'renovated']

# features that correlation is greater than "0.2"
feature2 = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'view',
            'grade', 'sqft_above', 'sqft_basement', 'lat', 'sqft_living15']

# numerical features with log_transform
feature3 = ['bedrooms', 'bathrooms', 'sqft_living_log', 'sqft_lot_log', 'floors','waterfront', 'view', 'condition',
            'grade', 'sqft_above', 'sqft_basement', 'zipcode', 'lat', 'long', 'sqft_living15_log','sqft_lot15_log',
            'age_of_house', 'renovated']

# numerical features with log_transform where correlation is greater that "0.2"
feature4 = ['bedrooms', 'bathrooms', 'sqft_living_log', 'floors', 'view',
            'grade', 'sqft_above', 'sqft_basement', 'lat', 'sqft_living15_log']


# In[ ]:


def correlation_of_each_feature(dataset, features):
    # get correlations of each features in dataset
    features.append('price_log')
    corrmat = dataset[features].corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    #plot heat map
    sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")

correlation_of_each_feature(df, feature1.copy())


# ## MAchine Learning

# In[ ]:


# Evaluation Matrix 
evaluation_df = pd.DataFrame(columns=['Name of Model','Feature Set', 'Target', 'R^2 of Training', 'R^2 of Testing', 'Mean Squaued Error Training',
                                      'Mean Squaued Error Testing'])


# In[ ]:


# function to split data into training and testing set
def feature_target(features, target):
    X = df[features]
    y = df[target]
    feature_train, feature_test, label_train, label_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return feature_train, feature_test, label_train, label_test


# In[ ]:


def model_xyz(model, feature_train, feature_test, label_train, label_test, model_name='Linear Regression', feature_set=1,
              target='price'):
    model.fit(feature_train, label_train)  
    y_pred_train = model.predict(feature_train)
    y_pred_test = model.predict(feature_test)
    r2_train = r2_score(label_train, y_pred_train)
    r2_test = r2_score(label_test, y_pred_test)
    rmse_train = mean_squared_error(label_train, y_pred_train)
    rmse_test = mean_squared_error(label_test, y_pred_test)
    
    r = evaluation_df.shape[0]
    evaluation_df.loc[r] = [model_name, feature_set, target, r2_train, r2_test, rmse_train, rmse_test]


# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature1, 'price')
lr = LinearRegression()
model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Simple linear Regression', feature_set= 1,
          target='price')


# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature2, 'price')
lr = LinearRegression()
model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Simple linear Regression', feature_set= 2,
          target='price')


# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')
lr = LinearRegression()
model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Simple linear Regression', feature_set= 3,
          target='price_log')


# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature4, 'price_log')
lr = LinearRegression()
model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Simple linear Regression', feature_set= 4,
          target='price_log')


# In[ ]:


evaluation_df


# I seen that model is not overfitting so no need of Regulariztion. Upto now transformed feature are perform well then other.

# Now I try polynomial features.

# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature1, 'price')
polyfeat = PolynomialFeatures(degree = 2)
feature_train = polyfeat.fit_transform(feature_train)
feature_test = polyfeat.fit_transform(feature_test)
lr = LinearRegression()
model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Linear Regression with degree 2',
          feature_set= 1, target='price')


# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature2, 'price')
polyfeat = PolynomialFeatures(degree = 2)
feature_train = polyfeat.fit_transform(feature_train)
feature_test = polyfeat.fit_transform(feature_test)
lr = LinearRegression()
model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Linear Regression with degree 2', 
          feature_set= 2, target='price')


# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')
polyfeat = PolynomialFeatures(degree = 2)
feature_train = polyfeat.fit_transform(feature_train)
feature_test = polyfeat.fit_transform(feature_test)
lr = LinearRegression()
model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Linear Regression with degree 2', 
          feature_set= 3, target='price_log')


# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature4, 'price_log')
polyfeat = PolynomialFeatures(degree = 2)
feature_train = polyfeat.fit_transform(feature_train)
feature_test = polyfeat.fit_transform(feature_test)
lr = LinearRegression()
model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Linear Regression with degree 2', 
          feature_set= 4, target='price_log')


# In[ ]:


evaluation_df


# #### **Woo thats great!!!!!!** 

# Linear regression with polynomial degree 2 with feature set **3** is given great result training **r^2 = 0.83** and testing **r^2 = 0.82**. That good but i try to improve it. Lets go with further degree of polynomial degree. 

# From now onward I only go with transformed features because its give higher r^2 then other features. 

# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')
polyfeat = PolynomialFeatures(degree = 3)
feature_train = polyfeat.fit_transform(feature_train)
feature_test = polyfeat.fit_transform(feature_test)
lr = LinearRegression()
model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Linear Regression with degree 3',
          feature_set= 3, target='price_log')


# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature4, 'price_log')
polyfeat = PolynomialFeatures(degree = 3)
feature_train = polyfeat.fit_transform(feature_train)
feature_test = polyfeat.fit_transform(feature_test)
lr = LinearRegression()
model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Linear Regression with degree 3',
          feature_set= 4, target='price_log')


# In[ ]:


evaluation_df.iloc[6:,]


# With polynomial degree *3* Training R^2 is **0.88** higher then testing R^2 **0.79**.Infact it is less then ploynomial degree *2* testing r^2 **0.81**.

# So it may be overfitting so we need to apply **Regularization**.

# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')
polyfeat = PolynomialFeatures(degree = 3)
feature_train = polyfeat.fit_transform(feature_train)
feature_test = polyfeat.fit_transform(feature_test)
lr = Lasso(alpha=10)
model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Lasso Regression with degree 3',
          feature_set= 3, target='price_log')


# In[ ]:


evaluation_df.iloc[8:,]


# Now with gridsearchcv I try to find best value for alpha.

# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')
polyfeat = PolynomialFeatures(degree = 3)
feature_train = polyfeat.fit_transform(feature_train)
feature_test = polyfeat.fit_transform(feature_test)
lr = Lasso()
search_grid={'alpha':[0.001,0.01,0.05,1,10,20]}
search=GridSearchCV(estimator=lr, param_grid=search_grid, 
                    scoring='neg_mean_squared_error', n_jobs=1, cv=5)


# In[ ]:


search.fit(feature_train, label_train)


# In[ ]:


search.best_params_


# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')
polyfeat = PolynomialFeatures(degree = 3)
feature_train = polyfeat.fit_transform(feature_train)
feature_test = polyfeat.fit_transform(feature_test)
lr = Lasso(alpha=20)
model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Lasso Regression with degree 3 with alpha=20',
          feature_set= 3, target='price_log')


# In[ ]:


evaluation_df


# That Mean squared error of polynomial degree 3 **(0.056)** is greater then Mean squared error of polynomial degree 2 **(0.45)**. that not good way to go further. 

# ### Decision Tree with AdaBoost

# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')
lr =  DecisionTreeRegressor()
search_grid={'max_depth':[6,7,8,9,10,11,12,13,14,15]}
search=GridSearchCV(estimator=lr, param_grid=search_grid, 
                    scoring='neg_mean_squared_error', n_jobs=1, cv=3)


# In[ ]:


search.fit(feature_train, label_train)


# In[ ]:


search.best_params_


# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')
lr =  DecisionTreeRegressor(max_depth=9)
model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Decision Tree Regressor with alpha 9',
          feature_set= 3, target='price_log')


# In[ ]:


evaluation_df


# Its **MSE(0.051)** is little higher then MSE of linear regressinon with polynomial degree **2(0.47)**

# Now i apply Adaboost let see what happen!...

# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')
ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth=9))

search_grid={'n_estimators':[200,300,400,500],'learning_rate':[0.05, 0.1, 0.3, 1]}

search=GridSearchCV(estimator=ada, param_grid=search_grid, 
                    scoring='neg_mean_squared_error', n_jobs=1, cv=3)


# In[ ]:


search.fit(feature_train, label_train)


# In[ ]:


search.best_params_


# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')
lr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=9), learning_rate=1, n_estimators=500)
model_xyz(lr, feature_train, feature_test, label_train, label_test,model_name= 'Decision Tree Regressor with alpha 9',
          feature_set= 3, target='price_log')


# In[ ]:


evaluation_df


# **Boom** MSE get down to **0.032** from **0.047**.

# In[ ]:


np.exp(0.032)


# ## Validatation

# Now we validate our model by cross-validation.

# In[ ]:


X = df[feature3]
y = df['price_log']
lr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=9), learning_rate=1, n_estimators=500)
scores = cross_val_score(lr, X, y, scoring='neg_mean_squared_error', cv=5)


# In[ ]:


scores.mean()


# Cool MSE is **0.033**  so my final model is Adaboost with decisionTreeRegressor.

# In[ ]:


feature_train, feature_test, label_train, label_test = feature_target(feature3, 'price_log')
lr = AdaBoostRegressor(DecisionTreeRegressor(max_depth=9), learning_rate=1, n_estimators=500)
model = lr.fit(feature_train, label_train)


# In[ ]:


pred = model.predict(feature_test)


# In[ ]:


validation_df  = pd.DataFrame()


# In[ ]:


validation_df['actual'] = np.exp(label_test)


# In[ ]:


validation_df['predetion'] = np.exp(pred)


# In[ ]:


validation_df


# ## Conclusion

# When we look at the evaluation table, **2nd degree polynomial (all features, with price_log as target variable)** is doing good job forpredicting outcome. But **Adaboost with DecisionTree** is doing best.
# Futher improvements can also be made by using feature engineering.
# If you like my notebook please do not forget to **Upvote**.
