#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"] = (20,15)


# In[ ]:


# additional imports:
import seaborn as sns
import re
import sys
from time import sleep
from tqdm.notebook import tqdm
import warnings;
warnings.filterwarnings('ignore');


# In[ ]:


df = pd.read_csv("../input/bengaluru-house-price-data/Bengaluru_House_Data.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.head()


# ## Data Cleaning:

# Lets check the null or nan values

# In[ ]:


df.isnull().sum()


# In[ ]:


#Remove unnecessary columns 
df1 = df.drop(['area_type','society','balcony','availability'],axis='columns')
df1.head()


# In[ ]:


df1.shape


# In[ ]:


df1.isnull().sum()


# In[ ]:


df1 = df1.dropna()
df1.isnull().sum()


# In[ ]:


# lets check 'size' column
df1['size'].unique()


# Here we need numeric values, so we will remove bedroom and BHK strings from all values.

# In[ ]:


# FUNCTION to remove string from row values.
# Nan values will be replaced by 0
def remove_string(x):
    x = str(x)
    if x == 'nan':
        x = np.NaN
    else:
        x = int(x.split(" ")[0])
    return x


# In[ ]:


# We create new column for the cleaned values of size column:
df1['BHK'] = df1['size'].apply(lambda x: remove_string(x))


# In[ ]:


df1['BHK'].unique()


# In[ ]:


df1[df1.BHK > 20]


# In[ ]:


df1.isnull().sum()


# In[ ]:


df1.total_sqft.unique()


# There are some values in range format. like 1133-1384

# In[ ]:


# Function to catch all non numeric and abnormal values:
def catch_abnormal_val(series):
    err_val = []
    for x in series:
        try:
            float(x)
        except:
            err_val.append(x)
    return err_val


# In[ ]:


catch_abnormal_val(df1['total_sqft'])


# Not just range values, we can see there are some numeric values written in sqft, perch, acres, yards, cents and ground formats.

# In[ ]:


# Lets modify the range format values first:
# function that will identfy range format values and convert them to single float value:
def convert_rng_val(x):
    values = x.split('-')
    if len(values) == 2:
        return (float(values[0])+float(values[1]))/2 #return float mean value of range
    try:
        return float(x) #return remaining values in float.
    except:
        return x #return other abnormal value as it is.


# In[ ]:


print(convert_rng_val('1200'))
print(convert_rng_val('1200-2349'))
print(convert_rng_val('1200sqft. Meter'))


# * sq Meters to sqft: 10.764 * sq.meters
# * sqYards to sqft: 9 * sqYards
# * gunta to sqft: 1089 * gunta
# * acres to sqft: 43560 * acres
# * perch to sqft: 272.25 * perch
# * Grounds to sqft: 2400 * ground
# * Cents to sqft: 435.6 * cent

# In[ ]:


def sqmt_to_sqft(x):
    """convert sq.meters to sqft"""
    return x * 10.764

def sqyards_to_sqft(x):
    """convert sq.yards to sqft"""
    return x * 9

def gunta_to_sqft(x):
    """convert gunta to sqft"""
    return x * 1089

def acres_to_sqft(x):
    """convert acres to sqft"""
    return x * 43560

def perch_to_sqft(x):
    """convert perch to sqft"""
    return x * 272.25

def grounds_to_sqft(x):
    """convert grounds to sqft"""
    return x * 2400

def cents_to_sqft(x):
    """convert cents to sqft"""
    return x * 435.6


# All functions are ready now will make one parent function to clean our 'total_sqft' column

# In[ ]:


def clean_total_sqft(y):
    try:
        y = float(y)
    except:
        if "-" in y:
            y = round(convert_rng_val(y),1)
        elif "Sq. Meter" in y:
            y = round(sqmt_to_sqft(float(re.findall('\d+',y)[0])),1)
        elif "Sq. Yards" in y:
            y = sqyards_to_sqft(float(re.findall('\d+',y)[0]))
        elif "Guntha" in y:
            y = gunta_to_sqft(float(re.findall('\d+',y)[0]))
        elif "Acres" in y:
            y = acres_to_sqft(float(re.findall('\d+',y)[0]))
        elif "Perch" in y:
            y = perch_to_sqft(float(re.findall('\d+',y)[0]))
        elif "Grounds" in y:
            y = grounds_to_sqft(float(re.findall('\d+',y)[0])) 
        elif "Cents" in y:
            y = round(cents_to_sqft(float(re.findall('\d+',y)[0])),1)
        return y
    return y


# In[ ]:


clean_total_sqft("13Sq. Yards")


# In[ ]:


# Lets clean our column and create a cleaned version of it:
df1['total_sqft_cleaned'] = df1['total_sqft'].apply(lambda x : clean_total_sqft(x))
# lets check for abnormal values now :
catch_abnormal_val(df1['total_sqft_cleaned'])


# In[ ]:


# Remove unecessary columns:
df2 = df1.drop(['size','total_sqft'], axis=1)


# All columns are cleaned and no nan values remaining.
# Now we can proceed to feature engineering.

# In[ ]:


df2.head()


# ## Feature Engineering

# In[ ]:


df3 = df2.copy()


# In[ ]:


df3['location'].unique()


# In[ ]:


len(df3['location'].unique())


# In[ ]:


#Lets check number of classes in each categorical columns:
categorical_cols = df3.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f'Number of classes in {col} : {df2[col].nunique()}')


# In[ ]:


# Creating new feature for detecting outliers:
df3['price_per_sqft'] = df3['price']*100000/df3['total_sqft_cleaned']
df3.head()


# In[ ]:


# Checking location statistics:
df3['location'] = df3['location'].apply(lambda x: x.strip())
location_stats = df3.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# We will tag the locations which are having very less counts as others.

# In[ ]:


# Locations with less than 10 count:
locations_stats_less_than_10 = location_stats[location_stats<=10]
locations_stats_less_than_10


# In[ ]:


df3['location'] = df3['location'].apply(lambda x : "other" if x in locations_stats_less_than_10 else x)


# In[ ]:


df3['location'].nunique()


# The unique values have been reduced.
# 
# Now we will detect and remove outliers first. Suppose we have a threshold value for per room sqft given by realestate expert. 
# Using this we can find out anomalies in our data and remove them.

# In[ ]:


# Per room sqft threshold be 300sqft: 
df3 = df3[~(df3.total_sqft_cleaned/df3.BHK < 300)]
df3.shape


# In[ ]:


df3['price_per_sqft'].describe()


# For normal distribution of data, we will keep price values which are near to mean and std.
# Outliers are all above mean+standard_deviation and below mean+standard_deviation.

# In[ ]:


# Function to remove outliers from price_per_sqft based on locations.
# As every location will have different price range.
def remove_price_outlier(df_in):
    df_out = pd.DataFrame()
    for key, subdf in df_in.groupby('location'):
        avg_price = np.mean(subdf.price_per_sqft)
        std_price = np.std(subdf.price_per_sqft)
        # data without outliers: 
        reduced_df = subdf[(subdf.price_per_sqft>(avg_price-std_price)) & (subdf.price_per_sqft<=(avg_price+std_price))]
        df_out =pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out
df4 = remove_price_outlier(df3)
df4.shape


# It was found that in some rows price of 2BHK is very less than 1 BHK. So we  will remove outliers based on BHK for each location. That is we can remove those n BHK apartments whose price_per_sqft is less than mean price_per_sqft of n-1 BHK.

# In[ ]:


# Function to remove BHK outliers:
def remove_bhk_outliers(df_in):
    exclude_indices = np.array([])
    for location, location_subdf in df_in.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_subdf in df_in.groupby('BHK'):
            bhk_stats[bhk] = {
                'mean':np.mean(bhk_subdf.price_per_sqft),
                'std':np.std(bhk_subdf.price_per_sqft),
                'count':bhk_subdf.shape[0]
            }
        for bhk, bhk_subdf in location_subdf.groupby('BHK'):
            stats = bhk_stats.get(bhk-1) #statistics of n-1 BHK
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices, bhk_subdf[bhk_subdf.price_per_sqft<(stats['mean'])].index.values)
    return df_in.drop(exclude_indices, axis='index')
        
df5 = remove_bhk_outliers(df4)
df5.shape


# In[ ]:


# Visualize to see number of data points for price_per_sqft
plt.hist(df5.price_per_sqft, rwidth=0.8)
plt.xlabel("Price Per Sqft.")
plt.ylabel("Count")


# In Data Cleaning section, we encountered that there exist number of bathrooms more than 10.
# Lets tackle that in this section.

# In[ ]:


df5[df5.bath>10]


# In[ ]:


# Visualize to see data points based on number of bathrooms:
plt.hist(df5.bath, rwidth=0.8)
plt.xlabel("Number of Bathrooms")
plt.ylabel("Count")


# We can consider the following for bathroom outlier, that **we cannot have (number of bathrooms) more than  (number of bedrooms)+2**

# In[ ]:


df5[df5.bath > df5.BHK+2]


# These rows are our outliers for bathrooms.

# In[ ]:


# Remove bathroom outliers:
df6 = df5[df5.bath<df5.BHK+2]
df6.shape


# In[ ]:


df6.head()


# Outliers removal is done. Now we can remove the extra column "price_per_sqft"

# In[ ]:


df7 = df6.drop(['price_per_sqft'], axis=1)
df7.head()


# As location variable is an categorical feature, we will create dummy columns for location feature using get dummies function.

# In[ ]:


location_dummies = pd.get_dummies(df7.location)
location_dummies.head()


# As this generated binary columns of locations, it is obvious that if any one the row value is 1 then rest are 0. So we will remove one column.
# Whenever there are N classes in a feature, we keep N-1 dummies for it. Here we will drop 'other' column

# In[ ]:


df8 = pd.concat([df7, location_dummies.drop('other', axis='columns')], axis='columns')
df8.head()


# In[ ]:


# Remove Location Column:
df9 = df8.drop(['location'], axis='columns')
df9.head()


# In[ ]:


df9.shape


# Feature engineering is done. We are now at 9th pipe stage and can proceed further for Prediction model building.

# ## Model Building
# ### Data Spliting: (Independent: X, Dependent:y) 

# In[ ]:


# Independent variables:
X = df9.drop('price', axis='columns')
X.head()


# In[ ]:


# Dependent Variable:
y = df9['price']
y.head()


# ### Data Spliting: (train and test)
# (X_train, y_train, X_test, y_test)
# 
# We will keep 20% of sample data for test and rest for training.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


# In[ ]:


# Linear Regression: 
from  sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print(f'Score: {lin_reg.score(X_test, y_test)}')


# In[ ]:


# K-fold validation for Linear Regression:
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
cv1 = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
cross_val_score(LinearRegression(), X, y, cv=cv1)


# ## Model Selection and Parameter Tuning

# Using GridSearch lets find out best model among Linear reg, Lasso reg, DecisionTree reg

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet


# In[ ]:


def find_best_model_grid_search(X, Y, tqdm=tqdm):
    algos = {
        'Linear_regression' : {
            'model' : LinearRegression(),
            'params': {
                'normalize':[True, False]
             }
          },  
         'Lasso' : {
             'model': Lasso(),
             'params': {
                  "max_iter": [1, 5, 10],
                 'alpha': [0.02, 0.024, 0.025, 0.026, 0.03, 0.05, 0.5, 1,2],
                 'selection':['random', 'cyclic'],
                  'normalize':[True, False]
             }
          },
         'Ridge' : {
             'model' : Ridge(),
             'params': {
                  "max_iter": [1, 5, 10],
                 'alpha': [0.05, 0.1, 0.5, 1, 5, 10, 200, 230, 250,265, 270, 275, 290, 300, 500],
                  'normalize':[True, False]
             }
         },
        'ElasticNet' : {
             'model' : ElasticNet(),
             'params' : {
                 "max_iter": [1, 5, 10],
                 'alpha': [0, 0.01, 0.02, 0.03, 0.05, 0.5, 1, 0.05, 0.1, 0.5, 1, 5, 10, 100],
                 'l1_ratio': np.arange(0.0, 1.0, 0.1),
                 'normalize':[True, False]
             } 
         },
          'Decision_tree': {
              'model': DecisionTreeRegressor(),
              'params': {
                  'criterion' : ['mse', 'friedman_mse'],
                  'splitter': ['best', 'random']
              }
          }
    }
    values = (algos.items())
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    print(f'Grid Search CV Initiated..' )    
    with tqdm(total=len(values), file=sys.stdout) as pbar:
        for algo_name, config in algos.items():
            pbar.set_description('Processed')
            gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
            gs.fit(X,Y)
            scores.append({
                'model': algo_name,
                'best_score': gs.best_score_,
                'best_params': gs.best_params_
            })
            pbar.update(1)
            print(f'Grid search CV for {algo_name} done')
        print("Grid Search CV completed!")
    return pd.DataFrame(scores,columns=['model','best_score','best_params'])


# In[ ]:


models = find_best_model_grid_search(X, y)
models


# Clearly, we can see Ridge is performing well. Lasso and Ridge are actually any other regression model. They are regularization methods of linear regressions.
# To know more about Lasso and Ridge:https://www.analyticsvidhya.com/blog/2017/06/a-comprehensive-guide-for-linear-ridge-and-lasso-regression/ .
# Lets collect its best parameters and proceed to re-train our model using these parameters.

# In[ ]:


# Ridge best parameters:
models.loc[3]['best_params']


# In[ ]:


# Re-train using best parameter:
model = Ridge(alpha=0.1, max_iter=1)
model.fit(X_train, y_train)


# In[ ]:


# Prediction:
ypred = model.predict(X_test)


# In[ ]:


# Visualising the test vs predicted data:
plt.scatter(ypred, y_test)
plt.title('Actual Price vs Predicted Price (in Lacs)')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')


# In[ ]:


# Calculate the absolute errors
errors = abs(ypred - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[ ]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / y_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[ ]:


X.columns


# In[ ]:


pd.DataFrame(model.coef_,X.columns,columns=['Coefficient'])


# Our Model is ready!! 
# 
# ## Predictions

# In[ ]:


# Prediction Function
def predict_price(location, sqft, bath, bhk, data=X):
    loc_index = np.where(data.columns==location)[0][0]
    x = np.zeros(len(data.columns)) #init a new array with zero values.
    x[0] = bath
    x[1] = bhk
    x[2] = sqft
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0]


# In[ ]:


predict_price('1st Phase JP Nagar',1000,2,2)


# In[ ]:


# Indira Nagar is most expensive in Bengaluru. Lets predict
predict_price('Indira Nagar',1000,2,2)


# ## Saving Model

# In[ ]:


# saving ml model as pickle file:
import pickle
with open('bengaluru_home_price_model.pickle','wb') as f:
    pickle.dump(model,f)


# In[ ]:


# saving column names as a json file:
import json
columns = {
    'data_columns' : [col.lower() for col in X.columns]
}
with open('columns.json', 'w') as f:
    f.write(json.dumps(columns))


# Done!!
