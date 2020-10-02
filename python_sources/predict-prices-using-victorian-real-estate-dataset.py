#!/usr/bin/env python
# coding: utf-8

# # Victorian Real Estate Data Analysis
# 
# In this project, I will analysis victorian real estate data to estimate price of sold properties. Let's begin!!!
# 
# The goal of this project is to identify important features among provided features to predict the sold price of a property. As we want to predict a continuous value, it is a regression problem. I will use the RandomForest regression algorithm to predict the price.
# 
# ## What you can learn from this project
# 1. Basic data handling and manipulation (remove column, move column, update row) using pandas
# 2. Converting object to desire data type (price is originally string, I will convert it to float)
# 3. Visualising data using matplotlib and seaborn
# 4. Application of gridspace for flexiable subplotting
# 5. Heatmap to view correlation
# 6. get_dummies() for handling text data
# 7. Creating a randomforest model using the training data
# 8. Evaluating the model using the testing data
# 9. How to use gridsearch with cross validation to tune hyperparameters

# In[2]:


# include the required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


# In[3]:


# read the dataset
df = pd.read_csv('../input/_sold.csv')


# ## Explore the dataset
# This function will explore the dataset. Specifically, it will explore the following information from the dataframe.
# 1. Shape of the dataframe (row, column)
# 2. Head and tail to view some entries
# 3. Info() to check datatypes, memory uses
# 4. Describe() to view several statistical information regarding the dataframe

# In[4]:


def explore_data(df):
    # check the shape of the dataframe
    print('The number of rows and columns are: ' + str(df.shape))
    print('-'*50)
    
    # List of columns
    print('Column names')
    print(list(df.columns))
    print('-'*50)
    
    # check the head and tail of the dataframe
    print('Some entries from the begining')
    print(df.head(5))
    print('-'*50)
    print('Some entries from the bottom')
    print(df.tail(5))
    print('-'*50)
    
    # information of the dataset
    print(df.info())
    print('-'*50)
    
    # Some statistical information
    print(df.describe().transpose())
    print('-'*50)
    


# In[5]:


explore_data(df)


# We want to predict price, it is not a floting point datatype, currently it is string having '$', ',' and 'Contact agent'. The following function will convert the price column into floating point.

# In[6]:


# We will replace $ and , with empty string and 'Contact agaent' with zero. 
def convert_price(price):
    new_price = price.replace(',','').replace('$','').replace('Contact agent','0')
    return float(new_price)


# In[7]:


# convert and update the column
df['price'] = df['price'].apply(convert_price)


# In[8]:


# changing zero price with the mean value
df['price'].replace(0, df['price'].mean()) 

# check the price column
df.describe()


# In[9]:


# check for missing data
df.isna().sum()


# We have some missing data, we will remove some of the columns from the analysis. The following function takes care of it.

# In[10]:


def remove_columns(df):
    # drop a list of columns
    drop_columns = ['streetAddress', 'suburb', 'region', 'listingId', 'title', 'dateSold', 'modifiedDate']
    df.drop(labels = drop_columns, axis=1, inplace=True)
    
    # drop the row which which na in the latitude and longitude
    df.dropna(axis=0, how='any', subset=['latitude', 'longitude'], inplace=True)


# In[11]:


remove_columns(df)


# We get rid of many columns. Now check the seven columns which we will use as the feature matrix to predict the price. Check the data distribution of each independent variable and the dependent variable.

# In[12]:


def data_visualisation(df):
    fig = plt.figure(constrained_layout=True, figsize=(15,15))
    gs = GridSpec(3, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    sns.distplot(df['latitude'], bins= 50)
    
    ax2 = fig.add_subplot(gs[0, 1])
    sns.distplot(df['longitude'], bins= 50)
    
    ax3 = fig.add_subplot(gs[0, 2])
    sns.distplot(df['postcode'], bins= 50)
    
    ax4 = fig.add_subplot(gs[1, 0])
    sns.distplot(df['bedrooms'], bins= 50)
    
    ax5 = fig.add_subplot(gs[1, 1])
    sns.distplot(df['bathrooms'], bins= 50)
    
    ax6 = fig.add_subplot(gs[1, 2])
    sns.distplot(df['parkingSpaces'], bins= 50)
    
    prop_type = df['propertyType'].value_counts(dropna=False)
    prop_dict = {}
    for i in range(len(prop_type.index)):
        prop_dict[prop_type.index[i]] = prop_type.values[i]
    ax7 = fig.add_subplot(gs[2, :-1])
    plt.bar(prop_dict.keys(), prop_dict.values())
    
    ax8 = fig.add_subplot(gs[2, -1])
    sns.distplot(df['price'], bins= 50)


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
data_visualisation(df)


# Also check the pairwise distribution

# In[14]:


sns.set()
g = sns.PairGrid(df)
g = g.map_diag(plt.hist)
g = g.map_offdiag(plt.scatter)


# In[15]:


# correlation matrix with sns heatmap
plt.figure(figsize=(8,8))
sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')


# In[16]:


# map propertyType to integer values and remove the old column from the dataset and get rid of the dummy variable trap
df = pd.get_dummies(data=df, columns=['propertyType'], drop_first=True)


# In[17]:


# move price at the end of the dataframe
cols = list(df.columns)
# remove the desire column from the dataframe
cols.pop(cols.index('price'))
# add it back to the dataframe
df = df[cols+['price']]


# In[18]:


# independent variable
X = df.iloc[:,:-1].values
# dependent variable
y = df.iloc[:,-1].values


# In[19]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, r2_score, explained_variance_score)


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[21]:


rf_regressor = RandomForestRegressor(criterion='mse',random_state=0, n_estimators=600, n_jobs=-1,
                                    max_features='sqrt')


# In[22]:


rf_regressor.fit(X_train, y_train)


# In[23]:


rf_pred = rf_regressor.predict(X_test)


# In[24]:


plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
sns.distplot(df['price'], bins=50, color='green')
plt.xlabel('Origin price')
plt.subplot(1,2,2)
sns.distplot(rf_pred, bins=50, color='red')
plt.xlabel('Predicted price')


# In[25]:


# mean squared error
mean_squared_error(y_test, rf_pred)


# In[26]:


# r2 score
r2_score(y_test, rf_pred)


# I used gridsearch to find n_estimators and max_features. I commented the following code, which can be used to tune hyperparameter

# In[27]:


# Applying grid search for hyperparameter tunning
# from sklearn.model_selection import GridSearchCV


# In[28]:


# parameters = [{'n_estimators': [300, 400, 500, 600, 700], 'max_features': ['auto', 'sqrt', 'log2']}]


# In[29]:


# grid_search = GridSearchCV(estimator=rf_regressor, param_grid=parameters, scoring='neg_mean_squared_error',
                            # n_jobs=-1,cv=10
                           # )


# In[30]:


# grid_search = grid_search.fit(X_train, y_train)


# In[31]:


# grid_search.best_params_


# That's all for now.   
# Thanks

# In[ ]:




