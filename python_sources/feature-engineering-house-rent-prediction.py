#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning Model Building Pipeline: Data Analysis
# 
# In this following notebook, we will go through the Data Analysis step in the Machine Learning model building pipeline. There will be a notebook for each one of the Machine Learning Pipeline steps:
# 
# 1. [Data Analysis](https://www.kaggle.com/rkb0023/exploratory-data-analysis-house-rent-prediction)
# 2. Feature Engineering
# 3. [Model Building](https://www.kaggle.com/rkb0023/model-building-house-rent-prediction)
# 
# **This is the notebook for step 2: Feature Engineering**
# 
# The dataset can be found in [iNeuron](https://challenge-ineuron.in/mlchallenge.php#) ML Challenge 2.
# 
# <hr>
# 
# ## Predicting Rent Price of Houses
# 
# The aim of the project is to build a machine learning model to predict the rent price of homes based on different explanatory variables describing aspects of residential houses. 
# 
# 
# <hr>

# ## House Prices dataset: Feature Engineering
# 
# In the following cells, we will engineer / pre-process the variables of the House Rent Dataset from iNeuron. We will engineer the variables so that we tackle:
# 
# 1. Missing values
# 2. Temporal variables
# 3. Non-Gaussian distributed variables
# 4. Categorical variables: remove rare labels
# 5. Categorical variables: convert strings to numbers
# 5. Standarise the values of the variables to the same range
# 
# Let's go ahead and load the dataset.

# In[ ]:


# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)

import warnings
warnings.simplefilter(action='ignore')


# In[ ]:


data = pd.read_csv('../input/houserentpredictiondataset/houseRent/housing_train.csv')
data.shape


# In[ ]:


data.info()


# ## Data Cleaning

# Getting appropriate data types. For example baths have a dtype of float, and it contains some decimal values. But it is a quantitative variable. So transforming it to remove decimal.

# In[ ]:


data['baths'] = np.ceil(data['baths'])
data['baths'] = data['baths'].astype(np.int)


# ## Missing values
# 
# ### Categorical variables

# In[ ]:


# make a list of the categorical variables that contain missing values
cat_var_na = ['laundry_options', 'parking_options']


# In[ ]:


def impute_missing_cat(data, var, modeof):
    return data.groupby(modeof)[var].transform(
        lambda x: x.fillna(x.mode()[0]))


# In[ ]:


data["laundry_options"] = impute_missing_cat(data, "laundry_options", "type")
data["parking_options"] = impute_missing_cat(data, "parking_options", "type")
data = data.dropna(subset=["state", "description"],axis=0)


# ### Numerical variables
# 

# In[ ]:


# make a list with the numerical variables that contain missing values
num_var_na = ['lat', 'long']


# In[ ]:


def impute_missing_num(data, var, meanof):
    return data.groupby(meanof)[var].transform(
        lambda x: x.fillna(x.mode()[0]))


# In[ ]:


data["lat"] = impute_missing_num(data, "lat", "region")
data["long"] = impute_missing_num(data, "long", "region")


# Is there any remaining missing value ? 

# In[ ]:


#Check remaining missing values if any 
all_data_na = (data.isnull().sum() / len(data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


# In[ ]:


data.shape


# ## Outliers

# In EDA, we decided to remove outliers according to the upper and lower bound of its interquartile range.

# In[ ]:


price_upper, price_lower = 2400, 1
sqfeet_upper, sqfeet_lower = 1762, 1
beds_upper, beds_lower = 3, 1
baths_upper, baths_lower = 3, 1

data = data[(data['price'] <= price_upper) & (data['price'] >= price_lower)]
data = data[(data['sqfeet'] <= sqfeet_upper) & (data['sqfeet'] >= sqfeet_lower)]
data = data[(data['beds'] <= beds_upper) & (data['beds'] >= beds_lower)]
data = data[(data['baths'] <= baths_upper) & (data['baths'] >= baths_lower)]


# In[ ]:


data.shape


# ## Getting More Features

# In[ ]:


data['premium_house'] = np.where((data['baths']>=data['beds'])&(data['beds']>1),1,0)
data['pets_allowed'] = np.where((data['cats_allowed']==1)&data['dogs_allowed']==1,1,0)
data['beds_per_sqfeet'] = data['beds'] / data['sqfeet']
data['baths_per_beds'] = data['baths'] / data['beds']


# ## Exploring *'description'* column

# In[ ]:


data.description[82226].lower()


# In[ ]:


[x in data.description[82226].lower() for x in ['pool', 'swimming','wi-fi','fireplace','grilling','gym','fence', 'court']]


# Let's get some intriguing new features

# In[ ]:


data['has_pool'] = data['description'].apply(lambda x: 1 if 'pool' in x.lower() or 'swimming' in x.lower() else 0)
data['has_grill'] = data['description'].apply(lambda x: 1 if 'grill' in x.lower() or 'grilling' in x.lower() else 0)
data['has_fireplace'] = data['description'].apply(lambda x: 1 if 'fireplace' in x.lower() or 'fire pits' in x.lower() else 0)
data['gym_nearby'] = data['description'].apply(lambda x: 1 if 'gym' in x.lower() or 'fitness' in x.lower() else 0)
data['school/clg_nearby'] = data['description'].apply(lambda x: 1 if 'school' in x.lower() or 'college' in x.lower() else 0)
data['wifi_facilities'] = data['description'].apply(lambda x: 1 if 'wifi' in x.lower() or 'wi-fi' in x.lower() else 0)
data['valet_service'] = data['description'].apply(lambda x: 1 if 'valet' in x.lower() else 0)
data['shopping_nearby'] = data['description'].apply(lambda x: 1 if 'shopping' in x.lower() else 0)
data['sports_playground'] = data['description'].apply(lambda x: 1 if 'sport' in x.lower()  or 'sports' in x.lower() 
                                                      or 'tennis' in x.lower() or 'soccer' in x.lower() 
                                                      or 'soccers' in x.lower() or 'court' in x.lower() else 0)
data['dining_nearby'] = data['description'].apply(lambda x: 1 if 'dining' in x.lower() else 0)


# In[ ]:


data.columns


# In[ ]:


for var in ['has_pool', 'has_grill', 'has_fireplace', 'gym_nearby',
       'school/clg_nearby', 'wifi_facilities', 'valet_service',
       'shopping_nearby', 'sports_playground', 'dining_nearby']:
    print(data[var].value_counts())


# ## Numerical variable transformation
# 
# We will log transform the positive numerical variables in order to get a more Gaussian-like distribution. This tends to help Linear machine learning models. 

# In[ ]:


for var in ['price','sqfeet','baths_per_beds','beds_per_sqfeet']:
    data[var] = np.log(data[var])


# In[ ]:


# check that data set does not contain null values in the engineered variables
[var for var in ['price','sqfeet','baths_per_beds','beds_per_sqfeet'] if data[var].isnull().sum() > 0]


# ## Categorical variables
# 
# ### Removing rare labels
# 
# First, we will group those categories within variables that are present in less than 1% of the observations. That is, all values of categorical variables that are shared by less than 1% of houses, well be replaced by the string "Rare".
# 
# To learn more about how to handle categorical variables visit our course [Feature Engineering for Machine Learning](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=UDEMY2018) in Udemy.

# In[ ]:


# let's capture the categorical variables in a list

cat_vars = ['region', 'type', 'laundry_options', 'parking_options', 'state']


# In[ ]:


frequent_ls = {
    'region': 
        ['denver', 'fayetteville', 'jacksonville', 'omaha / council bluffs', 'rochester'],
     'type': 
        ['apartment', 'condo', 'duplex', 'house', 'manufactured', 'townhouse'],
     'laundry_options': 
        ['laundry in bldg', 'laundry on site', 'w/d hookups', 'w/d in unit'],  
     'parking_options': 
        ['attached garage', 'carport', 'detached garage', 'off-street parking', 'street parking'],
     'state': 
        ['al', 'ar', 'az', 'ca', 'co', 'ct', 'fl', 'ga', 'ia', 'id', 'il', 'in', 'ks', 'ky', 'la', 
        'ma', 'md', 'mi', 'mn', 'ms', 'nc', 'nd', 'ne', 'nj', 'nm', 'nv', 'ny', 'oh']
}


for var in cat_vars:
    data[var] = np.where(data[var].isin(
        frequent_ls[var]), data[var], 'Rare')


# ### Encoding of categorical variables
# 
# Next, we need to transform the strings of the categorical variables into numbers. We will do it so that we capture the monotonic relationship between the label and the target.
# 
# To learn more about how to encode categorical variables visit our course [Feature Engineering for Machine Learning](https://www.udemy.com/feature-engineering-for-machine-learning/?couponCode=UDEMY2018) in Udemy.

# In[ ]:


# this function will assign discrete values to the strings of the variables,
# so that the smaller value corresponds to the category that shows the smaller
# mean house sale price


def replace_categories(data, var, target):

    # order the categories in a variable from that with the lowest
    # house sale price, to that with the highest
    ordered_labels = data.groupby([var])[target].mean().sort_values().index

    # create a dictionary of ordered categories to integer values
    ordinal_label = {k: i for i, k in enumerate(ordered_labels, 0)}

    # use the dictionary to replace the categorical strings by integers
    data[var] = data[var].map(ordinal_label)


# In[ ]:


for var in cat_vars:
    replace_categories(data, var, 'price')


# In[ ]:


# check absence of na in the train set
[var for var in data.columns if data[var].isnull().sum() > 0]


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


features = ['region', 'price', 'type', 'sqfeet', 'smoking_allowed', 'wheelchair_access', 
            'electric_vehicle_charge', 'comes_furnished', 'laundry_options', 'parking_options','lat', 'long', 
            'premium_house', 'pets_allowed', 'beds_per_sqfeet', 'baths_per_beds', 'has_pool', 'has_grill', 
            'has_fireplace', 'gym_nearby', 'school/clg_nearby', 'wifi_facilities', 'valet_service', 
            'shopping_nearby', 'sports_playground', 'dining_nearby']

data_final = data[features].copy()
data_final.head()


# In[ ]:


for feature in features:
    data_final[feature] = data_final[feature].astype(np.float64)


# ### Correlation Heatmap

# In[ ]:


corr_matrix = data.corr()
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)]= True

fig, ax = plt.subplots(figsize=(25,25)) 

sns.heatmap(corr_matrix, 
            annot=True, 
            square=True,
            fmt='.2g',
            mask=mask,
            ax=ax).set(
    title = 'Feature Correlation', xlabel = 'Columns', ylabel = 'Columns')

ax.set_yticklabels(corr_matrix.columns, rotation = 0)
ax.set_xticklabels(corr_matrix.columns)
sns.set_style({'xtick.bottom': True}, {'ytick.left': True})


# In[ ]:


data_final.to_csv('data_cleaned.csv', index=False)


# In[ ]:




