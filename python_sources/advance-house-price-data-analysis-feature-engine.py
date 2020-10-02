#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

pd.pandas.set_option('display.max_columns',None)


# In[ ]:


dataset= pd.read_csv('../input/house-prices-advanced-regression-techniques/housetrain.csv')
dataset.head()


# In[ ]:





# Data Analysis part 1

# In[ ]:


features_with_nan =[features for features in dataset.columns if dataset[features].isnull().sum()>1]

for feature in features_with_nan:
    print(feature, np.round(dataset[feature].isnull().mean(),4), '% missing values')


# In[ ]:


dataset.shape


# In[ ]:


for feature in features_with_nan:
    data= dataset.copy()
    
    data[feature]= np.where(data[feature].isnull(),1,0)
    
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.title(feature)
    plt.show()


# here with the relation between the missing vallue and dependent variable is clearly visible. so we need to replace these
# nan values with something meaningful which we will do in feature engineering section

# In[ ]:


# from the dataset, features like id is not required

print("Id of houses {}".format(len(dataset.Id)))


# # numerical variable

# In[ ]:


# list of numerical variables
numerical_features=[ feature for feature in dataset.columns if dataset[feature].dtypes !='O']

print('Number of numerical variables:' , len(numerical_features))

# visualizing the numerical variable
dataset[numerical_features].head()


# Temporal Variables( eg: DateTime Variables)
# From the dataset we have 4 year variables.

# In[ ]:


# list of variable that contain year information
year_feature=[feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

print(year_feature)


# In[ ]:


# exploring the content of these year variables
for feature in year_feature:
    print(feature, dataset[feature].unique())


# In[ ]:


#lets analyze the temporal Datetime variables

dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Median House Price')
plt.title('House price vs YearSold')


# In[ ]:


year_feature


# In[ ]:


## Numerical variables are usually of 2 type
## 1. Continous variable and Discrete Variables

discrete_feature=[feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_feature+['Id']]
print("Discrete Variables Count: {}".format(len(discrete_feature)))


# In[ ]:


discrete_feature


# In[ ]:


dataset[discrete_feature].head()


# In[ ]:


# finding relationship between them and sale price

for feature in discrete_feature:
    data= dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# In[ ]:


# Continous variable

continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature+year_feature+['Id']]
print("Continuous feature Count {}".format(len(continuous_feature)))


# In[ ]:


for feature in continuous_feature:
    data= dataset.copy()
    data[feature].hist(bins=25)
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.title(feature)
    plt.show()


# # EDA PART 2

# In[ ]:


## we will be using logarithmic transformation

data= dataset.copy()
for feature in continuous_feature:
    data= dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]= np.log(data[feature])
        data['SalePrice']= np.log(data['SalePrice'])
        plt.scatter(data[feature],data['SalePrice'])
        plt.xlabel(feature)
        plt.ylabel('SalePrice')
        plt.title(feature)
        plt.show()


# ### OUTLIERS

# In[ ]:


for feature in continuous_feature:
    data= dataset.copy()
    if 0 in data[feature].unique():
        pass
    else:
        data[feature]= np.log(data[feature])
        data.boxplot(column=feature)
        
        plt.ylabel(feature)
        plt.title(feature)
        plt.show()
        
    


# ### categorical variables

# In[ ]:


categorical_features= [ feature for feature in dataset.columns if data[feature].dtype=='O']
categorical_features
                      


# In[ ]:


dataset[categorical_features].head()


# In[ ]:


for feature in categorical_features:
    print('The feature is {} and number of categories are {}'.format(feature,len(dataset[feature].unique())))


# In[ ]:


## to find relation betwwen categorical_features and sale price

for feature in categorical_features:
    data= dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('SalePrice')
    plt.title(feature)
    plt.show()


# # Feature Engineering

# In[ ]:


## Always remember there way always be a chance of data leakage so we need to split the data first and then apply feature
## Engineering
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(dataset,dataset['SalePrice'],test_size=0.1,random_state=0)


# In[ ]:


X_train.shape, X_test.shape


# In[ ]:


## MISSING VALUES

## Let us capture all the nan values
## First lets handle Categorical features which are missing
features_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes=='O']

for feature in features_nan:
    print("{}: {}% missing values".format(feature,np.round(dataset[feature].isnull().mean(),4)))


# In[ ]:


## Replace missing with new label
def replace_cat_feature(dataset,features_nan):
    data= dataset.copy()
    
    data[features_nan]=data[features_nan].fillna('Missing')
    return data

dataset=replace_cat_feature(dataset,features_nan)

dataset[features_nan].isnull().sum()


# In[ ]:


dataset.head()


# In[ ]:



## Now lets check for numerical variables the contains missing values
numerical_with_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtypes!='O']

## We will print the numerical nan variables and percentage of missing values

for feature in numerical_with_nan:
    print("{}: {}% missing value".format(feature,np.around(dataset[feature].isnull().mean(),4)))


# In[ ]:


# replacing numerical missing values
for feature in numerical_with_nan:
    
    median_value= dataset[feature].median()
    # we will replace using median since there are outliers
    dataset[feature+'nan']=np.where(dataset[feature].isnull(),1,0)
    dataset[feature].fillna(median_value,inplace=True)

dataset[numerical_with_nan].isnull().sum()


# In[ ]:


dataset.head(50)


# In[ ]:


## Temporal Variables (Date Time Variables)

for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
       
    dataset[feature]=dataset['YrSold']-dataset[feature]


# In[ ]:


dataset.head()


# In[ ]:


dataset[['YearBuilt','YearRemodAdd','GarageYrBlt']].head()


# In[ ]:


num_features=['LotFrontage','LotArea','1stFlrSF','GrLivArea','SalePrice']

for feature in num_features:
    dataset[feature]=np.log(dataset[feature])


# In[ ]:


dataset.head()


# ### Handeling rare categorical Feature

# we will remove categorical features that are present less than 1 % of observations

# In[ ]:


categorical_features=[ feature for feature in dataset.columns if dataset[feature].dtype=='O']


# In[ ]:


categorical_features


# In[ ]:


for feature in categorical_features:
    temp=dataset.groupby(feature)['SalePrice'].count()/len(dataset)
    temp_df= temp[temp>0.01].index
    dataset[feature]=np.where(dataset[feature].isin(temp_df),dataset[feature],'rare_var')


# In[ ]:


dataset.head()


# In[ ]:





# ## Feature Scaling
# 

# In[ ]:


for feature in categorical_features:
    labels_ordered=dataset.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    dataset[feature]=dataset[feature].map(labels_ordered)


# In[ ]:


dataset.head()


# In[ ]:


feature_scale=[feature for feature in dataset.columns if feature not in ['Id','SalePrice']]

from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler()
scaler.fit(dataset[feature_scale])


# In[ ]:


scaler.transform(dataset[feature_scale
                        ])


# In[ ]:


# transform the train& test and add on the Id and SalePrice variable
data= pd.concat([dataset[['Id','SalePrice']].reset_index(drop=True),
                   pd.DataFrame(scaler.transform(dataset[feature_scale]),columns=feature_scale)],axis=1)


# In[ ]:


data.head()


# ## Feature Selection

# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel


# In[ ]:


## Capture the dependent feature
y_train=dataset[['SalePrice']]

## drop dependent feature from dataset
X_train=dataset.drop(['Id','SalePrice'],axis=1)


# In[ ]:


### apply feature selection
# first, I specify the Lasso Regression model, and I
# select a suitable alpha (equivalent of penalty).
# The bigger the alpha the less features that will be selected.

# Then I use the selectFromModel object from sklearn, which
# will select the features which coefficients are non-zero

feature_sel_model= SelectFromModel(Lasso(alpha=0.005,random_state=0))
feature_sel_model.fit(X_train,y_train)


# In[ ]:


feature_sel_model.get_support()


# In[ ]:


# lets print  the number of total and selected features

# this is how we can make a list of our selected_features
selected_feat = X_train.columns[(feature_sel_model.get_support())]

# let's print some stats
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
    np.sum(feature_sel_model.estimator_.coef_ == 0)))


# In[ ]:


selected_feat


# In[ ]:


X_train=X_train[selected_feat]


# In[ ]:




