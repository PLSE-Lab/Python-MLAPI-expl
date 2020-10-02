#!/usr/bin/env python
# coding: utf-8

# # EDA (Exploratory Data Analysis)
# 
# This file is to analyze the features being used for Income Prediction
# **Search WeirdStuff{num} and the visualisation/code regarding fact should show up **
# ## Interesting Finds
# 
# 1. WeirdStuff1: Instances is TOO highly correlated to Year of Record which in turn is highly correlated to Target Income.
#     - This may lead to doubting affect of Year of Record of if Instances may be useful in testing data as well.
#     
# 2. Some houses just dont exists during longe ranges of years
#     - We may need to find a way to restrict housing impact to only certain years. Because in test set there are sets that are not in training 
#     
# 3. WeirdStuff3: Gender female doing worse than f 
#     - Maybe best to keep them seperate
# 

# In[ ]:



get_ipython().system('pip install featexp')
get_ipython().system('pip install --upgrade numpy')
from featexp import get_univariate_plots

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(font_scale=1)

# Data Prediction
from sklearn import preprocessing
import xgboost
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
import lightgbm as lgb
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import LabelEncoder

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input/inc-ml/"))


# In[ ]:


# Importing the dataset

train_dt = pd.read_csv('../input/inc-ml/tcd-ml-1920-group-income-train.csv') 
test_dt = pd.read_csv('../input/inc-ml/tcd-ml-1920-group-income-test.csv')

Y = train_dt['Total Yearly Income [EUR]']
#Display columns and shape
print(train_dt.shape)
print(test_dt.shape)
train_dt.head()
test_dt.head()


# In[ ]:


print(train_dt['Instance'].nunique())
print(len(train_dt['Instance']))


# In[ ]:


get_univariate_plots(data=train_dt, target_col='Total Yearly Income [EUR]', 
                     features_list=['Wears Glasses'], bins=10)


# # Null Values in Training and Test Set 
# Not sure if its including blanks in other strings

# In[ ]:


#Counting Null Function
def null_values(df):
    
    sum_null = df.isnull().sum()
    total = df.isnull().count()
    percent_nullvalues = 100* sum_null / total 
    df_null = pd.DataFrame()
    df_null['Total'] = total
    df_null['Null_Count'] = sum_null
    df_null['Percent'] = round(percent_nullvalues,2)
    df_null = df_null.sort_values(by='Null_Count',ascending = False)
    df_null = df_null[df_null.Null_Count > 0]
    
    return(df_null)

# Training and Testing null functions
print(null_values(train_dt))
print(null_values(test_dt))


# # Unique values in each and .describe() stats of each feature
# Shows distribution of values and the unique values in each feature, both training and test set

# In[ ]:


train_dt.describe()


# In[ ]:


test_dt.describe()


# In[ ]:


# Seperate categorical and numeric features
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_df = train_dt.select_dtypes(include=numerics)
cat_df = list(set(train_dt.columns)-set(num_df))

# Display different types in training features
for feature in cat_df:
    print('---'+feature+'---')
    print(train_dt[feature].value_counts())
    print()

# Display different types in testing features
for feature in num_df:
    print('---'+feature+'---')
    print(test_dt[feature].value_counts())
    print()


# # Filling Missing values
# Replace missing values for **all to mode of  feature**

# In[ ]:


#Remove null values with most common values
for feature in train_dt.columns:
    train_dt = train_dt.fillna({
    feature:train_dt[feature].mode()[0]
    })


print(null_values(train_dt))

#Manually Encode Values


# # Feature Engineering
# 
# Creating new Features by means of 
# 1. Label Encoding
# 2. Dummy Encoding

# ## Dummy and Label Encoding 
# #### Label Encoding 
# - Hair Color
# - Gender (3,2,1,0)
# - Uni Degree (3,2,1,0)
# - Satisfation with employer	
# 
# #### Dummy Enconding variables
# - ...

# In[ ]:


train_dt['Gender'].value_counts()


# In[ ]:


train_dt['University Degree'].value_counts()


# In[ ]:


train_dt['Hair Color'].value_counts()


# In[ ]:


train_dt['Satisfation with employer'].value_counts()


# In[ ]:


test_dt


# In[ ]:


train_dt = pd.concat([train_dt, test_dt],axis='rows').reset_index()
for feature in train_dt.columns:
    train_dt = train_dt.fillna({
    feature:train_dt[feature].mode()[0]
    })


# In[ ]:


# Label Encode Gender, Degree, Hair-Color

# Hair
le_gender= train_dt['Gender']
le_gender = le_gender.str.replace('female','2')
le_gender = le_gender.str.replace('male','3')
le_gender = le_gender.str.replace('other','0')
le_gender = le_gender.str.replace('unknown','4')
le_gender = le_gender.str.replace('0','4')
le_gender = le_gender.str.replace('f','2')
le_gender = le_gender.apply(lambda v: int(v))
print(le_gender.value_counts())
le_gender = pd.DataFrame({'gender_le':le_gender}) 


# Degree
le_degree = train_dt['University Degree']
le_degree = le_degree.str.replace('No','0')
le_degree = le_degree.str.replace('Bachelor','1')
le_degree = le_degree.str.replace('Master','2')
le_degree = le_degree.str.replace('PhD','3')
le_degree = le_degree.str.replace('0','0')
le_degree = le_degree.apply(lambda v: int(v))
print(le_degree.value_counts())
le_degree = pd.DataFrame({'uni_le':le_degree}) 


# Hair Color
le = LabelEncoder()
le_hc = le.fit_transform(train_dt['Hair Color'])
le_hc = pd.DataFrame({'Hc_le':le_hc}) 

# Satisfation with employer
le_satisfaction = train_dt['Satisfation with employer']
le_satisfaction = le_satisfaction.str.replace('Unhappy','0')
le_satisfaction = le_satisfaction.str.replace('Average','1')
le_satisfaction = le_satisfaction.str.replace('Somewhat Happy','2')
le_satisfaction = le_satisfaction.str.replace('Happy','3')
le_satisfaction = le_satisfaction.apply(lambda v: int(v))
print(le_satisfaction.value_counts())
le_satisfaction = pd.DataFrame({'satisfaction_le':le_satisfaction}) 


#Additional Income
inc = pd.DataFrame()
inc['additionalInc'] = train_dt['Yearly Income in addition to Salary (e.g. Rental Income)']
inc['additionalInc'] = inc['additionalInc'].str.replace('EUR', '')
inc['additionalInc'] = pd.to_numeric(inc['additionalInc'], errors='coerce')


# ## Adding Dummy Variables
# Adding 
# - Gender
# - Uni Degree
# 
# Dropping Categorical variables* 

# In[ ]:


#Drop categorical columns
# train_dt = train_dt.drop(['Gender','University Degree','Hair Color'],axis=1)

# Add one encoded columns
train_dt = pd.concat([train_dt,le_gender,le_degree,le_hc,le_satisfaction,inc['additionalInc']],axis=1)


# In[ ]:


train_dt.head(3)


# ## Difference between Training and Test Sets
# Only missing values are 
# - 0 in Housing 
# - Countries 

# In[ ]:


# {'Antigua and Barbuda', 'Palau', 'Russia', 'Nigeria', 'Seychelles', 'Liechtenstein', 'Iran', 'Ethiopia', 
# 'San Marino', 'Philippines', 'Tonga', 'Marshall Islands', 'Monaco', 'Nauru', 'Germany', '0'}
for feature in ['Housing Situation','Country','Profession']:
    print('---'+feature+'---')
    tr_vc = train_dt[feature].unique()
    te_vc = test_dt[feature].unique()
    diff_vals = set(te_vc) - set(tr_vc) 
    print(diff_vals)


# ### Unique Countries

# In[ ]:


train_dt['Country'].unique()


# # Correlations
# Checking correlations between variables with Total yearly income and pair plots
# 
# ## MultiColinearity
# Multicollinearity increases the standard errors of the coefficients. That means, multicollinearity makes some variables statistically insignificant when they should be significant.
# 
# To avoid this we can do 3 things:
# 
# Completely remove those variables
# Make new feature by adding them or by some other operation.
# Use PCA, which will reduce feature set to small number of non-collinear features.

# In[ ]:


corr=train_dt.corr()["Total Yearly Income [EUR]"]
corr[np.argsort(corr, axis=0)[::-1]]


# In[ ]:


# WeirdStuff1
correlations=train_dt.corr()
attrs = correlations.iloc[:-1,:-1] # all except target

threshold = 0.005
important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0])     .unstack().dropna().to_dict()

unique_important_corrs = pd.DataFrame(
    list(set([(tuple(sorted(key)), important_corrs[key]) \
    for key in important_corrs])), 
        columns=['Attribute Pair', 'Correlation'])

# sorted by absolute value
unique_important_corrs = unique_important_corrs.ix[
    abs(unique_important_corrs['Correlation']).argsort()[::-1]]

unique_important_corrs


# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_df = train_dt.select_dtypes(include=numerics)
corrMatrix=train_dt[num_df.columns].corr()

sns.set(font_scale=1.10)
plt.figure(figsize=(10, 10))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features');


# In[ ]:


num_df


# In[ ]:


# Univariate Analysis
fig = plt.figure(figsize=(9,3))
for feature in num_df:
    sns.distplot(train_dt[feature], color="r", kde=False)
    plt.title("Distribution of " + feature)
    plt.ylabel("Number of Occurences")
    plt.xlabel(feature)
    plt.show()


# In[ ]:


f = (train_dt
         .loc[train_dt['Housing Situation'].isin(['Small House', 'Large House','Medium Apartment'])]
         .loc[:, ['Total Yearly Income [EUR]', 'Housing Situation', 'Gender']]
    )

sns.boxplot(x="Gender", y="Total Yearly Income [EUR]", hue='Housing Situation', data=f)


# In[ ]:


train_dt['Housing Situation'].value_counts()


# In[ ]:


# WeirdStuff2
sns.lmplot(x='Year of Record', y='Total Yearly Income [EUR]', col='Housing Situation', 
           data=train_dt.loc[train_dt['Housing Situation'].isin(['Small House','Medium House','Large House'])], 
           fit_reg=False)


# In[ ]:


sns.lmplot(x='Year of Record', y='Total Yearly Income [EUR]', col='Housing Situation', 
           data=train_dt.loc[train_dt['Housing Situation'].isin(['Small Apartment','Medium Apartment','Large Apartment'])], 
           fit_reg=False)


# In[ ]:


# train_dt.loc[(train_dt['Housing Situation'] == 'Castle') & (train_dt['Year of Record'] >=1978)] 
# test_dt.loc[(test_dt['Housing Situation'] == 'Large Apartment') & (test_dt['Year of Record'] >=1980) & (test_dt['Year of Record'] >=1990)] 
sns.lmplot(x='Year of Record', y='Total Yearly Income [EUR]', col='Housing Situation', 
           data=train_dt.loc[train_dt['Housing Situation'].isin(['nA','0','Castle'])], 
           fit_reg=False)


# In[ ]:


train_dt.loc[(train_dt['Housing Situation'] == 'Castle') & (train_dt['Total Yearly Income [EUR]'] >=300000)] 


# In[ ]:


sns.lmplot(x='Satisfation with employer', y='Total Yearly Income [EUR]', col='Gender', 
           data=train_dt.loc[train_dt['Housing Situation'].isin([0,1,2,3])], 
           fit_reg=False)


# In[ ]:


# WeirdStuff3: Gender female doing worse than f 
sns.lmplot(x='Crime Level in the City of Employement', y='Total Yearly Income [EUR]', col='Gender', 
           data=train_dt.loc[train_dt['gender_le'].isin([0,1,2,3])], 
           fit_reg=False)


# In[ ]:


train_dt.loc[(train_dt['Housing Situation'] == 'Castle') & (train_dt['Total Yearly Income [EUR]'] >=300000)] 


# In[ ]:


incs = train_dt.groupby('Year of Record').mean()
incs['Year'] = incs.index


# In[ ]:


incs.columns


# In[ ]:


plt.scatter(incs['Year'], incs['Total Yearly Income [EUR]'])
plt.show()


# In[ ]:


countries_mean = train_dt.groupby('Country').mean()
countries_mean['Country'] = countries_mean.index
countries_mean


# In[ ]:


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num_df = countries_mean.select_dtypes(include=numerics)
corrMatrix=train_dt[num_df.columns].corr()

sns.set(font_scale=1.0)
plt.figure(figsize=(12, 12))

sns.heatmap(corrMatrix, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='viridis',linecolor="white")
plt.title('Correlation between features');


# In[ ]:


testtest = test_dt.iloc[:,:-1]
testtest


# In[ ]:


train_dt.columns


# In[ ]:


get_univariate_plots(data=train_dt, target_col='Total Yearly Income [EUR]', 
                     features_list=['Crime Level in the City of Employement'], bins=10)
get_univariate_plots(data=train_dt, target_col='Total Yearly Income [EUR]', 
                     features_list=['Hc_le'], bins=10)
get_univariate_plots(data=train_dt, target_col='Total Yearly Income [EUR]', 
                     features_list=['uni_le'], bins=10)
get_univariate_plots(data=train_dt, target_col='Total Yearly Income [EUR]', 
                     features_list=['Year of Record'], bins=10)
get_univariate_plots(data=train_dt, target_col='Total Yearly Income [EUR]', 
                     features_list=['Wears Glasses'], bins=10)
get_univariate_plots(data=train_dt, target_col='Total Yearly Income [EUR]', 
                     features_list=['satisfaction_le'], bins=10)
get_univariate_plots(data=train_dt, target_col='Total Yearly Income [EUR]', 
                     features_list=['gender_le'], bins=10)
get_univariate_plots(data=train_dt, target_col='Total Yearly Income [EUR]', 
                     features_list=['Body Height [cm]'], bins=10)


# In[ ]:


num_train_df = train_dt.iloc[0:1048574,:].drop(['Housing Situation', 'Work Experience in Current Job [years]', 'Satisfation with employer', 'Gender', 'Country', 'Profession', 'University Degree', 'Hair Color', 'Yearly Income in addition to Salary (e.g. Rental Income)'],axis=1)
num_test_df = train_dt.iloc[1048574:,:].drop(['Housing Situation', 'Work Experience in Current Job [years]', 'Satisfation with employer', 'Gender', 'Country', 'Profession', 'University Degree', 'Hair Color', 'Yearly Income in addition to Salary (e.g. Rental Income)'],axis=1)
X_train, X_val, y_train, y_val = train_test_split(num_train_df, Y, test_size=0.35, random_state=42)
get_univariate_plots(data=X_train, target_col='Total Yearly Income [EUR]', data_test=X_val, features_list=['Hc_le'])
# (1048574, 17)
# (369438, 17)


# # Trend Correlation
# 
# Shows if features are noisy and inconsistent throughout data

# In[ ]:


from featexp import get_trend_stats
stats = get_trend_stats(data=X_train, target_col='Total Yearly Income [EUR]', data_test=X_val)
stats


# In[ ]:


stats = get_trend_stats(data=num_train_df, target_col='additionalInc', data_test=num_test_df)
stats


# In[ ]:


get_univariate_plots(data=num_train_df, target_col='additionalInc', data_test=num_test_df, features_list=['satisfaction_le'])

