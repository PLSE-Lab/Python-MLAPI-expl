#!/usr/bin/env python
# coding: utf-8

# Imports of libraries we'll be using for a start

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

import os


# **Read The Data**
# 
# Reading the CSV files into Pandas DataFrame

# In[ ]:


#First, taking a look at the list files that is available available
print(os.listdir("../input/"))


# In[ ]:


application_train_df = pd.read_csv('../input/application_train.csv')


# In[ ]:


application_test_df = pd.read_csv('../input/application_test.csv')


# Taking a look at what our application and test dataframes look like, including a quick glance at their descriptive statistics.

# In[ ]:


application_train_df.head()


# In[ ]:


application_test_df.head()


# In[ ]:


application_train_df.shape


# The application_train dataframe has 122 columns (121 features plus the target) and 307511 rows or observations.

# In[ ]:


application_test_df.shape


# While the application_test dataframe has 121 columns (features only) and 48744 rows or observations.

# In[ ]:


#A look at their descriptive statistics
application_train_df.describe()


# In[ ]:


application_test_df.describe()


# In[ ]:


application_train_df.info()


# In[ ]:


#Showing null details (missing values) of the application_train_df
application_train_df.isnull().sum()


# We can quickly see the columns with Null values in the dataframe and this will be helpful when we get to the cleansing aspect of the project in preparing the data for the Machine Model.

# **Exploratory Data Analysis(EDA)**
# 
# Having taking a glimpse at what our application dataframe looks like, now lets dive a little bit into exploring the data to uncover as many insights and trends that we can gained through visualization.

# Showing a countplot of the target column

# In[ ]:


plt.figure(figsize = (10,5))
sns.countplot(x='TARGET', data = application_train_df)


# The visual above shows that there are way more customers that are not having challenges meeting up with their loan obligations. However, it is still an issue to have that much clients struggling with their loans.

# In[ ]:


plt.figure(figsize = (10,5))
sns.countplot(x='NAME_CONTRACT_TYPE', data = application_train_df, hue='TARGET')


# Much of the loans are of the Cash category.

# In[ ]:


plt.figure(figsize = (10,5))
sns.countplot(x='CODE_GENDER', data = application_train_df, hue='TARGET')


# There are much more females than males based on the above. 

# In[ ]:


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.countplot(x='FLAG_OWN_CAR', data = application_train_df, order=('Y','N'),hue='TARGET')
plt.subplot(1,2,2)
sns.countplot(x='FLAG_OWN_REALTY', data = application_train_df, hue='TARGET')
plt.subplots_adjust(wspace = 0.8)


# The clients own more realty than they do own cars. 

# In[ ]:


ax = sns.factorplot(x="NAME_INCOME_TYPE", hue="NAME_CONTRACT_TYPE", col="TARGET", data=application_train_df, kind="count",size=4, aspect=1.2)
ax.set_xticklabels(rotation=90)


# In[ ]:


ax = sns.factorplot(hue="NAME_INCOME_TYPE", x="CODE_GENDER", 
                    col="TARGET",data=application_train_df, 
                    kind="count",size=4, aspect=1.2)


# In[ ]:


ax = sns.factorplot(x='OCCUPATION_TYPE', data = application_train_df,
                    hue = 'TARGET', kind = 'count', size=4, aspect=1.8)
ax.set_xticklabels(rotation=90)


# In[ ]:


plt.figure(figsize = (10,5))
sns.countplot(x='NAME_TYPE_SUITE', data = application_train_df)


# In[ ]:


sns.factorplot(x="AMT_INCOME_TOTAL", y="NAME_INCOME_TYPE", 
               hue="CODE_GENDER", data=application_train_df, size=4, 
                aspect=2.5, kind="bar", ci=False)


# How the total income earned varies by the income type categories. As perhaps would be expected, the "Businessman" category has the highest income. Interestingly, the "Maternity leave" category appears to provide a decent payday as well.

# In[ ]:


sns.factorplot(x="AMT_INCOME_TOTAL", y="OCCUPATION_TYPE", 
               data=application_train_df[application_train_df.OCCUPATION_TYPE.notnull()],
               size=4, aspect=2.5, kind="bar", ci=False)


# And finally, a glimpse of how the various occupation types compare regarding income total amount. Not a lot of variation, however, Managers earns the most.

# In[ ]:


plt.figure(figsize = (12,5))
sns.countplot(x='NAME_EDUCATION_TYPE', data = application_train_df)


# In[ ]:


plt.figure(figsize = (12,5))
sns.countplot(x='NAME_HOUSING_TYPE', data = application_train_df)


# In[ ]:


ax = sns.factorplot(x='ORGANIZATION_TYPE', data = application_train_df, kind = 'count', size=8, aspect=1.8)
ax.set_xticklabels(rotation=90)


# There seems to be a decent amount of business-minded individuals among the customer base of HomeCredit. 

# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(application_train_df['DAYS_BIRTH'], bins = 50, color = 'red')
plt.title('Showing the Age Distribution')


# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
sns.distplot(application_train_df['AMT_CREDIT'], bins = 30, kde = False, color = 'red')
plt.title('Showing the Credit Amount Distribution')


# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
sns.distplot(application_train_df['AMT_ANNUITY'].dropna(), bins = 30, color = 'red')
plt.title('Showing the Annuity Amount Distribution')


# In[ ]:


sns.set_style('whitegrid')
plt.figure(figsize=(10,5))
sns.distplot(application_train_df['AMT_GOODS_PRICE'].dropna(), bins = 30, color = 'red')
plt.title('Showing the Goods Price Amount Distribution')


# In[ ]:


sns.set_style('whitegrid')
sns.jointplot('AMT_CREDIT','AMT_GOODS_PRICE', data=application_train_df, size=8)


# In[ ]:


sns.lmplot(x='AMT_CREDIT', y='AMT_GOODS_PRICE', data=application_train_df, 
           hue='TARGET', size =7 , col='NAME_CONTRACT_TYPE')


# In[ ]:


sns.lmplot(x='AMT_CREDIT', y='AMT_GOODS_PRICE', data=application_train_df,
           hue='NAME_CONTRACT_TYPE', size = 7, col='TARGET')


# **Data Cleansing**
# 
# Now lets do a bit of cleaning the data having done some visuals to understand the dataset much better.
# 
# The heatmap below, shows the profile of the dataframe with missing values highlighted in yellow.

# In[ ]:


#A quick visual of the entire dataframe showing missing values. It can be seen that much of the apartment and 
# own car age data are missing.
plt.figure(figsize = (10,5))
sns.heatmap(application_train_df.isnull(), yticklabels=False, cbar = False, cmap = 'viridis')


# From the descriptive statisitics shown earlier, we saw that the maximum value for "DAYS_EMPLOYED", is not only positive which is an anomally of its own given that the column is supposed to be negative all through, but it is also too large to represent a reasonable figure. As a result, it is an anomally and would be treated as such. 
# First, let's take a look at the distribution of this column.
# 
# **Distribution of DAYS_EMPLOYED**

# In[ ]:


application_train_df['DAYS_EMPLOYED'].plot.hist()
plt.xlabel('DAYS_EMPLOYED')


# In[ ]:


anomalous_days_employed_train = application_train_df[application_train_df['DAYS_EMPLOYED'] == 365243]
print(len(anomalous_days_employed_train))


# There are 55374 outliers in the DAYS_EMPLOYED column in the application train dataset which is pretty significant. The concluded strategy to handle this is to temporarily replace these outliers with NaN values and subsequently treating them as such when dealing with the other NaN (missing) values in the DataFrame. Also, this will be carried out while cleaning the application_test dataframe as well.

# In[ ]:


#Replacing the outliers with NaNs
application_train_df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)


# Another chart of the **Distribution of DAYS_EMPLOYED** now looks like this. Which makes more sense than previously.

# In[ ]:


application_train_df['DAYS_EMPLOYED'].plot.hist()
plt.xlabel('DAYS_EMPLOYED')


# Time to carry out some basic cleansing operation with the perculiarities of the columns in mind starting with the categorical columns and then the numeric ones.

# In[ ]:


def impute_suite(NAME_TYPE_SUITE):
    if pd.isnull(NAME_TYPE_SUITE):
        return 'Unaccompanied'
    else:
        return NAME_TYPE_SUITE


# In[ ]:


application_train_df['NAME_TYPE_SUITE'] = application_train_df['NAME_TYPE_SUITE'].apply(impute_suite)


# In[ ]:


def impute_occupation(cols):
    OCCUPATION_TYPE=cols[0]
    AMT_INCOME_TOTAL=cols[1]
    if pd.isnull(OCCUPATION_TYPE):
        if AMT_INCOME_TOTAL > 150000.000:
            return 'Laborers'
        elif AMT_INCOME_TOTAL < 150000.000:
            return 'Sales staff'
    else:
        return OCCUPATION_TYPE


# In[ ]:


application_train_df['OCCUPATION_TYPE'] = application_train_df[['OCCUPATION_TYPE', 'AMT_INCOME_TOTAL']].apply(impute_occupation, axis=1)


# In[ ]:


def impute_fond(FONDKAPREMONT_MODE):
    if pd.isnull(FONDKAPREMONT_MODE):
        return 'reg oper account'
    else:
        return FONDKAPREMONT_MODE


# In[ ]:


application_train_df['FONDKAPREMONT_MODE'] = application_train_df['FONDKAPREMONT_MODE'].apply(impute_fond)


# In[ ]:


def impute_housetype(HOUSETYPE_MODE):
    if pd.isnull(HOUSETYPE_MODE):
        return 'block of flats'
    else:
        return HOUSETYPE_MODE


# In[ ]:


application_train_df['HOUSETYPE_MODE'] = application_train_df['HOUSETYPE_MODE'].apply(impute_housetype)


# In[ ]:


application_train_df['WALLSMATERIAL_MODE'] = application_train_df['WALLSMATERIAL_MODE'].fillna(application_train_df['WALLSMATERIAL_MODE'].value_counts().index[0])


# In[ ]:


application_train_df['EMERGENCYSTATE_MODE'] = application_train_df['EMERGENCYSTATE_MODE'].fillna(application_train_df['EMERGENCYSTATE_MODE'].value_counts().index[0])


# The concludes the categorical columns. Now to fill the numerical ones with column medians.

# In[ ]:


#Imputing NaN values in numeric columns with column median. This action completes the clean up.
application_train_df = application_train_df.fillna(application_train_df.median())


# **The Cleaned Application_Train DataFrame**

# In[ ]:


#Cleaned dataframe.
plt.figure(figsize = (10,5))
sns.heatmap(application_train_df.isnull(), yticklabels=False, cbar = False, cmap = 'viridis')


# **Features Correlation**
# 
# Let's take a look to see if there is any meaningful correlation between the features. 

# In[ ]:


corr = application_train_df.corr

plt.figure(figsize = (12,10))
sns.heatmap(corr(), annot = True)


# Well, not very important to us as it is right now. Too many features to be decipherable. 

# **Categorical Features Dummification**
# 
# Having cleaned up our dataframe, we now move on to create dummy variables for our categorical columns.  In doing so, we drop the first columns of each of the transformed dummy variable columns to avoid collinearity problems in our model.

# In[ ]:


application_train_df = pd.get_dummies(application_train_df, prefix_sep='_', drop_first=True)


# In[ ]:


print('Below is our newly transformed application_train_df now ready to be trained')
print('Application_train_df shape: ', application_train_df.shape)
application_train_df.head()


# **Application_Test DataFrame Cleanup**

# In[ ]:


#Showing null details (missing values) of the application_test_df
application_test_df.isnull().sum()


# Applying the same treatment to the "DAYS_EMPLOYED" column of the Application_test_df as that earlier applied to the Application_train_df. But first of, let's look at the distribution. 

# In[ ]:


application_test_df['DAYS_EMPLOYED'].plot.hist()
plt.xlabel('DAYS_EMPLOYED')


# In[ ]:


anomalous_days_employed_test = application_test_df[application_test_df['DAYS_EMPLOYED'] == 365243]
print(len(anomalous_days_employed_test))


# There are 9274 outliers in the "DAYS_EMPLOYED" column of the application_test_df out of 48744 which represents roughly 19% of the observations. This is comparable to that of the application_train_df which is roughly 18%.

# In[ ]:


#Replacing the outliers with NaNs
application_test_df['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)


# In[ ]:


application_test_df['DAYS_EMPLOYED'].plot.hist()
plt.xlabel('DAYS_EMPLOYED')


# In[ ]:


def impute_suite(NAME_TYPE_SUITE):
    if pd.isnull(NAME_TYPE_SUITE):
        return 'Unaccompanied'
    else:
        return NAME_TYPE_SUITE


# In[ ]:


application_test_df['NAME_TYPE_SUITE'] = application_test_df['NAME_TYPE_SUITE'].apply(impute_suite)


# In[ ]:


def impute_occupation(cols):
    OCCUPATION_TYPE=cols[0]
    AMT_INCOME_TOTAL=cols[1]
    if pd.isnull(OCCUPATION_TYPE):
        if AMT_INCOME_TOTAL > 150000.000:
            return 'Laborers'
        elif AMT_INCOME_TOTAL < 150000.000:
            return 'Sales staff'
    else:
        return OCCUPATION_TYPE


# In[ ]:


application_test_df['OCCUPATION_TYPE'] = application_test_df[['OCCUPATION_TYPE', 'AMT_INCOME_TOTAL']].apply(impute_occupation, axis=1)


# In[ ]:


def impute_fond(FONDKAPREMONT_MODE):
    if pd.isnull(FONDKAPREMONT_MODE):
        return 'reg oper account'
    else:
        return FONDKAPREMONT_MODE


# In[ ]:


application_test_df['FONDKAPREMONT_MODE'] = application_test_df['FONDKAPREMONT_MODE'].apply(impute_fond)


# In[ ]:


def impute_housetype(HOUSETYPE_MODE):
    if pd.isnull(HOUSETYPE_MODE):
        return 'block of flats'
    else:
        return HOUSETYPE_MODE


# In[ ]:


application_test_df['HOUSETYPE_MODE'] = application_test_df['HOUSETYPE_MODE'].apply(impute_housetype)


# In[ ]:


application_test_df['WALLSMATERIAL_MODE'] = application_test_df['WALLSMATERIAL_MODE'].fillna(application_test_df['WALLSMATERIAL_MODE'].value_counts().index[0])


# In[ ]:


application_test_df['EMERGENCYSTATE_MODE'] = application_test_df['EMERGENCYSTATE_MODE'].fillna(application_test_df['EMERGENCYSTATE_MODE'].value_counts().index[0])


# In[ ]:


application_test_df = application_test_df.fillna(application_test_df.median())


# **The Cleaned Application_Test DataFrame**

# In[ ]:


plt.figure(figsize = (10,5))
sns.heatmap(application_test_df.isnull(), yticklabels=False, cbar = False, cmap = 'viridis')


# In[ ]:


application_test_df = pd.get_dummies(application_test_df, prefix_sep='_', drop_first=True)


# In[ ]:


print('Below is our newly transformed application_test_df')
print('Application_test_df shape: ', application_test_df.shape)
application_test_df.head()


# **Aligning The Two DataFrames**
# 
# While the application test dataframe does not contain the "TARGET" column, it is however 3 columns less that the application train dataframe. So we have to align the two dataframes to be able to use them in our model. To accomplish that, we first drop the "TARGET" column in the application_train_df , align and then add it back.

# In[ ]:


print('Training Features shape: ', application_train_df.shape)
print('Testing Features shape: ', application_test_df.shape)


# In[ ]:


#Saving the Target column to re-add it again later
application_train_df_TARGET = application_train_df['TARGET']


# In[ ]:


application_train_df.drop('TARGET', axis=1, inplace = True)


# In[ ]:


application_train_df, application_test_df = application_train_df.align(application_test_df, join = 'inner', axis = 1)


# In[ ]:


#ALigned dataframes
print('Training Features shape: ', application_train_df.shape)
print('Testing Features shape: ', application_test_df.shape)


# Train and Test aligned. Now add back the 'Target'.

# In[ ]:


application_train_df = pd.concat([application_train_df, application_train_df_TARGET], axis=1)


# In[ ]:


application_train_df.head()


# In[ ]:


print('Training Features shape: ', application_train_df.shape)
print('Testing Features shape: ', application_test_df.shape)


# **Baseline Model**
# 
# Training our model with the cleaned application_test and application_train dataframes. We now load the outputted dataframes respectively.

# Next is to separate our features (X) from the label/Target(y) into two separate dataframes

# In[ ]:


X = application_train_df.drop('TARGET', axis = 1)
y = application_train_df['TARGET']

X_test = application_test_df #This will be called to test our trained and validated model.


# We now split our X and y variables into training and validation sets.

# In[ ]:


from sklearn.model_selection import train_test_split 


# In[ ]:


x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=1001)


# **LightGBM Framework**
# 
# We will be using the LightGBM framework for our classification. This was chosen for it's high performance in terms speed, memory usage as our data size is pretty large, as well as its focus on accuracy.

# In[ ]:


import lightgbm as lgbm


# In[ ]:


#Creating our Training and Validation Sets respectively
lgbm_train = lgbm.Dataset(data=x_train, label=y_train)
lgbm_valid = lgbm.Dataset(data=x_valid, label=y_valid)


# In[ ]:


#We now define our parameters
params = {}
params['task'] = 'train'
params['objective'] = 'binary'
params['boosting_type'] = 'gbdt'
params['num_iteration'] = 10000
params['learning_rate'] = 0.003
params['metric'] = 'auc'
params['num_leaves'] = 80
params['min_data_in_leaf'] = 100
params['max_depth'] = 8
params['min_child_weight'] = 80
params['reg_alpha'] = 0.05
params['reg_lambda'] = 0.08
params['min_split_gain'] = 0.03
params['sub_sample'] = 0.9
params['colsample_bytree'] = 0.95


# **Model Training**

# In[ ]:


lgbm_model = lgbm.train(params, lgbm_train, valid_sets=lgbm_valid, early_stopping_rounds=200, verbose_eval=200)


# **Features Importance**
# 
# Let's see a chart of the ranking of the features according to their importance to predicting the target.

# In[ ]:


#Limiting the number of features to displayed to 120
lgbm.plot_importance(lgbm_model, figsize=(12, 25), max_num_features=120)


# External sources 3 and 2 as well as customer's age are the top three most important features in predicting whether a customer would default or not. While we do not know the exact sources of the External sources data, it maybe inferred that it could probably be related to customer's credit scores which are provided by external agencies and are vital determinant of credit worthiness.

# **Predictions**

# In[ ]:


proba_predictions = lgbm_model.predict(X_test)
submit_lgbm = pd.DataFrame()
submit_lgbm['SK_ID_CURR'] = application_test_df['SK_ID_CURR']
submit_lgbm['TARGET'] = proba_predictions
submit_lgbm.to_csv("lgbm_baseline_model.csv", index=False)
submit_lgbm.head(20)


# In[ ]:




