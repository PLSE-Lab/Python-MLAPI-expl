#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# to handle datasets
import pandas as pd
import numpy as np
import scipy as sp
from scipy import stats
# for text / string processing
import re
 
# for plotting
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
 
# to divide train and test set
from sklearn.model_selection import train_test_split
 
# feature scaling
from sklearn.preprocessing import MinMaxScaler
 
# for tree binarisation
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
 
 
# to build the models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
 
# to evaluate the models
from sklearn.metrics import roc_auc_score
from sklearn import metrics
 
pd.pandas.set_option('display.max_columns', None)
 
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train=pd.read_csv('../input/titanic/train.csv')
test=pd.read_csv('../input/titanic/test.csv')
submission=pd.read_csv('../input/titanic/gender_submission.csv')


# **Util Function**

# In[ ]:


def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# **Summary of data**

# In[ ]:


resumetable(train)


# **There are a mixture of categorical and numerical variables. Numerical are those of type int and float. Categorical those of type object.**

# In[ ]:


# find categorical variables
categorical = [var for var in train.columns if train[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))

# find numerical variables
numerical = [var for var in train.columns if train[var].dtype!='O']
print('There are {} numerical variables'.format(len(numerical)))


# In[ ]:


# view of categorical variables
train[categorical].head()


# **Cabin and Ticket contain both numbers and letters. We could extract the numerical part and then the non-numerical part and generate 2 variables out of them, to see if that adds value to our predictive models.**

# In[ ]:


# view of numerical variables
train[numerical].head()


# 3 Discrete variables: Pclass, SibSp and Parch
# 
# 2 continuous variables: Fare and Age
# 
# 1 Id variable: PassengerId (it is a label for each of the passengers)
# 
# 1 binary: Survived (target variable).

# In[ ]:


# let's visualise the values of the discrete variables
for var in ['Pclass',  'SibSp', 'Parch']:
    print(var, ' values: ', train[var].unique())


# **Types of variables, summary:**
# 
# 5 categorical variables: from them 2 could be treated as mixed type of variables (numbers and strings)
# 
# 7 numerical variables: 3 discrete, 2 continuous, 1 Id, and 1 binary target

# ## Types of problems within the variables

# **Missing values**

# In[ ]:


# let's visualise the percentage of missing values
train.isnull().mean()


# **Three of the variables contain missing data,**
# 
# Age (~20%),
# 
# Cabin (~77%)
# 
#  Embarked (< 1%)

# ## Outliers

# In[ ]:


numerical = [var for var in numerical if var not in['Survived', 'PassengerId']]
numerical


# In[ ]:


# let's make boxplots to visualise outliers in the continuous variables 
# Age and Fare
 
plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = train.boxplot(column='Age')
fig.set_title('')
fig.set_ylabel('Age')
 
plt.subplot(1, 2, 2)
fig = train.boxplot(column='Fare')
fig.set_title('')
fig.set_ylabel('Fare')


# Both Age and Fare contain outliers. Let's find which valuers are the outliers

# In[ ]:


# first we plot the distributions to find out if they are Gaussian or skewed.
# Depending on the distribution, we will use the normal assumption or the interquantile
# range to find outliers
 
plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = train.Age.hist(bins=20)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Age')
 
plt.subplot(1, 2, 2)
fig = train.Fare.hist(bins=20)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Fare')


# Age is quite Gaussian and Fare is skewed, so I will use the Gaussian assumption for Age, and the interquantile range for Fare

# In[ ]:


# find outliers
 
# Age
Upper_boundary = train.Age.mean() + 3* train.Age.std()
Lower_boundary = train.Age.mean() - 3* train.Age.std()
print('Age outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_boundary, upperboundary=Upper_boundary))
 
# Fare
IQR = train.Fare.quantile(0.75) - train.Fare.quantile(0.25)
Lower_fence = train.Fare.quantile(0.25) - (IQR * 3)
Upper_fence = train.Fare.quantile(0.75) + (IQR * 3)
print('Fare outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))


# We should remove from the dataset Ages > 73
# 
# There are a few methods to handle outliers, one is top-coding, the other one is discretisation of variables.** I will use top-coding for Age,**

# **Outlies in discrete variables**

# Let's calculate the percentage of passengers for each of the values that can take the discrete variables in the titanic dataset. I will call outliers, those values that are present in less than 1% of the passengers. This is exactly the same as finding rare labels in categorical variables. Discrete variables, in essence can be pre-processed / engineered as if they were categorical. Keep this in mind.
# 

# In[ ]:


#  outlies in discrete variables
for var in ['Pclass',  'SibSp', 'Parch']:
    print(train[var].value_counts() / np.float(len(train)))
    print()


# **Pclass does not contain outliers,** as all its numbers are present in at least 20% of the passengers.
# 
# SibSp This variable indicates the number of of siblings / spouses aboard the Titanic. Values bigger than 4, are rare. So I will cap this variable at 4 **(top coding).**
# 
# Parch This variable indicates the number of parents / children aboard the Titanic. We can see that values > 2 are rare (present in less than 1% of passengers). Thus I will cap this variable at 2 (top-coding).

# **Number of labels: cardinality**

# In[ ]:


for var in categorical:
    print(var, ' contains ', len(train[var].unique()), ' labels')


# The variables Name, Ticket and Cabin are highly cardinal, i.e., they contain a lot of labels. In addition, those variables are not usable as such, and they require some manual preprocessing. I will do that before proceeding with the data exploration

# # Pre-processing of mixed type of variables 

# The variables Cabin and Ticket contain both numbers and letters. Let's create 2 variables for each extracting the numerical and categorical part.

# In[ ]:


# Cabin
train['Cabin_numerical'] = train.Cabin.str.extract('(\d+)') # extracts number from string
train['Cabin_numerical'] = train['Cabin_numerical'].astype('float') # parses the above variable to float type
 
train['Cabin_categorical'] = train['Cabin'].str[0] # captures first letter of string (the letter of the cabin)
 
# same for submission data
test['Cabin_numerical'] = test.Cabin.str.extract('(\d+)')
test['Cabin_numerical'] = test['Cabin_numerical'].astype('float')
 
test['Cabin_categorical'] = test['Cabin'].str[0]
 
train[['Cabin', 'Cabin_numerical', 'Cabin_categorical']].head()


# In[ ]:



# drop the original variable
train.drop(labels='Cabin', inplace=True, axis=1)
test.drop(labels='Cabin', inplace=True, axis=1)


# In[ ]:


#  Ticket
# extract the last bit of ticket as number
train['Ticket_numerical'] = train.Ticket.apply(lambda s: s.split()[-1])
train['Ticket_numerical'] = np.where(train.Ticket_numerical.str.isdigit(), train.Ticket_numerical, np.nan)
train['Ticket_numerical'] = train['Ticket_numerical'].astype('float')
 
# extract the first part of ticket as category
train['Ticket_categorical'] = train.Ticket.apply(lambda s: s.split()[0])
train['Ticket_categorical'] = np.where(train.Ticket_categorical.str.isdigit(), np.nan, train.Ticket_categorical)
 
# submission
test['Ticket_numerical'] = test.Ticket.apply(lambda s: s.split()[-1])
test['Ticket_numerical'] = np.where(test.Ticket_numerical.str.isdigit(), test.Ticket_numerical, np.nan)
test['Ticket_numerical'] = test['Ticket_numerical'].astype('float')
 
# extract the first part of ticket as category
test['Ticket_categorical'] = test.Ticket.apply(lambda s: s.split()[0])
test['Ticket_categorical'] = np.where(test.Ticket_categorical.str.isdigit(), np.nan, test.Ticket_categorical)
 
train[['Ticket', 'Ticket_numerical', 'Ticket_categorical']].head()


# In[ ]:


# let's explore the ticket categorical part a bit further
train.Ticket_categorical.unique()


# In[ ]:


# it contains several labels, some of them seem very similar apart from the punctuation
# I will try to reduce this number of labels a bit further
 
# remove non letter characters from string
text = train.Ticket_categorical.apply(lambda x: re.sub("[^a-zA-Z]", '', str(x)))
 
# to visualise the output and compare with input
pd.concat([text, train.Ticket_categorical], axis=1)


# In[ ]:


# set to upper case: we reduce the number of labels quite a bit
text = text.str.upper()
text.unique()


# In[ ]:


# process the variable in submission as well
train['Ticket_categorical'] = text
 
test['Ticket_categorical'] = test.Ticket_categorical.apply(lambda x: re.sub("[^a-zA-Z]", '', str(x)))
test['Ticket_categorical'] = test['Ticket_categorical'].str.upper()


# In[ ]:


# drop the original variable
train.drop(labels='Ticket', inplace=True, axis=1)
test.drop(labels='Ticket', inplace=True, axis=1)


# ## tailored preprocessing for the Titanic dataset

# **The variable Name contains 891 different values, one for each of the passengers. We wouldn't be able to use this variable as is. However, we can extract some data from it, for example the title. See below.**

# In[ ]:


def get_title(passenger):
    # extracts the title from the name variable
    line = passenger
    if re.search('Mrs', line):
        return 'Mrs'
    elif re.search('Mr', line):
        return 'Mr'
    elif re.search('Miss', line):
        return 'Miss'
    elif re.search('Master', line):
        return 'Master'
    else:
        return 'Other'
    
train['Title'] = train['Name'].apply(get_title)
test['Title'] = test['Name'].apply(get_title)
 
test[['Name', 'Title']].head()


# In[ ]:


# drop the original variable
train.drop(labels='Name', inplace=True, axis=1)
test.drop(labels='Name', inplace=True, axis=1)


# In[ ]:


# create a variable indicating family size (including the passenger)
# sums siblings and parents
 
train['Family_size'] = train['SibSp']+train['Parch']+1
test['Family_size'] = test['SibSp']+test['Parch']+1
 
print(train.Family_size.value_counts()/ np.float(len(train)))
 
(train.Family_size.value_counts() / np.float(len(train))).plot.bar()


# The new variable Family size is discrete, because it is the sum of 2 discrete variables. It takes a finite number of values, and large families were rare on the Titanic. In fact, families larger than 7 people were rare, so ** I will cap family size at 7.**

# In[ ]:


# variable indicating if passenger was a mother
train['is_mother'] = np.where((train.Sex =='female')&(train.Parch>=1)&(train.Age>18),1,0)
test['is_mother'] = np.where((test.Sex =='female')&(test.Parch>=1)&(test.Age>18),1,0)
 
train[['Sex', 'Parch', 'Age', 'is_mother']].head()


# In[ ]:


train.loc[train.is_mother==1, ['Sex', 'Parch', 'Age', 'is_mother']].head()


# In[ ]:


print('there were {} mothers in the Titanic'.format(train.is_mother.sum()))


# ### Types of problems within variables II

# Let's look for missing data, outliers, cardinality and rare labels in the newly created variables.

# **New numerical variables: Missing values**

# In[ ]:


train[['Cabin_numerical', 'Ticket_numerical', 'is_mother', 'Family_size']].isnull().mean()


# Cabin_numerical, as expected contains the same amount of missing data than the original variable Cabin.
# 
# Ticket, also contains a small percentage of missing values. The other newly created variables do not contain missing data, as expected.

# **New numerical variables: Outliers**

# In[ ]:


# first we plot the distributions to find out if they are Gaussian or skewed.
# Depending on the distribution, we will use the normal assumption or the interquantile
# range to find outliers
 
plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = train.Cabin_numerical.hist(bins=50)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Cabin number')
 
plt.subplot(1, 2, 2)
fig = train.Ticket_numerical.hist(bins=50)
fig.set_ylabel('Number of passengers')
fig.set_xlabel('Ticket number')


# In[ ]:


# let's visualise outliers with the boxplot and whiskers
plt.figure(figsize=(15,6))
plt.subplot(1, 2, 1)
fig = train.boxplot(column='Cabin_numerical')
fig.set_title('')
fig.set_ylabel('Cabin number')
 
plt.subplot(1, 2, 2)
fig = train.boxplot(column='Ticket_numerical')
fig.set_title('')
fig.set_ylabel('Ticket number')


# Cabin_numerical does not contain outliers. Ticket_numerical seems to contain a few outliers. Let's find out more about it.

# In[ ]:


# Ticket numerical
IQR = train.Ticket_numerical.quantile(0.75) - train.Ticket_numerical.quantile(0.25)
Lower_fence = train.Ticket_numerical.quantile(0.25) - (IQR * 3)
Upper_fence = train.Ticket_numerical.quantile(0.75) + (IQR * 3)
print('Ticket number outliers are values < {lowerboundary} or > {upperboundary}'.format(lowerboundary=Lower_fence, upperboundary=Upper_fence))
passengers = len(train[train.Ticket_numerical>Upper_fence]) / np.float(len(train))
print('Number of passengers with ticket values higher than {upperboundary}: {passengers}'.format(upperboundary=Upper_fence,                                                                                                  passengers=passengers))


#  will use equal width discretisation for this variable.

# **New categorical variables: Missing values**

# In[ ]:


train[['Cabin_categorical', 'Ticket_categorical', 'Title']].isnull().mean()


# As expected, Cabin contains the same amount of missing data as the original Cabin variable.
# 
# The other 2 variables do not show missing data.

# **New categorical variables: cardinality**

# In[ ]:


for var in ['Cabin_categorical', 'Ticket_categorical', 'Title']:
    print(var, ' contains ', len(train[var].unique()), ' labels')


# Title and Cabin are not highly cardinal, Ticket on the other hand has quite a few labels. Let's explore the percentage of passengers within each label to identify rare labels.

# **New categorical variables: rare labels**

# In[ ]:


# rare / infrequent labels (less than 1% of passengers)
for var in ['Cabin_categorical', 'Ticket_categorical', 'Title']:
    print(train[var].value_counts() / np.float(len(train)))
    print()


# Cabin contains the rare labels G and T: replace by most frequent category
# 
# Ticket contains a lot of infrequent labels: replace by rare
# 
# Title does not contain rare labels
# 
# Because the number of passengers in the rare cabins is so small, grouping them into a new category called rare, will be in itself rare, and may be prone to overfitting. This, in cabin, I will replace rare labels by the most frequent category.
# 
# In ticket_categorical, on the other hand, the number of infrequent labels is high, therefore grouping them into a new label makes sense.

# **Separate train and test set**

# In[ ]:


# Let's separate into train and test set
 
X_train, X_test, y_train, y_test = train_test_split(train, train.Survived, test_size=0.2,
                                                    random_state=0)
X_train.shape, X_test.shape


# In[ ]:


# let's group again the variables into categorical or numerical
# now considering the newly created variables
 
def find_categorical_and_numerical_variables(dataframe):
    cat_vars = [col for col in train.columns if train[col].dtypes == 'O']
    num_vars  = [col for col in train.columns if train[col].dtypes != 'O']
    return cat_vars, num_vars
                 
categorical, numerical = find_categorical_and_numerical_variables(train) 


# In[ ]:


categorical


# In[ ]:


numerical = [var for var in numerical if var not in ['Survived','PassengerId']]
numerical


# ## Engineering missing values in numerical variables

# In[ ]:


# print variables with missing data
for col in numerical:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())


# Age and ticket contains < 50% NA: create additional variable with NA + random sample imputation
# Cabin_numerical contains > 50% NA: impute NA by value far in the distribution

# In[ ]:


def impute_na(X_train, df, variable):
    # make temporary df copy
    temp = df.copy()
    
    # extract random from train set to fill the na
    random_sample = X_train[variable].dropna().sample(temp[variable].isnull().sum(), random_state=0)
    
    # pandas needs to have the same index in order to merge datasets
    random_sample.index = temp[temp[variable].isnull()].index
    temp.loc[temp[variable].isnull(), variable] = random_sample
    return temp[variable]


# In[ ]:


# Age and ticket
# add variable indicating missingness
for df in [X_train, X_test, test]:
    for var in ['Age', 'Ticket_numerical']:
        df[var+'_NA'] = np.where(df[var].isnull(), 1, 0)
    
# replace by random sampling
for df in [X_train, X_test, test]:
    for var in ['Age', 'Ticket_numerical']:
        df[var] = impute_na(X_train, df, var)
    
 
# Cabin numerical
extreme = X_train.Cabin_numerical.mean() + X_train.Cabin_numerical.std()*3
for df in [X_train, X_test, test]:
    df.Cabin_numerical.fillna(extreme, inplace=True)


# ### Engineering Missing Data in categorical variables

# In[ ]:


# print variables with missing data
for col in categorical:
    if X_train[col].isnull().mean()>0:
        print(col, X_train[col].isnull().mean())


# Embarked NA imputed by most frequent category, because NA is low
# 
# Cabin_categorical imputed by 'Missing', because NA is high

# In[ ]:


# add label indicating 'Missing' to Cabin categorical
# or replace by most frequent label in Embarked
 
for df in [X_train, X_test, test]:
    df['Embarked'].fillna(X_train['Embarked'].mode()[0], inplace=True)
    df['Cabin_categorical'].fillna('Missing', inplace=True)


# In[ ]:


# check absence of null values
X_train.isnull().sum()


# In[ ]:


X_test.isnull().sum()


# In[ ]:


# Fare in the test dataset contains one null value, I will replace it by the median 
test.Fare.fillna(X_train.Fare.median(), inplace=True)


# **Outliers in Numerical variables**

# As I was analysing the outliers at the beginning of the notebook, I was taking a note on the preprocessing that I thought would be more convenient for each one of them. The notes are summarised here:
# 
# Age: top-coding (73)
# 
# Fare: equal frequency binning
# 
# Sibsp: top-coding (4)
# 
# Parch: top-coding (2)
# 
# Family Size: top-coding (7)
# 
# Ticket_number: equal frequency binning

# In[ ]:


def top_code(df, variable, top):
    return np.where(df[variable]>top, top, df[variable])
 
for df in [X_train, X_test, test]:
    df['Age'] = top_code(df, 'Age', 73)
    df['SibSp'] = top_code(df, 'SibSp', 4)
    df['Parch'] = top_code(df, 'Parch', 2)
    df['Family_size'] = top_code(df, 'Family_size', 7)


# In[ ]:


# let's check that it worked
for var in ['Age',  'SibSp', 'Parch', 'Family_size']:
    print(var, ' max value: ', X_train[var].max())


# In[ ]:


# let's check that it worked
for var in ['Age',  'SibSp', 'Parch', 'Family_size']:
    print(var, ' max value: ', test[var].max())


# In[ ]:


X_train.head()


# In[ ]:


# test.Fare.isnull().sum()
test.Ticket_numerical.isnull().sum()


# Engineering rare labels in categorical variables

# In[ ]:


# find unfrequent labels in categorical variables
for var in categorical:
    print(var, X_train[var].value_counts()/np.float(len(X_train)))
    print()


# In[ ]:


def rare_imputation(variable, which='rare'):    
    # find frequent labels
    temp = X_train.groupby([variable])[variable].count()/np.float(len(X_train))
    frequent_cat = [x for x in temp.loc[temp>0.01].index.values]
    
    # create new variables, with Rare labels imputed
    if which=='frequent':
        # find the most frequent category
        mode_label = X_train.groupby(variable)[variable].count().sort_values().tail(1).index.values[0]
        X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], mode_label)
        X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], mode_label)
        test[variable] = np.where(test[variable].isin(frequent_cat), test[variable], mode_label)
    
    else:
        X_train[variable] = np.where(X_train[variable].isin(frequent_cat), X_train[variable], 'Rare')
        X_test[variable] = np.where(X_test[variable].isin(frequent_cat), X_test[variable], 'Rare')
        test[variable] = np.where(test[variable].isin(frequent_cat), test[variable], 'Rare')


# In[ ]:


rare_imputation('Cabin_categorical', 'frequent')
rare_imputation('Ticket_categorical', 'rare')


# In[ ]:


# let's check that it worked
for var in categorical:
    print(var, X_train[var].value_counts()/np.float(len(X_train)))
    print()


# In[ ]:


# let's check that it worked
for var in categorical:
    print(var, test[var].value_counts()/np.float(len(test)))
    print()


# **Encode categorical variables **

# Sex: one hot encoding
# 
# Remaining variables: replace by risk probability

# In[ ]:


for df in [X_train, X_test, test]:
    df['Sex']  = pd.get_dummies(df.Sex, drop_first=True)


# In[ ]:


print(X_train.Sex.unique())
print(X_test.Sex.unique())
print(test.Sex.unique())


# In[ ]:


def encode_categorical_variables(var, target):
        # make label to risk dictionary
        ordered_labels = X_train.groupby([var])[target].mean().to_dict()
        
        # encode variables
        X_train[var] = X_train[var].map(ordered_labels)
        X_test[var] = X_test[var].map(ordered_labels)
        test[var] = test[var].map(ordered_labels)
 
# enccode labels in categorical vars
for var in categorical:
    encode_categorical_variables(var, 'Survived')


# In[ ]:


variables_that_need_scaling = ['Pclass', 'Age', 'Sibsp', 'Parch', 'Cabin_numerical', 'Family_size']


# In[ ]:


training_vars = [var for var in X_train.columns if var not in ['PassengerId', 'Survived']]
training_vars


# In[ ]:


# fit scaler
scaler = MinMaxScaler() # create an instance
scaler.fit(X_train[training_vars])


# The scaler is now ready, we can use it in a machine learning algorithm when required. See below.

# # Machine Learning algorithm

# **xgboost**

# In[ ]:


xgb_model = xgb.XGBClassifier()
 
eval_set = [(X_test[training_vars], y_test)]
xgb_model.fit(X_train[training_vars], y_train, eval_metric="auc", eval_set=eval_set, verbose=False)
 
pred = xgb_model.predict_proba(X_train[training_vars])
print('xgb train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = xgb_model.predict_proba(X_test[training_vars])
print('xgb test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# **Random_forest**

# In[ ]:


rf_model = RandomForestClassifier()
rf_model.fit(X_train[training_vars], y_train)
 
pred = rf_model.predict_proba(X_train[training_vars])
print('RF train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = rf_model.predict_proba(X_test[training_vars])
print('RF test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# **Adaboost**

# In[ ]:


ada_model = AdaBoostClassifier()
ada_model.fit(X_train[training_vars], y_train)
 
pred = ada_model.predict_proba(X_train[training_vars])
print('Adaboost train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = ada_model.predict_proba(X_test[training_vars])
print('Adaboost test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# **Logistic regression**

# In[ ]:


logit_model = LogisticRegression()
logit_model.fit(scaler.transform(X_train[training_vars]), y_train)
 
pred = logit_model.predict_proba(scaler.transform(X_train[training_vars]))
print('Logit train roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
pred = ada_model.predict_proba(scaler.transform(X_test[training_vars]))
print('Logit test roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))


# Select threshold for maximum accuracy

# In[ ]:


pred_ls = []
for model in [xgb_model, rf_model, ada_model, logit_model]:
    pred_ls.append(pd.Series(model.predict_proba(X_test[training_vars])[:,1]))
 
final_pred = pd.concat(pred_ls, axis=1).mean(axis=1)
print('Ensemble test roc-auc: {}'.format(roc_auc_score(y_test,final_pred)))


# In[ ]:


tpr, tpr, thresholds = metrics.roc_curve(y_test, final_pred)
thresholds


# In[ ]:


accuracy_ls = []
for thres in thresholds:
    y_pred = np.where(final_pred>thres,1,0)
    accuracy_ls.append(metrics.accuracy_score(y_test, y_pred, normalize=True))
    
accuracy_ls = pd.concat([pd.Series(thresholds), pd.Series(accuracy_ls)],
                        axis=1)
accuracy_ls.columns = ['thresholds', 'accuracy']
accuracy_ls.sort_values(by='accuracy', ascending=False, inplace=True)
accuracy_ls.head()


# **Feature importance**

# In[ ]:


importance = pd.Series(rf_model.feature_importances_)
importance.index = training_vars
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(12,6))


# In[ ]:


importance = pd.Series(xgb_model.feature_importances_)
importance.index = training_vars
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(12,6))


# In[ ]:


importance = pd.Series(np.abs(logit_model.coef_.ravel()))
importance.index = training_vars
importance.sort_values(inplace=True, ascending=False)
importance.plot.bar(figsize=(12,6))


# <html><font size=5 color='red'>If you liked this kernel, please drop an UPVOTE. It motivates me to produce more quality content :)</font></html>

# In[ ]:




