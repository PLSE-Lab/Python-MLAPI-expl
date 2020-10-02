#!/usr/bin/env python
# coding: utf-8

# ## Machine Learning Model Building Pipeline: Data Analysis
# 
# 1. Data Analysis
# 2. Feature Engineering
# 3. Feature Selection
# 4. Model Building
# 
# **This is the notebook for step 1: Data Analysis**
# 
# ===================================================================================================
# 
# ## Predicting Sale Price of Houses
# 
# The aim of the project is to build a machine learning model to predict the sale price of homes based on different explanatory variables describing aspects of residential houses. 
# 
# ### Why is this important? 
# 
# Predicting house prices is useful to identify fruitful investments, or to determine whether the price advertised for a house is over or underestimated, before making a buying judgment.
# 
# ### What is the objective of the machine learning model?
# 
# We aim to minimise the difference between the real price, and the estimated price by our model. We will evaluate model performance using the mean squared error (mse) and the root squared of the mean squared error (rmse).

# # 1. Data Analysis
# 
# In the following cells, we will analyse the variables of the House Price Dataset from Kaggle. I will take you through the different aspects of the analysis that we will make over the variables, and introduce you to the meaning of each of the variables as well.
# Let's go ahead and load the dataset.

# In[ ]:


# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
#% matplotlib inline

# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


# load dataset
data = pd.read_csv("../input/train.csv")

# rows and columns of the data
print(data.shape)

# visualise the dataset
data.head()


# The house price dataset contains 1460 rows, i.e., houses, and 81 columns, i.e., variables. 
# **We will analyse the dataset to identify:**
# 1. Data Anslysis<br>
# 1.1. Missing values<br>
# 1.2. Numerical variables<br>
# 1.2.1. Distribution of the numerical variables<br>
# 1.2.1.1. Discrete Numeric variables<br>
# 1.2.1.2. Continuous Numeric variables<br>
# 1.2.2. Outliers<br>
# 1.3. Categorical variables<br>
# 1.3.1. Cardinality of the categorical variables<br>
# 1.3.2. Rare Labels<br>
# 1.4. Potential relationship between the variables and the target: SalePrice

# ## 1.1 Missing values
# 
# Let's go ahead and find out which variables of the dataset contain missing values

# In[ ]:


# make a list of the variables that contain missing values
vars_with_na = [var for var in data.columns if data[var].isnull().sum()>1]

# print the variable name and the percentage of missing values
for var in vars_with_na:
    print(var, np.round(data[var].isnull().mean(), 3),  ' % missing values')


# Our dataset contains a few variables with missing values. We need to account for this in our following notebook, where we will engineer the variables for use in Machine Learning Models.

# Relationship between values being missing and Sale Price
# 
# Let's evaluate the price of the house for those cases where the information is missing, for each variable.

# In[ ]:


def analyse_na_value(df, var):
    df = df.copy()
    
    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    df[var] = np.where(df[var].isnull(), 1, 0)
    
    # let's calculate the mean SalePrice where the information is missing or present
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.show()
    
for var in vars_with_na:
    analyse_na_value(data, var)


# We see that the fact that the information is missing for those variables, is important. We will capture this information when we engineer the variables in our next section.

# ## 1.2 Numerical variables
# 
# Let's go ahead and find out what numerical variables we have in the dataset

# In[ ]:


# list of numerical variables
num_vars = [var for var in data.columns if data[var].dtypes != 'O']

print('Number of numerical variables: ', len(num_vars))

# visualise the numerical variables
data[num_vars].head()


# From the above view of the dataset, we notice the variable Id, which is an indicator of the house. We will not use this variable to make our predictions, as there is one different value of the variable per each row, i.e., each house in the dataset. See below:

# In[ ]:


print('Number of House Id labels: ', len(data.Id.unique()))
print('Number of Houses in the Dataset: ', len(data))


# #### Temporal variables
# 
# From the above view we also notice that we have 4 year variables. Typically, we will not use date variables as is, rather we extract information from them. For example, the difference in years between the year the house was built and the year the house was sold. We need to take this into consideration in our sectio, where we will engineer our features.

# In[ ]:


# list of variables that contain year information
year_vars = [var for var in num_vars if 'Yr' in var or 'Year' in var]

year_vars


# In[ ]:


# let's explore the content of these year variables
for var in year_vars:
    print(var, data[var].unique())
    print()


# As you can see, it refers to years.
# 
# We can also explore the evolution of the sale price with the years in which the house was sold:

# In[ ]:


data.groupby('YrSold')['SalePrice'].median().plot()
plt.ylabel('Median House Price')
plt.title('Change in House price with the years')


# There has been a drop in the value of the houses. That is unusual, in real life, house prices typically go up as years go by.
# 
# 
# Let's go ahead and explore whether there is a relationship between the year variables and SalePrice. For this, we will capture the elapsed years between the Year variables and the year in which the house was sold:

# In[ ]:


# let's explore the relationship between the year variables and the house price in a bit of more details
def analyse_year_vars(df, var):
    df = df.copy()
    
    # capture difference between year variable and year the house was sold
    df[var] = df['YrSold'] - df[var]
    
    plt.scatter(df[var], df['SalePrice'])
    plt.ylabel('SalePrice')
    plt.xlabel(var)
    plt.show()
    
for var in year_vars:
    if var !='YrSold':
        analyse_year_vars(data, var)
    


# We see that there is a tendency to a decrease in price, with older features.

# ### 1.2.1 Distribution of numeric variable

# #### 1.2.1.1 Discrete variables
# 
# Let's go ahead and find which variables are discrete, i.e., show a finite number of values

# In[ ]:


#  list of discrete variables
discrete_vars = [var for var in num_vars if len(data[var].unique())<20 and var not in year_vars+['Id']]

print('Number of discrete variables: ', len(discrete_vars))


# In[ ]:


# let's visualise the discrete variables
data[discrete_vars].head()


# We can see that these variables tend to be Qualifications or grading scales, or refer to the number of rooms, or units. Let's go ahead and analyse their contribution to the house price.

# In[ ]:


def analyse_discrete(df, var):
    df = df.copy()
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.ylabel('SalePrice')
    plt.show()
    
for var in discrete_vars:
    analyse_discrete(data, var)


# We see that there is a relationship between the variable numbers and the SalePrice, but this relationship is not always monotonic. 
# 
# For example, for OverallQual, there is a monotonic relationship: the higher the quality, the higher the SalePrice.  
# 
# However, for OverallCond, the relationship is not monotonic. Clearly, some Condition grades, like 5, favour better selling prices, but higher values do not necessarily do so. We need to be careful on how we engineer these variables to extract the most for a linear model. 
# 
# There are ways to re-arrange the order of the discrete values of a variable, to create a monotonic relationship between the variable and the target.

# #### 1.2.1.2 Continuous variables
# 
# Let's go ahead and find the distribution of the continuous variables. We will consider continuous all those that are not temporal or discrete variables in our dataset.

# In[ ]:


# list of continuous variables
cont_vars = [var for var in num_vars if var not in discrete_vars+year_vars+['Id']]

print('Number of continuous variables: ', len(cont_vars))


# In[ ]:


# let's visualise the continuous variables
data[cont_vars].head()


# In[ ]:


# Let's go ahead and analyse the distributions of these variables
def analyse_continous(df, var):
    df = df.copy()
    df[var].hist(bins=20)
    plt.ylabel('Number of houses')
    plt.xlabel(var)
    plt.title(var)
    plt.show()
    
for var in cont_vars:
    analyse_continous(data, var)


# We see that all of the above variables, are not normally distributed, including the target variable 'SalePrice'. For linear models to perform best, we need to account for non-Gaussian distributions. We will transform our variables in the next section, during our feature engineering section.
# 
# Let's also evaluate here if a log transformation renders the variables more Gaussian looking:

# In[ ]:


# Let's go ahead and analyse the distributions of these variables
def analyse_transformed_continous(df, var):
    df = df.copy()
    
    # log does not take negative values, so let's be careful and skip those variables
    if 0 in data[var].unique():
        pass
    else:
        # log transform the variable
        df[var] = np.log(df[var])
        df[var].hist(bins=20)
        plt.ylabel('Number of houses')
        plt.xlabel(var)
        plt.title(var)
        plt.show()
    
for var in cont_vars:
    analyse_transformed_continous(data, var)


# We get a better spread of values for most variables when we use the logarithmic transformation. This engineering step will most likely add performance value to our final model.

# In[ ]:


# let's explore the relationship between the house price and the transformed variables
# with more detail
def transform_analyse_continous(df, var):
    df = df.copy()
    
    # log does not take negative values, so let's be careful and skip those variables
    if 0 in data[var].unique():
        pass
    else:
        # log transform
        df[var] = np.log(df[var])
        df['SalePrice'] = np.log(df['SalePrice'])
        plt.scatter(df[var], df['SalePrice'])
        plt.ylabel('SalePrice')
        plt.xlabel(var)
        plt.show()
    
for var in cont_vars:
    if var !='SalePrice':
        transform_analyse_continous(data, var)


# From the previous plots, we observe some monotonic associations between SalePrice and the variables to which we applied the log transformation, for example 'GrLivArea'.

# ### 1.2.2 Outliers

# In[ ]:


# let's make boxplots to visualise outliers in the continuous variables 

def find_outliers(df, var):
    df = df.copy()
    
    # log does not take negative values, so let's be careful and skip those variables
    if 0 in data[var].unique():
        pass
    else:
        df[var] = np.log(df[var])
        df.boxplot(column=var)
        plt.title(var)
        plt.ylabel(var)
        plt.show()
    
for var in cont_vars:
    find_outliers(data, var)


# The majority of the continuous variables seem to contain outliers. Outliers tend to affect the performance of linear model. So it is worth spending some time understanding if removing outliers will add performance value to our  final machine learning model.

# ## 1.3 Categorical variables
# Let's go ahead and analyse the categorical variables present in the dataset.

# In[ ]:


### Categorical variables

cat_vars = [var for var in data.columns if data[var].dtypes=='O']

print('Number of categorical variables: ', len(cat_vars))


# In[ ]:


# let's visualise the values of the categorical variables
data[cat_vars].head()


# ### 1.3.1 Number of labels: cardinality
# 
# Let's evaluate how many different categories are present in each of the variables.

# In[ ]:


for var in cat_vars:
    print(var, len(data[var].unique()), ' categories')


# All the categorical variables show low cardinality, this means that they have only few different labels. That is good as we won't need to tackle cardinality during our feature engineering lecture.
# 
# ### 1.3.2 Rare labels:
# 
# Let's go ahead and investigate now if there are labels that are present only in a small number of houses:

# In[ ]:


def analyse_rare_labels(df, var, rare_perc):
    df = df.copy()
    tmp = df.groupby(var)['SalePrice'].count() / len(df)
    return tmp[tmp<rare_perc]

for var in cat_vars:
    print(analyse_rare_labels(data, var, 0.01))
    print()


# ## 1.4 Potential relationship between IDV and sales price(target)
# Some of the categorical variables show multiple labels that are present in less than 1% of the houses. We will engineer these variables in our next section. Labels that are under-represented in the dataset tend to cause over-fitting of machine learning models. That is why we want to remove them.
# 
# Finally, we want to explore the relationship between the categories of the different variables and the house price:

# In[ ]:


for var in cat_vars:
    analyse_discrete(data, var)


# Clearly, the categories give information on the SalePrice. In the next section, we will transform these strings / labels into numbers, so that we capture this information and transform it into a monotonic relationship between the category and the house price.

# # 2.Feature Engineering
# 
# ## House Prices dataset: Feature Engineering
# 
# In the following cells, we will engineer / pre-process the variables of the House Price Dataset. We will engineer the variables so that we tackle:
# 
# 1. Missing values
# 2. Temporal variables
# 3. Non-Gaussian distributed variables
# 4. Categorical variables: remove rare labels
# 5. Categorical variables: convert strings to numbers
# 5. Standarise the values of the variables to the same range
# 
# ### Setting the seed
# 
# It is important to note that we are engineering variables and pre-processing data with the idea of deploying the model if we find business value in it. Therefore, from now on, for each step that includes some element of randomness, it is extremely important that we **set the seed**. This way, we can obtain reproducibility between our research and our development code.
# 
# This is perhaps one of the most important lessons that you need to take away from this course: **Always set the seeds**.

# ### Separate dataset into train and test
# 
# Before beginning to engineer our features, it is important to separate our data intro training and testing set. This is to avoid over-fitting. This step involves randomness, therefore, we need to set the seed.

# In[ ]:


# to handle datasets
import pandas as pd
import numpy as np

# for plotting
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import MinMaxScaler

# to visualise al the columns in the dataframe
pd.pandas.set_option('display.max_columns', None)


# In[ ]:


# Let's separate into train and test set
# Remember to set the seed (random_state for this sklearn function)

X_train, X_test, y_train, y_test = train_test_split(data, data.SalePrice,
                                                    test_size=0.1,
                                                    random_state=0) # we are setting the seed here
X_train.shape, X_test.shape


# ## 2.1 Missing values
# 
# For categorical variables, we will fill missing information by adding an additional category: "missing"

# In[ ]:


# make a list of the categorical variables that contain missing values
vars_with_na = [var for var in data.columns if X_train[var].isnull().sum()>1 and X_train[var].dtypes=='O']

# print the variable name and the percentage of missing values
for var in vars_with_na:
    print(var, np.round(X_train[var].isnull().mean(), 3),  ' % missing values')


# In[ ]:


# function to replace NA in categorical variables
def fill_categorical_na(df, var_list):
    X = df.copy()
    X[var_list] = df[var_list].fillna('Missing')
    return X


# In[ ]:


# replace missing values with new label: "Missing"
X_train = fill_categorical_na(X_train, vars_with_na)
X_test = fill_categorical_na(X_test, vars_with_na)

# check that we have no missing information in the engineered variables
X_train[vars_with_na].isnull().sum()


# In[ ]:


# check that test set does not contain null values in the engineered variables
[vr for var in vars_with_na if X_train[var].isnull().sum()>0]


# For numerical variables, we are going to add an additional variable capturing the missing information, and then replace the missing information in the original variable by the mode, or most frequent value:

# In[ ]:


# make a list of the numerical variables that contain missing values
vars_with_na = [var for var in data.columns if X_train[var].isnull().sum()>1 and X_train[var].dtypes!='O']

# print the variable name and the percentage of missing values
for var in vars_with_na:
    print(var, np.round(X_train[var].isnull().mean(), 3),  ' % missing values')


# In[ ]:


# replace the missing values
for var in vars_with_na:
    
    # calculate the mode
    mode_val = X_train[var].mode()[0]
    
    # train
    X_train[var+'_na'] = np.where(X_train[var].isnull(), 1, 0)
    X_train[var].fillna(mode_val, inplace=True)
    
    # test
    X_test[var+'_na'] = np.where(X_test[var].isnull(), 1, 0)
    X_test[var].fillna(mode_val, inplace=True)

# check that we have no more missing values in the engineered variables
X_train[vars_with_na].isnull().sum()


# In[ ]:


# check that we have the added binary variables that capture missing information
X_train[['LotFrontage_na', 'MasVnrArea_na', 'GarageYrBlt_na']].head()


# ## 2.2 Temporal variables
# 
# We remember from the previous lecture, that there are 4 variables that refer to the years in which something was built or something specific happened. We will capture the time elapsed between the that variable and the year the house was sold:

# In[ ]:


# let's explore the relationship between the year variables and the house price in a bit of more details

def elapsed_years(df, var):
    # capture difference between year variable and year the house was sold
    df[var] = df['YrSold'] - df[var]
    return df
for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
    X_train = elapsed_years(X_train, var)
    X_test = elapsed_years(X_test, var)
# check that test set does not contain null values in the engineered variables
[vr for var in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt'] if X_test[var].isnull().sum()>0]


# ## 2.3 Numerical variables: Non-Gaussian to gaussian
# 
# We will log transform the numerical variables that do not contain zeros in order to get a more Gaussian-like distribution. This tends to help Linear machine learning models. 

# In[ ]:


for var in ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']:
    X_train[var] = np.log(X_train[var])
    X_test[var]= np.log(X_test[var])
# check that test set does not contain null values in the engineered variables
[var for var in ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice'] if X_test[var].isnull().sum()>0]


# In[ ]:


# same for train set
[var for var in ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice'] if X_train[var].isnull().sum()>0]


# ## 2.4 Categorical variables: Removal of Rare variable
# 
# First, we will remove those categories within variables that are present in less than 1% of the observations:

# In[ ]:


# let's capture the categorical variables first
cat_vars = [var for var in X_train.columns if X_train[var].dtype == 'O']


# In[ ]:


def find_frequent_labels(df, var, rare_perc):
    # finds the labels that are shared by more than a certain % of the houses in the dataset
    df = df.copy()
    tmp = df.groupby(var)['SalePrice'].count() / len(df)
    return tmp[tmp>rare_perc].index

for var in cat_vars:
    frequent_ls = find_frequent_labels(X_train, var, 0.01)
    X_train[var] = np.where(X_train[var].isin(frequent_ls), X_train[var], 'Rare')
    X_test[var] = np.where(X_test[var].isin(frequent_ls), X_test[var], 'Rare')


# ## 2.5 Categorical variable: Convert string to number

# Next, we need to transform the strings of these variables into numbers. We will do it so that we capture the monotonic relationship between the label and the target:

# In[ ]:


# this function will assign discrete values to the strings of the variables, 
# so that the smaller value corresponds to the smaller mean of target

def replace_categories(train, test, var, target):
    ordered_labels = train.groupby([var])[target].mean().sort_values().index
    ordinal_label = {k:i for i, k in enumerate(ordered_labels, 0)} 
    train[var] = train[var].map(ordinal_label)
    test[var] = test[var].map(ordinal_label)


# In[ ]:


for var in cat_vars:
    replace_categories(X_train, X_test, var, 'SalePrice')


# In[ ]:


# check absence of na
[var for var in X_train.columns if X_train[var].isnull().sum()>0]


# In[ ]:


# check absence of na
[var for var in X_test.columns if X_test[var].isnull().sum()>0]


# In[ ]:


# let me show you what I mean by monotonic relationship between labels and target
def analyse_vars(df, var):
    df = df.copy()
    df.groupby(var)['SalePrice'].median().plot.bar()
    plt.title(var)
    plt.ylabel('SalePrice')
    plt.show()
    
for var in cat_vars:
    analyse_vars(X_train, var)


# We can now see monotonic relationships between the labels of our variables and the target (remember that the target is log-transformed, that is why the differences seem so small).

# ## 2.6 Feature Scaling
# 
# For use in linear models, features need to be either scaled or normalised. In the next section, I will scale features between the min and max values:

# In[ ]:


train_vars = [var for var in X_train.columns if var not in ['Id', 'SalePrice']]
len(train_vars)


# In[ ]:


X_train[['Id', 'SalePrice']].reset_index(drop=True)


# In[ ]:


# fit scaler
scaler = MinMaxScaler() # create an instance
scaler.fit(X_train[train_vars]) #  fit  the scaler to the train set for later use

# transform the train and test set, and add on the Id and SalePrice variables
X_train = pd.concat([X_train[['Id', 'SalePrice']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(X_train[train_vars]), columns=train_vars)],
                    axis=1)

X_test = pd.concat([X_test[['Id', 'SalePrice']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(X_test[train_vars]), columns=train_vars)],
                    axis=1)


# In[ ]:


X_train.head()


# In[ ]:


That concludes the feature engineering section for this dataset.


# In[ ]:


# check absence of missing values
X_train.isnull().sum()


# # 3. Feature Selection
# 
# In the following cells, we will select a group of variables, the most predictive ones, to build our machine learning models. 
# 
# ### Why do we need to select variables?
# 
# 1. For production: Fewer variables mean smaller client input requirements (e.g. customers filling out a form on a website or mobile app), and hence less code for error handling. This reduces the chances of bugs.
# 2. For model performance: Fewer variables mean simpler, more interpretable, less over-fitted models
# 
# 
# **We will select variables using the Lasso regression: Lasso has the property of setting the coefficient of non-informative variables to zero. This way we can identify those variables and remove them from our final models.**
# 
# ### Setting the seed
# 
# It is important to note, that we are engineering variables and pre-processing data with the idea of deploying the model if we find business value in it. Therefore, from now on, for each step that includes some element of randomness, it is extremely important that we **set the seed**. This way, we can obtain reproducibility between our research and our development code.

# In[ ]:





# In[ ]:


# capture the target
y_train = X_train['SalePrice']
y_test = X_test['SalePrice']

# drop unnecessary variables from our training and testing sets
X_train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
X_test.drop(['Id', 'SalePrice'], axis=1, inplace=True)


# ### Feature Selection
# 
# Let's go ahead and select a subset of the most predictive features. There is an element of randomness in the Lasso regression, so remember to set the seed.

# In[ ]:


# here I will do the model fitting and feature selection
# altogether in one line of code

# first, I specify the Lasso Regression model, and I
# select a suitable alpha (equivalent of penalty).
# The bigger the alpha the less features that will be selected.

# Then I use the selectFromModel object from sklearn, which
# will select the features which coefficients are non-zero
# to build the models
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
sel_ = SelectFromModel(Lasso(alpha=0.005, random_state=0)) # remember to set the seed, the random state in this function
sel_.fit(X_train, y_train)


# In[ ]:


# this command let's us visualise those features that were kept.
# Kept features have a True indicator
sel_.get_support()


# In[ ]:


# let's print the number of total and selected features

# this is how we can make a list of the selected features
selected_feat = X_train.columns[(sel_.get_support())]

# let's print some stats
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
    np.sum(sel_.estimator_.coef_ == 0)))


# In[ ]:


# print the selected features
selected_feat


# ### Identify the selected variables

# In[ ]:


# this is an alternative way of identifying the selected features 
# based on the non-zero regularisation coefficients:
selected_feats = X_train.columns[(sel_.estimator_.coef_ != 0).ravel().tolist()]
selected_feats


# ## House Prices dataset: Machine Learning Model build
# 
# In the following cells, we will finally build our machine learning models, utilising the engineered data and the pre-selected features. 
# 
# 
# ### Setting the seed
# 
# It is important to note, that we are engineering variables and pre-processing data with the idea of deploying the model if we find business value in it. Therefore, from now on, for each step that includes some element of randomness, it is extremely important that we **set the seed**. This way, we can obtain reproducibility between our research and our development code.
# 
# This is perhaps one of the most important lessons that you need to take away from this course: **Always set the seeds**.
# 
# Let's go ahead and load the dataset.

# In[ ]:



# here I will add this last feature, even though it was not selected in our previous step,
# because it needs key feature engineering steps that I want to discuss further during the deployment
# part of the course. 
selected_feats = selected_feats + ['LotFrontage'] 


# In[ ]:


# reduce the train and test set to the desired features

X_train = X_train[selected_feats]
X_test = X_test[selected_feats]

