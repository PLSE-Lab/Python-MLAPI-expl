#!/usr/bin/env python
# coding: utf-8

# # Titanic Data Wrangling
# The Titanic Survival Problem has become famous as a "toy" problem that illustrates important decisions about data for prediction tasks.  The survival problem is to create a predictive algorithm that can tell if an individual would have survived the titanic or not.  The principle is to use a data-driven approach to create the algorithm.  That is, use some of the survival data to create the algorithm and then use the remainder to test the algorithm.
# 
# This data is hosted at [kaggle](https://www.kaggle.com/c/titanic) and is used for tutorials of all kinds.  In this notebook we will focus almost exclusively on data wrangling to prepare the data for learning.
# 
# We are going to follow a multistep process.  
# 
# * First, we will look at the information we have for each passenger to understand what it tells us and to determine if it is useful.  We get the data in a spreadsheet format in which each row is a passenger and each column is a different type of data we know about that passenger (including whether or not that passenger survived).
# 
# * Next we will focus on trying to determine the quality of each type of data by looking at each column to see if it is related to survival or if the values are present for many of the passengers.  We will also look at relationships among the columns to see if they are correlated or not.
# 
# * Next we will look for inter-column relationships that can be used to define new columns with more predictive power that single columns.  We will also look for opportunities to group data within a column to provide classes of data with greater predictive power.
# 
# * Next we will make sure that all nominal variables have numerical codes.  This is important for a number of learning methods.
# 
# * Next we will drop the columns which we don't need.
# 
# * Next we will look at some techniques for "imputing" data points that are missing, but that we think might provide value.
# 
# * After imputation, we will revisit step 2 to see if there are correlations that show up with the richer data.  At this point we will use some useful visualizations to get a better intuition as to what the relationships in the data are.
# 
# * After acting on discoveries in the previous steps, we will perform some predictions to try to get an idea of what type of accuracy is possible.  We will use n-fold cross validation to help make our methods as robust and general as possible.

# ### The Passengers of the Titanic
# 
# Kaggle hosts [a description of the values in each column here](https://www.kaggle.com/c/titanic/data).  pandas will also give us some options for understanding the data.

# In[ ]:


import pandas as pd
import numpy as np
datapath = "../input/train.csv"
df = pd.read_csv(datapath)

# print column names
print(df.columns.values)

# look at the first 5 examples + column names
df.head()


# In[ ]:



#print the data types and quantities in each column:
df.info()


# With just a few methods, we can see alot about the data.  The `info` method in particular reveals important information about the types of data in each column, as well as the number of valid entries in each column.  
# 
# #### Data Types
# We have numeric data in the columns `['PassengerId' 'Survived' 'Pclass' 'Age' 'SibSp' 'Parch' 'Fare']`
# 
# and nominal data in the columns `['Name' 'Sex' 'Ticket' 'Cabin' 'Embarked']`
#  
# We will look closer at the data types below.
# 
# #### Missing Data
# There are 891 rows of data in total.  The output of `info()` reveals that there are three columns in which data is missing for some rows: `['Age' 'Cabin' 'Embarked']`

# In[ ]:


# `describe` reveals some statistics about our numeric columns.
df.describe()


# In[ ]:


# the 'O' option gives us some important info about nominal columns.
df.describe(include='O')


# ### Quality - What Data is Useful?
# Now that we know something about the data, Let's start thinking about what to do to best use what we have.  Let's look at each column separately to assertain their suitability for classifying survival.  Below is a quick summary of results
# 
# | Column Name | Type | Keep? | Comments |
# | --- | --- | --- | --- |
# | PassengerID | numeric | N | A different value for each row.  Not useful for learning classifiers. |
# | Survived | numeric | Y | This is the target value> |
# | Pclass | numeric | Y | Class of travel will likely affect survival. |
# | Name | nominal | N | There may be useful information here like title (Mr. Miss, Dr., etc.) |
# | Sex | nominal | Y | Change this field to 1 = female, 0 = male. |
# | Age | numeric | Y | Bin these values so that they are better for learning. Need to repair missing values. |
# | SibSp | numeric | Y | Traveling with family likely affects survival. |
# | Parch | numeric | Y | Traveling with family likely affects survival. |
# | Ticket | nominal | N | Ticket numbers are inconsistent and don't lend themselves to patterns. |
# | Fare | numeric | Y | Fare may affect survival. |
# | Cabin | nominal | N | We don't know enough about cabin to us it well. |
# | Embarked | nominal | Y | convert to a numeric value. |
# 
# Next let's remove the columns we know we won't use.

# In[ ]:


# remove columns we won't use
df = df.drop(['PassengerId', 'Ticket', 'Cabin'], axis=1)
df.head()


# #### Relationship to Survival
# the seaborn visualization library gives us some nice tools for visualizing relationship between values in different fields.  We will use this tools for each remaining column to determine if we want to continue using it for inference.

# In[ ]:


# first things first:  pairwise correlation of the numeric fields in df
df.corr()


# We will look only at the top two highly (absolute value) correlated items:
# 
# **Pclass and Fare** appear to be negatively correlated.  This makes sense because in general we would expact 1st class travel to be more expensive that 3rd class travel.
# 
# **Parch and SibSp** are positively correlated.  Apparently members of a family travel together often.
# 
# Notice that only numeric values show up here.  Later we will change nominal parameters to have nummeric values and rerun this analysis to them to the pairwise correlation matrix.
# 
# Next we will look at visualizing how columns compare to `Survived` values.

# #### Pclass

# In[ ]:


# What is the sample likelihood of survival for different passenger classes
df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

# what are the frequencies of Death and Survival given Passenger Class?
g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Pclass')


# Pclass appears to be very much related to survival in this sample.

# #### Age and AgeRange

# In[ ]:


# what are the frequencies of Death and Survival given Age?
g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Age', bins=20)


# It seems that Age has a strong affect on Survival.  Below we will use these histogram distributions to find good bin values for Age.  We will build bins around the ranges [0..17], [18..34], [35..49], [50..100].  

# In[ ]:


df['AgeRange'] = pd.cut(df['Age'], [0, 18, 35, 50,100], labels=[1, 2, 3, 4], include_lowest=True, right=True).astype(np.float)
df.head(10)


# In[ ]:


# What is the sample likelihood of survival for different Age Ranges
df[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# what are the frequencies of Death and Survival given Age range?
g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'AgeRange')


# It appears that young adults risk death on the titanic more than other categories of passengers.  Using `AgeRange`, we can see the afect of the values on survival:

# In[ ]:


# What is the sample likelihood of survival for different AgeRange?
df[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# #### SibSp - Siblings and Spouses

# In[ ]:


# What is the sample likelihood of survival for different SibSp?
df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# what are the frequencies of Death and Survival given SibSp?
g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'SibSp')


# One or two siblings or spouses seems to make survival more likely, but groups larger than 5 have no likelihood of survival.

# #### Parch - Parent or Child

# In[ ]:


# What is the sample likelihood of survival for different Parch?
df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# what are the frequencies of Death and Survival given Parch?
g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Parch')


# Again, small groups traveling together seem to have the best likelihood of survival.  

# #### Fare and FareClass

# In[ ]:


# what are the frequencies of Death and Survival given Fare?
g = sns.FacetGrid(df, col='Survived')
g.map(plt.hist, 'Fare', bins=20)


# It seems that lower fare indicates less chance of survival.  We will use this information to find a new variable related to `Fare` and `Pclass`.

# In[ ]:


g = sns.FacetGrid(df, col='Pclass', row='Survived')
g.map(plt.hist, 'Fare', )


# In[ ]:


# Bin the Fare into a FareClass column
df['FareClass'] = pd.cut(df['Fare'], [0, 50, 150, 275,1000], labels=[1, 2, 3, 4], include_lowest=True, right=True).astype(np.int8)
df.head()


# In[ ]:


# What is the sample likelihood of survival for different Parch?
df[['FareClass', 'Survived']].groupby(['FareClass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# #### Sex

# In[ ]:


# What is the sample likelihood of survival for different Sex?
df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# In[ ]:


# what are the frequencies of Death and Survival given Sex and Pclass?
g = sns.FacetGrid(df, col='Pclass', row='Sex')
g.map(plt.hist, 'Survived')


# Apparently sex has a very strong affect on survival.  If a passenger is female,  they already have a high likelihood of survival.  Also, if a passenger is male in passenger class 3, he is very likely to NOT survive.  Females in Pclass=3 are equally likely to survive or not, while females in other calsses have a high likelihood of survival.

# In[ ]:


# what are the frequencies of Death and Survival given Fare?
g = sns.FacetGrid(df, col='Survived', row='Sex')
g.map(plt.hist, 'Survived', bins=20)


# #### Embarked

# In[ ]:


# How does Embarkation and Sex compare to Survival?
g = sns.FacetGrid(df, col='Embarked', row='Sex')
g.map(plt.hist, 'Survived')


# In[ ]:


# How does Embarkation and FareClass compare to Survival?
g = sns.FacetGrid(df, col='FareClass', row='Embarked')
g.map(plt.hist, 'Survived')


# It appears that Embarkation has a strong association with survival conditional on FareClass and Sex.

# ### Inter-Related Columns
# 
# Before we look closer to the data, Let's work on deriving some new feature columns that seem likely to help.  We will add the following new fields:
# 
# * A `Title` field derived from the `Name` column
# * A Family size field derived from `SibSp` and `Parch`
# 
# Each of these changes requires us to use multiple columns to create and interpret the new field.

# #### Extract `Title` from `Name`

# In[ ]:


# Extract title from name using the fact that titles end with period.
df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# lets see if we have one for every row (there are 891 rows).
df['Title'].shape


# In[ ]:


# how many different values are there?
# crosstab the result with the Sex column to see how they are realated
titles = pd.crosstab(df['Title'], df['Sex'])
print(titles.shape)
titles


# Soon we will look at whether or not this field has any relationship with survival.  For the moment we will just hang onto this field. 

# #### Create `FamilySize` from `Parch` and `SibSp`

# In[ ]:


df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df.head()


# In[ ]:


# What is the sample likelihood of survival for different Family Sizes?
df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# ### Numeric Fields
# 
# Now we want to do a few things to change the remaining nominal data to numeric codes.  This is helpful for a couple of the decision algorithms we might choose to use.  
# 
# * A numeric field called `Gender` to replace `Sex`
# * A numeric field called `Depart` to replace `Embarked` 
# * A numeric field for `Title`

# In[ ]:


import numpy as np
df['Gender'] = df['Sex'].map({'male':0, 'female':1}).astype(np.uint8)
df['Depart'] = df['Embarked'].map({'S':1, 'C':2, 'Q':3}, na_action='ignore').astype(np.float)
titleArr = df['Title'].unique()
mapping = {v: k for k, v in dict(enumerate(titleArr)).items()}
df['NamePrefix'] = df['Title'].map(mapping, na_action='ignore').astype(np.uint8)
df.head(10)


# ### Drop Unneeded Columns
# There are many columns that we have transformed now.  Let's remove them since we wont need them for training.

# In[ ]:


df1 = df.drop(['Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title'], axis=1)
df1.head(10)


# ### Impute Missing Values
# Now that the columns are cleaner, lets make sure to fill in data that is missing so that we can use as many rows as possible.

# In[ ]:


df1.info()


# There are two columns with missing data, `AgeRange` and `Depart`.  We are going to **impute**, or guess the missing values in these columns.  There are a number of ways to do this:
# 
# * Pick a value that seems reasonable.  Often the **median** value of the field is used in this case.
# * Draw the value from a distribution.  Fix a distribution for the column's sample data and draw values from that distribution for the unknown values.  Typical distributions to use are categorical or normal.
# * Find rows with known values that are close to values in the row with unknown column.  This is known as **K-Nearest Neighbor (KNN)** imputation.
# 
# 

# In[ ]:


import numpy as np
# code to support knn imputation
def euclidean_distance(a, b):
    c = a-b
    return np.sqrt(c.dot(c))

# test
row1 = df1.iloc[0]
row2 = df1.iloc[1]
print(euclidean_distance(row1, row2))
print(euclidean_distance(row1, row1))

# test
# apply euclidean distance to every pair of rows
# return the (dis)similarity matrix with the distances between each pair of rows
def distance_matrix(df, distance_measure = euclidean_distance):
    x = df1.drop(['Survived'], axis=1).fillna(0).values
    print(x.shape)
    m = np.matrix(np.zeros(shape=(x.shape[0],x.shape[0])))
    for i in range(x.shape[0]):
        for j in range(i, x.shape[0]):
            m[i,j] = euclidean_distance(x[i], x[j])
    return m + m.T

# m is the (dis)similarity matrix with the distances between each pair of rows
m = distance_matrix(df1)
#print(m[:10,:10])

# remove all indexes that are nan on the field we want
AgeRange_nan_indexes = df1[df1['AgeRange'].isnull()].index.values
#print(AgeRange_nan_indexes)
m1 = np.delete(m, AgeRange_nan_indexes, axis=0)
print(m1.shape)
#print(np.sort(m1[:,5])[:5,0])

def get_k_nn_indexes(m, df_row, k):
    idxs = np.argsort(m[:,df_row], axis = 0)
    return idxs[:k,0]

# test
#print('get_k_nn_indexes(m1, 5, 9)')
#print(get_k_nn_indexes(m1, 5, 9))

def get_k_nn_values(df, idxs, col):
    icol = df.columns.get_loc(col)
    return df.values[idxs, icol]

# test
idxs = get_k_nn_indexes(m1, 5, 15)
#print('get_k_nn_values(df1, idxs, "AgeRange")')
#print(get_k_nn_values(df1, idxs, 'AgeRange'))

# select the most common value for the missing data among the top k samples
def select_best(knns):
    dfknn = pd.DataFrame(knns, columns=['values'])
    return dfknn['values'].value_counts().index[0]  # pick the top count of knns

# test
knns = get_k_nn_values(df1, idxs, 'AgeRange')
#print('select_best(knns)')
#print(select_best(knns))

# replace nan values in a dataframe
def replace_value(df, col, indexes, values):
    d = dict(zip(indexes, values))
    return df[col].fillna(d)

# test
# impute nans for a column
values = []
for idx in AgeRange_nan_indexes:
    impute_idxs = get_k_nn_indexes(m1, idx, 7)
    knns = get_k_nn_values(df1, impute_idxs, 'AgeRange')
    best = select_best(knns)
    values.append(best)
    
#print(values)
df2 = df1
df2['AgeRange'] = replace_value(df1, 'AgeRange', AgeRange_nan_indexes, values)
df2.info()
#df2[df2.isnull()].shape
df2

def impute_knn(m, df, column, nan_indexes, k):
    values = []
    for idx in nan_indexes:
        impute_idxs = get_k_nn_indexes(m, idx, k)
        knns = get_k_nn_values(df, impute_idxs, column)
        best = select_best(knns)
        values.append(best)
    return values

#test
Depart_nan_indexes = df1[df1['Depart'].isnull()].index.values
v = impute_knn(m1, df1, 'Depart', Depart_nan_indexes ,11)
df2['Depart'] = replace_value(df1, 'Depart', Depart_nan_indexes, v)
df2.info()


# In[ ]:


# To find the knn of a row, pick the column of m corresponding to a row with a missing value, 
#  sort the values of that vector ascending, then pick the top k values from the list.
#  Note that if the row index < k, you should pick k+1 rows and throw out the top value
#  since it will correspond to the row being selected.  Next pick the indexes of the top k 
#  values and get those rows from the dataframe.  Use the values in the columns in 
#  question to impute the missing values.
def impute_nans(df, cols, k=9):
    m = distance_matrix(df)
    for col in cols:
        # get the indexes to rows with nan entries in this column
        nan_indexes = df[df[col].isnull()].index.values
        #remove those rows from the m1 matrix
        m1 = np.delete(m, nan_indexes, axis=0)
        nan_values = impute_knn(m1, df, col, nan_indexes, k)
        df[col] = replace_value(df, col, nan_indexes, nan_values)
        
# test
impute_nans(df1, ['AgeRange', 'Depart'], k=11)
df1.head()


# ### Visualizations
# Now that we have cleaned up the data, we can look at the relationships of the data to survival again using some of the visualizations we've seen before.

# In[ ]:


g = sns.FacetGrid(df, row='FamilySize')
g.map(plt.hist, 'Survived')


# In[ ]:


g = sns.FacetGrid(df, row='AgeRange')
g.map(plt.hist, 'Survived')


# In[ ]:


g = sns.FacetGrid(df, row='NamePrefix')
g.map(plt.hist, 'Survived')


# `FamilySize`, `AgeRange`, and `NamePrefix` all have a number of values.  Some of these values correlate with survuival and some don't.  Sometimes the correlationis the result of very sparse data.  For example, `FamilySize = 11` correlates strongly with (non) survival.

# In[ ]:


# show a scattermatrix and a correlation matrix
df1.corr()


# The correlation matrix above shows the relationship between each pair of fields.  We will focus on correlations with `Survived`.  `Gender` has the strongest correlation.  Next is `Pclass` which is anti-correlated with `Survived`.  `NamePrefix` is slightly positively correlated with `Survived`. 

# In[ ]:


import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
scatter_matrix(df1, figsize=(15,15), diagonal='kde')


# The scatter matrix doesn't help us too much.  It is hard to see the results with the discrete value types.

# ### Learning from the Titanic
# 
# Now that we have data, lets see how good it is for prediction.  There are a few step to prepare for using this data for training.  
# - Separate the data into training and test sets or design cross validation strategy
# - Select an algorithm
# - Train a model 
# - Test the model

# #### Separating Data Into Training and Testing
# 

# In[ ]:


# train and test - we will use scikit-learn
from sklearn.linear_model import LogisticRegression
from patsy import dmatrices
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score

y,X = dmatrices('Survived ~ Pclass + AgeRange + FareClass + FamilySize + Gender + Depart + NamePrefix', 
                df1, return_type="dataframe")
y = np.ravel(y)

# in this case we hold out 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)

# create a logistic regression model
model_lr = LogisticRegression()
model_lr.fit(X_train,y_train)
model_lr.score(X_test, y_test) # check accuracy against the test data


# In[ ]:


# Lets try Support Vector Machine.
from sklearn import svm

model_svc = svm.SVC(kernel='rbf', C=1)
model_svc.fit(X_train, y_train)
model_svc.score(X_test, y_test)


# In[ ]:


# Now for a Random Forest
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
model_rf.score(X_test, y_test)


# Very nice result.  We can get more generalizable scores by looking at a boosted average of the test accuracy.  The most basic approach to this is to use cross validation.

# #### Cross-Validation
# Cross Validation is a way to provide robust accuracy scores when the quantity of data for training and testing is limited.  The idea is to split that data into a number of segments, then recombine the segments so that each segment is used as test data exactly once.  
# 
# So if we split the data into 3 segments and number them 1, 2, and 3, then we can combine to segments into training data and leave one for testing [[1,2],3].  Note that there are three different ways to do this; [[1,2],3], [[1,3],2],[1,[2,3]].  We train each of those and then test with the remaining segment.
# 
# This is done automatically using `cross_val_score`.

# In[ ]:


# logistic regression
scores_lm = cross_val_score(model_lr, X, y, cv=5)
print(scores_lm)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_lm.mean(), scores_lm.std() * 2))


# In[ ]:


# Support vector machine
scores_svc = cross_val_score(model_svc, X, y, cv=5)
print(scores_svc)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_svc.mean(), scores_svc.std() * 2))


# In[ ]:


# Random Forest
scores_rf = cross_val_score(model_rf, X, y, cv=5)
print(scores_rf)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_rf.mean(), scores_rf.std() * 2))

