#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data Processing and Cleaning
import numpy as np
import pandas as pd

# Data Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

# Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz

# Modeling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

#Miscellaneous
from tqdm import tqdm_notebook

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.


# ## Reading Data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train.head(3).T


# From a first peek at the data, we can spot missing values and features that are JSON objects. Let's dig a bit more into the dataset and its features

# In[ ]:


train.shape


# In[ ]:


test.shape


# The train data has less observations than the test data! Challenge accepted

# Editing erronous Data in train and test set (based on Discussion forums)

# In[ ]:


train.loc[train['id'] == 16,'revenue'] = 192864         
train.loc[train['id'] == 90,'budget'] = 30000000                  
train.loc[train['id'] == 118,'budget'] = 60000000       
train.loc[train['id'] == 149,'budget'] = 18000000       
train.loc[train['id'] == 313,'revenue'] = 12000000       
train.loc[train['id'] == 451,'revenue'] = 12000000      
train.loc[train['id'] == 464,'budget'] = 20000000       
train.loc[train['id'] == 470,'budget'] = 13000000       
train.loc[train['id'] == 513,'budget'] = 930000         
train.loc[train['id'] == 797,'budget'] = 8000000        
train.loc[train['id'] == 819,'budget'] = 90000000       
train.loc[train['id'] == 850,'budget'] = 90000000       
train.loc[train['id'] == 1007,'budget'] = 2              
train.loc[train['id'] == 1112,'budget'] = 7500000       
train.loc[train['id'] == 1131,'budget'] = 4300000        
train.loc[train['id'] == 1359,'budget'] = 10000000       
train.loc[train['id'] == 1542,'budget'] = 1             
train.loc[train['id'] == 1570,'budget'] = 15800000       
train.loc[train['id'] == 1571,'budget'] = 4000000        
train.loc[train['id'] == 1714,'budget'] = 46000000       
train.loc[train['id'] == 1721,'budget'] = 17500000       
train.loc[train['id'] == 1865,'revenue'] = 25000000      
train.loc[train['id'] == 1885,'budget'] = 12             
train.loc[train['id'] == 2091,'budget'] = 10             
train.loc[train['id'] == 2268,'budget'] = 17500000       
train.loc[train['id'] == 2491,'budget'] = 6              
train.loc[train['id'] == 2602,'budget'] = 31000000       
train.loc[train['id'] == 2612,'budget'] = 15000000       
train.loc[train['id'] == 2696,'budget'] = 10000000      
train.loc[train['id'] == 2801,'budget'] = 10000000       
train.loc[train['id'] == 335,'budget'] = 2 
train.loc[train['id'] == 348,'budget'] = 12
train.loc[train['id'] == 470,'budget'] = 13000000 
train.loc[train['id'] == 513,'budget'] = 1100000
train.loc[train['id'] == 640,'budget'] = 6 
train.loc[train['id'] == 696,'budget'] = 1
train.loc[train['id'] == 797,'budget'] = 8000000 
train.loc[train['id'] == 850,'budget'] = 1500000
train.loc[train['id'] == 1199,'budget'] = 5 
train.loc[train['id'] == 1282,'budget'] = 9              
train.loc[train['id'] == 1347,'budget'] = 1
train.loc[train['id'] == 1755,'budget'] = 2
train.loc[train['id'] == 1801,'budget'] = 5
train.loc[train['id'] == 1918,'budget'] = 592 
train.loc[train['id'] == 2033,'budget'] = 4
train.loc[train['id'] == 2118,'budget'] = 344 
train.loc[train['id'] == 2252,'budget'] = 130
train.loc[train['id'] == 2256,'budget'] = 1 
train.loc[train['id'] == 2696,'budget'] = 10000000


# In[ ]:


test.loc[test['id'] == 3033,'budget'] = 250 
test.loc[test['id'] == 3051,'budget'] = 50
test.loc[test['id'] == 3084,'budget'] = 337
test.loc[test['id'] == 3224,'budget'] = 4  
test.loc[test['id'] == 3594,'budget'] = 25  
test.loc[test['id'] == 3619,'budget'] = 500  
test.loc[test['id'] == 3831,'budget'] = 3  
test.loc[test['id'] == 3935,'budget'] = 500  
test.loc[test['id'] == 4049,'budget'] = 995946 
test.loc[test['id'] == 4424,'budget'] = 3  
test.loc[test['id'] == 4460,'budget'] = 8  
test.loc[test['id'] == 4555,'budget'] = 1200000 
test.loc[test['id'] == 4624,'budget'] = 30 
test.loc[test['id'] == 4645,'budget'] = 500 
test.loc[test['id'] == 4709,'budget'] = 450 
test.loc[test['id'] == 4839,'budget'] = 7
test.loc[test['id'] == 3125,'budget'] = 25 
test.loc[test['id'] == 3142,'budget'] = 1
test.loc[test['id'] == 3201,'budget'] = 450
test.loc[test['id'] == 3222,'budget'] = 6
test.loc[test['id'] == 3545,'budget'] = 38
test.loc[test['id'] == 3670,'budget'] = 18
test.loc[test['id'] == 3792,'budget'] = 19
test.loc[test['id'] == 3881,'budget'] = 7
test.loc[test['id'] == 3969,'budget'] = 400
test.loc[test['id'] == 4196,'budget'] = 6
test.loc[test['id'] == 4221,'budget'] = 11
test.loc[test['id'] == 4222,'budget'] = 500
test.loc[test['id'] == 4285,'budget'] = 11
test.loc[test['id'] == 4319,'budget'] = 1
test.loc[test['id'] == 4639,'budget'] = 10
test.loc[test['id'] == 4719,'budget'] = 45
test.loc[test['id'] == 4822,'budget'] = 22
test.loc[test['id'] == 4829,'budget'] = 20
test.loc[test['id'] == 4969,'budget'] = 20
test.loc[test['id'] == 5021,'budget'] = 40 
test.loc[test['id'] == 5035,'budget'] = 1 
test.loc[test['id'] == 5063,'budget'] = 14 
test.loc[test['id'] == 5119,'budget'] = 2 
test.loc[test['id'] == 5214,'budget'] = 30 
test.loc[test['id'] == 5221,'budget'] = 50 
test.loc[test['id'] == 4903,'budget'] = 15
test.loc[test['id'] == 4983,'budget'] = 3
test.loc[test['id'] == 5102,'budget'] = 28
test.loc[test['id'] == 5217,'budget'] = 75
test.loc[test['id'] == 5224,'budget'] = 3 
test.loc[test['id'] == 5469,'budget'] = 20 
test.loc[test['id'] == 5840,'budget'] = 1 
test.loc[test['id'] == 5960,'budget'] = 30
test.loc[test['id'] == 6506,'budget'] = 11 
test.loc[test['id'] == 6553,'budget'] = 280
test.loc[test['id'] == 6561,'budget'] = 7
test.loc[test['id'] == 6582,'budget'] = 218
test.loc[test['id'] == 6638,'budget'] = 5
test.loc[test['id'] == 6749,'budget'] = 8 
test.loc[test['id'] == 6759,'budget'] = 50 
test.loc[test['id'] == 6856,'budget'] = 10
test.loc[test['id'] == 6858,'budget'] =  100
test.loc[test['id'] == 6876,'budget'] =  250
test.loc[test['id'] == 6972,'budget'] = 1
test.loc[test['id'] == 7079,'budget'] = 8000000
test.loc[test['id'] == 7150,'budget'] = 118
test.loc[test['id'] == 6506,'budget'] = 118
test.loc[test['id'] == 7225,'budget'] = 6
test.loc[test['id'] == 7231,'budget'] = 85
test.loc[test['id'] == 5222,'budget'] = 5
test.loc[test['id'] == 5322,'budget'] = 90
test.loc[test['id'] == 5350,'budget'] = 70
test.loc[test['id'] == 5378,'budget'] = 10
test.loc[test['id'] == 5545,'budget'] = 80
test.loc[test['id'] == 5810,'budget'] = 8
test.loc[test['id'] == 5926,'budget'] = 300
test.loc[test['id'] == 5927,'budget'] = 4
test.loc[test['id'] == 5986,'budget'] = 1
test.loc[test['id'] == 6053,'budget'] = 20
test.loc[test['id'] == 6104,'budget'] = 1
test.loc[test['id'] == 6130,'budget'] = 30
test.loc[test['id'] == 6301,'budget'] = 150
test.loc[test['id'] == 6276,'budget'] = 100
test.loc[test['id'] == 6473,'budget'] = 100
test.loc[test['id'] == 6842,'budget'] = 30


# ## EDA

# ### Null and Missing Values

# In[ ]:


pd.DataFrame(train.isnull().sum()).T


# *Belongs to collection* and *homepage* have a lot of null values. Let's put a **percentage** to these missing values for both the train and test set

# In[ ]:


((pd.DataFrame(train.isnull().sum()).T)/len(train))*100


# In[ ]:


((pd.DataFrame(test.isnull().sum()).T)/len(test))*100


# 1. The distribution of null values across both the train and test set is roughly the same. This is good news!
# 2. *Budget* feature has 0 null values. But this doesn't mean that all values of this feature are meaningful. It could simply be due to the common practive of replacing null values with a dummy value like -1, 0, or 999
# 
# Let's look at the frequency of unique values held by the *Budget* feature to see if anything out of the ordinary pops up

# In[ ]:


pd.DataFrame(train.budget.value_counts()).T


# So the *Budget* feature has 812 values as 0. The likely cause is that information wasn't collected or available for those observations. It is unlikely that some of the movies had an extremely low budget, close to zero. 
# 
# We are immediately faced with a decision. How to deal with these missing values? Dropping more than 25% of the training set is not the best idea, and the *test* set is likely to have the same problem. Therefore, we should should look to fill values for these observations from external sources or the median value for budget. Let's deal with this later.

# ### Target Variable Problems

# Another issue we will have to take care of is the units of the budget and target variable, revenue. Let's deal with this in the data cleaning section later.

# In[ ]:


train[train['revenue'] < 10][['imdb_id', 'title']].T


# ### Visualizations

# Let's plot different variables in the dataset to potentially gain interesting insights about distributions and target variables.

# In[ ]:


fig = plt.figure(figsize=(18,15))
plt.subplots_adjust(hspace=0.5)

# Plot 1: Target Variable Distribution
plt.subplot2grid((4,2), (0,0))

sns.distplot(np.log1p(train['revenue']), kde=False, bins=40)

plt.title('Distribution for Target Variable', fontsize = 15)
plt.xlabel('Revenue - log(1+x)', fontsize=12)

# Plot 2: Revenue and Budget
plt.subplot2grid((4,2), (0,1))

sns.scatterplot(x = 'budget', y = 'revenue', data=train)

plt.title('Revenue vs. Budget', fontsize = 15)
plt.xlabel('Budget', fontsize=12)
plt.ylabel('Revenue', fontsize=12)

# Plot 3: Revenue, Runtime, and Popularity
ax = plt.subplot2grid((4,2), (1,0), projection='3d', rowspan = 2, colspan = 2)

x3d = np.array(train['runtime'])
y3d = np.array(train['popularity'])
z3d = np.array(train['revenue'])

# Unique category labels
color_labels = train['original_language'].unique()

# List of RGB triplets
rgb_values = sns.color_palette("Set2", len(color_labels))

# Map label to RGB
color_map = dict(zip(color_labels, rgb_values))
colors = train['original_language'].map(color_map)
colors = np.random.rand(len(train))

ax.scatter(
    x3d, y3d, z3d,
    c = colors,
    alpha = 0.8,
    )

ax.set_xlabel('Runtime')
ax.set_ylabel('Popularity')
ax.set_zlabel('Revenue')
ax.set_title('3D plot for Runtime, Popularity and Revenue', fontsize = 15)

# Plot 4: Correlation matrix
plt.subplot2grid((4,2), (3,0))

corr = train.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(corr, mask = mask, cmap = 'PiYG', annot = True, fmt=".2f")

plt.yticks(rotation=0) 
plt.xticks(rotation=0)
plt.title('Correlation Matrix for Train Data', fontsize = 15)

# Plot 5: Distribution of Films by Language
plt.subplot2grid((4,2), (3,1), colspan = 2)

top5_languages = train['original_language'].value_counts()[:5]
top5_languages.plot(kind = 'bar')

plt.yticks(rotation=0) 
plt.xticks(rotation=0)
plt.title('Distribution of Films by Language', fontsize = 15)

# Display Plot
plt.show()


# 1. **Target Variable Distribution:** It's never a bad idea to start by plotting the distribution of the target variable. Using np.log1p allows us to plot the several values between 0 and 1 million USD and get a much more uniform distribution.
# 2. **Revenue vs. Budget:** A scatterplot to explore the relationship between these two numerical features. We have plenty of zero values for budget
# 3. **3D Plot:** A 3D plot to explore the relationship between *Runtime, Popularity, and Revenue.*
# 4. **Correlation Matrix:** So revenue and budget are highly correlated. Let's try to predict the target variable with just with this one variable and get a baseline score and leaderboard rank!
# 5. **Distribution by Language** The data is dominated by English language films. French, Russian, Spanish, and Hindi films follow.

# ## Simple Models

# Before using external data or performing extensive cleaning and parsing of features, let us fit the simplest linear regression and Random Forest model to get a baseline score. Before we do that though, we need to take the log of the target variable, as the competition metric is RMSLE (Root Mean Squared Log Error)

# In[ ]:


train['revenue'] = np.log1p(train['revenue'])


# ### Linear Regression

# Let's perform a simple linear regression Model using just the budget feature.

# In[ ]:


x = train.budget.values.reshape(-1,1)
y = train.revenue
reg = LinearRegression().fit(x, y)


# In[ ]:


print(f'Regression Score: {reg.score(x, y)}')
print(f'Regression Coefficient: {reg.coef_[0]}')
print(f'Regression Intercept: {reg.intercept_}')


# In[ ]:


predictions = reg.predict(test['budget'].values.reshape(-1,1))


# #### Preparing Submission for Linear Regression

# In[ ]:


submission['revenue'] = np.round(np.expm1(predictions))


# In[ ]:


submission.to_csv('submission_budget_linreg.csv', index = False)


# This gives us a Kaggle score of *2.71*, and would place you around the top 75th percenticle of this competition. It is a significant improvement over the default predictions, which led to a Kaggle score of *3.79*. We have barely used any brainpower so this is not bad at all.

# ### Random Forest

# In[ ]:


rf_cols = ['budget', 'original_language', 'popularity', 'release_date', 'runtime', 'status', 'homepage', 'overview', 'revenue']
rf_train = train[rf_cols].copy()
rf_cols.remove('revenue')
rf_test = test[rf_cols].copy()


# #### Budget Feature
# Fill zero values for the budget feature in train and test data with median of the feature in the train set. We shall use only the train set to calculate median budget. This will avoid data leakage. We will also create another binary feature, *'budget_is_median'* which will hold the value 1 for indices that have been filled with median budget

# In[ ]:


median_budget = rf_train[rf_train['budget'] > 0]['budget'].median()
median_budget


# In[ ]:


def fill_budget(df, median_budget):
    df['budget_is_median'] = 0
    df.loc[df.budget == 0, 'budget_is_median'] = 1
    df.loc[df.budget == 0, 'budget'] = median_budget
    return df


# In[ ]:


rf_train = fill_budget(rf_train, median_budget)
rf_test = fill_budget(rf_test, median_budget)


# #### Original Language Feature
# We will label encode this categorical feature using sklearn. For now, let's do this in the simplest manner and not worry about any smart encoding. We will have to concatenate the train and test set in order before fitting the label encodings

# In[ ]:


rf_combined = pd.concat([rf_train, rf_test], sort=False)


# In[ ]:


le = LabelEncoder()
le.fit(rf_combined['original_language'])
rf_train['original_language'] = le.transform(rf_train['original_language'])
rf_test['original_language'] = le.transform(rf_test['original_language'])


# #### Status Feature
# We will deal with the *Status* categorical feature in the same manner. From our EDA section, we know that there are 2 missing values in the test set. Let's replace these values with the most common occurence of the variable, *'Released'*

# In[ ]:


rf_test.loc[rf_test['status'].isnull() == True, 'status'] = 'Released'
rf_combined.loc[rf_combined['status'].isnull() == True, 'status'] = 'Released'


# In[ ]:


le = LabelEncoder()
le.fit(rf_combined['status'])
rf_train['status'] = le.transform(rf_train['status'])
rf_test['status'] = le.transform(rf_test['status'])


# #### Homepage Feature

# This feature will store 0 for movies that don't have a homepage, and 1 for movies that do.

# In[ ]:


rf_train.loc[rf_train['homepage'].isnull() == True, 'homepage'] = 0
rf_train.loc[rf_train['homepage'].isnull() == False, 'homepage'] = 1

rf_test.loc[rf_test['homepage'].isnull() == True, 'homepage'] = 0
rf_test.loc[rf_test['homepage'].isnull() == False, 'homepage'] = 1


# #### Runtime Feature
# We have some null values for *Runtime* feature. There are also some values that have the value 0. Let's fill them with the median value for runtime from the train set.

# In[ ]:


median_runtime = rf_train['runtime'].median()
median_runtime


# In[ ]:


def fill_runtime(df, median_runtime):
    df['runtime_is_median'] = 0
    df.loc[df.runtime == 0, 'runtime_is_median'] = 1
    df.loc[df.runtime.isnull() == True, 'runtime_is_median'] = 1
    df.loc[df.runtime == 0, 'runtime'] = median_runtime
    df.loc[df.runtime.isnull() == True, 'runtime'] = median_runtime
    return df


# In[ ]:


rf_train = fill_runtime(rf_train, median_runtime)
rf_test = fill_runtime(rf_test, median_runtime)


# #### Release Data Feature

# Parse release date and extract features such as day, month, year!

# In[ ]:


from datetime import timedelta, date


# Filling missing data with external ground truth

# In[ ]:


rf_test.loc[rf_test['release_date'].isnull() == True, 'release_date'] = '10/19/2001'
test.loc[test['release_date'].isnull() == True, 'release_date'] = '10/19/2001'


# In[ ]:


def add_date_features(df, col, prefix):
    df[col] = pd.to_datetime(df[col])
    future = df[col] > pd.Timestamp(year=2017,month=12,day=31)
    df.loc[future, col] -= timedelta(days=365.25*100)
    
    df[prefix+'_day_of_week'] = df[col].dt.dayofweek
    df[prefix+'_day_of_year'] = df[col].dt.dayofyear
    df[prefix+'_month'] = df[col].dt.month
    df[prefix+'_year'] = df[col].dt.year
    df[prefix+'_day'] = df[col].dt.day
    df[prefix+'_is_year_end'] = df[col].dt.is_year_end
    df[prefix+'_is_year_start'] = df[col].dt.is_year_start
    df[prefix+'_week'] = df[col].dt.week
    df[prefix+'_quarter'] = df[col].dt.quarter    
    
    df.drop(col, axis = 1, inplace = True)

    return df


# In[ ]:


rf_train = add_date_features(rf_train, 'release_date', 'release')
rf_test = add_date_features(rf_test, 'release_date', 'release')


# ## Parsing JSON Features

# So far, we haven't parsed any JSON features. Let's go through the *production companies*, *production countries*, *cast*, *crew*, *keywords*, and *belongs to collection* feature and try and extract information that might help our model

# In[ ]:


def get_dictionary(s):
    try:
        d = eval(s)
    except:
        d = {}
    return d


# A note on the *eval* function being used.  The *eval()* method parses the expression passed to this method and runs python expression (code) within the program. In simple terms, the *eval()* method runs the python code (which is passed as an argument) within the program.

# In[ ]:


json_cols = ['production_companies', 'production_countries', 'cast', 'crew', 'Keywords', 'belongs_to_collection']
for col in json_cols:
    rf_train[col] = train[col]
    rf_train[col] = rf_train[col].apply(lambda x: get_dictionary(x))
    rf_test[col] = test[col]
    rf_test[col] = rf_test[col].apply(lambda x: get_dictionary(x))


# ## Feature Engineering

# 1. For features such as cast, crew, production companies, keywords, and production countries, get the length/size of the feature. 
# 2. If the movie belongs to a collection, extract the name of the collection, and label encode it.
# 3. For movie overview, count the number of words in overview
# 4. Feature interactions

# In[ ]:


for col in json_cols:
    # Get name of collection movie belongs to
    if col == 'belongs_to_collection':
        rf_train['collection_name'] = rf_train[col].apply(lambda row: row[0]['name'] if row != {} else '0')
        rf_test['collection_name'] = rf_test[col].apply(lambda row: row[0]['name'] if row != {} else '0')
        rf_combined = pd.concat([rf_train, rf_test], sort=False)
        le = LabelEncoder()
        le.fit(rf_combined['collection_name'])
        rf_train['collection_name'] = le.transform(rf_train['collection_name'])
        rf_test['collection_name'] = le.transform(rf_test['collection_name'])    
    
    # Size of feature
    rf_train[col] = rf_train[col].apply(lambda row: 0 if row is None else len(row))
    rf_test[col] = rf_test[col].apply(lambda row: 0 if row is None else len(row))

# Word count for overview
rf_train['overview_wordcount'] = rf_train['overview'].str.split().str.len()
rf_train.drop('overview', axis = 1, inplace = True)
rf_train.loc[rf_train['overview_wordcount'].isnull() == True, 'overview_wordcount'] = 0

rf_test['overview_wordcount'] = rf_test['overview'].str.split().str.len()
rf_test.drop('overview', axis = 1, inplace = True)
rf_test.loc[rf_test['overview_wordcount'].isnull() == True, 'overview_wordcount'] = 0

# Feature Interactions
rf_train['_budget_runtime_ratio'] = np.round(rf_train['budget']/rf_train['runtime'], 2)
rf_train['_budget_year_ratio'] = np.round(rf_train['budget']/(rf_train['release_year']*rf_train['release_year']), 2)
rf_train['_releaseYear_popularity_ratio'] = np.round(rf_train['release_year']/rf_train['popularity'], 2)

rf_test['_budget_runtime_ratio'] = np.round(rf_test['budget']/rf_test['runtime'], 2)
rf_test['_budget_year_ratio'] = np.round(rf_test['budget']/(rf_test['release_year']*rf_test['release_year']), 2)
rf_test['_releaseYear_popularity_ratio'] = np.round(rf_test['release_year']/rf_test['popularity'], 2)


# So this is what the training data looks like after performing all the required cleaning

# In[ ]:


rf_train.head()


# ## Training

# #### Peform Train and Validation Split

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(
    rf_train.drop('revenue', axis = 1), rf_train['revenue'], 
    test_size=0.1, 
    random_state=42
)


# #### Functions to evaluate our Random Forest

# In[ ]:


def rmse(y_pred, y_true):
    return np.sqrt(mean_squared_error(y_pred, y_true))

def print_rf_score(model):
    print(f'Train R2:   {model.score(X_train, y_train)}')
    print(f'Valid R2:   {model.score(X_valid, y_valid)}')
    print(f'Train RMSE: {rmse(model.predict(X_train), y_train)}')
    print(f'Valid RMSE: {rmse(model.predict(X_valid), y_valid)}')


# #### Random Forest with default hyperparameters

# In[ ]:


rf = RandomForestRegressor(n_jobs = -1, random_state = 42)
rf.fit(X_train, y_train)
print_rf_score(rf)


# Our validation R squared is very low compared to the training R squared, suggesting we are overfitting. Although the validation RMSE will place us in the top 40% of the competition, much better than our naive linear regression model.

# #### Drawing a Single Decision Tree

# Here, we will draw a single decision tree that is builds our Random Forest ensemble. We will make a small tree which is easy to visualize. To achieve this, we set *max_depth* to 3. We will also turn *bootstrap* to False in order to sample all of the data and build a deterministic tree.

# In[ ]:


rf = RandomForestRegressor(
    n_estimators = 1, 
    max_depth = 3, 
    bootstrap = False, 
    n_jobs = -1, 
    random_state = 42
)
rf.fit(X_train, y_train)
print_rf_score(rf)


# In[ ]:


# Export as dot file
export_graphviz(
    rf.estimators_[0], 
    out_file='tree.dot', 
    feature_names = X_train.columns,
    rounded = True, 
    proportion = False, 
    precision = 2, 
    filled = True,
    rotate = True
)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png', height = 600, width = 800)


# #### Creating a deeper Single Tree

# Previously, we set our *max_depth* to 3 to make the tree easy to visualize. Now, let's remove this hyperparameter and observe the difference in evaluation.

# In[ ]:


rf = RandomForestRegressor(
    n_estimators = 1, 
    bootstrap = False, 
    n_jobs = -1, 
    random_state = 42
)
rf.fit(X_train, y_train)
print_rf_score(rf)


# As expected, we have a training R squared of 1. This is because each observation in the training data is a leaf node and can be accounted for easily. Our R squared and validation score on test data, however, has decreased tremendously.

# #### Bagging Trees

# As we saw, our deep single tree massively overfit. Bagging, an important concept in ensembling, suggests that if we create a large number of such trees built on random samples of data and average their errors, we will get a good model. To illustrate this process better, let's return to our default random forest with no hyperparameter tuning.

# In[ ]:


rf = RandomForestRegressor(n_estimators = 10, n_jobs = -1, random_state = 42)
rf.fit(X_train, y_train)
print_rf_score(rf)


# The value of *n_estimators* is 10, which is the default. Let's grab predictions for each of these 10 trees and look at how they perform on the first validation sample.

# In[ ]:


tree_preds = np.stack([tree.predict(X_valid) for tree in rf.estimators_])
print(f' Individual Tree Predictions: {[np.around(tree_preds[:,0], 1)]}')
print(f' Mean of Tree Predictions:    {np.mean(tree_preds[:,0])}')
print(f' Ground truth for sample:     {y_valid[0]}')


# As is visible, the predictions for individual trees are all over the place but their average is fairly reasonable.

# #### Visualizing the imapct of additional Trees

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_scores_r2 = []\nvalid_scores_r2 = []\ntrain_scores_rmse = []\nvalid_scores_rmse = []\nfor trees in tqdm_notebook(range(1, 100)):\n    rf = RandomForestRegressor(n_estimators = trees, n_jobs = -1, random_state = 42)\n    rf.fit(X_train, y_train)\n    train_scores_r2.append(rf.score(X_train, y_train))\n    valid_scores_r2.append(rf.score(X_valid, y_valid))    \n    train_scores_rmse.append(rmse(rf.predict(X_train), y_train))\n    valid_scores_rmse.append(rmse(rf.predict(X_valid), y_valid))')


# In[ ]:


fig = plt.figure(figsize=(10,8))
plt.subplots_adjust(hspace=0.5)

# Plot 1: Train R2
plt.subplot2grid((2,2), (0,0))

plt.plot(train_scores_r2)
plt.title('Training Data R Squared', fontsize = 15)
plt.xlabel('Estimators', fontsize=12)

# Plot 2: Valid R2
plt.subplot2grid((2,2), (0,1))

plt.plot(valid_scores_r2, color='r')
plt.title('Validation Set R Squared', fontsize = 15)
plt.xlabel('Estimators', fontsize=12)

# Plot 1: Train RMSE
plt.subplot2grid((2,2), (1,0))

plt.plot(train_scores_rmse)
plt.title('Training Data RMSE', fontsize = 15)
plt.xlabel('Estimators', fontsize=12)

# Plot 2: Valid RMSE
plt.subplot2grid((2,2), (1,1))

plt.plot(valid_scores_rmse, color='r')
plt.title('Validation Set RMSE', fontsize = 15)
plt.xlabel('Estimators', fontsize=12)

plt.show()


# #### Out of Bag Evaluation

# To know if our validation score is worse because our model is overfitting or because the validation set is from a different distribution, or both, we can leverage a hyperparameter called *oob_score*, or Out of Bag score. The idea behind it is to calculate error on the training set while only including those trees in the calculation of a row's error where that row was not included in the training the tree. 
# 
# If you have a lot of trees, all of the rows in the dataset should appear a few times in the out of bag samples. This approach is beneficial as we can see if our model generalizes, even if we have a small amount of data. We can avoid creating a separate validation set and lose valuable training data.

# In[ ]:


rf = RandomForestRegressor(
    n_estimators = 20, 
    n_jobs = -1, 
    oob_score = True, 
    random_state = 42
)
rf.fit(X_train, y_train)
print_rf_score(rf)
print(f'OOB Score:  {rf.oob_score_}')


# The Out of Bag R2 score is in the same range as the validation R2 score, which is good news.

# #### Min_Samples_Leaf

# To prevent overfitting, we will tune min_samples_leaf. This will reduce the depth of our trees by a couple of levels

# In[ ]:


rf = RandomForestRegressor(
    n_estimators = 20, 
    min_samples_leaf = 4, 
    n_jobs = -1, 
    oob_score = True, 
    random_state = 42
)
rf.fit(X_train, y_train)
print_rf_score(rf)
print(f'OOB Score:  {rf.oob_score_}')


# #### Max_Features

# In[ ]:


rf = RandomForestRegressor(
    n_estimators = 20, 
    min_samples_leaf = 4, 
    max_features = 0.3, 
    n_jobs = -1, 
    oob_score = True, 
    random_state = 42
)
rf.fit(X_train, y_train)
print_rf_score(rf)
print(f'OOB Score:  {rf.oob_score_}')


# #### Random Forest with Hyperparameter Tuning

# In[ ]:


rf = RandomForestRegressor(
    n_estimators = 20, 
    min_samples_leaf = 4, 
    max_features = 0.3, 
    n_jobs = -1,
    oob_score = True, 
    random_state = 42,
)
rf.fit(X_train, y_train)
print_rf_score(rf)
print(f'OOB Score:  {rf.oob_score_}')


# #### Predictions for Random Forest model

# In[ ]:


predictions = np.expm1(rf.predict(rf_test))
submission['revenue'] = np.round(predictions)
submission.to_csv('submission_simple_rf.csv', index = False)


# ## Ensembling

# ### XGBoost

# In[ ]:


def xgtrain(X_train, X_valid, y_train, y_valid):
    regressor = XGBRegressor(
        n_estimators = 50000, 
        learning_rate = 0.001,
        max_depth = 6, 
        subsample = 0.3, 
        colsample_bytree = 0.2
        )
    
    regressor_ = regressor.fit(
        X_train.values, y_train.values, 
        eval_metric = 'rmse', 
        eval_set = [
            (X_train.values, y_train.values), 
            (X_valid.values, y_valid.values)
        ],
        verbose = 1000,
        early_stopping_rounds = 500,
        )
    
    return regressor_


# In[ ]:


get_ipython().run_cell_magic('time', '', 'regressor_ = xgtrain(X_train, X_valid, y_train, y_valid)')


# ### Light GBM

# In[ ]:


def lgbtrain(X_train, y_train, X_valid, y_valid):
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2_root'},
        'max_depth': 4,
        'learning_rate': 0.001,
        'feature_fraction': 0.3,
        'bagging_fraction': 1,
        'bagging_freq': 1,
    }
    
    gbm = lgb.train(
        params,
        lgb_train,
        valid_sets = lgb_eval,
        num_boost_round=50000,
        early_stopping_rounds=500,
        verbose_eval = 1000
    )
    
    return gbm


# In[ ]:


get_ipython().run_cell_magic('time', '', 'gbm = lgbtrain(X_train, y_train, X_valid, y_valid)')


# ## Feature Importances

# In[ ]:


feature_importances = pd.DataFrame(rf.feature_importances_, index = X_train.columns, columns=['importance'])
feature_importances = feature_importances.sort_values('importance', ascending=True)
feature_importances.plot(kind = 'barh', figsize = (10,8))
plt.show()


# ### Preparing Submission

# In[ ]:


predictions = np.expm1(rf.predict(rf_test)) + np.expm1(regressor_.predict(rf_test.values)) + np.expm1(gbm.predict(rf_test.values))
predictions /= 3


# In[ ]:


submission['revenue'] = np.round(predictions)


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission_ensemble.csv', index = False)


# Get Output files without committing

# In[ ]:


from IPython.display import FileLinks
FileLinks('.')


# In[ ]:




