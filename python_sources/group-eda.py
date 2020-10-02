#!/usr/bin/env python
# coding: utf-8

# 
# * A. Loading the data
# * B. General overview                   
# * C. Analysis of ndividual variables                
# * D. Prediction         

# In[ ]:


# Installing basic packages

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.ticker as ticker  #to change the display of units used in graphs

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
import os
print(os.listdir("../input"))


# In[ ]:


# uploading both the 'train' and 'test' data files into dataframes

train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')


# In[ ]:


print('train_df shape:', train_df.shape)
print('test_df shape:', test_df.shape)


# In[ ]:


# Train data: investigating all columns and missing values from each

print(train_df.info(verbose=True))


# From the previous table, 
# 1. the train data seems to be fairly complete. Not many NULLs
# 2. 22 columns. Most of the columns are categorical. Only 4 are values
# 3. I will set the value columns as float (revenue and budget)

# In[ ]:


# Test data: investigating all columns and missing values from each

print(test_df.info(verbose=True))


# In[ ]:


train_df.describe(include='all')


# After a quick view, I find:
# 
# 1. There are 5 columns with numerical values.
# 2. 'id' is included but should actually be the index.
# 3. The units are hard to read.

# In[ ]:


# setting the 'id' as index
train_df.set_index('id', inplace=True)
test_df.set_index('id', inplace=True)

# changing types a float for budget and revenue
train_df = train_df.astype({"budget":'float64', "revenue":'float64'})
test_df = test_df.astype({"budget":'float64'})

# adjusting display of large numbers using commas and showing only 2 decimal points
pd.set_option('display.float_format', '{:,.2f}'.format)


# In[ ]:


# check if it worked
print('Train data')
print(train_df.describe())
print("")
print('Test data')
print(test_df.describe())


# - The 'Test' data has similar characteristics as the 'Train' data
# - Both include 0.00 values in Budget, Popularity, Runtime and Revenue

# In[ ]:


sns.pairplot(train_df)


# - The last row of charts show correlations with revenue.
# - Budget seems to have some correlation.
# - Runtime seem only to show that outliers don't generate much revenue.
# - There seem to be 0 values in budget and revenue.

# **C. Analysis of individual variables**
# 
# 1. belongs_to_collection  
# 2. budget                   
# 3. genres                 
# 4. homepage                 
# 5. imdb_id                  
# 6. original_language    
# 7. original_title         
# 8. overview                 
# 9. popularity             
# 10. poster_path             
# 11. production_companies     
# 12. production_countries     
# 13. release_date             
# 14. runtime                
# 15. spoken_languages         
# 16. status                   
# 17. tagline                  
# 18. title                  
# 19. Keywords              
# 20. cast                   
# 21. crew                   
# 22. revenue 

# **C1. belongs to collection** 
# * New columns needed: 
# 1. Belongs_to a collection: Boolean, 1=yes, 0=no 
# 2. has_a_sequel: if there is a movie that folows the present one. Boolean, 1=yes, 0=no 
# 3. is_a_sequel: if there is a movie that precedes the present one. Boolean, 1=yes, 0=no 

# In[ ]:


#train_df['belongs_to_collection'].apply(lambda x: len(str(x)))
for x in train_df['belongs_to_collection'].apply(lambda x: len(str(x))):
    if x > 3:
        train_df['in_collection'] = 1
    else: 
        train_df['in_collection'] = 0

train_df['in_collection']

#train_df['belongs_to_collection'] = x = 1 if len(str(x))>3 else x = 0 for train_df['belongs_to_collection']


# In[ ]:





# In[ ]:


train_df['in_collection']


# **2. budget**

# In[ ]:


budget_ax = train_df['budget'].plot(kind='hist', bins= 100, figsize=(15,3))
budget_ax.set_title('Budget histogram')
budget_ax.xaxis.set_major_formatter(ticker.EngFormatter())


# In[ ]:


# Counting how many zero values there are per column
(train_df == 0).sum(axis=0)


# - Train_df: 812 rows have zero budget. 27% of total rows.
# - If valid data cannot be found to replace the zeroes, those rows will have to be dropped.

# In[ ]:


# analysing the rows with budget value > 0
train_df[train_df.budget>0].describe()


# - After excluding the movies with budget =0, there are still movies with unusual budgets ($1)

# In[ ]:


# analysing the rows where: [ 0 < budget value < 1,000 ]
train_df[(train_df.budget<1000) & (train_df.budget>0)].describe()


# - Of all movies with budget < 1,000, 75% were under 72.
# - These are very unusual films. Specially because there generated an average revenue of 9 millions

# In[ ]:


train_df[(train_df.budget>1000)].describe()


# In[ ]:


budget2_ax = train_df.budget[train_df.budget>1000].plot(kind='hist', bins= 100, figsize=(15,3))
budget2_ax.set_title('Histogram: Movies with budget > $1,000')
budget2_ax.xaxis.set_major_formatter(ticker.EngFormatter())


# In[ ]:


# Visualise using a box plot.
train_df.budget.plot(kind='box');


# - A large number of outliner above 100M budget

# ****Repeating the same analysis for the test_df****

# In[ ]:


# Counting how many zero values there are per column in the test_df
(test_df == 0).sum(axis=0)


# - Test_df: 1211 rows have zero budget. 27% of total rows.
# - Can we impute budget values in the test data? I don't think so.
# - Use average of train + test where budget > 0 ? Average by country?
# - **Other people went and found the data from other sources.**

# In[ ]:


test_df.describe()


# In[ ]:


budget3_ax = test_df.budget[test_df.budget>1000].plot(kind='hist', bins= 100, figsize=(15,3))
budget3_ax.set_title('Histogram: Test Movies with budget > $1,000')
budget3_ax.xaxis.set_major_formatter(ticker.EngFormatter())


# In[ ]:


test_df[(test_df.budget>1000)].describe()


# - The 'train' and 'test' dataset have similar distributions for 'budget'.

# **Analysing the relationship between 'budget' and 'revenue'**

# In[ ]:


# Analysing the relationship between 'budget' and 'revenue' 

fig, axe = plt.subplots(figsize=(8, 5))    
cool_chart = sns.scatterplot(ax=axe, data=train_df, x='budget',y='revenue',marker='o',
                             s=100,palette="magma", alpha=0.3)

cool_chart.xaxis.set_major_formatter(ticker.EngFormatter())
cool_chart.yaxis.set_major_formatter(ticker.EngFormatter())
plt.show()


# In[ ]:



fig, axe = plt.subplots(figsize=(8, 5))    
cool_chart = sns.scatterplot(ax=axe,data=train_df[train_df.budget >0],
                             x='budget', y='revenue', marker='o', s=100, palette="magma", alpha=0.3)

# zooming-in but setting limits on x=axis up to 100M and y-axis up to 200M
axe.set(xlim = (0,100000000), ylim = (0, 200000000))

cool_chart.xaxis.set_major_formatter(ticker.EngFormatter())
cool_chart.yaxis.set_major_formatter(ticker.EngFormatter())
plt.show()


# In[ ]:



fig, axe = plt.subplots(figsize=(8, 5))    
cool_chart = sns.scatterplot(ax=axe, data=train_df[train_df.budget >0],x='budget',y='revenue',
                             marker='o',alpha=0.3,s=100,palette="magma")

# zooming-in but setting limits on x=axis up to 100M and y-axis up to 200M
axe.set(xlim = (0,40000000), ylim = (0, 200000000))

cool_chart.xaxis.set_major_formatter(ticker.EngFormatter())
cool_chart.yaxis.set_major_formatter(ticker.EngFormatter())
plt.show()


# - Zooming-in to exclude outliers, we see a large of concentration of films with low budget and low revenue.
# - For budget, there also seems to be a concentration in rounded numbers. Many films have budgets of exactly 15, 20, 25, 30, 35, 40 millions of dollar.
# - budget's seem to be generated in rounded numbers

# These columns will not be analysed
# 
# 7. **original_title**   : Text string
# 8. **overview**         : Text string
# 9. **popularity**       : Rating give after a movie is released
# 10. **poster_path**     : Image

# **11. production company** 
# * Create new columns with production companies(0-n)

# **13. release dates**
# * New columns needed: Year, Month, Day, Day of the week, Quarter (trimester) as proxy to season.

# In[ ]:


train_df.release_date.head()


# In[ ]:


# creating a date function so we can apply it to the train and test datasets

def date_features(df):
# Changing the date column to an apropriate pandas date format
    df['release_date'] = pd.to_datetime(df['release_date'])

# Creating the new columns
    df['release_year'] = df['release_date'].dt.year
    df['release_month'] = df['release_date'].dt.month
    df['release_day'] = df['release_date'].dt.day
    df['release_quarter'] = df['release_date'].dt.quarter
    df['release_dayofweek'] = df['release_date'].dt.dayofweek   # where Mon=0 and Sun=6

    
    
    df.loc[df['release_year'] > 2019,'release_year']=df.loc[df['release_year']>2019,'release_year'].apply(lambda x: x - 100)
    
    
    return df


# In[ ]:


# applying the date function
train_df = date_features(train_df)

# checking if it worked
train_df.sort_values(by='release_year',ascending=False).head(10)[['title','revenue','release_year', 'release_quarter','release_month','release_day','release_dayofweek']]


# In[ ]:


train_df['release_year'].plot(kind='hist', bins= 100, figsize=(10,3))


# In[ ]:


train_df['release_year'].value_counts().sort_values()


# In[ ]:


sns.jointplot(x="release_year", y="revenue", data=train_df, color="g", alpha=0.3);


# In[ ]:


sns.jointplot(x="release_quarter", y="revenue", data=train_df, height=7, ratio=4, color="gray", alpha=0.3)


# In[ ]:


sns.jointplot(x="release_month", y="revenue", data=train_df, height=7, ratio=4, color="r", alpha=0.3)


# In[ ]:


# Visualise using a box plot.
train_df.boxplot(column='revenue', by='release_month', figsize=(9, 6));


# In[ ]:


sns.jointplot(x="release_dayofweek", y="revenue", data=train_df, height=7, ratio=4, color="b", alpha=0.3)


# In[ ]:


# Visualise using a box plot.
train_df.boxplot(column='revenue', by='release_dayofweek', figsize=(9, 6));


# **14. runtime**
# * The other numerical variable is 'runtime'. 
# * Let's analyse it along with 'revenue'.

# In[ ]:


#runtime analysis vs revenue

fig, ax = plt.subplots(figsize=(8, 5))    
palette = sns.color_palette("bright", 6)
cool_chart = sns.scatterplot(ax=ax, data=train_df, x='runtime', y='revenue', marker='o', s=100, palette="magma")
cool_chart.legend(bbox_to_anchor=(1, 1), ncol=1)
#ax.set(xlim = (50000,250000))

cool_chart.xaxis.set_major_formatter(ticker.EngFormatter())
cool_chart.yaxis.set_major_formatter(ticker.EngFormatter())
plt.show()


# * No obvious positive correlation. Most movies seem to be in the 60 - 190 minute runtime range.
# * It does not seem to be a predictor.

# **16. status**

# In[ ]:


train_df['status'].value_counts()


# In[ ]:


test_df['status'].value_counts()


# * The 'status' field contains only 3 values 'Released', 'Rumoured' and 'Post Production'.
# * The great majority of the values are 'Released', hence it won't contribute to the prediction.
# * We can drop this column

# **20. cast** 
# * Create new columns with Cast

# In[ ]:





# **21. crew** 
# * Create new columns with main crew: Director(0-n), Producer(0-n), Executive producer(0-n)

# In[ ]:





# In[ ]:





# **REVENUE PREDICTION**
# 1. ?
# 2. ?

# In[ ]:


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# Don't need to split train into subsets (train and test). Using the train , split using 'cv' folding method

# In[ ]:


# separating the features and getting them ready:
X = train_df.drop(['revenue'])
# separating the variable to predict and getting it ready:
y = train_df['revenue']

# Creating subsets from the data for model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Instantiating a regressor
regressor = LinearRegression()  

# Fitting the model with the train data
regressor.fit(X_train, y_train)

# Making a prediction

# Evaluating the score of the prediction

