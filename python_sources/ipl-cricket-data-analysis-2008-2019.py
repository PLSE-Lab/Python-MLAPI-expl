#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# #### The CRISP-DM method will be applied here to keep track of the analysis process. I learned this method from the Udacity Nanodegree Datascientist Program. 

# ## 1. Business Understanding

# I am a cricket player since my childhood - hence exploring this dataset is an immense fun which might help me to get to know some exciting insights on this game.
# 
# I will focus on the bewlow question below:
# 
# Q1: What is the win percentage of a team batting second at Wankhede Stadium during 2008 to 2019? 
# 
# Q2: Which are the weekdays Kolkata Knight Riders wins the most?
# 
# Q3: Which is the best IPL team and how many times it had won in the past?  
# 
# Q4: Predict the winner of the match by just using the matches dataset fields? - This is a classic binary classification problem
# 
# 
# Let us first import the necessary modules.!
# 
# 

# ### Load packages

# In[ ]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.model_selection import train_test_split

import lightgbm as lgb

from plotly import tools
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected=True)


# ### 2.Data Understanding

# In[ ]:


deliveries_df = pd.read_csv('/kaggle/input/ipldata/deliveries.csv')


# In[ ]:


deliveries_df.head()


# In[ ]:


matches_df = pd.read_csv('/kaggle/input/ipldata/matches.csv')


# In[ ]:


matches_df.head()


# In[ ]:


matches_df.describe()


# In[ ]:


matches_df.info()


# In[ ]:


#Missing vlaues checking
matches_df.isnull().sum()


# In[ ]:


# Get some basic stats on the data
print("Number of matches played so far in IPL : ", matches_df.shape[0])
print("Number of seasons in IPL : ", len(matches_df.season.unique()))
print("Number of Teams participated in IPL : ", len(matches_df.team1.unique()))
print("Number of Teams participated in IPL : ", len(matches_df.team2.unique()))


# #### Total number of matches played in each season

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='season', data=matches_df)
plt.title('The total number of matches played in each year')
plt.show()


# #### Number of matches in each venue

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='venue', data=matches_df)
plt.xticks(rotation='vertical')
plt.show()


# Maximum number of matches played at Eden Gardens. This is one of the famous ground in India - where some histroic matches are played

# #### Which team played maximum number of matches at Eden Gardens

# In[ ]:


df = pd.melt(matches_df, id_vars=['id','season'], value_vars=['team1', 'team2'])
df.head()


# In[ ]:


df.columns = ['id', 'season', 'varaible', 'Team']


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='Team', data=df)
plt.xticks(rotation='vertical')
plt.show()


# Seems like Mumbai Indians and Royal challengers Bangalore are the most matches played teams . Deccan chargers and Sunrisers Hyderabad are from the same City - similarly Delhi Daredevils and Delhi Capitals are also representing the same city
# 

# In[ ]:


eden_df = matches_df[matches_df['venue'] == 'Eden Gardens']


# In[ ]:


eden_df.head()


# In[ ]:


eden_df_1 = pd.melt(eden_df, id_vars=['id','season'], value_vars=['team1', 'team2'])
eden_df_1.head()


# In[ ]:


eden_df_1.columns = ['id', 'season', 'varaible', 'Team']


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='Team', data=eden_df_1)
plt.xticks(rotation='vertical')
plt.show()


# #### It seems obvious that Kolkata played lot many matches at Eden gardens - as this is a home ground for Kolkata

# ### 3.Data Preparation

# There are some necessary stpes to apply before continue exploring the dataset:
# 
# 1) Check for missing values - if available impute them or remove them from analysis
# 
# 2) There might be some teams with different names - we can club them if required
# 

# In[ ]:


matches_df.isnull().sum()


# In[ ]:


deliveries_df.isnull().sum()


# Let us remove the columns that have missing values in deliveries dataset

# In[ ]:


deliveries_df.drop(['player_dismissed', 'dismissal_kind', 'fielder'],axis=1,inplace=True)


# ##### Convert the date format into weekdays

# In[ ]:


matches_df['date'] = pd.to_datetime(matches_df['date'])


# In[ ]:


matches_df["WeekDay"] = matches_df["date"].dt.weekday


# In[ ]:


matches_df.head()


# ### Answer Questions based on dataset

# **Q1: What is the win percentage of a team batting second at Wankhede Stadium during 2008 to 2019?**

# In[ ]:


df = matches_df[(matches_df['toss_decision'] == 'field') &  (matches_df['venue'] == 'Wankhede Stadium') &
             (matches_df['season'] >= 2008) & (matches_df['season'] <= 2019)
             ]


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


print('The win percentage of a team batting second at Wankhede Stadium during 2008 to 2016 is {}%'.format((df[df['win_by_wickets']>0].shape[0])*100/ df.shape[0]))


# In[ ]:


df[(df['win_by_wickets']>0)]['winner'].value_counts()


# In[ ]:


df[df['win_by_wickets']>0]['winner'].value_counts().plot(kind='bar', color='Orange', figsize=(12,6))
plt.xlabel("Team")
plt.ylabel("Count")
plt.title('Top Teams who win batting second are')
plt.show()


# In[ ]:


plt.figure(figsize=(12,6))

plt.title('Top Teams who win batting second are')
sns.countplot(x='winner', data=df[df['win_by_wickets']>0])
plt.xlabel("Team")
plt.ylabel("Count")
plt.xticks(rotation='vertical')
plt.show()


# Q2: Which are the weekdays Kolkata Knight Riders wins the most?

# In[ ]:


df = matches_df[['id', 'WeekDay','winner']]
df = df[df['winner'] == 'Kolkata Knight Riders']


# In[ ]:


df.head()


# In[ ]:


df['WeekDay'].value_counts().plot(kind='bar', color='green', figsize=(12,6))
plt.xlabel("Team")
plt.ylabel("Count")
plt.title('Kolkata Knight Riders winning on weekdays - where Monday is 0 and Sunday is 6')


# KKR wins most of the matches on Wednesday(When compared to only weekdays)

# **Q3: Which is the best IPL team and how many times it had won in the past? **

# Best IPL team can be judged by the number of times it had won. To find which team won in each season - we will first identify the last match played in that year and we assume it should be finals. So Hence collect all the last matches played in each season.

# In[ ]:


df = matches_df.loc[matches_df.groupby('season').date.idxmax()]


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(12,6))

plt.title('Top winning teams in IPL history')
sns.countplot(x='winner', data=df)
plt.xlabel("Team")
plt.ylabel("Count")
plt.xticks(rotation='vertical')
plt.show()


# Mumbai Indians is the team which won maximum number of times - Hence this is the best team

# **Q4:  Predict the winner of the match by just using the matches fields?**

# In this analysis, we are going to look at the matches played only during the latest season 2019. So let us subset the dataset to get only these rows.
# 
# Also some matches are affected by rain and hence Duckworth-Lewis method are used for these matches and so using these matches for training our model might cause some error in our training and so let us neglect those matches as well.

# In[ ]:


# Let us take only the matches played in 2019 for this analysis #
matches_df_2019 = matches_df.ix[matches_df.season==2019,:]
matches_df_2019 = matches_df_2019.ix[matches_df_2019.dl_applied == 0,:]
matches_df_2019.head()


# Okay. Now that we are done with the pre-processing, let us create the variables that are needed for building our model.
# 
# I will be considering only the fields from matches dataframe only.
# 
# so let us start with these variables and I believe this will be a good starting point. As and when required we can add some more variables.

# In[ ]:


train_df = matches_df[matches_df['season'] != 2019]


# In[ ]:


test_df = matches_df[matches_df['season'] == 2019]


# In[ ]:


train_df.columns


# In[ ]:


train_df.head()


# In[ ]:


train_df = train_df[['city',  'team1', 'team2', 'toss_winner',
       'toss_decision', 'result', 'dl_applied', 'winner',  'venue',
        'WeekDay']]
test_df = test_df[['city',  'team1', 'team2', 'toss_winner',
       'toss_decision', 'result', 'dl_applied', 'winner',  'venue',
        'WeekDay']]


# In[ ]:


train_df.head()


# In[ ]:


train_df.isnull().sum()


# In[ ]:


test_df.isnull().sum()


# In[ ]:


train_df= train_df.dropna()
test_df = test_df.dropna()


# In[ ]:


train_df.head()


# In[ ]:


train_df.dtypes


# In[ ]:


test_df.dtypes


# In[ ]:


# Importing LabelEncoder and initializing it
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
# Iterating over all the common columns in train and test
for col in train_df.columns.values:
    # Encoding only categorical variables
    if train_df[col].dtypes=='object':
        print(col)
        # Using whole data to form an exhaustive list of levels
        data=train_df[col].append(test_df[col])
        le.fit(data.values) 
        train_df[col]=le.transform(train_df[col])
        test_df[col]=le.transform(test_df[col])


# In[ ]:


X = train_df.drop(['winner'],axis=1)
y = train_df['winner']

train_X = X
train_y = y

test_X = test_df.drop(['winner'],axis=1)
y_test = test_df['winner']


# ### 4.Modelling

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Split into training and test sets
X_train, X_valid , y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=17)

from sklearn import metrics
model = LogisticRegression()
model.fit(X_train,y_train)
prediction=model.predict(X_valid)
print('The accuracy of the Logistic Regression is', metrics.accuracy_score(prediction,y_valid))


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=500)
rf.fit(X_train, y_train)
y_valid_preds = rf.predict(X_valid)
print("The validation accuracy score is", metrics.accuracy_score(y_valid, y_valid_preds))

print("The test accuarcy score is", metrics.accuracy_score(y_test, rf.predict(test_X)))


# We are able to predict with a good start point accuracy with the randomforest model - but here we need to identify which of the features are important to predict the outcome. For this let us look at the feature importance.

# In[ ]:


coefs_df = pd.DataFrame()

coefs_df['Features'] = X_train.columns
coefs_df['Coefs'] = rf.feature_importances_
coefs_df.sort_values('Coefs', ascending=False).head(10)


# In[ ]:


coefs_df.set_index('Features', inplace=True)
coefs_df.sort_values('Coefs', ascending=False).head(10).plot(kind='bar', color='green', figsize=(12,6))


# From the above important features - if we ignore team2 and team1, the important features are toss_winner and venue.
# 
# So can we assume that if a team wins the toss - is it going to win the match? Let us look at descriptive statistics of the team winning toss and winning the match combination

# In[ ]:


def check_winner(a,b):
    if (a == b):
        return 1
    else:
        return 0

train_df['win_toss_win_match'] = train_df.apply(lambda row: check_winner(row['toss_winner'],row['winner']),axis=1)


# In[ ]:


train_df.head()


# In[ ]:


train_df['win_toss_win_match'].value_counts().plot(kind='bar', color='green')

plt.title('Winning Toss and Winning Match')
plt.xlabel("Team")
plt.ylabel("Count")
plt.xticks(rotation='vertical')
plt.show()


# Clearly from the above plot - we can see that if a team wins toss and wins match combination is high. Hence we proved the same from our model output accuracy.

# Here by just using the matches dataset fields we are able to predict the match outcome with a decent accuracy. We can build a robust model by applying more features from the deliveries dataset.
