#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows',100)


# In[ ]:


fifa_19 = pd.read_csv('../input/data.csv')


# In[ ]:


fifa_19.head()


# In[ ]:


fifa_19.drop('Unnamed: 0',axis=1,inplace=True)


# In[ ]:


fifa_19.shape


# In[ ]:


fifa_19.columns


# In[ ]:


fifa_19.dtypes


# ## Exploratory Data Analysis

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


plt.hist(fifa_19['Age'])
plt.xlabel("Players Age")
plt.ylabel("Number of Players")
plt.plot()


# In[ ]:


import seaborn as sns


# In[ ]:


g = sns.distplot(fifa_19['Age'],kde=False,rug = False)


# ## Let us look at how age varies with overall

# In[ ]:


plt.figure(figsize=(20,15))
sns.regplot(fifa_19['Age'],fifa_19['Overall'])
plt.title('Age vs Overall rating')
plt.show()


# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(fifa_19.corr(),linewidths=6)
plt.title('Correlation heatmap')
plt.show()


# In[ ]:


fifa_19 = fifa_19[['ID', 'Name', 'Age','Nationality','Overall',
                  'Potential', 'Club','Value', 'Wage', 'Special','Preferred Foot',
                'International Reputation', 'Weak Foot','Skill Moves', 'Work Rate',
                'Body Type','Position','Joined', 'Loaned From', 'Contract Valid Until',
               'Height', 'Weight', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
               'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
               'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Crossing',
               'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
               'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
               'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
               'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
               'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
               'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling',
               'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause']]


# In[ ]:


fifa_19.head(100)


# ## Seperate out Goal keepers from the outfield players
# <li> The metrics for these players is extremely different

# In[ ]:


goal_keepers = fifa_19.loc[fifa_19['Position']=='GK']


# ## Make sure we have only goalkeepers in the data frame
# 
# <li> very important to check the tail of goalkeepers dataframe 

# In[ ]:


goal_keepers.head()


# In[ ]:


goal_keepers.tail()


# ## Remove features that are not predictive for goalkeepers

# In[ ]:


goal_keepers.columns


# In[ ]:


goal_keepers = goal_keepers[['ID', 'Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club',
       'Value', 'Wage', 'Special', 'Preferred Foot',
       'International Reputation', 'Weak Foot','Body Type','Height', 'Weight',
       'ShortPassing','Strength','GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes',
       'Release Clause']]


# In[ ]:


outfield = fifa_19[fifa_19['Position'] != 'GK']


# In[ ]:


outfield.columns


# In[ ]:


outfield = outfield[['ID', 'Name', 'Age', 'Nationality', 'Overall', 'Potential', 'Club',
       'Value', 'Wage', 'Special', 'Preferred Foot',
       'International Reputation', 'Weak Foot', 'Skill Moves', 'Work Rate',
       'Body Type', 'Position', 'Joined', 'Loaned From',
       'Contract Valid Until', 'Height', 'Weight', 'LS', 'ST', 'RS', 'LW',
       'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM',
       'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB',
       'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',
       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',
       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',
       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',
       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',
       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle','Release Clause']]


# In[ ]:


outfield['Position'].value_counts()


# ## Let us first analyse outfield players

# In[ ]:


def split_columns(position):
    outfield[position + '_orig'], outfield[position+ '_improv'] = outfield[position].str.split('+',1).str
    outfield[position + '_orig'] = outfield[position + '_orig'].astype(float)
    outfield[position+ '_improv'] = outfield[position+ '_improv'].astype(float)
    outfield.drop(position,axis = 1,inplace=True)


# In[ ]:


positions = ['LS', 'ST', 'RS', 'LW','LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM','RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
for position in positions:
    split_columns(position)

## drop positions too
outfield.drop('Position',axis = 1,inplace= True)


# In[ ]:


## Check for the presence of NA's in the data
outfield.isnull().sum()


# ## It is very important to look at missing data
# #### Let us first look at missing data by storing these columns in a seperate dataframe
# #####  However it is very important to note that the missng values discussed below are missing-not-at-random and can be useful for other form of analysis
# 
# <ol>
# <li> Club column can be dropped as this might be due to player being a free agent</li>
# <li> Joined column can be dropped as it is not predictive in our analysis</li>
# <li> Loaned From can be dropped as the players may not be on loan</li>
# <li> Contract Valid Until can be dropped as these might not be predictive</li>
# <li> Release Clause can be dropped too as some players may not have a release clause or is not publicly available. Multiple 
#     Imputing can be done, however its usefulness is not known</li>
# </ol>
# 

# In[ ]:


outfield.drop(['Club','Joined','Loaned From','Contract Valid Until','Release Clause','Height','Weight'],axis =1 ,inplace=True)


# In[ ]:


null_data = outfield[outfield.isnull().any(axis=1)]


# In[ ]:


null_data.shape


# In[ ]:


null_data


# ### Here we can see that these players are valued extremely low and would not be of no economic interest
# #### Hence drop the rows with NA's

# In[ ]:


outfield = outfield.dropna()


# In[ ]:


outfield.shape


# In[ ]:


outfield.dtypes


# In[ ]:


outfield['Rating'] = outfield[['LS_orig','ST_orig','RS_orig','LW_orig','LF_orig','CF_orig','RF_orig','RW_orig','LAM_orig','CAM_orig','RAM_orig','LM_orig','LCM_orig','CM_orig','RCM_orig','RM_orig','LWB_orig','LDM_orig','CDM_orig','RDM_orig','RWB_orig','LB_orig','LCB_orig','CB_orig','RCB_orig','RB_orig']].max(axis = 1)
outfield.drop(['LS_orig','ST_orig','RS_orig','LW_orig','LF_orig','CF_orig','RF_orig','RW_orig','LAM_orig','CAM_orig','RAM_orig','LM_orig','LCM_orig','CM_orig','RCM_orig','RM_orig','LWB_orig','LDM_orig','CDM_orig','RDM_orig','RWB_orig','LB_orig','LCB_orig','CB_orig','RCB_orig','RB_orig'],axis = 1,inplace = True)


# In[ ]:


outfield['Rating_improv'] = outfield[['LS_improv','ST_improv','RS_improv','LW_improv','LF_improv','CF_improv','RF_improv','RW_improv','LAM_improv','CAM_improv','RAM_improv','LM_improv','LCM_improv','CM_improv','RCM_improv','RM_improv','LWB_improv','LDM_improv','CDM_improv','RDM_improv','RWB_improv','LB_improv','LCB_improv','CB_improv','RCB_improv','RB_improv']].max(axis = 1)
outfield.drop(['LS_improv','ST_improv','RS_improv','LW_improv','LF_improv','CF_improv','RF_improv','RW_improv','LAM_improv','CAM_improv','RAM_improv','LM_improv','LCM_improv','CM_improv','RCM_improv','RM_improv','LWB_improv','LDM_improv','CDM_improv','RDM_improv','RWB_improv','LB_improv','LCB_improv','CB_improv','RCB_improv','RB_improv'],axis = 1,inplace = True)


# In[ ]:


outfield.shape


# In[ ]:


outfield['Work Rate'].value_counts().sort_values(ascending = True)


# In[ ]:


outfield['Body Type'].value_counts()


# ### Let us drop body type too

# In[ ]:


## Dropping body types as this variable is not predictive for potential
outfield.drop('Body Type', axis = 1, inplace=True)


# In[ ]:


outfield.shape


# In[ ]:


outfield.head()


# In[ ]:


outfield.dtypes


# In[ ]:


outfield['Preferred Foot'] = outfield['Preferred Foot'].astype('category')


# In[ ]:


outfield['Work Rate'] = outfield['Work Rate'].astype("category",ordered = True,categories = ['Low/ Low','Low/ Medium','Low/ High','Medium/ Low','Medium/ Medium','Medium/ High','High/ Low','High/ Medium','High/ High']).cat.codes


# In[ ]:


outfield.head()


# In[ ]:


outfield = pd.get_dummies(outfield,columns = ['Preferred Foot'])


# In[ ]:


outfield.columns


# In[ ]:


## Only one preferred foot needs to be taken
X = outfield[['Age','Special', 'Preferred Foot_Left', 'International Reputation','Weak Foot', 'Skill Moves', 'Work Rate', 'Crossing', 'Finishing',
       'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle', 'Rating',
       'Rating_improv']]


# In[ ]:


X.shape


# In[ ]:


Y = outfield['Potential']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33, random_state = 2)


# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import math


# In[ ]:


learning_rate = [0.01,0.03,0.05,0.07,0.09]
min_samples_split = [2,3,4]
min_samples_leaf = [1,2]
max_depth = [2,3,4]
    
## Grid Search for hyper parameter tuning
mse_min = 9999
lr_min = 0.01
min_samples_split_min = 2
min_samples_leaf_min = 1
max_depth_min = 2
for lr in learning_rate:
    for mss in min_samples_split:
        for msl in min_samples_leaf:
            for md in max_depth:
                params = {'min_samples_split':mss,'learning_rate': lr, 'verbose':0,'max_depth':md,'min_samples_leaf':msl}
                clf = GradientBoostingRegressor(**params)
                clf.fit(X_train, Y_train)
                mse = mean_squared_error(Y_test, clf.predict(X_test))
                    
                if mse < mse_min:
                    mse_min = mse 
                    lr_min = lr
                    min_samples_split_min = mss
                    min_samples_leaf_min = msl
                    max_depth_min = md
                        
print("Best MSE is ",mse_min,"for ",lr_min,min_samples_split_min,min_samples_leaf_min,max_depth_min)


# In[ ]:


params = {'min_samples_split':min_samples_split_min,'learning_rate': lr_min, 'verbose':0,'max_depth':max_depth_min,'min_samples_leaf':min_samples_leaf_min}
clf = GradientBoostingRegressor(**params)
clf.fit(X_train, Y_train)    


# In[ ]:


predicted_potential = clf.predict(X_test)


# In[ ]:


X_test['Potential'] = Y_test
X_test['Predicted_Potential'] = predicted_potential


# In[ ]:


X_test['Predicted_Potential'] = round(X_test['Predicted_Potential'])


# In[ ]:


X_test


# In[ ]:


import numpy as np


# In[ ]:


## Let us look at which features were important for our prediction
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

## Plotting the relative importance of variables
fig = plt.gcf()
fig.set_size_inches(12.5,8.5)
plt.plot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[ ]:


## We see that rating is a very important parameter for predicting potential. Let us see if removing this has a significant
## impact on predicting potential

X = outfield[['Age','Special', 'Preferred Foot_Left', 'International Reputation','Weak Foot', 'Skill Moves', 'Work Rate', 'Crossing', 'Finishing',
       'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve',
       'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle',
       'Rating_improv']]


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33, random_state = 2)


# In[ ]:


learning_rate = [0.01,0.03,0.05,0.07,0.09]
min_samples_split = [2,3,4]
min_samples_leaf = [1,2]
max_depth = [2,3,4]
    
## Grid Search for hyper parameter tuning
mse_min = 9999
lr_min = 0.01
min_samples_split_min = 2
min_samples_leaf_min = 1
max_depth_min = 2
for lr in learning_rate:
    for mss in min_samples_split:
        for msl in min_samples_leaf:
            for md in max_depth:
                params = {'min_samples_split':mss,'learning_rate': lr, 'verbose':0,'max_depth':md,'min_samples_leaf':msl}
                clf = GradientBoostingRegressor(**params)
                clf.fit(X_train, Y_train)
                mse = mean_squared_error(Y_test, clf.predict(X_test))
                    
                if mse < mse_min:
                    mse_min = mse 
                    lr_min = lr
                    min_samples_split_min = mss
                    min_samples_leaf_min = msl
                    max_depth_min = md
                        
print("Best MSE is ",mse_min,"for ",lr_min,min_samples_split_min,min_samples_leaf_min,max_depth_min)


# In[ ]:


params = {'min_samples_split':min_samples_split_min,'learning_rate': lr_min, 'verbose':0,'max_depth':max_depth_min,'min_samples_leaf':min_samples_leaf_min}
clf = GradientBoostingRegressor(**params)
clf.fit(X_train, Y_train)


# In[ ]:


predicted_potential = clf.predict(X_test)


# In[ ]:


X_test['Predicted_Potential'] = predicted_potential
X_test['Predicted_Potential'] = round(X_test['Predicted_Potential'])
X_test['Potential'] = Y_test


# In[ ]:


X_test


# ### We see that the MSE increases on removing rating from the predictor variables
# ### Depending on the problem on hand we can choose one of the above models

# In[ ]:


goal_keepers = pd.get_dummies(goal_keepers,columns=['Preferred Foot'])


# In[ ]:


goal_keepers.drop('Preferred Foot_Left',axis = 1, inplace =True)


# In[ ]:


## Let us take into only the relevant features for goalkeepers
X = goal_keepers[['Age','Special', 'Preferred Foot_Right',
       'International Reputation', 'Weak Foot',
       'ShortPassing','Strength','GKDiving',
       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]


# In[ ]:


Y = goal_keepers['Potential']


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.33,random_state = 2)


# In[ ]:


learning_rate = [0.01,0.03,0.05,0.07,0.09]
min_samples_split = [2,3,4]
min_samples_leaf = [1,2]
max_depth = [2,3,4]
loss = ['ls','huber']
    
## Grid Search for hyper parameter tuning
mse_min = 9999
lr_min = 0.01
min_samples_split_min = 2
min_samples_leaf_min = 1
max_depth_min = 2
for lr in learning_rate:
    for mss in min_samples_split:
        for msl in min_samples_leaf:
            for md in max_depth:
                for ls in loss:
                    params = {'min_samples_split':mss,'learning_rate': lr, 'verbose':0,'max_depth':md,'min_samples_leaf':msl,'loss':ls}
                    clf = GradientBoostingRegressor(**params)
                    clf.fit(X_train, Y_train)
                    mse = mean_squared_error(Y_test, clf.predict(X_test))
                    
                    if mse < mse_min:
                        mse_min = mse 
                        lr_min = lr
                        min_samples_split_min = mss
                        min_samples_leaf_min = msl
                        max_depth_min = md
                        loss_min = ls
                        
print("Best MSE is ",mse_min,"for ",lr_min,min_samples_split_min,min_samples_leaf_min,max_depth_min,loss_min)


# In[ ]:


params = {'min_samples_split':min_samples_split_min,'learning_rate': lr_min, 'verbose':0,'max_depth':max_depth_min,'min_samples_leaf':min_samples_leaf_min,'loss':loss_min}
clf = GradientBoostingRegressor(**params)
clf.fit(X_train, Y_train)


# In[ ]:


predicted_potential = clf.predict(X_test)


# In[ ]:


X_test['Predicted_Potential'] = predicted_potential
X_test['Predicted_Potential'] = round(X_test['Predicted_Potential'])
X_test['Potential'] = Y_test


# In[ ]:


X_test


# ### Let us look at the features that were important in our prediction

# In[ ]:


## Let us look at which features were important for our prediction
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5

## Plotting the relative importance of variables
fig = plt.gcf()
fig.set_size_inches(12.5,8.5)
plt.plot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, X_train.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# ### Just like in the case of outfield players, the age of a player plays a significant role in their potential
# 
# <ol>
#     <li> GKPositioning was very important in his potential
#     <li> GKHandling is the next important variable in predicting potential 
# </ol>
# 
# #### We can see that predicted features shows which features are extremely important and agree with our initial hypothesis. 
# #### The same is true for outfield players too
