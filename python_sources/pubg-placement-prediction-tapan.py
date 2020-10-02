#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display
from fastai.imports import *
import xgboost as xgb
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
#setting display height,max_rows,max_columns and width to desired
pd.set_option('display.max_rows',500)
pd.set_option('display.max_columns',500)


# In[ ]:


# function to print Mean Absolute Error
def rmse(X,Y,t,m: RandomForestRegressor):
    print (" Mean Absolute Error in {} is {}".format(t,mean_absolute_error(m.predict(X),Y)))
#     res = ['mae train: ',mean_absolute_error(m.predict(X_train),Y_train),'mae value: ',mean_absolute_error(m.predict(X_test),Y_test)]
#     if hasattr(m,'oooob score '):
#         res.append(m.oob_score_)
#     print(res)
def rootmse(Y_pred,Y_test):
    print(" Mean Absolute Error is {}".format(mean_absolute_error(Y_pred,Y_test)))


# In[ ]:


train  = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
print ("Train Head -->")
display(train.head())
print ("Test Head -->")
display(test.head())


# In[ ]:


print (train.shape)


# In[ ]:


print (test.shape)


# In[ ]:


train.info()


# In[ ]:


train.isna().sum()


# In[ ]:


train[train['winPlacePerc'].isnull()]


# In[ ]:


#lets drop that NaN entry
train.drop(2744604,inplace = True)


# In[ ]:


test.isna().sum()


# In[ ]:


print ("Longest Kill Recored {} Average Kill Distance {}".format(train['longestKill'].max(),train['longestKill'].mean()))
print ("Max Assists Recorded {} Average Assists {}".format(train['assists'].max(),train['assists'].mean()))
print ("Max Boost Items used {} Average Boost Items Used {}".format(train['boosts'].max(),train['boosts'].mean()))
print ("Maximum DamageDealt  {} Average Damage Dealt {}".format(train['damageDealt'].max(),train['damageDealt'].mean()))
print ("Max Boost Items used {} Average Boost Items Used {}".format(train['boosts'].max(),train['boosts'].mean()))
print ("Max Heal Items used {} Average Heal Items Used {}".format(train['heals'].max(),train['heals'].mean()))
print ("Longest Kill Streak {} Average Kill Streak Used {}".format(train['killStreaks'].max(),train['killStreaks'].mean()))
print ("Maximum Kills {} Average Kills {}".format(train['kills'].max(),train['kills'].mean()))
print ("Maximum Revives {} Average Revives {}".format(train['revives'].max(),train['revives'].mean()))
print ("Maximum Team Kills {} Average Team Kills {}".format(train['teamKills'].max(),train['teamKills'].mean()))
print ("Maximum vehicleDestroys {} Average vehicleDestroys {}".format(train['vehicleDestroys'].max(),train['vehicleDestroys'].mean()))


# **Check Number of Players Joined in Game**

# In[ ]:


train['playerJoined'] = train.groupby('matchId')['matchId'].transform('count')
plt.figure(figsize = (15,10))
sns.countplot(train[train['playerJoined']>=60]['playerJoined'])
plt.title("Players Joined")
plt.show()
plt.figure(figsize = (15,10))
sns.countplot(train[train['playerJoined']<=60]['playerJoined'])
plt.title("Players Joined")
plt.show()


# In[ ]:


# modifing test dataset now
test['playerJoined'] = test.groupby('matchId')['matchId'].transform('count')
test.drop(test[test['playerJoined']<=50].index, inplace = True)
test.shape


# In[ ]:


# I think matches with less than 50 are not worth considering
# so gonna drop those rows
train.drop(train[train['playerJoined']<=50].index,inplace = True)
train.shape


# **Number of Player in Group**

# In[ ]:


train['playersInGroup'] = train.groupby('groupId')['groupId'].transform('count')
plt.figure(figsize=(15,10))
sns.countplot(train[train['playersInGroup']>=0]['playersInGroup'])
plt.title("Number of Players in Single Group")
plt.show()
train[train['playersInGroup']>4].shape


# In[ ]:


# test set
test['playersInGroup'] = test.groupby('groupId')['groupId'].transform('count')
test.drop(test[test['playersInGroup']>4].index, inplace = True)


# In[ ]:


#Groups with players greater than 4 are not valid
# as in PUBG max size of Group is 4 so we remove them
train.drop(train[train['playersInGroup']>4].index, inplace = True)


# **Killing**

# In[ ]:


#lets find some interesting things from data
print ('Max Kills Recored {} Average Kills person kills {} while 99% people kills {}'.format(train['kills'].max(),train['kills'].mean(),train['kills'].quantile(0.99)))
# 72 kills seems suspicious lets Plot


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(train[train['kills']>=1]['kills'])
plt.title("Number of Kills")
plt.show()


# In[ ]:


#lets check killing with winPlacePerc
# plt.figure(figsize = (15,10))
sns.jointplot(x="winPlacePerc",y="kills",data=train,height=10,ratio=3)
plt.title("WinplacePerc vs Number of Kills")
plt.show()


# There seems to be relation between kills and winplaceperc . the more the number of kills more the winplacePerc

# In[ ]:


#Team kills cannot be 4 or more so we have to remove this
plt.figure(figsize=(15,10))
sns.countplot(train[train['teamKills']>=4]['teamKills'])
plt.title("TeamMate Kills")
plt.show()
train[train['teamKills']>=4].shape


# In[ ]:


#removing teamKills outliers
train.drop(train[train['teamKills']>=4].index, inplace = True)
test.drop(test[test['teamKills']>=4].index, inplace = True)


# In[ ]:


print("Max number of HeadShots by Single Person {} Average Headshots {} While 99% percent people {} ".format(train['headshotKills'].max(),train['headshotKills'].mean(),train['headshotKills'].quantile(0.99)))
### remove  outlier headshots ###


# **Match Duration**

# In[ ]:


######### has to do something with MatchDuration for match duration with less than 5min to 10 minutes######
# train['check'] = train[train['matchDuration']<600]
plt.figure(figsize=(15,10))
sns.countplot(train[train['matchDuration']<600]['matchDuration'])
plt.title("Match With Duration less tha 10 Minutes")
plt.show()
train[train['matchDuration']<600].shape


# In[ ]:


#we will drop the rows with match Duration less than 10 minutes


# In[ ]:


print ("Unique id counts {} while data shape {}".format(train['Id'].nunique(),train.shape))


# In[ ]:


print("Max Number of Weapons acquired by individual {} Average Number of Weapons Acquired {} while 99% percentile {}".format(train['weaponsAcquired'].max(),train['weaponsAcquired'].mean(),train['weaponsAcquired'].quantile(0.99)))


# In[ ]:


#236 weapons acquired by an individual is of course an outlier
#lets find outliers using weapons acquired
plt.figure(figsize=(15,10))
sns.countplot(train[train['weaponsAcquired']>50]['weaponsAcquired'])
plt.show()
train[train['weaponsAcquired']>50].shape


# In[ ]:


#we will remove rows with weapons acquired more than 40 as they seems suspicious
train.drop(train[train['weaponsAcquired']>50].index, inplace = True)
test.drop(test[test['weaponsAcquired']>50].index, inplace = True)
#lets plot WeaponsAcquired vs winPlacePerc
# plt.figure(figsize=(15,10))
sns.jointplot(x="winPlacePerc",y="weaponsAcquired",data=train,height=10,ratio=3,color="blue")
plt.title("WinPlacePerc vs WeaponsAcquired Realtion")
plt.show()


# In[ ]:


# train.sort_values(by = ['groupId']).head()


# Lets Check If There are no entries with MatchType solo but has assists and Players Knocked

# In[ ]:


# train['checkSolo'] = [1 if (x == "solo" or x == "solo-fpp") else 0 for x in train['matchType']]
train['assistsCheck'] = ["false" if ((x == "solo" or x == "solo-fpp") and y!=0) else "true" for x,y in zip(train['matchType'],train['assists'])]
print ("Number of assists in Solo :",train[train['assistsCheck']=="false"].shape)
train['DBNOCheck'] = ["false" if ((x == "solo" or x == "solo-fpp") and y!=0) else "true" for x,y in zip(train['matchType'],train['DBNOs'])]
print ("Number of Knocks in Solos",train[train['DBNOCheck']=="false"].shape)


# In[ ]:


# train['checkSolo'] = [1 if (x == "solo" or x == "solo-fpp") else 0 for x in train['matchType']]
test['assistsCheck'] = ["false" if ((x == "solo" or x == "solo-fpp") and y!=0) else "true" for x,y in zip(test['matchType'],test['assists'])]
print ("Number of assists in Solo :",test[test['assistsCheck']=="false"].shape)
test['DBNOCheck'] = ["false" if ((x == "solo" or x == "solo-fpp") and y!=0) else "true" for x,y in zip(test['matchType'],test['DBNOs'])]
print ("Number of Knocks in Solos",test[test['DBNOCheck']=="false"].shape)


# In[ ]:


test.drop(test[test['assistsCheck']== "false"].index, inplace = True)


# There are 38848 assists in solo that can't be possible as there are no teammates in solo.

# In[ ]:


#lets remove these outliers from dataset
train.drop(train[train['assistsCheck']=="false"].index, inplace = True)


# Let's Check WinplacePerc with Vehicle Destroys

# In[ ]:


# #plot winPlacePerc with Vehicle Destroyed
# plt.figure(figsize = (15,10))
# sns.countplot(train[train['vehicleDestroys']>0]['vehicleDestroys'])
# plt.title("Vehicle Destroyed")
# plt.show()
# # plt.figure(figsize = (15,10))
# sns.jointplot(x="winPlacePerc",y="vehicleDestroys",data=train,height=10,ratio=3,color="lime")
# plt.title("Vehicle Destroyed jointplot")
# plt.show()


# Let's Look for Players with Kills but has not travelled a bit in the match and Players won the match with no distance Travelled

# In[ ]:


print ("Maximum walk Distance Tracelled {} Average walk Distance Travelled {}".format(train['walkDistance'].max(),train['walkDistance'].mean()))
print ("Maximum ride Distance Tracelled {} Average ride Distance Travelled {}".format(train['rideDistance'].max(),train['rideDistance'].mean()))
print ("Maximum swim Distance Tracelled {} Average swim Distance Travelled {}".format(train['swimDistance'].max(),train['swimDistance'].mean()))


# In[ ]:


print ("Maximum Total Distance Travelled by a Person in Single Match {} Average Total Distance Travelled {}".format((train['walkDistance']+train['swimDistance']+train['rideDistance']).max(),(train['walkDistance']+train['swimDistance']+train['rideDistance']).mean()))


# Let's Plot the distance travelled

# In[ ]:


plt.figure(figsize = (15,10))
sns.countplot(train[train['walkDistance']>5000]['walkDistance'])
plt.title("Distance Covered by Foot")
plt.show()


# In[ ]:


plt.figure(figsize = (15,10))
sns.countplot(train[train['rideDistance']>1200]['rideDistance'])
plt.title("Distance Covered by Ride")
plt.show()


# In[ ]:


plt.figure(figsize = (15,10))
sns.countplot(train[train['swimDistance']>1200]['swimDistance'])
plt.title("Distance Covered by Swimming")
plt.show()


# In[ ]:


#lets find jointplot for WinPlacePerc vs all distances
#first Walk Distance
sns.jointplot(x="winPlacePerc",y="walkDistance",data=train,height=10,ratio=3)
plt.show()


# In[ ]:


#now check ride distance
sns.jointplot(x="winPlacePerc",y="rideDistance",data=train,height=10,ratio=3,color="black")
plt.show()


# In[ ]:


#now check swim distance
sns.jointplot(x="winPlacePerc",y="swimDistance",data=train,height=10,ratio=3,color="pink")
plt.show()


# In[ ]:


#at last total distancce
train['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']
sns.jointplot(x="winPlacePerc",y="totalDistance",data=train,height=10,ratio=3,color="green")
plt.show()


# In[ ]:


#at last total distancce
test['totalDistance'] = test['walkDistance'] + test['rideDistance'] + test['swimDistance']
sns.jointplot(x="winPlacePerc",y="totalDistance",data=test,height=10,ratio=3,color="green")
plt.show()


# from all those distance plot we can observe more you travel the higher is your WinPlacePerc i.e winning chances increaase the more you travel

# Let's check people with Total Distance travelled 0 and weapons Acquired and Kills for outliers in the data

# In[ ]:


df = train.copy()
df = df[(df['totalDistance']==0) & (df['weaponsAcquired']>=4)]
print ("{} peoples cheated who donot move a bit but acquired weapons ".format(df['Id'].count()))
# df.sort_values(by=['groupId']).head()
train[train['groupId']=="082950bbdd1d97"].head()


# These data are no outliers as there teammates increased thier winPlacePerc not cheated

# In[ ]:


# df = train.copy()
# df = df[(df['totalDistance']==0) & (df['kills']!=0)]
# df.shape
# df.head()
df = train.copy()
df = df[(df['totalDistance']==0) & (df['kills']!=0)]
print ("{} peoples cheated who donot move a bit but acquired weapons ".format(df['Id'].count()))
df.sort_values(by=['groupId']).head()


# In[ ]:


#lets see an entry and observe
train[train['groupId']=="0000a9f58703c5"].head()


# In[ ]:


plt.figure(figsize=(15,10))
sns.countplot(train[train['numGroups']>1]['numGroups'])
plt.title("Number of Groups")
plt.show()


# Let's Change Categorical Data into One-Hot Encoding

# In[ ]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


# In[ ]:


## not ajidosaid iahsf
# train['matchType'] = [1 if match =="solos" 2 elif match =="duos" else 3 for match in train['matchType']]
# df.loc[df.set_of_numbers <= 4, 'equal_or_lower_than_4?'] = 'True' 
# df.loc[df.set_of_numbers > 4, 'equal_or_lower_than_4?'] = 'False' 
# train.loc[train.matchType == "solo",'matchType'] = 1
# train.loc[train.matchType == "duo",'matchType'] = 2
# train.loc[train.matchType == "squad",'matchType'] = 3
# train.loc[train.matchType == "solo-fpp",'matchType'] = 4
# train.loc[train.matchType == "duo-fpp",'matchType'] = 5
# train.loc[train.matchType == "squad-fpp",'matchType'] = 6
train.loc[(train.matchType != "solo") & (train.matchType != "duo") & (train.matchType != "squad") & (train.matchType != "solo-fpp") & (train.matchType != "duo-fpp") & (train.matchType != "squad-fpp"),'matchType'] = "others"
train['matchType'] = LabelEncoder().fit_transform(train['matchType'])
train.head()


# In[ ]:


test.loc[(test.matchType != "solo") & (test.matchType != "duo") & (test.matchType != "squad") & (test.matchType != "solo-fpp") & (test.matchType != "duo-fpp") & (test.matchType != "squad-fpp"),'matchType'] = "others"
test['matchType'] = LabelEncoder().fit_transform(test['matchType'])
test.head()


# In[ ]:


#drop unneccesary columns from dataset
train = train.drop(['assistsCheck','DBNOCheck'], axis=1)
train.head()


# In[ ]:


#drop unneccesary columns from dataset
test = test.drop(['assistsCheck','DBNOCheck'], axis=1)
test.head()


# In[ ]:


#Let's turn groupId and matchId in category values
train['groupId'] = train['groupId'].astype('category')
train['matchId'] = train['matchId'].astype('category')

#category codinf for groupId and matchId
train['groupId'] = train['groupId'].cat.codes
train['matchId'] = train['matchId'].cat.codes

train.head()


# In[ ]:


#Let's turn groupId and matchId in category values
test['groupId'] = test['groupId'].astype('category')
test['matchId'] = test['matchId'].astype('category')

#category codinf for groupId and matchId
test['groupId'] = test['groupId'].cat.codes
test['matchId'] = test['matchId'].cat.codes

test.head()


# In[ ]:


#Let's finally drop Id Column and do some Machine Learning
train = train.drop(['Id'],axis = 1)
test = test.drop(['Id'],axis = 1)
test.shape


# In[ ]:


# train.head()
train.info()


# In[ ]:


# #Let's split the dataset into Training and cross validation set
# train,test = train_test_split(train,test_size=0.3)
# X_train = train.copy()
# X_train = X_train.drop(['winPlacePerc'], axis = 1)
# Y_train = train['winPlacePerc']
# X_test = test.copy()
# X_test = X_test.drop(['winPlacePerc'], axis = 1)
# Y_test = test['winPlacePerc']


# In[ ]:


# final prediction
Y_train = train['winPlacePerc']
X_train = train.drop(['winPlacePerc'], axis = 1)


# In[ ]:


print ("Training Data Shape {} and Test Data Shape {} ".format(train.shape,test.shape))
print ("Training Data Shape {} and Test Data Shape {} ".format(X_train.shape,Y_train.shape))


# In[ ]:


m1 = RandomForestRegressor(n_estimators = 40, min_samples_leaf = 3, max_features = 'sqrt', n_jobs = -1)
m1.fit(X_train,Y_train)
print ("Random Joke")
precdiction_m1 = m1.predict(test)
# rmse(X_train,Y_train,"Train",m1)
# rmse(X_test,Y_test,"Test",m1)


# In[ ]:





# In[ ]:


m2 = RandomForestRegressor(n_estimators = 70, min_samples_leaf = 5, max_features = 0.5, n_jobs = -1)
m2.fit(X_train,Y_train)
print ("Random Joke 2")
precdiction_m2 = m2.predict(test)
# rmse(X_train,Y_train,"Train",m2)
# rmse(X_test,Y_test,"Test",m2)


# In[ ]:


# m3 = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 5, max_features = 0.5, n_jobs = -1)
# m3.fit(X_train,Y_train)
# rmse(X_train,Y_train,"Train",m3)
# rmse(X_test,Y_test,"Test",m3)


# In[ ]:


# # #XGboost for regression
# regr = xgb.XGBRegrssor(colsample_bytree=0.2,gamm=0.0,learning_rate=0.01,max_depth=4,min_child_weight=1.5,n_estimators=7200,reg_alpha=0.9,reg_lambda=0.6,subsample=0.2,seed=42,silent=1)
# regr.fit(X_train,Y_train)
# Y_pred = regr.predict(X_test)
# rootmse(Y_pred,Y_test)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


# #let's do hyperParameter Tuning
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# print(random_grid)


# In[ ]:


# ##########has to do one hot encoding############
# #let's try new Setting
# # Use the random grid to search for best hyperparameters
# # First create the base model to tune
# rf = RandomForestRegressor()
# # Random search of parameters, using 3 fold cross validation, 
# # search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# # Fit the random search model
# rf_random.fit(X_train,Y_train)


# In[ ]:


# #Let's check best Random Parameters
# rf_random.best_params_


# In[ ]:


# #Let's Compare Accuracy of Different Models
# def evaluate(model, test_features, test_labels):
#     predictions = model.predict(test_features)
#     errors = abs(predictions - test_labels)
#     mape = 100 * np.mean(errors / test_labels)
#     accuracy = 100 - mape
#     print('Model Performance')
#     print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
#     print('Accuracy = {:0.2f}%.'.format(accuracy))
    
#     return accuracy


# In[ ]:


#Base Model given
base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(X_train,Y_train)
prediction_base_model = base_model.predict(test)
# base_accuracy_given = evaluate(base_model, X_test, Y_test)


# In[ ]:


# #Model Created
# base_accuracy_m1 = evaluate(m1,X_test,Y_test)
# base_accuracy_m2 = evaluate(m2,X_test,Y_test)
# base_accuracy_m3 = evaluate(m3,X_test,Y_test)
# # base_accuracy_regr = evaluate(regr,X_test,Y_test)


# In[ ]:


# #radnom Best Estimator
# train_features = X_train
# train_labels = Y_train
# test_features = X_test
# test_labels = Y_test
# # best_random = rf_random.best_estimator_
# # random_accuracy = evaluate(best_random, test_features, test_labels)
# # print (" Hyper Random Accracy {} \n m1 Accuracy {} \n m2 Accuracy {} \n m3 Accuracy {} \n xGBoost {} \n best_random {}".format(base_accuracy_given,base_accuracy_m1,base_accuracy_m2,base_accuracy_m3,base_accuracy_regr,random_accuracy))


# In[ ]:


# #GridSearch Model
# from sklearn.model_selection import GridSearchCV
# # Create the parameter grid based on the results of random search 
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300, 1000]
# }
# # Create a based model
# rf = RandomForestRegressor()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2)


# In[ ]:


# # Fit the grid search to the data
# grid_search.fit(train_features, train_labels)
# grid_search.best_params_
# #chagne into one hot encoding


# In[ ]:


# best_grid = grid_search.best_estimator_
# grid_accuracy = evaluate(best_grid, X_test, Y_test)
# # print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy_given) / base_accuracy_given))


# The models are as follows:
# 
# 1. average: original baseline computed by predicting historical average max temperature for each day in test set
# 2. one_year: model trained using a single year of data
# 3. four_years_all: model trained using 4.5 years of data and expanded features (see Part One for details)
# 4. four_years_red: model trained using 4.5 years of data and subset of most important features
# 5. best_random: best model from random search with cross validation
# 6. first_grid: best model from first grid search with cross validation (selected as the final model)
# 7. second_grid: best model from second grid search

# In[ ]:




