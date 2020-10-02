#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
from pylab import rcParams
rcParams['figure.figsize'] = 25, 10
warnings.filterwarnings("ignore")


# In[ ]:


train=pd.read_csv('../input/train_V2.csv')


# In[ ]:


test=pd.read_csv('../input/test_V2.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


print("Shape of train dataset {}; Shape of test dataset {}, number of columns: {}".format(train.shape[0],test.shape[0], len(test.columns)))


# NaN data in target column.

# In[ ]:


train[train.winPlacePerc.isnull()]


# Drop this record

# In[ ]:


train.drop(train[train.winPlacePerc.isnull()].index,inplace=True)


# In[ ]:


train.info()


# In[ ]:


train_floats=train.select_dtypes(include=['float64']).drop(axis=1,labels='winPlacePerc')
test_floats=test.select_dtypes(include=['float64'])

train_int=train.select_dtypes(include=['int64'])
test_int=test.select_dtypes(include=['int64'])


# In[ ]:


train_int.head()


# In[ ]:


plt.figure(figsize=(12,8))
sns.lineplot(data=test_int.rankPoints.value_counts())


# In[ ]:


# Correlation heatmap
corr = train.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Create heatmap
heatmap = sns.heatmap(corr)


# In[ ]:


corr_columns=corr[abs(corr.winPlacePerc)>0.3].winPlacePerc.index


# We can filtred some other records(over 9000 kills, headshots and etc.), but I don't think that's good idea 

# In[ ]:


train[corr_columns].head()


# In[ ]:


train[train.groupId==train.groupId.unique()[0]].matchId.shape[0]>4


# In[ ]:


print('Uniqie Ids: ',len(train['Id'].unique()),' Unique groups: ',len(train['groupId'].unique()),' matchs: ',len(train['matchId'].unique()))


# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(test['longestKill'],bins=10)
plt.show()


# In[ ]:


train.head()


# ## Walk Distance

# In[ ]:


plt.figure(figsize=(12,4))
sns.distplot(train.walkDistance,bins=100)
plt.show()


# ## Walk Distance = 0 and Kills>0 

# In[ ]:


train=train.drop(train[(train.walkDistance==0) & (train.kills>0)].index)


# ## boosts and assits

# In[ ]:


train.head()


# In[ ]:


train['assists'].value_counts()


# In[ ]:


train[train.assists.isin([12,13,14,15,17,20,21,22])]


# In[ ]:


train.matchType.value_counts()


# Add new feature:

# In[ ]:


train["assists_and_boosts"]=train.assists+train.boosts
test["assists_and_boosts"]=test.assists+test.boosts


# ## Swim Distance

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(train.swimDistance,bins=10)
plt.show()


# ## Ride Distance

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(train[train['rideDistance']>20000].rideDistance,bins=10)
plt.show()


# ## Sumup feature

# In[ ]:


train['fullDistance']=train['swimDistance']+train['walkDistance']+train['rideDistance']
test['fullDistance']=test['swimDistance']+test['walkDistance']+test['rideDistance']


# In[ ]:


print("The average person uses {:.1f} boost items, 99% of people use {} or less, while the doctor used {}.".format(train['fullDistance'].mean(), train['fullDistance'].quantile(0.99), train['fullDistance'].max()))


# ## Weapons Acquired

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(train.weaponsAcquired,bins=100)
plt.show()


# In[ ]:


print("The average person uses {:.1f} boost items, 99% of people use {} or less, while the doctor used {}.".format(train['weaponsAcquired'].mean(), train['weaponsAcquired'].quantile(0.99), train['weaponsAcquired'].max()))


# ## Kills

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(train.kills,bins=100)
plt.show()


# In[ ]:


print("The average person uses {:.1f} boost items, 99% of people use {} or less, while the killer used {}.".format(train['kills'].mean(), train['kills'].quantile(0.99), train['kills'].max()))


# # headshot features

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(train.headshotKills,bins=100)
plt.show()


# In[ ]:


train[["headshotKills","kills"]].head()


# In[ ]:


train['headkill_ratio']=(train.headshotKills/train.kills).fillna(0)
test['headkill_ratio']=(test.headshotKills/test.kills).fillna(0)


# In[ ]:


train_ratio_pie=train['headkill_ratio'].value_counts()[1:]
test_ratio_pie=test['headkill_ratio'].value_counts()[1:]


# In[ ]:


train_ratio_pie.quantile(.75),train_ratio_pie.quantile(.95)


# In[ ]:


train_ratio_pie95=train_ratio_pie[train_ratio_pie>train_ratio_pie.quantile(.95)]
test_ratio_pie95=test_ratio_pie[test_ratio_pie>test_ratio_pie.quantile(.95)]


# In[ ]:


train_ratio_pie95labels=['1','1/2','1/3','1/4','2/3','1/5','2/5','1/6','3/4','3/5','1/7','2/7','3/7']
test_ratio_pie95labels=['1','1/2','1/3','1/4','2/3','1/5','2/5','1/6']
plt.figure(figsize=[25,15])
plt.subplot(221)

plt.title("popular headkill_ratios: ")
plt.pie(x=train_ratio_pie95,labels=train_ratio_pie95labels)

plt.subplot(222)

plt.title("popular headkill_ratios: ")
plt.pie(x=test_ratio_pie95,labels=test_ratio_pie95labels)
plt.show()


# In[ ]:


print('quantile 85% headkill ratio : {}, 99% {:.1f}, headshot killer:{:.1f}'.format(train['headkill_ratio'].quantile(.85),train['headkill_ratio'].quantile(0.99),train['headkill_ratio'].max()))


# In[ ]:


print('quantile 85% headkill ratio : {}, 99% {:.1f}, headshot killer:{:.1f}'.format(test['headkill_ratio'].quantile(.85),test['headkill_ratio'].quantile(0.99),test['headkill_ratio'].max()))


# ## Road kills

# ### Lets create new feature!!!
# roadKills/rideDistance	

# In[ ]:


train['distance_to_kill']=(train["rideDistance"]/train["roadKills"]).fillna(0.0).replace(np.inf,0)
test['distance_to_kill']=(test["rideDistance"]/test["roadKills"]).fillna(0.0).replace(np.inf,0)
train['road_killer']=(train['roadKills']>0).astype(int)
test['road_killer']=(test['roadKills']>0).astype(int)


# In[ ]:


train_roadkiller=train.road_killer.value_counts()
test_roadkiller=test.road_killer.value_counts()

plt.figure(figsize=[25,10])

plt.subplot(2,2,1)
plt.title("train road killer ratio: ")
plt.pie(x=train_roadkiller,labels=['not','road killer'])

plt.subplot(2,2,2)
plt.title("test road killer ratio: ")
plt.pie(x=test_roadkiller,labels=['not','road killer'])
plt.show()


# ## Let's see corr again!

# In[ ]:


# Correlation heatmap
corr = train.corr()

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Create heatmap
heatmap = sns.heatmap(corr)


# In[ ]:


new_corr_columns=corr[abs(corr.winPlacePerc)>0.1].winPlacePerc.index


# In[ ]:


new_corr_columns


# In[ ]:


train[new_corr_columns].info()


# In[ ]:


new_corr_columns.drop(['winPlacePerc'])


# In[ ]:


corr_train=train[new_corr_columns]
corr_test=test[new_corr_columns.drop(['winPlacePerc'])]


# In[ ]:


corr_train.head()


# # See some data

# In[ ]:


Y_trainingData=corr_train.winPlacePerc
corr_train=corr_train.drop('winPlacePerc',axis=1)


# In[ ]:


Y_trainingData.hist()


# In[ ]:


corr_train.head()


# In[ ]:


ids=['Id','groupId','matchId']
matchType=['matchType']


# ## Normalize some features

# In[ ]:


from sklearn.preprocessing import normalize
norm_corr_train=pd.DataFrame(normalize(corr_train),columns=new_corr_columns.drop(['winPlacePerc']))
norm_corr_test=pd.DataFrame(normalize(corr_test),columns=new_corr_columns.drop(['winPlacePerc']))


# In[ ]:


norm_corr_train.head()


# In[ ]:


norm_corr_test.head()


# # get Dummies from 'matchType'

# In[ ]:


matchType_train=pd.get_dummies(train['matchType'])
matchType_test=pd.get_dummies(test['matchType'])


# In[ ]:


X_train=norm_corr_train.join(matchType_train)
X_test=norm_corr_test.join(matchType_test)


# In[ ]:


X_train=X_train.fillna(0)
X_test=X_test.fillna(0)


# In[ ]:


from sklearn import utils
from sklearn import preprocessing


# In[ ]:


lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(Y_trainingData)
print(training_scores_encoded)
print(utils.multiclass.type_of_target(Y_trainingData))
print(utils.multiclass.type_of_target(Y_trainingData.astype('float64')))
print(utils.multiclass.type_of_target(training_scores_encoded))


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train, training_scores_encoded, train_size=0.6, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.63, random_state=1)


# In[ ]:


print("X shape: ",X_train.shape,"train shape: ",x_train.shape,"train answers shape: ",y_train.shape,"x_val.shape: ",x_val.shape,"y_val.shape: ",y_val.shape)


# In[ ]:


x_test.shape,y_test.shape


# ## XGBoost

# In[ ]:


import xgboost as xgb


# In[ ]:


reg_boost=xgb.XGBRegressor(objective='reg:linear',colsample_bytree=0.3,learning_rate=0.1,max_depth=5,alpha=10,n_estimators=100)

fit the model
# In[ ]:


get_ipython().run_line_magic('timeit', '')
reg_boost.fit(x_train,y_train)


# In[ ]:


get_ipython().run_line_magic('timeit', '')
train_predict = reg_boost.predict(x_train)


# In[ ]:


get_ipython().run_line_magic('timeit', '')
test_predict = reg_boost.predict(x_test)


# In[ ]:


X_trainDF=pd.DataFrame({"Predict":train_predict,"True":y_train})
X_testDF=pd.DataFrame({"Predict":test_predict,"True":y_test})


# In[ ]:


plt.figure(figsize=[25,10])
plt.subplot(211)
plt.title('train model')
plt.plot(X_trainDF[100000:100100])


plt.subplot(212)
plt.title('test model')
plt.plot(X_testDF[100000:100100])
plt.show()


# In[ ]:


from sklearn.metrics import mean_squared_error
train_rmse=np.sqrt(mean_squared_error(y_train,train_predict ))
test_rmse = np.sqrt(mean_squared_error(y_test,test_predict ))

print("train RMSE: {}, test RMSE:{}".format(train_rmse,test_rmse))


# In[ ]:


from sklearn.metrics import mean_absolute_error
print("train mean absolute error {}, test mean absolute error {}".format(mean_absolute_error(y_train,train_predict),mean_absolute_error(y_test,test_predict)))


# In[ ]:


submit_test_predict=reg_boost.predict(X_test)


# In[ ]:





# In[ ]:


submission=pd.DataFrame({'Id':test['Id'],'winPlacePerc':submit_test_predict})
submission.to_csv('submission.csv', index=False)


# In[ ]:




