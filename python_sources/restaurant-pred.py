#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings(action="ignore")
import os
#import featuretools as ft


# In[ ]:


print(os.listdir("../input"))


# In[ ]:


raw_data1 = pd.read_excel("../input/Data_Train.xlsx")
test_data = pd.read_excel("../input/Data_Test.xlsx")
raw_data1['source'] = 'train'
test_data['source'] = 'test'
raw_data = pd.concat([raw_data1,test_data],ignore_index=True)


# In[ ]:


raw_data.info(memory_usage='deep')


# In[ ]:


raw_data.columns


# In[ ]:


raw_data.head(5)
raw_data = raw_data[['CITY', 'CUISINES', 'LOCALITY', 'RATING',
       'TIME', 'TITLE', 'VOTES', 'source', 'COST']]


# In[ ]:


print("The train shape is", raw_data1.shape)
print("The test shape is", test_data.shape)


# In[ ]:


raw_data.isnull().sum()


# In[ ]:


#raw_data.drop(['RESTAURANT_ID'],inplace=True,axis=1)


# In[ ]:


raw_data.isnull().sum()


# In[ ]:


raw_data.head(5)


# In[ ]:


raw_data.dtypes


# In[ ]:


def unique_counts(df,features,p=False):
    for i in features:
        if p:
         print("The number of unique values for",i, df[i].value_counts())
         print("The number of unique values for",i, len(df[i].value_counts()))
         print("-"*100)
        else :
            print("The number of unique values for",i, len(df[i].value_counts()))
            print("-"*100)


# In[ ]:


ft = ['CITY','CUISINES','LOCALITY','TITLE',]
unique_counts(raw_data,ft)


# In[ ]:


raw_data['CITY'].fillna('NA', inplace=True)
raw_data['LOCALITY'].fillna('XXX', inplace=True)
raw_data['RATING'].fillna(0, inplace=True)
raw_data['VOTES'].fillna("0 votes",inplace=True)


# In[ ]:


def comma_seprated_categories(df, col):
    cols = list(df[col])
    max = 1
    for i in cols:
        if len(i.split(',')) > max:
            max = len(i.split(','))
    print("Max number of",col,"in a cell",max)
    
    all_cats = []
    for i in cols :
     if len(i.split(',')) == 1:
         all_cats.append(i.split(',')[0].strip().upper())
     else :
        for it in range(len(i.split(','))):
            all_cats.append(i.split(',')[it].strip().upper())
    print("\n\nNumber of Unique",col,": ", len(pd.Series(all_cats).unique()))
    print("\n\nUnique",col,":\n", pd.Series(all_cats).unique()) 
    return list(pd.Series(all_cats).unique())


# In[ ]:


all_titles = comma_seprated_categories(raw_data,col='TITLE')
all_cuisines =comma_seprated_categories(raw_data,col='CUISINES')


# In[ ]:





# In[ ]:


raw_data.isnull().sum()


# In[ ]:


#comma_seprated_categories(raw_data,col='TITLE')
comma_seprated_categories(raw_data,col='CUISINES')


# In[ ]:


raw_data[raw_data['source']=='train']['COST'].describe()


# In[ ]:


sns.distplot((raw_data[raw_data['source']=='train']['COST']))
plt.show()


# In[ ]:


np.log(raw_data[raw_data['source']=='train']['COST']).describe()


# In[ ]:


sns.distplot(np.log(raw_data[raw_data['source']=='train']['COST']))
plt.show()


# In[ ]:


print(len(raw_data['CITY'].str.split(" ",expand = True)[0].value_counts()))
print(len(raw_data['CITY'].value_counts()))


# In[ ]:


raw_data['CUISINES'].str.split(",",expand = True).head(5)
# Max 8 cusines in any given row


# In[ ]:


len(raw_data['TITLE'].unique())


# In[ ]:


raw_data['TITLE_1'] = raw_data['TITLE'].str.split(",",expand = True)[0]
raw_data['TITLE_2'] = raw_data['TITLE'].str.split(",",expand = True)[1]


# In[ ]:


raw_data.drop(['TITLE'],inplace=True,axis=1)


# In[ ]:


raw_data.head(5)


# In[ ]:


len( raw_data['LOCALITY'].str.split(" ",expand = True)[0].unique())


# In[ ]:


def mapping(df,col,n=25):
 print(col,n)
 vc = df[col].value_counts()
 replacements = {}
 for col, s in vc.items():
    if s[s<n].any():
        replacements[col] = 'other'
 return replacements


# In[ ]:


sns.boxplot(x='TITLE_1',y='COST', data=raw_data)
sns.set(rc={'figure.figsize':(40,30)})
sns.set(font_scale=1)   
plt.show()


# In[ ]:


raw_data.isnull().sum()


# In[ ]:


raw_data['TITLE_2'].fillna("None", inplace=True)


# In[ ]:


raw_data.isnull().sum()


# In[ ]:


raw_data['CUISINES_1'] = raw_data['CUISINES'].str.split(",",expand = True)[0]
raw_data['CUISINES_2'] = raw_data['CUISINES'].str.split(",",expand = True)[1]
raw_data['CUISINES_3'] = raw_data['CUISINES'].str.split(",",expand = True)[2]
raw_data['CUISINES_4'] = raw_data['CUISINES'].str.split(",",expand = True)[3]
raw_data['CUISINES_5'] = raw_data['CUISINES'].str.split(",",expand = True)[4]
raw_data['CUISINES_6'] = raw_data['CUISINES'].str.split(",",expand = True)[5]
raw_data['CUISINES_7'] = raw_data['CUISINES'].str.split(",",expand = True)[6]
raw_data['CUISINES_8'] = raw_data['CUISINES'].str.split(",",expand = True)[7]


# In[ ]:


raw_data.isnull().sum()


# In[ ]:


cus_list = []
for i in range(1,9):
    i = str(i)
    cus_list.append("CUISINES_"+i)
    


# In[ ]:


for i in cus_list:
    raw_data[i].fillna("NAA", inplace=True)
all_cuisines.append("NAA")


# In[ ]:


raw_data.isnull().sum()


# In[ ]:


raw_data.dtypes


# In[ ]:


raw_data['VOTES'] = raw_data['VOTES'].str.split(" ", expand=True)[0]
raw_data['VOTES']  = pd.to_numeric(raw_data['VOTES'])


# In[ ]:


raw_data.dtypes


# In[ ]:


rates = list(raw_data['RATING'])

for i in range(len(rates)) :
    try:
       rates[i] = float(rates[i])
    except :
       rates[i] = np.nan


# In[ ]:


raw_data['RATING'] = rates


# In[ ]:


raw_data.isnull().sum()


# In[ ]:


raw_data['RATING'].fillna(0.0,inplace=True)


# In[ ]:


raw_data.isnull().sum()


# In[ ]:


raw_data.dtypes


# In[ ]:


raw_data.drop(['TIME'], inplace=True,axis=1)


# In[ ]:


raw_data.drop(['CUISINES'], inplace=True,axis=1)


# In[ ]:


raw_data.columns


# In[ ]:


raw_data = raw_data[['CITY', 'LOCALITY', 'RATING', 'VOTES', 'source', 'TITLE_1',
       'TITLE_2', 'CUISINES_1', 'CUISINES_2', 'CUISINES_3', 'CUISINES_4',
       'CUISINES_5', 'CUISINES_6', 'CUISINES_7', 'CUISINES_8','COST']]


# In[ ]:


raw_data.head(10)


# In[ ]:


#correlation matrix
corrmat = raw_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()


# In[ ]:


raw_data.corr()


# In[ ]:


#scatter plot experince/saleprice

var = 'RATING'
data = pd.concat([raw_data['COST'], raw_data[var]], axis=1)
#plt.figure(figsize=(40,20))
#plt.xlabel('xlabel', fontsize=50)
#plt.ylabel('ylabel', fontsize=50)
data.plot.scatter(x=var, y='COST', figsize = (40,20), s=120,grid=True,fontsize=40,);
plt.show()


# In[ ]:


plt.figure(figsize=(40,20))
plt.xlabel('xlabel', fontsize=50)
plt.ylabel('ylabel', fontsize=50)
sns.distplot(raw_data["RATING"])
plt.show()


# In[ ]:


raw_data['RATING'].describe()


# In[ ]:


raw_data['VOTES'].describe()


# In[ ]:


plt.figure(figsize=(40,20))
plt.xlabel('xlabel', fontsize=50)
plt.ylabel('ylabel', fontsize=50)
sns.distplot(raw_data["VOTES"])
plt.show()


# In[ ]:


raw_data.isnull().sum()


# In[ ]:


def mapping(df,col,n=25):
 print(col,n)
 vc = df[col].value_counts()
 replacements = {}
 for col, s in vc.items():
    if s[s<n].any():
        replacements[col] = 'other'
 return replacements
local = mapping(raw_data,'LOCALITY',n=40)
raw_data['LOCALITY'] = raw_data['LOCALITY'].replace(local)


# In[ ]:


from sklearn.preprocessing import LabelEncoder


le_titles = LabelEncoder()
le_cuisines = LabelEncoder()
le_city = LabelEncoder()
le_locality = LabelEncoder()


le_titles.fit(all_titles)
le_cuisines.fit(all_cuisines)

le_city.fit(raw_data['CITY'])
le_locality.fit(raw_data['LOCALITY'])


# In[ ]:


raw_data['TITLE_1']=raw_data['TITLE_1'].str.upper()
raw_data['TITLE_2']=raw_data['TITLE_2'].str.upper()


# In[ ]:


for i in cus_list:
    raw_data[i] = raw_data[i].str.upper()
    raw_data[i] = raw_data[i].str.strip()


# In[ ]:





# In[ ]:


raw_data['TITLE_1'] = le_titles.transform(raw_data['TITLE_1'])
raw_data['TITLE_2'] = le_titles.transform(raw_data['TITLE_2'])


raw_data['CUISINES_1'] = le_cuisines.transform(raw_data['CUISINES_1'])
raw_data['CUISINES_2'] = le_cuisines.transform(raw_data['CUISINES_2'])
raw_data['CUISINES_3'] = le_cuisines.transform(raw_data['CUISINES_3'])
raw_data['CUISINES_4'] = le_cuisines.transform(raw_data['CUISINES_4'])
raw_data['CUISINES_5'] = le_cuisines.transform(raw_data['CUISINES_5'])
raw_data['CUISINES_6'] = le_cuisines.transform(raw_data['CUISINES_6'])
raw_data['CUISINES_7'] = le_cuisines.transform(raw_data['CUISINES_7'])
raw_data['CUISINES_8'] = le_cuisines.transform(raw_data['CUISINES_8'])


raw_data['CITY'] = le_city.transform(raw_data['CITY'])
raw_data['LOCALITY'] = le_locality.transform(raw_data['LOCALITY'])


# In[ ]:


raw_data.head(5)


# In[ ]:


le_cuisines.inverse_transform([88])


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
raw_data['VOTES']= sc.fit_transform(raw_data['VOTES'].reshape(len(raw_data['VOTES']),1))
raw_data['RATING']= sc.fit_transform(raw_data['RATING'].reshape(len(raw_data['RATING']),1))


# In[ ]:


raw_data.head(5)


# In[ ]:





# In[ ]:


raw_data.drop(["CUISINES_6","CUISINES_7","CUISINES_8"], axis=1,inplace=True)


# In[ ]:


raw_data.head(5)


# In[ ]:


#Divide into test and train:
train = raw_data.loc[raw_data['source']=="train"]
test = raw_data.loc[raw_data['source']=="test"]
train.drop(["source"], inplace=True, axis=1)
test.drop(["source","COST"], inplace=True, axis=1)


# In[ ]:


train['COST'] = np.log(train['COST'])


# In[ ]:


from sklearn import model_selection
from sklearn.metrics import mean_squared_error
import xgboost
import numpy as np
from sklearn.ensemble import RandomForestRegressor
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(train.drop(["COST"], axis=1), train['COST'])


# In[ ]:


from sklearn.grid_search import GridSearchCV


# In[ ]:


RF_G= param_grid={
            'max_depth': [4,8,10,12,14,16],
            'n_estimators': [10,20,30,40,50],
            'min_samples_split': [2, 5, 10]
        },


# In[ ]:


gsearchRF1 = GridSearchCV(estimator = RandomForestRegressor(),param_grid=RF_G,scoring='neg_mean_squared_log_error')


# In[ ]:


gsearchRF1.fit(train_x,train_y)


# In[ ]:


gsearchRF1.best_params_


# In[ ]:


RF_G_2 = param_grid={
            'max_depth': [12],
            'n_estimators': [40,50,60],
            'min_samples_split': [10,12,15,19,24]
        },


# In[ ]:


gsearchRF2 = GridSearchCV(estimator = RandomForestRegressor(),param_grid=RF_G_2,scoring='neg_mean_squared_log_error')


# In[ ]:


gsearchRF2.fit(train_x,train_y)


# In[ ]:


gsearchRF2.best_params_


# In[ ]:


RF1 = RandomForestRegressor(max_depth=14, min_samples_split=10, n_estimators=50, n_jobs=-1)


# In[ ]:


RF1.fit(train_x,train_y)


# In[ ]:


RF1.score(valid_x,valid_y)


# In[ ]:


sub1 = np.exp(RF1.predict(test))


# In[ ]:


pd.DataFrame(sub1).to_excel("./subimission_01.xlsx")


# In[ ]:


import xgboost


# In[ ]:


param_test1 = {'n_estimators':list(range(20,121,10))}
gsearch1 = GridSearchCV(estimator = xgboost.XGBRegressor(learning_rate=0.1, min_samples_split=100,min_samples_leaf=50,max_depth=8,max_features='sqrt',subsample=0.8,random_state=10), 
param_grid = param_test1, scoring='neg_mean_squared_log_error',n_jobs=4,iid=False, cv=5)
get_ipython().run_line_magic('time', 'gsearch1.fit(train_x,train_y)')


# In[ ]:


gsearch1.best_params_


# In[ ]:


param_test2 = {'max_depth':list(range(5,16,2))}
gsearch2 = GridSearchCV(estimator = xgboost.XGBRegressor(learning_rate=0.1, n_estimators=120, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test2, scoring='neg_mean_squared_log_error',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train_x,train_y)
#print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)


# In[ ]:


gsearch2.best_params_


# In[ ]:


param_test3 = {'min_child_weight':[1,2,3,4]}
gsearch3 = GridSearchCV(estimator = xgboost.XGBRegressor(learning_rate=0.1, n_estimators=120, max_features='sqrt', subsample=0.8, random_state=10,max_depth=7), 
param_grid = param_test3, scoring='neg_mean_squared_log_error',n_jobs=4,iid=False, cv=5)
gsearch3.fit(train_x,train_y)
#print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)


# In[ ]:


gsearch3.best_params_


# In[ ]:


param_test4 = {'gamma':[i/10.0 for i in range(1,10)]}
gsearch4 = GridSearchCV(estimator = xgboost.XGBRegressor(learning_rate=0.1, n_estimators=120, max_features='sqrt', subsample=0.8, random_state=10,max_depth=7, min_child_weight=3), 
param_grid = param_test4, scoring='neg_mean_squared_log_error',n_jobs=4,iid=False, cv=5)
gsearch4.fit(train_x,train_y)


# In[ ]:


gsearch4.best_params_


# In[ ]:


xg_algo = xgboost.XGBRegressor(n_estimators=120, max_depth=7, min_child_weight=3,
                              gamma=0.6)


# In[ ]:


xg_algo.fit(train_x,train_y)


# In[ ]:


xg_algo.score(valid_x,valid_y)


# In[ ]:


sub9 = np.exp(xg_algo.predict(test))
pd.DataFrame(sub9).to_excel("./subimission_09.xlsx")


# In[ ]:


xgboost.plot_importance(xg_algo)
plt.figure(figsize=(40,20))
#plt.xlabel(fontsize=50)
#plt.ylabel(fontsize=50)
plt.show()


# In[ ]:


sub2 = np.exp(xg_algo.predict(test))
pd.DataFrame(sub2).to_excel("./subimission_02.xlsx")


# In[ ]:


sub1


# In[ ]:


sub2


# In[ ]:


param_test5 = {'n_estimators':list(range(120,180,10)),
              'max_depth':list(range(5,16,2)),
              'min_child_weight':[1,2,3,4],
              'gamma':[i/10.0 for i in range(1,10)]}


# In[ ]:


gsearch5 = GridSearchCV(estimator = xgboost.XGBRegressor(learning_rate=0.1, max_features='sqrt', subsample=0.8, random_state=10), 
param_grid = param_test5, scoring='neg_mean_squared_log_error',n_jobs=-1,iid=False, cv=5)
gsearch5.fit(train_x,train_y)


# In[ ]:


gsearch5.best_params_


# In[ ]:


xg_boost2 = xgboost.XGBRegressor(max_depth=11, gamma=0.6,min_child_weight=2, n_estimators=120)


# In[ ]:


xg_boost2.fit(train_x, train_y)


# In[ ]:


xg_boost2.score(valid_x, valid_y)


# In[ ]:


sub9 = np.exp(xg_boost2.predict(test))
pd.DataFrame(sub9).to_excel("./subimission_09.xlsx")


# In[ ]:


xgboost.plot_importance(xg_boost2)
plt.show()


# In[ ]:


sub3 = (sub2 + sub4) /  2
pd.DataFrame(sub2).to_excel("./subimission_03.xlsx")


# In[ ]:


sub5 = np.exp(xg_boost2.predict(test))
#pd.DataFrame(sub5).to_excel("./subimission_05.xlsx")


# In[ ]:


sub5


# In[ ]:


from keras import layers
from keras import models
from keras.layers import Dropout
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation = 'relu', input_shape = (train_x.shape[1],)))
    model.add(Dropout(.2))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(Dropout(.2))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(Dropout(.2))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))
    model.add(layers.Dense(64, activation = 'relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['mse'])
    return model


# In[ ]:


NN_model = build_model()


# In[ ]:


NN_model.fit(train_x, train_y, epochs=30, verbose=1,validation_data=(valid_x,valid_y))


# In[ ]:


sub6 = np.exp(NN_model.predict(test))


# In[ ]:


sub66 = (sub5 + sub6)/2
pd.DataFrame(sub66).to_excel("./sub66.xlsx")


# In[ ]:


import pandas as pd


# In[ ]:


pd.__version__


# In[ ]:




