#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra|
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
#Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import export_graphviz
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing" Shift+Enter) will list the files in the input directory
import ast
import datetime
from tqdm import tqdm
import os
print(os.listdir("../input"))
from pylab import rcParams
rcParams['figure.figsize'] = 25, 10
# Any results you write to the current directory are saved as output.
import xgboost as xgb


# In[ ]:


train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")


# In[ ]:


train.head()


# In[ ]:


test.head()


# 
# 
# # Trying to get dicts

# 
# 
# 
# 'id' in genres is just code of genres. So id is useless

# In[ ]:


def getting_from_dicts(i,column='name'):

    if pd.isna(i):
        #print(i,'nan')
        return 'NaN'
    elif i=='[]':
        return 'NaN'
    elif type(ast.literal_eval(i[1:-1]))==dict:
        #print(i,'dict')
        return ast.literal_eval(i[1:-1])[column] 
    elif type(ast.literal_eval(i[1:-1]))==tuple:
        #print(i,'tuple')
        genres_of_cinema=[]
        for ii in range(len(ast.literal_eval(i[1:-1]))):
            genres_of_cinema.append(ast.literal_eval(i[1:-1])[ii][column])
        return genres_of_cinema
    


# In[ ]:


train_genres=pd.DataFrame()
test_genres=pd.DataFrame()
train_genres['genres_list']=train.genres.apply(getting_from_dicts)
test_genres['genres_list']=test.genres.apply(getting_from_dicts)


# Let's see what we get

# In[ ]:


train_genres['genres_list']


# In[ ]:


type(train_genres['genres_list'][2])


# In[ ]:


test_genres['genres_list'].head()


# So, we got lists of genres in every film or just string.

# ## Creating new columns

# How to handle this lists of genres? We'll create new columns for every genre.

# In[ ]:


def creating_columns(dateframe,column='genres_list'):
    genres=pd.Series()
    for i in tqdm(range(dateframe[column].shape[0])):
        if type(dateframe.at[i,column])==list:
            for ii in range(len(dateframe.at[i,column])):
                genres=genres.append(pd.Series(dateframe.at[i,column][ii]))
        elif type(dateframe.at[i,column])==str:
            genres=genres.append(pd.Series(dateframe.at[i,column]))
    # creating columns
    for i in tqdm(genres.unique()):
        dateframe[i]=pd.Series(np.zeros(shape=(dateframe.shape[0],)))
    #filling columns
    for i in tqdm(range(dateframe[column].shape[0])):
        if type(dateframe.at[i,column])==list:
            for ii in range(len(dateframe.at[i,column])):
                dateframe.at[i,dateframe.at[i,column][ii]]=1
        elif type(dateframe[column].iloc[i])==str:
            dateframe.at[i,dateframe.at[i,column]]=1


# ## Working with crew

# Cast of film are making film's boxoffice in general. So,  we'll get celebs from cast_list:

# In[ ]:


train_cast=pd.DataFrame()
test_cast=pd.DataFrame()
train_cast['cast_list']=train.cast.apply(getting_from_dicts)
test_cast['cast_list']=test.cast.apply(getting_from_dicts)


# Function thats creating zero columns:

# In[ ]:


def creatin_zero_col(train,test):
    for i in test.columns.symmetric_difference(train.columns):
        train[i]=pd.Series(np.zeros(shape=(train.shape[0],)))
    for i in train.columns.symmetric_difference(test.columns):
        test[i]=pd.Series(np.zeros(shape=(test.shape[0],)))


# In[ ]:


creating_columns(train_genres)
creating_columns(test_genres)


# In[ ]:


creating_columns(train_cast,column='cast_list')
creating_columns(test_cast,column='cast_list')


# Let's see what we get:

# In[ ]:


train_cast.head()


# In[ ]:


train_cast.columns


# In[ ]:


test_cast.head()


# In[ ]:


test_cast.columns


# ## Genres

# In this part we add zero columns in DataFrames. In test data some genres and in train data some other genres. That way we'll made DataFrames same columns:

# ### Difference between train_genres and test_genres

# Check if columns are equal.

# In[ ]:


test_genres.columns.symmetric_difference(train_genres.columns)


# In[ ]:


creatin_zero_col(train_genres,test_genres)


# In[ ]:


test_genres.columns.symmetric_difference(train_genres.columns)


# In[ ]:


train_genres.drop(columns='genres_list',inplace=True)
test_genres.drop(columns='genres_list', inplace=True)


# ## Cast

# ### Difference between train_cast and test_cast

# In[ ]:


train_cast.drop('cast_list',axis=1,inplace=True)


# In[ ]:


train_cast.columns


# In[ ]:


cast_stat=train_cast.apply(pd.value_counts)


# In[ ]:


top_cast=cast_stat.T[cast_stat.T[1.0]>17].index


# In[ ]:


train_cast=train_cast[top_cast]
train_cast.drop('NaN',axis=1,inplace=True)


# In[ ]:


test_cast=test_cast[top_cast]
test_cast.drop('NaN',axis=1,inplace=True)


# In[ ]:


plt.figure(figsize=(40, 10))
sns.barplot(data=train_cast[train_cast.columns[:24]])


# In[ ]:


plt.figure(figsize=(40, 10))
sns.barplot(data=train_cast[train_cast.columns[24:]])


# In[ ]:


plt.figure(figsize=(25, 6))
sns.barplot(data=train_genres)


# In[ ]:


plt.figure(figsize=(23, 6))
sns.barplot(data=test_genres, label='test Dataset')


# ## Spoken Languages

# In[ ]:


train_spoken=pd.DataFrame()
test_spoken=pd.DataFrame()
train_spoken['spoken_languages_list']=train.spoken_languages.apply(getting_from_dicts,args=('iso_639_1',))
test_spoken['spoken_languages_list']=test.spoken_languages.apply(getting_from_dicts,args=('iso_639_1',))
creating_columns(train_spoken,column='spoken_languages_list')
creating_columns(test_spoken,column='spoken_languages_list')


# In[ ]:


train_spoken.columns.symmetric_difference(test_spoken.columns)


# In[ ]:


creatin_zero_col(train_spoken,test_spoken)


# In[ ]:


train_spoken.columns.symmetric_difference(test_spoken.columns)


# In[ ]:


train_spoken.drop(columns='spoken_languages_list',inplace=True)
test_spoken.drop(columns='spoken_languages_list',inplace=True)


# In[ ]:


print('spoken english train {} %, spoken english test {} %'.format(round(train_spoken.en.value_counts().iloc[0]/train_spoken.shape[0],2)*100,round(test_spoken.en.value_counts().iloc[0]/test_spoken.shape[0],2)*100))


# In[ ]:


plt.subplot(221)
plt.title('train')

train_spoken.en.value_counts().plot.bar()

plt.subplot(222)

test_spoken.en.value_counts().plot.bar()
plt.title('test')
plt.show()


# # Production companies

# In[ ]:


train_companies=pd.DataFrame()
test_companies=pd.DataFrame()
train_companies['companies']=train.production_companies.apply(getting_from_dicts)
test_companies['companies']=test.production_companies.apply(getting_from_dicts)
creating_columns(train_companies,column='companies')
creating_columns(test_companies,column='companies')
train_companies.drop(columns='companies',inplace=True)
test_companies.drop(columns='companies',inplace=True)


# In[ ]:


train_companies.head()


# In[ ]:


col_comp=[]
for gen in train_companies:
    per=100-round(train_companies[gen].value_counts().iloc[0]/train_companies.shape[0]*100)
    if(per>1):
        print("{} {}%".format(gen,per))
        col_comp.append(gen)
col_comp


# In[ ]:


train_companies_top=train_companies[col_comp]
test_companies_top=test_companies[col_comp]


# In[ ]:


plt.figure(figsize=(36, 8))
sns.barplot(data=train_companies_top)


# In[ ]:


plt.figure(figsize=(36, 8))
sns.barplot(data=test_companies_top)


# # Year

# In[ ]:


train.release_date=pd.to_datetime(train.release_date)
test.release_date=pd.to_datetime(test.release_date)


# In[ ]:


train['weekday']=train.release_date.dt.weekday_name
test['weekday']=test.release_date.dt.weekday_name


# In[ ]:


train['year']=train.release_date.dt.year
test['year']=test.release_date.dt.year


# In[ ]:


train['month']=train.release_date.dt.month
test['month']=test.release_date.dt.month


# In[ ]:


months={1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}

train_months=train['month'].value_counts().index.map(months)
test_months=test['month'].value_counts().index.map(months)


# ### Train data relese month

# In[ ]:


plt.figure(figsize=(26,6))
plt.pie(train['month'].value_counts(),labels=train_months)
plt.title("month of release")
plt.show()


# ### Test data relese month

# In[ ]:


plt.figure(figsize=(26,6))
plt.pie(test['month'].value_counts(),labels=test_months)
plt.title("month of release")
plt.show()


# ## New features from "release_date"

# In[ ]:


def date_features(df, col, prefix):
    df_new=pd.DataFrame()
    today = datetime.datetime.today()
    df[col] = pd.to_datetime(df[col])
    cinema_start=datetime.date(year=1895,month=3,day=22)
    df_new[prefix+'_day_of_week'] = df[col].dt.dayofweek
    df_new[prefix+'_day_of_year'] = df[col].dt.dayofyear
    df_new[prefix+'_month'] = df[col].dt.month
    df_new[prefix+'_year'] = df[col].apply(lambda x: x.year-100 if (x>today) else x.year)
    df_new[prefix+'_day'] = df[col].dt.day
    df_new[prefix+'_is_year_end'] = df[col].dt.is_year_end
    df_new[prefix+'_is_year_start'] = df[col].dt.is_year_start
    df_new[prefix+'_week'] = df[col].dt.week
    df_new[prefix+'_quarter'] = df[col].dt.quarter    
    
   

    return df_new
    


# In[ ]:


today=datetime.datetime.now()
today.date()


# In[ ]:





# In[ ]:


release_year_train=date_features(train,'release_date','release')
release_year_test=date_features(test,'release_date','release')


# In[ ]:


release_year_train.head()


# In[ ]:


release_year_test.head()


# # Dummies variables

# In[ ]:


train_status=pd.get_dummies(train.status)
train_or_lan=pd.get_dummies(train.original_language,prefix='original')
test_status=pd.get_dummies(test.status)
test_or_lan=pd.get_dummies(test.original_language,prefix='original')


# In[ ]:


train_status.columns.symmetric_difference(test_status.columns)


# In[ ]:


creatin_zero_col(train_status,test_status)
train_status.columns.symmetric_difference(test_status.columns)


# In[ ]:


train_or_lan.columns.symmetric_difference(test_or_lan.columns)


# In[ ]:


creatin_zero_col(train_or_lan,test_or_lan)
train_or_lan.columns.symmetric_difference(test_or_lan.columns)


# In[ ]:


train_or_lan.original_en.value_counts()


# In[ ]:


test_or_lan.original_en.value_counts()


# ## Set Y 

# In[ ]:


Y=train.revenue


# ## Creating final datasets

# In[ ]:


print("train original language:{},test original language:{},train status:{},test status:{},train genres:{},test genres:{}".format(train_or_lan.original_en.shape,test_or_lan.original_en.shape,train_status.shape,test_status.shape,train_genres.shape,test_genres.shape))


# ## see train dataset

# In[ ]:


train.runtime.describe()


# In[ ]:


train[train.runtime.isna()]


# In[ ]:


test[test.runtime.isna()]


# In[ ]:


print("train runtime std: {}, test runtime std: {}".format(train.runtime.std(),test.runtime.std()))


# In[ ]:


train.runtime.fillna(train.runtime.std(),inplace=True)
test.runtime.fillna(test.runtime.std(),inplace=True)


# In[ ]:


print("train runtime count {},test runtime count {}".format(train.runtime.count(),test.runtime.count()))


# In[ ]:


print("train budget count {}, test budget count {}".format(train.budget.count(),test.budget.count()))


# In[ ]:


print("train popularity count {}, test popularity count {}".format(train.popularity.count(),test.popularity.count()))


# ## Normalization

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


train[["runtime","budget","popularity"]].head()


# In[ ]:


x=train.runtime.values.astype(float)
min_max_scaler_train = MinMaxScaler()
min_max_scaler_test = MinMaxScaler()
min_max_scaler_train.fit(train[["runtime","budget","popularity"]])
min_max_scaler_test.fit(test[["runtime","budget","popularity"]])


# In[ ]:



norm_train=min_max_scaler_train.transform(train[["runtime","budget","popularity"]])
norm_test=min_max_scaler_test.transform(test[["runtime","budget","popularity"]])


# In[ ]:


norm_train=pd.DataFrame(norm_train,columns=["runtime","budget","popularity"])
norm_test=pd.DataFrame(norm_test,columns=["runtime","budget","popularity"])


# In[ ]:


#train_or_lan,test_or_lan,train_status,test_status,train_genres,test_genres


# ### Concatenate

# In[ ]:


train_work=pd.DataFrame()
test_work=pd.DataFrame()
test_or_lan.original_en.shape,train_or_lan.original_en.shape


# In[ ]:


train_work=train_status.join(train_or_lan["original_en"])
train_work=train_work.join(train_genres)
train_work=train_work.join(norm_train)
#train_work=train_work.join(train_or_lan.original_en)
train_work=train_work.join(train_spoken.en)
train_work=train_work.join(release_year_train)
train_work=train_work.join(train_cast)


# In[ ]:


test_work=test_status.join(test_or_lan["original_en"])
test_work=test_work.join(test_genres)
test_work=test_work.join(norm_test)
#test_work=test_work.join(test_or_lan.original_en)
test_work=test_work.join(test_spoken.en)
test_work=test_work.join(release_year_test)
test_work=test_work.join(test_cast)


# # l2 regularization

# In[ ]:


from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(train_work,Y)


# In[ ]:


clf.score(train_work,Y)


# In[ ]:


clf.get_params()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_work, Y, test_size=0.75, random_state=42)


# # XGBoosting 

# In[ ]:


dtrain=xgb.DMatrix(X_train,label=y_train)
dvalid=xgb.DMatrix(X_test,label=y_test)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]


# In[ ]:


xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9, 
            'max_depth': 15,
'subsample': 0.9, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
'eval_metric': 'rmse', 'objective': 'reg:linear'}


# In[ ]:


model = xgb.train(xgb_pars, dtrain, 15, watchlist, early_stopping_rounds=2,
      maximize=False, verbose_eval=1)
print('Modeling RMSLE %.5f' % model.best_score)


# In[ ]:


xgb.plot_importance(model, max_num_features=28, height=0.7)


# In[ ]:


train_predict=model.predict(dtrain)
valid_predict =model.predict(dvalid)


# In[ ]:


from sklearn.metrics import mean_squared_error
err_train= mean_squared_error(y_train,train_predict)
err_test= mean_squared_error(y_test,valid_predict)
print("train error: {}, test error: {}".format(err_train,err_test))


# In[ ]:


X_trainDF=pd.DataFrame({"Predict":train_predict,"True":y_train})
X_testDF=pd.DataFrame({"Predict":valid_predict,"True":y_test})


# In[ ]:


len(train_predict)


# In[ ]:


X_trainDF.reset_index()[['Predict',"True"]].loc[450:500].plot()


# In[ ]:


X_testDF.reset_index()[['Predict','True']][1000:1100].plot()


# In[ ]:


dtest=xgb.DMatrix(test_work)


# In[ ]:


names=model.feature_names


# In[ ]:


dtest=xgb.DMatrix(test_work[names])


# In[ ]:


test_predicted =model.predict(dtest)


# In[ ]:


pred_df=pd.DataFrame({'id':test.id,'revenue':test_predicted})
pred_df.to_csv('submission.csv',index=False)


# In[ ]:




