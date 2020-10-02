#!/usr/bin/env python
# coding: utf-8

# This is my first attempt with minimal feature engineering and basic logistic regression model
# on Rican Household PovertyCosta model.

# In[ ]:


#import basic libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
print('train data shape :',train_df.shape)
print('test data shape :',test_df.shape)


# #### Here, we can see that number of observations in training data set is very less than test dataset (almost 40%).

# In[ ]:


#snapshot of training data
train_df.head()


# In[ ]:


#check distribution of target classes in training dataset
y=train_df['Target']
#y=train_df.query('parentesco1==1')['Target']
y.value_counts()


# In[ ]:


#check nulls in train dataset
train_null = train_df.isnull().sum()
train_null[train_null > 0]


# In[ ]:


#check nulls in test dataset
test_null = test_df.isnull().sum()
test_null[test_null > 0]


# #### We can see from above two cells that columns v2a1, v18q1, rez_esc are missing for most of the observations in both training and test dataset.

# In[ ]:


train_df.info()


# From discussion in [thread](https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#358941), poverty level is not consistent throughout the household. As per suggested by organizers, it is a data discrepany. So we will try to handle it here.

# In[ ]:


# Groupby the household and figure out the number of unique values
train_grphh = train_df.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)
err_train = train_grphh[train_grphh !=True]
print('Number of households with incorrect poverty level :',len(err_train))


# It is clarified in discussion, that correct poverty level is poverty level of head of the family. We can identity it using parentesco1 column with value 1. Let's use this to correct poverty level in errorneous records.

# In[ ]:


#let's correct the poverty level in incorrect records
for household in err_train.index:
    #find correct poverty level
    target = int(train_df[(train_df['idhogar']==household) & (train_df['parentesco1']==1.0)]['Target'])
    #set correct poverty level
    train_df.loc[train_df['idhogar']==household,'Target'] = target


# In[ ]:


#Align training and test dataset to find common features
train_df,test_df = train_df.align(test_df,join='inner',axis=1)

print('Training Features shape: ', train_df.shape)
print('Testing Features shape: ', test_df.shape)


# In[ ]:


#let's join train and test data
data = pd.concat([train_df,test_df],axis=0)
data.head()


# In[ ]:


#run some checks
data[data['hhsize'] != data['hogar_total']].shape


# ## Handle Missing values

# ### We have seen that majorly columns v2a1,v18q1 and rez_esc have missing values. Let's handle them one by one.
# 
# 

# #### v18q1 - denotes number of tables in household. We have another column v18q represents  whether household owns a table or not.

# In[ ]:


#get household heads
#hh = data.loc[data['parentesco1'] == 1]
#check null values for flag v18q
#hh.groupby('v18q')['v18q1'].apply(lambda x: x.isnull().sum())


# It seems value of v18q1 is null only for records where v18q is 0 i.e. no tablet in house. So, we can replace nan with 0 here.

# In[ ]:


#replace null values with 0
#data['v18q1'].fillna(0,inplace=True)


# In[ ]:


#similarly fill v2al for missing rent payment
# Fill in households that own the house with 0 rent payment
#data.loc[(data['tipovivi1'] == 1), 'v2a1'] = 0

# Create missing rent payment column
#data['v2a1-missing'] = data['v2a1'].isnull()

#data['v2a1-missing'].value_counts()


# In[ ]:


#rez_esc
# If individual is over 19 or younger than 7 and missing years behind, set it to 0
# data.loc[((data['age'] > 19) | (data['age'] < 7)) & (data['rez_esc'].isnull()), 'rez_esc'] = 0
# data.loc[data['rez_esc'] > 5, 'rez_esc'] = 5


# In[ ]:


#check distinct values in object(categorical) column
cat_cols = data.nunique()==2
cat_cols = list(cat_cols[cat_cols].keys())


# In[ ]:


cols = ['edjefe', 'edjefa','dependency']
data[cols] = data[cols].replace({'no': 0, 'yes':1}).astype(float)
# data = pd.get_dummies(data,columns=cat_cols)
# print('data shape after one hot encoding :',data.shape)
#interaction features
data['hogar_mid'] = data['hogar_adul'] - data['hogar_mayor']
data['bedroom%'] =  data['bedrooms']/data['rooms']
data['person/rooms'] = data['rooms']/data['hhsize']
data['male_ratio'] = data['r4h3']/data['r4t3']
data['female_ratio'] = data['r4m3']/data['r4t3']
data['female_per_room'] = data['r4m3']/data['rooms']
data['female_per_bedroom']  = data['r4m3']/data['bedrooms']
data['hogarmid_per_bedroom']  = data['hogar_mid']/data['bedrooms']
#data['hogar_nin%'] = data['hhsize']/(data['hogar_nin']+1)
data['escolari_age'] = data['escolari']/data['age']
data['hogar_nin_per_room'] = data['rooms']/(data['hogar_nin']+1)
data['male_per_bedroom'] = data['r4h3']/data['bedrooms']
data['male_per_room'] = data['r4h3']/data['rooms']
data['dependencyXmale']= data['dependency']*data['r4h3']
data['dependencyXhogar_mid'] = data['dependency']*data['hogar_mid']
data['dependencyXhogar_adul'] = data['dependency']*data['hogar_adul']
data['dependency_prod_sum']=data['dependencyXmale'] + data['dependencyXhogar_mid'] + data['dependencyXhogar_adul']
data['overcrowding_room_and_bedroom'] = (data['hacdor'] + data['hacapo'])/2

data['no_appliances'] = data['refrig'] + data['computer'] + data['television']
#data['mobile_per_bedrooms'] = (data['qmobilephone'])/data['bedrooms']
#data['mobile_per_person'] = data['r4t3']/(data['qmobilephone']+1)
#data['mobile_per_male'] = data['r4h3']/(data['qmobilephone']+1)
#data['mobile_per_female'] = data['r4m3']/(data['qmobilephone']+1)
#data['escolari_age_diff'] = data['age'] - data['escolari']
#aggregation features
# other_list = ['escolari', 'age']
# for item in other_list:
#     for function in ['mean','std','min','max','sum']:
#         group_data = data[item].groupby(data['idhogar']).agg(function)
#         new_col = item + '_' + function
#         data[new_col] = group_data


# In[ ]:


#aggregation columns
df_group = pd.DataFrame()
other_list = ['escolari', 'age', 'escolari_age']
for item in other_list:
    for function in ['mean','std','min','max','sum']:
        group_data = data[item].groupby(data['idhogar']).agg(function)
        new_col = item + '_' + function
        df_group[new_col] = group_data
df_group = df_group.reset_index()
data = pd.merge(data, df_group, on='idhogar')


# In[ ]:


#drop columns contains mostly null values
data.drop(labels=['v2a1','v18q1','rez_esc','Id','idhogar'],axis=1,inplace=True)
print('data shape after dropping null columns :',data.shape)


# In[ ]:


#impute missing values
from sklearn.preprocessing import MinMaxScaler, Imputer
# Median imputation of missing values
#imputer = Imputer(strategy = 'median')
# Fit on the training data
#imputer.fit(data)

# Transform both training and testing data
#data = imputer.transform(data)
data.fillna(-1,inplace=True)
# scaler = MinMaxScaler()
# data = pd.DataFrame(scaler.fit_transform(data[list(datacols-set(cat_cols))]))


# In[ ]:


train_df = data[:len(train_df)]
test_df = data[len(train_df):]
#train_df = train_df.query('parentesco1==1')
print('Training data shape: ', train_df.shape)
print('Testing data shape: ', test_df.shape)


# In[ ]:


#modelling
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score,make_scorer
import lightgbm as lgb
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, StratifiedKFold

# Custom scorer for cross validation
scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')


# In[ ]:


def macro_f1_score(labels, predictions):
    # Reshape the predictions as needed
    predictions = predictions.reshape(len(np.unique(labels)), -1 ).argmax(axis = 0)
    
    metric_value = f1_score(labels, predictions, average = 'macro')
    
    # Return is name, value, is_higher_better
    return 'macro_f1', metric_value, True


# In[ ]:


X = train_df
logR = LogisticRegression(class_weight='balanced',C=0.0005)
cv_score = cross_val_score(logR, X, y, cv = 10, scoring = scorer)
print(f'10 Fold Cross Validation F1 Score = {round(cv_score.mean(), 4)} with std = {round(cv_score.std(), 4)}')


# In[ ]:


logR.fit(X,y)
preds_log = logR.predict(test_df)
sub_log = pd.DataFrame({'Id':test_df.index, 'Target':preds_log})
sub_log.to_csv('sub_log1.csv', index=False)


# In[ ]:




