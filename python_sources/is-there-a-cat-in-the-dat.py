#!/usr/bin/env python
# coding: utf-8

# # Is There a Cat in the Dat ? 
# By : Hesdham Asem
# 
# ______
# 
# a simple clean data , which depend on categorical featurs , & we need to classify it to know whether there will be a cat or not
# 
# let's start by importing libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb


# then read the data

# In[ ]:


train = pd.read_csv('../input/cat-in-the-dat/train.csv')  
test = pd.read_csv('../input/cat-in-the-dat/test.csv')  

print(f'Train data Shape is {train.shape}')
print(f'Test data Shape is {test.shape}')


# 300K sample size for training & 200K for testing , great . 
# 
# now to define needed functions

# In[ ]:


def Drop(feature) :
    global data
    data.drop([feature],axis=1, inplace=True)
    data.head()
    
def UniqueAll(show_value = True) : 
    global data
    for col in data.columns : 
        print(f'Length of unique data for   {col}   is    {len(data[col].unique())} ')
        if show_value == True  : 
            print(f'unique values ae {data[col].unique()}' )
            print('-----------------------------')
            
def Encoder(feature , new_feature, drop = True) : 
    global data
    enc  = LabelEncoder()
    enc.fit(data[feature])
    data[new_feature] = enc.transform(data[feature])
    if drop == True : 
        data.drop([feature],axis=1, inplace=True)
        
def CPlot(feature) : 
    global data
    sns.countplot(x=feature, data=data,facecolor=(0, 0, 0, 0),linewidth=5,edgecolor=sns.color_palette("dark", 3))
    
def Mapp(feature , new_feature ,f_dict, drop_feature = True) : 
    global data
    data[new_feature] = data[feature].map(f_dict)
    if drop_feature == True : 
        data.drop([feature],axis=1, inplace=True)
    else :
        data.head()
def Unique(feature) : 
    global data
    print(f'Number of unique vaure are {len(list(data[feature].unique()))} which are : \n {list(data[feature].unique())}')


# ____
# 
# as usual , start with heading data to have a look to it 

# In[ ]:


train.head()


# & here is test data

# In[ ]:


test.head()


# ____
# 
# # Forming the Data
# 
# since this example depend on categorical data , we have to slice features (X) from output (y) from training data , then concatenate X from training data to features from text data . 
# 
# & this step to make same data processing (like label encoder & so ) for all features 
# 
# so first to slice X_train & X_test

# In[ ]:


X_train = train.drop(['id' , 'target'], axis=1, inplace=False)
X_test = test.drop(['id'], axis=1, inplace=False)

X_train.shape , X_test.shape


# now to concatenate them together into X

# In[ ]:


X = pd.concat([X_train , X_test])
X.shape


# how it looks ? 

# In[ ]:


X.head()


# ______
# 
# # Data Processing
# 
# 
# no we'll call it data , so it be suitable for all functions we define , which depend on global data

# In[ ]:


data = X


# now for plotting some features , to be sure its values are well represented

# In[ ]:


CPlot('bin_0')


# In[ ]:


CPlot('bin_1')


# In[ ]:


CPlot('bin_2')


# In[ ]:


CPlot('bin_3')


# _____
# 
# great . 
# 
# for bin 3 , since it got T for True & F for False , let's map it to new feature bin 03, with values 1 , 0

# In[ ]:


Mapp('bin_3' , 'bin_03' , {'T':1 , 'F':0} , True)


# how it looks now

# In[ ]:


data.head()


# plot it 

# In[ ]:


CPlot('bin_03')


# ____
# 
# we'll repeat it for bin 4 , Yes & No will be 1 & 0

# In[ ]:


Mapp('bin_4' , 'bin_04' , {'Y':1 , 'N':0} , True)


# plot it

# In[ ]:


CPlot('bin_04')


# _____
# 
# & since we use number of unique values for feature on alot of things , let's show them 

# In[ ]:


UniqueAll(False)


# looks fine , may be except few features which got a high number of unique values , so it might not be very helpful in training
# 
# _____
# 
# # Label Encoding
# 
# now we need to apply label encoding to some categorical features , so it be ready for training 
# 
# let's start with features : 'nom_0' , 'nom_1' , 'nom_2' , 'nom_3' , 'nom_4'
# 

# In[ ]:


for C in ['nom_0' , 'nom_1' , 'nom_2' , 'nom_3' , 'nom_4'] : 
    enc  = LabelEncoder()
    enc.fit(X[C])
    X[C] = enc.transform(X[C])


# how it looks now

# In[ ]:


data.head()


# plot them 

# In[ ]:


CPlot('nom_0')


# In[ ]:


CPlot('nom_1')


# In[ ]:


CPlot('nom_2')


# In[ ]:


CPlot('nom_3')


# In[ ]:


CPlot('nom_4')


# _____
# 
# for nom_5 & nom_6 , let's have a look to their unique values

# In[ ]:


Unique('nom_5')


# In[ ]:


Unique('nom_6')


# ok , we'll continue label encode them

# In[ ]:


for C in ['nom_5' , 'nom_6' , 'nom_7' , 'nom_8' , 'nom_9']: 
    enc  = LabelEncoder()
    enc.fit(X[C])
    X[C] = enc.transform(X[C])


# how it looks

# In[ ]:


data.head()


# and it might not be helpful to plot features with very high number of unique values 
# 
# ______
# 
# now to plot other features 

# In[ ]:


CPlot('ord_0')


# In[ ]:


CPlot('ord_1')


# In[ ]:


CPlot('ord_2')


# In[ ]:


CPlot('ord_3')


# In[ ]:


CPlot('ord_4')


# In[ ]:


CPlot('ord_5')


#  again to label encode them 

# In[ ]:


for C in ['ord_0' , 'ord_1' , 'ord_2' , 'ord_3' , 'ord_4' , 'ord_5']: 
    enc  = LabelEncoder()
    enc.fit(X[C])
    X[C] = enc.transform(X[C])

data.head()


# so now we are ready for Build the model & train the data 
# 
# ______
# 
# # Build the Model
# 
# first to prepare the data for training by defining trainging & testing data again 
# 

# In[ ]:


train_data = data.iloc[:train.shape[0],:]
test_data=  data.iloc[train.shape[0]:,:]
train_data.shape , test_data.shape


# now to define X & y 

# In[ ]:


X = train_data
y = train['target']
X.shape , y.shape


# let's apply minmaxscalerfrom sklearn , to make the model faster 

# In[ ]:


scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
X = scaler.fit_transform(X)


# then to split it into training & testing data

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44, shuffle =True)

print('X_train shape is ' , X_train.shape)
print('X_test shape is ' , X_test.shape)
print('y_train shape is ' , y_train.shape)
print('y_test shape is ' , y_test.shape)


# now it's time to apply lgb classification model , with round numbers = 25K & shown parameters

# In[ ]:


num_round = 25000

parameters = {'num_leaves': 128,
             'min_data_in_leaf': 20, 
             'objective':'binary',
             'max_depth': 8,
             'learning_rate': 0.001,
             "min_child_samples": 20,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9 ,
             "bagging_seed": 44,
             "metric": 'auc',
             "verbosity": -1}


traindata = lgb.Dataset(X_train, label=y_train)
testdata = lgb.Dataset(X_test, label=y_test)

LGBModel = lgb.train(parameters, traindata, num_round, valid_sets = [traindata, testdata],
                     verbose_eval=50, early_stopping_rounds = 600)


# accuracy might be better by using more round numbers 
# 
# _____
# 
# now to predict test data , but first we have to apply same scaler model to test data

# In[ ]:


test = scaler.transform(test_data)
test.shape


# ok , now predicting testing data

# In[ ]:


y_pred = LGBModel.predict(test)
y_pred.shape


# how it looks like ? 

# In[ ]:


y_pred[:10]


# great , now to open sample_submission , to read id columns from it

# In[ ]:


data = pd.read_csv('../input/cat-in-the-dat/sample_submission.csv')  

print(f'Test data Shape is {data.shape}')
data.head()


# at last we concatenate id column with the result

# In[ ]:


idd = data['id']
FinalResults = pd.DataFrame(y_pred,columns= ['target'])
FinalResults.insert(0,'id',idd)
FinalResults.head()


# & export the result file

# In[ ]:


FinalResults.to_csv("sample_submission.csv",index=False)


# In[ ]:





# In[ ]:




