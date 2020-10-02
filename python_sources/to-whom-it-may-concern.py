#!/usr/bin/env python
# coding: utf-8

# # The things you are about to see are the ramblings of a mad man. If any of this triggers a lightbulb at any point you may be mad as well. Enjoy 
# *** For those with a short attention span scroll down till you see the flashing stop sign.***

# ![](https://media.giphy.com/media/X6w1HKaFXLAAw/giphy.gif)

# #  Predict which water pumps are faulty

# **The goal is to identify with above 60% accuracy which water wells are faulty or non functional.** I will be using data from Taarifa and the Tanzanian Ministry of Water. The submission of my predictions will be in the format of .CSV with columns for 'id' as well as 'status_group'. Lets start by loading the data and getting a feel for it. 

# In[ ]:


import pandas as pd


# In[ ]:


df_feats = pd.read_csv('../input/ds1-kaggle-challenge/train_features.csv')
df_label = pd.read_csv('../input/ds1-kaggle-challenge/train_labels.csv') 


# In[ ]:


df_label.describe(include='object')


# **The df_label data frame contains the id along with status of the well. The status will be our target.**

# In[ ]:


#let us see the whole column profile of the data frame 
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
df_feats.head()


# In[ ]:


df_feats.shape, df_label.shape


# **The df_feats Data Frame contains all the features we will use to predict the status of any given well.** For the first iteration I will run a simple baseline. We see the shape of the features df is 59400 by 40 and the shape of the label df is 59400 by 2. Lets check the distribution of the status of the wells. 

# In[ ]:


df_label.status_group.value_counts(normalize = True)


# **Overall it appears the majority of wells are functional(54.3%), non functional is the second highest(38.42%). Functional needing repair rounds out the data set(7.26%).** If I were to make a blind prediction saying that all the wells were functional I would be correct around 54 percent of the time. Not bad but not nearly conclusive or useful for the real world. Lets dig deeper. 

# For simplicity I will combine the two data frames to process before splitting. 
# 

# In[ ]:


full = pd.DataFrame.merge(df_label,df_feats)


# In[ ]:


full.head()


# In[ ]:


full.isnull().sum()


# **After merging the data frames together I find there are some Null values that may skew our results or otherwise break our functions during the process.** For now we will drop all instances that are missing values, but later we may impute some values to help our models predict better if neeeded.  

# In[ ]:


clean = full.dropna(axis = 1)


# In[ ]:


clean.isna().sum()


# **Now that we have no NaN values lets make some test and tarin sets with our data. We want to predict status so we will call that the 'y' variable. All other features will be called our 'X' matrix of features.** 

# In[ ]:


from sklearn.model_selection import train_test_split
X1 = clean.drop(columns = ['status_group',], axis = 1)
y = clean['status_group']
X_train, X_test, y_train, y_test = train_test_split(X1, y,test_size = .5, random_state=42)


# In[ ]:


X_train.head()


# In[ ]:


X_train.isna().sum().sum()


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import category_encoders as ce
import numpy as np 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df
      


# In[ ]:


X_train_DC = dummyEncode(X_train)
X_train_DC.head()
X_test_DC = dummyEncode(X_test)
X_test_DC.head()
X = dummyEncode(X1)


# In[ ]:


X_train_DC.isna().sum().sum()
X_train_DC.shape


# In[ ]:


model= LogisticRegression()
model.fit(X_train_DC, y_train)
y_pred = model.predict(X_test_DC)
accuracy_score(y_test, y_pred)


# In[ ]:


pipeline = make_pipeline(ce.OneHotEncoder(use_cat_names=True),
                         StandardScaler(), LogisticRegression(solver ='lbfgs',n_jobs=-1, multi_class = 'auto',C=2))
pipeline.fit(X_train_DC, y_train)


# In[ ]:


y_pred = pipeline.predict(X_train)


# In[ ]:


pred = pd.DataFrame(y_pred, X_train_DC['id'])


# In[ ]:


pred.columns = ['status_group']


# In[ ]:


pred.head()
pred.shape
pred.head()


# In[ ]:


newsub = pd.DataFrame(pred)
newsub.shape
sub_2 = newsub.index
subm = pd.DataFrame( newsub['status_group'],sub_2)
subm.head()
subm.reset_index(inplace = True)


# In[ ]:


#subm.to_csv('C:/Users/dakot/Documents/GitHub/sumbission1.csv',columns = ['id','status_group'], index = False )


# In[ ]:


subm.shape


# Now to make it work for the actual test set. 
# 

# In[ ]:


df_test = pd.read_csv('../input/ds1-kaggle-challenge/test_features.csv') 


# In[ ]:


df_test.head()


# In[ ]:


df_test.isna().sum()
nona = df_test.dropna(axis = 1)


# In[ ]:


nona.shape


# In[ ]:


X = dummyEncode(nona)


# In[ ]:


X.head()


# In[ ]:


pipeline.fit(X_test, y_test)
y_preds = pipeline.predict(X)


# In[ ]:


y_preds.shape


# In[ ]:


preds = pd.DataFrame(y_preds, X['id'])
preds.columns = ['status_group']
preds.head()


# In[ ]:


newsubs = pd.DataFrame(preds)
newsubs.shape
sub_2s = newsubs.index
subms = pd.DataFrame( newsubs['status_group'],sub_2s)
subms.head()
subms.reset_index(inplace = True)


# In[ ]:


subms.head()


# **The above got me a baseline of .53754 on kaggle. we can do better than that. **

# In[ ]:


from sklearn import tree
from sklearn.metrics import classification_report
 
clf = tree.DecisionTreeClassifier(random_state=42)
clf = clf.fit(X_train, y_train)
 
y_pred2 = clf.predict(X)
#print(classification_report(y_test, y_pred2))
#print('\nAccuracy: {0:.4f}'.format(accuracy_score(y_test, y_pred2)))


# In[ ]:


y_pred2.shape


# Lets try to automate the formatting for submission. 

# In[ ]:


def format(predictions):
    pre = pd.DataFrame(predictions, X['id'])
    pre.columns = ['status_group']
    new = pd.DataFrame(pre)
    sub_2s = new.index
    subs = pd.DataFrame( new['status_group'],sub_2s)
    subs.reset_index(inplace = True)
    print(subs.head(),subs.shape)
    subs.to_csv('C:/Users/dakot/Documents/GitHub/sumbission1.csv',columns = ['id','status_group'], index = False )
    return 'YAY!'


# # Decision Tree Classifier leads!
# kaggle score for the tree without a pipline  = 0.71054
# Now lets pipeline this baby!

# In[ ]:


pipeline = make_pipeline(ce.OneHotEncoder(use_cat_names=True),
                         StandardScaler(), LogisticRegression(solver ='lbfgs',n_jobs=-1, multi_class = 'auto',C=2))
pipeline.fit(X_train, y_train)


# In[ ]:


pred3 = pipeline.predict(X)


# In[ ]:


pred3


# **Standard scaled one hot encoded log_reg =  0.63769** 

# ## Ok for real this time decision tree in a pipeline 
# 

# In[ ]:


treepipe = make_pipeline(ce.OneHotEncoder(use_cat_names=True),
                         StandardScaler(),tree.DecisionTreeClassifier(random_state=42) )
treepipe.fit(X_train, y_train)


# In[ ]:


tpred = treepipe.predict(X_test)
print(accuracy_score(y_test,tpred))
pred4 = treepipe.predict(X)


# **Score = 0.71040**

# In[ ]:


from sklearn.preprocessing import RobustScaler
treepipe2 = make_pipeline(ce.OneHotEncoder(use_cat_names=True),
                         RobustScaler(),tree.DecisionTreeClassifier(random_state=42) )
treepipe2.fit(X_train, y_train)
pred = treepipe.predict(X_test)


# In[ ]:


accuracy_score(y_test,pred)


# In[ ]:


pred5 = treepipe2.predict(X)


# In[ ]:


pred5


# **Score = 0.71040**

# # Ok this far I've dropped all rows with NAN's, lets fix some of the columns and see if that helps

# In[ ]:


# the training data set 
full.isna().sum()


# # MONKEY PATCHING TIME!!!!

# In[ ]:


full.funder.fillna(full.funder.describe().top,inplace = True)
full.installer.fillna(full.installer.describe().top,inplace = True)
full.subvillage.fillna(full.subvillage.describe().top, inplace = True)
full.public_meeting.fillna(full.public_meeting.describe().top,inplace = True)
full.scheme_management.fillna(full.scheme_management.describe().top, inplace = True)
full.scheme_name.fillna(full.scheme_name.describe().top, inplace = True)
full.permit.fillna(full.permit.describe().top,inplace = True)


# In[ ]:


full.isna().sum().sum()


# In[ ]:


full.columns


# **Monkey patch complete on training data. Lets see the effects**

# In[ ]:


Xi = full.drop(columns= ['status_group','date_recorded'], axis = 1)
yi = full['status_group']


# In[ ]:


Xi.shape, yi.shape


# In[ ]:


# DJ split that S*&&%
X_train, X_test, y_train, y_test = train_test_split(Xi, yi,test_size = .5, random_state=42)
#now encode it
X_trains = dummyEncode(X_train)
X_tests = dummyEncode(X_test)


# In[ ]:


#how does it like the trees
cl = tree.DecisionTreeClassifier(random_state=42)
cl = clf.fit(X_trains, y_train)
 
y_predictor = clf.predict(X_tests)
print(classification_report(y_test, y_predictor))
print('\nAccuracy: {0:.4f}'.format(accuracy_score(y_test, y_predictor)))
#accuracy of .699 for the train data when split how about the test data


# # Monkey Patch All The DATAS!!!!!!

# In[ ]:


test = df_test
print(test.shape)
test.funder.fillna(test.funder.describe().top,inplace = True)
test.installer.fillna(test.installer.describe().top,inplace = True)
test.subvillage.fillna(test.subvillage.describe().top, inplace = True)
test.public_meeting.fillna(test.public_meeting.describe().top,inplace = True)
test.scheme_management.fillna(test.scheme_management.describe().top, inplace = True)
test.scheme_name.fillna(test.scheme_name.describe().top, inplace = True)
test.permit.fillna(test.permit.describe().top,inplace = True)


# In[ ]:


test.head()


# In[ ]:


Xt = test.drop(columns = ['date_recorded'], axis = 1)
XT = dummyEncode(Xt)
#TREE ME!!!
cl = tree.DecisionTreeClassifier(random_state=42)
cl = clf.fit(X_trains, y_train)
 
y_predictors = clf.predict(XT)
# print(classification_report(y_test, y_predictor))
# print('\nAccuracy: {0:.4f}'.format(accuracy_score(y_test, y_predictor)))


# **Your submission scored 0.68881**

# In[ ]:


#lets hit the pipe testing with standard scale then robust, log_reg and tree
logpipe = make_pipeline(RobustScaler(),
                        tree.DecisionTreeClassifier(random_state=42) )
logpipe.fit(X_trains, y_train)
predlog = logpipe.predict(X_test)
accuracy_score(y_test,predlog)
# yeah .64 is no bueno with standard scaler logistic regression
#robust scale log_regression is.63 which doesnt tickle my fancy 
# Standard scale D tree gives .69 but im not impressed
#robust scale Dtree gives a slightly higher .699


# In[ ]:


# What about different encoding?
import category_encoders as ce
encoder = ce.HashingEncoder()
hashingpipe = make_pipeline(ce.HashingEncoder(),RobustScaler(),
                        tree.DecisionTreeClassifier(random_state=42) )
hashingpipe.fit(X_train, y_train)
predlogs = hashingpipe.predict(X_test)
accuracy_score(y_test,predlogs)


# # Allright so far im not breaking through this barrier. So far the best score came from dropping all the nan values and a decision tree. Time for a change of thought..  
# 

# ![](https://media1.tenor.com/images/b9f65b516415511fcca2ffcca76d7c8e/tenor.gif?itemid=12454910)![image.png](attachment:image.png)

# In[ ]:


df_test1 = pd.read_csv('../input/ds1-kaggle-challenge/test_features.csv') 


# # Lets find Null values and replace them or send them to the dumpster fire out back. 

# ![image.png](attachment:image.png)

# **Some values could potentially be zero, but would be quite unlikely. For instance the hieght of the gps recorded in "gps_hieght" COULD be exactly 0.0 but is highly unlikely and is probably the effect of someone just not recording an actual hieght. So lets fix those instances!**

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
df_test1.isna().sum()
df_test1['gps_height'].replace(0.0, np.nan, inplace=True)
df_test1['population'].replace(0.0, np.nan, inplace=True)
df_test1['amount_tsh'].replace(0.0, np.nan, inplace=True)
df_test1.isnull().sum()


# ** Better results were found when grouping subsets together then applying the mean or median of those groupings to the missing values. The code below contains many iterations worth of groupings and modifications, some are useful in the position they're in and others are not but as a whole this code block is operational and the outcome is the one expected. Also a column has been added ('years_service') to track the years since construction because the older the well the more likely it is fatigued and will fail. **

# In[ ]:


df_test1['gps_height'].fillna(df_test1.groupby(['region', 'district_code'])['gps_height'].transform('mean'), inplace=True)
df_test1['gps_height'].fillna(df_test1.groupby(['region'])['gps_height'].transform('mean'), inplace=True)
df_test1['gps_height'].fillna(df_test1['gps_height'].mean(), inplace=True)
df_test1['population'].fillna(df_test1.groupby(['region', 'district_code'])['population'].transform('median'), inplace=True)
df_test1['population'].fillna(df_test1.groupby(['region'])['population'].transform('median'), inplace=True)
df_test1['population'].fillna(df_test1['population'].median(), inplace=True)
df_test1['amount_tsh'].fillna(df_test1.groupby(['region', 'district_code'])['amount_tsh'].transform('median'), inplace=True)
df_test1['amount_tsh'].fillna(df_test1.groupby(['region'])['amount_tsh'].transform('median'), inplace=True)
df_test1['amount_tsh'].fillna(df_test1['amount_tsh'].median(), inplace=True)
df_test1.isnull().sum()
features=['amount_tsh', 'gps_height', 'population']
scaler = MinMaxScaler(feature_range=(0,20))
df_test1[features] = scaler.fit_transform(df_test1[features])
df_test1[features].head(20)
df_test1.isna().sum()
df_test1['longitude'].replace(0.0, np.nan, inplace=True)
df_test1['latitude'].replace(0.0, np.nan, inplace=True)
df_test1['construction_year'].replace(0.0, np.nan, inplace=True)
df_test1['latitude'].fillna(df_test1.groupby(['region', 'district_code'])['latitude'].transform('mean'), inplace=True)
df_test1['longitude'].fillna(df_test1.groupby(['region', 'district_code'])['longitude'].transform('mean'), inplace=True)
df_test1['longitude'].fillna(df_test1.groupby(['region'])['longitude'].transform('mean'), inplace=True)
df_test1['construction_year'].fillna(df_test1.groupby(['region', 'district_code'])['construction_year'].transform('median'), inplace=True)
df_test1['construction_year'].fillna(df_test1.groupby(['region'])['construction_year'].transform('median'), inplace=True)
df_test1['construction_year'].fillna(df_test1.groupby(['district_code'])['construction_year'].transform('median'), inplace=True)
df_test1['construction_year'].fillna(df_test1['construction_year'].median(), inplace=True)
df_test1['date_recorded'] = pd.to_datetime(df_test1['date_recorded'])
df_test1['years_service'] = df_test1.date_recorded.dt.year - df_test1.construction_year
print(df_test1.isnull().sum())


# # Alright thats a lot of code for a little cleaning. Lets dump some trash 

# ** These categories called Garbage were the features I found to be redundant, colinear, or otherwise too far gone to be worth my limited time in this competition. At some point I will revisit these columns to see which can be spared when time is less restricted. **

# In[ ]:


garbage=['wpt_name','num_private','subvillage','region_code','recorded_by','management_group',
         'extraction_type_group','extraction_type_class','scheme_name','payment',
        'quality_group','quantity_group','source_type','source_class','waterpoint_type_group',
        'ward','public_meeting','permit','date_recorded','construction_year']
df_test1.drop(garbage,axis=1, inplace=True)


# When looking through columns individually I noticed some variation in placement of capital letters so lets fix that. 

# In[ ]:


#take out any random capital letters in the entries
df_test1.waterpoint_type = df_test1.waterpoint_type.str.lower()
df_test1.funder = df_test1.funder.str.lower()
df_test1.basin = df_test1.basin.str.lower()
df_test1.region = df_test1.region.str.lower()
df_test1.source = df_test1.source.str.lower()
df_test1.lga = df_test1.lga.str.lower()
df_test1.management = df_test1.management.str.lower()
df_test1.quantity = df_test1.quantity.str.lower()
df_test1.water_quality = df_test1.water_quality.str.lower()
df_test1.payment_type=df_test1.payment_type.str.lower()
df_test1.extraction_type=df_test1.extraction_type.str.lower()


# In[ ]:


df_test1.columns


# # A few more lose ends to tie up...

# In[ ]:


df_test1["funder"].fillna("other", inplace=True)
df_test1["scheme_management"].fillna("other", inplace=True)
df_test1["installer"].fillna("other", inplace=True)
df_test1.isna().sum()


# ![image.png](attachment:image.png)

# In[ ]:


df_test1.head()
df_test1.shape


# **It was at this point I realized I was working on my test data and not the train, so now in order to test the models I need to clean the training data. **

# # Now that process again on the training data....YAY!
# lets try to automate that mess 

# In[ ]:


#AUTOMATE ALL THE THINGS!!!
def MrClean(df):
    df_t= df
    df_t['gps_height'].replace(0.0, np.nan, inplace=True)
    df_t['population'].replace(0.0, np.nan, inplace=True)
    df_t['amount_tsh'].replace(0.0, np.nan, inplace=True)
    df_t['gps_height'].fillna(df_t.groupby(['region', 'district_code'])['gps_height'].transform('mean'), inplace=True)
    df_t['gps_height'].fillna(df_t.groupby(['region'])['gps_height'].transform('mean'), inplace=True)
    df_t['gps_height'].fillna(df_t['gps_height'].mean(), inplace=True)
    df_t['population'].fillna(df_t.groupby(['region', 'district_code'])['population'].transform('median'), inplace=True)
    df_t['population'].fillna(df_t.groupby(['region'])['population'].transform('median'), inplace=True)
    df_t['population'].fillna(df_t['population'].median(), inplace=True)
    df_t['amount_tsh'].fillna(df_t.groupby(['region', 'district_code'])['amount_tsh'].transform('median'), inplace=True)
    df_t['amount_tsh'].fillna(df_t.groupby(['region'])['amount_tsh'].transform('median'), inplace=True)
    df_t['amount_tsh'].fillna(df_t['amount_tsh'].median(), inplace=True)
    features=['amount_tsh', 'gps_height', 'population']
    scaler = MinMaxScaler(feature_range=(0,20))
    df_t[features] = scaler.fit_transform(df_t[features])
    df_t['longitude'].replace(0.0, np.nan, inplace=True)
    df_t['latitude'].replace(0.0, np.nan, inplace=True)
    df_t['construction_year'].replace(0.0, np.nan, inplace=True)
    df_t['latitude'].fillna(df_t.groupby(['region', 'district_code'])['latitude'].transform('mean'), inplace=True)
    df_t['longitude'].fillna(df_t.groupby(['region', 'district_code'])['longitude'].transform('mean'), inplace=True)
    df_t['longitude'].fillna(df_t.groupby(['region'])['longitude'].transform('mean'), inplace=True)
    df_t['construction_year'].fillna(df_t.groupby(['region', 'district_code'])['construction_year'].transform('median'), inplace=True)
    df_t['construction_year'].fillna(df_t.groupby(['region'])['construction_year'].transform('median'), inplace=True)
    df_t['construction_year'].fillna(df_t.groupby(['district_code'])['construction_year'].transform('median'), inplace=True)
    df_t['construction_year'].fillna(df_t['construction_year'].median(), inplace=True)
    df_t['date_recorded'] = pd.to_datetime(df_t['date_recorded'])
    df_t['years_service'] = df_t.date_recorded.dt.year - df_t.construction_year
   
    garbage=['wpt_name','num_private','subvillage','region_code','recorded_by','management_group',
         'extraction_type_group','extraction_type_class','scheme_name','payment',
        'quality_group','quantity_group','source_type','source_class','waterpoint_type_group',
        'ward','public_meeting','permit','date_recorded','construction_year']
    df_t.drop(garbage,axis=1, inplace=True)
    df_t.waterpoint_type = df_t.waterpoint_type.str.lower()
    df_t.funder = df_t.funder.str.lower()
    df_t.basin = df_t.basin.str.lower()
    df_t.region = df_t.region.str.lower()
    df_t.source = df_t.source.str.lower()
    df_t.lga = df_t.lga.str.lower()
    df_t.management = df_t.management.str.lower()
    df_t.quantity = df_t.quantity.str.lower()
    df_t.water_quality = df_t.water_quality.str.lower()
    df_t.payment_type=df_t.payment_type.str.lower()
    df_t.extraction_type=df_t.extraction_type.str.lower()
    df_t["funder"].fillna("other", inplace=True)
    df_t["scheme_management"].fillna("other", inplace=True)
    df_t["installer"].fillna("other", inplace=True)
    return df_t


# In[ ]:


#Full is the df of both the train_features csv and train_labels merged 
full = pd.DataFrame.merge(df_label,df_feats)
full.shape
print(full.columns)


# In[ ]:


#Call out mrclean!
soclean =  MrClean(full)


# ![](https://media1.tenor.com/images/e92324f31edc80fe7beb1c80194de76f/tenor.gif?itemid=9643872)

# In[ ]:


soclean.head()
soclean.isna().sum()


# # Mr. Clean did his job, no missing values! 

# ![](https://media1.giphy.com/media/ujvW8qiDfbCJRnIHPq/giphy.gif?cid=3640f6095c5c88d06450675932107813)

# # TIME FOR TREES

# In[ ]:


yc = soclean['status_group']
Xc = soclean


# In[ ]:


Xc.head()


# In[ ]:


Xc.drop(columns = ['status_group'], axis = 1, inplace = True)


# In[ ]:


Xc.columns


# In[ ]:


#split this ish 
X_train, X_test, y_train, y_test = train_test_split(Xc, yc,test_size = .2, random_state=42)


# In[ ]:


X_train.head()


# In[ ]:


# TREES!!!
cleanpipe = make_pipeline(ce.OneHotEncoder(use_cat_names=True),StandardScaler(),
                        tree.DecisionTreeClassifier(random_state=42) )
cleanpipe.fit(X_train, y_train)
preds = cleanpipe.predict(X_test)
accuracy_score(y_test,preds)


# In[ ]:


preddi = cleanpipe.predict(df_test1)


# In[ ]:


preddi.shape


# **0.71973 now for some other models**

# # Lets set up a different way, it may make it easier to work with. 

# In[ ]:


full = pd.DataFrame.merge(df_label,df_feats)
full.shape
print(full.columns)
soclean =  MrClean(full)
train = soclean
test = df_test1


# In[ ]:


train.shape,test.shape


# In[ ]:


target = train.pop('status_group')
train['train']=1
test['train']=0


# In[ ]:


combo = pd.concat([train, test])
combo.info()


# In[ ]:


combo['funder'] = pd.factorize(combo['funder'])[0]
combo['installer'] = pd.factorize(combo['installer'])[0]
combo['scheme_management'] = pd.factorize(combo['scheme_management'])[0]
combo['extraction_type'] = pd.factorize(combo['extraction_type'])[0]
combo['management'] = pd.factorize(combo['management'])[0]
combo['payment_type'] = pd.factorize(combo['payment_type'])[0]
combo['water_quality'] = pd.factorize(combo['water_quality'])[0]
combo['quantity'] = pd.factorize(combo['quantity'])[0]
combo['source'] = pd.factorize(combo['source'])[0]
combo['waterpoint_type'] = pd.factorize(combo['waterpoint_type'])[0]
combo['basin'] = pd.factorize(combo['basin'])[0]
combo['region'] = pd.factorize(combo['region'])[0]
combo['lga'] = pd.factorize(combo['lga'])[0]
combo['district_code'] = pd.factorize(combo['district_code'])[0]
combo['years_service'] = pd.factorize(combo['years_service'])[0]
combo.head()


# In[ ]:


train_df = combo[combo["train"] == 1]
test_df = combo[combo["train"] == 0]
train_df.drop(["train"], axis=1, inplace=True)
train_df.drop(['id'],axis=1, inplace=True)
test_df.drop(["train"], axis=1, inplace=True)


# In[ ]:


X = train_df
y = target


# In[ ]:


X.shape,y.shape


# In[ ]:


y.head()


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
model_rfc = RandomForestClassifier(n_estimators=1000, n_jobs = -1)


# In[ ]:


score = cross_val_score(model_rfc, X, y, cv=3, n_jobs = -1)


# In[ ]:


score.mean()


# In[ ]:


X_test=test_df
X_test.shape


# In[ ]:


model_rfc.fit(X,y)


# In[ ]:


X.info()
importances = model_rfc.feature_importances_
importances


# In[ ]:


X_test.shape, X.shape


# In[ ]:


a=X_test['id']
X_test.drop(['id'],axis=1, inplace=True)
y_pred = model_rfc.predict(X_test)


# In[ ]:


y_pred


# In[ ]:


a.head()


# In[ ]:


y_pred.shape,a.shape


# In[ ]:


y_pred=pd.DataFrame(y_pred)
y_pred['id']=a
y_pred.columns=['status_group','id']
y_pred=y_pred[['id','status_group']]


# In[ ]:


y_pred.head()


# # Random forest gets the lead. 0.81487

# In[ ]:


from xgboost import XGBClassifier
modelxgb = XGBClassifier(objective = 'multi:softmax', booster = 'gbtree', nrounds = 'min.error.idx', 
                      num_class = 4, maximize = False, eval_metric = 'merror', eta = .2,
                      max_depth = 14, colsample_bytree = .4)


# In[ ]:


#print(cross_val_score(modelxgb, X, y, cv=3,n_jobs = -1))
modelxgb.fit(X,y)


# In[ ]:


y_preds = modelxgb.predict(X_test)


# In[ ]:


y_preds=pd.DataFrame(y_preds)
y_preds['id']=a
y_preds.columns=['status_group','id']
y_preds=y_preds[['id','status_group']]


# In[ ]:


y_preds.shape


# In[ ]:


y_preds.head()


# # XGboost falls short with 81292

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1000)


# In[ ]:


scores = (cross_val_score(knn, X, y, cv=3,n_jobs = -1))
scores.mean()


# # With that low of cross vals I wont even submit the KNN 

# In[ ]:


cl = tree.DecisionTreeClassifier(random_state=42)


# In[ ]:


cl.fit(X,y)


# In[ ]:


y_predcl = cl.predict(X_test)


# In[ ]:


y_predcl=pd.DataFrame(y_predcl)
y_predcl['id']=a
y_predcl.columns=['status_group','id']
y_predcl=y_predcl[['id','status_group']]


# In[ ]:


y_predcl.head()


# In[ ]:


y_predcl.shape


# # Decision tree classifier gets 0.70566

# In[ ]:


log = LogisticRegression(solver ='saga',n_jobs=-1, multi_class = 'auto',C=1.0)


# In[ ]:


print(cross_val_score(log, X, y, cv=3,n_jobs = -1))
log.fit(X,y)


# In[ ]:


y_predlog =log.predict(X_test)


# In[ ]:


y_predlog=pd.DataFrame(y_predlog)
y_predlog['id']=a
y_predlog.columns=['status_group','id']
y_predlog=y_predlog[['id','status_group']]


# In[ ]:


y_predlog.head()


# # logistic regression = 0.61359

# In[ ]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score


# In[ ]:


clf = ExtraTreesClassifier(n_estimators=500, max_depth=None,
                           min_samples_split=10, random_state=0)
scores = cross_val_score(clf, X, y, cv=5)


# In[ ]:


scores.mean()


# clf = RandomForestClassifier(n_estimators=2000, max_depth=None,
#                              min_samples_split=10, random_state=0  
#                              )
# scores = cross_val_score(clf, X, y, cv=5)
# scores.mean()    
# 0.8134174558068722
# 
# clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
#                              min_samples_split=10, random_state=0 , 
#                              n_jobs=-1)
# scores = cross_val_score(clf, X, y, cv=5)
# scores.mean()                               
# 0.8135689539533196
# 
# 

# In[ ]:


clf = RandomForestClassifier(n_estimators=1000, max_depth=None,
                             min_samples_split=8, random_state=0 , 
                             n_jobs=-1)
scores = cross_val_score(clf, X, y, cv=5)
scores.mean()                               


# In[ ]:


clf.fit(X,y)


# In[ ]:


pred_rfc =clf.predict(X_test)


# In[ ]:


pred_rfc=pd.DataFrame(pred_rfc)
pred_rfc ['id']=a
pred_rfc .columns=['status_group','id']
pred_rfc = pred_rfc[['id','status_group']]


# In[ ]:


pred_rfc.head()


# In[ ]:


from xgboost import XGBClassifier
modelxgb = XGBClassifier(objective = 'multi:softmax', booster = 'gbtree', nrounds = 'min.error.idx', 
                      num_class = 3, maximize = False, eval_metric = 'merror', eta = .1,
                      max_depth = 14, colsample_bytree = .4)


# In[ ]:


score = (cross_val_score(modelxgb, X, y, cv=5,n_jobs = -1))
score.mean()


# In[ ]:


modelxgb.fit(X,y)


# In[ ]:


predict = modelxgb.predict(X_test)


# In[ ]:


predict=pd.DataFrame(predict)
predict ['id']=a
predict .columns=['status_group','id']
predict = predict[['id','status_group']]


# In[ ]:


predict.head()


# # XGB Score = 0.81292

# # Thanks for reading and be sure to vote!

# ![](https://media3.giphy.com/media/hLPNDUZ3ntKEM/200w.webp?cid=3640f6095c5c99ae696f576c41729322)

# In[ ]:




