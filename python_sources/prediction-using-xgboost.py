#!/usr/bin/env python
# coding: utf-8

# # Import Required Library

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve

sns.set(style='white', context='notebook', palette='deep')


# # Load Dataset

# In[ ]:


df_train=pd.read_csv('../input/cat-in-the-dat-ii/train.csv')
df_test=pd.read_csv('../input/cat-in-the-dat-ii/test.csv')


# # analysis of data

# In[ ]:


df_train.head()
df_test.head()


# In[ ]:


df=pd.concat([df_train.drop(['target'],axis=1),df_test],axis=0,sort=False)
df


# # check for missing value
# 
# ***Simple imputer can also be to fill the missing value  "most-frequent" parameter
# 
# 
# 
# ***simple imputer consume to much time on cpu

# In[ ]:


df.isnull().sum()


# In[ ]:


g=sns.countplot(df_train['target'])


# In[ ]:


#first find the the corr between diff col
df.corr().abs()


# In[ ]:


df_train.groupby(['target','bin_0']).count().id


# In[ ]:


df_train.groupby(['target','bin_1']).count().id


# In[ ]:


df_train.groupby(['target','bin_2']).count().id


# ### from the above observation we conclude :data is linear separable we cannot find any type of outliear
# 
# 

# # How we fill such large missing value

# In[ ]:


print(df['bin_0'].describe())
print(df['bin_0'].value_counts())
print(df['bin_0'].isnull().sum())


g=sns.countplot(df['bin_0'])

df['bin_0'].fillna(value=0,inplace=True)


# In[ ]:


print(df['bin_1'].describe())
print(df['bin_1'].value_counts())
print(df['bin_1'].isnull().sum())
sns.countplot(df['bin_1'])
df['bin_1'].fillna(value=0,inplace=True)


# In[ ]:


print(df['bin_2'].describe())
print(df['bin_2'].value_counts())
print(df['bin_2'].isnull().sum())
sns.countplot(df['bin_2'])
df['bin_2'].fillna(value=0,inplace=True)


# In[ ]:


print(df['bin_3'].describe())
print(df['bin_3'].value_counts())
print(df['bin_3'].isnull().sum())
sns.countplot(df['bin_3'])
df['bin_3'].fillna(value='F',inplace=True)


# In[ ]:


print(df['bin_4'].describe())
print(df['bin_4'].value_counts())
print(df['bin_4'].isnull().sum())
sns.countplot(df['bin_4'])
df['bin_4'].fillna(value='N',inplace=True)


# In[ ]:


print(df['nom_0'].describe())
print(df['nom_0'].value_counts())
print(df['nom_0'].isnull().sum())
sns.countplot(df['nom_0'])
df['nom_0'].fillna(value='Red',inplace=True)


# In[ ]:


print(df['nom_1'].describe())
print(df['nom_1'].value_counts())
print(df['nom_1'].isnull().sum())
sns.countplot(df['nom_1'])
df['nom_1'].fillna(value='Triangle',inplace=True)


# In[ ]:


print(df['nom_2'].describe())
print(df['nom_2'].value_counts())
print(df['nom_2'].isnull().sum())
sns.countplot(df['nom_2'])
df['nom_2'].fillna(value='Hamster',inplace=True)


# In[ ]:


df['nom_2'].unique()


# In[ ]:


print(df['nom_3'].describe())
print(df['nom_3'].value_counts())
print(df['nom_3'].isnull().sum())
sns.countplot(df['nom_3'])
df['nom_3'].fillna(value='India',inplace=True)


# In[ ]:


print(df['nom_4'].describe())
print(df['nom_4'].value_counts())
print(df['nom_4'].isnull().sum())
sns.countplot(df['nom_4'])
df['nom_4'].fillna(value='Theremin',inplace=True)


# In[ ]:


print(df['nom_5'].describe())
print(df['nom_5'].value_counts())
print(df['nom_5'].isnull().sum())
#sns.countplot(df['nom_5'])
df['nom_5'].fillna(value='360a16627',inplace=True)


# In[ ]:


print(df['nom_6'].describe())
print(df['nom_6'].value_counts())
print(df['nom_6'].isnull().sum())
#sns.countplot(df['nom_6'])
df['nom_6'].fillna(value='9fa481341',inplace=True)


# In[ ]:


print(df['nom_7'].describe())
print(df['nom_7'].value_counts())
print(df['nom_7'].isnull().sum())
#sns.countplot(df['nom_7'])
df['nom_7'].fillna(value='86ec768cd',inplace=True)


# In[ ]:


print(df['nom_8'].describe())
print(df['nom_8'].value_counts())
print(df['nom_8'].isnull().sum())
#sns.countplot(df['nom_8'])
df['nom_8'].fillna(value='d7e75499d',inplace=True)


# In[ ]:


print(df['nom_9'].describe())
print(df['nom_9'].value_counts())
print(df['nom_9'].isnull().sum())
#sns.countplot(df['nom_9'])
df['nom_9'].fillna(value='8f3276a6e',inplace=True)


# In[ ]:


print(df['ord_0'].describe())
print(df['ord_0'].value_counts())
print(df['ord_0'].isnull().sum())
sns.countplot(df['ord_0'])
df['ord_0'].fillna(value=1,inplace=True)


# In[ ]:


print(df['ord_1'].describe())
print(df['ord_1'].value_counts())
print(df['ord_1'].isnull().sum())
sns.countplot(df['ord_1'])
df['ord_1'].fillna(value='Novice',inplace=True)


# In[ ]:


print(df['ord_2'].describe())
print(df['ord_2'].value_counts())
print(df['ord_2'].isnull().sum())
sns.countplot(df['ord_2'])
df['ord_2'].fillna(value='Freezing',inplace=True)


# In[ ]:


print(df['ord_3'].describe())
print(df['ord_3'].value_counts())
print(df['ord_3'].isnull().sum())
sns.countplot(df['ord_3'])
df['ord_3'].fillna(value='n',inplace=True)


# In[ ]:


print(df['ord_4'].describe())
print(df['ord_4'].value_counts())
print(df['ord_4'].isnull().sum())
sns.countplot(df['ord_4'])
df['ord_4'].fillna(value='N', inplace=True)


# In[ ]:


print(df['ord_5'].describe())
print(df['ord_5'].value_counts())
print(df['ord_5'].isnull().sum())
#sns.countplot(df['ord_5'])
df['ord_5'].fillna(value='Fl',inplace=True)


# In[ ]:


print(df['day'].describe())
print(df['day'].value_counts())
print(df['day'].isnull().sum())
sns.countplot(df['day'])
df['day'].fillna(inplace=True,value=3)


# In[ ]:


print(df['month'].describe())
print(df['month'].value_counts())
print(df['month'].isnull().sum())
sns.countplot(df['month'])
df['month'].fillna(value=8,inplace=True)


# In[ ]:


df.isnull().sum()


# In[ ]:


obj=LabelEncoder()
df['bin_3']=obj.fit_transform(df['bin_3'])
df['bin_4']=obj.fit_transform(df['bin_4'])
df['nom_0']=obj.fit_transform(df['nom_0'])
df['nom_2']=obj.fit_transform(df['nom_2'])
df['nom_3']=obj.fit_transform(df['nom_3'])
df['nom_4']=obj.fit_transform(df['nom_4'])
df['nom_5']=obj.fit_transform(df['nom_5'])
df['nom_6']=obj.fit_transform(df['nom_6'])
df['nom_7']=obj.fit_transform(df['nom_7'])
df['nom_8']=obj.fit_transform(df['nom_8'])
df['nom_9']=obj.fit_transform(df['nom_9'])
df['nom_1']=obj.fit_transform(df['nom_1'])
df['ord_2']=obj.fit_transform(df['ord_2'])


df['ord_1']=obj.fit_transform(df['ord_1'])
df['ord_3']=obj.fit_transform(df['ord_3'])
df['ord_4']=obj.fit_transform(df['ord_4'])
df['ord_5']=obj.fit_transform(df['ord_5'])


#df['nom_1']=df['nom_1'].map({'Trapezoid':1,'Star':2,'Circle':1,'Triangle':0,'Polygon':0,'Square':2})
#df['nom_2']=df['nom_2'].map({'Hamster':0,'Axolotl':0,'Lion':1,'Dog':1,'Cat':2,'Sanke':2})
#df['ord_2']=df['ord_2'].map({'Freezing':0,'Cold':0,'Warm':1,'Boiling Hot':1,'Lava Hot':2,'Hot':2})


# In[ ]:


df.head()


# In[ ]:


df.corr().abs()


# #  Why Normalisation

# sometime algorithm define priority on the  basis of label value:  
#     item    label
# 1.  Red     0
# 2.  Blue    1
# 3.  Green   2
# 
# Now green is considered as highest priority (This can also be calculated usin score value)
# 
# 2. 
# some column contain huge variation in their value some col contain (max value upto 3)  and other may contain (max value upto 1220) by normalistion it lies b/w -1 to 1
# 
# 3. Normalised data is easily and fast process by cpu 

# In[ ]:


#col=['nom_5','nom_6','nom_7','nom_8','nom_9','ord_3','ord_4','ord_5','day','month']
col=['bin_0','bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0','nom_1',
       'nom_2', 'nom_3', 'nom_4', 'nom_5','nom_6', 'nom_7', 'nom_8','nom_9' ,
       'ord_0', 'ord_1', 'ord_2','ord_3', 'ord_4', 'ord_5', 'day', 'month']
from sklearn.preprocessing import StandardScaler
obj=StandardScaler()
df[col]=obj.fit_transform(df[col])


# In[ ]:


df.describe()


# # Prepare  train  and test data

# In[ ]:


dtrain=df[df['id']<600000] # train data
dtest=df[df['id']>=600000] # test data


# In[ ]:


dtrain=dtrain.drop('id',axis=1)
dtest=dtest.drop('id',axis=1)


# In[ ]:


dtrain
dtest


# In[ ]:


X=dtrain # divide the train data into two part X_train and X_test
y=df_train['target']


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1,test_size=0.25)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Using Xgboost

# In[ ]:


import xgboost as xgb
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np


# In[ ]:


model = xgb.XGBClassifier(objective ='binary:logistic',
                      colsample_bytree = 0,
                      learning_rate = 0.1,
                      max_depth = 15,
                      n_estimators = 400,
                      scale_pos_weight = 2,
                      random_state = 2020,
                      subsample = 0.8)


# In[ ]:


model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False,)


# In[ ]:


preds_val = model.predict_proba(X_test)[:,1]


# In[ ]:


score = roc_auc_score(y_test ,preds_val)
print("score: %f" % (score))


# In[ ]:


model.fit(X,y)


# In[ ]:


y_pred = model.predict_proba(dtest)[:,1]


# In[ ]:


Id=pd.Series(range(600000,1000000),name='id')
Id
submission=pd.DataFrame({'id':Id,'target':y_pred})
submission.head()


# In[ ]:


submission.to_csv('submission.csv',index=False)


# In[ ]:


indices=np.argsort(model.feature_importances_)
plt.figure(figsize=(10,10))
g = sns.barplot(y=X_train.columns[indices][:40],x = model.feature_importances_[indices][:40] , orient='h')

