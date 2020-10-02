#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score,f1_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from imblearn.over_sampling import SMOTE


# In[ ]:


train=pd.read_csv('train.csv')


# In[ ]:


train.iloc[:,-1:]


# In[ ]:


train['REPORTEDDT']=pd.to_datetime(train['REPORTEDDT'],format="%Y-%m-%d")
train['LOSSDT']=pd.to_datetime(train['LOSSDT'],format="%Y-%m-%d")

train['Days']=train['REPORTEDDT']-train['LOSSDT']

train['Days']=train['Days']/np.timedelta64(1, 'D')

train[train['Days']<0]

train.drop(columns=['REPORTEDDT','LOSSDT'],inplace=True)
train['Days']=train['Days'].abs()

from dateutil.parser import parse

from datetime import date, timedelta
train1=train[train['BIRTHDT'].notnull()]
train1['age'] = train1.apply(lambda x:(date.today() - parse(x['BIRTHDT']).date()) // timedelta(days=365.2425),axis=1)

train1.loc[(train1['age']>120) | (train1['age']<0),'age']=int(train1[(train1['age']<120) & (train1['age']>0)]['age'].mean())

train.loc[train['BIRTHDT'].notnull(),'age']=train1['age'].values

train['age'].fillna(int(train['age'].mean()),inplace=True)
train.drop('BIRTHDT',axis=1,inplace=True)

train.drop(columns=['CLAIMNO'],inplace=True)
train['distance']=np.sqrt( (train['INSUREDLAT']-train['CLMNT_LAT'])**2 + (train['INSUREDLON']-train['CLMNT_LON'])**2 )
print(train.columns)

train.drop(columns=['INSUREDNA_cleaned_root','CLMNT_LAT','CLMNT_LON','INSUREDLAT','INSUREDLON','CLMNT_NA_cleaned_root','Prov_Name_All_final_root','N_PAYTO_NAME_cleaned_root','N_REFRING_PHYS_final_root','N_PRVDR_NAME_NONPHYS_cleaned_root'],inplace=True)
for i in train.columns:
    if(train[i].dtypes =='object'):
        train[i].fillna(train[i].mode()[0],inplace=True)
    else:
        train[i].fillna(int(train[i].mean()),inplace=True)


                        
train['SICCD']=train['SICCD'].replace('\t','nan')

for i in train.columns:
    print(i,train[i].nunique())

train.values[0]
rest=train[['Days','age','distance']]

# label=train[['SICCD','NATUREOFINJURYCD','LOSSSTATECD','WORK_LOC_STATECD','CLMNT_STATECD','INSUREDSTATECD']]

# label_encoder = preprocessing.LabelEncoder() 
# # Encode labels in column 'species'. 
# label=label.apply(label_encoder.fit_transform)

dummy=pd.get_dummies(train[['CAUSEOFINJURY_FLTRCD','BODYPART_FLTRCD','RTRN_TOWORKIND','SICCD','NATUREOFINJURYCD','LOSSSTATECD','WORK_LOC_STATECD','CLMNT_STATECD','INSUREDSTATECD']])

final=pd.concat([dummy,rest],axis=1)


# In[ ]:


haha=train[train['TARGET']==1]


# In[ ]:


haha.reset_index(inplace=True)


# In[ ]:


count=0
for i in range(len(haha)):
        if(haha.loc[i,'LOSSSTATECD']==haha.loc[i,'INSUREDSTATECD']):
            count+=1


# In[ ]:


count/len(haha)


# In[ ]:


count1=0
for i in range(len(train)):
        if(train.loc[i,'LOSSSTATECD']==train.loc[i,'INSUREDSTATECD']):
            count1+=1


# In[ ]:


count1/len(train)


# In[ ]:


len(set(haha['LOSSSTATECD'])-set(haha['INSUREDSTATECD'])


# In[ ]:


for i in train.columns:
    print(i,train[i].nunique())


# In[ ]:


x=final
y=train['TARGET']


# In[ ]:


y.value_counts()


# In[ ]:


final.iloc[0]


# In[ ]:





# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.4, random_state=0)


# In[ ]:


sm = SMOTE(random_state=27, sampling_strategy=1.0)
X_train, y_train = sm.fit_sample(X_train, y_train)


# In[ ]:


get_ipython().run_line_magic('pinfo', 'SMOTE')


# In[ ]:


clf = RandomForestClassifier(random_state=100,class_weight='balanced') 
clf.fit(X_train, y_train) 
y_pred = clf.predict(X_test) 
    
clf.score(X_test, y_test)


# In[ ]:


# scores = cross_val_score(clf, x, y, cv=10,n_jobs=4)
# scores                      


# In[ ]:


recall_score(y_test,y_pred)


# In[ ]:


precision_score(y_test,y_pred)


# In[ ]:


f1_score(y_test,y_pred)


# In[ ]:


test=pd.read_csv('test.csv')


# In[ ]:


len(dummy.columns)


# In[ ]:


columns=dummy.columns


# In[ ]:


def add_missing_dummy_columns( d, columns ):
    missing_cols = set( columns ) - set( d.columns )
    for c in missing_cols:
        d[c] = 0


# In[ ]:


def fix_columns( d, columns ):  

    add_missing_dummy_columns( d, columns )

    # make sure we have all the columns we need
    assert( set( columns ) - set( d.columns ) == set())

    extra_cols = set( d.columns ) - set( columns )
    if extra_cols:
        print( "extra columns:", extra_cols)

    d = d[ columns ]
    return d


# In[ ]:


test['REPORTEDDT']=pd.to_datetime(test['REPORTEDDT'],format="%Y-%m-%d")
test['LOSSDT']=pd.to_datetime(test['LOSSDT'],format="%Y-%m-%d")

test['Days']=test['REPORTEDDT']-test['LOSSDT']

test['Days']=test['Days_to_Rep']/np.timedelta64(1, 'D')

test[test['Days']<0]

test.drop(columns=['REPORTEDDT','LOSSDT'],inplace=True)
test['Days']=test['Days'].abs()
from datetime import date, timedelta
test1=test[test['BIRTHDT'].notnull()]
test1['age'] = test1.apply(lambda x:(date.today() - parse(x['BIRTHDT']).date()) // timedelta(days=365.2425),axis=1)
test1.loc[(test1['age']>120) | (test1['age']<0),'age']=int(test1[(test1['age']<120) & (test1['age']>0)]['age'].mean())
test.loc[test['BIRTHDT'].notnull(),'age']=test1['age'].values
test['age'].fillna(int(test['age'].mean()),inplace=True)
test.drop('BIRTHDT',axis=1,inplace=True)
test.drop(columns=['CLAIMNO'],inplace=True)
test['distance']=np.sqrt( (test['INSUREDLAT']-test['CLMNT_LAT'])**2 + (test['INSUREDLON']-test['CLMNT_LON'])**2 )
print(test.columns)

test.drop(columns=['INSUREDNA_cleaned_root','CLMNT_LAT','CLMNT_LON','INSUREDLAT','INSUREDLON','CLMNT_NA_cleaned_root','Prov_Name_All_final_root','N_PAYTO_NAME_cleaned_root','N_REFRING_PHYS_final_root','N_PRVDR_NAME_NONPHYS_cleaned_root'],inplace=True)

for i in test.columns:
    if(test[i].dtypes =='object'):
        test[i].fillna(test[i].mode()[0],inplace=True)
    else:
        test[i].fillna(int(test[i].mean()),inplace=True)
        
test['SICCD']=test['SICCD'].replace('\t','nan')
rest=test[['Days','age','distance']]

# label=train[['SICCD','NATUREOFINJURYCD','LOSSSTATECD','WORK_LOC_STATECD','CLMNT_STATECD','INSUREDSTATECD']]

# label_encoder = preprocessing.LabelEncoder() 
# # Encode labels in column 'species'. 
# label=label.apply(label_encoder.fit_transform)

dummy=pd.get_dummies(test[['CAUSEOFINJURY_FLTRCD','BODYPART_FLTRCD','RTRN_TOWORKIND','SICCD','NATUREOFINJURYCD','LOSSSTATECD','WORK_LOC_STATECD','CLMNT_STATECD','INSUREDSTATECD']])
dummy=fix_columns(dummy,columns)
print(dummy.columns)
final=pd.concat([dummy,rest],axis=1)


# In[ ]:


# final['RTRN_TOWORKIND_nan']=0


# In[ ]:


t=(clf.predict_proba(final))


# In[ ]:


p=[i[1] for  i in t]


# In[ ]:


np.unique(p)


# In[ ]:


X_test.columns


# In[ ]:


final.columns


# In[ ]:


columns


# In[ ]:


t=pd.read_csv('test.csv')


# In[ ]:


sub=pd.DataFrame(columns=['CLAIMNO','TARGET'])


# In[ ]:


sub['CLAIMNO']=t['CLAIMNO']


# In[ ]:


sub['TARGET']=p


# In[ ]:


sub.to_csv('SUB_4.csv',index=False)

