#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# ### Importing the train and test data

# In[ ]:


train_df=pd.read_csv('/kaggle/input/fraud-prediction/train.csv')
test_df=pd.read_csv('/kaggle/input/fraud-prediction/test.csv')
df_desc=pd.read_csv('/kaggle/input/fraud-prediction/column_Desc.csv')


# In[ ]:


from xgboost import XGBClassifier
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set()

import datetime


# ### Let's see the column info and it's information about missing value and unique values:

# In[ ]:


def uniquelo(df):
    col=[]
    vals=[]
    for a in df.columns:
        col.append(a)
        vals.append(len(df[a].unique()))
    percent_missing = df.isnull().sum() * 100 / len(df)
    d = {'Unique_val': vals,'percent_missing': percent_missing, 'Data_Types' :df.dtypes.values}
    return pd.DataFrame(data=d)


# In[ ]:


out=uniquelo(train_df)
# out.sort_values('percent_missing', inplace=True,ascending=False)
out['Column Info']=df_desc.Description.values
out


# ### There are lot of variables which aren't much useful, with too many unique values or too many missing values. So they are not useful for our classification problem. We will also create some derived values using information already present in our other columns.
# 
# ### Let's start

# #### First we create Age Column:

# In[ ]:


train_df['BIRTHDT'] = pd.to_datetime(train_df['BIRTHDT'], format='%Y-%m-%d')
l = [(datetime.datetime.now()-i).days/365 for i in train_df['BIRTHDT']]
l = [int(i) if np.isnan(i)==False else np.nan for i in l]
train_df['AGE'] = l


# In[ ]:


test_df['BIRTHDT'] = pd.to_datetime(test_df['BIRTHDT'], format='%Y-%m-%d')
l = [(datetime.datetime.now()-i).days/365 for i in test_df['BIRTHDT']]
l = [int(i) if np.isnan(i)==False else np.nan for i in l]
test_df['AGE'] = l


# ### Now we shall drop columns which either have too many unique values or too many missing values or might not be useful for our classification

# In[ ]:


train_df.drop(['BIRTHDT','REPORTEDDT','LOSSDT','N_PAYTO_NAME_cleaned_root','Prov_Name_All_final_root','INSUREDNA_cleaned_root',
                   'N_REFRING_PHYS_final_root','CLMNT_NA_cleaned_root','N_PRVDR_NAME_NONPHYS_cleaned_root'],axis=1,inplace=True)

test_df.drop(['BIRTHDT','REPORTEDDT','LOSSDT','N_PAYTO_NAME_cleaned_root','Prov_Name_All_final_root','INSUREDNA_cleaned_root',
                   'N_REFRING_PHYS_final_root','CLMNT_NA_cleaned_root','N_PRVDR_NAME_NONPHYS_cleaned_root'],axis=1,inplace=True)


# ### Let's check our dataset now:

# In[ ]:


uniquelo(train_df)


# ### Since there was no reference of using with to find distance, I first removed the variables and then fit the model, but using them in the model somehow gave better results. So that's why keeping them.

# ### Now filling null values:

# In[ ]:


for a in train_df.columns:
    if (train_df[a].dtype!='object') and (a!='TARGET'):
        train_df[a].fillna(train_df[a].mean(),inplace=True)
        test_df[a].fillna(train_df[a].mean(),inplace=True)
        
    elif (a=='TARGET') or (a=='CLAIMNO'):
        continue
    
    else:
        train_df[a].fillna(train_df[a].mode()[0],inplace=True)
        test_df[a].fillna(train_df[a].mode()[0],inplace=True)


# ### Now making our Target as Response Variable:

# In[ ]:


y = train_df['TARGET']
train_df.drop(['TARGET'],axis=1,inplace=True)


# ### Making dataset of explanatory variables, and dummyfying categorical ones:

# ### We see below that there are some labels in some columns that are present in train dataset but not in test, and also vice-versa. So we will after dummyfying, create columns of these values and assign it 0, so that we get fixed set of columns in our both train and test data:

# In[ ]:


x = pd.DataFrame()
test = pd.DataFrame()

for i in train_df.columns:
    if (train_df[i].dtype!='object') or (i=='CLAIMNO'):
        continue
    
    else:
        train_dummy = pd.get_dummies(train_df[i])
        test_dummy = pd.get_dummies(test_df[i])
#         if(a.columns[0] in (b.columns)):
#             b.drop(a.columns[0],axis=1,inplace=True)
#         a.drop(a.columns[0],axis=1,inplace=True)
        missing_test = set(train_dummy.columns)-set(test_dummy.columns)
#         train_miss = set(b.columns)-set(a.columns)

        for k in missing_test:
            test_dummy[k] = 0
            
        for p in train_dummy.columns:
            #print(j)
            x[i+'_'+str(p)] = train_dummy[p]
            test[i+'_'+str(p)] = test_dummy[p]


# ### I've already performed Cross-Validation on Colab and since it was k-fold(5), it consumed whole RAM, so just mentioning the k-fold cv without using it and also reducing the number of estimators while performing the rest of the modelling part and giving the output below:
# 
# ### I used RandomForest with Bagging

# In[ ]:


from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import model_selection


# In[ ]:


clf = RandomForestClassifier(random_state=42,verbose=3, n_estimators=50)


# In[ ]:


model = BaggingClassifier(base_estimator = clf, n_estimators = 10, n_jobs=-1, verbose=3, random_state = 240)


# #### The k-fold I didn't perform here:
# kfold = model_selection.KFold(n_splits = 3, random_state = 240)
# 
# results = model_selection.cross_val_score(model, x, y, cv = kfold, verbose=3, n_jobs=-1)

# In[ ]:


model.fit(x,y)


# ### Here's the final output:

# In[ ]:


l = model.predict_proba(test)[:,1]

claim_no = test_df['CLAIMNO']

output = pd.DataFrame(columns=['CLAIMNO','TARGET'])

output['CLAIMNO'] = claim_no

output['TARGET'] = l

output.to_csv('For_Kernel3.csv',index=False)

output


# ### My eventual output was a weighted ensemble of other techniques too, but here just giving some base model as output.
