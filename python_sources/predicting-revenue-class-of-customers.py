#!/usr/bin/env python
# coding: utf-8

# A financial institution is planning to roll out a stock market trading facilitation service for their
# existing account holders. This service costs significant amount of money for the bank in terms of
# infra, licensing and people cost. To make the service offering profitable, they charge a percentage
# base commission on every trade transaction. However this is not a unique service offered by
# them, many of their other competitors are offering the same service and at lesser commission
# some times. To retain or attract people who trade heavily on stock market and in turn generate a
# good commission for institution, they are planning to offer discounts as they roll out the service
# to entire customer base.
# 
# Problem is , that this discount, hampers profits coming from the customers who do not trade in
# large quantities . To tackle this issue , company wants to offer discounts selectively. To be able to
# do so, they need to know which of their customers are going to be heavy traders or money
# makers for them.
# 
# To be able to do this, they decided to do a beta run of their service to a small chunk of their
# customer base [approx 10000 people]. For these customers they have manually divided them
# into two revenue categories 1 and 2. Revenue one category is the one which are money makers
# for the bank, revenue category 2 are the ones which need to be kept out of discount offers.
# We need to use this study's data to build a prediction model which should be able to identify if a
# customer is potentially eligible for discounts [falls In revenue grid category 1]. Lets get the data
# and begin.

# In[4]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')


# In[5]:


train_file = '../input/rg_train.csv'
test_file = '../input/rg_test.csv'


# In[6]:


bd_train = pd.read_csv(train_file)
bd_test = pd.read_csv(test_file)


# Lets add Revenue.Grid to test data and combine both test and train data for data preprocessing

# In[7]:


bd_test['Revenue.Grid'] = np.nan
bd_train['data'] = 'train'
bd_test['data'] = 'test'


# In[8]:


bd_test=bd_test[bd_train.columns] #to ensure same order of columns


# Concatenating train and test data

# In[9]:


bd_all = pd.concat([bd_train, bd_test], axis = 0)


# Lets take a look at the data

# In[10]:


bd_all.head()


# In[11]:


bd_all.dtypes


# In[12]:


bd_all.nunique()


# 1. REF_NO,post_code , post_area : drop (too many unique values)
# 2. children : Zero : 0 , 4+ : 4 and then convert to numeric
# 3. age_band, family income : string processing and then to numeric
# 4. status , occupation , occupation_partner , home_status,self_employed, 
#    TVArea , Region , gender : create dummies
# 5. Revenue Grid : 1,2 : 1,0 (some functions need the target to be 1/0 in 
#    binary classification).
# 

# In[13]:


bd_all.drop(['REF_NO','post_code','post_area'],axis=1,inplace=True)


# In[14]:


bd_all['children']=np.where(bd_all['children']=='Zero',0,bd_all['children'])
bd_all['children']=np.where(bd_all['children'].str[:1]=='4',4,bd_all['children'])
bd_all['children']=pd.to_numeric(bd_all['children'],errors='coerce')


# In[15]:


bd_all['Revenue.Grid']=(bd_all['Revenue.Grid']==1).astype(int)


# In[16]:


bd_all['family_income'].value_counts(dropna=False)


# In[17]:


bd_all['family_income']=bd_all['family_income'].str.replace(',',"")
bd_all['family_income']=bd_all['family_income'].str.replace('<',"")
k=bd_all['family_income'].str.split('>=',expand=True)


# In[18]:


for col in k.columns:
    k[col]=pd.to_numeric(k[col],errors='coerce')


# In[19]:


bd_all['fi']=np.where(bd_all['family_income']=='Unknown',np.nan,
    np.where(k[0].isnull(),k[1],
    np.where(k[1].isnull(),k[0],0.5*(k[0]+k[1]))))


# In[20]:


bd_all['age_band'].value_counts(dropna=False)


# In[21]:


k=bd_all['age_band'].str.split('-',expand=True)
for col in k.columns:
    k[col]=pd.to_numeric(k[col],errors='coerce')


# In[22]:


bd_all['ab']=np.where(bd_all['age_band'].str[:2]=='71',71,
             np.where(bd_all['age_band']=='Unknow',np.nan,0.5*(k[0]+k[1])))


# In[23]:


del bd_all['age_band']
del bd_all['family_income']


# In[24]:


cat_vars=bd_all.select_dtypes(['object']).columns
cat_vars=list(cat_vars)
cat_vars.remove('data')


# In[25]:


for col in cat_vars:
    dummy=pd.get_dummies(bd_all[col],drop_first=True,prefix=col)
    bd_all=pd.concat([bd_all,dummy],axis=1)
    del bd_all[col]
    print(col)
del dummy


# imputing missing values

# In[26]:


for col in bd_all.columns:
    if col=='data' or bd_all[col].isnull().sum()==0:
        continue
    bd_all.loc[bd_all[col].isnull(),col]=bd_all.loc[bd_all['data']=='train',col].mean()


# In[27]:


bd_all.loc[bd_all[col].isnull(),col]=bd_all.loc[bd_all['data']=='train',col].mean()


# Separating data

# In[28]:


train1=bd_all[bd_all['data']=='train']
del train1['data']
test1=bd_all[bd_all['data']=='test']
test1.drop(['Revenue.Grid','data'],axis=1,inplace=True)


# In[29]:


from sklearn.linear_model import LogisticRegression


# In[30]:


params={'class_weight':['balanced',None],
        'penalty':['l1','l2'],
# these are L1 and L2 written in lower case
# dont confuse them with numeric eleven and tweleve
        'C':np.linspace(0.0001,1000,10)}

# we can certainly try much higher ranges and number of values for theparameter 'C'
# grid search in this case , will be trying out 2*2*10=40 possiblecombination
# and will give us cross validated performance for all


# In[31]:


model=LogisticRegression(fit_intercept=True)


# In[32]:


from sklearn.model_selection import GridSearchCV


# In[33]:


grid_search=GridSearchCV(model,param_grid=params,cv=10,scoring="roc_auc")
# note that scoring is now roc_auc as we are solving a classification problem


# In[34]:


x_train=train1.drop('Revenue.Grid',axis=1)
y_train=train1['Revenue.Grid']


# In[35]:


grid_search.fit(x_train,y_train)


# In[36]:


# predict_proba for predciting probabilities
# just predict, predicts hard classes considering 0.5 as score cutoff
# which is not always a great idea, we'll see in a moment
test_prediction = grid_search.predict_proba(test1)


# In[37]:


test_prediction


# In[38]:


# this will tell you which probability belongs to which class
grid_search.classes_


# Finding cutoff on the basis of max KS

# In[39]:


train_score=grid_search.predict_proba(x_train)[:,1]
real = y_train


# In[40]:


cutoffs = np.linspace(.001,0.999, 999)


# In[41]:


KS=[]


# In[42]:


for cutoff in cutoffs:
    predicted=(train_score>cutoff).astype(int)
    TP=((real==1)&(predicted==1)).sum()
    FP=((real==0)&(predicted==1)).sum()
    TN=((real==0)&(predicted==0)).sum()
    FN=((real==1)&(predicted==0)).sum()
    ks=(TP/(TP+FN))-(FP/(TN+FP))
    KS.append(ks)


# In[43]:


temp=pd.DataFrame({'cutoffs':cutoffs,'KS':KS})


# In[44]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[45]:


sns.lmplot(x='cutoffs',y='KS',data=temp,fit_reg=False)


# In[46]:


cutoffs[KS==max(KS)][0]


# In[47]:


test_hard_classes=(test_prediction>cutoffs[KS==max(KS)][0]).astype(int)


# In[48]:


test_hard_classes[:,0]


# In[49]:


output = pd.DataFrame({'REF_NO' : bd_test.REF_NO, 'Revenue.Grid':test_hard_classes[:,0]})
output.to_csv('submission.csv', index=False)


# In[ ]:




