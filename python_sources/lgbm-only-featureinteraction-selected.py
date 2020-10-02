#!/usr/bin/env python
# coding: utf-8

# In[16]:


#Importing Libraries
import warnings
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
warnings.simplefilter("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook
from sklearn.metrics import f1_score
import xgboost
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import os
from bayes_opt import BayesianOptimization
print(os.listdir("../input"))


# In[17]:


#Reading Data
df=pd.read_csv("../input/train.csv")
del df['id']
print(df.shape)
display(df.head())


# In[18]:


#Reading Test data
test=pd.read_csv("../input/test.csv")
idx=test['id']
del test['id']
print(test.shape)
display(test.head())


# In[19]:


#Plot target
sns.distplot(df.loss)


# In[20]:


#Categorical Interactions Top50
cat_int=['cat12|cat80', 'cat79|cat80', 'cat1|cat12|cat80',
       'cat12|cat80|cat81', 'cat12|cat79|cat80', 'cat1|cat80',
       'cat103|cat12|cat80', 'cat57|cat80', 'cat12|cat72|cat80',
       'cat1|cat103|cat80', 'cat57|cat79|cat80', 'cat1|cat80|cat81',
       'cat12|cat57|cat80', 'cat2|cat57|cat80', 'cat10|cat80',
       'cat12|cat81', 'cat103|cat12', 'cat1|cat12', 'cat1|cat114|cat80',
       'cat10|cat103|cat80', 'cat1|cat111|cat80', 'cat10|cat111|cat80',
       'cat12|cat53', 'cat1|cat103', 'cat12|cat53|cat80',
       'cat10|cat101|cat80', 'cat12|cat53|cat81', 'cat12|cat79',
       'cat1|cat81', 'cat2|cat80|cat81', 'cat81|cat87', 'cat12|cat72',
       'cat103|cat111', 'cat1|cat72', 'cat10|cat114|cat81', 'cat4|cat5',
       'cat103|cat111|cat80', 'cat10|cat72|cat80', 'cat80|cat81',
       'cat103|cat111|cat12', 'cat1|cat2|cat72', 'cat10|cat81|cat87',
       'cat2|cat57', 'cat1|cat72|cat79', 'cat79|cat81|cat87',
       'cat2|cat44', 'cat114|cat2', 'cat13|cat53', 'cat111|cat13|cat53',
       'cat10|cat81']

for each in cat_int:
    l=each.split("|")
    if(len(l)==3):
        a=l[0]
        b=l[1]
        c=l[2]
        df[a+"_"+b+"_"+c]=df[a]+df[b]+df[c]
        test[a+"_"+b+"_"+c]=test[a]+test[b]+test[c]
    else:
        a=l[0]
        b=l[1]
        df[a+"_"+b]=df[a]+df[b]
        test[a+"_"+b]=test[a]+test[b]


# In[ ]:


#Label Encoding All Categorical Columns
c=list(df.columns)
for each in c:
    if each[0:3]=='cat':
        res=pd.factorize(pd.concat((df[each],test[each])).values,sort=True)[0]
        df[each]=res[0:188318]
        test[each]=res[188318:]


# In[ ]:


#X and Y
c=list(df.columns)
c.remove('loss')
X=df.loc[:,c]
Y=df.loss
X.head()


# In[ ]:


#Transformation on Y
Y=np.log1p(Y)
sns.distplot(Y)


# In[ ]:


#Numerical Feature interactions selected
X['mul_cont2_cont7']=X['cont2']*X['cont7']
X['add_cont2_cont7']=X['cont2']+X['cont7']
X['add_cont2_cont12']=X['cont2']+X['cont12']
X['add_cont2_cont11']=X['cont2']+X['cont11']
X['sub_cont1_cont7']=X['cont1']-X['cont7']
X['add_cont7_cont14']=X['cont7']+X['cont14']
X['div_cont2_cont4']=X['cont2']/X['cont4']
X['add_cont3_cont7']=X['cont3']+X['cont7']

#Numerical Feature interactions selected
test['mul_cont2_cont7']=test['cont2']*test['cont7']
test['add_cont2_cont7']=test['cont2']+test['cont7']
test['add_cont2_cont12']=test['cont2']+test['cont12']
test['add_cont2_cont11']=test['cont2']+test['cont11']
test['sub_cont1_cont7']=test['cont1']-test['cont7']
test['add_cont7_cont14']=test['cont7']+test['cont14']
test['div_cont2_cont4']=test['cont2']/test['cont4']
test['add_cont3_cont7']=test['cont3']+test['cont7']


# In[ ]:


# #Numerical tranformations 
# col_num=['cont1','cont2','cont3','cont4','cont5','cont6','cont7','cont8','cont9'
#         ,'cont10','cont12','cont12','cont13','cont14']
# for each in col_num:
#     X[each]=np.log1p(X[each])[0]
#     test[each]=np.log1p(test[each])[0]


# In[ ]:


X.shape


# In[ ]:


params={'bagging_fraction': 0.5021393514653055,
'feature_fraction': 0.6190345305094423,
'lambda_l1': 0.4932225704923126,
'max_depth': 39,
'min_data_in_leaf': 22,
'min_gain_to_split': 0.2,
'num_leaves': 45,
'silent':True,
'objective':'fair',
'random_state':60,
'n_estimators':10000,
'learning_rate':0.03}

# params={'silent':True,
# 'objective':'fair',
# 'random_state':60,
# 'n_estimators':10000,
# 'learning_rate':0.03}


# In[ ]:


r=X.nunique()
r.sort_values(ascending=False)


# In[14]:


#5 Fold Cross Validation
P=np.zeros((test.shape[0]))
trainALL=[]
valALL=[]
kf = KFold(n_splits=5,random_state=60)
k=1
for train_index, test_index in kf.split(X):
    print("FOLD :",k)
    X_train, X_val = X.loc[train_index,:], X.loc[test_index,:]
    Y_train, Y_val = Y[train_index], Y[test_index]
    
#     #Mean Encoding
#     temp=X_train.copy()
#     temp['target']=Y_train
#     dic=temp.groupby('cat116')['target'].mean()
#     X_train['mean_f1']=0
#     mask=np.array(X_train['cat116'].apply(lambda x: x in list(dic.index) ).values)
#     X_train.loc[mask,'mean_f1']=X_train.loc[mask,'cat116'].map(lambda x:dic[x])
#     X_train.loc[~mask,'mean_f1']=X_train['mean_f1'].mean()
    
#     X_val['mean_f1']=0
#     mask=np.array(X_val['cat116'].apply(lambda x: x in list(dic.index) ).values)
#     X_val.loc[mask,'mean_f1']=X_val.loc[mask,'cat116'].map(lambda x:dic[x])
#     X_val.loc[~mask,'mean_f1']=X_val['mean_f1'].mean()
#     #END
    
    reg=lgb.LGBMRegressor(**params)
    reg.fit(X_train, Y_train,eval_metric='mae',eval_set=[(X_val,Y_val)],verbose=False,early_stopping_rounds=50)
    
    trainR=mean_absolute_error(np.expm1(Y_train),np.expm1(reg.predict(X_train)))
    valR=mean_absolute_error( np.expm1(Y_val),np.expm1(reg.predict(X_val)))
    print("MAE Train : ",trainR)
    print("MAE Test : ",valR)
    print("****************")
    trainALL.append(trainR)
    valALL.append(valR)
    
    #Feature Importance
#     if(k==1):
#         fI=pd.DataFrame(index=reg.booster_.feature_name())
#         fI['imp']=reg.booster_.feature_importance(importance_type='gain')
#     else:
#         fI.loc[reg.booster_.feature_name(),'imp']+=reg.booster_.feature_importance(importance_type='gain')
    
    k+=1
    P+=np.expm1(reg.predict(test))


# In[15]:


print("Train Score: ",np.mean(trainALL),"   Std dev:",np.std(trainALL))
print("Val Score:  ",np.mean(valALL),"   Std dev:",np.std(valALL))


# In[ ]:


dfx=pd.DataFrame()
dfx['id']=idx
dfx['loss']=P/5
dfx.to_csv("lgb_avgfolds.csv",index=False)


# In[ ]:




