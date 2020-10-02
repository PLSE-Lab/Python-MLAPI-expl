# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import gc
import lightgbm as lgb
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import xgboost as xg




# Any results you write to the current directory are saved as output.

train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")

def reduce_cat(data,data_2, name_variable,max_cat):
    
    if len(data[name_variable].dropna().unique()) > max_cat :
    
        rcat=list(data[name_variable].value_counts().iloc[:max_cat].index)
    
        data.loc[(~data[name_variable].isin(rcat))&(data[name_variable].notnull()),name_variable]="other_"+name_variable
        
        data_2.loc[(~data_2[name_variable].isin(rcat))&(data_2[name_variable].notnull()),name_variable]="other_"+name_variable

    
    return data,data_2
    
for x in train,test:
    train.loc[train["DG6"]!=2,"DG6"]==0
    test.loc[test["DG6"]!=2,"DG6"]==0
    
for x in train,test:
    train.loc[train["DL1"]!=7,"DL1"]==0
    test.loc[test["DL1"]!=7,"DL1"]==0
    
a=[x for x in train.columns if "OTHERS"  not in x]

train=train[a]

categorical=["AA3","AA5","AA6","AA8","DG3","DG3A","DG14","DL2","DL5","DL27","DL28","MT1A","MT5","MT6","MT6A",
"MT6B","MT7A","MT9","MT11","FF13","MM10B","MM12","MM13","MM14","MM18","MM19","MM20","MM21","MM28","MM30",
"MM34","MM41","IFI5_1","IFI5_2","IFI5_3","IFI24","FL4","FL9A","FL9B","FL9C","FL10","FB2","FB19","FB20",
"FB21","FB24","FB25","GN1","GN2","GN3","GN4","GN5"]

a=[x for x in train.columns if "G2P2" in x]
b=[x for x in train.columns if "MT13" in x]
c=[x for x in train.columns if "MT14_" in x]
d=[x for x in train.columns if "MM11" in x]
e=[x for x in train.columns if "FB28_" in x]

s=a+b+c+d

categorical.extend(s)

alt=[x for x in a if x not in ["train_id","is_female"] and x not in categorical]
for x in alt:
    train.loc[train[x]==96,x]=np.nan
    test.loc[test[x]==96,x]=np.nan
    train.loc[train[x]==99,x]=np.nan
    test.loc[test[x]==99,x]=np.nan
    
el=[]
for x in train.columns:
    if len(train.loc[train[x].isnull()])/len(train)<0.99 :
        el.append(x)
        
        
train=train[el]
elt=[x for x in el if x in test]
test=test[elt]


categorical=[x for x in categorical if x in train.columns]


categorical_2=train.select_dtypes(include=[object]).columns.tolist()


categorical.extend(categorical_2)

for x in categorical:
    train,test=reduce_cat(train,test,x,max_cat=4)
    
train=pd.get_dummies(train,columns=categorical)
test=pd.get_dummies(test,columns=categorical)

no_usar=["train_id","is_female"]

features=[x for x in train.columns if x not in no_usar ]

kf_previo=KFold(n_splits=5,random_state=256,shuffle=True)

i=1

r=[]


for train_index,test_index in kf_previo.split(train):

    
    params = {
        "objective":"binary:logistic",
        "tree_method":"hist", 
        "grow_policy":"depthwise",

        'eta': 0.01,
        'colsample_bytree': 0.5,
        'max_depth': 9,
        'subsample': 0.85,
        'silent': 1,
        'verbose_eval': True,
        "eval_metric":"auc"
    }


    xgtrain = xg.DMatrix(train.loc[train_index,features], label=train.loc[train_index,"is_female"])
    xgtest = xg.DMatrix(train.loc[test_index,features],label=train.loc[test_index,"is_female"])

    model = xg.train(params=params, dtrain=xgtrain,evals=[(xgtest,"test")], num_boost_round=10000,early_stopping_rounds=50,verbose_eval=False)

    test["IS_FEMALE_FOLD_"+str(i)]=model.predict(xg.DMatrix(test[features]), ntree_limit=model.best_ntree_limit)
   
    print ("Fold_"+str(i))
    
    a=roc_auc_score(train.loc[test_index,"is_female"],model.predict(xg.DMatrix(train.loc[test_index,features]), ntree_limit=model.best_iteration))
    
    r.append(a)
    
    print (roc_auc_score(train.loc[test_index,"is_female"],model.predict(xg.DMatrix(train.loc[test_index,features]), ntree_limit=model.best_iteration)))
    print ("")
    
    i=i+1
    
print ("mean: "+str(np.mean(np.array(r))))
print ("std: "+str(np.std(np.array(r))))  



a=[x for x in test.columns if "FOLD" in x]

test["is_female"]=test[a].mean(axis=1)

test["test_id"]=range(0,len(test))

test[["test_id","is_female"]].to_csv("submission_32_xgboost_cleaned_data.csv",index=False)




