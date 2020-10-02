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

a=[x for x in train.columns if "OTHERS"  not in x]

train=train[a]

for x in train , test:
    
    
    x["p_h"] = (x["DG3"]==2)*1+(x["MT1A"]==2)*1+(x["MT6"]==2)*1+ (x["MM18"]==2)*1+(x["MM19"]==2)*1+(x["FL4"]==2)*1+\
    (x["GN1"]==2)*1+(x["GN2"]==2)*1+(x["GN3"]==2)*1+(x["GN4"]==2)*1+(x["GN5"]==2)*1+(x["MT7A"]==1)*1
    

for x in train,test :   
    x["selected_nulls"]=x[["DG3","MT1A","MT6","MM18","MM19","FL4","GN1","GN2","GN3","GN4","GN5","MT7A"]].notnull().sum(axis=1)
    x["r1"]=x["p_h"]/x["selected_nulls"]
    x["r2"]=x["p_h"]/12
    
    
alt=[x for x in a if x not in ["train_id","is_female"]]
for x in alt:
    train.loc[train[x]==96,x]=np.nan
    test.loc[test[x]==96,x]=np.nan
    
categorical=["AA3","AA5","AA6","AA8","DG3","DG3A","DG6","DG14","DL1","DL2","DL5","DL27","DL28","MT1A","MT5","MT6","MT6A",
"MT6B","MT7A","MT9","MT11","FF13","MM10B","MM12","MM13","MM14","MM18","MM19","MM20","MM21","MM28","MM30",
"MM34","MM41","IFI5_1","IFI5_2","IFI5_3","IFI24","FL4","FL9A","FL9B","FL9C","FL10","FB2","FB19","FB20",
"FB21","FB24","FB25","GN1","GN2","GN3","GN4","GN5"]


a=[x for x in train.columns if "G2P2" in x]
b=[x for x in train.columns if "MT13" in x]
c=[x for x in train.columns if "MT14_" in x]
d=[x for x in train.columns if "MM11" in x]
e=[x for x in train.columns if "FB28_" in x]


s=a+b+c+d+e

categorical.extend(s)

el=[]
for x in train.columns:
    if len(train.loc[train[x].isnull()])/len(train)<0.95 :
        el.append(x)
        
train=train[el]
elt=[x for x in el if x in test]
test=test[elt]


categorical=[x for x in categorical if x in train.columns]

categorical_2=train.select_dtypes(include=[object]).columns.tolist()


categorical.extend(categorical_2)


no_usar=["train_id","is_female"]

features=[x for x in train.columns if x not in no_usar]

cat_ind=[features.index(x) for x in categorical]

for l in categorical:
    le = preprocessing.LabelEncoder()
    le.fit(list(train[l].dropna())+list(test[l].dropna()))

    train.loc[~train[l].isnull(),l]=le.transform(train.loc[~train[l].isnull(),l])
    test.loc[~test[l].isnull(),l]=le.transform(test.loc[~test[l].isnull(),l])
    
    
kf_previo=KFold(n_splits=5,random_state=256,shuffle=True)

i=1

r=[]

for train_index,test_index in kf_previo.split(train):

    lgb_train = lgb.Dataset(train.loc[train_index,features].values,train.loc[train_index,"is_female"].values.ravel())
    lgb_eval = lgb.Dataset(train.loc[test_index,features].values, train.loc[test_index,"is_female"].values.ravel(), reference=lgb_train)

    params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': { 'auc'},
            'learning_rate': 0.01,
            "num_leaves":100,
            # "max_depth" : 11,
            'feature_fraction': 0.5,
             "bagging_freq": 1,
            'bagging_fraction': 0.8,
          # "lambda_l1":1,
        # "lambda_l2":1,
         "is_unbalance" : False,
            'verbose': 1
    }




    lgbm3 = lgb.train(params,lgb_train,num_boost_round=13100,valid_sets=lgb_eval,early_stopping_rounds=50,verbose_eval=0,categorical_feature=cat_ind)
    test["IS_FEMALE_FOLD_"+str(i)]=lgbm3.predict(test[features].values, num_iteration=lgbm3.best_iteration)
    
    print ("Fold_"+str(i))
    a= roc_auc_score(train.loc[test_index,"is_female"],lgbm3.predict(train.loc[test_index,features].values, num_iteration=lgbm3.best_iteration))
    r.append(a)
    print (a)
    print ("")
    
    i=i+1

print ("mean: "+str(np.mean(np.array(r))))
print ("std: "+str(np.std(np.array(r))))      


a=[x for x in test.columns if "FOLD" in x]

test["is_female"]=test[a].mean(axis=1)


test["test_id"]=range(0,len(test))


test[["test_id","is_female"]].to_csv("submission_37_cleaned_data_2.csv",index=False)

