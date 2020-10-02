#!/usr/bin/env python
# coding: utf-8

# ## IEEE Fraud: LightGBM GPU+Augmentation
# ### Hi welcome to this kernel in this kernel i am using lightgbm and augmentation technique. <br>
# ### Reference and credits: <br>
# 1. https://www.kaggle.com/kirankunapuli/ieee-fraud-lightgbm-with-gpu <br>
# 2. https://www.kaggle.com/jiweiliu/lgb-2-leaves-augment <br>
# 3. https://www.kaggle.com/niteshx2/beginner-explained-lgb-2-leaves-augment <br>
# Thanks to authors of the above kernels!
# 

# ## LightGBM GPU Installation

# In[ ]:


get_ipython().system('rm -r /opt/conda/lib/python3.6/site-packages/lightgbm')


# In[ ]:


get_ipython().system('git clone --recursive https://github.com/Microsoft/LightGBM')


# In[ ]:


get_ipython().system('apt-get install -y -qq libboost-all-dev')


# ### Build and re-install LightGBM with GPU support

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cd LightGBM\nrm -r build\nmkdir build\ncd build\ncmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..\nmake -j$(nproc)')


# In[ ]:


get_ipython().system('cd LightGBM/python-package/;python3 setup.py install --precompile')


# In[ ]:


get_ipython().system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
get_ipython().system('rm -r LightGBM')


# In[ ]:


# Latest Pandas version
get_ipython().system("pip install -q 'pandas==0.25' --force-reinstall")


# ## Imports

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold, StratifiedKFold,TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import os
print(os.listdir("../input"))
import warnings
import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


# In[ ]:


print("Pandas version:", pd.__version__)


# In[ ]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:


import gc
gc.enable()


# In[ ]:


import lightgbm as lgb
print("LightGBM version:", lgb.__version__)


# ## Preprocessing

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


train_transaction = pd.read_csv('../input/train_transaction.csv')
test_transaction = pd.read_csv('../input/test_transaction.csv')

train_identity = pd.read_csv('../input/train_identity.csv')
test_identity = pd.read_csv('../input/test_identity.csv')

sample_submission = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape)
print(test.shape)


# In[ ]:


train.head()


# In[ ]:


predictions = test[['TransactionID_x']]


# In[ ]:


#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
#credits : https://www.kaggle.com/mjbahmani/reducing-memory-size-for-ieee
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist


# In[ ]:


train, NAlist = reduce_mem_usage(train)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


# In[ ]:


test, NAlist = reduce_mem_usage(test)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


# In[ ]:


train = train.sort_values('TransactionID_x')


# In[ ]:


train = train.set_index('TransactionID_x')
test = test.set_index('TransactionID_x')


# In[ ]:


train.head()


# In[ ]:


y_train = train['isFraud'].copy()
del train_transaction, train_identity, test_transaction, test_identity
gc.collect()


# In[ ]:


test.head()


# In[ ]:


# Drop target, fill in NaNs
X_train = train.drop('isFraud', axis=1)
X_test = test.copy()
del train, test
gc.collect()


# In[ ]:


X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)


# In[ ]:


# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))


# Note : Here i am taking only 475000 rows in the data because of memory issues in kernel.If you have good hardware you can take full data frame for training.

# In[ ]:


X_tr = X_train[:475000]
y_tr = y_train[:475000]


# In[ ]:


def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


# ## Modeling

# In[ ]:


params = {'max_bin' :63,
    'num_leaves' : 255,
    'num_iterations' : 500,
    'learning_rate' : 0.05,
    'tree_learner' :'serial',
    'is_training_metric' : False,
    'min_data_in_leaf' : 1,
    'min_sum_hessian_in_leaf' : 100,
    'sparse_threshold':1.0,
    'device' : 'gpu',
    'num_thread' : -1,
    'save_binary': True,
    'seed': 42,
    'feature_fraction_seed' : 42,
    'bagging_seed' : 42,
    'drop_seed' : 42,
    'data_random_seed' : 42,
    'objective' : 'binary',
    'boosting_type' : 'gbdt',
    'verbose' : 1,
    'metric' : 'auc',
    'is_unbalance' : True,
    'boost_from_average' : False}


# In[ ]:


feature_importance_df = pd.DataFrame()
val_aucs = []
skf = KFold(n_splits=5)
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_tr, y_tr)):
    X_tra, y_tra = X_tr.iloc[trn_idx], y_tr.iloc[trn_idx]
    X_valid, y_valid = X_tr.iloc[val_idx], y_tr.iloc[val_idx]
    print('Fold:',fold+1)
    
    N = 3
    p_valid,yp = 0,0
    for i in range(N):
        X_t, y_t = augment(X_tra.values, y_tra.values)
        X_t = pd.DataFrame(X_t)
        X_t = X_t.add_prefix('var_')
    
        trn_data = lgb.Dataset(X_t, label=y_t)
        val_data = lgb.Dataset(X_valid, label=y_valid)
        evals_result = {}
        lgb_clf = lgb.train(params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000,
                        evals_result=evals_result
                       )
        p_valid += lgb_clf.predict(X_valid)
        yp += lgb_clf.predict(X_test)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = X_train.columns
    fold_importance_df["importance"] = lgb_clf.feature_importance()
    fold_importance_df["fold"] = fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    val_score = roc_auc_score(y_valid, p_valid)
    val_aucs.append(val_score)
    predictions['fold{}'.format(fold+1)] = yp/N


# ## Let's see what this augmentation does to our dataframe

# In[ ]:


print("Distribution of 1s in original data : {} / {} ".format(np.sum(y_train) , len(y_train)))
print("Percentage of 1s in original data  : {}".format(np.sum(y_train)*100.0/len(y_train)))
print('--'*30)
print("Distribution of 1s in subset of data(450000 rows) : {} / {} ".format(np.sum(y_tr) , len(y_tr)))
print("Percentage of 1s in subset of data(450000 rows)  : {}".format(np.sum(y_tr)*100.0/len(y_tr)))
print('--'*30)
print("Percentage of 1s in augmented data : {}".format(np.sum(y_t)*100.0/len(y_t)))
print("Distribution of 1s in augmented data : {} / {} ".format(np.sum(y_t) , len(y_t)))


# * We can see that after augmentation the percentage of 1s was increased by approximatly 1.5%.This technique is more like oversampling but,here we oversample both classes,rather than just one. 

# In[ ]:


mean_auc = np.mean(val_aucs)
std_auc = np.std(val_aucs)
print('Mean auc is {:2f} and standard deviation of auc is {:2f}'.format(mean_auc,std_auc))


# ## Feature Importances

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:30].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]
plt.figure(figsize=(14,10))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Feature importance top 50(averaged over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


# In[ ]:


predictions['isFraud'] = np.mean(predictions[[col for col in predictions.columns if col not in ['TransactionID_x', 'isFraud']]].values, axis=1)
predictions.to_csv('lgb_all_predictions.csv', index=None)


# ## Submission

# In[ ]:


sub_df = pd.DataFrame({"TransactionID":predictions["TransactionID_x"].values})
sub_df["isFraud"] = predictions['isFraud']
sub_df.to_csv("lgb_submission.csv", index=False)
sub_df.head()


# ### That's all for now.Suggestions are welcome for further improvement of kernel,Upvote if you like this kernel.You can achieve better score with using full dataset for training, hyperparameter tuning,feature engineering,building different models and stacking the predictions.Thankyou! <br>
# ### Happy learning
# 
