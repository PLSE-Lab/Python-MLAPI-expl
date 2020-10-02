#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# data analysis libraries:
import numpy as np
import pandas as pd

# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns

# to ignore warnings:
import warnings
warnings.filterwarnings('ignore')

# to display all columns:
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_auc_score,roc_curve
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# In[ ]:


# Read train and test data with pd.read_csv():
train_id= pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
test_id = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")
train_tr = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
test_tr = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")


# In[ ]:


train_id.head()


# In[ ]:


test_id.head()


# In[ ]:


train_tr.head()


# In[ ]:


test_tr.head()


# In[ ]:


train_id.info()


# In[ ]:


test_id.info()


# In[ ]:


train_tr.info()


# In[ ]:


train_tr.info()


# In[ ]:


test_tr.info()


# In[ ]:


train=pd.merge(train_tr, train_id, on = "TransactionID",how='left')
train.head()


# In[ ]:


test=pd.merge(test_tr, test_id, on = "TransactionID",how="left")
test.head()


# In[ ]:


del train_id, train_tr, test_id, test_tr


# In[ ]:


train.head()


# In[ ]:


test.columns=train.columns.drop("isFraud")


# In[ ]:


train['Trans_min_mean'] = train['TransactionAmt'] - train['TransactionAmt'].mean()
train['Trans_min_std'] = train['Trans_min_mean'] / train['TransactionAmt'].std()
test['Trans_min_mean'] = test['TransactionAmt'] - test['TransactionAmt'].mean()
test['Trans_min_std'] = test['Trans_min_mean'] / test['TransactionAmt'].std()

train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')

train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')
train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')
train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')
train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')
test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')
test['id_02_to_mean_card4'] =test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')
test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')
test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')

train['D15_to_mean_card1'] =train['D15'] / train.groupby(['card1'])['D15'].transform('mean')
train['D15_to_mean_card4'] =train['D15'] / train.groupby(['card4'])['D15'].transform('mean')
train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')
train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')
test['D15_to_mean_card1'] =test['D15'] / test.groupby(['card1'])['D15'].transform('mean')
test['D15_to_mean_card4'] =test['D15'] / test.groupby(['card4'])['D15'].transform('mean')
test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')
test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')



train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')
train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')
train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')
test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')
test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')

train['TransactionAmt'] = np.log(train['TransactionAmt'])
test['TransactionAmt'] = np.log(test['TransactionAmt'])


# In[ ]:


new_f=train.columns[-10:]
new_f


# In[ ]:


get_ipython().run_cell_magic('time', '', '# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n# WARNING! THIS CAN DAMAGE THE DATA \ndef reduce_mem_usage2(df):\n    """ iterate through all the columns of a dataframe and modify the data type\n        to reduce memory usage.        \n    """\n    start_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage of dataframe is {:.2f} MB\'.format(start_mem))\n    \n    for col in df.columns:\n        col_type = df[col].dtype\n        \n        if col_type != object:\n            c_min = df[col].min()\n            c_max = df[col].max()\n            if str(col_type)[:3] == \'int\':\n                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                    df[col] = df[col].astype(np.int8)\n                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                    df[col] = df[col].astype(np.int16)\n                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                    df[col] = df[col].astype(np.int32)\n                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                    df[col] = df[col].astype(np.int64)  \n            else:\n                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                    df[col] = df[col].astype(np.float16)\n                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                    df[col] = df[col].astype(np.float32)\n                else:\n                    df[col] = df[col].astype(np.float64)\n        else:\n            df[col] = df[col].astype(\'category\')\n\n    end_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage after optimization is: {:.2f} MB\'.format(end_mem))\n    print(\'Decreased by {:.1f}%\'.format(100 * (start_mem - end_mem) / start_mem))\n    \n    return df')


# In[ ]:


train=reduce_mem_usage2(train)
test=reduce_mem_usage2(test)


# In[ ]:


train.columns


# In[ ]:


train.shape, test.shape


# In[ ]:


train.head()


# In[ ]:


emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 
      'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft',
      'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo',     
      'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 
      'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink',
      'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other',
      'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 
      'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 
      'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo',
      'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other',
      'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft',
      'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 
      'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 
      'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 
      'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 
      'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 
      'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 
      'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other',
      'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']

# https://www.kaggle.com/c/ieee-fraud-detection/discussion/100499#latest-579654
for c in ['P_emaildomain', 'R_emaildomain']:
     train[c + '_bin'] = train[c].map(emails)
     test[c + '_bin'] = test[c].map(emails)

     train[c + '_suffix'] = train[c].map(lambda x: str(x).split('.')[-1])
     test[c + '_suffix'] = test[c].map(lambda x: str(x).split('.')[-1])

     train[c + '_suffix'] = train[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')
     test[c + '_suffix'] = test[c + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')


# In[ ]:


main_columns=[  'TransactionAmt_to_std_card1', 'TransactionAmt_to_std_card4',
       'id_02_to_mean_card1', 'id_02_to_mean_card4', 'id_02_to_std_card1',
       'id_02_to_std_card4', 'D15_to_mean_card1', 'D15_to_mean_card4',
       'D15_to_std_card1', 'D15_to_std_card4']
cat_columns=['ProductCD','M5','id_38','P_emaildomain', 'R_emaildomain']


# In[ ]:


def agg_func(main_columns, cat_columns, aggregations=["mean"], train_df=train, test_df=test, 
              fillna=True, usena=False):
   
    for main_column in main_columns:  
        for col in cat_columns:
            for agg_type in aggregations:
                new_col_name = main_column+'_'+col+'_'+agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])
                if usena: temp_df.loc[temp_df[main_column]==-1,main_column] = np.nan
                temp_df = (temp_df.groupby([col])[main_column].agg([agg_type])).rename(columns={agg_type: new_col_name}).reset_index()
                
                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   

                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                test_df[new_col_name]  = test_df[col].map(temp_df).astype('float32')


# In[ ]:


agg_func(main_columns,cat_columns,aggregations=['mean',"std"])


# In[ ]:


print(train.shape)
test.shape


# In[ ]:


train_cat = train.select_dtypes(include=['object'])
train_cat_columns=train_cat.columns
train_cat_columns
train_columns=train.columns


# In[ ]:


test_cat = test.select_dtypes(include=['object'])
test_cat_columns=test_cat.columns
del test_cat, train_cat


# In[ ]:


from sklearn import preprocessing
for i in train_cat_columns: 
    lbe=preprocessing.LabelEncoder()
    train[i]=lbe.fit_transform(train[i].astype(str))


# In[ ]:


for i in test_cat_columns:    
    test[i]=lbe.fit_transform(test[i].astype(str))


# In[ ]:


train_columns=train.columns
train_columns=train_columns.drop("isFraud")
test.columns=train_columns
test.columns


# In[ ]:


test.shape


# In[ ]:


train.shape


# In[ ]:


for i in train_cat_columns:
    if test[i].max()== train[i].max():
            test = pd.get_dummies(test, columns = [i])
            train=pd.get_dummies(train, columns = [i])

    


# In[ ]:


train.shape, test.shape


# In[ ]:


train_TransactionID= train["TransactionID"]
test_TransactionID=test["TransactionID"]
X= train.sort_values('TransactionDT').drop([ 'TransactionDT', 'TransactionID'], axis=1)
y = train.sort_values('TransactionDT')['isFraud']
test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)
del train


# In[ ]:


X=X.drop("isFraud", axis=1)


# In[ ]:


X.head()


# In[ ]:


X.shape, test.shape


# In[ ]:


params = {'num_leaves': 491,
          'min_child_weight': 0.03454472573214212,
          'feature_fraction': 0.3797454081646243,
          'bagging_fraction': 0.4181193142567742,
          'min_data_in_leaf': 106,
          'objective': 'binary',
          'max_depth': -1,
          'learning_rate': 0.006883242363721497,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'auc',
          "verbosity": -1,
          'reg_alpha': 0.3899927210061127,
          'reg_lambda': 0.6485237330340494,
          'random_state': 47
         }


# In[ ]:


import seaborn as sns
import lightgbm as lgb
import gc
from time import time
import datetime
from tqdm import tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score
warnings.simplefilter('ignore')
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


folds = TimeSeriesSplit(n_splits=5)

aucs = list()
feature_importances = pd.DataFrame()
feature_importances['feature'] = X.columns

training_start_time = time()
for fold, (trn_idx, test_idx) in enumerate(folds.split(X, y)):
    start_time = time()
    print('Training on fold {}'.format(fold + 1))
    
    trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
    clf = lgb.train(params, trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
    
    feature_importances['fold_{}'.format(fold + 1)] = clf.feature_importance()
    aucs.append(clf.best_score['valid_1']['auc'])
    
    print('Fold {} finished in {}'.format(fold + 1, str(datetime.timedelta(seconds=time() - start_time))))
print('-' * 30)
print('Training has finished.')
print('Total training time is {}'.format(str(datetime.timedelta(seconds=time() - training_start_time))))
print('Mean AUC:', np.mean(aucs))
print('-' * 30)


# In[ ]:


feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(folds.n_splits)]].mean(axis=1)
feature_importances.to_csv('feature_importances.csv')

plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
plt.title('50 TOP feature importance over {} folds average'.format(folds.n_splits));


# In[ ]:


# clf right now is the last model, trained with 80% of data and validated with 20%
best_iter = clf.best_iteration


# In[ ]:


clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)
clf.fit(X, y)


# In[ ]:


#set the output as a dataframe and convert to csv file named submission.csv
predictions = clf.predict_proba(test)[:, 1]
output = pd.DataFrame({ "TransactionID" : test_TransactionID, "isFraud": predictions })
output.to_csv('submission_lgbm.csv', index=False)


# In[ ]:


output


# In[ ]:




