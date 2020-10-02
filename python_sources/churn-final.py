#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[ ]:


# data analysis libraries:
import numpy as np
import pandas as pd

# data visualization libraries:
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

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
import numpy as np, pandas as pd, os, gc
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np,gc # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 500)
import numpy as np
import pandas as pd 
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
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from warnings import filterwarnings
filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
import gc
import time
from contextlib import contextmanager


# # Helper Functions

# In[ ]:


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


# In[ ]:


# FREQUENCY ENCODE TOGETHER
def encode_FE(df1, df2, cols):
    for col in cols:
        df = pd.concat([df1[col],df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')
        print(nm,', ',end='')
        
# LABEL ENCODE
def encode_LE(col,train,test,verbose=False):
    df_comb = pd.concat([train[col],test[col]],axis=0)
    df_comb,_ = df_comb.factorize(sort=True)
    nm = col
    if df_comb.max()>32000: 
        train[nm] = df_comb[:len(train)].astype('int32')
        test[nm] = df_comb[len(train):].astype('int32')
    else:
        train[nm] = df_comb[:len(train)].astype('int16')
        test[nm] = df_comb[len(train):].astype('int16')
    del df_comb; x=gc.collect()
    if verbose: print(nm,', ',end='')
        
# GROUP AGGREGATION MEAN AND STD
# https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda
def encode_AG(main_columns, uids, aggregations, train_df, test_df, 
              fillna=True, usena=False):
    # AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS
    for main_column in main_columns:  
        for col in uids:
            for agg_type in aggregations:
                new_col_name = main_column+'_'+col+'_'+agg_type
                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])
                if usena: temp_df.loc[temp_df[main_column]==-1,main_column] = np.nan
                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(
                                                        columns={agg_type: new_col_name})

                temp_df.index = list(temp_df[col])
                temp_df = temp_df[new_col_name].to_dict()   

                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')
                test_df[new_col_name]  = test_df[col].map(temp_df).astype('float32')
                
                if fillna:
                    train_df[new_col_name].fillna(-1,inplace=True)
                    test_df[new_col_name].fillna(-1,inplace=True)
                
                print("'"+new_col_name+"'",', ',end='')
                
# COMBINE FEATURES
def encode_CB(col1,col2,train,test):
    nm = col1+'_'+col2
    train[nm] = train[col1].astype(str)+'_'+train[col2].astype(str)
    test[nm] = test[col1].astype(str)+'_'+test[col2].astype(str) 
    encode_LE(nm,train,test)
# GROUP AGGREGATION NUNIQUE
def encode_AG2(main_columns, uids, train_df, test_df):
    for main_column in main_columns:  
        for col in uids:
            comb = pd.concat([train_df[[col]+[main_column]],test_df[[col]+[main_column]]],axis=0)
            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()
            train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')
            test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')
            print(col+'_'+main_column+'_ct, ',end='')


# In[ ]:


get_ipython().run_cell_magic('time', '', '# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n# WARNING! THIS CAN DAMAGE THE DATA \ndef reduce_mem_usage2(df):\n    """ iterate through all the columns of a dataframe and modify the data type\n        to reduce memory usage.        \n    """\n    start_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage of dataframe is {:.2f} MB\'.format(start_mem))\n    \n    for col in df.columns:\n        col_type = df[col].dtype\n        \n        if col_type != object:\n            c_min = df[col].min()\n            c_max = df[col].max()\n            if str(col_type)[:3] == \'int\':\n                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                    df[col] = df[col].astype(np.int8)\n                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                    df[col] = df[col].astype(np.int16)\n                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                    df[col] = df[col].astype(np.int32)\n                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                    df[col] = df[col].astype(np.int64)  \n            else:\n                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                    df[col] = df[col].astype(np.float16)\n                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                    df[col] = df[col].astype(np.float32)\n                else:\n                    df[col] = df[col].astype(np.float64)\n        else:\n            df[col] = df[col].astype(\'category\')\n\n    end_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage after optimization is: {:.2f} MB\'.format(end_mem))\n    print(\'Decreased by {:.1f}%\'.format(100 * (start_mem - end_mem) / start_mem))\n    \n    return df')


# In[ ]:


def base_models(X,y,test,test_Exited):
    
   
    
    
    from sklearn.model_selection import cross_val_score, KFold 
  

    
    results = []
    A = []
    
    names = ["LogisticRegression","GaussianNB","KNN",
             "CART","RF","GBM","XGBoost","LightGBM","CatBoost"]
    
    
    classifiers = [LogisticRegression(),GaussianNB(), KNeighborsClassifier(),
                  RandomForestClassifier(), GradientBoostingClassifier(),
                  XGBClassifier(), LGBMClassifier(), CatBoostClassifier(verbose = False)]
    
    
    for name, clf in zip(names, classifiers):
        
        folds = StratifiedKFold(n_splits=5)
        cv_results = cross_val_score(clf, X, y, cv = folds, scoring = "accuracy")
        results.append(cv_results)
        A.append(name)
        clf.fit(X, y)
        predictions = clf.predict_proba(test)[:, 1]
        real_test_y=pd.DataFrame(test_Exited)
        real_test_y["predictions"]=predictions
        real_test_y.loc[:,"predictions"]=round((real_test_y.loc[:,"predictions"]).astype("int") )
        z=accuracy_score(real_test_y.loc[:,"Exited"],real_test_y.loc[:,"predictions"] )
        test_score="   test score : "
        msg = "%s: %f (%f) %s %f" % (name, cv_results.mean() , cv_results.std(),test_score, z)
        print(msg)
       


# In[ ]:


def love():
    print('\n'.join([''.join([(' I_Love_Data_Science_'[(x-y) % len('I_Love_Data_Science_')] if ((x*0.05)**2+(y*0.1)**2-1)**3-(x*0.05)**2*(y*0.1)**3 <= 0 else ' ') for x in range(-30, 30)]) for y in range(15, -15, -1)]))


# # Combine

# In[ ]:


def combine():
     with timer('Combining :'):   
        print('Combining Start...')
        # Read train and test data with pd.read_csv():
        data=pd.read_csv("/kaggle/input/churn-modelling/Churn_Modelling.csv")
        train, test = train_test_split(data, test_size=0.30, random_state=4)
        test_Exited=test["Exited"]
        test=test.drop(["Exited"], axis=1)
        
        

        return train,test,test_Exited


# > # Preprocessing and Feature Engineering

# In[ ]:


def pre_processing_and_feature_engineering():
    train,test,test_Exited=combine()   
    
    with timer('Preprocessing and Feature Engineering'):
        print('-' * 30)
        print('Preprocessing and Feature Engineering start...')
        print('-' * 10)
        print("we can drop RowNumber column")
        train = train.drop(["RowNumber"],axis=1)
        test = test.drop(["RowNumber"],axis=1)
        print("-")
        print("Creating New Features...")
        bins = [0, 20, 30, 40, 50, 60, 70, np.inf]
        mylabels = [ 'Child', 'Young', 'Young Adult', 'Adult', 'Senior',"Old","Death"]
        train['AgeGroup'] = pd.cut(train["Age"], bins, labels = mylabels)
        test['AgeGroup'] = pd.cut(test["Age"], bins, labels = mylabels)
        main_columns=test._get_numeric_data().columns.drop(['CustomerId'])
        categorical_columns=test.columns.drop(main_columns)
        categorical_columns=categorical_columns.drop(['CustomerId'])           
        bins = [0, 585, 653, 718,np.inf]
        mylabels = [ 'v_low', 'low', 'medium', 'high']
        train["CreditScoreGroup"] = pd.cut(train["CreditScore"], bins, labels = mylabels)
        test["CreditScoreGroup"] = pd.cut(test["CreditScore"], bins, labels = mylabels)
        bins = [-1, 97006,  127290,np.inf]
        mylabels = [ 'low',  'medium', 'high']
        train["BalanceGroup"] = pd.cut(train["Balance"], bins, labels = mylabels)
        test["BalanceGroup"] = pd.cut(test["Balance"], bins, labels = mylabels)
        bins = [0, 50607,  99041,149383,np.inf]
        mylabels = [ 'low',  'medium', 'high',"v_high"]
        train["EstimatedSalaryGroup"] = pd.cut(train["EstimatedSalary"], bins, labels = mylabels)
        test["EstimatedSalaryGroup"] = pd.cut(test["EstimatedSalary"], bins, labels = mylabels)
        # FREQUENCY ENCODE TOGETHER
        encode_FE(train, test, [ 'NumOfProducts', 'Tenure'])
        # COMBINE FEATURES
        encode_CB('NumOfProducts','Tenure',train,test)
        encode_CB('NumOfProducts_Tenure','Gender',train,test)
        encode_CB("EstimatedSalaryGroup","BalanceGroup",train,test)
        encode_CB('EstimatedSalaryGroup_BalanceGroup',"CreditScoreGroup",train,test)
        # GROUP AGGREGATION MEAN AND STD
        encode_AG(["Balance","EstimatedSalary","CreditScore"], ["NumOfProducts",'NumOfProducts_Tenure','EstimatedSalaryGroup_BalanceGroup_CreditScoreGroup'], ["mean","std"], train, test)
        # GROUP AGGREGATION NUNIQUE
        encode_AG2([ 'Surname'], ['NumOfProducts_Tenure','EstimatedSalaryGroup_BalanceGroup_CreditScoreGroup'], train, test)
        print("Creating New Features Finished")
        print('-' * 10)
        print('Label Coding and One Hot Encoding Start...')
        train=pd.get_dummies(train, columns = ["Geography"])
        test = pd.get_dummies(test, columns = ["Geography"])
        train=pd.get_dummies(train, columns = ["AgeGroup"])
        test = pd.get_dummies(test, columns = ["AgeGroup"])

        main_columns=test._get_numeric_data().columns.drop(['CustomerId'])
        categorical_columns=test.columns.drop(main_columns)
        categorical_columns=categorical_columns.drop(['CustomerId'])
        from sklearn import preprocessing
        for i in categorical_columns:
                lbe=preprocessing.LabelEncoder()
                train[i]=lbe.fit_transform(train[i].astype(str))
                test[i]=lbe.fit_transform(test[i].astype(str))
   
        
        
        
       
       
          
        
                
        print('-' * 10)
        return train,test,test_Exited


# In[ ]:


def modeling():
    train,test,test_Exited=pre_processing_and_feature_engineering()
    with timer('Machine Learning '):
        train_CustomerId= train["CustomerId"]
        test_CustomerId=test["CustomerId"]

        X= train.drop(['CustomerId',"Exited"], axis=1)
        y = train['Exited']
        test = test.drop(['CustomerId'], axis=1)
        
        print('-')
        print('-' * 20)
        print('Machine Learning start ... ')
        base_models(X,y,test,test_Exited)
        params=  {
                    'task': 'train',
                    'boosting_type': 'gbdt',
                    'objective': 'binary',
                    'metric': 'auc',
                    'num_leaves': 35,
                    'learning_rate': 0.05,
                    'verbose': -1}
        from sklearn.model_selection import StratifiedKFold
        folds = StratifiedKFold(n_splits=5)
        folds.get_n_splits(X, y)

        aucs = list()

        fold=-1
        
        for trn_idx, test_idx in folds.split(X, y):
            fold=fold+1
            
            print('Training on fold {}'.format(fold + 1))

            trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
            val_data = lgb.Dataset(X.iloc[test_idx], label=y.iloc[test_idx])
            clf = lgb.train(params,trn_data, 10000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)


            aucs.append(clf.best_score['valid_1']['auc'])

            
        print('-' * 30)
        print('Training has finished.')
       
        print('Mean AUC:', np.mean(aucs))
        print('-' * 30)
        #clf right now is the last model, trained with 80% of data and validated with 20%
        best_iter = clf.best_iteration
        clf = lgb.LGBMClassifier(**params, num_boost_round=best_iter)
        clf.fit(X, y)
        
        predictions = clf.predict_proba(test)[:, 1]
        real_test_y=pd.DataFrame(test_Exited)
        real_test_y["predictions"]=predictions
        real_test_y.loc[:,"predictions"]=round(real_test_y.loc[:,"predictions"] ).astype(int)
        print("Final model real accuracy LGBM :",accuracy_score(real_test_y.loc[:,"Exited"],real_test_y.loc[:,"predictions"] ))
        love()
        feature_imp = pd.DataFrame(sorted(zip(clf.feature_importances_,X.columns)), columns=['Value','Feature'])
        feature_imp.to_excel('feature_importances.xlsx')
        plt.figure(figsize=(20, 10))
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))
        plt.title('LightGBM Features (avg over folds)')
        plt.tight_layout()
        plt.show()
        plt.savefig('lgbm_importances-01.png')

        return 


# # main

# In[ ]:


def main():
    with timer('Full Model Run '):
        print("Full Model Run Start...")
        print('-' * 50)
        modeling() 
        print('-' * 50)


# In[ ]:


if __name__ == "__main__":
     main()

