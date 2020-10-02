#!/usr/bin/env python
# coding: utf-8

# # Do feature engineering to improve LightGBM prediction
# This kernel closely follows https://www.kaggle.com/mlisovyi/feature-engineering-lighgbm-with-f1-macro, but instead of making accent on precise prediction, we look what is going on under the hood. 
# 
# The key points:
# - The main point is **to demonstrate the origin of low F1 scores that we achieve on these data**
# - The studies start here: [Confusion matrix studies](#F1-score-across-different-classes)
# - Spoiler: no magical solution. Any feedback/advice is welcome

# # Initial imports and helper functions

# In[ ]:


import numpy as np # linear algebra
import pandas as pd 

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# The following categorical mapping originates from [this kernel](https://www.kaggle.com/mlisovyi/categorical-variables-encoding-function).

# In[ ]:


from sklearn.preprocessing import LabelEncoder

def encode_data(df):
    '''
    The function does not return, but transforms the input pd.DataFrame
    
    Encodes the Costa Rican Household Poverty Level data 
    following studies in https://www.kaggle.com/mlisovyi/categorical-variables-in-the-data
    and the insight from https://www.kaggle.com/c/costa-rican-household-poverty-prediction/discussion/61403#359631
    
    The following columns get transformed: edjefe, edjefa, dependency, idhogar
    The user most likely will simply drop idhogar completely (after calculating houshold-level aggregates)
    '''
    
    yes_no_map = {'no': 0, 'yes': 1}
    
    df['dependency'] = df['dependency'].replace(yes_no_map).astype(np.float32)
    
    df['edjefe'] = df['edjefe'].replace(yes_no_map).astype(np.float32)
    df['edjefa'] = df['edjefa'].replace(yes_no_map).astype(np.float32)
    
    df['idhogar'] = LabelEncoder().fit_transform(df['idhogar'])


# **Drop less important features:**

# In[ ]:


def drop_features(df):
    # Drop SQB variables, as they are just squres of other vars 
    df.drop([f_ for f_ in df.columns if f_.startswith('SQB') or f_ == 'agesq'], axis=1, inplace=True)
    # Drop id's
    df.drop(['Id', 'idhogar'], axis=1, inplace=True)
    # Drop repeated columns
    df.drop(['hhsize', 'female', 'area2'], axis=1, inplace=True)
    return df


# In[ ]:


def convert_OHE2LE(df):
    tmp_df = df.copy(deep=True)
    for s_ in ['pared', 'piso', 'techo', 'abastagua', 'sanitario', 'energcocinar', 'elimbasu', 
               'epared', 'etecho', 'eviv', 'estadocivil', 'parentesco', 
               'instlevel', 'lugar', 'tipovivi',
               'manual_elec']:
        if 'manual_' not in s_:
            cols_s_ = [f_ for f_ in df.columns if f_.startswith(s_)]
        elif 'elec' in s_:
            cols_s_ = ['public', 'planpri', 'noelec', 'coopele']
        sum_ohe = tmp_df[cols_s_].sum(axis=1).unique()
        #deal with those OHE, where there is a sum over columns == 0
        if 0 in sum_ohe:
            print('The OHE in {} is incomplete. A new column will be added before label encoding'
                  .format(s_))
            # dummy colmn name to be added
            col_dummy = s_+'_dummy'
            # add the column to the dataframe
            tmp_df[col_dummy] = (tmp_df[cols_s_].sum(axis=1) == 0).astype(np.int8)
            # add the name to the list of columns to be label-encoded
            cols_s_.append(col_dummy)
            # proof-check, that now the category is complete
            sum_ohe = tmp_df[cols_s_].sum(axis=1).unique()
            if 0 in sum_ohe:
                 print("The category completion did not work")
        tmp_cat = tmp_df[cols_s_].idxmax(axis=1)
        tmp_df[s_ + '_LE'] = LabelEncoder().fit_transform(tmp_cat).astype(np.int16)
        if 'parentesco1' in cols_s_:
            cols_s_.remove('parentesco1')
        tmp_df.drop(cols_s_, axis=1, inplace=True)
    return tmp_df


# # Read in the data and clean it up

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.info(verbose=False)


# In[ ]:


def process_df(df_):
    # fix categorical features
    encode_data(df_)
    #fill in missing values based on https://www.kaggle.com/mlisovyi/missing-values-in-the-data
    for f_ in ['v2a1', 'v18q1', 'meaneduc', 'SQBmeaned']:
        df_[f_] = df_[f_].fillna(0)
    df_['rez_esc'] = df_['rez_esc'].fillna(-1)
    # drop useless columns
    return drop_features(df_)

train = process_df(train)
test = process_df(test)


# Now, let's define `train_test_apply_func` helper function to apply a custom function to a concatenated test+train dataset

# In[ ]:


def train_test_apply_func(train_, test_, func_):
    test_['Target'] = 0
    xx = pd.concat([train_, test_])

    xx_func = func_(xx)
    train_ = xx_func.iloc[:train_.shape[0], :]
    test_  = xx_func.iloc[train_.shape[0]:, :].drop('Target', axis=1)

    del xx, xx_func
    return train_, test_


# In[ ]:


train, test = train_test_apply_func(train, test, convert_OHE2LE)


# In[ ]:


train.info(verbose=False)


# Compare the number of features with `int64` type to the previous info summary. The difference comes from convertion of OHE into LE (`convert_OHE2LE` function)

# # VERY IMPORTANT
# > Note that ONLY the heads of household are used in scoring. All household members are included in test + the sample submission, but only heads of households are scored.

# In[ ]:


X = train.query('parentesco1==1')

# pull out the target variable
y = X['Target'] - 1
X = X.drop(['Target'], axis=1)


# # Model fitting
# 
# We will use LightGBM classifier - LightGBM allows to build very sophysticated models with a very short training time.

# ## Use test subset for early stopping criterion
# 
# This allows us to avoid overtraining and we do not need to optimise the number of trees. We also use F1 macro-averaged score to decide when to stop
# 

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=314, stratify=y)


# In[ ]:


from sklearn.metrics import f1_score
def evaluate_macroF1_lgb(truth, predictions):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.reshape(len(np.unique(truth)),-1).argmax(axis=0)
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', f1, True) 

import lightgbm as lgb
fit_params={"early_stopping_rounds":300, 
            "eval_metric" : evaluate_macroF1_lgb, 
            "eval_set" : [(X_train,y_train), (X_test,y_test)],
            'eval_names': ['train', 'early_stop'],
            'verbose': 100,
            'categorical_feature': 'auto'}


# # LightGBM optimal parameters
# 
# The parameters are optimised with a random search in this kernel: https://www.kaggle.com/mlisovyi/lighgbm-hyperoptimisation-with-f1-macro
# 

# In[ ]:


opt_parameters = {'colsample_bytree': 0.89, 'min_child_samples': 90, 'num_leaves': 14, 'subsample': 0.96}


# # Fit a LGBM classifier

# In[ ]:


def train_lgbm_model(X_, y_, random_state_=None, opt_parameters_={}, fit_params_={}, lr_=0.05):
    clf  = lgb.LGBMClassifier(max_depth=-1, learning_rate=lr_, objective='multiclass',
                             random_state=random_state_, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced')
    clf.set_params(**opt_parameters_)
    return clf.fit(X_, y_, **fit_params_)

clf_final = train_lgbm_model(X_train, y_train, 
                       random_state_=314, 
                       opt_parameters_=opt_parameters,
                       fit_params_=fit_params)


# So the final macro F_1 score on the early-stop test sample is `~0.44`
# # F1 score across different classes
# Let's see if all classes show similar performance

# In[ ]:


from sklearn.metrics import classification_report
def print_report(clf_, X_tr, y_tr, X_tt, y_tt):
    print('------------ Train sample -------------\n', 
          classification_report(y_tr, clf_.predict(X_tr)))
    print('------------ Test sample -------------\n', 
          classification_report(y_tt, clf_.predict(X_tt)))

print_report(clf_final, X_train, y_train, X_test, y_test) 


# As we can see:
# 
#  - in the train sample the the model is able to predict all classes with roughly the same F1 score (the third column)
#  - in the test sample only the last class (*"non vulnerable households"*) can be predicted reliably, while others have very low precision and recall
#  
#  Supposedly this is due to poor separation between three poorest classes, as is shown in this kernel: https://www.kaggle.com/mlisovyi/cluster-analysis-tsne-mds-isomap
#  
#  How can we address it?
#  
#  
# # Let's try to build a model to distinguish between poor classes only:

# In[ ]:


def drop_classes(X_, y_, to_drop_=3):
    XY = pd.concat([X_, y_], axis=1)
    XY = XY.query('Target != @to_drop_')
    return XY.drop('Target', axis=1), XY['Target']

X_train_wo3, y_train_wo3 = drop_classes(X_train, y_train, 3)
X_test_wo3,  y_test_wo3  = drop_classes(X_test, y_test, 3)


# In[ ]:


import copy
fit_params_wo3 = copy.deepcopy(fit_params)
fit_params_wo3['eval_set'] = [(X_train_wo3, y_train_wo3), (X_test_wo3, y_test_wo3)]
fit_params_wo3['verbose'] = 100


# In[ ]:


clf_wo3 = train_lgbm_model(X_train_wo3, y_train_wo3, 
                       random_state_=314, 
                       opt_parameters_=opt_parameters,
                       fit_params_=fit_params_wo3)


# In[ ]:


print_report(clf_final, X_train_wo3, y_train_wo3, X_test_wo3, y_test_wo3) 


# # Nope, the model can not well distinguish between the 3 poorest classes

# In[ ]:





# In[ ]:




