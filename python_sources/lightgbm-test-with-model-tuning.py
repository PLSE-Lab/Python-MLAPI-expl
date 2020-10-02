# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


import numpy as np
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.simplefilter(action = 'ignore', category = FutureWarning)

@contextmanager

def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns = categorical_columns, dummy_na = nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns



# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv("../input/application_train.csv", nrows = num_rows)
    test_df = pd.read_csv("../input/application_test.csv", nrows = num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index()
    #Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    EDUCATION_TYPE_LVL = {'Academic degree':1,
                      'Higher education':1,
                      'Incomplete higher':2,
                      'Lower secondary':2,
                      'Secondary / secondary special':2}
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].map(EDUCATION_TYPE_LVL)
    NAME_INCOME_TYPE_LVL = {'Businessman':1,
                    'Pensioner':1,
                    'State servant':1,
                    'Commercial associate':2,
                    'Working':3,
                    'Unemployed':3,
                    'Maternity leave':3,
                    'Student':3                    
                    }
    
    df['NAME_INCOME_TYPE'] = df['NAME_INCOME_TYPE'].map(NAME_INCOME_TYPE_LVL)
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    #NaN values for DAYS_EMPLOYED: 365243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace = True)
    #Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_PER_AGE'] = df['AMT_INCOME_TOTAL'] / df['DAYS_BIRTH']
    df['INCOME_PER_EMPLOYED'] = df['AMT_INCOME_TOTAL'] / df['DAYS_EMPLOYED']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / df['CNT_CHILDREN']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['AMT_CREDIT_USED_RATE'] = df['AMT_GOODS_PRICE']/df['AMT_CREDIT']
    
    del test_df
    gc.collect()
    return df




# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('../input/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('../input/bureau_balance.csv', nrows = num_rows)
    #bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'STATUS':['max']}
#                       'MONTHS_BALANCE':['max', 'min', 'size']}
#    for col in bb_cat:
#        bb_aggregations[col] = ['mean']

    bb['STATUS'].replace('C', np.nan, inplace = True)
    bb['STATUS'].replace('X', np.nan, inplace = True)
    bb['STATUS'] = bb['STATUS'].notnull().astype('int')
    
#    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
#    bb_agg.columns = pd.Index([e[0] + '_' + e[1].upper() for e in bb_agg.columns.tolist()])

    bb_agg_1 = bb[bb['MONTHS_BALANCE'] >= -10].groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg_1.columns = pd.Index([e[0] + '_' + e[1].upper() + '_10m'.upper() for e in bb_agg_1.columns.tolist()])
    bb_agg_3 = bb[bb['MONTHS_BALANCE'] >= -30].groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg_3.columns = pd.Index([e[0] + '_' + e[1].upper() + '_30m'.upper() for e in bb_agg_3.columns.tolist()])
    bb_agg_6 = bb[bb['MONTHS_BALANCE'] >= -60].groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg_6.columns = pd.Index([e[0] + '_' + e[1].upper() + '_60m'.upper() for e in bb_agg_6.columns.tolist()])
    bb_agg_10 = bb[bb['MONTHS_BALANCE'] >= -100].groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg_10.columns = pd.Index([e[0] + '_' + e[1].upper() + '_100m'.upper() for e in bb_agg_10.columns.tolist()])
    
    bureau = bureau.join(bb_agg_1, how = 'left', on = 'SK_ID_BUREAU')
    bureau = bureau.join(bb_agg_3, how = 'left', on = 'SK_ID_BUREAU')
    bureau = bureau.join(bb_agg_6, how = 'left', on = 'SK_ID_BUREAU')
    bureau = bureau.join(bb_agg_10, how = 'left', on = 'SK_ID_BUREAU')

    del bb, bb_agg_1, bb_agg_3, bb_agg_6, bb_agg_10
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    
    num_aggregations = {
#            'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
            'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
            'DAYS_CREDIT_UPDATE': ['mean'],
            'CREDIT_DAY_OVERDUE': ['max', 'mean'],
            'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
            'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
            'AMT_CREDIT_SUM_OVERDUE': ['mean', 'max'],
            'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
            'AMT_ANNUITY': ['max', 'mean', 'sum'],
            'CNT_CREDIT_PROLONG': ['sum', 'max'],
#            'MONTHS_BALANCE_MIN': ['min'],
#            'MONTHS_BALANCE_MAX': ['max'],
#            'MONTHS_BALANCE_SIZE': ['mean', 'sum'],
            'SK_ID_BUREAU':['nunique'],
            'STATUS_MAX_10M':['max'],
            'STATUS_MAX_30M':['max'],
            'STATUS_MAX_60M':['max'],
            'STATUS_MAX_100M':['max']
            }
    
    # Bureau and bureau_balance categorical features
    
    cat_aggregations = {}
    for cat in bureau_cat:
        cat_aggregations[cat] = ['mean', 'sum']
    
#    for cat in bb_cat:
#        cat_aggregations[cat + '_MEAN'] = ['mean']
    
#    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
#    bureau_agg.columns = pd.Index(['BURO_' + e[0] + '_' + e[1].upper() for e in bureau_agg.columns.tolist()])
    bureau_agg_1 = bureau[bureau['DAYS_CREDIT'] >= -1*360].groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg_1.columns = pd.Index(['BURO_' + e[0] + '_' + e[1].upper() + '_1y'.upper() for e in bureau_agg_1.columns.tolist()])
    bureau_agg_3 = bureau[bureau['DAYS_CREDIT'] >= -3*360].groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg_3.columns = pd.Index(['BURO_' + e[0] + '_' + e[1].upper() + '_3y'.upper() for e in bureau_agg_3.columns.tolist()])
    bureau_agg_6 = bureau[bureau['DAYS_CREDIT'] >= -6*360].groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg_6.columns = pd.Index(['BURO_' + e[0] + '_' + e[1].upper() + '_6y'.upper() for e in bureau_agg_6.columns.tolist()])
    bureau_agg_10 = bureau[bureau['DAYS_CREDIT'] >= -10*360].groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg_10.columns = pd.Index(['BURO_' + e[0] + '_' + e[1].upper() + '_10y'.upper() for e in bureau_agg_10.columns.tolist()])

    bureau_agg = bureau_agg_10.join(bureau_agg_1, how = 'left', on = 'SK_ID_CURR')
    bureau_agg = bureau_agg.join(bureau_agg_3, how = 'left', on = 'SK_ID_CURR')
    bureau_agg = bureau_agg.join(bureau_agg_6, how = 'left', on = 'SK_ID_CURR')
    
    
    del bureau_agg_1, bureau_agg_3, bureau_agg_6, bureau_agg_10
    gc.collect()
    
    # Bureau: Active credits - using only numerical aggregations
    active1 = bureau[(bureau['CREDIT_ACTIVE_Active'] == 1) & (bureau['DAYS_CREDIT'] >= -1*360)]
    active_agg1 = active1.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg1.columns = pd.Index(['ACTIVE_' + e[0] + '_' + e[1].upper() + '_1y'.upper() for e in active_agg1.columns.tolist()])
    active3 = bureau[(bureau['CREDIT_ACTIVE_Active'] == 1) & (bureau['DAYS_CREDIT'] >= -3*360)]
    active_agg3 = active3.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg3.columns = pd.Index(['ACTIVE_' + e[0] + '_' + e[1].upper() + '_3y'.upper() for e in active_agg3.columns.tolist()])
    active6 = bureau[(bureau['CREDIT_ACTIVE_Active'] == 1) & (bureau['DAYS_CREDIT'] >= -6*360)]
    active_agg6 = active6.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg6.columns = pd.Index(['ACTIVE_' + e[0] + '_' + e[1].upper() + '_6y'.upper() for e in active_agg6.columns.tolist()])
    active10 = bureau[(bureau['CREDIT_ACTIVE_Active'] == 1) & (bureau['DAYS_CREDIT'] >= -10*360)]
    active_agg10 = active10.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg10.columns = pd.Index(['ACTIVE_' + e[0] + '_' + e[1].upper() + '_10y'.upper() for e in active_agg10.columns.tolist()])

    bureau_agg = bureau_agg.join(active_agg1, how = 'left', on = 'SK_ID_CURR')
    bureau_agg = bureau_agg.join(active_agg3, how = 'left', on = 'SK_ID_CURR')
    bureau_agg = bureau_agg.join(active_agg6, how = 'left', on = 'SK_ID_CURR')
    bureau_agg = bureau_agg.join(active_agg10, how = 'left', on = 'SK_ID_CURR')
    
    
    del active1, active3, active6, active10
    del active_agg1, active_agg3, active_agg6, active_agg10
    
    gc.collect()
    
    
    # Bureau: Closed credits - using only numerical aggregations
    closed1 = bureau[(bureau['CREDIT_ACTIVE_Closed'] == 1) & (bureau['DAYS_CREDIT'] >= -1*360)]
    closed_agg1 = closed1.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg1.columns = pd.Index(['CLOSED_' + e[0] + '_' + e[1].upper() + '_1y'.upper() for e in closed_agg1.columns.tolist()])
    closed3 = bureau[(bureau['CREDIT_ACTIVE_Closed'] == 1) & (bureau['DAYS_CREDIT'] >= -3*360)]
    closed_agg3 = closed3.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg3.columns = pd.Index(['CLOSED_' + e[0] + '_' + e[1].upper() + '_3y'.upper() for e in closed_agg3.columns.tolist()])
    closed6 = bureau[(bureau['CREDIT_ACTIVE_Closed'] == 1) & (bureau['DAYS_CREDIT'] >= -6*360)]
    closed_agg6 = closed6.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg6.columns = pd.Index(['CLOSED_' + e[0] + '_' + e[1].upper() + '_6y'.upper() for e in closed_agg6.columns.tolist()])
    closed10 = bureau[(bureau['CREDIT_ACTIVE_Closed'] == 1) & (bureau['DAYS_CREDIT'] >= -10*360)]
    closed_agg10 = closed10.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg10.columns = pd.Index(['CLOSED_' + e[0] + '_' + e[1].upper() + '_10y'.upper() for e in closed_agg10.columns.tolist()])

    bureau_agg = bureau_agg.join(closed_agg1, how = 'left', on = 'SK_ID_CURR')
    bureau_agg = bureau_agg.join(closed_agg3, how = 'left', on = 'SK_ID_CURR')
    bureau_agg = bureau_agg.join(closed_agg6, how = 'left', on = 'SK_ID_CURR')
    bureau_agg = bureau_agg.join(closed_agg10, how = 'left', on = 'SK_ID_CURR')
    
    del bureau
    del closed1, closed3, closed6, closed10
    del closed_agg1, closed_agg3, closed_agg6, closed_agg10
    
    gc.collect()
    return bureau_agg


# Preprocess previous_application.csv
def previous_application(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('../input/previous_application.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category = True)
    
    # Days 365243 values -> nan
    
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    
    # Add feature: value ask/value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION']/prev['AMT_CREDIT']
    prev['AMT_CREDIT_USED_RATE'] = prev['AMT_GOODS_PRICE']/prev['AMT_CREDIT']
    prev['PAYMENT_RATE'] = prev['AMT_ANNUITY'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    
    num_aggregations = {'AMT_ANNUITY': ['min', 'max', 'mean'],
            'AMT_APPLICATION': ['min', 'max', 'mean'],
            'AMT_CREDIT': ['min', 'max', 'mean'],
            'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
            'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
            'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
            'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
            'DAYS_DECISION': ['min', 'max', 'mean'],
            'CNT_PAYMENT': ['mean', 'sum'],
            'AMT_CREDIT_USED_RATE':['min', 'max', 'mean', 'var'],
            'PAYMENT_RATE':['min', 'max', 'mean', 'var']
            }
    
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
        
        
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + '_' + e[1].upper() for e in prev_agg.columns.tolist()])
    
    # Previous Applications:Approved Applications -only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + '_' + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how = 'left', on = 'SK_ID_CURR')
    
    #Previous Application: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + '_' + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how = 'left', on = 'SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg


# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('../input/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category = True)
    
    # Features
    aggregations = {'MONTHS_BALANCE': ['max', 'mean', 'size'],
            'SK_DPD': ['max', 'mean'],
            'SK_DPD_DEF': ['max', 'mean']}
    
    for cat in cat_cols:
        aggregations[cat] = ['mean', 'sum']
        
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + '_' + e[1].upper() for e in pos_agg.columns.tolist()])
    
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg



# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('../input/installments_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category = True)
    
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    
    
    for cat in cat_cols:
        aggregations[cat] = ['mean', 'sum']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg


# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('../input/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    cc['AMT_DRAWINGS_ATM_CURRENT_RATE'] = cc['AMT_DRAWINGS_ATM_CURRENT']/cc['AMT_DRAWINGS_CURRENT']
    cc['AMT_DRAWINGS_POS_CURRENT_RATE'] = cc['AMT_DRAWINGS_POS_CURRENT']/cc['AMT_DRAWINGS_CURRENT']
    cc['AMT_DRAWINGS_PER'] = cc['AMT_DRAWINGS_CURRENT']/cc['CNT_DRAWINGS_CURRENT']
    cc['AMT_DRAWINGS_ATM_PER'] = cc['AMT_DRAWINGS_ATM_CURRENT']/cc['CNT_DRAWINGS_ATM_CURRENT']
    cc['AMT_DRAWINGS_POS_PER'] = cc['AMT_DRAWINGS_POS_CURRENT']/cc['CNT_DRAWINGS_POS_CURRENT']
    
    aggregations = {'AMT_DRAWINGS_ATM_CURRENT_RATE':['mean', 'max', 'min'],
                    'AMT_DRAWINGS_POS_CURRENT_RATE':['mean', 'max', 'min'],
                    'AMT_DRAWINGS_PER':['mean', 'max', 'min'],
                    'AMT_DRAWINGS_ATM_PER':['mean', 'max', 'min'],
                    'AMT_DRAWINGS_POS_PER':['mean', 'max', 'min']
                    
            }
    
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    for cat in cat_cols:
        aggregations[cat] = ['mean', 'sum']
    cc1 = cc[cc['MONTHS_BALANCE'] >= -10]
    cc_agg1 = cc1.groupby('SK_ID_CURR').agg(aggregations)
    cc_agg1.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() + '_10m'.upper() for e in cc_agg1.columns.tolist()])
    cc3 = cc[cc['MONTHS_BALANCE'] >= -30]
    cc_agg3 = cc3.groupby('SK_ID_CURR').agg(aggregations)
    cc_agg3.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() + '_30m'.upper() for e in cc_agg3.columns.tolist()])
    cc6 = cc[cc['MONTHS_BALANCE'] >= -60]
    cc_agg6 = cc6.groupby('SK_ID_CURR').agg(aggregations)
    cc_agg6.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() + '_60m'.upper() for e in cc_agg6.columns.tolist()])
    cc10 = cc[cc['MONTHS_BALANCE'] >= -100]
    cc_agg10 = cc10.groupby('SK_ID_CURR').agg(aggregations)
    cc_agg10.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() + '_100m'.upper() for e in cc_agg10.columns.tolist()])
    cc_agg = cc_agg10.join(cc_agg6, how = 'left', on = 'SK_ID_CURR')
    cc_agg = cc_agg.join(cc_agg3, how = 'left', on = 'SK_ID_CURR')
    cc_agg = cc_agg.join(cc_agg1, how = 'left', on = 'SK_ID_CURR')

    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

    del cc, cc1, cc3, cc6, cc10 
    del cc_agg1, cc_agg3, cc_agg6, cc_agg10
    gc.collect()
    return cc_agg


# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code


def kfold_lightgbm(df, num_folds, stratified = False, debug = False):
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
   
    
    print('Starting LightGBM. Train shape: {}, test shape: {}'.format(train_df.shape, test_df.shape))

    
    del df
    gc.collect()
    
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = 1001)
    else:
        folds = KFold(n_splits = num_folds, shuffle = True, random_state = 1001)
    
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    #feats = [f for f in feats if f in cols_sel_bureau]
    OrderedDict = dict([('max_depth', 0.0),
              ('n_estimators', 92.0),
              ('learning_rate', 5.0),
              ('subsample', 3.0),
              ('colsample_bytree', 2.0),
              ('min_child_weight', 0.0),
              ('max_bin', 5.0),
              ('num_leaves', 8.0),
              ('reg_alpha', 1.0),
              ('reg_lambda', 6.0),
              ('min_split_gain', 20.0),
              ('min_data_in_leaf', 9.0)])
              
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
        print('train_x shape: {}, valid_x shape: {}'.format(train_x.shape, valid_x.shape))
        max_depth = int(OrderedDict['max_depth'] + 5)
        n_estimators = int(OrderedDict['n_estimators']*100 + 500)
        learning_rate = OrderedDict['learning_rate']*0.02 + 0.05
        subsample = OrderedDict['subsample']*0.1 + 0.6
        colsample_bytree = OrderedDict['colsample_bytree']*0.1 + 0.6
        min_child_weight = int(OrderedDict['min_child_weight']*10 + 2)
        max_bin = int(OrderedDict['max_bin']*10 + 5)
        num_leaves = int(OrderedDict['num_leaves']*5 + 5)
        reg_alpha = OrderedDict['reg_alpha']*0.02 + 0.02
        reg_lambda = OrderedDict['reg_lambda']*0.02 + 0.02
        min_split_gain = OrderedDict['min_split_gain']*0.02 + 0.02
        min_data_in_leaf = int(OrderedDict['min_data_in_leaf']*100)
    
        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            colsample_bytree=colsample_bytree,
            subsample=subsample,
            max_depth=max_depth,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_split_gain=min_split_gain,
            min_child_weight=min_child_weight,
            silent=-1,
            verbose=-1, 
            is_unbalance = True,
            max_bin = max_bin,
            min_data_in_leaf = min_data_in_leaf,
            feature_fraction = 0.8
            )
    
        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)
    
        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        #sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
        fold_importance_df = pd.DataFrame()
        booster = clf.booster_
        fold_importance_df["feature"] = booster.feature_name()
        fold_importance_df["importance"] = booster.feature_importance(importance_type = 'split')
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del train_x, train_y, valid_x, valid_y
        gc.collect()
    
    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    if not debug:
        sub_preds = clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1]
        test_df['TARGET'] = sub_preds
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    return feature_importance_df


# Display/plot feature importance
def display_importances(feature_importance_df_):
    cols = feature_importance_df_[['feature', 'importance']].groupby('feature').mean().sort_values(by = 'importance', ascending = False)[:40].index
    best_feature = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize = (8, 10))
    sns.barplot(x = 'importance', y = 'feature', data = best_feature.sort_values(by = 'importance', ascending = False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')



def lightgbm_param(max_depth, n_estimators, learning_rate, subsample, colsample_bytree, min_child_weight, max_bin, num_leaves, reg_alpha, reg_lambda, min_split_gain, min_data_in_leaf):
    global df
    train_df = df[df['TARGET'].notnull()]

    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    train_x, valid_x, train_y, valid_y = train_test_split(train_df[feats], train_df['TARGET'], test_size = 0.3, random_state = 11)
    
    max_depth = int(max_depth + 5)
    n_estimators = int(n_estimators*100 + 500)
    learning_rate = learning_rate*0.02 + 0.05
    subsample = subsample*0.1 + 0.6
    colsample_bytree = colsample_bytree*0.1 + 0.6
    min_child_weight = int(min_child_weight*10 + 2)
    max_bin = int(max_bin*10 + 5)
    num_leaves = int(num_leaves*5 + 5)
    reg_alpha = reg_alpha*0.02 + 0.02
    reg_lambda = reg_lambda*0.02 + 0.02
    min_split_gain = min_split_gain*0.02 + 0.02
    min_data_in_leaf = int(min_data_in_leaf*100)
    
#    print('max_depth:' + str(max_depth))
#    print('n_estimators:' + str(n_estimators))
#    print('learning_rate:' + str(learning_rate))
#    print('subsample:' + str(subsample))
#    print('colsample_bytree:' + str(colsample_bytree))
#    print('min_child_weight:' + str(min_child_weight))
#    print('max_bin:' + str(max_bin))
#    print('num_leaves:' + str(num_leaves))
#    print('reg_alpha:' + str(reg_alpha))
#    print('reg_lambda:' + str(reg_lambda))
#    print('min_split_gain:' + str(min_split_gain))
#    print('min_data_in_leaf:' + str(min_data_in_leaf))
    
    
    
    clf = LGBMClassifier(
    nthread=4,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    num_leaves=num_leaves,
    colsample_bytree=colsample_bytree,
    subsample=subsample,
    max_depth=max_depth,
    reg_alpha=reg_alpha,
    reg_lambda=reg_lambda,
    min_split_gain=min_split_gain,
    min_child_weight=min_child_weight,
    silent=-1,
    verbose=-1, 
    is_unbalance = True,
    max_bin = max_bin,
    min_data_in_leaf = min_data_in_leaf,
    feature_fraction = 0.8
    )
    
    clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
    eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)
    
    oof_preds = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
    auc = roc_auc_score(valid_y, oof_preds)
    return auc


debug = False
num_rows = 10000 if debug else None
df = application_train_test(num_rows)

with timer('Process bureau and bureau_balance'):
    bureau = bureau_and_balance(num_rows)
    print('Bureau df shape:', bureau.shape)
    df =df.join(bureau, how = 'left', on = 'SK_ID_CURR')
    del bureau
    gc.collect()
    
with timer('Process previous_applications'):
    prev = previous_application(num_rows)
    print('Previous applications df shape:', prev.shape)
    df =df.join(prev, how = 'left', on = 'SK_ID_CURR')
    del prev
    gc.collect()
    
with timer('Process POS-CASH balance'):
    pos = pos_cash(num_rows)
    print('Pos-cash balance df shape:', pos.shape)
    df =df.join(pos, how = 'left', on = 'SK_ID_CURR')
    del pos
    gc.collect()
    
with timer('Process installments payments'):
    ins = installments_payments(num_rows)
    print('Installments payments df shape', ins.shape)
    df =df.join(ins, how = 'left', on = 'SK_ID_CURR')
    del ins
    gc.collect()
    
with timer('Process credit card balance'):
    cc = credit_card_balance(num_rows)
    print('Credit card balance df shape', cc.shape)
    df =df.join(cc, how = 'left', on = 'SK_ID_CURR')
    del cc
    gc.collect()

missing_rate = df.isnull().sum()/df.shape[0]
missing_rate = missing_rate.reset_index()
missing_cols = missing_rate[missing_rate[0] > 0.7]['index'].tolist()
feats = [f for f in df.columns if f not in missing_cols]
df = df[feats]
df_pearson = df[df['TARGET'].notnull()].drop(columns = ['index']).corr()
df_pearson = df_pearson['TARGET']
df_pearson = df_pearson[(df_pearson < -0.02) | df_pearson > 0.02].index.values
df = df[df_pearson]


with timer('Run LightGBM with kfold'):
    submission_file_name = 'submission.csv'
    feat_importance = kfold_lightgbm(df, num_folds = 6, stratified = False, debug = debug)


        


#from sklearn.metrics import roc_auc_score
#import numpy as np
#from pyGPGO.covfunc import matern32
#from pyGPGO.acquisition import Acquisition
#from pyGPGO.surrogates.GaussianProcess import GaussianProcess
#from pyGPGO.GPGO import GPGO
#from sklearn.cross_validation import train_test_split
#
#
###search for best parameter
#space = {'max_depth':('int', [0, 5]),
#         'n_estimators':('int', [0, 100]),
#         'learning_rate':('int', [0, 10]),
#         'subsample':('int', [0, 4]),
#         'colsample_bytree':('int', [0, 4]),
#         'min_child_weight':('int', [0, 6]),
#         'max_bin':('int', [0, 6]),
#         'num_leaves':('int', [0, 10]),
#         'reg_alpha':('int', [0, 10]),
#         'reg_lambda':('int', [0, 10]),
#         'min_split_gain':('int', [0, 30]),
#         'min_data_in_leaf':('int', [0, 20])
#         }
#
#
##algo = partial(tpe.suggest, n_startup_jobs = 1)
##best = fmin(lightgbm_param, space, algo = algo, max_evals = 4)
#
#cov = matern32()
#gp = GaussianProcess(cov)
#acq = Acquisition(mode='ExpectedImprovement')
#
#
#np.random.seed(1337)
#gpgo = GPGO(gp, acq, lightgbm_param, space)
#gpgo.run(max_iter=15)
#gpgo.getResult()

