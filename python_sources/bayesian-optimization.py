# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from bayes_opt import BayesianOptimization

import gc


def days_employed(x):
    if x > 300000:
        x = -np.random.randint(20000, high=25000)
    return x


def one_hot_encoder(df):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def amt_rank_log(df):
    amt_columns = [col for col in df.columns if 'AMT' in col and 'AMT_REQ' not in col]

    df_amt_rank = df[amt_columns].rank()
    rank_columns = []
    for col in df_amt_rank.columns:
        col = col + '_RANK'
        rank_columns.append(col)
    df_amt_log = df[amt_columns].apply(np.log10)
    log_columns = []
    for col in df_amt_log.columns:
        col = col + '_LOG'
        log_columns.append(col)
    df = df.join(df_amt_rank, rsuffix='_RANK').join(df_amt_log, rsuffix='_LOG')
    
    return df, rank_columns, log_columns


def func_application():
    application_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv').set_index(['SK_ID_CURR']).sort_index()
    application_test = pd.read_csv('../input/home-credit-default-risk/application_test.csv').set_index(['SK_ID_CURR']).sort_index()
    
    application = application_train.append(application_test).sort_index()
    
    del application_train, application_test
    gc.collect()
    
    application = application[application['CODE_GENDER'] != 'XNA']
    
    inc_by_org_train = application[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    
    application['DAYS_EMPLOYED'] = application['DAYS_EMPLOYED'].apply(days_employed)
    application['CREDIT_TO_GOODS_RATIO'] = application['AMT_CREDIT'] / application['AMT_GOODS_PRICE']
    application['ANNUITY_TO_INCOME_RATIO'] = application['AMT_ANNUITY'] / application['AMT_INCOME_TOTAL']
    application['AMT_REQ_CREDIT_BUREAU_TOTAL'] = application['AMT_REQ_CREDIT_BUREAU_HOUR'] + application['AMT_REQ_CREDIT_BUREAU_DAY'] + application['AMT_REQ_CREDIT_BUREAU_WEEK'] + application['AMT_REQ_CREDIT_BUREAU_MON'] + application['AMT_REQ_CREDIT_BUREAU_QRT'] + application['AMT_REQ_CREDIT_BUREAU_YEAR']
    application['ANNUITY_TO_CREDIT_RATIO'] = application['AMT_ANNUITY'] / application['AMT_CREDIT']
    application['NEW_EMPLOY_TO_BIRTH_RATIO'] = application['DAYS_EMPLOYED'] / application['DAYS_BIRTH']
    application['NEW_ANNUITY_TO_INCOME_RATIO'] = application['AMT_ANNUITY'] / application['AMT_INCOME_TOTAL']
    application['AMT_INCOME_PER_CHLD'] = application['AMT_INCOME_TOTAL'] / application['CNT_CHILDREN']
    application['NEW_SOURCES_PROD'] = application['EXT_SOURCE_1'] * application['EXT_SOURCE_2'] * application['EXT_SOURCE_3']
    application['NEW_EXT_SOURCES_MEAN'] = application[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    application['NEW_SCORES_STD'] = application[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    application['NEW_SCORES_max'] = application[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
    application['NEW_SCORES_STD'] = application['NEW_SCORES_STD'].fillna(application['NEW_SCORES_STD'].mean())
    application['NEW_CAR_TO_BIRTH_RATIO'] = application['OWN_CAR_AGE'] / application['DAYS_BIRTH']
    application['NEW_CAR_TO_EMPLOY_RATIO'] = application['OWN_CAR_AGE'] / application['DAYS_EMPLOYED']
    application['NEW_PHONE_TO_BIRTH_RATIO'] = application['DAYS_LAST_PHONE_CHANGE'] / application['DAYS_BIRTH']
    application['NEW_PHONE_TO_BIRTH_RATIO'] = application['DAYS_LAST_PHONE_CHANGE'] / application['DAYS_EMPLOYED']
    application['NEW_CREDIT_TO_INCOME_RATIO'] = application['AMT_CREDIT'] / application['AMT_INCOME_TOTAL']
    application['NEW_INC_BY_ORG'] = application['ORGANIZATION_TYPE'].map(inc_by_org_train)
    application['AMT_INCOME_PER_FAM_MEMBERS'] = application['AMT_INCOME_TOTAL'] / application['CNT_FAM_MEMBERS']
    application['missing_EXT_SOURCE_1'] = application['EXT_SOURCE_1'].isna() 
    application['missing_EXT_SOURCE_2'] = application['EXT_SOURCE_2'].isna() 
    application['missing_EXT_SOURCE_3'] = application['EXT_SOURCE_3'].isna()
    application['NEW_EXT_SOURCES_1_TO_MEAN_RATIO'] = application['EXT_SOURCE_1'] / application['NEW_EXT_SOURCES_MEAN']
    application['NEW_EXT_SOURCES_2_TO_MEAN_RATIO'] = application['EXT_SOURCE_2'] / application['NEW_EXT_SOURCES_MEAN']
    application['NEW_EXT_SOURCES_3_TO_MEAN_RATIO'] = application['EXT_SOURCE_3'] / application['NEW_EXT_SOURCES_MEAN']
    
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'EMERGENCYSTATE_MODE']:
        application[bin_feature], uniques = pd.factorize(application[bin_feature])
        
    application, app_cat_cols = one_hot_encoder(application)
    application, rank_cols, log_cols = amt_rank_log(application)
    
    del rank_cols, log_cols, app_cat_cols
    gc.collect()
    
    train = application[pd.notnull(application['TARGET'])]
    test = application[pd.isnull(application['TARGET'])]
    
    del application 
    gc.collect()
    
    X = train.drop(['TARGET'], axis=1)
    y = train['TARGET']

    X_test = test.drop(['TARGET'], axis=1)
    
    del train, test
    gc.collect()
    
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # del X, y
    # gc.collect()
    
    return X, y, X_test
    print('Application done')
    

def func_previous_applications():
    prev = pd.read_csv('../input/home-credit-default-risk/previous_application.csv').set_index(['SK_ID_PREV'])
    
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
    
    prev['INTEREST_RATE'] = ((prev['AMT_ANNUITY'] * prev['CNT_PAYMENT'] / prev['AMT_CREDIT']) ** (1 / (prev['CNT_PAYMENT'] / 12))) - 1
    
    prev['APP_TO_CREDIT_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    prev['APP_TO_GOODS_PRICE_RATIO'] = prev['AMT_APPLICATION'] / prev['AMT_GOODS_PRICE']
    prev['CREDIT_TO_GOODS_PRICE_RATIO'] = prev['AMT_CREDIT'] / prev['AMT_GOODS_PRICE']
    
    prev, cat_columns = one_hot_encoder(prev)
    prev, rank_columns, log_columns = amt_rank_log(prev)
    
    rel_columns = ['APP_TO_CREDIT_RATIO', 'APP_TO_GOODS_PRICE_RATIO', 'CREDIT_TO_GOODS_PRICE_RATIO', 'INTEREST_RATE']
    abs_columns = [col for col in prev.columns if col not in rel_columns and col not in cat_columns and col != 'SK_ID_CURR' and 'FLAG' not in col and col not in rank_columns and col not in log_columns]
    cat_columns.extend([col for col in prev.columns if 'FLAG' in col])
    
    # Previous applications log features
    log_aggregations = {}
    for col in log_columns:
        log_aggregations[col] = ['min', 'max', 'mean']
    # Previous applications rank features
    rank_aggregations = {}
    for col in rank_columns:
        rank_aggregations[col] = ['sum', 'mean']
    # Previous applications relative features
    rel_aggregations = {} 
    for col in rel_columns:
        rel_aggregations[col] = ['min', 'max', 'mean', 'var']
    # Previous applications absolute features
    abs_aggregations = {} 
    for col in abs_columns:
        abs_aggregations[col] = ['min', 'max', 'mean']
    # Previous applications categorical features
    cat_aggregations = {}
    for col in cat_columns:
        cat_aggregations[col] = ['sum', 'mean']
        
    prev_agg = prev.groupby('SK_ID_CURR').agg({**rel_aggregations, **abs_aggregations, **cat_aggregations, **log_aggregations, **rank_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg({**rel_aggregations, **abs_aggregations, **log_aggregations, **rank_aggregations})
    approved_agg.columns = pd.Index(['PREV_APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left')
    
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg({**rel_aggregations, **abs_aggregations, **log_aggregations, **rank_aggregations})
    refused_agg.columns = pd.Index(['PREV_REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left')

    del refused, refused_agg, approved, approved_agg, prev #revolving, revolving_agg, cash, cash_agg, consumer, consumer_agg, 
    gc.collect()
    
    return prev_agg
    print('Previous_applications done')


def func_bureau():
    bb = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv')
    
    bb, cat_columns = one_hot_encoder(bb)
    
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    bb_cat_aggregations = {}
    for col in cat_columns:
        bb_cat_aggregations[col] = ['mean']
        
    bb_agg = bb.groupby('SK_ID_BUREAU').agg({**bb_aggregations, **bb_cat_aggregations})
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    
    month = bb[bb['MONTHS_BALANCE'] >= -1]
    month_agg = month.groupby('SK_ID_BUREAU').agg(bb_cat_aggregations)
    month_agg.columns = pd.Index(['1_MONTH_' + e[0] + "_" + e[1].upper() for e in month_agg.columns.tolist()])
    
    iii_month = bb[bb['MONTHS_BALANCE'] >= -3]
    iii_month_agg = iii_month.groupby('SK_ID_BUREAU').agg(bb_cat_aggregations)
    iii_month_agg.columns = pd.Index(['3_MONTH_' + e[0] + "_" + e[1].upper() for e in iii_month_agg.columns.tolist()])
    
    year = bb[bb['MONTHS_BALANCE'] >= -12]
    year_agg = year.groupby('SK_ID_BUREAU').agg(bb_cat_aggregations)
    year_agg.columns = pd.Index(['YEAR_' + e[0] + "_" + e[1].upper() for e in year_agg.columns.tolist()])
    
    bb_agg = bb_agg.join(month_agg).join(iii_month_agg).join(year_agg)
    
    del bb, month_agg, iii_month_agg, year_agg
    gc.collect()
    
    bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv').set_index('SK_ID_BUREAU')
    
    bureau['CREDIT_ENDDATE_VS_ENDDATE_FACT'] = bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT_ENDDATE']
    bureau['CREDIT_ENDDATE_TO_ENDDATE_FACT'] = bureau['DAYS_ENDDATE_FACT'] - bureau['DAYS_CREDIT_ENDDATE']
    bureau['AMT_CREDIT_CARD_CREDIT_DEBT_VS_CREDIT_LIMIT'] = bureau['AMT_CREDIT_SUM_DEBT'] - bureau['AMT_CREDIT_SUM_LIMIT']
    bureau['AMT_CREDIT_CARD_CREDIT_DEBT_VS_CREDIT_LIMIT'] = np.where(bureau['CREDIT_TYPE'] == 'Credit card', bureau['AMT_CREDIT_CARD_CREDIT_DEBT_VS_CREDIT_LIMIT'], 0)
    bureau['CREDIT_CARD_CREDIT_DEBT_TO_CREDIT_LIMIT'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM_LIMIT']
    bureau['CREDIT_CARD_CREDIT_DEBT_TO_CREDIT_LIMIT'] = np.where(bureau['CREDIT_TYPE'] == 'Credit card', bureau['AMT_CREDIT_CARD_CREDIT_DEBT_VS_CREDIT_LIMIT'], 0)
    
    bureau = bureau.join(bb_agg, how='left')
    
    bureau, cat_columns = one_hot_encoder(bureau)
    bureau, rank_columns, log_columns = amt_rank_log(bureau)
    
    rel_columns = ['CREDIT_ENDDATE_TO_ENDDATE_FACT', 'CREDIT_CARD_CREDIT_DEBT_TO_CREDIT_LIMIT']
    period_columns = bb_agg.drop(['MONTHS_BALANCE_MAX', 'MONTHS_BALANCE_MIN', 'MONTHS_BALANCE_SIZE'], axis=1).columns
    abs_columns = [col for col in bureau.columns if col not in rel_columns and col not in cat_columns and col != 'SK_ID_CURR' and col not in rank_columns and col not in log_columns and col not in period_columns]
    
    # Previous applications period features
    period_aggregations = {}
    for col in period_columns:
        period_aggregations[col] = ['mean']
    # Previous applications log features
    log_aggregations = {}
    for col in log_columns:
        log_aggregations[col] = ['min', 'max', 'mean']
    # Previous applications log features
    log_aggregations = {}
    for col in log_columns:
        log_aggregations[col] = ['min', 'max', 'mean']
    # Previous applications rank features
    rank_aggregations = {}
    for col in rank_columns:
        rank_aggregations[col] = ['sum', 'mean']
    # Previous applications relative features
    rel_aggregations = {} 
    for col in rel_columns:
        rel_aggregations[col] = ['min', 'max', 'mean', 'var']
    # Previous applications absolute features
    abs_aggregations = {} 
    for col in abs_columns:
        abs_aggregations[col] = ['min', 'max', 'mean']
    # Previous applications categorical features
    cat_aggregations = {}
    for col in cat_columns:
        cat_aggregations[col] = ['sum', 'mean']
        
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**rel_aggregations, **abs_aggregations, **cat_aggregations, **log_aggregations, **rank_aggregations, **period_aggregations})
    bureau_agg.columns = pd.Index(['BUREAU_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg({**rel_aggregations, **abs_aggregations, **log_aggregations, **rank_aggregations, **period_aggregations})
    closed_agg.columns = pd.Index(['BUREAU_CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left')
    
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg({**rel_aggregations, **abs_aggregations, **log_aggregations, **rank_aggregations, **period_aggregations})
    active_agg.columns = pd.Index(['BUREAU_ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left')
    
    del active, active_agg, closed, closed_agg, bureau, bb_agg
    gc.collect()
    
    return bureau_agg
    print('Bureau done')
    

def func_pos_cash_balance():
    pos = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv').sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])
    
    pos, cat_columns = one_hot_encoder(pos)
    
    pos['CATCH_DPD'] = np.where(pos['SK_DPD'] > 0, 1, 0)
    pos['CATCH_DPD_DEF'] = np.where(pos['SK_DPD_DEF'] > 0, 1, 0)
    
    pos_aggregations = {
        'SK_ID_CURR': ['mean'],
        'MONTHS_BALANCE': ['min', 'max'], 
        'CNT_INSTALMENT': ['min', 'max', 'sum', 'mean'],
        'CNT_INSTALMENT_FUTURE': ['min', 'max', 'sum'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean'],
        'CATCH_DPD': ['sum'],
        'CATCH_DPD_DEF': ['sum']
    }
    cat_aggregations = {
        'SK_ID_CURR': ['mean']
    }
    for col in cat_columns:
        cat_aggregations[col] = ['last']
        
    pos_agg_prev = pos.groupby('SK_ID_PREV').agg(pos_aggregations)
    pos_agg_prev.columns = pd.Index([e[0] + "_" + e[1].upper() for e in pos_agg_prev.columns.tolist()])
    pos_agg_prev = pos_agg_prev.rename(columns={'SK_ID_CURR_MEAN':'SK_ID_CURR'})
    pos_agg_1 = pos_agg_prev.groupby('SK_ID_CURR').mean()
    
    pos_agg_prev_cat = pos.groupby('SK_ID_PREV').agg(cat_aggregations)
    pos_agg_prev_cat.columns = pd.Index([e[0] + "_" + e[1].upper() for e in pos_agg_prev_cat.columns.tolist()])
    pos_agg_prev_cat = pos_agg_prev_cat.rename(columns={'SK_ID_CURR_MEAN':'SK_ID_CURR'})
    pos_agg_2 = pos_agg_prev_cat.groupby('SK_ID_CURR').sum()
    
    pos_agg = pos_agg_1.join(pos_agg_2)
    pos_agg.columns = ['POS_' + col for col in pos_agg.columns.tolist()]
    
    del pos_agg_2, pos_agg_1, pos_agg_prev_cat, pos_agg_prev
    gc.collect()
    
    pos_month = pos[pos['MONTHS_BALANCE'] >= -2]
    
    pos_agg_prev = pos_month.groupby('SK_ID_PREV').agg(pos_aggregations)
    pos_agg_prev.columns = pd.Index(['POS_MONTHS_' + e[0] + "_" + e[1].upper() for e in pos_agg_prev.columns.tolist()])
    pos_agg_prev = pos_agg_prev.rename(columns={'POS_MONTHS_SK_ID_CURR_MEAN':'SK_ID_CURR'})
    pos_agg_1 = pos_agg_prev.groupby('SK_ID_CURR').mean()
    
    pos_agg_prev_cat = pos_month.groupby('SK_ID_PREV').agg(cat_aggregations)
    pos_agg_prev_cat.columns = pd.Index(['POS_MONTHS_' + e[0] + "_" + e[1].upper() for e in pos_agg_prev_cat.columns.tolist()])
    pos_agg_prev_cat = pos_agg_prev_cat.rename(columns={'POS_MONTHS_SK_ID_CURR_MEAN':'SK_ID_CURR'})
    pos_agg_2 = pos_agg_prev_cat.groupby('SK_ID_CURR').sum()
    
    pos_agg = pos_agg.join(pos_agg_1.join(pos_agg_2), how='left')
    
    del pos_month, pos_agg_2, pos_agg_1, pos_agg_prev_cat, pos_agg_prev
    gc.collect()
    
    pos_III_month = pos[pos['MONTHS_BALANCE'] >= -4]
    
    pos_agg_prev = pos_III_month.groupby('SK_ID_PREV').agg(pos_aggregations)
    pos_agg_prev.columns = pd.Index(['POS_III_MONTHS_' + e[0] + "_" + e[1].upper() for e in pos_agg_prev.columns.tolist()])
    pos_agg_prev = pos_agg_prev.rename(columns={'POS_III_MONTHS_SK_ID_CURR_MEAN':'SK_ID_CURR'})
    pos_agg_1 = pos_agg_prev.groupby('SK_ID_CURR').mean()
    
    pos_agg_prev_cat = pos_III_month.groupby('SK_ID_PREV').agg(cat_aggregations)
    pos_agg_prev_cat.columns = pd.Index(['POS_III_MONTHS_' + e[0] + "_" + e[1].upper() for e in pos_agg_prev_cat.columns.tolist()])
    pos_agg_prev_cat = pos_agg_prev_cat.rename(columns={'POS_III_MONTHS_SK_ID_CURR_MEAN':'SK_ID_CURR'})
    pos_agg_2 = pos_agg_prev_cat.groupby('SK_ID_CURR').sum()
    
    pos_agg = pos_agg.join(pos_agg_1.join(pos_agg_2), how='left')
    
    del pos_III_month, pos_agg_2, pos_agg_1, pos_agg_prev_cat, pos_agg_prev
    gc.collect()
    
    pos_year = pos[pos['MONTHS_BALANCE'] >= -12]
    
    pos_agg_prev = pos_year.groupby('SK_ID_PREV').agg(pos_aggregations)
    pos_agg_prev.columns = pd.Index(['POS_YEAR_' + e[0] + "_" + e[1].upper() for e in pos_agg_prev.columns.tolist()])
    pos_agg_prev = pos_agg_prev.rename(columns={'POS_YEAR_SK_ID_CURR_MEAN':'SK_ID_CURR'})
    pos_agg_1 = pos_agg_prev.groupby('SK_ID_CURR').mean()
    
    pos_agg_prev_cat = pos_year.groupby('SK_ID_PREV').agg(cat_aggregations)
    pos_agg_prev_cat.columns = pd.Index(['POS_YEAR_' + e[0] + "_" + e[1].upper() for e in pos_agg_prev_cat.columns.tolist()])
    pos_agg_prev_cat = pos_agg_prev_cat.rename(columns={'POS_YEAR_SK_ID_CURR_MEAN':'SK_ID_CURR'})
    pos_agg_2 = pos_agg_prev_cat.groupby('SK_ID_CURR').sum()
    
    pos_agg = pos_agg.join(pos_agg_1.join(pos_agg_2), how='left')
    
    del pos_year, pos_agg_2, pos_agg_1, pos_agg_prev_cat, pos_agg_prev, pos
    gc.collect()
    
    return pos_agg
    print('Pos_cash_balance done')
    

def func_credit_card_balance():
    cc = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv').sort_values(by=['SK_ID_PREV', 'MONTHS_BALANCE'])
    
    cc['PAYMENT_CURRENT_LOSS'] = np.where(cc['AMT_PAYMENT_CURRENT'] < cc['AMT_INST_MIN_REGULARITY'], 1, 0)
    cc['PAYMENT_TOTAL_CURRENT_LOSS'] = np.where(cc['AMT_PAYMENT_TOTAL_CURRENT'] < cc['AMT_INST_MIN_REGULARITY'], 1, 0)
    cc['AMT_PAYMENT_CURRENT_LOSS'] = np.where(cc['AMT_PAYMENT_CURRENT'] < cc['AMT_INST_MIN_REGULARITY'], cc['AMT_INST_MIN_REGULARITY'] - cc['AMT_PAYMENT_CURRENT'], 0)
    cc['AMT_PAYMENT_TOTAL_CURRENT_LOSS'] = np.where(cc['AMT_PAYMENT_TOTAL_CURRENT'] < cc['AMT_INST_MIN_REGULARITY'], cc['AMT_INST_MIN_REGULARITY'] - cc['AMT_PAYMENT_TOTAL_CURRENT'], 0)
    
    ratio_balance_list = []
    ratio_credit_limit_list = []
    
    grouped = cc.groupby(by='SK_ID_PREV')
    
    for sk_id_prev in cc['SK_ID_PREV'].unique():
        group = grouped.get_group(sk_id_prev)
        
        if group.iloc[0, :]['AMT_BALANCE'] != 0:
            ratio_balance = group.iloc[-1, :]['AMT_BALANCE'] / group.iloc[0, :]['AMT_BALANCE']
        else:
            ratio_balance = 0
            
        if group.iloc[0, :]['AMT_CREDIT_LIMIT_ACTUAL'] != 0:
            ratio_credit_limit = group.iloc[-1, :]['AMT_CREDIT_LIMIT_ACTUAL'] / group.iloc[0, :]['AMT_CREDIT_LIMIT_ACTUAL']
        else:
            ratio_credit_limit = 0
        
        ratio_balance_list.extend(np.full(group.shape[0], ratio_balance).tolist())
        ratio_credit_limit_list.extend(np.full(group.shape[0], ratio_credit_limit).tolist())
        
    del grouped, group
    gc.collect()
    
    new_cc = cc
    
    new_cc['RATIO_BALANCE'] = ratio_balance_list
    new_cc['RATIO_CREDIT_LIMIT'] = ratio_credit_limit_list
    
    del ratio_balance_list, ratio_credit_limit_list
    gc.collect()
    
    new_cc, cat_columns = one_hot_encoder(new_cc)
    
    rel_columns = ['RATIO_BALANCE', 'RATIO_CREDIT_LIMIT']
    abs_columns = [col for col in new_cc.columns if col not in rel_columns and col not in cat_columns and col != 'SK_ID_CURR' and col != 'SK_ID_PREV' and col != 'MONTHS_BALANCE']
    
    cat_aggregations = {}
    for col in cat_columns:
        cat_aggregations[col] = ['last']
        
    cc_aggregations = {} 
    for col in rel_columns:
        cc_aggregations[col] = ['mean']
        
    for col in abs_columns:
        cc_aggregations[col] = ['min', 'max', 'mean']
    
    cc_aggregations['SK_ID_CURR'] = ['mean']
    cat_aggregations['SK_ID_CURR'] = ['mean']
    cc_aggregations['MONTHS_BALANCE'] = ['min', 'max', 'size']
    
    cc_agg_prev = new_cc.groupby('SK_ID_PREV').agg(cc_aggregations)
    cc_agg_prev.columns = pd.Index([e[0] + "_" + e[1].upper() for e in cc_agg_prev.columns.tolist()])
    cc_agg_prev = cc_agg_prev.rename(columns={'SK_ID_CURR_MEAN': 'SK_ID_CURR'})
    cc_agg_1 = cc_agg_prev.groupby('SK_ID_CURR').mean()
    
    cc_agg_prev_cat = new_cc.groupby('SK_ID_PREV').agg(cat_aggregations)
    cc_agg_prev_cat.columns = pd.Index([e[0] + "_" + e[1].upper() for e in cc_agg_prev_cat.columns.tolist()])
    cc_agg_prev_cat = cc_agg_prev_cat.rename(columns={'SK_ID_CURR_MEAN': 'SK_ID_CURR'})
    cc_agg_2 = cc_agg_prev_cat.groupby('SK_ID_CURR').sum()
    
    cc_agg = cc_agg_1.join(cc_agg_2)
    cc_agg.columns = ['CC_' + col for col in cc_agg.columns.tolist()]
    
    del cc_agg_1, cc_agg_2, cc_agg_prev, cc_agg_prev_cat
    gc.collect()
    
    ratio_balance_list_m = []
    ratio_credit_limit_list_m = []
    
    grouped_m = cc[cc['MONTHS_BALANCE'] >= -3].groupby(by='SK_ID_PREV')
    
    for sk_id_prev in cc[cc['MONTHS_BALANCE'] >= -3]['SK_ID_PREV'].unique():
        group = grouped_m.get_group(sk_id_prev)
        
        if group.iloc[0, :]['AMT_BALANCE'] != 0:
            ratio_balance = group.iloc[-1, :]['AMT_BALANCE'] / group.iloc[0, :]['AMT_BALANCE']
        else:
            ratio_balance = 0
            
        if group.iloc[0, :]['AMT_CREDIT_LIMIT_ACTUAL'] != 0:
            ratio_credit_limit = group.iloc[-1, :]['AMT_CREDIT_LIMIT_ACTUAL'] / group.iloc[0, :]['AMT_CREDIT_LIMIT_ACTUAL']
        else:
            ratio_credit_limit = 0
        
        ratio_balance_list_m.extend(np.full(group.shape[0], ratio_balance).tolist())
        ratio_credit_limit_list_m.extend(np.full(group.shape[0], ratio_credit_limit).tolist())
        
    del grouped_m, group
    gc.collect()
    
    new_cc_m = cc[cc['MONTHS_BALANCE'] >= -3]
    
    new_cc_m['RATIO_BALANCE_M'] = ratio_balance_list_m
    new_cc_m['RATIO_CREDIT_LIMIT_M'] = ratio_credit_limit_list_m
    
    del ratio_balance_list_m, ratio_credit_limit_list_m
    gc.collect()
    
    new_cc_m, cat_columns = one_hot_encoder(new_cc_m)
    
    cat_aggregations = {}
    for col in cat_columns:
        cat_aggregations[col] = ['last']
    cat_aggregations['SK_ID_CURR'] = ['mean']
    
    cc_agg_prev_m = new_cc_m.groupby('SK_ID_PREV').agg(cc_aggregations)
    cc_agg_prev_m.columns = pd.Index(['CC_MONTHS_' + e[0] + "_" + e[1].upper() for e in cc_agg_prev_m.columns.tolist()])
    cc_agg_prev_m = cc_agg_prev_m.rename(columns={'CC_MONTHS_SK_ID_CURR_MEAN': 'SK_ID_CURR'})
    cc_agg_1 = cc_agg_prev_m.groupby('SK_ID_CURR').mean()
    
    cc_agg_prev_cat_m = new_cc_m.groupby('SK_ID_PREV').agg(cat_aggregations)
    cc_agg_prev_cat_m.columns = pd.Index(['CC_MONTHS_' + e[0] + "_" + e[1].upper() for e in cc_agg_prev_cat_m.columns.tolist()])
    cc_agg_prev_cat_m = cc_agg_prev_cat_m.rename(columns={'CC_MONTHS_SK_ID_CURR_MEAN': 'SK_ID_CURR'})
    cc_agg_2 = cc_agg_prev_cat_m.groupby('SK_ID_CURR').sum()
    
    cc_agg = cc_agg.join(cc_agg_1.join(cc_agg_2), how='left')
    
    del cc_agg_1, cc_agg_2, cc_agg_prev_m, cc_agg_prev_cat_m
    gc.collect()
    
    ratio_balance_list_y = []
    ratio_credit_limit_list_y = []
    
    grouped_y = cc[cc['MONTHS_BALANCE'] >= -12].groupby(by='SK_ID_PREV')
    
    for sk_id_prev in cc[cc['MONTHS_BALANCE'] >= -12]['SK_ID_PREV'].unique():
        group = grouped_y.get_group(sk_id_prev)
        
        if group.iloc[0, :]['AMT_BALANCE'] != 0:
            ratio_balance = group.iloc[-1, :]['AMT_BALANCE'] / group.iloc[0, :]['AMT_BALANCE']
        else:
            ratio_balance = 0
            
        if group.iloc[0, :]['AMT_CREDIT_LIMIT_ACTUAL'] != 0:
            ratio_credit_limit = group.iloc[-1, :]['AMT_CREDIT_LIMIT_ACTUAL'] / group.iloc[0, :]['AMT_CREDIT_LIMIT_ACTUAL']
        else:
            ratio_credit_limit = 0
        
        ratio_balance_list_y.extend(np.full(group.shape[0], ratio_balance).tolist())
        ratio_credit_limit_list_y.extend(np.full(group.shape[0], ratio_credit_limit).tolist())
        
    del grouped_y, group
    gc.collect()
    
    new_cc_y = cc[cc['MONTHS_BALANCE'] >= -12]
    
    new_cc_y['RATIO_BALANCE_Y'] = ratio_balance_list_y
    new_cc_y['RATIO_CREDIT_LIMIT_Y'] = ratio_credit_limit_list_y
    
    del ratio_balance_list_y, ratio_credit_limit_list_y
    gc.collect()
    
    new_cc_y, cat_columns = one_hot_encoder(new_cc_y)
    
    cat_aggregations = {}
    for col in cat_columns:
        cat_aggregations[col] = ['last']
    cat_aggregations['SK_ID_CURR'] = ['mean']
    
    cc_agg_prev_y = new_cc_y.groupby('SK_ID_PREV').agg(cc_aggregations)
    cc_agg_prev_y.columns = pd.Index(['CC_YEAR_' + e[0] + "_" + e[1].upper() for e in cc_agg_prev_y.columns.tolist()])
    cc_agg_prev_y = cc_agg_prev_y.rename(columns={'CC_YEAR_SK_ID_CURR_MEAN': 'SK_ID_CURR'})
    cc_agg_1 = cc_agg_prev_y.groupby('SK_ID_CURR').mean()
    
    cc_agg_prev_cat_y = new_cc_y.groupby('SK_ID_PREV').agg(cat_aggregations)
    cc_agg_prev_cat_y.columns = pd.Index(['CC_YEAR_' + e[0] + "_" + e[1].upper() for e in cc_agg_prev_cat_y.columns.tolist()])
    cc_agg_prev_cat_y = cc_agg_prev_cat_y.rename(columns={'CC_YEAR_SK_ID_CURR_MEAN': 'SK_ID_CURR'})
    cc_agg_2 = cc_agg_prev_cat_y.groupby('SK_ID_CURR').sum()
    
    cc_agg = cc_agg.join(cc_agg_1.join(cc_agg_2), how='left')
    
    del cc_agg_1, cc_agg_2, cc_agg_prev_y, cc_agg_prev_cat_y
    gc.collect()
    
    return cc_agg
    print('Credit_card_balance done')


def func_installments_payments():
    ins = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv')
    
    ins, cat_columns = one_hot_encoder(ins)
    
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    
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
    for cat in cat_columns:
        aggregations[cat] = ['mean']
        
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    
    del ins
    gc.collect()
    
    return ins_agg
    print('Installments_payments done')
    

def join_tables(X_train, X_test, df):
    X_train = X_train.join(df, how='left')
    X_test = X_test.join(df, how='left')
    
    del df
    gc.collect()
    
    return X_train, X_test


def min_max_scale(train, test, columns):

    for col in columns:
        scaler = MinMaxScaler()

        null_index_train = train[col].isnull()
        null_index_test = test[col].isnull()

        if train.loc[~null_index_train, [col]].shape[0] == 0 or test.loc[~null_index_test, [col]].shape[0] == 0:
            train = train.drop(col, axis=1)
            test = test.drop(col, axis=1)
            print(col + ' dropped')
            continue

        scaler.fit(train.loc[~null_index_train, [col]])

        train.loc[~null_index_train, [col]] = scaler.transform(train.loc[~null_index_train, [col]])
        test.loc[~null_index_test, [col]] = scaler.transform(test.loc[~null_index_test, [col]])

        train[col].fillna(0, inplace=True)
        test[col].fillna(0, inplace=True)

        print(col + ' scaled')

    print('Scaling done')
    return train, test


def lr_maximize(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=101)

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_train = X_train.fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.fillna(0)

    def lr_evaluate(
            C,
            intercept_scaling,
            solver,
            max_iter
    ):
        score_list = []

        solver_dict = {
            1: ['newton-cg', 'l2'],
            2: ['lbfgs', 'l2'],
            3: ['liblinear', 'l1'],
            4: ['sag', 'l2'],
            5: ['saga', 'l1']
        }

        clf = LogisticRegression(
            penalty=solver_dict[int(round(solver))][1],
            C=C,
            intercept_scaling=intercept_scaling,
            solver = solver_dict[int(round(solver))][0],
            max_iter=int(max_iter),
            n_jobs=-1
        )

        clf.fit(X_train, y_train)

        score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        score_list.append(score)

        return np.mean(score_list)

    xgb_bo = BayesianOptimization(lr_evaluate, {
        'C': (0.5, 1.5),
        'intercept_scaling': (0.5, 1.5),
        'solver': (1, 5),
        'max_iter': (100, 1000),
    })

    xgb_bo.maximize(init_points=5, n_iter=40)


def dnn_maximize(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    float_cols = X.select_dtypes(include='float').columns

    X_train, X_test = min_max_scale(X_train, X_test, float_cols)
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    print('NaN filled')

    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    print('Dummies done')

    X_train = np.array(X_train).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    print('Converted to float32')

    def dnn_evaluate(batch_size, hls, learning_rate):

        hls = int(round(hls))

        tf_train_dataset = tf.placeholder(np.float32, shape=(int(batch_size), X_train.shape[1]))
        tf_train_labels = tf.placeholder(np.float32, shape=(int(batch_size), y_train.shape[1]))
        tf_test_dataset = tf.constant(X_test)

        if hls == 0:

            n_nodes_il = int((X_train.shape[1] - y_train.shape[1]) / 2)
            n_nodes_ol = y_train.shape[1]

            input_layer = {'weights': tf.Variable(tf.truncated_normal([X_train.shape[1], n_nodes_il])),
                           'biases': tf.Variable(tf.zeros([n_nodes_il]))}

            output_layer = {'weights': tf.Variable(tf.truncated_normal([n_nodes_il, n_nodes_ol])),
                            'biases': tf.Variable(tf.zeros([n_nodes_ol]))}

            il = tf.add(tf.matmul(tf_train_dataset, input_layer['weights']), input_layer['biases'])
            il = tf.nn.sigmoid(il)

            ol = tf.add(tf.matmul(il, output_layer['weights']), output_layer['biases'])

            il_test = tf.add(tf.matmul(tf_test_dataset, input_layer['weights']), input_layer['biases'])
            il_test = tf.nn.sigmoid(il_test)

            ol_test = tf.add(tf.matmul(il_test, output_layer['weights']), output_layer['biases'])

        else:

            nodes_in_layers = np.linspace(X_train.shape[1], y_train.shape[1], hls + 3).astype(int)

            layers = []

            for i in range(1, hls + 3):
                layer = {'weights': tf.Variable(tf.truncated_normal([nodes_in_layers[i - 1], nodes_in_layers[i]])),
                         'biases': tf.Variable(tf.zeros([nodes_in_layers[i]]))}
                layers.append(layer)

            activation_functions_train = []

            il = tf.nn.sigmoid(tf.add(tf.matmul(tf_train_dataset, layers[0]['weights']), layers[0]['biases']))
            activation_functions_train.append(il)

            for i in range(1, hls + 1):
                hl = tf.nn.sigmoid(
                    tf.add(tf.matmul(activation_functions_train[i - 1], layers[i]['weights']), layers[i]['biases']))
                activation_functions_train.append(hl)

            ol = tf.add(tf.matmul(activation_functions_train[-1], layers[-1]['weights']), layers[-1]['biases'])

            activation_functions_test = []

            il = tf.nn.sigmoid(tf.add(tf.matmul(tf_test_dataset, layers[0]['weights']), layers[0]['biases']))
            activation_functions_test.append(il)

            for i in range(1, hls + 1):
                hl = tf.nn.sigmoid(
                    tf.add(tf.matmul(activation_functions_test[i - 1], layers[i]['weights']), layers[i]['biases']))
                activation_functions_test.append(hl)

            ol_test = tf.add(tf.matmul(activation_functions_test[-1], layers[-1]['weights']), layers[-1]['biases'])

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=ol)
        )

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        train_prediction = tf.nn.softmax(ol)
        test_prediction = tf.nn.softmax(ol_test)

        num_steps = 15000

        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 44
        config.inter_op_parallelism_threads = 44

        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()
            print('Initialized')

            roc_auc = None
            roc_auc_step = None

            for step in range(num_steps):
                offset = (step * int(batch_size)) % (y_train.shape[0] - int(batch_size))

                batch_data = X_train[offset:(offset + int(batch_size)), :]
                batch_labels = y_train[offset:(offset + int(batch_size)), :]

                feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}

                _, l, predictions_train, predictions_test = sess.run([optimizer, loss, train_prediction, test_prediction], feed_dict=feed_dict)

                if roc_auc is None:
                    roc_auc = roc_auc_score(y_test[:, -1], predictions_test[:, -1])

                if roc_auc_step is None:
                    roc_auc_step = step

                if roc_auc_score(y_test[:, -1], predictions_test[:, -1]) > roc_auc:
                    roc_auc = roc_auc_score(y_test[:, -1], predictions_test[:, -1])
                    roc_auc_step = step

                if step - roc_auc_step > 2500:
                    break

                if step % 10 == 0:
                    print("Minibatch loss at step %d: %f" % (step, l))
                    print('Minibatch ROC AUC: ' + '{0:.4f}'.format(roc_auc_score(batch_labels[:, -1],
                                                                                 predictions_train[:, -1])))
                    # valid_predictions = valid_prediction.eval(feed_dict={})[:, -1]

                    print('Validation ROC AUC: ' + '{0:.4f}'.format(roc_auc_score(y_test[:, -1],
                                                                                  predictions_test[:, -1])))
            # print('ROC AUC: ' + '{0:.4f}'.format(roc_auc_score(y_test[:, -1], test_prediction.eval()[:, -1])))

            return roc_auc

    xgbBO = BayesianOptimization(dnn_evaluate, {
        'batch_size': (10000, 40000),
        'hls': (1, 5),
        'learning_rate': (0.0005, 0.001)
    })

    xgbBO.maximize(init_points=5, n_iter=40)


def xgb_maximize(X, y):

    def sk_xgb_evaluate(
            booster,
            num_leaves,
            max_depth,
            learning_rate,
            min_child_weight,
            max_delta_step,
            colsample_bytree,
            colsample_bylevel,
            subsample,
            reg_alpha,
            reg_lambda
    ):
        score_list = []

        # for i in [101, 202, 303, 404]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.fillna(-99999)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.fillna(-99999)

        boster_dict = {0: 'gbtree', 1: 'gblinear', 2: 'dart'}
        # ratio = y_train[y_train == 0].shape[0] / y_train[y_train == 1].shape[0]
        clf = XGBClassifier(objective='binary:logistic',
                            n_jobs=16,
                            seed=123,
                            booster=boster_dict[int(round(booster))],
                            num_leaves=int(num_leaves),
                            n_estimators=10000,
                            max_depth=int(max_depth),
                            learning_rate=learning_rate,
                            min_child_weight=min_child_weight,
                            max_delta_step=max_delta_step,
                            colsample_bytree=colsample_bytree,
                            colsample_bylevel=colsample_bylevel,
                            subsample=subsample,
                            reg_alpha=reg_alpha,
                            reg_lambda=reg_lambda #, scale_pos_weight=ratio
                            )

        clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric='auc',
                early_stopping_rounds=200, verbose=1000)

        score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        score_list.append(score)

        return roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

    xgbBO = BayesianOptimization(sk_xgb_evaluate, {
        'booster': (0, 2),
        'num_leaves': (10, 50),
        'max_depth': (10, 50),
        'learning_rate': (0.01, 0.03),
        'min_child_weight': (1, 90),
        'max_delta_step': (0.01, 0.15),
        'colsample_bytree': (0.1, 1),
        'colsample_bylevel': (0.1, 1),
        'subsample': (0.1, 0.99),
        'reg_alpha': (0.01, 1),
        'reg_lambda': (0.01, 1),
    })

    xgbBO.maximize(init_points=5, n_iter=40)


def lgb_maximize(train_X, train_y,
                 valid_X, valid_y,
                 test_X, test_y):
    
    def sk_lgb_evaluate(num_leaves,
                        max_depth,
                        learning_rate,
                        min_child_weight,
                        colsample_bytree,
                        subsample,
                        reg_alpha,
                        reg_lambda):

        # ratio = y_train[y_train == 0].shape[0] / y_train[y_train == 1].shape[0]
        clf = LGBMClassifier(objective='binary',
                             silent=1,
                             nthread=16,
                             seed=123,
                             num_leaves=int(num_leaves),
                             n_estimators=10000,
                             max_depth=int(max_depth),
                             learning_rate=learning_rate,
                             min_child_weight=min_child_weight,
                             colsample_bytree=colsample_bytree,
                             subsample=subsample,
                             reg_alpha=reg_alpha,
                             reg_lambda=reg_lambda,
                            #  scale_pos_weight=ratio
                             )

        clf.fit(train_X, train_y, eval_set=[(train_X, train_y), (valid_X, valid_y)], eval_metric='auc',
                early_stopping_rounds=100, verbose=500)

        score = roc_auc_score(test_y, clf.predict_proba(test_X)[:, -1])

        return score

    xgbBO = BayesianOptimization(sk_lgb_evaluate, {
        'num_leaves': (10, 50),
        'max_depth': (10, 50),
        'learning_rate': (0.01, 0.03),
        'min_child_weight': (1, 90),
        'colsample_bytree': (0.1, 1),
        'subsample': (0.1, 0.99),
        'reg_alpha': (0.01, 1),
        'reg_lambda': (0.01, 1),
    })

    xgbBO.maximize(init_points=1, n_iter=1)


def rfc_maximize(X, y):

    def rfc_evaluate(
            min_samples_split,
            min_samples_leaf,
            min_weight_fraction_leaf,
            min_impurity_decrease
    ):
        score_list = []

        for i in [101, 202, 303, 404]:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i)
            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            X_train = X_train.fillna(-9999)
            X_test = X_test.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.fillna(-9999)

            # weights = y_train.shape[0] / (2 * np.bincount(y_train))
            weight_class_zero = y_train.shape[0] / (2 * y_train[y_train == 0].shape[0])
            weight_class_one = y_train.shape[0] / (2 * y_train[y_train == 1].shape[0])

            clf = RandomForestClassifier(
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                min_impurity_decrease=min_impurity_decrease,
                # class_weight={0: weight_class_zero, 1: weight_class_one},
                n_jobs=-1
            )

            clf.fit(X_train, y_train)

            score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
            score_list.append(score)

        return np.mean(score_list)

    xgb_bo = BayesianOptimization(rfc_evaluate, {
        'min_samples_split': (0.0001, 0.3),
        'min_samples_leaf': (0.0001, 0.3),
        'min_weight_fraction_leaf': (0., 0.3),
        'min_impurity_decrease': (0., 0.001),
    })

    xgb_bo.maximize(init_points=5, n_iter=40)


def main():

    # X, y, test = func_application()
    # print('Application done')

    # previous_applications = func_previous_applications()
    # X = X.join(previous_applications, how='left')
    # test = test.join(previous_applications, how='left')
    # del previous_applications
    # print('Previous_applications done')

    # bureau = func_bureau()
    # X = X.join(bureau, how='left')
    # test = test.join(bureau, how='left')
    # del bureau
    # print('Bureau done')

    # pos_cash_balance = func_pos_cash_balance()
    # X = X.join(pos_cash_balance, how='left')
    # test = test.join(pos_cash_balance, how='left')
    # del pos_cash_balance
    # print('Pos_cash_balance done')

    # credit_card_balance = func_credit_card_balance()
    # X = X.join(credit_card_balance, how='left')
    # test = test.join(credit_card_balance, how='left')
    # del credit_card_balance
    # print('Credit_card_balance done')

    # installments_payments = func_installments_payments()
    # X = X.join(installments_payments, how='left')
    # test = test.join(installments_payments, how='left')
    # del installments_payments
    # print('Installments_payments done')
    
    # gc.collect()
    
    X = pd.read_csv('../input/my-dataset/X.csv').set_index('SK_ID_CURR')
    y = pd.read_csv('../input/my-dataset/y.csv').set_index('SK_ID_CURR')
    # test = pd.read('../input/my-dataset/test.csv')
    imp = pd.read_csv('../input/my-dataset/data_importances.csv')
    cols = imp.loc[:800, 'Feature'].values
    
    del imp
    gc.collect()
    
    X = X[cols]
    
    del cols
    gc.collect()
    
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    del X, y
    gc.collect()
    
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.33, random_state=101)
    
    del X_train_valid, y_train_valid
    gc.collect()
    
    print('Data created')
    
    lgb_maximize(X_train, y_train,
                 X_valid, y_valid,
                 X_test, y_test)


if __name__ == "__main__":
    main()