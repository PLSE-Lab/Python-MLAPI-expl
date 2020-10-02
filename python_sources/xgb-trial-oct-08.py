#keras for house sold price forcast
"""
Created on Sat Oct  1 00:38:55 2016

@author: Chia-Ta
https://www.kaggle.com/cttsai/


"""

import itertools
import time
from random import randint, randrange
import datetime as dt
import pandas as pd
import numpy as np
from math import expm1, log1p, log10, log2, sqrt, ceil
from sklearn.cluster import KMeans, AffinityPropagation
#from sklearn.cross_validation import StratifiedKFold, LabelKFold, ShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder

#import matplotlib.pyplot as plt

#keras
import xgboost as xgb


seed = 101472016
np.random.seed(seed)
t_digit = 1
scr_digit = 2
#hash_digit = (2 ** 5) * (2 ** 10) #32*1024


#useful variables
target, target_id = 'SalePrice', 'Id'
    
###############################################################################
#XGB
###############################################################################
def evaluation(train, test, params, xgb_param):

    filestem = '_{}_md{}_eta{:.3f}'.format(
        xgb_param['eval_metric'], xgb_param['max_depth'], xgb_param['eta'])    
    
    
    #prepare data for learning
    test_pred = None
    collect_test_pred = pd.DataFrame()
    #scaling y
    real_y = train[target]
    train_scl_y, scl_low, scl_range = rescale_column(train[target])
    #train_scl_y = real_y.apply(log1p)
    #covert to np array as input
    #X_id = train[target_id]
    mask = [target_id, target]
    X = train.drop(mask, axis=1)#.as_matrix()
    y = train_scl_y#.as_matrix()
        
    test_id = test[target_id]
    X_t = test.drop(mask, axis=1)
    dtest = xgb.DMatrix(X_t)     
    del train, test

#Block of Model learning begin#################################################
    nr_round, nr_min_round = params['nr_round'], params['nr_min_round']
    nr_run, va_ratio = params['nr_SS'], params['SS_va_ratio']    

    count, score = 0, 0.0
    
    score_in_iters = []
    
    ss = ShuffleSplit(n_splits=nr_run, test_size=va_ratio, train_size=1-va_ratio, random_state=0)
    #np.random.rand
    for ind_tr, ind_va in ss.split(X):
        
        iter_start_ckp = time.time()
        
        #split
        y_train, y_valid = y[ind_tr], y[ind_va]
        
        X_train = X.iloc[ind_tr]
        X_valid = X.iloc[ind_va]
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid) 
        
        watchlist  = [(dtrain, 'train'), (dvalid, 'eval')]
        gbm = xgb.train(xgb_param, dtrain, nr_round, 
                        evals=watchlist, 
                        early_stopping_rounds=nr_min_round)

        
#Block of Model learning End###################################################        
        count += 1
        
        #validation in real
        y_valid_pred = gbm.predict(dvalid).flatten()
        y_valid_pred = scaling_column(pd.Series(y_valid_pred), scl_low, scl_range)
        #y_valid_pred = pd.Series(y_valid_pred).apply(expm1)
        
        y_valid_real = real_y.iloc[ind_va]

        mae = mean_absolute_error(y_valid_real.tolist(), y_valid_pred)
        rmse = sqrt(mean_squared_error(y_valid_real.tolist(), y_valid_pred))
        r2 = r2_score(y_valid_real.tolist(), y_valid_pred)
            
#        score = r2
        score_in_iters.append(rmse)            
            
        print('\niter {:02d} validation real scale: rmse={:.2f}, mae={:.2f}, r2={:.3f}, in {} s\n'.format(
        count, rmse, mae, r2, round(time.time() - iter_start_ckp, t_digit)))
        
        if params['test_mode']:
            if r2 >= 0.9:
            #only model with good quality to predict test
                #stem = 'test' + filestem + '_iter_{:02d}_r2_{:.3f}'.format(count, score)
                stem = 'test' + '_iter_{:02d}_r2_{:.3f}'.format(count, score)
                test_pred = gbm.predict(dtest).flatten()
                test_pred = scaling_column(pd.Series(test_pred), scl_low, scl_range)
    #            #test_pred = pd.Series(test_pred).apply(expm1)
    #            create_submission(test_id, test_pred, prefix=stem, digit=-2)
                collect_test_pred[stem] = pd.Series(test_pred)
        #break

    if params['test_mode']:
        test_pred = collect_test_pred.mean(axis=1)

    top_k = int(ceil(params['nr_SS'] / 4))

    score_in_iters = sorted(score_in_iters)
    for i in range(top_k):
        score += score_in_iters[i]

    return test_pred, (score/top_k)


###############################################################################
#analytic
    
###############################################################################
#file IO
###############################################################################
def load_data(filename):
    
    print('\nLoad ' + filename)
    
    start_time = time.time()

    data = pd.read_csv(filename, dtype={target:float, target_id: int})

    print('Load in {} samples: {:.2f} minutes'.format(len(data), (time.time() - start_time)/60))

    return data


def create_submission(output_id, output_val, prefix = '', digit = 6):
    now = dt.datetime.now()
    filename = 'submit_' + prefix + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv.gz'
    print('Make submission:{}\n'.format(filename))    

    output_val = output_val.apply(lambda x: round(x, digit))
    submission = pd.DataFrame(data={target_id: output_id, target: output_val})
    submission.to_csv(filename, index=False, header=True, compression='gzip')
    
    return 


###############################################################################
#data processing
###############################################################################
def rescale_column(data):
    #sliced dataframe == series, axis=0:'columns' to each row
    data = data.apply(log1p)
    scl_low, scl_range = data.min(), (data.max() - data.min())
    data = data.apply(lambda scl: (scl - scl_low)/scl_range)#scale to 0~1
    return data, scl_low, scl_range


def scaling_column(data, scl_low=0, scl_range=0):#input pd.Series
    data = data.apply(lambda scl: scl * scl_range + scl_low).apply(expm1)#back
    return data#, scl_low, scl_range


def mask_columns(columns = {}, mask = {}):
    for str1 in mask:
        #if columns.count(str1) > 0:
        if str1 in columns:
            columns.remove(str1)
    return columns


def merge_columns(df, columns=[], str_name='', drop=False):      
        
    df[str_name] = ''
    for i in columns:
        df[str_name] += ':' + df[i].astype(str)
        
    if drop == True:
            df.drop(columns, axis=1, inplace=True)
            
    return df


def process_f_cat(df, mask = [], excl_prefix='', minor_cutoff=0):
    
    start_time = time.time()
        
    columns = mask_columns(df.columns.tolist(), mask)
    list_onehot = []
    #encode
    count = 0
    for c in columns:
        if str(c).startswith(excl_prefix) == False:
            
            list_onehot.append(c)
            
            df[c] = df[c].fillna('na').astype(str)

            if minor_cutoff > 0:
                freq = df[c].value_counts(dropna=False)        
                minor = freq.loc[lambda x : x < minor_cutoff].index.tolist()
                if len(minor) > 2:
                    df[c].replace(to_replace=minor, value='minor', inplace=True)

            df[c] = LabelEncoder().fit_transform(df[c].values)
            count += 1
    
    df = pd.get_dummies(df, columns=list_onehot)
    
    print('\nOne-hot encoding {} features on {} samples'.format(len(list_onehot), len(df)))    
    print('Encoded samples: {:.2f} minutes\n'.format((time.time() - start_time)/60))

    return df


def process_f_cluster(train, test, columns=[], nr_clusters= 2 ** 4, name = ''):
    
    #cluster_algo = AffinityPropagation()
    cluster_algo = KMeans(n_clusters = nr_clusters)
    clustered_train = cluster_algo.fit_predict(train[columns].as_matrix())
    clustered_test = cluster_algo.predict(test[columns].as_matrix())
            
    clustered_train = pd.DataFrame({name: clustered_train})
    #train[cluster_id] = clustered_train[cluster_id]
    train.loc[:, name] = clustered_train.loc[:, name]
    
    clustered_test = pd.DataFrame({name: clustered_test})
    #test[cluster_id] = clustered_test[cluster_id]
    test.loc[:, name] = clustered_test.loc[:, name]

    df = pd.concat([train, test])
    
    return df
    
    
def process_feature(train, test, params, mask=[]):
    
    num_prefix = params['dense_prefix']
    
    #before start
    df = pd.concat([train, test])
    df.fillna(0, inplace=True)
        
    #clustering
    #set of features
    fLivArea = ['GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'MasVnrArea', 'BsmtFinSF1', 'LotArea', '2ndFlrSF', 'BsmtUnfSF']
    fYear = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
    fCount = ['OverallQual', 'GarageCars', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'HalfBath', 'BsmtFullBath', 'BedroomAbvGr', 'KitchenAbvGr']
    fPoarch = ['3SsnPorch', 'EnclosedPorch', 'OpenPorchSF', 'ScreenPorch', 'WoodDeckSF']

    if params['cluster_num']:        
        
        nr_clusters = 2 ** 5
        start_time = time.time()        
        print('Using {} clusterd groups as features:'.format(nr_clusters))
        
        cluster_f = [fLivArea, fYear, fCount, fPoarch]
        count = 0
        
        for c in cluster_f:
            count += 1
            df = process_f_cluster(df[:len(train)], df[len(train):], columns=c, 
                                      nr_clusters= nr_clusters, name='cluster_{:02d}'.format(count))
                                    
        print('Clustered {} subsets of features: {}, {:.2f} minutes\n'.format(
                len(cluster_f), cluster_f, (time.time() - start_time)/60))


        #year
    #newly added, #detailed date and year duration
    YrDU = ['RAYrDU', 'BltSoYrDU', 'RASoYrDU', 'GSoYrDU']

    df['YMSold'] = df['YrSold'].astype(str) + '-' + df['MoSold'].astype(str)
    #Year diff
    df['RAYrDU'] = df['YearRemodAdd'] - df['YearBuilt']
    df['BltSoYrDU'] = df['YrSold'] - df['YearBuilt']    
    df['RASoYrDU'] = df['YrSold'] - df['YearRemodAdd']
    df['GSoYrDU'] = df['YrSold'] - df['GarageYrBlt']#Nan need handel(-1)

    for c in YrDU:
        df.ix[df[c] < 0, c] = -1.0 #NA handle; (YrDu should always >= -1)
        df[c] = df[c].apply(lambda x: (x - df[c].min()) / (df[c].max() - df[c].min()))
    #year feature


    #numeric
    GrSF = ['1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'LotArea', 'GarageArea']
    BsmtSF = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
    PoarchSF = ['3SsnPorch', 'EnclosedPorch', 'OpenPorchSF', 'ScreenPorch', 'WoodDeckSF']
    Others = ['LotFrontage', 'MasVnrArea', 'PoolArea', 'MiscVal']
    
    numeric_f = []
    numeric_f.extend(GrSF + BsmtSF + PoarchSF + Others)
    start_time = time.time()

    for c in numeric_f:
        #print(df[c].value_counts(dropna=False))
        df[c].fillna(0, inplace=True) #0 has more sense than NA
        df[c] = df[c].apply(log1p)
        df[c] = df[c].apply(lambda x: (x - df[c].median()) / (df[c].max() - df[c].min()))        

    #rename
    numeric_f.extend(YrDU)
    for c in numeric_f:
        df.rename(columns={c: num_prefix + c}, inplace=True)#last in the loop, otherwise
    
    print('Process {} numeric features of {} samples in {:.2f} minutes\n'.format(
            len(numeric_f), len(df), (time.time() - start_time)/60))
    #finish process numeric features


    #merge to reduse features
    if params['merge_cat']:
        df = merge_columns(df, columns=['Street', 'Alley'], str_name='StrAl', drop=True)
        df = merge_columns(df, columns=['RoofStyle', 'RoofMatl'], str_name='Roof_', drop=False)
        df = merge_columns(df, columns=['Exterior1st', 'Exterior2nd'], str_name='Exterior12', drop=True)
        df = merge_columns(df, columns=['Exterior12', 'ExterCond'], str_name='ExterCond', drop=False)
        df = merge_columns(df, columns=['Exterior12', 'ExterQual'], str_name='Exter1Qual', drop=False)  
        df.drop(['ExterCond', 'ExterQual'], axis=1, inplace=True)    
        df = merge_columns(df, columns=['Heating', 'HeatingQC', 'CentralAir'], str_name='Heat_AC', drop=True)
        df = merge_columns(df, columns=['KitchenAbvGr', 'Fireplaces', 'GarageCars'], str_name='KFG_Nr', drop=True)
        df = merge_columns(df, columns=['KitchenQual', 'FireplaceQu', 'GarageQual'], str_name='KFG_Qual', drop=True)
        df = merge_columns(df, columns=['GarageCond', 'GarageFinish'], str_name='GarageCF', drop=True)
        df = merge_columns(df, columns=['GarageType', 'PavedDrive'], str_name='GarageTPaveD', drop=True)    
        df = merge_columns(df, columns=['MoSold', 'YrSold'], str_name='YMSold', drop=False)    

      
    if params['precess_feat_only']:
       for c in df:
           print(df[c].value_counts(dropna=False))
   

    #clearning
    list_to_drop = ['Utilities']
    list_to_drop.extend(fYear) 
    df.drop(list_to_drop, axis=1, inplace=True)
   
    df = process_f_cat(df, mask, excl_prefix=num_prefix, minor_cutoff=params['minor_cutoff'])

    return df


###############################################################################
def main():


    #dataframe
    train, test, data, train_size = None, None, None, None

    #input
    path='../input/'
    train_file = 'train.csv'
    test_file = 'test.csv'
    train_pkl_path, test_pkl_path = train_file + '.pkl', test_file + '.pkl'
    
    params = {'to_pkl': False, 'from_pkl': False, 
              'dense_prefix': 'numeric_', 
              'cluster_num': True, 
              'merge_cat': True,
              'minor_cutoff': 8, #0.5%
              'precess_feat_only': False,
              'save_best': True}    

    
    #from plk; skip major processing
#pickle to saving time#########################################################
    if params['from_pkl']:
        train = pd.read_pickle(train_pkl_path)
        train_size = len(train)
        test = pd.read_pickle(test_pkl_path)
        data = pd.concat([train, test])
        print('Load in {}, {} samples from pickle'.format(len(train), len(test)))

    else:
#feature processing begin######################################################
        #read samples
        train = load_data(path + train_file)
        train_size = len(train)
        test = load_data(path + test_file)
        data = process_feature(train, test, params, mask=[target_id, target])
#output: produce data, train_size
#feature processing end########################################################    
    if params['precess_feat_only']:
        return

    #split data later may save some memory
    train = data[:train_size]
    test = data[train_size:]
    del data
    
    #to_pkl; skip major for later
    if params['to_pkl']:
        train.to_pickle(train_pkl_path)
        test.to_pickle(test_pkl_path)
        print('saving to pickle: {}, {}'.format(train_pkl_path, test_pkl_path))

    params['nr_round'] = 1000
    params['nr_min_round'] = 25
    params['nr_SS'] = 20
    params['SS_va_ratio'] = 0.1 
    params['probes'] = 25

    #prepare data for learning, model fit
    xgb_param = {'max_depth':12, 'eta':0.025, 'silent':1, 'objective':'reg:logistic'}
    xgb_param['booster'] = 'gblinear'    
    #xgb_param['booster'] = 'gbtree'
    xgb_param['eval_metric'] = 'rmse'
    xgb_param['nthread'] = 8
    
    if xgb_param['booster'] == 'gbtree':    
        xgb_param['subsample'] = 0.75
        xgb_param['colsample_bytree']= 0.75
        xgb_param['min_child_weight'] = 0
        
    elif xgb_param['booster'] == 'gblinear':
        xgb_param['lambda_bias'] = 0.0

    min_score = 9223372036854775807

    best_xgb_param = xgb_param

    params['test_mode'] = False

    #simulated annealing
    md_lower, md_upper = 8, 24
    eta_lower, eta_upper = -10, -2

    md = randint(md_lower, md_upper) #+-2
    pow_eta = randrange(eta_lower, eta_upper) #+-2
    
    for i in range(params['probes']):
        
        md = randint(md - 2, md_upper + 2)
        if md > md_upper:
            md = md_upper
        elif md < md_lower:
            md = md_lower
        
        xgb_param['max_depth'] = int(md)
        
        pow_eta = randrange(pow_eta - 1, pow_eta + 1)
        if pow_eta > eta_lower:
            pow_eta = eta_lower
        elif pow_eta < eta_lower:
            pow_eta = eta_lower
        
        xgb_param['eta'] = 2 ** pow_eta    
    
        test_pred, score = evaluation(train, test, params, xgb_param)
        
        if min_score > score:
            min_score = score
            best_xgb_param = xgb_param  
            
        print('Probes {:003d}th: eta {:.3f}, max_depth {}, score={:.3f} (best={:.3f})\n\n'.format(i, 
                  xgb_param['eta'], xgb_param['max_depth'], score, min_score))
    
    print('Found pptimial: eta {:.3f}, max_depth {}, best={:.3f}'.format( 
                  best_xgb_param['eta'], best_xgb_param['max_depth'], min_score))    
    
    params['test_mode'] = True
    test_pred, score = evaluation(train, test, params, best_xgb_param)
    create_submission(test[target_id], test_pred, prefix='xgb_test_mean_s{:.3f}'.
                        format(min_score), digit=-2)

    return
    
###############################################################################
###############################################################################

main()



