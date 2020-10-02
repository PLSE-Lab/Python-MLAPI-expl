#keras for house sold price forcast
"""
Created on Sat Oct  1 00:38:55 2016

@author: Chia-Ta
https://www.kaggle.com/cttsai/
"""


import itertools
import time
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
from keras.preprocessing import sequence
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, Dropout, Reshape, Merge
from keras.layers.advanced_activations import PReLU, LeakyReLU, ELU, SReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
from keras.optimizers import SGD
from keras.models import Model, load_model


seed = 101472016
np.random.seed(seed)
t_digit = 1
scr_digit = 2
#hash_digit = (2 ** 5) * (2 ** 10) #32*1024


#useful variables
target, target_id = 'SalePrice', 'Id'
    
###############################################################################
#keras
###############################################################################
def design_keras_model(data, params, mask=[]):
    
    #seperate prefix; #dense layers for numeric featureas
    dense_prefix = params['dense_prefix']
    
    #param
    dim = 2 ** 7
    hidden = 2 ** 8 
    L2_reg = 10 ** -9
    opt_dot_merged = True

    if params['run_simple']:
        dim = 2 ** 2
        hidden = 2 ** 3                
        
    columns = mask_columns(data.columns.tolist(), mask)
    print('\nApplied {} features'.format(len(columns)))
    for i, j in enumerate(columns, start=1):
        print(i, ': ', j)    
        
    print('\ndesign model structure')    
    dim_in_num, dim_in_cat = 0, 0
    inputs = []
    num_c_layers = []
    embed_c_layers = []
        
    for c in columns:
        if str(c).startswith(dense_prefix):
            inputs_c = Input(shape=(1,), dtype='float32')
            inputs.append(inputs_c)
            
            num_c_layers.append(inputs_c)
            dim_in_num += 1

        else:
            inputs_c = Input(shape=(1,), dtype='int32')
            inputs.append(inputs_c)
            
            num_c = len(np.unique(data[c].values))
            
            vdim = dim        
            if params['run_simple'] == False:
                vdim = int(2 ** ceil(log10(num_c) + 1))
                if vdim > dim:
                    vdim = dim
            
            embed_c = Embedding(
                    num_c,
                    vdim,
                    dropout=0.1,
                    input_length=1
                    )(inputs_c)
            
            embed_c_layers.append(embed_c)#Output(None, 1, dim)
            dim_in_cat += vdim
            
    del data
    #end adding inputs

    #hidden
    print('inputs: num {}, cat {}, cat dim {}'.format(
            len(num_c_layers), len(embed_c_layers), dim_in_cat))
    hidden_num = int(2 ** ceil(log2(len(num_c_layers)) + 1))    
    hidden_cat = int(2 ** ceil(log1p(dim_in_cat) + 1))
    
    if params['run_simple']:
        hidden_num = hidden        
        hidden_cat = hidden   
        
    print('hidden nodes for num {}, cat {}, higher {}'.format(hidden_num, hidden_cat, hidden))
    
    #layers for nnumeric    
    concat_numeric = Merge(mode='concat')(num_c_layers)
    concat_numeric = Dense(hidden_num, activation='sigmoid')(concat_numeric)
    concat_numeric = Dropout(0.25)(concat_numeric)

    #layers for embed
    concat_embed = Merge(mode='concat')(embed_c_layers)
    concat_embed = Flatten()(concat_embed)
    concat_embed = Dense(hidden_cat, activation='sigmoid')(concat_embed)
    concat_embed = Dropout(0.25)(concat_embed)

    merge_layers = [concat_numeric, concat_embed]

    
    if opt_dot_merged:
        dot_numeric = Reshape((1, hidden_num))(concat_numeric)
        dot_embed = Reshape((1, hidden_cat))(concat_embed)
        #numeric x embed
        dot_concat = Merge(mode='dot', dot_axes=1)([dot_numeric, dot_embed])#Output(hidden, hidden)        
        dot_concat = Flatten()(dot_concat)           
        dot_concat = Dense(int(sqrt(hidden_cat * hidden_num)), activation='sigmoid')(dot_concat)
        dot_concat = Dropout(0.25)(dot_concat)
        merge_layers.append(dot_concat)
        #embed x embed
        dot_concat = Merge(mode='dot', dot_axes=1)([dot_embed, dot_embed])#Output(hidden, hidden)        
        dot_concat = Flatten()(dot_concat)           
        dot_concat = Dense(hidden_cat, activation='sigmoid')(dot_concat)
        dot_concat = Dropout(0.25)(dot_concat)
        merge_layers.append(dot_concat)
        

    #deep layers which stacking from the flatten layer
    deep = Merge(mode='concat')(merge_layers)
    deep = Dropout(0.25)(deep)  
    deep = Dense(hidden, activation='sigmoid')(deep)
    deep = Dropout(0.5)(deep) 
    #'softplus', 'sigmoid', 'relu'
    
    #set output
    #W_regularizer=l2(0.01), activity_regularizer=activity_l2(0.01)
    outputs = Dense(1, activation='softplus', 
            W_regularizer=l2(L2_reg), activity_regularizer=activity_l2(L2_reg)
        )(deep)
    
    model = Model(input=inputs, output=outputs)
    model.compile(
        loss = params['loss'],
        optimizer = params['optimizer'],
        #metrics=['msle', 'mse'],
        )
     
    #print(model.summary())
    
    return model


def evaluation(train, test, params, model):
    #prepare data for learning
    test_pred = None
    #scaling y
    real_y = train[target]
    #train_scl_y, scl_low, scl_range = rescale_column(train[target])
    train_scl_y = real_y.apply(log1p)
    #covert to np array as input
    #X_id = train[target_id]
    mask = [target_id, target]
    X = train.drop(mask, axis=1).as_matrix()
    y = train_scl_y.as_matrix()
    
    test_id = test[target_id]
    X_t = test.drop(mask, axis=1).as_matrix()
    X_t = [X_t[:,i] for i in range(X_t.shape[1])]
    del train, test

#Block of Model learning begin#################################################
    #model params
    default_batch_size = (2 ** 7)
    nr_epoch, min_epoch = 500, 20
    nr_run, va_ratio = 20, 0.25

    if params['run_simple']:
        nr_epoch, min_epoch = 5, 3
        nr_run, va_ratio = 5, 0.45

    #option
    set_callback = []

    if params['early_stop']:
        ckp_tr = EarlyStopping(monitor='loss', patience=min_epoch)
        ckp_va = EarlyStopping(monitor='val_loss', patience=min_epoch)
        set_callback.extend([ckp_tr, ckp_va])
    
    if params['save_best']:
        filepath  = 'keras_house_best.hdf5'
        #filepath  = 'keras_house_epoch{epoch:02d}_s{val_loss:.4f}.hdf5'
        ckp_sv_best = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True)
        #filepath  = 'keras_house_weights_epoch{epoch:02d}_s{val_loss:.4f}.h5'
        #ckp_sv_best = ModelCheckpoint(filepath, monitor='val_loss', save_best_only=True, save_weights_only=True)
        params['best_model'] = filepath        
        set_callback.append(ckp_sv_best)
    
    
    #start training and evaluation
    count, score = 0, 0.0
    
    ss = ShuffleSplit(n_splits=nr_run, test_size=va_ratio, train_size=1-va_ratio, random_state=0)
    #np.random.rand
    for ind_tr, ind_va in ss.split(X):
        
        iter_start_ckp = time.time()
        
        #split
        y_train, y_valid = y[ind_tr], y[ind_va]
        
        X_train = np.array(X[ind_tr])
        X_train = [X_train[:,i] for i in range(X.shape[1])]
        
        X_valid = np.array(X[ind_va])
        X_valid = [X_valid[:,i] for i in range(X.shape[1])]
        
        model.fit(
            X_train, y_train,
            batch_size=default_batch_size, 
            nb_epoch=nr_epoch, verbose=2, shuffle=True,
            validation_data=[X_valid, y_valid],
            callbacks=set_callback,
        )
#Block of Model learning End###################################################        
        count += 1
        
        #validation in real
        if params['run_simple'] == False:
            y_valid_pred = model.predict(X_valid, batch_size=default_batch_size).flatten()
            #y_valid_pred = scaling_column(pd.Series(y_valid_pred), scl_low, scl_range)
            y_valid_pred = pd.Series(y_valid_pred).apply(expm1)
        
            y_valid_real = real_y.iloc[ind_va]

            mae = mean_absolute_error(y_valid_real.as_matrix(), y_valid_pred)
            rmse = sqrt(mean_squared_error(y_valid_real.as_matrix(), y_valid_pred))
            r2 = r2_score(y_valid_real.as_matrix(), y_valid_pred)
            
            score = r2            
            
            print('\niter {:02d} validation real scale: rmse={:.2f}, mae={:.2f}, r2={:.3f}, in {} s\n'.format(
                count, rmse, mae, r2, round(time.time() - iter_start_ckp, t_digit)))
        
            if r2 >= 0.8:
            #only model with good quality to predict test
                stem = 'test' + '_iter_{:02d}_r2_{:.3f}'.format(count, score)
                test_pred = model.predict(X_t, batch_size=default_batch_size).flatten()
                #test_pred = scaling_column(pd.Series(test_pred), scl_low, scl_range)
                test_pred = pd.Series(test_pred).apply(expm1)
                create_submission(test_id, test_pred, prefix=stem, digit=-2)
        #break

    if params['save_best']:
        model = load_model(params['best_model'])
        X = [X[:,i] for i in range(X.shape[1])]
        score = model.evaluate(X, train_scl_y, batch_size=default_batch_size, verbose=2)
        test_pred = model.predict(X_t, batch_size=default_batch_size).flatten()
        test_pred = pd.Series(test_pred).apply(expm1)
        
    return test_pred, sqrt(score)


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
    #encode
    count = 0
    for c in columns:
        if str(c).startswith(excl_prefix) == False:
            
            df[c] = df[c].fillna('na').astype(str)

            if minor_cutoff > 0:
                freq = df[c].value_counts(dropna=False)        
                minor = freq.loc[lambda x : x < minor_cutoff].index.tolist()
                if len(minor) > 2:
                    df[c].replace(to_replace=minor, value='minor', inplace=True)

            df[c] = LabelEncoder().fit_transform(df[c].values)
            count += 1
    
    print('\nOne-hot encoding {} features on {} samples'.format(count, len(df)))    
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
              'run_simple': False, 
              'early_stop': True, 'save_best': True}    
    params['loss'] = 'mean_squared_error'
    #params['loss'] = 'mean_squared_logarithmic_error'
    #params['loss'] = 'mean_absolute_error'
    params['optimizer'] = 'adam'
    #params['optimizer'] = 'sgd'
    #params['optimizer'] = 'RMSprop'
    
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

    #design model
    model = design_keras_model(data, params, mask=[target_id, target])    

    #split data later may save some memory
    train = data[:train_size]
    test = data[train_size:]
    del data
    
    #to_pkl; skip major for later
    if params['to_pkl']:
        train.to_pickle(train_pkl_path)
        test.to_pickle(test_pkl_path)
        print('saving to pickle: {}, {}'.format(train_pkl_path, test_pkl_path))

    #prepare data for learning, model fit
    test_pred, score = evaluation(train, test, params, model)
    
    create_submission(test[target_id], test_pred, prefix='keras_test_best_model_s{:.3f}'.format(score), digit=-2)
    
    return
    
###############################################################################
###############################################################################

main()



