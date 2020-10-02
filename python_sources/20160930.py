# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_auc_score
from scipy.io import loadmat
from operator import itemgetter
import random
import os
import time
import glob

import sys
import os
import numpy as np
import pandas as pd
from math import *
from scipy.io import loadmat
from scipy.stats import skew, kurtosis
#import pyeeg 
# pyeeg is the one that has very good fractal dimensions 
# computation but not installed here

random.seed(2016)
np.random.seed(2016)

def mat_to_data(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return ndata

def corr(data,type_corr):
    C = np.array(data.corr(type_corr))
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    w,v = np.linalg.eig(C)
    #print(w)
    x = np.sort(w)
    x = np.real(x)
    return x

def calculate_features(file_name):
    f = mat_to_data(file_name)
    fs = f['iEEGsamplingRate'][0,0]
    eegData = f['data']
    [nt, nc] = eegData.shape
    print((nt, nc))
    subsampLen = floor(fs * 60)
    numSamps = int(floor(nt / subsampLen));      # Num of 1-min samples
    sampIdx = range(0,(numSamps+1)*subsampLen,subsampLen)
    #print(sampIdx)
    feat = [] # Feature Vector
    for i in range(1, numSamps+1):
        print('processing file {} epoch {}'.format(file_name,i))
        epoch = eegData[sampIdx[i-1]:sampIdx[i], :]

        # compute Shannon's entropy, spectral edge and correlation matrix
        # segments corresponding to frequency bands
        lvl = np.array([0.1, 4, 8, 12, 30, 70, 180])  # Frequency levels in Hz
        lseg = np.round(nt/fs*lvl).astype('int')
        D = np.absolute(np.fft.fft(epoch, n=lseg[-1], axis=0))
        D[0,:]=0                                # set the DC component to zero
        D /= D.sum()                      # Normalize each channel               
        
        dspect = np.zeros((len(lvl)-1,nc))
        for j in range(len(dspect)):
            dspect[j,:] = 2*np.sum(D[lseg[j]:lseg[j+1],:], axis=0)

        # Find the shannon's entropy
        spentropy = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)

        # Find the spectral edge frequency
        sfreq = fs
        tfreq = 40
        ppow = 0.5

        topfreq = int(round(nt/sfreq*tfreq))+1
        A = np.cumsum(D[:topfreq,:])
        B = A - (A.max()*ppow)
        spedge = np.min(np.abs(B))
        spedge = (spedge - 1)/(topfreq-1)*tfreq

        # Calculate correlation matrix and its eigenvalues (b/w channels)
        data = pd.DataFrame(data=epoch)
        type_corr = 'pearson'
        lxchannels = corr(data, type_corr)
        
        # Calculate correlation matrix and its eigenvalues (b/w freq)
        data = pd.DataFrame(data=dspect)
        lxfreqbands = corr(data, type_corr)
        
        # Spectral entropy for dyadic bands
        # Find number of dyadic levels
        ldat = int(floor(nt/2.0))
        no_levels = int(floor(log(ldat,2.0)))
        seg = floor(ldat/pow(2.0, no_levels-1))

        # Find the power spectrum at each dyadic level
        dspect = np.zeros((no_levels,nc))
        for j in range(no_levels-1,-1,-1):
            dspect[j,:] = 2*np.sum(D[int(floor(ldat/2.0))+1:ldat,:], axis=0)
            ldat = int(floor(ldat/2.0))

        # Find the Shannon's entropy
        spentropyDyd = -1*np.sum(np.multiply(dspect,np.log(dspect)), axis=0)

        # Find correlation between channels
        data = pd.DataFrame(data=dspect)
        lxchannelsDyd = corr(data, type_corr)
        
        # Fractal dimensions
        no_channels = nc
        #fd = np.zeros((2,no_channels))
        #for j in range(no_channels):
        #    fd[0,j] = pyeeg.pfd(epoch[:,j])
        #    fd[1,j] = pyeeg.hfd(epoch[:,j],3)
        #    fd[2,j] = pyeeg.hurst(epoch[:,j])

        #[mobility[j], complexity[j]] = pyeeg.hjorth(epoch[:,j)
        # Hjorth parameters
        # Activity
        activity = np.var(epoch)

        # Mobility
        mobility = np.divide(np.std(np.diff(epoch)), np.std(epoch))

        # Complexity
        complexity = np.divide(np.divide(np.diff(np.diff(epoch)),
                                         np.std(np.diff(epoch))), mobility)
        # Statistical properties
        # Skewness
        sk = skew(epoch)

        # Kurtosis
        kurt = kurtosis(epoch)

        # compile all the features
        feat = np.concatenate((feat,
                               spentropy.ravel(),
                               spedge.ravel(),
                               lxchannels.ravel(),
                               lxfreqbands.ravel(),
                               spentropyDyd.ravel(),
                               lxchannels.ravel(),
                               #fd.ravel(),
                               activity.ravel(),
                               mobility.ravel(),
                               complexity.ravel(),
                               sk.ravel(),
                               kurt.ravel()
                                ))

    return feat


def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
    outfile.close()


def get_importance(gbm, features):
    create_feature_map(features)
    importance = gbm.get_fscore(fmap='xgb.fmap')
    importance = sorted(importance.items(), key=itemgetter(1), reverse=True)
    return importance


def intersect(a, b):
    return list(set(a) & set(b))


def print_features_importance(imp):
    for i in range(len(imp)):
        print("# " + str(imp[i][1]))
        print('output.remove(\'' + imp[i][0] + '\')')


def mat_to_pandas(path):
    mat = loadmat(path)
    names = mat['dataStruct'].dtype.names
    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}
    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])


def create_simple_csv():
    # TRAIN
    print('Create train.csv...')
    files = sorted(glob.glob("../input/train_*/*.mat"))
    out = open("simple_train.csv", "w")
    out.write("Id,patient_id")
    for i in range(16):
        out.write(",avg_" + str(i))
    out.write(",file_size,result\n")
    for fl in files:
        # print('Go for ' + fl)
        id_str = os.path.basename(fl)[:-4]
        arr = id_str.split("_")
        patient = int(arr[0])
        id = int(arr[1])
        result = int(arr[2])
        new_id = patient*100000 + id
        try:
            tables = mat_to_pandas(fl)
        except:
            continue
        out.write(str(new_id))
        out.write("," + str(patient))
        for f in sorted(list(tables.columns.values)):
            mean = tables[f].mean()
            out.write("," + str(mean))
        out.write("," + str(os.path.getsize(fl)))
        out.write("," + str(result) + "\n")
        # break
    out.close()

    # TEST
    print('Create test.csv...')
    files = sorted(glob.glob("../input/test_*/*.mat"))
    out = open("simple_test.csv", "w")
    out.write("Id,patient_id")
    for i in range(16):
        out.write(",avg_" + str(i))
    out.write(",file_size\n")
    for fl in files:
        # print('Go for ' + fl)
        id_str = os.path.basename(fl)[:-4]
        arr = id_str.split("_")
        patient = int(arr[0])
        id = int(arr[1])
        new_id = patient*100000 + id
        try:
            tables = mat_to_pandas(fl)
        except:
            continue
        out.write(str(new_id))
        out.write("," + str(patient))
        for f in sorted(list(tables.columns.values)):
            mean = tables[f].mean()
            out.write("," + str(mean))
        out.write("," + str(os.path.getsize(fl)))
        out.write("\n")
        # break
    out.close()

def create_simple_csv2():
    # TRAIN
    print('Create train.csv...')
    files = sorted(glob.glob("../input/train_*/*.mat"))
    out = open("simple_train.csv", "w")
    out.write("Id,patient_id")
    for i in range(11):
        out.write(",feat_" + str(i))
    out.write(",file_size,result\n")
    for fl in files:
        # print('Go for ' + fl)
        id_str = os.path.basename(fl)[:-4]
        arr = id_str.split("_")
        patient = int(arr[0])
        id = int(arr[1])
        result = int(arr[2])
        new_id = patient*100000 + id
        try:
            tables = calculate_features(f1)
        except:
            continue
        out.write(str(new_id))
        out.write("," + str(patient))
        for f in sorted(list(tables.columns.values)):
            #mean = tables[f].mean()
            out.write("," + str(tables[f]))
        out.write("," + str(os.path.getsize(fl)))
        out.write("," + str(result) + "\n")
        # break
    out.close()

    # TEST
    print('Create test.csv...')
    files = sorted(glob.glob("../input/test_*/*.mat"))
    out = open("simple_test.csv", "w")
    out.write("Id,patient_id")
    for i in range(16):
        out.write(",avg_" + str(i))
    out.write(",file_size\n")
    for fl in files:
        # print('Go for ' + fl)
        id_str = os.path.basename(fl)[:-4]
        arr = id_str.split("_")
        patient = int(arr[0])
        id = int(arr[1])
        new_id = patient*100000 + id
        try:
            tables = mat_to_pandas(fl)
        except:
            continue
        out.write(str(new_id))
        out.write("," + str(patient))
        for f in sorted(list(tables.columns.values)):
            mean = tables[f].mean()
            out.write("," + str(mean))
        out.write("," + str(os.path.getsize(fl)))
        out.write("\n")
        # break
    out.close()

def run_single(train, test, features, target, random_state=1):
    eta = 0.2
    max_depth = 5
    subsample = 0.77
    colsample_bytree = 0.8
    start_time = time.time()

    print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
    params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": eta,
        "tree_method": 'exact',
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "silent": 1,
        "seed": random_state,
    }
    num_boost_round = 700
    early_stopping_rounds = 25
    test_size = 0.207

    kf = KFold(len(train.index), n_folds=int(round(1/test_size, 0)), shuffle=True, random_state=random_state)
    train_index, test_index = list(kf)[0]
    print('Length of train: {}'.format(len(train_index)))
    print('Length of valid: {}'.format(len(test_index)))

    X_train, X_valid = train[features].as_matrix()[train_index], train[features].as_matrix()[test_index]
    y_train, y_valid = train[target].as_matrix()[train_index], train[target].as_matrix()[test_index]

    print('Length train:', len(X_train))
    print('Length valid:', len(X_valid))

    dtrain = xgb.DMatrix(X_train, y_train)
    dvalid = xgb.DMatrix(X_valid, y_valid)

    watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
    gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
                    early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

    print("Validating...")
    check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
    score = roc_auc_score(y_valid, check)
    print('Check error value: {:.6f}'.format(score))

    imp = get_importance(gbm, features)
    print('Importance array: ', imp)

    print("Predict test set...")
    test_prediction = gbm.predict(xgb.DMatrix(test[features].as_matrix()), ntree_limit=gbm.best_iteration+1)

    print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))
    return test_prediction.tolist(), score


def create_submission(score, test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(score) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('File,Class\n')
    total = 0
    for id in test['Id']:
        patient = id // 100000
        fid = id % 100000
        str1 = str(patient) + '_' + str(fid) + '.mat' + ',' + str(prediction[total])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('Id')
    # output.remove('file_size')
    return sorted(output)


def read_test_train():
    print("Load train.csv...")
    train = pd.read_csv("simple_train.csv")
    print("Load test.csv...")
    test = pd.read_csv("simple_test.csv")
    print("Process tables...")
    features = get_features(train, test)
    return train, test, features


print('XGBoost: {}'.format(xgb.__version__))
create_simple_csv2()
train, test, features = read_test_train()
print('Length of train: ', len(train))
print('Length of test: ', len(test))
print('Features [{}]: {}'.format(len(features), sorted(features)))
test_prediction, score = run_single(train, test, features, 'result')
create_submission(score, test, test_prediction)
