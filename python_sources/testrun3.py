import pandas as pd
import numpy as np
import xgboost as xgb

train =  np.genfromtxt('../input/train.csv', delimiter=',')
test =  np.genfromtxt('../input/test.csv', delimiter=',')
train = np.delete(train, (0), axis=0)
test = np.delete(test, (0), axis=0)

train_features = train[:,1:785]
train_labels = train[:,0]


features_train = train_features[0:20000]
features_test = train_features[20001:30000]

labels_train = train_labels[0:20000]
labels_test = train_labels[20001:30000]


dtrain = xgb.DMatrix(features_train, labels_train)
dtest = xgb.DMatrix(features_test, labels_test)

param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 10

watchlist = [ (dtrain,'train'), (dtest, 'test') ]
num_round = 50
bst = xgb.train(param, dtrain, num_round, watchlist );
# get prediction
pred = bst.predict( dtest );

print ('predicting, classification error=%f' %(sum(int(pred[i]) != labels_test[i] for i in range(len(labels_test))) / float(len(labels_test)) ))
# The competition datafiles are in the directory ../input
# Read competition data files:
#train = pd.read_csv("../input/train.csv")
#test  = pd.read_csv("../input/test.csv")

# Write to the log:
#print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
#print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs