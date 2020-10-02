#!/usr/bin/env python
# coding: utf-8

# '''
# Credit to this kernel:
# https://www.kaggle.com/remidi/neural-compression-auto-encoder-lb-0-55/code
# 
# I do some change and make it work for Santander . It is my first time to use denoising autoencoder.
# Please provide feedback and upvote if you like it :)
# '''

# In[ ]:



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from sklearn.base import BaseEstimator,TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from sklearn.linear_model import ElasticNetCV, LassoLarsCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import NMF
from sklearn.cluster import FeatureAgglomeration
import scipy
from sklearn.ensemble import RandomForestRegressor
import random

random.seed(1234)

import warnings
warnings.filterwarnings('ignore')


class StackingEstimator(BaseEstimator, TransformerMixin):
    
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self
    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # add class prodiction as a synthetic feature
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# copy from https://www.kaggle.com/mathormad/knowledge-distillation-with-nn-rankgauss
class GaussRankScaler():

    def __init__( self ):
        self.epsilon = 1e-9
        self.lower = -1 + self.epsilon
        self.upper =  1 - self.epsilon
        self.range = self.upper - self.lower

    def fit_transform( self, X ):

        i = np.argsort( X, axis = 0 )
        j = np.argsort( i, axis = 0 )

        assert ( j.min() == 0 ).all()
        assert ( j.max() == len( j ) - 1 ).all()

        j_range = len( j ) - 1
        self.divider = j_range / self.range

        transformed = j / self.divider
        transformed = transformed - self.upper
        transformed = scipy.special.erfinv( transformed )
        ############
        # transformed = transformed - np.mean(transformed)

        return transformed

target_col='target'
id_col='ID_code'
submission = pd.read_csv('../input/sample_submission.csv')
id_test = submission[id_col].values
# function for auto encoder with a compressed components n_comp = 12
def neural_compression_v2(train, test):
    dataset = pd.concat([train.drop(target_col, axis=1), test], axis=0)
    ids = dataset[id_col]
    dataset.drop(id_col, axis=1, inplace=True)
    y_train = train[target_col]
    
    cat_vars = [c for c in dataset.columns if dataset[c].dtype == 'object']
    for c in cat_vars:
        t_data = pd.get_dummies(dataset[c], prefix=c)
        dataset = pd.concat([dataset, t_data], axis=1)

    dataset.drop(cat_vars, axis=1, inplace=True)
    # We scale both train and test data so that our NN works better.
    sc = StandardScaler()
#     sc = GaussRankScaler()# Gauss Rank does not work...
    sc.fit_transform(dataset)

    dataset = sc.fit_transform(dataset)

    train = dataset[:train.shape[0]]
    test = dataset[train.shape[0]:]

    print("one hot encoded train shape :: {}".format(train.shape))
    print("one hot encoded test shape :: {}".format(test.shape))
    
    ''' neural network compression code '''
    
    import keras
    from keras import regularizers
    from keras.layers import Input, Dense,BatchNormalization,Dropout
    from keras.models import Model
    from keras.regularizers import l2
    # adding some noise to data before feed them to nn
    train = train + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=train.shape) 
    test = test + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=test.shape)
    l2_reg_embedding = 1e-5
    print(keras.__version__)
    init_dim = train.shape[1]

    input_row = Input(shape=(init_dim, ))
    encoded = Dense(512, activation='elu',kernel_regularizer=l2(l2_reg_embedding))(input_row)
    encoded = Dropout(0.2)(encoded)
    encoded=BatchNormalization()(encoded)
    encoded = Dense(256, activation='elu',kernel_regularizer=l2(l2_reg_embedding))(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(128, activation='elu',kernel_regularizer=l2(l2_reg_embedding))(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(64, activation='elu',kernel_regularizer=l2(l2_reg_embedding))(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(32, activation='elu',kernel_regularizer=l2(l2_reg_embedding))(encoded)

    encoded = Dense(16, activation='elu')(encoded)
    
    decoded = Dense(32, activation='elu',kernel_regularizer=l2(l2_reg_embedding))(encoded)
    encoded = Dropout(0.2)(encoded)
    decoded = Dense(64, activation='elu',kernel_regularizer=l2(l2_reg_embedding))(decoded)
    encoded = Dropout(0.2)(encoded)
    decoded = Dense(128, activation='elu',kernel_regularizer=l2(l2_reg_embedding))(decoded)
    encoded = Dropout(0.2)(encoded)
    decoded = Dense(256, activation='elu',kernel_regularizer=l2(l2_reg_embedding))(decoded)
    encoded = Dropout(0.2)(encoded)
    encoded = BatchNormalization()(encoded)
    decoded = Dense(512, activation='elu',kernel_regularizer=l2(l2_reg_embedding))(decoded)
    decoded = Dense(init_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_row, outputs=decoded)
    autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
    #we use the train data to train
    autoencoder.fit(train, train,
                    batch_size=512,verbose=2,
                    shuffle=True, validation_data=(test, test), epochs=4)

    # compressing the data
    encoder = Model(inputs=input_row, outputs=encoded)
    train_compress = encoder.predict(train,batch_size=4048)
    test_compress = encoder.predict(test,batch_size=4048)

    # denoising the data
    denoised_train = autoencoder.predict(train,batch_size=4048)
    denoised_test = autoencoder.predict(test,batch_size=4048)
    
    return train_compress, test_compress, denoised_train, denoised_test

train_compress, test_compress, denoised_train, denoised_test = neural_compression_v2(train, test)



for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))
        

n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop([target_col], axis=1))
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train.drop([target_col], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train.drop([target_col], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train.drop([target_col], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train.drop([target_col], axis=1))
srp_results_test = srp.transform(test)

# NMF
# nmf = NMF(n_components=n_comp, init='nndsvdar', random_state=420)
# nmf_results_train = nmf.fit_transform(train.drop([target_col], axis=1))
# nmf_results_test = nmf.transform(test)

# FAG
fag = FeatureAgglomeration(n_clusters=n_comp, linkage='ward')
fag_results_train = fag.fit_transform(train.drop([target_col], axis=1))
fag_results_test = fag.transform(test)

usable_columns = list(set(train.columns) - set([target_col]))

# Append decomposition components to datasets
for i in range(1, n_comp + 1):
    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]
    
    # train['nmf_' + str(i)] = nmf_results_train[:, i - 1]
    # test['nmf_' + str(i)] = nmf_results_test[:, i - 1]
    
#     train['fag_' + str(i)] = fag_results_train[:, i - 1]
#     test['fag_' + str(i)] = fag_results_test[:, i - 1]

for j in range(1, train_compress.shape[1]):
#     train['aen_' + str(j)] = train_compress[:, j-1]
#     test['aen_' + str(j)] = test_compress[:, j-1]
    train['aen_' + str(j)] = denoised_train[:, j-1]
    test['aen_' + str(j)] = denoised_test[:, j-1]
    
    
    


y = train[target_col].values


# finaltrainset and finaltestset are data to be used only the stacked model (does not contain PCA, SVD... arrays) 
finaltrainset = train[usable_columns].values
finaltestset = test[usable_columns].values

#--training & test stratified split
X=train.drop(target_col, axis=1)


# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(X, y,stratify=y, test_size=0.10,random_state=2019)


'''Train the xgb model then predict the test data'''
print('running ....... ')

# lgb_params = {
#     'num_leaves': 25,
#     'learning_rate': 0.02,
#     'max_depth': 2,#7
#     'subsample': 0.8,
#     'colsample_bytree':0.8,
#     'objective': 'binary',
#     'eval_freq':100,
#     'verbose': 0,
#     'metric': 'auc',

# }
lgb_params = {
        'bagging_freq': 5,
        'bagging_fraction': 0.38,
        'boost_from_average':'false',
        'boost': 'gbdt',
        'feature_fraction': 0.045,
        'learning_rate': 0.0095,
        'max_depth': -1,  
        'metric':'auc',
        'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 13,
        'num_threads': 8,
        'tree_learner': 'serial',
        'objective': 'binary', 
        'verbosity': 1
    }


import lightgbm as lgb
dtrain = lgb.Dataset(X_train, y_train)
dvalid = lgb.Dataset(X_valid, y_valid)
# dtest = lgb.Dataset(test)

num_boost_rounds = 50000# maybe overfitting
# train model
model = lgb.train(lgb_params, dtrain,valid_sets=dvalid, num_boost_round=num_boost_rounds,early_stopping_rounds=500)
y_pred = model.predict(test)


# In[ ]:


'''Train the stacked models then predict the test data'''

stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_depth=3, max_features=0.55, min_samples_leaf=18, min_samples_split=14, subsample=0.7)),
    LassoLarsCV()
)
X_train2, X_valid2, y_train2, y_valid2 = train_test_split(finaltrainset, y,stratify=y, test_size=0.10,random_state=2019)
stacked_pipeline.fit(X_train2, y_train2)
results = stacked_pipeline.predict(finaltestset)

'''auc Score on the entire Train data when averaging'''

print('auc score on train data:')
lgb_preds=model.predict(X_valid)
stack_score=roc_auc_score(y_valid2,stacked_pipeline.predict(X_valid2)*0.2855 + lgb_preds*0.7145)
print('stack_score:{}'.format(stack_score))
score = roc_auc_score(y_valid, lgb_preds)
print("lgboost score : {}".format(score))


'''Save submission to csv file'''
sub = pd.DataFrame()
sub[id_col] = id_test
# sub[target_col] = y_pred*0.75 + results*0.25
sub[target_col] = y_pred
sub.to_csv('lgb_stack_submission{}.csv'.format(stack_score), index=False)


# It seems that stacked_pipeline make performance worse. 
