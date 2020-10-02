#!/usr/bin/env python
# coding: utf-8

# <div>
#     <h2> Network Intrusion Detection using Linear Models, GBDT Ensembles and Deep Learning - A comparative study using state-of-the-art tools and libraries </h2>
#     <p> In this study, we use the <a href='https://www.unb.ca/cic/datasets/nsl.html'>NSL KDD dataset</a> to predict the probability of occurence of 23 different classes of attacks on a network. Here, we use three different categories of models - Linear Models including Logistic Regression and Stochastic Gradient Descent (SGD) classifier; Gradient Boosting Decision Tree emsembles including LightGBM (LGBM) and XGBoost; and a Deep Neural Network (DNN) classifier. We also train a stacked model consisting of all these models as base learners. Finally, we compare the performances of all the models for Network Intrusion Detection using the NSL-KDD dataset and draw useful conclusions.</p>
# </div>

# In[ ]:


# import required packages

import pandas as pd 
import numpy as np
import os, gc, time, warnings

from scipy.misc import imread
from scipy import sparse
import scipy.stats as ss
from scipy.sparse import csr_matrix, hstack, vstack

import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS
from PIL import Image
import matplotlib_venn as venn
import pydot, graphviz
from IPython.display import Image

import string, re, nltk, collections
from nltk.util import ngrams
from nltk.corpus import stopwords
import spacy
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer   

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import tensorflow as tf
import keras.backend as K
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate, BatchNormalization
from keras.layers import GRU, LSTM, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback


# In[ ]:


# settings

os.environ['OMP_NUM_THREADS'] = '4'
start_time = time.time()
color = sns.color_palette()
sns.set_style("dark")
warnings.filterwarnings("ignore")

eng_stopwords = set(stopwords.words("english"))
lem = WordNetLemmatizer()
ps = PorterStemmer()
tokenizer = TweetTokenizer()

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# print the names of files available in the root directory
print(os.listdir('../input'))


# In[ ]:


# import the dataset

train = pd.read_csv('../input/nslkdd-dataset/KDDTrain.csv')
test = pd.read_csv('../input/nslkdd-dataset/KDDTest.csv')


# In[ ]:


print("Training data information...")
train.info()


# In[ ]:


train.head(10)


# In[ ]:


print('Test data information...')
test.info()


# <div>
#     <h3> Feature Engineering </h3>
#     <p> We will extract useful features from the existing ones, and convert some features into formats suitable for training on different categories of models. </p>
# </div>

# In[ ]:


# obtaining a new target variable for each attack class

attack_classes = ['back', 'buffer_overflow', 'ftp_write', 'guess_passwd', 'imap', 'ipsweep', 'land', 
                  'loadmodule', 'multihop', 'neptune', 'nmap', 'normal', 'perl', 'phf', 'pod', 'portsweep',
                  'rootkit', 'satan', 'smurf', 'teardrop', 'warezmaster']

dos_classes = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']
probe_classes = ['ipsweep', 'satan', 'nmap', 'portsweep']
r2l_classes = ['ftp_write', 'guess_passwd', 'imap', 'phf', 'multihop', 'wazermaster']
u2r_classes = ['buffer_overflow', 'loadmodule', 'rootkit', 'perl']
normal_classes = ['normal']

train_label = pd.DataFrame()
test_label = pd.DataFrame()

for attack_type in attack_classes:
    train_label[attack_type] = train['attack_class'].apply(lambda x : int(x == attack_type))
    test_label[attack_type] = test['attack_class'].apply(lambda x : int(x == attack_type))    


# In[ ]:


# extracting numerical labels from categorical data

encoder = LabelEncoder()

train['protocol_type_label'] = encoder.fit_transform(train['protocol_type'])
test['protocol_type_label'] = encoder.fit_transform(test['protocol_type'])

train['service_label'] = encoder.fit_transform(train['service'])
test['service_label'] = encoder.fit_transform(test['service'])

train['flag_label'] = encoder.fit_transform(train['flag'])
test['flag_label'] = encoder.fit_transform(test['flag'])


# In[ ]:


# removing useless columns

train.drop(['attack_class', 'num_learners'], axis = 1, inplace = True)
test.drop(['attack_class', 'num_learners'], axis = 1, inplace = True)


# In[ ]:


print("Training data information...")
train.info()


# In[ ]:


# creating dataframes for storing training data for stacked model

stacked_train_df = {}
stacked_test_df = {}

for attack_type in attack_classes:
    stacked_train_df[attack_type] = pd.DataFrame()
    stacked_test_df[attack_type] = pd.DataFrame()


# In[ ]:


# preparing data for training on models

x_train = train.copy(deep = True)
x_train.drop(['protocol_type', 'service', 'flag'], axis = 1, inplace = True)

x_test = test.copy(deep = True)
x_test.drop(['protocol_type', 'service', 'flag'], axis = 1, inplace = True)


# <div>
#     <h3>Linear Models</h3>
# </div>

# In[ ]:


# logistic regression classifier

def getLRClf():
    clf = LogisticRegression(C = 0.2, solver = 'liblinear')
    return clf


# In[ ]:


# training on logistic regression classifier

lr_accuracy = []
lr_precision = []
lr_recall = []
lr_tn = []
lr_fp = []
lr_fn = []
lr_tp = []
lr_dos_accuracy = []
lr_probe_accuracy = []
lr_r2l_accuracy = []
lr_u2r_accuracy = []
lr_normal_accuracy = []

for attack_type in attack_classes:
    clf = getLRClf()
    clf.fit(x_train, train_label[attack_type])
    y_pred = clf.predict(x_test)
    stacked_train_df[attack_type]['logistic_regression'] = clf.predict(x_train)
    stacked_test_df[attack_type]['logistic_regression'] = y_pred
    lr_accuracy += [accuracy_score(test_label[attack_type], y_pred)]
    if attack_type in dos_classes:
        lr_dos_accuracy += [lr_accuracy[-1]]
    if attack_type in probe_classes:
        lr_probe_accuracy += [lr_accuracy[-1]]
    if attack_type in r2l_classes:
        lr_r2l_accuracy += [lr_accuracy[-1]]
    if attack_type in u2r_classes:
        lr_u2r_accuracy += [lr_accuracy[-1]]
    if attack_type in normal_classes:
        lr_normal_accuracy += [lr_accuracy[-1]]
    lr_precision += [precision_score(test_label[attack_type], y_pred)]
    lr_recall += [recall_score(test_label[attack_type], y_pred)]
    cm = confusion_matrix(test_label[attack_type], y_pred).ravel()
    if len(cm) > 1:
        lr_tn += [cm[0]]
        lr_fp += [cm[1]]
        lr_fn += [cm[2]]
        lr_tp += [cm[3]]
    else:
        lr_tn += [0]
        lr_fp += [0]
        lr_fn += [0]
        lr_tp += [0]
    
mean_lr_accuracy = np.mean(lr_accuracy)
mean_lr_precision = np.mean(lr_precision)
mean_lr_recall = np.mean(lr_recall)

print("Logistic Regression Classifier...")
print("Mean Accuracy score : " + str(mean_lr_accuracy))
print("Mean Precision score : " + str(mean_lr_precision))
print("Mean Recall score : " + str(mean_lr_recall))
print("Mean accuracy DOS attacks : " + str(np.mean(lr_dos_accuracy)))
print("Mean accuracy Probe attacks : " + str(np.mean(lr_probe_accuracy)))
print("Mean accuracy R2L attacks : " + str(np.mean(lr_r2l_accuracy)))
print("Mean accuracy U2R attacks : " + str(np.mean(lr_u2r_accuracy)))
print("Mean accuracy Normal class : " + str(np.mean(lr_normal_accuracy)))


# In[ ]:


# graphical comparison

n_groups = 4
scores = [np.mean(lr_dos_accuracy), np.mean(lr_probe_accuracy), np.mean(lr_r2l_accuracy), np.mean(lr_u2r_accuracy)]
scores = [item-0.90 for item in scores]

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.30
opacity = 0.8
 
rects = plt.bar(index, scores, bar_width, alpha = opacity, align = 'center', label = 'Average Accuracy')

rows = ['DoS', 'Probe', 'R2L', 'U2R']

plt.xlabel('Attack Category')
plt.ylabel('Average Accuracy Score')
plt.title('Accuracy scores for different attack categories using LR Classifier')
plt.xticks(index, rows)
plt.yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10], [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00])
plt.legend()

fig = plt.tight_layout(rect = (0, 0, 1.4, 1.4))
plt.show()


# In[ ]:


# SGD classifier

def getSGDClf():
    clf = SGDClassifier(max_iter = 1000, tol = 1e-3, learning_rate = 'optimal')
    return clf


# In[ ]:


# training on SGD classifier

sgd_accuracy = []
sgd_precision = []
sgd_recall = []
sgd_tn = []
sgd_fp = []
sgd_fn = []
sgd_tp = []
sgd_dos_accuracy = []
sgd_probe_accuracy = []
sgd_r2l_accuracy = []
sgd_u2r_accuracy = []
sgd_normal_accuracy = []

for attack_type in attack_classes:
    clf = getSGDClf()
    clf.fit(x_train, train_label[attack_type])
    y_pred = clf.predict(x_test)
    stacked_train_df[attack_type]['sgd'] = clf.predict(x_train)
    stacked_test_df[attack_type]['sgd'] = y_pred
    sgd_accuracy += [accuracy_score(test_label[attack_type], y_pred)]
    if attack_type in dos_classes:
        sgd_dos_accuracy += [sgd_accuracy[-1]]
    if attack_type in probe_classes:
        sgd_probe_accuracy += [sgd_accuracy[-1]]
    if attack_type in r2l_classes:
        sgd_r2l_accuracy += [sgd_accuracy[-1]]
    if attack_type in u2r_classes:
        sgd_u2r_accuracy += [sgd_accuracy[-1]]
    if attack_type in normal_classes:
        sgd_normal_accuracy += [sgd_accuracy[-1]]
    sgd_precision += [precision_score(test_label[attack_type], y_pred)]
    sgd_recall += [recall_score(test_label[attack_type], y_pred)]
    cm = confusion_matrix(test_label[attack_type], y_pred).ravel()
    if len(cm) > 1:
        sgd_tn += [cm[0]]
        sgd_fp += [cm[1]]
        sgd_fn += [cm[2]]
        sgd_tp += [cm[3]]
    else:
        sgd_tn += [0]
        sgd_fp += [0]
        sgd_fn += [0]
        sgd_tp += [0]
    
mean_sgd_accuracy = np.mean(sgd_accuracy)
mean_sgd_precision = np.mean(sgd_precision)
mean_sgd_recall = np.mean(sgd_recall)
    
print("SGD Classifier...")
print("Mean Accuracy score : " + str(mean_sgd_accuracy))
print("Mean Precision score : " + str(mean_sgd_precision))
print("Mean Recall score : " + str(mean_sgd_recall))
print("Mean accuracy DOS attacks : " + str(np.mean(sgd_dos_accuracy)))
print("Mean accuracy Probe attacks : " + str(np.mean(sgd_probe_accuracy)))
print("Mean accuracy R2L attacks : " + str(np.mean(sgd_r2l_accuracy)))
print("Mean accuracy U2R attacks : " + str(np.mean(sgd_u2r_accuracy)))
print("Mean accuracy Normal class : " + str(np.mean(sgd_normal_accuracy)))


# In[ ]:


# graphical comparison

n_groups = 4
scores = [np.mean(sgd_dos_accuracy), np.mean(sgd_probe_accuracy), np.mean(sgd_r2l_accuracy), np.mean(sgd_u2r_accuracy)]
scores = [item-0.90 for item in scores]

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.30
opacity = 0.8
 
rects = plt.bar(index, scores, bar_width, alpha = opacity, align = 'center', label = 'Average Accuracy')

rows = ['DoS', 'Probe', 'R2L', 'U2R']

plt.xlabel('Attack Category')
plt.ylabel('Average Accuracy Score')
plt.title('Accuracy scores for different attack categories using SGD Classifier')
plt.xticks(index, rows)
plt.yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10], [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00])
plt.legend()

fig = plt.tight_layout(rect = (0, 0, 1.4, 1.4))
plt.show()


# <div><h3>Gradient Boosting Decision Tree (GBDT) Ensemble Models</h3></div>

# In[ ]:


# lgbm classifier

import lightgbm as lgb

def getlgbclf(d_train, valid_sets):
    params = {'learning_rate': 0.2, 'application': 'binary', 'num_leaves': 31, 'verbosity': -1,
          'bagging_fraction': 0.8, 'feature_fraction': 0.6, 'nthread': 4, 'lambda_l1': 1, 'lambda_l2': 1}
    
    clf = lgb.train(params, train_set = d_train, num_boost_round = 300, early_stopping_rounds = 100,
                    valid_sets = valid_sets, verbose_eval = False)   
    
    return clf


# In[ ]:


# training on lgbm classifier

lgb_accuracy = []
lgb_precision = []
lgb_recall = []
lgb_tn = []
lgb_fp = []
lgb_fn = []
lgb_tp = []
lgb_dos_accuracy = []
lgb_probe_accuracy = []
lgb_r2l_accuracy = []
lgb_u2r_accuracy = []
lgb_normal_accuracy = []

for attack_type in attack_classes:
    d_train = lgb.Dataset(x_train, label = train_label[attack_type])
    d_test = lgb.Dataset(x_test, label = test_label[attack_type])
    valid_sets = [d_train, d_test]
    clf = getlgbclf(d_train, valid_sets)
    y_pred = (clf.predict(x_test) >= 0.5).astype(int)
    stacked_train_df[attack_type]['lgbm'] = (clf.predict(x_train) >= 0.5).astype(int)
    stacked_test_df[attack_type]['lgbm'] = y_pred
    lgb_accuracy += [accuracy_score(test_label[attack_type], y_pred)]
    if attack_type in dos_classes:
        lgb_dos_accuracy += [lgb_accuracy[-1]]
    if attack_type in probe_classes:
        lgb_probe_accuracy += [lgb_accuracy[-1]]
    if attack_type in r2l_classes:
        lgb_r2l_accuracy += [lgb_accuracy[-1]]
    if attack_type in u2r_classes:
        lgb_u2r_accuracy += [lgb_accuracy[-1]]
    if attack_type in normal_classes:
        lgb_normal_accuracy += [lgb_accuracy[-1]]
    lgb_precision += [precision_score(test_label[attack_type], y_pred)]
    lgb_recall += [recall_score(test_label[attack_type], y_pred)]
    cm = confusion_matrix(test_label[attack_type], y_pred).ravel()
    if len(cm) > 1:
        lgb_tn += [cm[0]]
        lgb_fp += [cm[1]]
        lgb_fn += [cm[2]]
        lgb_tp += [cm[3]]
    else:
        lgb_tn += [0]
        lgb_fp += [0]
        lgb_fn += [0]
        lgb_tp += [0]
    
mean_lgb_accuracy = np.mean(lgb_accuracy)
mean_lgb_precision = np.mean(lgb_precision)
mean_lgb_recall = np.mean(lgb_recall)
    
print("LGBM Classifier...")
print("Mean Accuracy score : " + str(mean_lgb_accuracy))
print("Mean Precision score : " + str(mean_lgb_precision))
print("Mean Recall score : " + str(mean_lgb_recall))
print("Mean accuracy DOS attacks : " + str(np.mean(lgb_dos_accuracy)))
print("Mean accuracy Probe attacks : " + str(np.mean(lgb_probe_accuracy)))
print("Mean accuracy R2L attacks : " + str(np.mean(lgb_r2l_accuracy)))
print("Mean accuracy U2R attacks : " + str(np.mean(lgb_u2r_accuracy)))
print("Mean accuracy Normal class : " + str(np.mean(lgb_normal_accuracy)))


# In[ ]:


# graphical comparison

n_groups = 4
scores = [np.mean(lgb_dos_accuracy), np.mean(lgb_probe_accuracy), np.mean(lgb_r2l_accuracy), np.mean(lgb_u2r_accuracy)]
scores = [item-0.90 for item in scores]

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.30
opacity = 0.8
 
rects = plt.bar(index, scores, bar_width, alpha = opacity, align = 'center', label = 'Average Accuracy')

rows = ['DoS', 'Probe', 'R2L', 'U2R']

plt.xlabel('Attack Category')
plt.ylabel('Average Accuracy Score')
plt.title('Accuracy scores for different attack categories using LGBM Classifier')
plt.xticks(index, rows)
plt.yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10], [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00])
plt.legend()

fig = plt.tight_layout(rect = (0, 0, 1.4, 1.4))
plt.show()


# In[ ]:


# XGBoost classifier

import xgboost as xgb

def getxgbclf(d_train, eval_list):
    params = {'booster' : 'gbtree', 'nthread' : 4, 'eta' : 0.2, 'max_depth' : 6, 'min_child_weight' : 4,
          'subsample' : 0.7, 'colsample_bytree' : 0.7, 'objective' : 'binary:logistic'}

    clf = xgb.train(params, d_train, num_boost_round = 300, early_stopping_rounds = 100, 
                    evals = evallist, verbose_eval = False)
    return clf


# In[ ]:


# training on XGBoost classifier

xgb_accuracy = []
xgb_precision = []
xgb_recall = []
xgb_tn = []
xgb_fp = []
xgb_fn = []
xgb_tp = []
xgb_dos_accuracy = []
xgb_probe_accuracy = []
xgb_r2l_accuracy = []
xgb_u2r_accuracy = []
xgb_normal_accuracy = []

for attack_type in attack_classes:
    d_train = xgb.DMatrix(x_train, label = train_label[attack_type])
    d_test = xgb.DMatrix(x_test, label = test_label[attack_type])
    evallist = [(d_train, 'train'), (d_test, 'valid')]
    clf = getxgbclf(d_train, evallist)
    y_pred = (clf.predict(d_test) >= 0.5).astype(int)
    stacked_train_df[attack_type]['xgb'] = (clf.predict(d_train) >= 0.5).astype(int)
    stacked_test_df[attack_type]['xgb'] = y_pred
    xgb_accuracy += [accuracy_score(test_label[attack_type], y_pred)]
    if attack_type in dos_classes:
        xgb_dos_accuracy += [xgb_accuracy[-1]]
    if attack_type in probe_classes:
        xgb_probe_accuracy += [xgb_accuracy[-1]]
    if attack_type in r2l_classes:
        xgb_r2l_accuracy += [xgb_accuracy[-1]]
    if attack_type in u2r_classes:
        xgb_u2r_accuracy += [xgb_accuracy[-1]]
    if attack_type in normal_classes:
        xgb_normal_accuracy += [xgb_accuracy[-1]]
    xgb_precision += [precision_score(test_label[attack_type], y_pred)]
    xgb_recall += [recall_score(test_label[attack_type], y_pred)]
    cm = confusion_matrix(test_label[attack_type], y_pred).ravel()
    if len(cm) > 1:
        xgb_tn += [cm[0]]
        xgb_fp += [cm[1]]
        xgb_fn += [cm[2]]
        xgb_tp += [cm[3]]
    else:
        xgb_tn += [0]
        xgb_fp += [0]
        xgb_fn += [0]
        xgb_tp += [0]
    
mean_xgb_accuracy = np.mean(xgb_accuracy)
mean_xgb_precision = np.mean(xgb_precision)
mean_xgb_recall = np.mean(xgb_recall)
    
print("XGBoost Classifier...")
print("Mean Accuracy score : " + str(mean_xgb_accuracy))
print("Mean Precision score : " + str(mean_xgb_precision))
print("Mean Recall score : " + str(mean_xgb_recall))
print("Mean accuracy DOS attacks : " + str(np.mean(xgb_dos_accuracy)))
print("Mean accuracy Probe attacks : " + str(np.mean(xgb_probe_accuracy)))
print("Mean accuracy R2L attacks : " + str(np.mean(xgb_r2l_accuracy)))
print("Mean accuracy U2R attacks : " + str(np.mean(xgb_u2r_accuracy)))
print("Mean accuracy Normal class : " + str(np.mean(xgb_normal_accuracy)))


# In[ ]:


# graphical comparison

n_groups = 4
scores = [np.mean(xgb_dos_accuracy), np.mean(xgb_probe_accuracy), np.mean(xgb_r2l_accuracy), np.mean(xgb_u2r_accuracy)]
scores = [item-0.90 for item in scores]

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.30
opacity = 0.8
 
rects = plt.bar(index, scores, bar_width, alpha = opacity, align = 'center', label = 'Average Accuracy')

rows = ['DoS', 'Probe', 'R2L', 'U2R']

plt.xlabel('Attack Category')
plt.ylabel('Average Accuracy Score')
plt.title('Accuracy scores for different attack categories using XGBoost Classifier')
plt.xticks(index, rows)
plt.yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10], [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00])
plt.legend()

fig = plt.tight_layout(rect = (0, 0, 1.4, 1.4))
plt.show()


# <div><h3>Deep Learning Model</h3></div>

# In[ ]:


# Deep Neural Network classifier

def getdnnclf():
    clf = Sequential()
    clf.add(Dense(1024, input_dim = 41, activation = 'relu'))
    clf.add(BatchNormalization())
    clf.add(Dense(1024, activation = 'relu'))
    clf.add(BatchNormalization())
    clf.add(Dense(512, activation = 'relu'))
    clf.add(Dense(64, activation = 'relu'))
    clf.add(Dense(1, activation = 'sigmoid'))
    clf.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return clf


# In[ ]:


# training on DNN classifier

dnn_accuracy = []
dnn_precision = []
dnn_recall = []
dnn_tn = []
dnn_fp = []
dnn_fn = []
dnn_tp = []
dnn_dos_accuracy = []
dnn_probe_accuracy = []
dnn_r2l_accuracy = []
dnn_u2r_accuracy = []
dnn_normal_accuracy = []

for attack_type in attack_classes:
    clf = getdnnclf()
    clf.fit(x_train, train_label[attack_type], batch_size = 1024, epochs = 5, 
            validation_data = (x_test, test_label[attack_type]), verbose = 0)
    y_pred = (clf.predict(x_test) >= 0.5).astype(int)
    stacked_train_df[attack_type]['dnn'] = (clf.predict(x_train) >= 0.5).astype(int)
    stacked_test_df[attack_type]['dnn'] = y_pred
    dnn_accuracy += [accuracy_score(test_label[attack_type], y_pred)]
    if attack_type in dos_classes:
        dnn_dos_accuracy += [dnn_accuracy[-1]]
    if attack_type in probe_classes:
        dnn_probe_accuracy += [dnn_accuracy[-1]]
    if attack_type in r2l_classes:
        dnn_r2l_accuracy += [dnn_accuracy[-1]]
    if attack_type in u2r_classes:
        dnn_u2r_accuracy += [dnn_accuracy[-1]]
    if attack_type in normal_classes:
        dnn_normal_accuracy += [dnn_accuracy[-1]]
    dnn_precision += [precision_score(test_label[attack_type], y_pred)]
    dnn_recall += [recall_score(test_label[attack_type], y_pred)]
    cm = confusion_matrix(test_label[attack_type], y_pred).ravel()
    if len(cm) > 1:
        dnn_tn += [cm[0]]
        dnn_fp += [cm[1]]
        dnn_fn += [cm[2]]
        dnn_tp += [cm[3]]
    else:
        dnn_tn += [0]
        dnn_fp += [0]
        dnn_fn += [0]
        dnn_tp += [0]
    
mean_dnn_accuracy = np.mean(dnn_accuracy)
mean_dnn_precision = np.mean(dnn_precision)
mean_dnn_recall = np.mean(dnn_recall)
    
print("Deep Neural Network Classifier...")
print("Mean Accuracy score : " + str(mean_dnn_accuracy))
print("Mean Precision score : " + str(mean_dnn_precision))
print("Mean Recall score : " + str(mean_dnn_recall))
print("Mean accuracy DOS attacks : " + str(np.mean(dnn_dos_accuracy)))
print("Mean accuracy Probe attacks : " + str(np.mean(dnn_probe_accuracy)))
print("Mean accuracy R2L attacks : " + str(np.mean(dnn_r2l_accuracy)))
print("Mean accuracy U2R attacks : " + str(np.mean(dnn_u2r_accuracy)))
print("Mean accuracy Normal class : " + str(np.mean(dnn_normal_accuracy)))


# In[ ]:


# graphical comparison

n_groups = 4
scores = [np.mean(dnn_dos_accuracy), np.mean(dnn_probe_accuracy), np.mean(dnn_r2l_accuracy), np.mean(dnn_u2r_accuracy)]
scores = [item-0.90 for item in scores]

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.30
opacity = 0.8
 
rects = plt.bar(index, scores, bar_width, alpha = opacity, align = 'center', label = 'Average Accuracy')

rows = ['DoS', 'Probe', 'R2L', 'U2R']

plt.xlabel('Attack Category')
plt.ylabel('Average Accuracy Score')
plt.title('Accuracy scores for different attack categories using DNN Classifier')
plt.xticks(index, rows)
plt.yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10], [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00])
plt.legend()

fig = plt.tight_layout(rect = (0, 0, 1.4, 1.4))
plt.show()


# <div><h3>Stacked Model</h3>
#     <p> Stacked Model contains all the models we have trained above as base learners. We will use a Logistic Regression classifier to combine the outputs of all the base learners to give a single output with greater accuracy.
#     </p>
# </div>

# In[ ]:


# training on stacked classifier

stacked_accuracy = []
stacked_precision = []
stacked_recall = []
stacked_tn = []
stacked_fp = []
stacked_fn = []
stacked_tp = []
stacked_dos_accuracy = []
stacked_probe_accuracy = []
stacked_r2l_accuracy = []
stacked_u2r_accuracy = []
stacked_normal_accuracy = []

for attack_type in attack_classes:
    clf = getLRClf()
    clf.fit(stacked_train_df[attack_type], train_label[attack_type])
    y_pred = clf.predict(stacked_test_df[attack_type])
    stacked_accuracy += [accuracy_score(test_label[attack_type], y_pred)]
    if attack_type in dos_classes:
        stacked_dos_accuracy += [stacked_accuracy[-1]]
    if attack_type in probe_classes:
        stacked_probe_accuracy += [stacked_accuracy[-1]]
    if attack_type in r2l_classes:
        stacked_r2l_accuracy += [stacked_accuracy[-1]]
    if attack_type in u2r_classes:
        stacked_u2r_accuracy += [stacked_accuracy[-1]]
    if attack_type in normal_classes:
        stacked_normal_accuracy += [stacked_accuracy[-1]]
    stacked_precision += [precision_score(test_label[attack_type], y_pred)]
    stacked_recall += [recall_score(test_label[attack_type], y_pred)]
    cm = confusion_matrix(test_label[attack_type], y_pred).ravel()
    if len(cm) > 1:
        stacked_tn += [cm[0]]
        stacked_fp += [cm[1]]
        stacked_fn += [cm[2]]
        stacked_tp += [cm[3]]
    else:
        stacked_tn += [0]
        stacked_fp += [0]
        stacked_fn += [0]
        stacked_tp += [0]
    
mean_stacked_accuracy = np.mean(stacked_accuracy)
mean_stacked_precision = np.mean(stacked_precision)
mean_stacked_recall = np.mean(stacked_recall)
    
print("Stacked Model Classifier...")
print("Mean Accuracy score : " + str(mean_stacked_accuracy))
print("Mean Precision score : " + str(mean_stacked_precision))
print("Mean Recall score : " + str(mean_stacked_recall))
print("Mean accuracy DOS attacks : " + str(np.mean(stacked_dos_accuracy)))
print("Mean accuracy Probe attacks : " + str(np.mean(stacked_probe_accuracy)))
print("Mean accuracy R2L attacks : " + str(np.mean(stacked_r2l_accuracy)))
print("Mean accuracy U2R attacks : " + str(np.mean(stacked_u2r_accuracy)))
print("Mean accuracy Normal class : " + str(np.mean(stacked_normal_accuracy)))


# In[ ]:


# graphical comparison

n_groups = 4
scores = [np.mean(stacked_dos_accuracy), np.mean(stacked_probe_accuracy), np.mean(stacked_r2l_accuracy), np.mean(stacked_u2r_accuracy)]
scores = [item-0.90 for item in scores]

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.30
opacity = 0.8
 
rects = plt.bar(index, scores, bar_width, alpha = opacity, align = 'center', label = 'Average Accuracy')

rows = ['DoS', 'Probe', 'R2L', 'U2R']

plt.xlabel('Attack Category')
plt.ylabel('Average Accuracy Score')
plt.title('Accuracy scores for different attack categories using Stacked Model Classifier')
plt.xticks(index, rows)
plt.yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10], [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00])
plt.legend()

fig = plt.tight_layout(rect = (0, 0, 1.4, 1.4))
plt.show()


# In[ ]:


# models comparison

columns = ['Average Accuracy Score', 'Average Precision Score', 'Average Recall Score']
rows = ['Log Reg', 'SGD', 'LGBM', 'XGBoost', 'DNN', 'Stacked Model']
lr_scores = [mean_lr_accuracy, mean_lr_precision, mean_lr_recall]
sgd_scores = [mean_sgd_accuracy, mean_sgd_precision, mean_sgd_recall]
lgb_scores = [mean_lgb_accuracy, mean_lgb_precision, mean_lgb_recall]
xgb_scores = [mean_xgb_accuracy, mean_xgb_precision, mean_xgb_recall]
dnn_scores = [mean_dnn_accuracy, mean_dnn_precision, mean_dnn_recall]
stacked_scores = [mean_stacked_accuracy, mean_stacked_precision, mean_stacked_recall]

table = pd.DataFrame(data = [lr_scores, sgd_scores, lgb_scores, xgb_scores, dnn_scores, stacked_scores], columns = columns, index = rows)
print(table)


# In[ ]:


acc_df = pd.DataFrame()

acc_df['attack_type'] = attack_classes
acc_df['lr_accuracy_scores'] = lr_accuracy

print(acc_df)


# In[ ]:


acc_df.drop(['lr_accuracy_scores'], axis = 1, inplace = True)
acc_df['sgd_accuracy_scores'] = sgd_accuracy

print(acc_df)


# In[ ]:


acc_df.drop(['sgd_accuracy_scores'], axis = 1, inplace = True)
acc_df['lgb_accuracy_scores'] = lgb_accuracy

print(acc_df)


# In[ ]:


acc_df.drop(['lgb_accuracy_scores'], axis = 1, inplace = True)
acc_df['xgb_accuracy_scores'] = xgb_accuracy

print(acc_df)


# In[ ]:


acc_df.drop(['xgb_accuracy_scores'], axis = 1, inplace = True)
acc_df['dnn_accuracy_scores'] = dnn_accuracy

print(acc_df)


# In[ ]:


acc_df.drop(['dnn_accuracy_scores'], axis = 1, inplace = True)
acc_df['stacked_accuracy_scores'] = stacked_accuracy

print(acc_df)


# In[ ]:


prec_df = pd.DataFrame()

prec_df['attack_type'] = attack_classes
prec_df['lr_precision_scores'] = lr_precision

print(prec_df)


# In[ ]:


prec_df.drop(['lr_precision_scores'], axis = 1, inplace = True)
prec_df['sgd_precision_scores'] = sgd_precision

print(prec_df)


# In[ ]:


prec_df.drop(['sgd_precision_scores'], axis = 1, inplace = True)
prec_df['lgb_precision_scores'] = lgb_precision

print(prec_df)


# In[ ]:


prec_df.drop(['lgb_precision_scores'], axis = 1, inplace = True)
prec_df['xgb_precision_scores'] = xgb_precision

print(prec_df)


# In[ ]:


prec_df.drop(['xgb_precision_scores'], axis = 1, inplace = True)
prec_df['dnn_precision_scores'] = dnn_precision

print(prec_df)


# In[ ]:


prec_df.drop(['dnn_precision_scores'], axis = 1, inplace = True)
prec_df['stacked_precision_scores'] = stacked_precision

print(prec_df)


# In[ ]:


rec_df = pd.DataFrame()

rec_df['attack_type'] = attack_classes
rec_df['lr_recall_scores'] = lr_recall

print(rec_df)


# In[ ]:


rec_df.drop(['lr_recall_scores'], axis = 1, inplace = True)
rec_df['sgd_recall_scores'] = sgd_recall

print(rec_df)


# In[ ]:


rec_df.drop(['sgd_recall_scores'], axis = 1, inplace = True)
rec_df['lgb_recall_scores'] = lgb_recall

print(rec_df)


# In[ ]:


rec_df.drop(['lgb_recall_scores'], axis = 1, inplace = True)
rec_df['xgb_recall_scores'] = xgb_recall

print(rec_df)


# In[ ]:


rec_df.drop(['xgb_recall_scores'], axis = 1, inplace = True)
rec_df['dnn_recall_scores'] = dnn_recall

print(rec_df)


# In[ ]:


rec_df.drop(['dnn_recall_scores'], axis = 1, inplace = True)
rec_df['stacked_recall_scores'] = stacked_recall

print(rec_df)


# In[ ]:


cf_df = pd.DataFrame()

cf_df['attack_type'] = attack_classes
cf_df['lr_tn'] = lr_tn
cf_df['lr_fp'] = lr_fp
cf_df['lr_fn'] = lr_fn
cf_df['lr_tp'] = lr_tp

print('Total samples : ' + str(x_train.shape[0]))
print(cf_df)


# In[ ]:


cf_df.drop(['lr_tn', 'lr_fp', 'lr_fn', 'lr_tp'], axis = 1, inplace = True)

cf_df['sgd_tn'] = sgd_tn
cf_df['sgd_fp'] = sgd_fp
cf_df['sgd_fn'] = sgd_fn
cf_df['sgd_tp'] = sgd_tp

print('Total samples : ' + str(x_train.shape[0]))
print(cf_df)


# In[ ]:


cf_df.drop(['sgd_tn', 'sgd_fp', 'sgd_fn', 'sgd_tp'], axis = 1, inplace = True)

cf_df['lgb_tn'] = lgb_tn
cf_df['lgb_fp'] = lgb_fp
cf_df['lgb_fn'] = lgb_fn
cf_df['lgb_tp'] = lgb_tp

print('Total samples : ' + str(x_train.shape[0]))
print(cf_df)


# In[ ]:


cf_df.drop(['lgb_tn', 'lgb_fp', 'lgb_fn', 'lgb_tp'], axis = 1, inplace = True)

cf_df['xgb_tn'] = xgb_tn
cf_df['xgb_fp'] = xgb_fp
cf_df['xgb_fn'] = xgb_fn
cf_df['xgb_tp'] = xgb_tp

print('Total samples : ' + str(x_train.shape[0]))
print(cf_df)


# In[ ]:


cf_df.drop(['xgb_tn', 'xgb_fp', 'xgb_fn', 'xgb_tp'], axis = 1, inplace = True)

cf_df['dnn_tn'] = dnn_tn
cf_df['dnn_fp'] = dnn_fp
cf_df['dnn_fn'] = dnn_fn
cf_df['dnn_tp'] = dnn_tp

print('Total samples : ' + str(x_train.shape[0]))
print(cf_df)


# In[ ]:


cf_df.drop(['dnn_tn', 'dnn_fp', 'dnn_fn', 'dnn_tp'], axis = 1, inplace = True)

cf_df['stacked_tn'] = stacked_tn
cf_df['stacked_fp'] = stacked_fp
cf_df['stacked_fn'] = stacked_fn
cf_df['stacked_tp'] = stacked_tp

print('Total samples : ' + str(x_train.shape[0]))
print(cf_df)


# <div><h3>Comparing Performances of all the models trained</h3></div>

# In[ ]:


acc_scores = [[mean_lr_accuracy], [mean_sgd_accuracy], [mean_lgb_accuracy], [mean_xgb_accuracy], 
          [mean_dnn_accuracy], [mean_stacked_accuracy]]
prec_scores = [[mean_lr_precision], [mean_sgd_precision], [mean_lgb_precision], [mean_xgb_precision], 
          [mean_dnn_precision], [mean_stacked_precision]]
rec_scores = [[mean_lr_recall], [mean_sgd_recall], [mean_lgb_recall], [mean_xgb_recall], 
          [mean_dnn_recall], [mean_stacked_recall]]


# In[ ]:


# graphical comparison

n_groups = 6
acc = [item[0]-0.80 for item in acc_scores]

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.30
opacity = 0.8
 
rects = plt.bar(index, acc, bar_width, alpha = opacity, align = 'center', label = 'Average Accuracy')

plt.xlabel('Model')
plt.ylabel('Average Accuracy Score')
plt.title('Graphical Comparison of Average Accuracy for all models')
plt.xticks(index, rows)
plt.yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12,
           0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20], 
           [0.80, 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 
            0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00])
plt.legend()

fig = plt.tight_layout(rect = (0, 0, 1.4, 1.4))
plt.show()


# In[ ]:


# graphical comparison

n_groups = 6
prec = [item[0] for item in prec_scores]

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.30
opacity = 0.8
 
rects = plt.bar(index, prec, bar_width, alpha = opacity, align = 'center', label = 'Average Precision')

plt.xlabel('Model')
plt.ylabel('Average Precision Score')
plt.title('Graphical Comparison of Average Precision for all models')
plt.xticks(index, rows)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.legend()

fig = plt.tight_layout(rect = (0, 0, 1.4, 1.4))
plt.show()


# In[ ]:


# graphical comparison

n_groups = 6
rec = [item[0] for item in rec_scores]

fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.30
opacity = 0.8
 
rects = plt.bar(index, rec, bar_width, alpha = opacity, align = 'center', label = 'Average Recall')

plt.xlabel('Model')
plt.ylabel('Average Recall Score')
plt.title('Graphical Comparison of Average Recall for all models')
plt.xticks(index, rows)
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.legend()

fig = plt.tight_layout(rect = (0, 0, 1.4, 1.4))
plt.show()


# In[ ]:




