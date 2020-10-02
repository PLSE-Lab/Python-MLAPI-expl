#!/usr/bin/env python
# coding: utf-8

# **Load packages**

# In[ ]:


import seaborn as sns
import pandas as pd # (e.g. pd.read_csv)
import matplotlib.pylab as plt
import numpy as np # linear algebra
import warnings
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
import lightgbm as lgb
import gc, datetime, random
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

#from ml import simple

warnings.simplefilter("ignore")
plt.style.use('ggplot')
color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]


# **Load Data**

# In[ ]:


transaction = pd.read_csv('../input/anomaly-detection/creditcard.csv')


# In[ ]:


print('transaction shape is {}'.format(transaction.shape))
transaction = transaction.sort_values("Time")
transaction.head(20)


# **Visualize Transaction Distribution by `Amount` during `Time`**

# In[ ]:


# plot all transaction
sns.lmplot( x="Time", y="Amount", data=transaction, fit_reg=False, hue='Class', height=8, aspect=17/8.27)
plt.title("Transaction Amount during Time")

# plot only fraudulent transaction
transaction[(transaction['Class'] == 1)].plot(x='Time', y='Amount', style='.', figsize=(15, 3), label='Fraudulent Transaction', color = 'orange')


# The fraudulent transaction occur mainly on small  `Amount` which do not exceed 2000.
# It seems no specific Time that we show a cluster of fraudes. Maybe near to Time : 40000.

# **Recognize categorical and numerical Variables**

# In[ ]:


transaction.info()


# **Add Index to each transaction**

# In[ ]:


transaction = transaction.reset_index()
transaction.head(20)


# ** UMBALANCED Proportion of classes**

# In[ ]:


total = len(transaction)
plt.figure(figsize=(12,5))
plt.subplot(121)
plot_tr = sns.countplot(x='Class', data=transaction)
plot_tr.set_title("Fraud Transactions Distribution \n 0: No Fraud | 1: Fraud", fontsize=18)
plot_tr.set_xlabel("Is fraud?", fontsize=16)
plot_tr.set_ylabel('Count', fontsize=16)
for p in plot_tr.patches:
    height = p.get_height()
    plot_tr.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/total*100),
            ha="center", fontsize=15) 


# **Summary classes**

# In[ ]:


transaction.groupby('Class').size()


# As expected, we have a small ratio of fraud case compared to normal transaction. Most supervised machine learning classification algorithms are sensitive to unbalance in the predictor classes, and special techniques would have to be used to account for this unbalance (Under-sampling the normal transaction, Over-sampling the frand transaction). but now we will continu we our raw data.

# **Check for missing Data**

# In[ ]:


data_null = transaction.isnull().sum()/len(transaction) * 100
data_null


# Good! seems to be clean.

# **Explore the must important Variables**

# In[ ]:


np.warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize':(15,50)})
for num, alpha in enumerate(list(transaction.drop(columns =['index', 'Class'], axis = 1).columns)):
    plt.subplot(10,3,num+1)
    yes = transaction[(transaction['Class'] == 1)][alpha]
    no = transaction[(transaction['Class'] == 0) ][alpha]
    plt.hist(yes[yes>0], alpha=0.75, label='Fraud', color='r')
    plt.hist(no[no>0], alpha=0.25, label='Not Fraud', color='g')
    plt.ylim(0, 1000)
    plt.legend(loc='upper right')
    plt.ylabel("Amount is limit to 1000 (250000)")
    plt.title('Histogram of values  in column ' + str(alpha) )
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
   


# We can delete some non importante column to improve model. Time, V3, V6, V7, V9, V10, V12, V14, V16 seem to be not important.

# **Split Training and Testing Data**

# In[ ]:


#we are going to divide 70/30 the data into training and testing set proportion
train,test=train_test_split(transaction,test_size=0.3)


# In[ ]:


print(train.shape)
print(test.shape)


# **Visualize distribution of trainning and testing datasets**

# In[ ]:


sns.lmplot( x="Time", y="Amount", data=train, fit_reg=False, hue='Class', height=8, aspect=17/8.27)
plt.title("Training distribution")
train.groupby('Class').size()


# In[ ]:


sns.lmplot( x="Time", y="Amount", data=test, fit_reg=False, hue='Class', height=8, aspect=17/8.27)
plt.title("Testing distribution")
test.groupby('Class').size()


# **Preprocessing**

# In[ ]:


# Extract xTrain
xTrain = train.drop(columns= ['index', 'Class'])
# Extract xTest
xTest = test.drop(columns= ['index', 'Class'])
# Extract yTrain
yTrain = train[('Class')]
# Extract y Test
yTest = test[('Class')]

xTrain.sort_values('Time')


# In[ ]:


print('xTrain:', xTrain.shape)
print('yTrain:', yTrain.shape)
print('xTest:', xTest.shape)
print('yTest:', yTest.shape)


# **LGBM**

# In[ ]:


def makePredictions(train, test, target, lgb_params, NFOLDS=6):
    folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

    X,y = train, target   
    P = test

    predictions = np.zeros(len(test))
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print('Fold:',fold_)
        tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]
            
        print(len(tr_x),len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)

        vl_data = lgb.Dataset(vl_x, label=vl_y)  

        estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets = [tr_data, vl_data],
            verbose_eval = 200,
        )   
        
        pp_p = estimator.predict(P)
        predictions += pp_p/NFOLDS
        
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()
        
    test['prediction'] = predictions
    
    return test


# In[ ]:


def make_predictions(train, test, features_columns, target, lgb_params, NFOLDS=2):
    
    folds = GroupKFold(n_splits=NFOLDS)

    X,y = train[features_columns], train[target]    
    P,P_y = test[features_columns], test[target]  
    split_groups =  train['Time']

    test = test[['index',target]]    
    predictions = np.zeros(len(test))
    oof = np.zeros(len(train))
    
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_groups)):
        print('Fold:',fold_)
        tr_x, tr_y = X.iloc[trn_idx,:], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx,:], y[val_idx]
            
        print(len(tr_x),len(vl_x))
        tr_data = lgb.Dataset(tr_x, label=tr_y)
        vl_data = lgb.Dataset(vl_x, label=vl_y)  

        estimator = lgb.train(
            lgb_params,
            tr_data,
            valid_sets = [tr_data, vl_data],
            verbose_eval = 200,
        )   
        
        pp_p = estimator.predict(P)
        predictions += pp_p/NFOLDS
        
        oof_preds = estimator.predict(vl_x)
        oof[val_idx] = (oof_preds - oof_preds.min())/(oof_preds.max() - oof_preds.min())

        if LOCAL_TEST:
            feature_imp = pd.DataFrame(sorted(zip(estimator.feature_importance(),X.columns)), columns=['Value','Feature'])
            print(feature_imp)
        
        del tr_x, tr_y, vl_x, vl_y, tr_data, vl_data
        gc.collect()
        
    test['prediction'] = predictions
    print('OOF AUC:', metrics.roc_auc_score(y, oof))
    if LOCAL_TEST:
        print('Holdout AUC:', metrics.roc_auc_score(test[TARGET], test['prediction']))
    
    return test


# In[ ]:


lgb_params3 = {
                'objective':'binary',
                'boosting_type':'gbdt',
                'metric':'auc',
                'n_jobs':-1,
                'learning_rate':0.007,
                'num_leaves': 2**8,
                'max_depth':-1,
                'tree_learner':'serial',
                'colsample_bytree': 0.85,
                'subsample_freq':1,
                'subsample':0.85,
                'n_estimators':1800,
                'max_bin':255,
                'verbose':-1,
                'seed': 42,
                #'early_stopping_rounds':100,
                'reg_alpha':0.3,
                'reg_lamdba':0.243
            } 


# In[ ]:


best_params = {'objective':'binary',
               'boosting_type':'gbdt',
               'metric':'auc',
               'reg_lambda': 0.4,
               'reg_alpha': 0.30000000000000004,
               'num_leaves': 500,
               'min_data_in_leaf': 120,
               'learning_rate': 0.05,
               'feature_fraction': 0.4,
               'bagging_fraction': 0.2,
               'class_weight':None,
               'colsample_bytree':1.0,
               'importance_type':'split',
               'max_depth':-1,
               'min_child_samples':20,
               'min_child_weight':0.001,
               'min_split_gain':0.0,
               'n_estimators':1800, 
               'n_jobs':-1,
               'num_leaves':31,
               'learning_rate': 0.05,
               'pre_dispatch':'2*n_jobs',
               'random_state':None, 
               'refit':True,
               'return_train_score':False, 
               'scoring':'roc_auc',
               'tree_learner':'serial',
               'seed': 42,
               'verbose': -1}


# In[ ]:


variables = list(train.drop(columns= ['index', 'Class']).columns)
variables


# In[ ]:


list(test.columns)
#test = test.drop(columns= ['index', 'Class'])
test.head()


# In[ ]:


TARGET = 'Class'
LOCAL_TEST = True
predictions_1 = make_predictions(train, test, variables ,TARGET, best_params, NFOLDS=6)


# In[ ]:


predictions_2 = makePredictions(xTrain, xTest, yTrain, best_params, NFOLDS=6)


# In[ ]:


Mat1 = pd.DataFrame({"Class":yTest, "Prediction": predictions_1['prediction']})
Mat2 = pd.DataFrame({"Class":yTest, "Prediction": predictions_2['prediction']})
Mat1


# In[ ]:


Mat2


# **Confusion Marix**

# In[ ]:


def plot_conf_Mat(Matrix):
# Creates a confusion matrix
    cm = confusion_matrix(Matrix['Class'].astype(np.int64), Matrix['Prediction'].astype(np.int64)) 

# Transform to df for easier plotting
    cm_df = pd.DataFrame(cm,
                     index = ['Class','Prediction'], 
                     columns = ['Class','Prediction'])

    plt.figure(figsize=(5.5,4))
    sns.heatmap(cm, annot=True)
    plt.title('CM \nAccuracy:{0:.3f}'.format(accuracy_score(Matrix['Class'].astype(np.int64), Matrix['Prediction'].astype(np.int64))))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

plot_conf_Mat(Mat1)
plot_conf_Mat(Mat2)


# **WHEN WE REMOVE NO IMPORTANTE VARIABLES**

# In[ ]:


train_drp = train.drop(columns=[ 'V3', 'V6', 'V7', 'V9', 'V10', 'V12', 'V14', 'V16'], axis = 1)
test_drp = train.drop(columns=['V3', 'V6', 'V7', 'V9', 'V10', 'V12', 'V14', 'V16'], axis = 1)


#indices = (0, 2, 5, 6, 8, 9, 11, 13, 15)
#for i in (indices):
#    variables.pop(i)
variables_drp = variables
#variables_drp.remove('Time')  
variables_drp.remove('V3')  
variables_drp.remove('V6')
variables_drp.remove('V7') 
variables_drp.remove('V9')  
variables_drp.remove('V10')
variables_drp.remove('V12')
variables_drp.remove('V14')
variables_drp.remove('V16')  


TARGET = 'Class'
LOCAL_TEST = True
predictions_1 = make_predictions(train_drp, test_drp, variables_drp ,TARGET, best_params, NFOLDS=6)

