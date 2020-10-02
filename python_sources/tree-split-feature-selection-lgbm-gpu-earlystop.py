#!/usr/bin/env python
# coding: utf-8

# # Tree Split Feature Selection - LGBM GPU EarlyStop  
# _By Nick Brooks_
# 
# V1 - 29/07/2019 - First Commit <br>
# V2 - 03/08/2019 - PCA, Metric Convergence, Submission with 21 features <br>
# V3 - 05/08/2019 - Fix GPU implementation <br>
# 
# **Motivation:** <br>
# How much of these features have actual signal? How does predictive power react when the number of features is decreased? What can PCA tell us about the  amount of variance in the features? Does reducing the number of features lead to smoother convergence?
# 
# **Methodology:** <br>
# *Tree-Split Feature Selection* - Experiment with Iteratively removing feature using Gradient Boosting Ensemble split importance. LGBM is trained with a single validation fold and shuffles the train / validation set each iteration.
# 
# Very hyped for the GPU speed boost to Gradient Boosting Algorithmns, this will enable many fun experiments.
# 
# **Other Links:** <Br>
# [ELI5 Permutation Importance](https://eli5.readthedocs.io/en/latest/blackbox/permutation_importance.html)
#     
# **My Other Fraud Notebooks:** <br>
# https://www.kaggle.com/nicapotato/auc-performance-vs-training-size-gpu-catboost <br>
# https://www.kaggle.com/nicapotato/fraud-shap-xgboost <br>

# #### GPU Installation from [kirankunapuli](https://www.kaggle.com/kirankunapuli/)
# Source: https://www.kaggle.com/kirankunapuli/ieee-fraud-lightgbm-with-gpu/comments

# In[ ]:


get_ipython().system('rm -r /opt/conda/lib/python3.6/site-packages/lightgbm')
get_ipython().system('git clone --recursive https://github.com/Microsoft/LightGBM')
get_ipython().system('apt-get install -y -qq libboost-all-dev')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'cd LightGBM\nrm -r build\nmkdir build\ncd build\ncmake -DUSE_GPU=1 -DOpenCL_LIBRARY=/usr/local/cuda/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda/include/ ..\nmake -j$(nproc)')


# In[ ]:


get_ipython().system('cd LightGBM/python-package/;python3 setup.py install --precompile')


# In[ ]:


get_ipython().system('mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd')
get_ipython().system('rm -r LightGBM')


# In[ ]:


# Latest Pandas version
get_ipython().system("pip install -q 'pandas==0.25' --force-reinstall")


# In[ ]:


import lightgbm as lgb
import pandas as pd 
print("LGBM version:", lgb.__version__)
print("Pandas version:", pd.__version__)


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os
import platform
print("Python Version:", platform.python_version())

import warnings
warnings.filterwarnings("ignore")

import time
notebookstart = time.time()

from contextlib import contextmanager
import gc
import pprint

import numpy as np

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA

seed = 24
np.random.seed(seed)

pd.set_option('display.max_columns', 500)
pd.options.display.max_rows = 999
pd.set_option('max_colwidth', 500)

print("LGBM version:", lgb.__version__)


# In[ ]:


print("Define DF Schema..")

target_var = 'isFraud'

schema = {
    "TransactionDT":       "int32",
    "TransactionAmt":    "float32",
    "ProductCD":          "object",
    "card1":               "int16",
    "card2":             "float32",
    "card3":             "float32",
    "card4":              "object",
    "card5":             "float32",
    "card6":              "object",
    "addr1":             "float32",
    "addr2":             "float32",
    "dist1":             "float32",
    "dist2":             "float32",
    "P_emaildomain":      "object",
    "R_emaildomain":      "object",
    "C1":                "float32",
    "C2":                "float32",
    "C3":                "float32",
    "C4":                "float32",
    "C5":                "float32",
    "C6":                "float32",
    "C7":                "float32",
    "C8":                "float32",
    "C9":                "float32",
    "C10":               "float32",
    "C11":               "float32",
    "C12":               "float32",
    "C13":               "float32",
    "C14":               "float32",
    "D1":                "float32",
    "D2":                "float32",
    "D3":                "float32",
    "D4":                "float32",
    "D5":                "float32",
    "D6":                "float32",
    "D7":                "float32",
    "D8":                "float32",
    "D9":                "float32",
    "D10":               "float32",
    "D11":               "float32",
    "D12":               "float32",
    "D13":               "float32",
    "D14":               "float32",
    "D15":               "float32",
    "M1":                 "object",
    "M2":                 "object",
    "M3":                 "object",
    "M4":                 "object",
    "M5":                 "object",
    "M6":                 "object",
    "M7":                 "object",
    "M8":                 "object",
    "M9":                 "object",
    "V1":                "float32",
    "V2":                "float32",
    "V3":                "float32",
    "V4":                "float32",
    "V5":                "float32",
    "V6":                "float32",
    "V7":                "float32",
    "V8":                "float32",
    "V9":                "float32",
    "V10":               "float32",
    "V11":               "float32",
    "V12":               "float32",
    "V13":               "float32",
    "V14":               "float32",
    "V15":               "float32",
    "V16":               "float32",
    "V17":               "float32",
    "V18":               "float32",
    "V19":               "float32",
    "V20":               "float32",
    "V21":               "float32",
    "V22":               "float32",
    "V23":               "float32",
    "V24":               "float32",
    "V25":               "float32",
    "V26":               "float32",
    "V27":               "float32",
    "V28":               "float32",
    "V29":               "float32",
    "V30":               "float32",
    "V31":               "float32",
    "V32":               "float32",
    "V33":               "float32",
    "V34":               "float32",
    "V35":               "float32",
    "V36":               "float32",
    "V37":               "float32",
    "V38":               "float32",
    "V39":               "float32",
    "V40":               "float32",
    "V41":               "float32",
    "V42":               "float32",
    "V43":               "float32",
    "V44":               "float32",
    "V45":               "float32",
    "V46":               "float32",
    "V47":               "float32",
    "V48":               "float32",
    "V49":               "float32",
    "V50":               "float32",
    "V51":               "float32",
    "V52":               "float32",
    "V53":               "float32",
    "V54":               "float32",
    "V55":               "float32",
    "V56":               "float32",
    "V57":               "float32",
    "V58":               "float32",
    "V59":               "float32",
    "V60":               "float32",
    "V61":               "float32",
    "V62":               "float32",
    "V63":               "float32",
    "V64":               "float32",
    "V65":               "float32",
    "V66":               "float32",
    "V67":               "float32",
    "V68":               "float32",
    "V69":               "float32",
    "V70":               "float32",
    "V71":               "float32",
    "V72":               "float32",
    "V73":               "float32",
    "V74":               "float32",
    "V75":               "float32",
    "V76":               "float32",
    "V77":               "float32",
    "V78":               "float32",
    "V79":               "float32",
    "V80":               "float32",
    "V81":               "float32",
    "V82":               "float32",
    "V83":               "float32",
    "V84":               "float32",
    "V85":               "float32",
    "V86":               "float32",
    "V87":               "float32",
    "V88":               "float32",
    "V89":               "float32",
    "V90":               "float32",
    "V91":               "float32",
    "V92":               "float32",
    "V93":               "float32",
    "V94":               "float32",
    "V95":               "float32",
    "V96":               "float32",
    "V97":               "float32",
    "V98":               "float32",
    "V99":               "float32",
    "V100":              "float32",
    "V101":              "float32",
    "V102":              "float32",
    "V103":              "float32",
    "V104":              "float32",
    "V105":              "float32",
    "V106":              "float32",
    "V107":              "float32",
    "V108":              "float32",
    "V109":              "float32",
    "V110":              "float32",
    "V111":              "float32",
    "V112":              "float32",
    "V113":              "float32",
    "V114":              "float32",
    "V115":              "float32",
    "V116":              "float32",
    "V117":              "float32",
    "V118":              "float32",
    "V119":              "float32",
    "V120":              "float32",
    "V121":              "float32",
    "V122":              "float32",
    "V123":              "float32",
    "V124":              "float32",
    "V125":              "float32",
    "V126":              "float32",
    "V127":              "float32",
    "V128":              "float32",
    "V129":              "float32",
    "V130":              "float32",
    "V131":              "float32",
    "V132":              "float32",
    "V133":              "float32",
    "V134":              "float32",
    "V135":              "float32",
    "V136":              "float32",
    "V137":              "float32",
    "V138":              "float32",
    "V139":              "float32",
    "V140":              "float32",
    "V141":              "float32",
    "V142":              "float32",
    "V143":              "float32",
    "V144":              "float32",
    "V145":              "float32",
    "V146":              "float32",
    "V147":              "float32",
    "V148":              "float32",
    "V149":              "float32",
    "V150":              "float32",
    "V151":              "float32",
    "V152":              "float32",
    "V153":              "float32",
    "V154":              "float32",
    "V155":              "float32",
    "V156":              "float32",
    "V157":              "float32",
    "V158":              "float32",
    "V159":              "float32",
    "V160":              "float32",
    "V161":              "float32",
    "V162":              "float32",
    "V163":              "float32",
    "V164":              "float32",
    "V165":              "float32",
    "V166":              "float32",
    "V167":              "float32",
    "V168":              "float32",
    "V169":              "float32",
    "V170":              "float32",
    "V171":              "float32",
    "V172":              "float32",
    "V173":              "float32",
    "V174":              "float32",
    "V175":              "float32",
    "V176":              "float32",
    "V177":              "float32",
    "V178":              "float32",
    "V179":              "float32",
    "V180":              "float32",
    "V181":              "float32",
    "V182":              "float32",
    "V183":              "float32",
    "V184":              "float32",
    "V185":              "float32",
    "V186":              "float32",
    "V187":              "float32",
    "V188":              "float32",
    "V189":              "float32",
    "V190":              "float32",
    "V191":              "float32",
    "V192":              "float32",
    "V193":              "float32",
    "V194":              "float32",
    "V195":              "float32",
    "V196":              "float32",
    "V197":              "float32",
    "V198":              "float32",
    "V199":              "float32",
    "V200":              "float32",
    "V201":              "float32",
    "V202":              "float32",
    "V203":              "float32",
    "V204":              "float32",
    "V205":              "float32",
    "V206":              "float32",
    "V207":              "float32",
    "V208":              "float32",
    "V209":              "float32",
    "V210":              "float32",
    "V211":              "float32",
    "V212":              "float32",
    "V213":              "float32",
    "V214":              "float32",
    "V215":              "float32",
    "V216":              "float32",
    "V217":              "float32",
    "V218":              "float32",
    "V219":              "float32",
    "V220":              "float32",
    "V221":              "float32",
    "V222":              "float32",
    "V223":              "float32",
    "V224":              "float32",
    "V225":              "float32",
    "V226":              "float32",
    "V227":              "float32",
    "V228":              "float32",
    "V229":              "float32",
    "V230":              "float32",
    "V231":              "float32",
    "V232":              "float32",
    "V233":              "float32",
    "V234":              "float32",
    "V235":              "float32",
    "V236":              "float32",
    "V237":              "float32",
    "V238":              "float32",
    "V239":              "float32",
    "V240":              "float32",
    "V241":              "float32",
    "V242":              "float32",
    "V243":              "float32",
    "V244":              "float32",
    "V245":              "float32",
    "V246":              "float32",
    "V247":              "float32",
    "V248":              "float32",
    "V249":              "float32",
    "V250":              "float32",
    "V251":              "float32",
    "V252":              "float32",
    "V253":              "float32",
    "V254":              "float32",
    "V255":              "float32",
    "V256":              "float32",
    "V257":              "float32",
    "V258":              "float32",
    "V259":              "float32",
    "V260":              "float32",
    "V261":              "float32",
    "V262":              "float32",
    "V263":              "float32",
    "V264":              "float32",
    "V265":              "float32",
    "V266":              "float32",
    "V267":              "float32",
    "V268":              "float32",
    "V269":              "float32",
    "V270":              "float32",
    "V271":              "float32",
    "V272":              "float32",
    "V273":              "float32",
    "V274":              "float32",
    "V275":              "float32",
    "V276":              "float32",
    "V277":              "float32",
    "V278":              "float32",
    "V279":              "float32",
    "V280":              "float32",
    "V281":              "float32",
    "V282":              "float32",
    "V283":              "float32",
    "V284":              "float32",
    "V285":              "float32",
    "V286":              "float32",
    "V287":              "float32",
    "V288":              "float32",
    "V289":              "float32",
    "V290":              "float32",
    "V291":              "float32",
    "V292":              "float32",
    "V293":              "float32",
    "V294":              "float32",
    "V295":              "float32",
    "V296":              "float32",
    "V297":              "float32",
    "V298":              "float32",
    "V299":              "float32",
    "V300":              "float32",
    "V301":              "float32",
    "V302":              "float32",
    "V303":              "float32",
    "V304":              "float32",
    "V305":              "float32",
    "V306":              "float32",
    "V307":              "float32",
    "V308":              "float32",
    "V309":              "float32",
    "V310":              "float32",
    "V311":              "float32",
    "V312":              "float32",
    "V313":              "float32",
    "V314":              "float32",
    "V315":              "float32",
    "V316":              "float32",
    "V317":              "float32",
    "V318":              "float32",
    "V319":              "float32",
    "V320":              "float32",
    "V321":              "float32",
    "V322":              "float32",
    "V323":              "float32",
    "V324":              "float32",
    "V325":              "float32",
    "V326":              "float32",
    "V327":              "float32",
    "V328":              "float32",
    "V329":              "float32",
    "V330":              "float32",
    "V331":              "float32",
    "V332":              "float32",
    "V333":              "float32",
    "V334":              "float32",
    "V335":              "float32",
    "V336":              "float32",
    "V337":              "float32",
    "V338":              "float32",
    "V339":              "float32",
    "id_01":             "float32",
    "id_02":             "float32",
    "id_03":             "float32",
    "id_04":             "float32",
    "id_05":             "float32",
    "id_06":             "float32",
    "id_07":             "float32",
    "id_08":             "float32",
    "id_09":             "float32",
    "id_10":             "float32",
    "id_11":             "float32",
    "id_12":              "object",
    "id_13":             "float32",
    "id_14":             "float32",
    "id_15":              "object",
    "id_16":              "object",
    "id_17":             "float32",
    "id_18":             "float32",
    "id_19":             "float32",
    "id_20":             "float32",
    "id_21":             "float32",
    "id_22":             "float32",
    "id_23":              "object",
    "id_24":             "float32",
    "id_25":             "float32",
    "id_26":             "float32",
    "id_27":              "object",
    "id_28":              "object",
    "id_29":              "object",
    "id_30":              "object",
    "id_31":              "object",
    "id_32":             "float32",
    "id_33":              "object",
    "id_34":              "object",
    "id_35":              "object",
    "id_36":              "object",
    "id_37":              "object",
    "id_38":              "object",
    "DeviceType":         "object",
    "DeviceInfo":         "object",
    "is_fraud":			  "int8"
}


# In[ ]:


@contextmanager
def timer(name):
    """
    Time Each Process
    """
    t0 = time.time()
    yield
    print('\n[{}] done in {} Minutes\n'.format(name, round((time.time() - t0)/60,2)))

def fraud_preprocessing(debug = None):
    print("Starting Pre-Processing..")
    with timer("Load Tables"):
        train_transaction = pd.read_csv('../input/train_transaction.csv',
                                        index_col='TransactionID', nrows= debug, dtype = schema)
        test_transaction = pd.read_csv('../input/test_transaction.csv',
                                       index_col='TransactionID', nrows= debug, dtype = schema)

        train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
        test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

    with timer("Merge Tables"):
        train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
        test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

        print("Train Shape: {} Rows, {} Columns".format(*train.shape))
        print("Test Shape: {} Rows, {} Columns".format(*test.shape))

        y = train[target_var].copy()
        del train_transaction, train_identity, test_transaction, test_identity

        traindex = train.index
        testdex = test.index

        df = pd.concat([train.drop(target_var,axis=1),test],axis = 0)
        del train, test
        
    with timer("Feature Engineering"):
        print("** crickets **")

    with timer("Label Encode"):
        categorical_cols = []
        # Label Encoding
        for f in df.columns:
            if df[f].dtype=='object': 
                categorical_cols += [f]
                lbl = preprocessing.LabelEncoder()
                df[f] = lbl.fit_transform(df[f].astype(str))
#                 df[f] = df[f].astype('category')
        print("Total Shape: {} Rows, {} Columns".format(*df.shape))
                
    return df, y, traindex, testdex, categorical_cols


# #### Prepare Data

# In[ ]:


debug = None
df, y, traindex, testdex, cat_cols = fraud_preprocessing(debug = debug)
sample_submission = pd.read_csv('../input/sample_submission.csv',
                                index_col='TransactionID',
                                nrows = debug)

X = df.loc[traindex,:]
feat_names = X.columns.tolist() 
test = df.loc[testdex,:]
del df ; gc.collect();


# In[ ]:


print("None Fraud: {}%, Fraud: {}%".format(*y.value_counts(normalize=True)))
print("Randomness Score AUC: {}".format(
    metrics.roc_auc_score(y,np.array([y.value_counts(normalize=True)[0]]*y.shape[0]))))


# ### LGBM Model

# In[ ]:


metric = 'auc'
split_size = .5
n_estimators = 5000

# Parameters From
# https://www.kaggle.com/vincentlugat/ieee-lgb-bayesian-opt
params = {
 'boosting_type': 'gbdt',
 'device': 'gpu',
 'feature_fraction': 0.8,
 'is_unbalance': False,
 'learning_rate': 0.09,
 'max_depth': 20,
 'metric': 'auc',
 'min_data_in_leaf': 1,
 'nthread': -1,
 'num_boost_round': 231,
 'num_leaves': 4095,
 'objective': 'binary',
 'reg_alpha': 1.0,
 'reg_lambda': 1.0,
 'seed': 24,
 'subsample': 0.4,
 'subsample_for_bin': 500,
 'tree_learner': 'serial',
 'verbose': -1}


# In[ ]:


feature_subset = feat_names
X_train, X_valid, y_train, y_valid = train_test_split(
    X[feature_subset], y, test_size=split_size,
    random_state=seed, shuffle=True,stratify=y)

print("Light Gradient Boosting Multi-Class Classifier: ")
lgtrain = lgb.Dataset(X_train, y_train, categorical_feature = cat_cols, free_raw_data=False)
lgvalid = lgb.Dataset(X_valid, y_valid, categorical_feature = cat_cols, free_raw_data=False)

evals_result = {} 
with timer("LGBM"):
    lgb_model = lgb.train(
            params,
            lgtrain,
            valid_sets=[lgtrain, lgvalid],
            valid_names=['train','valid'],
            verbose_eval=300,
            num_boost_round = n_estimators,
            early_stopping_rounds=100,
            evals_result=evals_result
            )
    
# -------------------------------------------------------
print('Plot metrics during training...')
f,ax = plt.subplots(figsize = [6,5])
lgb.plot_metric(evals_result, metric=metric, ax = ax)
plt.show()
# -------------------------------------------------------


# In[ ]:


with timer("LGBM Feature Importance"):
    # Importance to DataFrame
    model_importance = pd.DataFrame(
                 [X_train.columns.tolist(),
                  lgb_model.feature_importance(importance_type = 'gain').tolist(),
                  lgb_model.feature_importance(importance_type = 'split').tolist()]).T
    model_importance.columns = ['feature','gain','split']

    print("Feature Importance")
    # Feature Importance Plot
    f, ax = plt.subplots(1,2,figsize=[20,10])
    lgb.plot_importance(lgb_model, max_num_features=50, ax=ax[0], importance_type='gain')
    ax[0].set_title("Light GBM GAIN Feature Importance")

    lgb.plot_importance(lgb_model, max_num_features=50, ax=ax[1], importance_type='split')
    ax[1].set_title("Light GBM SPLIT Feature Importance")
    plt.tight_layout(pad=1)
    plt.savefig('sales_propensity_feature_import.png')
    
    del X_train, X_valid, y_train, y_valid, lgb_model; gc.collect();


# #### Principle Component Analysis
# How features explain the majority of the variance amongst these features?
# 
# [Jake VanderPlas's PythonDataScienceHandbook](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html)

# In[ ]:


pca = PCA().fit(X.fillna(0).values)

f, ax = plt.subplots(1,2,figsize =[10,5])
ax[0].plot(np.cumsum(pca.explained_variance_ratio_), 'k-')
ax[0].set_xlabel('number of components')
ax[0].set_ylabel('cumulative explained variance')
ax[0].set_title("PCA - All Features")

ax[1].plot(np.cumsum(pca.explained_variance_ratio_[:20]),'xg-')
ax[1].set_xlabel('number of components')
ax[1].set_ylabel('cumulative explained variance')
ax[1].set_title("PCA - Top 20 Features")

plt.tight_layout(pad=0)
plt.show()

del pca; gc.collect();


# ## Iterative Split Feature Importance, Feature Selection with Early Stopping

# In[ ]:


def feature_count_trade_off(feat_list, step, importance_type = 'perm_score'):
    """
    feat_list:
        list of features to test - Iteratively refit model after shaving off worst features with step size
    importance_type:
        "perm_score": Feature Permutation
        "gain": Decision Tree Gain
        "split": Decisino Tree Split
    """
    
    feature_size_experiment = []
    feature_subset = feat_list
    intervals =  [2] + list(np.logspace(2, 8.5, num=20, base=2.0).astype(int).clip(0,X.shape[1]))
    intervals = intervals[::-1]
    print("Number of Experiments: {}".format(len(intervals)))
    print("Intervals to test: {}".format(intervals))
    for i in intervals:
        feature_subset = feature_subset[:i]
        sbst_X_train, sbst_X_valid, y_train, y_valid = train_test_split(
            X[feature_subset], y, test_size=split_size,
            random_state=seed, shuffle=True,stratify=y)
        
        sb_cat_cols = [x for x in cat_cols if x in feature_subset]
        lgtrain = lgb.Dataset(sbst_X_train, y_train, categorical_feature = sb_cat_cols)
        lgvalid = lgb.Dataset(sbst_X_valid, y_valid, categorical_feature = sb_cat_cols)

        experiment_lgb = lgb.train(
                params,
                lgtrain,
                valid_sets=[lgtrain, lgvalid],
                valid_names=['train','valid'],
                num_boost_round = n_estimators,
                verbose_eval=0,
                early_stopping_rounds=50
                )
        
        val_pred = experiment_lgb.predict(sbst_X_valid)
        
        # Get Metrics
        loss = metrics.log_loss(y_valid, val_pred)
        auc = metrics.roc_auc_score(y_valid, val_pred)
        feature_num = len(feature_subset)

        # Importance to DataFrame
        model_importance = pd.DataFrame(
                     [sbst_X_valid.columns.tolist(),
                      experiment_lgb.feature_importance(importance_type = 'gain').tolist(),
                      experiment_lgb.feature_importance(importance_type = 'split').tolist()]).T
        model_importance.columns = ['feature','gain','split']
        
        # Pass list to next iteration
        if importance_type == 'perm_score':
            model_importance = model_importance.sort_values(by='perm_score', ascending=False) # ascending True if minimization metric
            best_features_after_iteration = model_importance['feature'].values
        elif importance_type == 'gain':
            model_importance = model_importance.sort_values(by='gain', ascending=False)
            best_features_after_iteration = model_importance['feature'].values
        elif importance_type == 'split':
            model_importance = model_importance.sort_values(by='split', ascending=False)
            best_features_after_iteration = model_importance['feature'].values
            
        feature_size_experiment.append(
            [
             loss,
             feature_num,
             experiment_lgb.best_score['train'][metric],
             experiment_lgb.best_score['valid'][metric],
             experiment_lgb.best_iteration,
             feature_subset,
             np.mean(model_importance[importance_type].values),
             np.std(model_importance[importance_type].values)
            ]
        )
            
        feature_subset = best_features_after_iteration
        
    # Full Data
    feature_size_experiment_df = pd.DataFrame(feature_size_experiment,
                                              columns = ['valid_logloss',
                                                         'feature_count',
                                                         'train_{}'.format(metric),
                                                         'valid_{}'.format(metric),
                                                         'boosting_round',
                                                         'feature_set',
                                                         'average_importance_type',
                                                         'std_avg_importance_type'
                                                        ])
    
    # Plot
    f, ax = plt.subplots(1,4,figsize=[17,6])
    ax[0].plot(feature_size_experiment_df.feature_count, feature_size_experiment_df['valid_{}'.format(metric)], 'xr-')
    ax[0].set_title("AUC vs. Feature Count")
    ax[0].set_xlabel("Feature Count")
    ax[0].set_ylabel("AUC")
    ax[1].plot(feature_size_experiment_df.feature_count, feature_size_experiment_df.valid_logloss, 'o-')
    ax[1].set_title("Logloss vs. Feature Count")
    ax[1].set_xlabel("Feature Count")
    ax[1].set_ylabel("Logloss")
    ax[2].plot(feature_size_experiment_df.feature_count, feature_size_experiment_df.boosting_round, 'xg-')
    ax[2].set_title("Boosting Rounds vs. Feature Count")
    ax[2].set_xlabel("Feature Count")
    ax[2].set_ylabel("Boosting Rounds")
    ax[3].errorbar(feature_size_experiment_df.feature_count, feature_size_experiment_df['average_importance_type'],
               yerr=feature_size_experiment_df['std_avg_importance_type'], label='both limits (default)', fmt =  'r-')
    ax[3].set_title("Average {} Importance \nvs. Row Count in Thousands".format(importance_type))
    ax[3].set_xlabel("Boosting Rounds")
    ax[3].set_ylabel("Avg {}".format(importance_type))
    
    plt.tight_layout(pad=0)
    
    return feature_size_experiment_df


# In[ ]:


IMPORTANCE = 'split'
with timer("Feature Selection Experiment - RANKED BY {}".format(IMPORTANCE)):
    perm_expr = feature_count_trade_off(feat_list = model_importance.sort_values(by=IMPORTANCE, ascending=False)['feature'].values,
                                        step = 25,
                                        importance_type = IMPORTANCE)
    plt.savefig('{}_perm_score_experiment.png'.format(IMPORTANCE))
    plt.show()


# In[ ]:


# Output Table:
display(perm_expr.iloc[-25:])
perm_expr.to_csv("full_output_table_{}.csv".format(IMPORTANCE))


# #### Run Submission set on Final Features:

# In[ ]:


final_features = [
"TransactionDT",
"card1",
"TransactionAmt",
"addr1",
"card2",
"P_emaildomain",
"dist1",
"D15",
"id_31",
"D4",
"id_02",
"card5",
"D10",
"D2",
"D11",
"C13",
"id_19",
"id_20",
"D1",
"D8",
"D5",
"M4",
"D3",
"M5",
"M6",
"C2",
"C1",
"D9",
"dist2",
"id_33",
"id_13",
"C14",
"id_06",
"V307",
"C6",
"id_05",
"V310",
"C11",
"M8",
"R_emaildomain",
"M9",
"C9",
"M3",
"id_01",
"V313",
"D14",
"V127",
"V130",
"card6",
"V62",
"V315",
"id_30",
"V314",
"V264",
"C5",
"M7",
"V83",
"V76",
"V308",
"M2",
"V317",
"V87",
"V283",
"V20",
"V306",
"D6",
"V78",
"V312"]

cat_cols = [x for x in cat_cols if x in final_features]


# In[ ]:


get_ipython().run_cell_magic('time', '', '# SOURCE: https://www.kaggle.com/artkulak/ieee-fraud-simple-baseline-0-9383-lb\n# Added Convergence Plot\nEPOCHS = 3\nkf = KFold(n_splits = EPOCHS, shuffle = True)\ny_preds = np.zeros(sample_submission.shape[0])\ny_oof = np.zeros(X.shape[0])\nf,ax = plt.subplots(figsize = [8,5])\nfor i, (tr_idx, val_idx) in enumerate(kf.split(X, y)):\n    evals_result = {}\n    lgtrain = lgb.Dataset(X.iloc[tr_idx, :][final_features], y.iloc[tr_idx], categorical_feature = cat_cols)\n    lgvalid = lgb.Dataset(X.iloc[val_idx, :][final_features], y.iloc[val_idx], categorical_feature = cat_cols)\n    clf = lgb.train(\n            params,\n            lgtrain,\n            valid_sets=[lgtrain, lgvalid],\n            valid_names=[\'train\',\'valid\'],\n            verbose_eval=300,\n            early_stopping_rounds=100,\n            evals_result=evals_result,\n            )\n    y_pred_train = clf.predict(X.iloc[val_idx, :][final_features])\n    y_oof[val_idx] = y_pred_train\n    print(\'ROC AUC {}\\n\'.format(roc_auc_score(y.iloc[val_idx], y_pred_train)))\n    y_preds += clf.predict(test.loc[:,final_features]) / EPOCHS\n    \n    evals_result[\'train_{}\'.format(i)] = evals_result.pop(\'train\')\n    evals_result[\'valid_{}\'.format(i)] = evals_result.pop(\'valid\')\n    lgb.plot_metric(evals_result, metric=metric, ax = ax)\n\nplt.title("GPU LGBM Metric Convergence over {} Folds".format(EPOCHS))\nplt.show()')


# #### Submit

# In[ ]:


# When doing feature selection, make sure you use the same subset on test set.
# LGBM will not break, but it will give you broken predictions.. -_-
assert X[final_features].shape[1] == test[final_features].shape[1]

sample_submission['isFraud'] = y_preds
sample_submission.to_csv('{}_feats_{}fold_lgbm_gpu.csv'.format(len(final_features),EPOCHS))


# In[ ]:


print("Notebook Runtime: %0.2f Hours"%((time.time() - notebookstart)/60/60))


# **Reflection:** <br>
# Ideally I wanted to use feature permutation to do this iterative feature selection, but it is too computationally expensive (even on CPU kernel)
