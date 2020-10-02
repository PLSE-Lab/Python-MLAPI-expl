#!/usr/bin/env python
# coding: utf-8

# In[10]:


import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (f1_score, precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Preprocessing

# In[ ]:


uci_credit = pd.read_csv('../input/UCI_Credit_Card.csv')
uci_credit.head()


# For variables `PAY_x`, -2 means that the card holder had not used the card during the month, and -1 means that the card was used but paid off before the payment date. As -2 and -1 are special value, we need disinguish the numerical value from the logical meaning. While -2 and -1 both represent zero default, two sets of extra variables are created to hold the logical meaning.

# In[3]:


# clean: PAY_x  -1 - payoff, -2 - no use
for var in uci_credit.columns[6:12]:
    uci_credit[var + '_no_use'] = np.where(uci_credit[var] == -2, 1, 0)
    uci_credit[var + '_payoff'] = np.where(uci_credit[var] == -1, 1, 0)
    uci_credit[var] = np.where(uci_credit[var] < 0, 0, uci_credit[var])


# Both numerical and categorical variables exists in the dataset. We will handle them in different ways. Here we split the original dataset into two parts: one only contain numerical variables, and the other one only categorical variables.

# In[4]:


cont_variables = [
    'LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5',
    'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5',
    'PAY_AMT6', 'PAY_0_no_use', 'PAY_0_payoff', 'PAY_2_no_use', 'PAY_2_payoff',
    'PAY_3_no_use', 'PAY_3_payoff', 'PAY_4_no_use', 'PAY_4_payoff',
    'PAY_5_no_use', 'PAY_5_payoff', 'PAY_6_no_use', 'PAY_6_payoff'
]
disc_variables = ['SEX', 'EDUCATION', 'MARRIAGE']

cont_data = uci_credit[cont_variables].values
disc_data = uci_credit[disc_variables].values
print('Number of numerical variables:\t\t', cont_data.shape[1])
print('Number of categorical variables:\t', disc_data.shape[1])
target = uci_credit['default.payment.next.month'].values


# Next we split training (80%) and testing datasets, one-hot encode the categorical variables, merge the numerical part and categorical together, and get the final datasets for modeling.

# In[5]:


train_cont_data, test_cont_data, train_disc_data, test_disc_data, train_target, test_target = train_test_split(
    cont_data,
    disc_data,
    target,
    train_size=0.8,
    random_state=1000,
    shuffle=True)

def one_hot_coding(train, test):
    one_hot_coder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    train = one_hot_coder.fit_transform(train)
    test = one_hot_coder.transform(test)
    return train, test


train_disc_data, test_disc_data = one_hot_coding(train_disc_data,
                                                 test_disc_data)
# print(train_disc_data.shape)
# print(test_disc_data.shape)

train_data = np.concatenate((train_cont_data, train_disc_data), axis=1)
test_data = np.concatenate((test_cont_data, test_disc_data), axis=1)


# # GBDT Model (ver. 1)
# 
# For a preliminary study, we just use the variables in the dataset to build a model, to set a baseline for all of our following study.
# 
# Basic steps:
# 
# - Normalization
# - Model tuning with 10-fold cross validation
# - Model evaluation on the testing set

# In[6]:


def normalize(train, test):
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test

train_data, test_data = normalize(train_data, test_data)


# In[7]:


classifier = GradientBoostingClassifier(random_state=1000)
optimizer = GridSearchCV(classifier,
                        param_grid={
                            'learning_rate':[0.1, 0.05, 0.01],
                            'n_estimators':[100, 150, 200, 250],
                            'max_depth':[2, 3, 4, 5],
                            'subsample':[0.6, 0.8, 1.0],
                            'max_features':['sqrt', 'log2']
                        },
                        cv=10,
                        scoring='roc_auc', 
                        n_jobs=4,
                        verbose=1)
optimizer.fit(train_data, train_target)
print('best score (roc_auc):', optimizer.best_score_)


# In[8]:


feat_imp = optimizer.best_estimator_.feature_importances_
feat_imp_df = pd.DataFrame({'Feature#' + str(i):[imp] for i, imp in enumerate(feat_imp)}, index=['Importance']).T
feat_imp_df.sort_values(by='Importance', ascending=False).head()


# The result above shows the top 5 important features are:
# 
# - `PAY_0`: Last terms of default
# - `BILLAMT_1`: Newest bill amount
# - `BILLAMT_2`: Last bill amount
# - `LIMIT_BAL`: Credit limit
# - `PAY_AMT_1`: Last payment
# 
# All of these variable are very reasonable to be important.

# In[9]:


estimator = optimizer.best_estimator_
train_pred = estimator.predict_proba(train_data)[:, 1]
test_pred = estimator.predict_proba(test_data)[:, 1]

gbdt_v1_train_auc = roc_auc_score(train_target, train_pred)
gbdt_v1_test_auc = roc_auc_score(test_target, test_pred)

gbdt_v1_train_roc = roc_curve(train_target, train_pred)
gbdt_v1_test_roc = roc_curve(test_target, test_pred)

def ks(truth, score):
    
    def ecdf(score):
        def p(x):
            return np.sum(score <= x) / len(score)
        return p
            
    pos_dist = ecdf(score[truth == 1])
    neg_dist = ecdf(score[truth == 0])
    
    xs = [0.001 * i for i in range(1001)]
    ks = [abs(pos_dist(x) - neg_dist(x)) for x in xs]
    return np.max(ks)
    
gbdt_v1_train_ks = ks(train_target, train_pred)
gbdt_v1_test_ks = ks(test_target, test_pred)

plt.plot(gbdt_v1_train_roc[0], gbdt_v1_train_roc[1], 'r-', label='Train')
plt.plot(gbdt_v1_test_roc[0], gbdt_v1_test_roc[1], 'b--', label='Test')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.suptitle('Receiver Operating Characteristic Curve')
plt.legend(loc='best')
plt.show()


# In[10]:


gbdt_v1_train_pr = precision_recall_curve(train_target, train_pred)
gbdt_v1_test_pr = precision_recall_curve(test_target, test_pred)

plt.plot(gbdt_v1_train_pr[1], gbdt_v1_train_pr[0], 'r-', label='Train')
plt.plot(gbdt_v1_test_pr[1], gbdt_v1_test_pr[0], 'b--', label='Test')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.suptitle('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()


# In[11]:


print('\tAUC\tKS\nTrain:  {:<8}{:<8}\nTest:   {:<8}{:<8}'.format(
    round(gbdt_v1_train_auc, 4),
    round(gbdt_v1_train_ks, 4),
    round(gbdt_v1_test_auc, 4),
    round(gbdt_v1_test_ks, 4)
))


# ### Conclusion
# 
# 1. The model is somewhat overfit. There exists a big performance drop between training and testing dataset.
# 
# 2. The reliable AUC for version 1 is about 0.79, and KS around 0.45,  which is not bad in most cases.

# # GBDT Model (ver. 2, with derived variables)
# 
# The performance of the former model is almost as good enough. However, there is still much information lying behind. What if we extract them explicitly as much as possible? Here we go.
# 
# ### Historical default information
# 
# - Maximum historical default period
# - Times of default
# - Terms since last default (0-6, 6 means no default record in recent 6 months)
# - Maximum bill amount of default
# - Minimum bill amount of default
# - Aerage bill amount of defaul

# In[12]:


def default_info(cont_data):
    default = cont_data[:, 2:8]
    billamt = cont_data[:, 8:14]
    
    max_default = np.max(default, axis=1)
    cnt_default = np.sum(np.where(default > 0, 1, 0), axis=1)
    _ = np.c_[default, np.ones((default.shape[0], 1))]
    terms_last = np.argmax(_, axis=1)
    
    _ = np.where(default > 0, 1, np.nan) * billamt
    max_billamt_default = np.nanmax(_, axis=1)
    min_billamt_default = np.nanmin(_, axis=1)
    avg_billamt_default = np.nanmean(_, axis=1)
    
    _ =  np.c_[max_default, 
               cnt_default,
               terms_last, 
               max_billamt_default,
               min_billamt_default,
               avg_billamt_default]
    return np.where(np.isnan(_), 0, _)

train_default_info = default_info(train_cont_data)
test_default_info = default_info(test_cont_data)


# ### Bill amount variables
# 
# - Min bill amount in 6 months
# - Max bill amount in 6 months
# - Average bill amount in 6 months
# - Range of bill amount in 6 months
# - Range / (Average + epsilon) in 6 months
# - (Latest  - Average) / (Range + epsilon) in 6 months
# - Min bill amount in 3 months
# - Max bill amount in 3 months
# - Average bill amount in 3 months
# - Range of bill amount in 3 months
# - Range / (Average + epsilon) in 3 months
# - (Latest  - Average) / (Range + epsilon) in 3 months
# 
# - Min bill amount in 6 months / balance amount
# - Max bill amount in 6 months / balance amount
# - Average bill amount in 6 months / balance amount
# - Range of bill amount in 6 months / balance amount
# - Min bill amount in 3 months / balance amount
# - Max bill amount in 3 months / balance amount
# - Average bill amount in 3 months / balance amount
# - Range of bill amount in 3 months / balance amount

# In[13]:


def bill_amount_vars(cont_data, epsilon=1e-8):
    balance_limit = cont_data[:, 0].reshape((-1, 1))
    bill_amt = cont_data[:, 8:14]
    
    min_amt = np.min(bill_amt, axis=1)
    max_amt = np.max(bill_amt, axis=1)
    avg_amt = np.mean(bill_amt, axis=1)
    range_amt = max_amt - min_amt
    variation = range_amt / (avg_amt + epsilon)
    latest_var = (bill_amt[:,0] - avg_amt) / (range_amt + epsilon)
    min_amt_3m = np.min(bill_amt[:, 0:3], axis=1)
    max_amt_3m = np.max(bill_amt[:, 0:3], axis=1)
    avg_amt_3m = np.mean(bill_amt[:, 0:3], axis=1)
    range_amt_3m = max_amt_3m - min_amt_3m
    variation_3m = range_amt_3m / (avg_amt_3m + epsilon)
    latest_var_3m = (bill_amt[:,0] - avg_amt_3m) / (range_amt_3m + epsilon)
    
    _ = np.c_[min_amt,
              max_amt,
              avg_amt,
              range_amt,
              variation, 
              latest_var,
              min_amt_3m,
              max_amt_3m,
              avg_amt_3m,
              range_amt_3m,
              variation_3m,
              latest_var_3m]
    tmp = np.c_[min_amt,
                max_amt,
                avg_amt,
                range_amt,
                min_amt_3m,
                max_amt_3m,
                avg_amt_3m,
                range_amt_3m,] / balance_limit
    return np.c_[_, tmp]

train_bill_vars = bill_amount_vars(train_cont_data)
test_bill_vars = bill_amount_vars(test_cont_data)


# ### Payment amount variables
# 
# __Note__: Payment amount should be compared with bill amount in last month
# 
# - Max payment
# - Min payment
# - Avg payment
# - Payment range
# - Payment range / (avg payment + epsilon)
# - (Lastest payment - avg payment) / (payment range + epsilon)
# 
# - Max payment 3m
# - Min payment 3m
# - Avg payment 3m
# - Payment range 3m
# - Payment range 3m / (avg payment 3m + epsilon)
# - (Lastest payment 3m - avg payment 3m) / (payment range 3m + epsilon)
# 
# - Max difference of payment and bill amount
# - Min difference of payment and bill amount
# - Avg difference of payment and bill amount
# - Max difference of payment and bill amount / balance limit
# - Min difference of payment and bill amount / balance limit
# - Avg difference of payment and bill amount / balance limit

# In[14]:


def pay_vars(cont_data, epsilon=1e-8):
    bal_limit = cont_data[:,0]
    payment = cont_data[:, 14:20]
    billamt = cont_data[:, 9:14]
    
    max_pay = np.max(payment, axis=1)
    min_pay = np.min(payment, axis=1)
    avg_pay = np.mean(payment, axis=1)
    pay_range = max_pay - min_pay
    pay_var = pay_range / (avg_pay + epsilon)
    pay_last = (payment[:,0] - avg_pay) / (pay_range + epsilon)
    
    max_pay_3m = np.max(payment[:, 0:3], axis=1)
    min_pay_3m = np.min(payment[:, 0:3], axis=1)
    avg_pay_3m = np.mean(payment[:, 0:3], axis=1)
    pay_range_3m = max_pay_3m - min_pay_3m
    pay_var_3m = pay_range_3m / (avg_pay_3m + epsilon)
    pay_last_3m = (payment[:,0] - avg_pay_3m) / (pay_range_3m + epsilon)
    
    diff_pay_bill = payment[:, 0:5] - billamt[:,0:5]
    max_diff = np.max(diff_pay_bill, axis=1)
    min_diff = np.min(diff_pay_bill, axis=1)
    avg_diff = np.mean(diff_pay_bill, axis=1)
    max_diff_r = max_diff / bal_limit
    min_diff_r = min_diff / bal_limit
    avg_diff_r = avg_diff / bal_limit
    last_diff_r = (diff_pay_bill[:, 0] - avg_diff) / (avg_diff + epsilon)
    
    return np.c_[
        max_pay, 
        min_pay, 
        avg_pay, 
        pay_range,
        pay_var, 
        pay_last,
        max_pay_3m, 
        min_pay_3m, 
        avg_pay_3m, 
        pay_range_3m,
        pay_var_3m, 
        pay_last_3m,
        max_diff,
        min_diff, 
        avg_diff, 
        max_diff_r, 
        min_diff_r, 
        avg_diff_r, 
        last_diff_r
    ]

train_pay_vars = pay_vars(train_cont_data)
test_pay_vars = pay_vars(test_cont_data)


# ### Features integration

# In[15]:


train_data_v2 = np.c_[train_cont_data, 
                      train_disc_data, 
                      train_default_info,
                      train_bill_vars,
                      train_pay_vars]
test_data_v2 = np.c_[test_cont_data, 
                     test_disc_data, 
                     test_default_info,
                     test_bill_vars,
                     test_pay_vars]

train_data_v2, test_data_v2 = normalize(train_data_v2, test_data_v2)


# ### Modeling

# In[16]:


classifier_v2 = GradientBoostingClassifier(random_state=1000)
optimizer_v2 = GridSearchCV(classifier_v2,
                        param_grid={
                            'learning_rate':[0.1, 0.05, 0.01],
                            'n_estimators':[100, 150, 200, 250],
                            'max_depth':[2, 3, 4, 5],
                            'subsample':[0.6, 0.8, 1.0],
                            'max_features':['sqrt', 'log2']
                        },
                        cv=10,
                        scoring='roc_auc', 
                        n_jobs=4,
                        verbose=1)
optimizer_v2.fit(train_data_v2, train_target)
print('best score (roc_auc):', optimizer_v2.best_score_)


# In[17]:


feat_imp_v2 = optimizer_v2.best_estimator_.feature_importances_
feat_imp_df_v2 = pd.DataFrame({'Feature#' + str(i):[imp] for i, imp in enumerate(feat_imp_v2)},
                              index=['Importance']).T
feat_imp_df_v2.sort_values(by='Importance', ascending=False).head()


# In[18]:


estimator_v2 = optimizer_v2.best_estimator_
train_pred_v2 = estimator_v2.predict_proba(train_data_v2)[:, 1]
test_pred_v2 = estimator_v2.predict_proba(test_data_v2)[:, 1]

gbdt_v2_train_auc = roc_auc_score(train_target, train_pred_v2)
gbdt_v2_test_auc = roc_auc_score(test_target, test_pred_v2)

gbdt_v2_train_roc = roc_curve(train_target, train_pred_v2)
gbdt_v2_test_roc = roc_curve(test_target, test_pred_v2)
    
gbdt_v2_train_ks = ks(train_target, train_pred_v2)
gbdt_v2_test_ks = ks(test_target, test_pred_v2)

plt.plot(gbdt_v2_train_roc[0], gbdt_v2_train_roc[1], 'r-', label='Train')
plt.plot(gbdt_v2_test_roc[0], gbdt_v2_test_roc[1], 'b--', label='Test')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.suptitle('Receiver Operating Characteristic Curve')
plt.legend(loc='best')
plt.show()


# In[19]:


gbdt_v2_train_pr = precision_recall_curve(train_target, train_pred_v2)
gbdt_v2_test_pr = precision_recall_curve(test_target, test_pred_v2)

plt.plot(gbdt_v2_train_pr[1], gbdt_v2_train_pr[0], 'r-', label='Train')
plt.plot(gbdt_v2_test_pr[1], gbdt_v2_test_pr[0], 'b--', label='Test')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.suptitle('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()


# In[20]:


print('\tAUC\tKS\nTrain:  {:<8}{:<8}\nTest:   {:<8}{:<8}'.format(
    round(gbdt_v2_train_auc, 4),
    round(gbdt_v2_train_ks, 4),
    round(gbdt_v2_test_auc, 4),
    round(gbdt_v2_test_ks, 4)
))


# ### Compare of Ver. 1 and Ver. 2

# In[21]:


plt.plot(gbdt_v1_test_roc[0], gbdt_v1_test_roc[1], 'r-', label='Model v1')
plt.plot(gbdt_v2_test_roc[0], gbdt_v2_test_roc[1], 'b-', label='Model v2')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.suptitle('Receiver Operating Characteristic Curve')
plt.legend(loc='best')
plt.show()


# In[22]:


plt.plot(gbdt_v1_test_pr[1], gbdt_v1_test_pr[0], 'r-', label='Model v1')
plt.plot(gbdt_v2_test_pr[1], gbdt_v2_test_pr[0], 'b-', label='Model v2')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.suptitle('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()


# From the plots we can that the second version slightly exceeds the first one in the testing set. However, essentially, not a big difference was made.

# # XGBoost Model with Derived Variables
# 
# In Kaggle, XGBoost is widely used for its robustness and good performance. Let's see how much it will beat x

# In[53]:


train_dmat = xgb.DMatrix(train_data_v2, label=train_target)
test_dmat = xgb.DMatrix(test_data_v2, label=test_target)
#
#xgboost.cv(params, dtrain, num_boost_round=10, nfold=3, stratified=False, 
#folds=None, metrics=(), obj=None, feval=None, maximize=False, early_stopping_rounds=None, 
#fpreproc=None, as_pandas=True, verbose_eval=None, show_stdv=True, seed=0, callbacks=None, shuffle=True)
#
xgb_model = xgb.train(
    {
        "eta": 0.01,
        "max_depth": 3,
        "subsample": 0.4,
        "alpha": 3,
        "lambda": 1,
        "eval_metric": "auc"
    },
    dtrain=train_dmat,
    num_boost_round=1000)


# ### Evaluation

# In[54]:


train_pred_xgb = xgb_model.predict(train_dmat)
test_pred_xgb = xgb_model.predict(test_dmat)

xgb_train_auc = roc_auc_score(train_target, train_pred_xgb)
xgb_test_auc = roc_auc_score(test_target, test_pred_xgb)

xgb_train_roc = roc_curve(train_target, train_pred_xgb)
xgb_test_roc = roc_curve(test_target, test_pred_xgb)
    
xgb_train_ks = ks(train_target, train_pred_xgb)
xgb_test_ks = ks(test_target, test_pred_xgb)

plt.plot(xgb_train_roc[0], xgb_train_roc[1], 'r-', label='Train')
plt.plot(xgb_test_roc[0], xgb_test_roc[1], 'b--', label='Test')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.suptitle('Receiver Operating Characteristic Curve')
plt.legend(loc='best')
plt.show()


# In[55]:


xgb_train_pr = precision_recall_curve(train_target, train_pred_xgb)
xgb_test_pr = precision_recall_curve(test_target, test_pred_xgb)

plt.plot(xgb_train_pr[1], xgb_train_pr[0], 'r-', label='Train')
plt.plot(xgb_test_pr[1], xgb_test_pr[0], 'b--', label='Test')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.suptitle('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()


# In[56]:


print('\tAUC\tKS\nTrain:  {:<8}{:<8}\nTest:   {:<8}{:<8}'.format(
    round(xgb_train_auc, 4),
    round(xgb_train_ks, 4),
    round(xgb_test_auc, 4),
    round(xgb_test_ks, 4)
))


# ### Comparation between GBDT ver. 2 and XGBoost

# In[58]:


plt.plot(gbdt_v2_test_roc[0], gbdt_v2_test_roc[1], 'r-', label='GBDT v2')
plt.plot(xgb_test_roc[0], xgb_test_roc[1], 'b-', label='XGBoost')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.suptitle('Receiver Operating Characteristic Curve')
plt.legend(loc='best')
plt.show()


# In[59]:


plt.plot(gbdt_v2_test_pr[1], gbdt_v2_test_pr[0], 'r-', label='Model v2')
plt.plot(xgb_test_pr[1], xgb_test_pr[0], 'b-', label='XGBboost')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.suptitle('Precision-Recall Curve')
plt.legend(loc='best')
plt.show()

