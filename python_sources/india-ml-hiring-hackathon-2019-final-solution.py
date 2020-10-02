#!/usr/bin/env python
# coding: utf-8

# # Analytics Vidya India ML Hiring Hackathon 2019
# 
# <h2 id="problem-statement">Problem Statement</h2>
# <div class="page" title="Page 1">
# <div class="section">
# <div class="layoutArea">
# <div class="column">
# <h2 id="loan-delinquency-prediction">Loan Delinquency Prediction</h2>
# Loan default prediction is one of the most critical and crucial problem faced by financial institutions and organizations as it has a noteworthy effect on the profitability of these institutions. In recent years, there is a tremendous increase in the volume of non &ndash; performing loans which results in a jeopardizing effect on the growth of these institutions.&nbsp;
# <p>Therefore, to maintain a healthy portfolio, the banks put stringent monitoring and evaluation measures in place to ensure timely repayment of loans by borrowers. Despite these measures, a major proportion of loans become delinquent. Delinquency occurs when a borrower misses a payment against his/her loan.</p>
# <p>Given the information like mortgage details, borrowers related details and payment details, our objective is to identify the delinquency status of loans for the next month given the delinquency status for the previous 12 months (in number of months)</p>
# <h2 id="data-description">Data Description</h2>
# </div>
# </div>
# * * </div>
# </div>
# <h3 id="trainzip">train.zip</h3>
# <p>train.zip contains train.csv.&nbsp;<strong>train.csv</strong>&nbsp;contains the training data with details on loan as described in the last section</p>
# <h3 id="data-dictionary">Data Dictionary</h3>
# <table dir="ltr" border="1" cellspacing="0" cellpadding="0"><colgroup><col width="100" /><col width="368" /></colgroup>
# <tbody>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Variable&quot;}"><strong>Variable</strong></td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Description&quot;}"><strong>Description</strong></td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;loan_id&quot;}">loan_id</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Unique loan ID&quot;}">Unique loan ID</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;source&quot;}">source</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Loan origination channel&quot;}">Loan origination channel</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;financial_institution&quot;}">financial_institution</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Name of the bank&quot;}">Name of the bank</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;interest_rate&quot;}">interest_rate</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Loan interest rate&quot;}">Loan interest rate</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;unpaid_principal_bal&quot;}">unpaid_principal_bal</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Loan unpaid principal balance&quot;}">Loan unpaid principal balance</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;loan_term&quot;}">loan_term</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Loan term (in days)&quot;}">Loan term (in days)</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;origination_date&quot;}">origination_date</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Loan origination date&quot;}">Loan origination date (YYYY-MM-DD)</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;first_payment_date&quot;}">first_payment_date</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;First instalment payment date&quot;}">First instalment payment date</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;loan_to_value&quot;}">loan_to_value</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Loan to value ratio&quot;}">Loan to value ratio</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;number_of_borrowers&quot;}">number_of_borrowers</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Number of borrowers&quot;}">Number of borrowers</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;debt_to_income_ratio&quot;}">debt_to_income_ratio</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Debt-to-income ratio&quot;}">Debt-to-income ratio</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;borrower_credit_score&quot;}">borrower_credit_score</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Borrower credit score&quot;}">Borrower credit score</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;loan_purpose&quot;}">loan_purpose</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Loan purpose&quot;}">Loan purpose</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;insurance_percent&quot;}">insurance_percent</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Mortgage insurance percent&quot;}">Loan Amount percent covered by insurance</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;co-borrower_credit_score&quot;}">co-borrower_credit_score</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Co-borrower credit score&quot;}">Co-borrower credit score</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;insurance_type&quot;}">insurance_type</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Mortgage insurance type&quot;}">0 - Premium paid by borrower, 1 - Premium paid by Lender</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;m1 to m12&quot;}">m1 to m12</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Month-wise loan performance (deliquency in months)&quot;}">Month-wise loan performance (deliquency in months)</td>
# </tr>
# <tr>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;m13&quot;}">m13</td>
# <td data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;target, loan deliquency status (0 = non deliquent, 1 = deliquent)&quot;}">target, loan deliquency status (0 = non deliquent, 1 = deliquent)</td>
# </tr>
# </tbody>
# </table>
# <h3 id="testzip"><br />test.zip</h3>
# <p>test.zip contains test.csv which has details of all loans for which the participants are to submit the delinquency status - 0/1 (not probability)</p>
# <h3 id="sample_submissionzip">sample_submission.zip</h3>
# <p>sample_submission.zip contains the submission format for the predictions against the test set. A single csv needs to be submitted as a solution.</p>
# <div class="page" title="Page 1">
# <div class="section">
# <div class="layoutArea">
# <div class="column">
# <h2 id="evaluation-metric">Evaluation Metric</h2>
# Submissions are evaluated on F1-Score between the predicted class and the observed target.</div>
# <div class="column">&nbsp;</div>
# </div>
# </div>
# </div>
# <div class="page" title="Page 2">
# <h2 id="public-and-private-split">Public and Private Split</h2>
# Test data is further randomly divided into Public (40%) and Private (60%) data.
# <ul>
# <li>Your initial responses will be checked and scored on the Public data.</li>
# <li>The final rankings would be based on your private score which will be published once the competition is over.</li>
# </ul>
# <h2 id="hackathon-rules">Hackathon Rules</h2>
# <ol>
# <li>Setting the final submission is mandatory. Without a final submission, the submission corresponding to best public score will be taken as final submission</li>
# <li>Use of external datasets is not allowed</li>
# <li>Use of loan_id variable as an input to the model is not allowed</li>
# <li>You can only make 15 submissions per day</li>
# <li>Code file is mandatory while setting&nbsp;final submission. For GUI based tools, please upload a zip file of snapshots of steps taken by you, else upload code file.</li>
# <li>The code file uploaded should be pertaining to your final submission.</li>
# <li>No submission will be accepted after the contest deadline</li>
# </ol>
# </div>
# <p>&nbsp;</p>
# 
# # LB Score 0.34
# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
import xgboost as xgb
from sklearn.decomposition import PCA,KernelPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from keras import Sequential
from keras import layers
from keras import backend as K
from keras.layers.core import Dense
from keras import regularizers
from keras.layers import Dropout
from keras.constraints import max_norm
import tensorflow as tf
import keras
from lightgbm import LGBMClassifier


# Data processing, metrics and modeling
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold,KFold

from datetime import datetime
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc,precision_recall_curve
from sklearn import metrics
from sklearn import preprocessing


# In[ ]:


train = pd.read_csv("../input/india-ml-hiring-hackathon-2019/train.csv")
test = pd.read_csv("../input/india-ml-hiring-hackathon-2019/test.csv")
sub = pd.read_csv("../input/india-ml-hiring-hackathon-2019/sample_submission.csv")


# In[ ]:


# I have used pandans profiling for EDA , based on that we removed number_of_borrowers 
train = train.drop(['loan_id','number_of_borrowers'],axis=1)
test = test.drop(['loan_id','number_of_borrowers'],axis=1)


# In[ ]:


target = train['m13']
train=train.drop('m13',axis=1)


# In[ ]:


# Lets try something

train.head()


# In[ ]:


#train['LTV_by_DTI'] = train['loan_to_value'] / train['debt_to_income_ratio']
#test['LTV_by_DTI'] = test['loan_to_value'] / test['debt_to_income_ratio']


# In[ ]:


train = pd.get_dummies(train)
test = pd.get_dummies(test)


# In[ ]:


# Align train and test

train_labels = target

# Align the training and testing data, keep only columns present in both dataframes
train_df, test_df = train.align(test, join = 'inner', axis = 1)

# Add the target back in
train_df['m13'] = train_labels

print('Training Features shape: ', train_df.shape)
print('Testing Features shape: ', test_df.shape)


# In[ ]:


from imblearn.under_sampling import TomekLinks


# In[ ]:


tl = TomekLinks()


# In[ ]:


train_df = train_df.reindex(
    np.random.permutation(train_df.index))


# In[ ]:


y = train_df['m13']
X = train_df.drop('m13',axis=1)


# In[ ]:


from imblearn.over_sampling import SVMSMOTE


# In[ ]:


sm = SVMSMOTE(random_state=42)


# In[ ]:


X_res, y_res = sm.fit_resample(X, y)


# In[ ]:


X_res, y_res = tl.fit_resample(X_res, y_res)


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=10,stratify=y_res)


# In[ ]:


from sklearn.metrics import f1_score

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True


# In[ ]:


def run_lgb(X_train, X_test, y_train, y_test, test_df):
    params = {
        "objective" : "binary",
       "n_estimators":10000,
       "reg_alpha" : 2.0,
       "reg_lambda":2.1,
       "n_jobs":-1,
       "colsample_bytree":.8,
       "min_child_weight":0.8,
       "subsample":0.8715623,
       "min_data_in_leaf":20,
       "nthread":4,
       "metric" : "f1",
       "num_leaves" : 100,
       "learning_rate" : 0.01,
       "verbosity" : -1,
       "seed": 120,
       "max_bin":60,
       'max_depth':15,
       'min_gain_to_split':.0222415,
       'scale_pos_weight':1
    }
    
    lgtrain = lgb.Dataset(X_train, label=y_train)
    lgval = lgb.Dataset(X_test, label=y_test)
    evals_result = {}
    model = lgb.train(params, lgtrain, 10000, 
                      valid_sets=[lgtrain, lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=100, 
                      evals_result=evals_result,feval=lgb_f1_score)
    
    pred_test_y = model.predict(test_df, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result


# In[ ]:


pred_test, model, evals_result = run_lgb(X_train, X_test, y_train, y_test, test_df)
print("LightGBM Training Completed...")


# In[ ]:


sub['m13'] = pred_test


# In[ ]:


sub['m13'] = sub['m13'].apply(lambda x : 1 if (x>=0.40) else 0)


# In[ ]:


sub['m13'].sum()


# In[ ]:


sub.to_csv('subm.csv',index=False)


# In[ ]:


from IPython.display import HTML 
import pandas as pd 
import numpy as np 
import base64 


# In[ ]:


def create_download_link(df, title = "Download CSV file", filename = "subm.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()) 
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>' 
    html = html.format(payload=payload,title=title,filename=filename) 
    return HTML(html)


# In[ ]:


create_download_link(sub)


# ## F1 score of this solution was 0.34 and had 50th Rank on LB

# ## Catbooster

# In[ ]:


get_ipython().system('pip install catboost')


# In[ ]:


import catboost

class ModelOptimizer:
    best_score = None
    opt = None
    
    def __init__(self, model, X_train, y_train, categorical_columns_indices=None, n_fold=3, seed=1994, early_stopping_rounds=30, is_stratified=True, is_shuffle=True):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.categorical_columns_indices = categorical_columns_indices
        self.n_fold = n_fold
        self.seed = seed
        self.early_stopping_rounds = early_stopping_rounds
        self.is_stratified = is_stratified
        self.is_shuffle = is_shuffle
        
        
    def update_model(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.model, k, v)
            
    def evaluate_model(self):
        pass
    
    def optimize(self, param_space, max_evals=10, n_random_starts=2):
        start_time = time.time()
        
        @use_named_args(param_space)
        def _minimize(**params):
            self.model.set_params(**params)
            return self.evaluate_model()
        
        opt = gp_minimize(_minimize, param_space, n_calls=max_evals, n_random_starts=n_random_starts, random_state=2405, n_jobs=-1)
        best_values = opt.x
        optimal_values = dict(zip([param.name for param in param_space], best_values))
        best_score = opt.fun
        self.best_score = best_score
        self.opt = opt
        
        print('optimal_parameters: {}\noptimal score: {}\noptimization time: {}'.format(optimal_values, best_score, time.time() - start_time))
        print('updating model with optimal values')
        self.update_model(**optimal_values)
        plot_convergence(opt)
        return optimal_values
    
class CatboostOptimizer(ModelOptimizer):
    def evaluate_model(self):
        validation_scores = catboost.cv(
        catboost.Pool(self.X_train, 
                      self.y_train, 
                      cat_features=self.categorical_columns_indices),
        self.model.get_params(), 
        nfold=self.n_fold,
        stratified=self.is_stratified,
        seed=self.seed,
        early_stopping_rounds=self.early_stopping_rounds,
        shuffle=self.is_shuffle,
        verbose=100,
        plot=False)
        self.scores = validation_scores
        test_scores = validation_scores.iloc[:, 2]
        best_metric = test_scores.max()
        return 1 - best_metric


# In[ ]:


get_ipython().system('pip install scikit-optimize')


# In[ ]:


from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import time


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score
m=CatBoostClassifier(n_estimators=3000,random_state=1994,eval_metric='AUC',max_depth=12,learning_rate=0.029,od_wait=50
                     ,l2_leaf_reg=5,bagging_temperature=0.85,random_strength=100,
                     use_best_model=True)
m.fit(X_train,y_train,eval_set=[(X_train,y_train),(X_test, y_test)], early_stopping_rounds=100,verbose=100)


# In[ ]:


test_pred = m.predict(test_df)


# In[ ]:


test_pred.sum()

