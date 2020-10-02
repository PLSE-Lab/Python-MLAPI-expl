#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report,roc_auc_score
from matplotlib import pyplot
import xgboost as xgb
from scipy import stats
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


df_application = pd.read_csv('../input/application_train.csv')
df_application_test = pd.read_csv('../input/application_test.csv')
df_application.head()


# In[3]:


df_application['Source'] = 'Train'
df_application_test['Source'] = 'Test'    
df = pd.concat((df_application,df_application_test),axis = 0,sort = False)
cat_cols = [col for col in df.columns if (df[col].dtype == object) & (col != 'Source' )]
le = preprocessing.LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col].fillna("Missing"))
df.head()
df_train = df[df['Source'] == "Train"].drop('Source', axis =1)
df_test = df[df['Source'] == "Test"].drop('Source', axis =1)
del df


# In[4]:


df_bureau = pd.read_csv("../input/bureau.csv")
df_bureau_balance = pd.read_csv("../input/bureau_balance.csv")
##Create simple feature
df_bureau_balance["MONTHS_BALANCE"]= np.abs(df_bureau_balance["MONTHS_BALANCE"])
df_bureau_balance["Period"] = np.where((df_bureau_balance["MONTHS_BALANCE"] < 7),"short",np.where((df_bureau_balance["MONTHS_BALANCE"] < 13),"medium","long"))
#df_bureau_balance = pd.get_dummies(df_bureau_balance,prefix = "STATUS",columns = "STATUS",dummy_na = True)
df_bureau_balance["Period_status"] = df_bureau_balance["Period"].astype(str) + "_" + df_bureau_balance["STATUS"]
df_bureau_balance.head(5)
#df_bureau_balance = pd.get_dummies(df_bureau_balance,prefix = "Period_status",columns = "Period_status")


# In[5]:


df_bureau_balance = df_bureau_balance.groupby(["SK_ID_BUREAU","Period_status"])                                      .agg({"MONTHS_BALANCE" : ["count","min","max","mean"]})                                      .reset_index()
df_bureau_balance.columns =  [''.join(col).strip() for col in df_bureau_balance.columns.values]
df_bureau_balance.head()


# In[6]:


df_bureau_balance = df_bureau_balance.pivot_table(index = 'SK_ID_BUREAU',columns = 'Period_status',values = ['MONTHS_BALANCEcount','MONTHS_BALANCEmin','MONTHS_BALANCEmax','MONTHS_BALANCEmean']).reset_index()
df_bureau_balance.columns =  [''.join(col).strip() for col in df_bureau_balance.columns.values]
df_bureau = pd.merge(df_bureau,df_bureau_balance, how="left", on = "SK_ID_BUREAU")
df_bureau=pd.get_dummies(df_bureau,prefix =['CREDIT_ACTIVE','CREDIT_CURRENCY','CREDIT_TYPE'] ,columns = ['CREDIT_ACTIVE','CREDIT_CURRENCY','CREDIT_TYPE'],dummy_na = True)
df_bureau.head(5)


# In[7]:


ohe_cols = df_bureau.columns[14:].tolist()


# In[8]:


df_bureau_features = df_bureau.groupby(['SK_ID_CURR'])                         .agg({'SK_ID_BUREAU' : ['nunique'],
                              'DAYS_CREDIT'  : ['min','max','mean','std'],
                              'CREDIT_DAY_OVERDUE' :['min','max','mean','std'],
                              'DAYS_CREDIT_ENDDATE':['min','max','mean','std'],
                              'DAYS_ENDDATE_FACT': ['min','max','mean','std'],
                              'AMT_CREDIT_MAX_OVERDUE' : ['mean','min','max'],
                              'CNT_CREDIT_PROLONG' : ['mean','min','max'],
                              'AMT_CREDIT_SUM' : ['min','max','mean','std'],
                              'AMT_CREDIT_SUM_DEBT' : ['sum','min','max'],
                              'AMT_CREDIT_SUM_LIMIT': ['sum','min','max'],
                              'AMT_CREDIT_SUM_OVERDUE':['sum','min','max'],
                              'DAYS_CREDIT_UPDATE' : ['sum','min','max'],
                              'AMT_ANNUITY' : ['sum','min','max','mean']
                             })
df_bureau_features = df_bureau_features.reset_index()
df_bureau_features.columns =  [''.join(col).strip() for col in df_bureau_features.columns.values]
df_bureau_features_ohe = df_bureau[['SK_ID_CURR'] + ohe_cols].groupby(['SK_ID_CURR']).mean().reset_index()
df_bureau_features = pd.merge(df_bureau_features,df_bureau_features_ohe,how = 'left',on = 'SK_ID_CURR')
del df_bureau_features_ohe
df_train = pd.merge(df_train,df_bureau_features,how = 'left',on = 'SK_ID_CURR')
df_test = pd.merge(df_test,df_bureau_features,how = 'left',on = 'SK_ID_CURR')


# In[9]:


###previous applicatin data
df_previous_application = pd.read_csv("../input/previous_application.csv")
cat_cols = [col for col in df_previous_application.columns if (df_previous_application[col].dtype == object) & ((col != 'SK_ID_CURR' ) | (col != 'SK_ID_PREV'))]
df_previous_application = pd.get_dummies(df_previous_application,prefix = cat_cols,columns = cat_cols)
df_previous_application.head()


# In[10]:


##Create current application ->  credit card balance, installment monthly balancemapping
##Current Application features
df_POS_CASH_balance = pd.read_csv("../input/POS_CASH_balance.csv")
df_POS_CASH_balance = pd.get_dummies(df_POS_CASH_balance, columns= ["NAME_CONTRACT_STATUS"])
df_POS_CASH_balance_current = df_POS_CASH_balance.drop('SK_ID_PREV',axis = 1).groupby('SK_ID_CURR').mean().reset_index()
df_POS_CASH_balance_previous = df_POS_CASH_balance.drop('SK_ID_CURR',axis = 1).groupby('SK_ID_PREV').mean().reset_index()
del df_POS_CASH_balance
df_POS_CASH_balance_current.head()


# In[11]:


##installment history
df_installments_payments = pd.read_csv("../input/installments_payments.csv")
df_installments_payments_current = df_installments_payments.drop('SK_ID_PREV',axis= 1).groupby('SK_ID_CURR').mean().reset_index()
df_installments_payments_previous = df_installments_payments.drop('SK_ID_CURR',axis= 1).groupby('SK_ID_PREV').mean().reset_index()
del df_installments_payments
df_installments_payments_current.head()


# In[12]:


##Credit card history
df_credit_card_balance = pd.read_csv("../input/credit_card_balance.csv")
df_credit_card_balance = pd.get_dummies(df_credit_card_balance, columns= ['NAME_CONTRACT_STATUS'])
df_credit_card_balance_current = df_credit_card_balance.drop('SK_ID_PREV',axis = 1).groupby('SK_ID_CURR').mean().reset_index()
df_credit_card_balance_previous = df_credit_card_balance.drop('SK_ID_CURR',axis = 1).groupby('SK_ID_PREV').mean().reset_index()
del df_credit_card_balance
df_credit_card_balance_current.head()


# In[13]:


###Append to train and test sets
df_train = df_train.merge(df_POS_CASH_balance_current, on = 'SK_ID_CURR',how = 'left',suffixes=['','_POS_bal_curr'])                    .merge(df_installments_payments_current, on = 'SK_ID_CURR',how = 'left', suffixes = ['','_installments_curr'])                    .merge(df_credit_card_balance_current, on = 'SK_ID_CURR',how = 'left',suffixes=['','_credit_card_bal_curr'])
        
df_test =  df_test.merge(df_POS_CASH_balance_current, on = 'SK_ID_CURR',how = 'left',suffixes=['','_POS_bal_curr'])                   .merge(df_installments_payments_current, on = 'SK_ID_CURR',how = 'left',suffixes = ['','_installments_curr'])                   .merge(df_credit_card_balance_current, on = 'SK_ID_CURR',how = 'left',suffixes=['','_credit_card_bal_curr']) 
df_train.head()


# In[14]:


###Leavin out SK_ID_PREV for now. Will use it as a temporal variable later on
df_previous_application = df_previous_application.merge(df_POS_CASH_balance_previous, on = 'SK_ID_PREV',how = 'left',suffixes=['','_POS_bal_past'])                                                  .merge(df_installments_payments_previous, on = 'SK_ID_PREV',how = 'left',suffixes = ['','_installments_past'])                                                  .merge(df_credit_card_balance_previous, on = 'SK_ID_PREV',how = 'left',suffixes=['','_credit_card_bal_past'])
        
df_previous_application = df_previous_application.drop("SK_ID_PREV",axis= 1).groupby(['SK_ID_CURR']).mean().reset_index()

df_train = df_train.merge(df_previous_application, on = 'SK_ID_CURR',how = 'left',suffixes= ('','_past_appl'))
df_test = df_test.merge(df_previous_application, on = 'SK_ID_CURR',how = 'left',suffixes= ('','_past_appl'))
df_previous_application.head()

del df_previous_application


# In[15]:



##Boosting trial
df1 = df_train.sample(frac = 1)
msk = np.random.rand(len(df1))
eval_set = df1[msk >= 0.95]
train = df1[msk < 0.95]


# In[101]:
#train_cols = cat_var+numeric_var
#Train matrices
X_train = train.drop(['TARGET','SK_ID_CURR'],axis = 1)
Y_train = train['TARGET']
#Eval matrices
X_eval = eval_set.drop(['TARGET','SK_ID_CURR'],axis = 1)
Y_eval = eval_set['TARGET']
#Y_eval.shape = (len(Y_eval),1)

##Try model , eval_metric = 'accuracy', eval_set = eval_set
# 1. XGB eval_set = [(X_train, Y_train), (X_eval, Y_eval)]
def xgb_(X_train = X_train, Y_train = Y_train,
         params = {
                 "objective"        :['multi:softmax']
                 ,"max_depth"        :[2]
                 ,'eta'              :[0.1]
        },
        fit_params = {
                'eval_metric'      :['mlogloss']
                ,'eval_set'         :[(X_eval,Y_eval)]},
        X_eval = X_eval, Y_eval = Y_eval):

    
    ######### Apply xgb
    #d_train = xgb.DMatrix(X_train , label = Y_train)
    #d_eval =  xgb.DMatrix(X_eval , label = Y_eval)
    print("# Tuning hyper-parameters for accuracy" )
    print()
    model_xgb = xgb.XGBClassifier()
   
    clf = GridSearchCV(model_xgb, param_grid = params,
                       fit_params = fit_params,cv =3, scoring ="roc_auc")
    clf.fit(X_train, Y_train)
    print("\nBest parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    
    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = Y_eval, clf.predict(X_eval)
    print(classification_report(y_true, y_pred))
    print()
    print(confusion_matrix(y_true, y_pred))
    print()
    model_xgb = clf.best_estimator_
    xgb.plot_importance(model_xgb,max_num_features = 20)
    
    #Eval model
    Y_dev_pred = model_xgb.predict_proba(X_eval)[:,1]
    score = roc_auc_score(Y_eval,Y_dev_pred)
    # retrieve performance metrics
    results = model_xgb.evals_result()
    metrics = fit_params['eval_metric']
    epochs = len(results['validation_0'][metrics[0]])
    x_axis = range(0, epochs)
    # plot log loss
    
    for metric in metrics:
        fig, ax = pyplot.subplots()
        ax.plot(x_axis, results['validation_0'][metric], label='Train')
        ax.plot(x_axis, results['validation_1'][metric], label='Validation/Hold out set')
        ax.legend()
        pyplot.ylabel('%s' %(metric) )
        pyplot.title('XGBoost %s' %(metric))
        
        pyplot.show()

    return model_xgb,score,clf,results,model_xgb.feature_importances_


# In[16]:


xgb_params = {
     'learning_rate'    :[0.02]
    ,'reg_lambda'       :[16]
    ,"max_depth"        :[9]
    ,'silent'           :[False]
    ,'n_estimators'     :[1000]
    ,'colsample_bytree' :[0.5]
    ,'nthread'          :[-1]
    ,'subsample'        :[0.5]
    ,'objective'        :["binary:logistic"]
    ,'scale_pos_weight' :[2]
    #,'tree_method'      :['gpu_hist']
    #,'min_child_weight' :[10]
    }
fit_params = {
    'eval_metric'       :['auc']
    ,'eval_set'         :[(X_train,Y_train),(X_eval,Y_eval)]
    ,'early_stopping_rounds' : 30
  #  ,'early_stopping_rounds' :[5]
}

model_xgb,score_xgb,xgb_gridsearch,results,feats = xgb_(X_train = X_train,X_eval = X_eval,params = xgb_params,fit_params = fit_params)


# In[17]:


X_test = df_test.drop(['TARGET','SK_ID_CURR'],axis = 1)
Y_test = model_xgb.predict_proba(X_test)[:,1]


# In[18]:


df_application_test["TARGET"] = Y_test
df_submit = df_application_test[["SK_ID_CURR","TARGET"]]
df_submit.to_csv('submission_appl_bureau.csv', index=False)

