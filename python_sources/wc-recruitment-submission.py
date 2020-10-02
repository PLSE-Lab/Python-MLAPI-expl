#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from tqdm import tqdm_notebook
import os


# In[ ]:


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


# In[ ]:


train=pd.read_csv('../input/webclubrecruitment2019/TRAIN_DATA.csv')
test=pd.read_csv('../input/webclubrecruitment2019/TEST_DATA.csv')
sample_submission=pd.read_csv('../input/webclubrecruitment2019/SAMPLE_SUB.csv')


# In[ ]:


train.drop('Unnamed: 0',axis=1,inplace=True)
test.rename({'Unnamed: 0':'Id'},axis=1,inplace=True)
print("Train Shape: ",train.shape)
print("Test Shape: ",test.shape)


# In[ ]:


from scipy.stats import pearsonr
def find_corr(data,target):
    return np.abs(pearsonr(data,target)[0])

def col_target_corr(data,target_col='Class',n=5):
    corr_vals={}
    for col in data.columns:
        corr_vals[col]=find_corr(data[col],data[target_col])
    corr_vals=dict(sorted(corr_vals.items(),key=lambda x:x[1],reverse=True)[:n+1])
    fig,ax=plt.subplots(1,1,figsize=(13,5))
    sns.barplot(x=[*corr_vals.values()],y=[*corr_vals.keys()],ax=ax)
    ax.set_title("Correlation Plot",fontsize=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.show()
    return dict(list(corr_vals.items())[1:])


# In[ ]:


from scipy.stats import boxcox
def outlier_detection(data,q_25=None,q_75=None):
    if(q_25==None):
        q_25=data.quantile(0.25)
        q_75=data.quantile(0.75)
    k=1.5
    outlier_min=q_25-k*(q_75-q_25)
    outlier_max=q_75+k*(q_75-q_25)
    outliers=[x for x in data if (x<outlier_min)|(x>outlier_max)]
    corrected_vals=data.apply(lambda x:outlier_min if x<outlier_min else outlier_max if x>outlier_max else x)
    return outliers,corrected_vals

def correct_skewness(data,min_val=None):
    if(min_val==None):
        min_val=data.min()
    shifted_data=data.apply(lambda x:x-min_val+1)
    return pd.Series(boxcox(shifted_data)[0])

def continuous_preprocessing(train,test):
    train_gen_feats={}
    test_gen_feats={}
    min_V1=min(train['V1'].min(),test['V1'].min())
    train_gen_feats['V1_skew']=correct_skewness(train['V1'],min_V1)
    test_gen_feats['V1_skew']=correct_skewness(test['V1'],min_V1)
    
    train_gen_feats['V6_thresh']=train['V6'].apply(lambda x:1 if x>582 else 0)
    test_gen_feats['V6_thresh']=test['V6'].apply(lambda x:1 if x>582 else 0)
    
    V6_q25=train['V6'].quantile(0.25)
    V6_q75=train['V6'].quantile(0.75)
    train_gen_feats['V6_outlier']=outlier_detection(train['V6'],V6_q25,V6_q75)[1]
    test_gen_feats['V6_outlier']=outlier_detection(test['V6'],V6_q25,V6_q75)[1]
    
    train_gen_feats['V10_thresh']=train['V10'].apply(lambda x:1 if x>25.5 else 0)
    test_gen_feats['V10_thresh']=test['V10'].apply(lambda x:1 if x>25.5 else 0)
    
    train_gen_feats=pd.DataFrame.from_dict(train_gen_feats)
    train_gen_feats.index=train.index
    test_gen_feats=pd.DataFrame.from_dict(test_gen_feats)
    test_gen_feats.index=test.index
    return train_gen_feats,test_gen_feats

train_feats,test_feats=continuous_preprocessing(train,test)
train_new=pd.concat([train,train_feats],axis=1)
test_new=pd.concat([test,test_feats],axis=1)


# In[ ]:


def categorical_preprocessing(train,test):
    train_gen_feats={}
    test_gen_feats={}
    
    #train_gen_feats['V2_group']=train['V2'].apply(lambda x:1 if x in [4,2,5,6,10,8] else 0)
    #test_gen_feats['V2_group']=test['V2'].apply(lambda x:1 if x in [4,2,5,6,10,8] else 0)
    train_gen_feats['V3_group']=train['V3'].apply(lambda x:1 if x==1 else 0)
    test_gen_feats['V3_group']=test['V3'].apply(lambda x:1 if x==1 else 0)
    train_gen_feats['V4_group']=train['V4'].apply(lambda x:1 if x==2 else 0)
    test_gen_feats['V4_group']=test['V4'].apply(lambda x:1 if x==2 else 0)
    train_gen_feats['V9_group']=train['V9'].apply(lambda x:1 if x==2 else 0)
    test_gen_feats['V9_group']=test['V9'].apply(lambda x:1 if x==2 else 0)
    #train_gen_feats['V11_group1']=train['V11'].apply(lambda x:1 if x==10 else 0)
    #test_gen_feats['V11_group1']=test['V11'].apply(lambda x:1 if x==10 else 0)
    #train_gen_feats['V11_group2']=train['V11'].apply(lambda x:1 if x in [10,2] else 0)
    #test_gen_feats['V11_group2']=test['V11'].apply(lambda x:1 if x in [10,2] else 0)
    train_gen_feats['V16_group']=train['V16'].apply(lambda x:1 if x==0 else 0)
    test_gen_feats['V16_group']=test['V16'].apply(lambda x:1 if x==0 else 0)
    
    train_gen_feats=pd.DataFrame.from_dict(train_gen_feats)
    train_gen_feats.index=train.index
    test_gen_feats=pd.DataFrame.from_dict(test_gen_feats)
    test_gen_feats.index=test.index
    return train_gen_feats,test_gen_feats

train_feats,test_feats=categorical_preprocessing(train,test)
train_new=pd.concat([train_new,train_feats],axis=1)
test_new=pd.concat([test_new,test_feats],axis=1)


# In[ ]:


def cont_cont_preprocessing(train,test):
    train_gen_feats={}
    test_gen_feats={}
    
    train_gen_feats['V1_V10']=train.apply(lambda x:1 if (x['V1']<36)&(x['V10']<23.5) else 0,axis=1)
    test_gen_feats['V1_V10']=test.apply(lambda x:1 if (x['V1']<36)&(x['V10']<23.5) else 0,axis=1)
    train_gen_feats['V6_V10']=train.apply(lambda x:1 if (x['V6']<71808)&(x['V10']<23.5) else 0,axis=1)
    test_gen_feats['V6_V10']=test.apply(lambda x:1 if (x['V6']<71808)&(x['V10']<23.5) else 0,axis=1)
    train_gen_feats['V10_V13']=train.apply(lambda x:1 if (x['V10']<23.5)&(x['V13']<16.5) else 0,axis=1)
    test_gen_feats['V10_V13']=test.apply(lambda x:1 if (x['V10']<23.5)&(x['V13']<16.5) else 0,axis=1)
    
    train_gen_feats=pd.DataFrame.from_dict(train_gen_feats)
    train_gen_feats.index=train.index
    test_gen_feats=pd.DataFrame.from_dict(test_gen_feats)
    test_gen_feats.index=test.index
    return train_gen_feats,test_gen_feats

train_feats,test_feats=cont_cont_preprocessing(train,test)
train_new=pd.concat([train_new,train_feats],axis=1)
test_new=pd.concat([test_new,test_feats],axis=1)


# In[ ]:


def cat_cat_preprocessing(train,test):
    train_gen_feats={}
    test_gen_feats={}
    
    train_gen_feats['V3_V16']=train.apply(lambda x:1 if (x['V3']==1)&(x['V16']==3) else 0,axis=1)
    test_gen_feats['V3_V16']=test.apply(lambda x:1 if (x['V3']==1)&(x['V16']==3) else 0,axis=1)
    #train_gen_feats['V4_V11']=train.apply(lambda x:1 if (x['V4']==1)&(x['V11']==10) else 0,axis=1)
    #test_gen_feats['V4_V11']=test.apply(lambda x:1 if (x['V4']==1)&(x['V11']==10) else 0,axis=1)
    #train_gen_feats['V11_V16']=train.apply(lambda x:1 if (x['V11']==10)&(x['V16']==3) else 0,axis=1)
    #test_gen_feats['V11_V16']=test.apply(lambda x:1 if (x['V11']==10)&(x['V16']==3) else 0,axis=1)
    
    train_gen_feats=pd.DataFrame.from_dict(train_gen_feats)
    train_gen_feats.index=train.index
    test_gen_feats=pd.DataFrame.from_dict(test_gen_feats)
    test_gen_feats.index=test.index
    return train_gen_feats,test_gen_feats

train_feats,test_feats=cat_cat_preprocessing(train,test)
train_new=pd.concat([train_new,train_feats],axis=1)
test_new=pd.concat([test_new,test_feats],axis=1)


# In[ ]:


def cont_cat_preprocessing(train,test):
    train_gen_feats={}
    test_gen_feats={}
    
    train_gen_feats['V1_V3']=train.apply(lambda x:1 if (x['V1']<62)&(x['V3']==1) else 0,axis=1)
    test_gen_feats['V1_V3']=test.apply(lambda x:1 if (x['V1']<62)&(x['V3']==1) else 0,axis=1)
    #train_gen_feats['V6_V3']=train.apply(lambda x:1 if (x['V6']<3807)&(x['V3']==1) else 0,axis=1)
    #test_gen_feats['V6_V3']=test.apply(lambda x:1 if (x['V6']<3807)&(x['V3']==1) else 0,axis=1)
    #test_gen_feats['V10_V3']=test.apply(lambda x:1 if (x['V10']<31)&(x['V3']==1) else 0,axis=1)
    #train_gen_feats['V13_V3']=train.apply(lambda x:1 if (x['V13']<56.11)&(x['V3']==1) else 0,axis=1)
    #test_gen_feats['V13_V3']=test.apply(lambda x:1 if (x['V13']<56.11)&(x['V3']==1) else 0,axis=1)
    
    train_gen_feats=pd.DataFrame.from_dict(train_gen_feats)
    train_gen_feats.index=train.index
    test_gen_feats=pd.DataFrame.from_dict(test_gen_feats)
    test_gen_feats.index=test.index
    return train_gen_feats,test_gen_feats

train_feats,test_feats=cont_cat_preprocessing(train,test)
train_new=pd.concat([train_new,train_feats],axis=1)
test_new=pd.concat([test_new,test_feats],axis=1)


# In[ ]:


train_new.drop(['V14','V15'],axis=1,inplace=True)
test_new.drop(['V14','V15'],axis=1,inplace=True)

train_new.drop(['V5'],axis=1,inplace=True)
test_new.drop(['V5'],axis=1,inplace=True)


# In[ ]:


corr_vals=col_target_corr(train_new,'Class',20)

print("New Train Shape: ",train_new.shape)
print("New Test Shape: ",test_new.shape)


# In[ ]:


def normalize(X,test_X):
    norm_X=X.copy()
    norm_test_X=test_X.copy()
    for col in X.columns:
        tot_data=pd.concat([X[col],test_X[col]])
        tot_mean=tot_data.mean()
        tot_std=tot_data.std()
        norm_X[col]=X[col].apply(lambda x:(x-tot_mean)/tot_std)
        norm_test_X[col]=test_X[col].apply(lambda x:(x-tot_mean)/tot_std)
    return norm_X,norm_test_X


# In[ ]:


from sklearn.model_selection import train_test_split

raw_X=train_new.drop(['Class'],axis=1)
Y=train_new['Class']
feat_names=[*raw_X.columns]

raw_test_X=test_new.drop(['Id'],axis=1)
test_id=test_new['Id']

X,test_X=normalize(raw_X,raw_test_X)


# In[ ]:


from sklearn.metrics import confusion_matrix,roc_auc_score,balanced_accuracy_score,fbeta_score,recall_score
train_perc=0.75;val_perc=0.1;test_perc=0.15


# # General Models

# In[ ]:


def cv_training(X,Y,model_func,model_params,cv=5):
    train_scores={}
    test_scores={}
    for ind in range(cv):
        random_state=np.random.randint(100)
        train_x,test_x,train_y,test_y=train_test_split(X,Y,stratify=Y,test_size=test_perc,random_state=random_state)
        
        model=model_func(**model_params)
        model.fit(train_x,train_y)
        
        pred=model.predict_proba(train_x)
        pred=[x[1] for x in pred]
        binary_pred=[1 if x>=0.5 else 0 for x in pred]
        acc_score=balanced_accuracy_score(train_y,binary_pred)*100
        auc_score=roc_auc_score(train_y,pred)
        f2_score=fbeta_score(train_y,binary_pred,2)
        train_scores[ind+1]={
            'acc':acc_score,
            'auc':auc_score,
            'f2':f2_score
        }
        
        pred=model.predict_proba(test_x)
        pred=[x[1] for x in pred]
        binary_pred=[1 if x>=0.5 else 0 for x in pred]
        acc_score=balanced_accuracy_score(test_y,binary_pred)*100
        auc_score=roc_auc_score(test_y,pred)
        f2_score=fbeta_score(test_y,binary_pred,2)
        test_scores[ind+1]={
            'acc':acc_score,
            'auc':auc_score,
            'f2':f2_score
        }
    return train_scores,test_scores

def training_and_prediction(X,Y,test_X,model_func,model_params,cv=5):
    final_preds=np.zeros((cv,test_X.shape[0]),dtype=np.float32)
    for ind in range(cv):
        random_state=np.random.randint(100)
        train_x,test_x,train_y,test_y=train_test_split(X,Y,stratify=Y,test_size=test_perc,random_state=random_state)
        
        model=model_func(**model_params)
        model.fit(train_x,train_y)
        
        pred=model.predict_proba(train_x)
        pred=[x[1] for x in pred]
        binary_pred=[1 if x>=0.5 else 0 for x in pred]
        acc_score=balanced_accuracy_score(train_y,binary_pred)*100
        auc_score=roc_auc_score(train_y,pred)
        f2_score=fbeta_score(train_y,binary_pred,2)
        print("Training Accuracy: {acc:.2f}%".format(acc=acc_score))
        print("Training AUC Score: {auc:.3f}".format(auc=auc_score))
        print("Training F2 Score: {f2:.3f}".format(f2=f2_score))
        print()
        pred=model.predict_proba(test_x)
        pred=[x[1] for x in pred]
        binary_pred=[1 if x>=0.5 else 0 for x in pred]
        acc_score=balanced_accuracy_score(test_y,binary_pred)*100
        auc_score=roc_auc_score(test_y,pred)
        f2_score=fbeta_score(test_y,binary_pred,2)
        print("Testing Accuracy: {acc:.2f}%".format(acc=acc_score))
        print("Testing AUC Score: {auc:.3f}".format(auc=auc_score))
        print("Testing F2 Score: {f2:.3f}".format(f2=f2_score))
        
        final_pred=model.predict_proba(test_X)
        final_pred=[x[1] for x in final_pred]
        final_preds[ind]=final_pred
        print('\n\n')
        
    return final_preds


# In[ ]:


def get_model_performance(model_func,model_params,model_name):
    print("Model - ",model_name)
    train_scores,test_scores=cv_training(X,Y,model_func,model_params)
    
    avg_acc=np.mean([x['acc'] for x in train_scores.values()])
    avg_auc=np.mean([x['auc'] for x in train_scores.values()])
    avg_f2=np.mean([x['f2'] for x in train_scores.values()])
    print("Average Training Accuracy: {acc:.2f}%".format(acc=avg_acc))
    print("Average Training AUC Score: {auc:.3f}".format(auc=avg_auc))
    print("Average Training F2 Score: {f2:.3f}".format(f2=avg_f2))
    print()
    avg_acc=np.mean([x['acc'] for x in test_scores.values()])
    avg_auc=np.mean([x['auc'] for x in test_scores.values()])
    avg_f2=np.mean([x['f2'] for x in test_scores.values()])
    print("Average Testing Accuracy: {acc:.2f}%".format(acc=avg_acc))
    print("Average Testing AUC Score: {auc:.3f}".format(auc=avg_auc))
    print("Average Training F2 Score: {f2:.3f}".format(f2=avg_f2))
    print('\n\n')
    
def get_model_submission(model_func,model_params,model_name):
    print("Model - ",model_name)
    final_preds=training_and_prediction(X,Y,test_X,model_func,model_params)
    final_pred=final_preds.mean(axis=0)
    submission={'Id':test_id,'PredictedValue':final_pred}
    submission=pd.DataFrame.from_dict(submission)
    submission.to_csv(model_name+'_submission.csv',index=False)

    sns.distplot(submission['PredictedValue'],bins=100)
    plt.show()

    return submission


# In[ ]:


from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


# In[ ]:


#get_model_performance(LogisticRegression,{},"Logistic Regression")
#get_model_performance(RandomForestClassifier,{},"RF Classifier")
#get_model_performance(ExtraTreesClassifier,{},"ET Classifier")


# In[ ]:


#sub1=get_model_submission(LogisticRegression,{},"Logistic Regression")
#sub2=get_model_submission(RandomForestClassifier,{},"RF Classifier")
#sub3=get_model_submission(ExtraTreesClassifier,{},"ET Classifier")


# # XGB Model

# In[ ]:


import xgboost as xgb
def f2_score_func(y,t):
    t=t.get_label()
    y_bin=[1.0 if x>=0.5 else 0.0 for x in y]
    return 'f2',fbeta_score(t,y_bin,2)

def recall_score_func(y,t):
    t=t.get_label()
    y_bin=[1.0 if x>=0.5 else 0.0 for x in y]
    return 'f2',recall_score(t,y_bin)

def train_xgb_model(xgb_train,xgb_val,xgb_params={},boosting_rounds=100,plot_perf=False,verbose_eval=True,feval='auc'):
    evallist=[(xgb_train,'train'),(xgb_val,'eval')]

    eval_history={}
    record_eval=xgb.callback.record_evaluation(eval_history)
    early_stop=xgb.callback.early_stop(25,verbose=False,maximize=True)
    callbacks=[record_eval,early_stop]

    xgb_model=xgb.train(
        xgb_params,xgb_train,num_boost_round=boosting_rounds,
        evals=evallist,callbacks=callbacks,verbose_eval=verbose_eval,feval=feval)
    
    if(plot_perf==True):
        fig,ax=plt.subplots(1,2,figsize=(15,7))
        xgb.plot_importance(xgb_model,ax=ax[0])
        ax[1].plot([*eval_history['train']['auc']],label='Train AUC',color='blue')
        ax[1].plot([*eval_history['eval']['auc']],label='Test AUC',color='red')
        ax[1].legend(loc='best')
        ax[1].set_title("Training Performance")
        ax[1].set_xlabel("Rounds")
        ax[1].set_ylabel("AUC")
        plt.show()
    
    return xgb_model

def xgb_cv_training(X,Y,cv=5,xgb_params={},verbose_eval=True,feval='auc'):
    train_scores={}
    test_scores={}
    for ind in range(cv):
        random_state=np.random.randint(100)
        train_x,test_x,train_y,test_y=train_test_split(X,Y,stratify=Y,test_size=test_perc,random_state=random_state)
        train_x,val_x,train_y,val_y=train_test_split(train_x,train_y,stratify=train_y,test_size=val_perc/(val_perc+train_perc),random_state=random_state)

        xgb_train=xgb.DMatrix(train_x,label=train_y,feature_names=feat_names)
        xgb_val=xgb.DMatrix(val_x,label=val_y,feature_names=feat_names)
        
        xgb_model=train_xgb_model(
            xgb_train,xgb_val,xgb_params,boosting_rounds=1000,verbose_eval=verbose_eval,feval=feval)
        
        train_matrix=xgb.DMatrix(train_x,feature_names=feat_names)
        pred=xgb_model.predict(train_matrix)
        binary_pred=[1 if x>=0.5 else 0 for x in pred]
        acc_score=balanced_accuracy_score(train_y,binary_pred)*100
        auc_score=roc_auc_score(train_y,pred)
        f2_score=recall_score(train_y,binary_pred)
        train_scores[ind+1]={
            'acc':acc_score,
            'auc':auc_score,
            'f2':f2_score
        }
        
        test_matrix=xgb.DMatrix(test_x,feature_names=feat_names)
        pred=xgb_model.predict(test_matrix)
        binary_pred=[1 if x>=0.5 else 0 for x in pred]
        acc_score=balanced_accuracy_score(test_y,binary_pred)*100
        auc_score=roc_auc_score(test_y,pred)
        f2_score=recall_score(test_y,binary_pred)
        test_scores[ind+1]={
            'acc':acc_score,
            'auc':auc_score,
            'f2':f2_score
        }
    return train_scores,test_scores

def xgb_training_and_prediction(X,Y,test_X,cv=5,xgb_params={},feval='auc'):
    
    xgb_final_test=xgb.DMatrix(test_X,feature_names=feat_names)
    final_predictions=np.zeros((cv,test_X.shape[0]),dtype=np.float32)
    for ind in range(cv):
        print("CV ",ind+1)
        random_state=np.random.randint(100)
        print("Random State: ",random_state)
        train_x,test_x,train_y,test_y=train_test_split(X,Y,stratify=Y,test_size=test_perc,random_state=random_state)
        train_x,val_x,train_y,val_y=train_test_split(train_x,train_y,stratify=train_y,test_size=val_perc/(val_perc+train_perc),random_state=random_state)

        xgb_train=xgb.DMatrix(train_x,label=train_y,feature_names=feat_names)
        xgb_val=xgb.DMatrix(val_x,label=val_y,feature_names=feat_names)
        
        xgb_model=train_xgb_model(xgb_train,xgb_val,xgb_params,boosting_rounds=100,feval=feval)
        
        train_matrix=xgb.DMatrix(train_x,feature_names=feat_names)
        pred=xgb_model.predict(train_matrix)
        binary_pred=[1 if x>=0.5 else 0 for x in pred]
        acc_score=balanced_accuracy_score(train_y,binary_pred)*100
        auc_score=roc_auc_score(train_y,pred)
        f2_score=fbeta_score(train_y,binary_pred,2)
        print("Training Accuracy Score: {score:.2f}%".format(score=acc_score))
        print("Training AUC Score: {score:.3f}".format(score=auc_score))
        print("Training F2 Score: {score:.3f}".format(score=f2_score))
        print()
        test_matrix=xgb.DMatrix(test_x,feature_names=feat_names)
        pred=xgb_model.predict(test_matrix)
        binary_pred=[1 if x>=0.5 else 0 for x in pred]
        acc_score=balanced_accuracy_score(test_y,binary_pred)*100
        auc_score=roc_auc_score(test_y,pred)
        f2_score=fbeta_score(test_y,binary_pred,2)
        print("Testing Accuracy Score: {score:.2f}%".format(score=acc_score))
        print("Testing AUC Score: {score:.3f}".format(score=auc_score))
        print("Testing F2 Score: {score:.3f}".format(score=f2_score))
        print("\n\n")
        
        final_pred=xgb_model.predict(xgb_final_test)
        final_predictions[ind]=final_pred
    return final_predictions


# In[ ]:


xgb_params={
    'objective':'binary:logistic',
    'eval_metric':'auc',
    'n_estimators':130,
    'max_depth':5,
    'min_child_weight':3,
    'gamma':0,
    'subsample':0.7,
    'colsample_bytree':0.9,
    'colsample_bylevel':0.7,
    'scale_pos_weight':95/5,
    'learning_rate':0.05,
    'verbosity':0,
    'max_delta_step':0,
    'lambda':1,
    'alpha':0
}


# In[ ]:


final_preds=xgb_training_and_prediction(X,Y,test_X,cv=10,xgb_params=xgb_params,feval=recall_score_func)

final_pred=final_preds.mean(axis=0)
submission1={'Id':test_id,'PredictedValue':final_pred}
submission1=pd.DataFrame.from_dict(submission1)
submission1.to_csv('submission1.csv',index=False)

sns.distplot(submission1['PredictedValue'],bins=100)
plt.show()

submission1.head()


# In[ ]:


xgb_params={
    'objective':'binary:logistic',
    'eval_metric':'auc',
    'n_estimators':100,
    'max_depth':6,
    'min_child_weight':5,
    'gamma':3,
    'subsample':0.9,
    'colsample_bytree':0.5,
    'colsample_bylevel':0.8,
    'scale_pos_weight':95/5,
    'learning_rate':0.1,
    'verbosity':0,
    'alpha':5,
    'lambda':1
}


# In[ ]:


final_preds=xgb_training_and_prediction(X,Y,test_X,cv=10,xgb_params=xgb_params,feval=recall_score_func)

final_pred=final_preds.mean(axis=0)
submission2={'Id':test_id,'PredictedValue':final_pred}
submission2=pd.DataFrame.from_dict(submission2)
submission2.to_csv('submission2.csv',index=False)

sns.distplot(submission2['PredictedValue'],bins=100)
plt.show()

submission2.head()


# # LightGBM

# In[ ]:


import lightgbm as lgb
def f2_score_func(y,t):
    t=t.get_label()
    y_bin=[1.0 if x>=0.5 else 0.0 for x in y]
    return 'f2',fbeta_score(t,y_bin,2),True

def recall_score_func(y,t):
    t=t.get_label()
    y_bin=[1.0 if x>=0.5 else 0.0 for x in y]
    return 'recall',recall_score(t,y_bin),True

def train_lgb_model(lgb_train,lgb_val,lgb_params={},boosting_rounds=100,plot_perf=False,verbose_eval=True):

    eval_history={}
    record_eval=lgb.callback.record_evaluation(eval_history)
    early_stopping=lgb.early_stopping(5,False)
    callbacks=[record_eval,early_stopping]

    lgb_model=lgb.train(
        lgb_params,lgb_train,num_boost_round=boosting_rounds,
        valid_sets=lgb_val,callbacks=callbacks,verbose_eval=verbose_eval,feval=f2_score_func)
    
    if(plot_perf==True):
        fig,ax=plt.subplots(1,2,figsize=(15,7))
        xgb.plot_importance(lgb_model,ax=ax[0])
        ax[1].plot([*eval_history['train']['auc']],label='Train AUC',color='blue')
        ax[1].plot([*eval_history['eval']['auc']],label='Test AUC',color='red')
        ax[1].legend(loc='best')
        ax[1].set_title("Training Performance")
        ax[1].set_xlabel("Rounds")
        ax[1].set_ylabel("AUC")
        plt.show()
    
    return lgb_model

def lgb_cv_training(X,Y,cv=5,lgb_params={},verbose_eval=True):
    train_scores={}
    test_scores={}
    for ind in range(cv):
        random_state=np.random.randint(100)
        train_x,test_x,train_y,test_y=train_test_split(X,Y,stratify=Y,test_size=test_perc,random_state=random_state)
        train_x,val_x,train_y,val_y=train_test_split(train_x,train_y,stratify=train_y,test_size=val_perc/(val_perc+train_perc),random_state=random_state)

        lgb_train=lgb.Dataset(train_x,label=train_y,feature_name=feat_names)
        lgb_val=lgb.Dataset(val_x,label=val_y,feature_name=feat_names)
        
        lgb_model=train_lgb_model(
            lgb_train,lgb_val,lgb_params,boosting_rounds=1000,verbose_eval=verbose_eval)
        
        train_matrix=train_x.copy()
        pred=lgb_model.predict(train_matrix)
        binary_pred=[1 if x>=0.5 else 0 for x in pred]
        acc_score=balanced_accuracy_score(train_y,binary_pred)*100
        auc_score=roc_auc_score(train_y,pred)
        f2_score=recall_score(train_y,binary_pred)
        train_scores[ind+1]={
            'acc':acc_score,
            'auc':auc_score,
            'f2':f2_score
        }
        
        test_matrix=test_x.copy()
        pred=lgb_model.predict(test_matrix)
        binary_pred=[1 if x>=0.5 else 0 for x in pred]
        acc_score=balanced_accuracy_score(test_y,binary_pred)*100
        auc_score=roc_auc_score(test_y,pred)
        f2_score=recall_score(test_y,binary_pred)
        test_scores[ind+1]={
            'acc':acc_score,
            'auc':auc_score,
            'f2':f2_score
        }
    return train_scores,test_scores

def lgb_training_and_prediction(X,Y,test_X,cv=5,lgb_params={}):
    
    lgb_final_test=test_X.copy()
    final_predictions=np.zeros((cv,test_X.shape[0]),dtype=np.float32)
    for ind in range(cv):
        print("CV ",ind+1)
        random_state=np.random.randint(100)
        print("Random State: ",random_state)
        train_x,test_x,train_y,test_y=train_test_split(X,Y,stratify=Y,test_size=test_perc,random_state=random_state)
        train_x,val_x,train_y,val_y=train_test_split(train_x,train_y,stratify=train_y,test_size=val_perc/(val_perc+train_perc),random_state=random_state)

        lgb_train=lgb.Dataset(train_x,label=train_y,feature_name=feat_names)
        lgb_val=lgb.Dataset(val_x,label=val_y,feature_name=feat_names)
        
        lgb_model=train_lgb_model(lgb_train,lgb_val,lgb_params,boosting_rounds=100)
        
        train_matrix=train_x.copy()
        pred=lgb_model.predict(train_matrix)
        binary_pred=[1 if x>=0.5 else 0 for x in pred]
        acc_score=balanced_accuracy_score(train_y,binary_pred)*100
        auc_score=roc_auc_score(train_y,pred)
        f2_score=recall_score(train_y,binary_pred)
        print("Training Accuracy Score: {score:.2f}%".format(score=acc_score))
        print("Training AUC Score: {score:.3f}".format(score=auc_score))
        print("Training F2 Score: {score:.3f}".format(score=f2_score))
        print()
        test_matrix=test_x.copy()
        pred=lgb_model.predict(test_matrix)
        binary_pred=[1 if x>=0.5 else 0 for x in pred]
        acc_score=balanced_accuracy_score(test_y,binary_pred)*100
        auc_score=roc_auc_score(test_y,pred)
        f2_score=recall_score(test_y,binary_pred)
        print("Testing Accuracy Score: {score:.2f}%".format(score=acc_score))
        print("Testing AUC Score: {score:.3f}".format(score=auc_score))
        print("Testing F2 Score: {score:.3f}".format(score=f2_score))
        print("\n\n")
        
        final_pred=lgb_model.predict(lgb_final_test)
        final_predictions[ind]=final_pred
    return final_predictions


# In[ ]:


lgb_params={
    'scale_pos_weight':95/5,
    'objective':'binary',
    'metrics':'f2',
    'max_depth':10,
    'min_data_in_leaf':15,
    'num_leaves':31,
    'max_bin':255,
    'bagging_fraction':0.6,
    'feature_fraction':0.8
}


# In[ ]:


final_preds=lgb_training_and_prediction(X,Y,test_X,cv=10,lgb_params=lgb_params)
final_pred=final_preds.mean(axis=0)
submission3={'Id':test_id,'PredictedValue':final_pred}
submission3=pd.DataFrame.from_dict(submission3)
submission3.to_csv('submission3.csv',index=False)

sns.distplot(submission3['PredictedValue'],bins=100)
plt.show()

submission3.head()


# In[ ]:


lgb_params={
    'scale_pos_weight':95/5,
    'objective':'binary',
    'metrics':'f2',
    'max_depth':5,
    'min_data_in_leaf':3,
    'num_leaves':15,
    'max_bin':63,
    'bagging_fraction':0.6,
    'feature_fraction':0.8,
    'lambda':5,
    'min_gain_to_split':0
}


# In[ ]:


final_preds=lgb_training_and_prediction(X,Y,test_X,cv=10,lgb_params=lgb_params)
final_pred=final_preds.mean(axis=0)
submission4={'Id':test_id,'PredictedValue':final_pred}
submission4=pd.DataFrame.from_dict(submission4)
submission4.to_csv('submission4.csv',index=False)

sns.distplot(submission4['PredictedValue'],bins=100)
plt.show()

submission4.head()


# In[ ]:


final_pred=(submission1['PredictedValue']+submission2['PredictedValue']+submission3['PredictedValue']+submission4['PredictedValue'])/4.0
submission={'Id':test_id,'PredictedValue':final_pred}
submission=pd.DataFrame.from_dict(submission)
submission.to_csv('submission.csv',index=False)

sns.distplot(submission['PredictedValue'],bins=100)
plt.show()

submission.head()


# In[ ]:




