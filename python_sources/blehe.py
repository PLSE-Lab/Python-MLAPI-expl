# -*- coding: utf-8 -*-
"""
Created on Sun Jun 04 01:06:39 2017

@author: Kanishka105457
"""


from sklearn.svm import SVC
from pandas import DataFrame,read_csv,notnull,isnull,get_dummies,concat,unique
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier,AdaBoostClassifier,RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold,train_test_split
from sklearn.metrics import accuracy_score,log_loss,mean_squared_error,r2_score,confusion_matrix,classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
seed=7
np.random.seed(seed)
class Sign:
    def load_data(self):
        self.train_csv=read_csv('../input/my-test-3/train.csv',header='infer')
        sss=StratifiedShuffleSplit(self.train_csv['SignFacing (Target)'],n_iter=1,test_size=0.3,random_state=seed)
        for train_index, test_index in sss:
            self.train, self.test = self.train_csv.drop('SignFacing (Target)',axis=1).iloc[train_index,:],self.train_csv.drop('SignFacing (Target)',axis=1).iloc[test_index,:]
            self.Y_train, self.Y_test = self.train_csv['SignFacing (Target)'][train_index].as_matrix(), self.train_csv['SignFacing (Target)'][test_index].as_matrix()
        self.train = self.train.sample(frac=1).reset_index(drop=True)
        self.train_corr=self.train.corr()
        self.train['Source']='train'
        self.test['Source']='test'
        self.df=self.train.append(self.test)
        self.df.loc[:,'SignDist']=np.log(10000/(self.df.loc[:,'SignWidth']*self.df.loc[:,'SignHeight']))
        self.df.loc[:,'SignAspectRatio']=np.log(self.df.loc[:,'SignAspectRatio'])
        self.df.loc[:,'SignWidth']=np.log(self.df.loc[:,'SignWidth'])
        self.df.loc[:,'SignHeight']=np.log(self.df.loc[:,'SignHeight'])
        #del self.df['DetectedCamera']
        del self.df['SignHeight']
        del self.df['SignWidth']
        
        #wrg_f_angle=(self.df.DetectedCamera=='Front')&(self.df.AngleOfSign>300)
        #self.df.loc[wrg_f_angle,'AngleOfSign']=360-self.df.loc[wrg_f_angle,'AngleOfSign']
        #Scaling the angles to a scale of 0 to 1:
        print('Train:',self.train.shape,' Test:',self.test.shape,' DF:',self.df.shape)
        print('Y_train:',self.Y_train,'Y_test',self.Y_test)
        pass
    def preprocess_data(self):
        cat_list=list(self.df.columns[self.df.dtypes=='object'])
        num_list=list(self.df.columns[self.df.dtypes!='object'])
        cat_list.pop(0)
        scale=MinMaxScaler()
        scale.fit(self.df.loc[:,num_list])
        num_df=self.df.loc[:,num_list]
        #pca=PCA(n_components=3)
        cat_temp=get_dummies(self.df[cat_list[0]],prefix=cat_list[0])
        for i_col in range(1,len(cat_list)):
            cat_temp=concat([cat_temp,get_dummies(self.df.loc[:,cat_list[i_col]],prefix=cat_list[i_col])],axis=1)
        self.proc_df=np.concatenate([num_df,cat_temp.as_matrix()],axis=1)
        #pca.fit(self.proc_df)
        #a=pca.transform(self.proc_df)
        #self.proc_df=np.concatenate([a,self.proc_df],axis=1)
        train_index=self.proc_df[:,self.proc_df.shape[1]-1]==1
        test_index=self.proc_df[:,self.proc_df.shape[1]-2]==1
        #vt=VarianceThreshold(threshold=0.01)
        #self.proc_df=vt.fit_transform(self.proc_df)
        self.proc_train=self.proc_df[train_index,:]
        self.proc_test=self.proc_df[test_index,:]
        print('Proc_df:',self.proc_df.shape)
        print('Proc Train:',self.proc_train.shape,' Proc Test:',self.proc_test.shape,' Proc DF:',self.proc_df.shape)
        pass
    def feature_importance(self):
        cf=RandomForestClassifier(n_estimators=1000,random_state=7,criterion='gini',min_samples_split=10,max_features='auto')
#        params={'n_jobs': [-1], 'n_estimators': [1000], 'random_state': [7], 'criterion': ['gini','entropy'], 'min_samples_split': [10],'max_features':['auto']}
#        cf=GridSearchCV(rf,param_grid=params,cv=3,scoring='neg_log_loss',verbose=3)
        cf.fit(self.proc_train,self.Y_train['SignFacing (Target)'])
#        print('GridCV Best Score: ',cf.best_score_,'GridCV',cf.best_params_)
        self.rforest=cf
        return DataFrame(data=cf.feature_importances_)
    def Gradboosting(self):
        grad=GradientBoostingClassifier()
        params={}
        #cf=GridSearchCV(grad,param_grid=params,cv=5,scoring='neg_log_loss',verbose=3)
        grad.fit(self.proc_train,self.Y_train)
        #print('GridCV Best Score: ',cf.best_score_,'GridCV',cf.best_params_)
        self.grad_prediction_test=grad.predict(self.proc_test)
        self.grad_prediction_train=grad.predict(self.proc_train)
        print('Train Confusion Matrix:')
        print(confusion_matrix(self.Y_train,self.grad_prediction_train))
        print('Test Confusion Matrix:')
        print(confusion_matrix(self.Y_test,self.grad_prediction_test))
        print('Train classification report:')
        print(classification_report(self.Y_train,self.grad_prediction_train))
        print('Test classification report:')
        print(classification_report(self.Y_test,self.grad_prediction_test))
        #print('Train log loss: ',log_loss(self.Y_train,self.grad_prediction_train),' and test log_loss: ',log_loss(self.Y_test,self.grad_prediction_test))
        print('Train accuracy: ',accuracy_score(self.Y_train,self.grad_prediction_train),' and test accuracy: ',accuracy_score(self.Y_test,self.grad_prediction_test))
        self.gradtree=grad
    def stack_learners(self,learners,stacker):
        P_train={}
        P_test={}
        C_train={}
        C_test={}
        learns=list(learners.keys())
        for i in learners:
            cf=GridSearchCV(learners[i]['Model'],param_grid=learners[i]['Params'],scoring='neg_log_loss',verbose=3,cv=5)
            cf.fit(self.proc_train,self.Y_train['SignFacing (Target)'])
            P_train[i]=cf.predict_proba(self.proc_train)
            P_test[i]=cf.predict_proba(self.proc_test)
            C_train[i]=cf.predict(self.proc_train)
            C_test[i]=cf.predict(self.proc_test)
        Compiled=P_test[learns[0]]
        #for i in range(0,len(learns)-1):
        #    Compiled=Compiled+P_test[learns[i]]
        Compiled=Compiled
        sub=DataFrame(Compiled,columns=['Front','Left','Rear','Right'])
        sub['Id'] = self.test['Id']
        sub = sub[['Id','Front','Left','Rear','Right']]
        sub.to_csv("stacker_test.csv", index=False)
        print(P_test.keys())
        #self.new_train=DataFrame(data=C)
        #self.new_test=DataFrame(data=D)
        #self.compile_results()
        #print(self.new_train.head())
        #print(stacker)
        #cf=GridSearchCV(stacker['Model'],param_grid=stacker['Params'],scoring='neg_log_loss',verbose=3,cv=10)
        #cf.fit(self.stack_train,self.Y_train['SignFacing (Target)'])
        #print('GridCV Best Score: ',cf.best_score_,'GridCV',cf.best_params_)
        #self.stack_predictions_prob=cf.predict_proba(self.stack_test)
sg=Sign()
sg.load_data()
sg.preprocess_data()
#sg.stack_learners({'XGB':{'Model':xgb.XGBClassifier(),'Params':{'n_estimators':[200],'max_delta_step':[1,2,3],'booster':['gbtree'],'subsample':[1],'learning_rate':[0.1]}}},'No stacker')
#'XGB':{'Model':xgb.XGBClassifier(),'Params':{'n_estimators':[100],'booster':['gbtree'],'subsample':[1],'learning_rate':[0.1]}}
#'RF1':{'Model':RandomForestClassifier(),'Params':{'n_jobs': [-1], 'n_estimators': [500], 'random_state': [7], 'criterion': ['gini'], 'min_samples_split': [5],'class_weight':['balanced']}},'RF':{'Model':RandomForestClassifier(),'Params':{'n_jobs': [-1], 'n_estimators': [500], 'random_state': [7], 'criterion': ['entropy'], 'min_samples_split': [5],'class_weight':['balanced']}},'NN':{'Model':MLPClassifier(),'Params':{'random_state':[seed],'hidden_layer_sizes':[(30,30,30)],'activation':['relu'],'solver':['adam'],'batch_size':[500],'early_stopping':[False]}}}
sg.Gradboosting()
#columns = ['Front','Left','Rear','Right']
#sub = DataFrame(data=sg.xg_predictions_prob, columns=columns)
#sub['Id'] = sg.test['Id']
#sub = sub[['Id','Front','Left','Rear','Right']]
#sub.to_csv("xgb_tuned_v1.csv", index=False)