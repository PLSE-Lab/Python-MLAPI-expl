""" Trail various ML alorithms, first we'll get the results with no tuning and then find out the results with Tuning. 
    Use StandardScaler to scale the data and also Onehotencoder to convert categorical columns to separate columns. 
    Use a Column Transformer to apply the various transformations. 
    Also split the data into 3 data sets - Training, Validation and Test. 
    Test Data will be used for final hypertuned best parameters and identify the one with the lowest loss. 
"""

#Import Necessary Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss

# Read Input Data
heart = pd.read_csv('../input/heart.csv')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
heart[heart.thal == 0] = 1
target = heart['target'].values
features = heart.drop('target',axis=1)

heart.shape
target.shape
features.shape

#Perform EDA
#heart.head()
#heart.describe()
#heart.info()
#heart.corr()

#Visual EDA Co-relation
#heart.style.background_gradient()
#plt.show()

#Preprocessing of Data
ct = ColumnTransformer([
    ('scaler',StandardScaler(),['age','trestbps','chol','thalach','oldpeak']),
    ('encoder',OneHotEncoder(),['sex','cp','fbs','restecg','exang','slope','ca','thal'])])
features_new = ct.fit_transform(features)
#X_train_new = ct.fit_transform(X_train)

#Create Training,Validation & Test Datasets
X_train,X_test,y_train,y_test = train_test_split(features_new,target,test_size=0.20,random_state=1)
X_train_red,X_valid,y_train_red,y_valid = train_test_split(X_train,y_train,test_size=0.20,random_state=1)

#Model Definitions
models = []
models.append(('LR', LogisticRegression(random_state=1)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier(random_state=1)))
models.append(('NB', GaussianNB()))
models.append(('RF',RandomForestClassifier(random_state=1)))
models.append(('XGB',xgb.XGBClassifier(seed=1)))

results = pd.DataFrame(columns=['Model','CV Score','Train Accuracy','Validation Accuracy','Train Logloss','Valid Logloss'])

for index,model_detail in enumerate(models):
    model_name= model_detail[0]
    classifier = model_detail[1]        
    cv = cross_val_score(classifier,X_train_red,y_train_red,cv=5,n_jobs=-1)
    classifier.fit(X_train_red,y_train_red)
    y_pred = classifier.predict(X_train_red)
    y_predprob = classifier.predict_proba(X_train_red)
    
    score = (accuracy_score(y_train_red, y_pred).round(2))*100
    Train_logloss = log_loss(y_train_red, y_predprob[:,1])

    y_pred1 = classifier.predict(X_valid)
    y_predprob1 = classifier.predict_proba(X_valid)
    
    score1 = (accuracy_score(y_valid, y_pred1).round(2))*100
    Valid_logloss = log_loss(y_valid, y_predprob1[:,1])
    
    results.at[index,'Model'] = model_name
    results.at[index,'CV Score'] = cv.mean().round(2)*100
    results.at[index,'Train Accuracy'] = score
    results.at[index,'Validation Accuracy'] = score1
    results.at[index,'Train Logloss'] = Train_logloss
    results.at[index,'Valid Logloss'] = Valid_logloss
    
#Parameter Tuning
paramlr = {'penalty' : ['l2'],'C' : [0.25]}   ##'penalty' : ['l2'],'C' : [0.25],'tol':np.logspace(-1,1,20)}
paramlda = {'solver':['svd']} ##{'solver':['svd','lsqr'],'n_components' : np.arange(3,15,1)}
paramknn = {'n_neighbors':[8],'weights':['distance'],'algorithm':['ball_tree'],'p':[1]} ##'n_neighbors':np.arange(8,9,1),'weights':['uniform','distance'],'algorithm':['ball_tree','kd_tree'],'p':[1,2,3]
paramdtc = {"criterion": ['gini'],"max_depth": [6],"max_features": [10],'min_samples_leaf' :[7]} ##"criterion": ["gini", "entropy"],"max_depth": [5,6,None],"max_features": np.arange(7,10,1),'min_samples_leaf' :np.arange(3,25,1)
paramnb = {} #'priors' :None,'var_smoothing' :1e-09
paramrf = {'n_estimators': [30],'criterion':["gini"],'max_depth':[None],'min_samples_leaf' :[3]} ##'n_estimators': np.arange(15,22,1),'criterion':["gini", "entropy"],'max_depth':[3,4,None],'min_samples_leaf' :np.arange(3,5,1)
paramxgb = {'booster':['gbtree'],'max_depth': [3],'eta':[1],'num_feature':[6,7],'gamma':[1],'subsample':[0.75],'colsample_bytree':[0.9],'n_estimators': [500]}
parameters = [paramlr,paramlda,paramknn,paramdtc,paramnb,paramrf,paramxgb]

results1 = pd.DataFrame(columns=['Model','Train Accuracy','Best score','Valid Accuracy','Train Logloss','Valid Logloss','Best Estimator'])

#Run the different Models
for index,model_detail in enumerate(zip(models,parameters)):
    model = model_detail[0]
    parameter = model_detail[1]
    model_name= model[0]
    classifier1 = model[1]
    cv1 = GridSearchCV(classifier1,parameter,cv=5,scoring ='roc_auc',refit=True,n_jobs=-1)
    cv1.fit(X_train_red,y_train_red)
    y_pred_par = cv1.predict(X_train_red)
    y_predprob_par = cv1.predict_proba(X_train_red)
    
    score_par = (accuracy_score(y_train_red, y_pred_par).round(2))*100
    Train_logloss_par = log_loss(y_train_red, y_predprob_par[:,1])

    y_pred1_par = cv1.predict(X_valid)
    y_predprob1_par = cv1.predict_proba(X_valid)
    
    score1_par = (accuracy_score(y_valid, y_pred1_par).round(2))*100
    Valid_logloss_par = log_loss(y_valid, y_predprob1_par[:,1])
    
    results1.at[index,'Model'] = model_name
    results1.at[index,'Train Accuracy'] = score_par
    results1.at[index,'Best score'] = (cv1.best_score_)
    #results1.at[index,'Best params'] = cv1.best_params_
    results1.at[index,'Best Estimator'] = cv1.best_estimator_
    results1.at[index,'Valid Accuracy'] = score1_par
    results1.at[index,'Train Logloss'] = Train_logloss_par
    results1.at[index,'Valid Logloss'] = Valid_logloss_par
    
print(results)
print(results1)
#cv1.best_params_

final_results = pd.DataFrame(columns=['Prob_LR','Prob_LDA','Prob_KNN','Prob_DTC','Prob_NB','Prob_RF','Prob_XGB','Max_Prob'])
model_logloss = pd.DataFrame(columns=['Model', 'Logloss'])

#for i in range(6):
#    final_results.at[i,'Model'] = models[i][0]

#for i in range(6):
for index,row in results1.iterrows():
    fin_class = row['Best Estimator']
    
    fin_class.fit(X_train,y_train)
    y_pred_test = fin_class.predict(X_test)
    y_predprob_test = fin_class.predict_proba(X_test)
    
    logloss_test = log_loss(y_test, y_predprob_test[:,1])
    
    if row['Model'] == 'LR':
        LR = y_predprob_test[:,1]
    elif row['Model'] == 'LDA':
        LDA = y_predprob_test[:,1]
    elif row['Model'] == 'KNN':
        KNN = y_predprob_test[:,1]
    elif row['Model'] == 'DTC':
        DTC = y_predprob_test[:,1]
    elif row['Model'] == 'NB':
        NB = y_predprob_test[:,1]
    elif row['Model'] == 'RF':
        RF = y_predprob_test[:,1]
    elif row['Model'] == 'XGB':
        XGB = y_predprob_test[:,1]
        
final_results['Prob_LR'] = LR 
final_results['Prob_LDA'] = LDA
final_results['Prob_KNN'] = KNN
final_results['Prob_DTC'] = DTC
final_results['Prob_NB'] = NB
final_results['Prob_RF'] = RF
final_results['Prob_XGB'] = XGB

for i,row in final_results.iterrows():
    print(row.max())
    if row.max() > 0.5:
        final_results.at[i,'Max_Prob'] = row.max()
    else:
        final_results.at[i,'Max_Prob'] = row.min()
    
final_results

        
