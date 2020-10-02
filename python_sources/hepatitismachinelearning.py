#!/usr/bin/env python
# coding: utf-8

# Reference - https://www.kaggle.com/kalpakchepurwar/hepatitisdataeda/notebook

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from time import time


# In[ ]:


df=pd.read_csv('../input/final.csv')
df.head()


# In[ ]:


df.drop('Unnamed: 0',axis=1,inplace=True)
df.head(2)


# In[ ]:


df.isnull().sum()


# In[ ]:


df.describe()


# In[ ]:


df['age'].unique()


# In[ ]:


df['age']=np.where((df['age'] <18) ,'Teenager/Child',
                               np.where((df['age'] >=18) & (df['age'] <=25),'Young',
                                np.where((df['age'] >=25) & (df['age'] <=40),'Adult',
                               'Old')))


# In[ ]:


df['age'].value_counts()


# In[ ]:


df=pd.get_dummies(df)
df.head()


# In[ ]:


df.columns


# In[ ]:


x=df.iloc[:,1:]
y=df['class']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[ ]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()


# In[ ]:


scale=['bilirubin', 'alk_phosphate', 'sgot', 'albumin']
x_train[scale]=sc.fit_transform(x_train[scale])
x_test[scale]=sc.transform(x_test[scale])


# ### Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression
reg=LogisticRegression()


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


params_reg= {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,'max_iter':[100,150,200,250,300]}


# In[ ]:


reg_cv=RandomizedSearchCV(reg,params_reg,cv=10,random_state=42)
reg_cv.fit(x_train,y_train)


# In[ ]:


print(reg_cv.best_score_)
print(reg_cv.best_params_)


# In[ ]:


log_reg=LogisticRegression(max_iter=250,C=0.1)
start_reg=time()
log_reg.fit(x_train,y_train)
end_reg=time()
time_reg=end_reg-start_reg


# In[ ]:


log_train_time = log_reg.score(x_train,y_train)
log_test_time = log_reg.score(x_test,y_test)
print('Training score: ',log_reg.score(x_train,y_train))
print('Testing score: ',log_reg.score(x_test,y_test))
print('Training time: ',time_reg)


# In[ ]:


y_predict_reg=log_reg.predict(x_test)
y_predict_reg


# In[ ]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[ ]:


accuracy_score(y_test,y_predict_reg)


# In[ ]:


cm_reg=confusion_matrix(y_test,y_predict_reg)
cm_reg


# In[ ]:


print(classification_report(y_test,y_predict_reg))


# ### Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[ ]:


params_dt={'criterion':['gini','entropy'],'min_samples_split':np.arange(2,10),'splitter':['best','random'],'max_depth':[2,3,4,5,6],'max_features':['auto','sqrt','log2',None]}


# In[ ]:


dt_cv=RandomizedSearchCV(dt,params_dt,cv=10,random_state=15)
dt_cv.fit(x_train,y_train)


# In[ ]:


print(dt_cv.best_score_)
print(dt_cv.best_params_)


# In[ ]:


decision_tree=DecisionTreeClassifier(splitter='random',min_samples_split=2,max_features='log2',max_depth=6,criterion='entropy',random_state=10)
start_dt=time()
decision_tree.fit(x_train,y_train)
end_dt=time()
time_dt=end_dt-start_dt


# In[ ]:


dt_train_time = decision_tree.score(x_train,y_train)
dt_test_time = decision_tree.score(x_test,y_test)
print('Training score: ',decision_tree.score(x_train,y_train))
print('Testing score: ',decision_tree.score(x_test,y_test))
print('Training time: ',time_dt)


# In[ ]:


y_predict_dt=decision_tree.predict(x_test)
y_predict_dt


# In[ ]:


cm_dt=confusion_matrix(y_test,y_predict_dt)
print(cm_dt)


# In[ ]:


print(classification_report(y_test,y_predict_dt))


# ### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[ ]:


params_rf={'n_estimators':[5,10,15,20,50,100,200,300,400,500],'criterion':['entropy','gini'],'max_depth':[2,3,4,5,6],'max_features':['auto','sqrt','log2',None],'bootstrap':[True,False]}


# In[ ]:


rf_cv=RandomizedSearchCV(rf,params_rf,cv=10,random_state=15)
rf_cv.fit(x_train,y_train)


# In[ ]:


print(rf_cv.best_score_)
print(rf_cv.best_params_)


# In[ ]:


random_forest=RandomForestClassifier(n_estimators=300,max_features='log2',max_depth=5,criterion='gini',bootstrap=False,random_state=0)
start_rf=time()
random_forest.fit(x_train,y_train)
end_rf=time()
time_rf=end_rf-start_rf


# In[ ]:


rf_train_time = random_forest.score(x_train,y_train)
rf_test_time = random_forest.score(x_test,y_test)
print('Training score: ',random_forest.score(x_train,y_train))
print('Testing score: ',random_forest.score(x_test,y_test))
print('Training time: ',time_rf)


# In[ ]:


y_predict_rf=random_forest.predict(x_test)
y_predict_rf


# In[ ]:


cm_rf=confusion_matrix(y_test,y_predict_rf)
print(cm_rf)


# In[ ]:


print(classification_report(y_test,y_predict_rf))


# ### KNN

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[ ]:


params_knn={'n_neighbors':[5,6,7,8,9,10]}


# In[ ]:


knn_cv=RandomizedSearchCV(knn,params_knn,cv=10,random_state=42)
knn_cv.fit(x_train,y_train)
print(knn_cv.best_score_)
print(knn_cv.best_params_)


# In[ ]:


KNN=KNeighborsClassifier(n_neighbors=10)
start_knn=time()
KNN.fit(x_train,y_train)
end_knn=time()
time_knn=end_knn-start_knn


# In[ ]:


knn_train_time = KNN.score(x_train,y_train)
knn_test_time = KNN.score(x_test,y_test)
print('Training score: ',KNN.score(x_train,y_train))
print('Testing score: ',KNN.score(x_test,y_test))
print('Training time: ',time_knn)


# In[ ]:


y_predict_knn=KNN.predict(x_test)
y_predict_knn


# In[ ]:


cm_knn=confusion_matrix(y_test,y_predict_knn)
print(cm_knn)


# In[ ]:


print(classification_report(y_test,y_predict_knn))


# ### Support Vector Classifier

# In[ ]:


from sklearn.svm import SVC
svc=SVC()


# In[ ]:


params_svm={'kernel':['linear','poly','rbf','sigmoid'],'C':list(np.arange(0.1,0.6)),'gamma':[0.0001,0.001,0.01,0.1,1,10,100,0.02,0.03,0.04,0.05],'degree':[1,2,3,4,5,6]}


# In[ ]:


svm_cv=RandomizedSearchCV(svc,params_svm,cv=10,random_state=7)
svm_cv.fit(x_train,y_train)


# In[ ]:


print(svm_cv.best_score_)
print(svm_cv.best_params_)


# In[ ]:


SVM=SVC(kernel='linear',gamma=1,degree=2,C=0.1)
start_svm=time()
SVM.fit(x_train,y_train)
end_svm=time()
time_svm=end_svm-start_svm


# In[ ]:


SVM_train_time = SVM.score(x_train,y_train)
SVM_test_time = SVM.score(x_test,y_test)
print('Training score: ',SVM.score(x_train,y_train))
print('Testing score: ',SVM.score(x_test,y_test))
print('Training time: ',time_svm)


# In[ ]:


y_predict_svm=SVM.predict(x_test)
y_predict_svm


# In[ ]:


cm_svm=confusion_matrix(y_test,y_predict_svm)
print(cm_svm)


# In[ ]:


print(classification_report(y_test,y_predict_svm))


# ### Gaussian Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()


# In[ ]:


start_gnb=time()
gnb.fit(x_train,y_train)
end_gnb=time()
time_gnb=end_gnb-start_gnb


# In[ ]:


gnb_train_time = gnb.score(x_train,y_train)
gnb_test_time = gnb.score(x_test,y_test)
print('Training score: ',gnb.score(x_train,y_train))
print('Testing score: ',gnb.score(x_test,y_test))
print('Training time: ',time_gnb)


# In[ ]:


y_predict_gnb=gnb.predict(x_test)
y_predict_gnb


# In[ ]:


cm_gnb=confusion_matrix(y_test,y_predict_gnb)
print(cm_gnb)


# In[ ]:


print(classification_report(y_test,y_predict_gnb))


# ## Compairing Training Accuracy of Different Models

# In[ ]:


model_training_time = pd.Series(data=[knn_train_time,log_train_time,dt_train_time,rf_train_time,SVM_train_time,gnb_train_time],
                          index=['KNN','Logistic','DecisionTreeClassifier','RandomForestClassifier','Support Vector','Naive Bayes'])
fig= plt.figure(figsize=(10,7))
model_training_time.sort_values().plot.barh()
plt.title('Model Training Accuracy')


# ## Compairing Testing Accuracy of Different Models
# 

# In[ ]:


model_testing_time = pd.Series(data=[knn_test_time,log_test_time,dt_test_time,rf_test_time,SVM_test_time,gnb_test_time],
                          index=['KNN','Logistic','DecisionTreeClassifier','RandomForestClassifier','Support Vector','Naive Bayes'])
fig= plt.figure(figsize=(10,7))
model_testing_time.sort_values().plot.barh()
plt.title('Model Testing Accuracy')


# ## Comparing Confusion Matrix of different Classifiers

# In[ ]:


knn_con = confusion_matrix(y_test, y_predict_knn)
log_con = confusion_matrix(y_test, y_predict_reg)
nb_con = confusion_matrix(y_test, y_predict_gnb)
dtc_con = confusion_matrix(y_test, y_predict_dt)
rf_con = confusion_matrix(y_test, y_predict_rf)
svm_con = confusion_matrix(y_test, y_predict_svm)


plt.figure(figsize=(20,10))
plt.subplot(2,4,1)
plt.title("KNeighborsClassifier")
sns.heatmap(knn_con,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.subplot(2,4,2)
plt.title("LogisticRegression")
sns.heatmap(log_con,annot=True,cmap="Oranges",fmt="d",cbar=False)
plt.subplot(2,4,3)
plt.title("GaussianNB")
sns.heatmap(nb_con,annot=True,cmap="Greens",fmt="d",cbar=False)
plt.subplot(2,4,4)
plt.title("DecisionTreeClassifier")
sns.heatmap(dtc_con,annot=True,cmap="Purples",fmt="d",cbar=False)
plt.subplot(2,4,5)
plt.title("RandomForestClassifier")
sns.heatmap(rf_con,annot=True,cmap="Purples",fmt="d",cbar=False)
plt.subplot(2,4,6)
plt.title("Support Vector Classifier")
sns.heatmap(svm_con,annot=True,cmap="Greens",fmt="d",cbar=False)
plt.show()


# ### Comparing Training times of different Classifiers

# In[ ]:


training_times=[time_dt,time_rf,time_knn,time_gnb,time_svm,time_reg]
algo=['Decision Tree Classifier','Random Forest Classifier','KNN','Gaussian Naive Bayes','Support Vector Classifier','Logistic Regression']


# In[ ]:


plt.figure(figsize=(15,5))
sns.barplot(y=algo,x=training_times,palette='ocean')
plt.xlabel('Training Time')
plt.grid()
plt.show()


# ### Bagging Classifier

# In[ ]:


from sklearn.ensemble import BaggingClassifier


# In[ ]:


df_bagging=DataFrame(columns=['Base Estimator','Training Score','Testing Score','Time Taken','Case Of','Diff(Training-Testing)'])
to_bag=[decision_tree,random_forest,log_reg,KNN,SVM,gnb]
algo_name=['Decision Tree','Random Forest','Logistic Regression','KNN','Support Vector Classifier','Gaussian Naive Bayes Classifier']
for i in range(len(to_bag)):
    case_of=''
    difference=0
    bag=BaggingClassifier(to_bag[i],bootstrap=True,random_state=0)
    start=time()
    bag.fit(x_train,y_train)
    end=time()
    time_taken=end-start
    if bag.score(x_train,y_train)>bag.score(x_test,y_test):
        case_of='Overfitting'
    else:
        case_of='Underfitting'
    difference=bag.score(x_train,y_train)-bag.score(x_test,y_test)
    df_bagging.loc[i]=[algo_name[i],bag.score(x_train,y_train),bag.score(x_test,y_test),time_taken,case_of,difference]
df_bagging


# ### Adaboost Classifier

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


df_adaboost=DataFrame(columns=['Base Estimator','Training Score','Testing Score','Time Taken','Case Of','Diff(Training-Testing)'])
to_bag=[decision_tree,random_forest,log_reg,KNN,SVM,gnb]
to_boost=[decision_tree,random_forest,log_reg,SVM,gnb]
algo_name=['Decision Tree','Random Forest','Logistic Regression','Support Vector Classifier','Gaussian Naive Bayes Classifier']
for i in range(len(to_boost)):
    boost=AdaBoostClassifier(to_boost[i],n_estimators=100,algorithm='SAMME',random_state=7)
    start=time()
    boost.fit(x_train,y_train)
    end=time()
    time_taken=end-start
    if boost.score(x_train,y_train)>boost.score(x_test,y_test):
        case_of='Overfitting'
    else:
        case_of='Underfitting'
    difference=boost.score(x_train,y_train)-boost.score(x_test,y_test)
    df_adaboost.loc[i]=[algo_name[i],boost.score(x_train,y_train),boost.score(x_test,y_test),time_taken,case_of,difference]
df_adaboost


# ### Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
params_gb={'learning_rate':[0.1,0.2,0.3,0.4,0.5,1,2,0.01,0.02,0.05],'n_estimators':[100,150,200,300],
           'max_depth':[2,3,4,5,6],'min_samples_split':list(np.arange(1,10)),'criterion':['friedman_mse','mse','mae']}
GB_cv=RandomizedSearchCV(gb,params_gb,cv=10,random_state=7)
GB_cv.fit(x_train,y_train)


# In[ ]:


print('Best score: ',GB_cv.best_score_)
print('Best parameters: ',GB_cv.best_params_)


# In[ ]:


GB=GradientBoostingClassifier(n_estimators=150,max_depth=2,learning_rate=0.5,min_samples_split=8,criterion='mse')


# In[ ]:


GB.fit(x_train,y_train)
GB.score(x_test,y_test)


# In[ ]:


df_gboost=DataFrame(columns=['Base Estimator','Training Score','Testing Score','Time Taken','Case Of','Diff(Training-Testing)'])
to_boost=[decision_tree,random_forest,log_reg,SVM,gnb]
algo_name=['Decision Tree','Random Forest','Logistic Regression','Support Vector Classifier','Gaussian Naive Bayes Classifier']
for i in range(len(to_boost)):
    gboost=GradientBoostingClassifier(n_estimators=200,max_depth=2,learning_rate=0.2,min_samples_split=8,criterion='mse')
    start=time()
    gboost.fit(x_train,y_train)
    end=time()
    time_taken=end-start
    if gboost.score(x_train,y_train)>boost.score(x_test,y_test):
        case_of='Overfitting'
    else:
        case_of='Underfitting'
    difference=boost.score(x_train,y_train)-boost.score(x_test,y_test)
    df_gboost.loc[i]=[algo_name[i],gboost.score(x_train,y_train),gboost.score(x_test,y_test),time_taken,case_of,difference]
df_gboost


# ### USING SMOTE

# In[ ]:


from imblearn.over_sampling import SMOTE
sm=SMOTE(sampling_strategy=1,k_neighbors=5,random_state=0)
x_train_res,y_train_res=sm.fit_sample(x_train,y_train)


# In[ ]:


print(x_train_res.shape)
print(y_train_res.shape)


# ### Logistic Regression

# In[ ]:


params_reg= {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,'max_iter':[100,150,200,250,300]}


# In[ ]:


reg_cv1=RandomizedSearchCV(reg,params_reg,cv=10,random_state=0)
reg_cv1.fit(x_train_res,y_train_res)
print(reg_cv1.best_score_)
print(reg_cv1.best_params_)


# In[ ]:


LR=LogisticRegression(max_iter=100,C=10,random_state=0)
LR.fit(x_train_res,y_train_res)
print('Training score: ',LR.score(x_train_res,y_train_res))
print('Testing score: ',LR.score(x_test,y_test))


# In[ ]:


y_predict_LR=LR.predict(x_test)
print('CONFUSION MATRIX:  ')
print(confusion_matrix(y_test,y_predict_LR))
print()
print('CLASSIFICATION REPORT: ')
print()      
print(classification_report(y_test,y_predict_LR))


# ### Decision Tree Classifier

# In[ ]:


params_dt={'criterion':['gini','entropy'],'min_samples_split':np.arange(2,10),'splitter':['best','random'],'max_depth':[2,3,4,5,6],'max_features':['auto','sqrt','log2',None]}


# In[ ]:


dt_cv1=RandomizedSearchCV(dt,params_dt,cv=15,random_state=42)
dt_cv1.fit(x_train_res,y_train_res)
print(dt_cv1.best_score_)
print(dt_cv1.best_params_)


# In[ ]:


DT=DecisionTreeClassifier(splitter='best',min_samples_split=7,max_features=None,max_depth=5,criterion='entropy')
DT.fit(x_train_res,y_train_res)
print('Training score: ',DT.score(x_train_res,y_train_res))
print('Testing score: ',DT.score(x_test,y_test))


# In[ ]:


y_predict_DT=DT.predict(x_test)
print('CONFUSION MATRIX:  ')
print(confusion_matrix(y_test,y_predict_DT))
print()
print('CLASSIFICATION REPORT: ')
print()      
print(classification_report(y_test,y_predict_DT))


# ### Random Forest Classifier

# In[ ]:


params_rf={'n_estimators':[5,10,15,20,50,100,200,300,400,500],'criterion':['entropy','gini'],'max_depth':[2,3,4,5,6],'max_features':['auto','sqrt','log2',None],'bootstrap':[True,False]}


# In[ ]:


rf_cv1=RandomizedSearchCV(rf,params_rf,cv=10,random_state=0)
rf_cv1.fit(x_train_res,y_train_res)
print(rf_cv1.best_score_)
print(rf_cv1.best_params_)


# In[ ]:


RF=RandomForestClassifier(n_estimators=500,max_features='log2',max_depth=6,criterion='gini',bootstrap=True)
RF.fit(x_train_res,y_train_res)
print('Training score: ',RF.score(x_train_res,y_train_res))
print('Testing score: ',RF.score(x_test,y_test))


# In[ ]:


y_predict_RF=RF.predict(x_test)
print('CONFUSION MATRIX:  ')
print(confusion_matrix(y_test,y_predict_RF))
print()
print('CLASSIFICATION REPORT: ')
print()      
print(classification_report(y_test,y_predict_RF))


# ### KNN

# In[ ]:


params_knn={'n_neighbors':[5,6,7,8,9,10]}


# In[ ]:


knn_cv1=RandomizedSearchCV(knn,params_knn,cv=10,random_state=0)
knn_cv1.fit(x_train_res,y_train_res)
print(knn_cv1.best_score_)
print(knn_cv1.best_params_)


# In[ ]:


Knn=KNeighborsClassifier(n_neighbors=5)
Knn.fit(x_train_res,y_train_res)
print('Training score: ',RF.score(x_train_res,y_train_res))
print('Testing score: ',RF.score(x_test,y_test))


# In[ ]:


y_predict_KNN=Knn.predict(x_test)
print('CONFUSION MATRIX:  ')
print(confusion_matrix(y_test,y_predict_KNN))
print()
print('CLASSIFICATION REPORT: ')
print()      
print(classification_report(y_test,y_predict_KNN))


# ### Support Vector Classifier

# In[ ]:


params_svm={'kernel':['linear','poly','rbf','sigmoid'],'C':list(np.arange(0.1,0.6)),'gamma':[0.0001,0.001,0.01,0.1,1,10,100,0.02,0.03,0.04,0.05],'degree':[1,2,3,4,5,6]}


# In[ ]:


svm_cv1=RandomizedSearchCV(svc,params_svm,cv=10,random_state=7)
svm_cv1.fit(x_train_res,y_train_res)
print(svm_cv1.best_score_)
print(svm_cv1.best_params_)


# In[ ]:


SVM1=SVC(kernel='poly',gamma=10,degree=3,C=0.1)
SVM1.fit(x_train_res,y_train_res)
print('Training score: ',SVM1.score(x_train_res,y_train_res))
print('Testing score: ',SVM1.score(x_test,y_test))


# In[ ]:


y_predict_SVM1=SVM1.predict(x_test)
print('CONFUSION MATRIX:  ')
print(confusion_matrix(y_test,y_predict_SVM1))
print()
print('CLASSIFICATION REPORT: ')
print()      
print(classification_report(y_test,y_predict_SVM1))


# ### Gaussian Naive Bayes Classifier

# In[ ]:


from sklearn.naive_bayes import GaussianNB
GNB=GaussianNB()
GNB.fit(x_train_res,y_train_res)
print('Training score: ',GNB.score(x_train_res,y_train_res))
print('Testing score: ',GNB.score(x_test,y_test))


# In[ ]:


y_predict_GNB=GNB.predict(x_test)
print('CONFUSION MATRIX:  ')
print(confusion_matrix(y_test,y_predict_GNB))
print()
print('CLASSIFICATION REPORT: ')
print()      
print(classification_report(y_test,y_predict_GNB))


# ### Bagging Classifier 

# In[ ]:


df_bagging_sm=DataFrame(columns=['Base Estimator','Training Score','Testing Score','Time Taken','Case Of','Diff(Training-Testing)'])
to_bag=[DT,RF,LR,Knn,SVM1,GNB]
algo_name=['Decision Tree','Random Forest','Logistic Regression','KNN','Support Vector Classifier','Gaussian Naive Bayes Classifier']
for i in range(len(to_bag)):
    case_of=''
    difference=0
    BAG=BaggingClassifier(to_bag[i],bootstrap=True,random_state=0)
    start=time()
    BAG.fit(x_train_res,y_train_res)
    end=time()
    time_taken=end-start
    if BAG.score(x_train_res,y_train_res)>BAG.score(x_test,y_test):
        case_of='Overfitting'
    else:
        case_of='Underfitting'
    difference=BAG.score(x_train_res,y_train_res)-BAG.score(x_test,y_test)
    df_bagging_sm.loc[i]=[algo_name[i],BAG.score(x_train_res,y_train_res),BAG.score(x_test,y_test),time_taken,case_of,difference]
df_bagging_sm


# ### Adaboost Classifier

# In[ ]:


df_adaboost_sm=DataFrame(columns=['Base Estimator','Training Score','Testing Score','Time Taken','Case Of','Diff(Training-Testing)'])
to_boost=[DT,RF,LR,SVM1,GNB]
algo_name=['Decision Tree','Random Forest','Logistic Regression','Support Vector Classifier','Gaussian Naive Bayes Classifier']
for i in range(len(to_boost)):
    boost=AdaBoostClassifier(to_boost[i],n_estimators=100,algorithm='SAMME',random_state=0)
    start=time()
    boost.fit(x_train_res,y_train_res)
    end=time()
    time_taken=end-start
    if boost.score(x_train_res,y_train_res)>boost.score(x_test,y_test):
        case_of='Overfitting'
    else:
        case_of='Underfitting'
    difference=boost.score(x_train_res,y_train_res)-boost.score(x_test,y_test)
    df_adaboost_sm.loc[i]=[algo_name[i],boost.score(x_train_res,y_train_res),boost.score(x_test,y_test),time_taken,case_of,difference]
df_adaboost_sm


# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gb=GradientBoostingClassifier()
params_gb={'learning_rate':[0.1,0.2,0.3,0.4,0.5,1,2,0.01,0.02,0.05],'n_estimators':[100,150,200,300],
           'max_depth':[2,3,4,5,6],'min_samples_split':list(np.arange(1,10)),'criterion':['friedman_mse','mse','mae']}
GB_cv1=RandomizedSearchCV(gb,params_gb,cv=10,random_state=7)
GB_cv1.fit(x_train_res,y_train_res)


# In[ ]:


print('Best score: ',GB_cv.best_score_)
print('Best parameters: ',GB_cv.best_params_)


# In[ ]:


GB_sm=GradientBoostingClassifier(n_estimators=300,min_samples_split=7,max_depth=3,learning_rate=0.5,criterion='mse')
GB_sm.fit(x_train_res,y_train_res)
print('Training score: ',GB_sm.score(x_train_res,y_train_res))
print('Testing score: ',GB_sm.score(x_test,y_test))


# In[ ]:


df_gboost_sm=DataFrame(columns=['Base Estimator','Training Score','Testing Score','Time Taken','Case Of','Diff(Training-Testing)'])
to_boost=[DT,RF,LR,SVM1,GNB]
algo_name=['Decision Tree','Random Forest','Logistic Regression','Support Vector Classifier','Gaussian Naive Bayes Classifier']
for i in range(len(to_boost)):
    gboost=GradientBoostingClassifier()
    start=time()
    gboost.fit(x_train_res,y_train_res)
    end=time()
    time_taken=end-start
    if gboost.score(x_train_res,y_train_res)>gboost.score(x_test,y_test):
        case_of='Overfitting'
    else:
        case_of='Underfitting'
    difference=gboost.score(x_train_res,y_train_res)-gboost.score(x_test,y_test)
    df_gboost_sm.loc[i]=[algo_name[i],gboost.score(x_train_res,y_train_res),gboost.score(x_test,y_test),time_taken,case_of,difference]
df_gboost_sm


# ## Conclusion

# #### In all the Cases Logistic Regression performed well and got least difference between Training and Testing Score.

# In[ ]:




