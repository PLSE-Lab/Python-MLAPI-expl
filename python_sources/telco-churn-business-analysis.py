#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,accuracy_score,auc,log_loss,roc_auc_score,f1_score
get_ipython().run_line_magic('matplotlib', 'notebook')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Treatment

# In[ ]:


data=pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data.iloc[:6]


# In[ ]:


data.describe().T


# In[ ]:


data['TotalCharges']=data['TotalCharges'].apply(lambda i:np.NaN if i==' ' else float(i))


# In[ ]:


data=data.dropna()
data.info()  


# In[ ]:


data.describe().T


# In[ ]:


data.columns


# # EDA

# In[ ]:


sns.countplot(x=data['Churn'])


# In[ ]:


df=pd.read_csv('../input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
kerugian=df.groupby(by='Churn').sum()['MonthlyCharges']
print(kerugian)
sns.barplot(x=data['Churn'],y=data['MonthlyCharges'],estimator=sum)


# #### From this chart we can see that Telco loss $13900 revenue from Churned Customer

# In[ ]:


sns.boxplot(x='tenure',data=data)


# In[ ]:


sns.boxplot(x='MonthlyCharges',data=data)


# In[ ]:


sns.boxplot(x='TotalCharges',data=data)


# In[ ]:


columnCat=['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 'InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod']
plt.figure(figsize=(30,30))
for item in range(len(columnCat)):
    plt.subplot(4,4,item+1)
    plt.title(columnCat[item])
    sns.countplot(x=data[columnCat[item]],hue=data['Churn'])
    if columnCat[item]=='PaymentMethod':
        plt.xticks(rotation=90)
plt.show()


# # Countplot Summary
# #### Gender=feature gender doesnt have correlation with Churn
# #### Senior Citizen= Senior Citizen has imbalance data and almost half of senior citizen churned
# #### Partner= Customer that doesnt have partner likely will churn although the correlation will be really small
# #### Dependent=Customer that not dependent has higher chance to churn
# ## Telco Service
# #### Phone Service=Imbalance data between customer that has phone service and not
# #### multiple line=No Correlation with Churn
# #### Internet service=Customer that use fiber optic has higher chance to churn
# #### online security=Customer that doesnt have online security service tend to have high Churn rate
# #### online backup=Customer that doesnt have online backup service tend to have high Churn rate
# #### device protection=Customer that doesnt have device protection service tend to have high Churn rate
# #### tech support=Customer that doesnt have tech support service tend to have high Churn rate
# #### Streaming TV=No Correlation with Churn
# #### streaming movies=No Correlation with Churn
# #### contract=The longer the contract the churn rate tend to be decreased
# #### paperless billing=customer that use paperless billing has higher churn rate
# #### payment method=Customer that use electronic check has the highest churn rate than other payment method
# 
# 
# 
# ## Feature that might have strong correlation with target(Churn):
# #### payment method
# #### Contract
# #### online security
# #### internet service
# #### tech support 
# #### Online Backup

# In[ ]:


label= preprocessing.LabelEncoder()
data['Churn']=label.fit_transform(data['Churn'])
data['Churn'].head()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True)


# In[ ]:


columnNum=['tenure','MonthlyCharges','TotalCharges']
plt.figure(figsize=(20,10))
for item in range(0,len(columnNum)):
    plt.title(columnNum[item-1])
    plt.subplot(1,3,item+1)
    sns.distplot(data[data['Churn']==0][columnNum[item]],kde=True,color='blue',bins=20)
    sns.distplot(data[data['Churn']==1][columnNum[item]],kde=True,color='red',bins=20)


# # Distplot Summary
# #### Tenure=New user will likely to churn
# #### Monthly Charges= High monthly charges means higher churn rate
# #### Total Charge=higher total charges of custommer smaller the chance of them to churn

# In[ ]:


plt.scatter(data['TotalCharges'],data['tenure'],c=data['Churn'])
plt.xlabel('TotalCharges')
plt.ylabel('Tenure')


# In[ ]:


plt.scatter(data['tenure'],data['MonthlyCharges'],c=data['Churn'])
plt.xlabel('tenure')
plt.ylabel('MonthlyCharges')


# In[ ]:


plt.scatter(data['TotalCharges'],data['MonthlyCharges'],c=data['Churn'])
plt.xlabel('TotalCharges')
plt.ylabel('MonthlyCharges')


# #### Tenure and Monthly charges dont have any correlation
# #### The longer Customer use telco service, total charges will increased
# #### Client that has high monthly charge tend to have high total charges too

# In[ ]:


sns.FacetGrid(data,col='Contract',hue='Churn').map(plt.scatter,'MonthlyCharges','tenure').fig.set_size_inches(15,10)


# #### Customer that has contract month to month and high monthly charges tend will be churn

# In[ ]:


sns.FacetGrid(data,col='Contract',row='PaymentMethod',hue='Churn').map(plt.scatter,'MonthlyCharges','tenure').fig.set_size_inches(15,10)


# #### Customer that has contract month to month tend to be churn but the highest number of churn customer is if they use electronic check as payment method

# In[ ]:


sns.FacetGrid(data,col='StreamingMovies',row='StreamingTV',hue='Churn').map(plt.scatter,'MonthlyCharges','tenure').fig.set_size_inches(15,10)


# In[ ]:


print('Mean Monthly charges internet DSL with streaming movies and TV',data[(data['InternetService']=='DSL')&(data['OnlineBackup']=='No')&(data['DeviceProtection']=='No')&(data['OnlineSecurity']=='No')&(data['TechSupport']=='No')&(data['StreamingMovies']=='Yes')&(data['StreamingTV']=='Yes')]['MonthlyCharges'].mean())
print('Mean Monthly charges internet  with streaming movies and TV',data[(data['InternetService']=='Fiber optic')&(data['OnlineBackup']=='No')&(data['DeviceProtection']=='No')&(data['OnlineSecurity']=='No')&(data['TechSupport']=='No')&(data['StreamingMovies']=='Yes')&(data['StreamingTV']=='Yes')]['MonthlyCharges'].mean())
print(data.groupby(by=['InternetService','OnlineBackup','OnlineSecurity','TechSupport','StreamingMovies','StreamingTV','DeviceProtection'])['MonthlyCharges'].mean())
print(data[(data['InternetService']=='Fiber optic')&(data['OnlineBackup']=='No')&(data['DeviceProtection']=='No')&(data['OnlineSecurity']=='No')&(data['TechSupport']=='No')&(data['StreamingMovies']=='Yes')&(data['StreamingTV']=='No')]['MonthlyCharges'].mean())


# In[ ]:


sns.FacetGrid(data,col='InternetService',row='TechSupport',hue='Churn').map(plt.scatter,'MonthlyCharges','tenure').fig.set_size_inches(15,10)


# In[ ]:


sns.FacetGrid(data,col='OnlineBackup',row='OnlineSecurity',hue='Churn').map(plt.scatter,'MonthlyCharges','tenure').fig.set_size_inches(15,10)


# # Conclusion From EDA
# ### payment method : telco need to reduce payment method using electronic check
# ### Contract : telco need to reduce customer with month to month contract
# ## Telco Feature
# #### Fiber Optic really expensive,either decrease the cost or focus on DSL Service
# #### Telco need to increase the number of customer that using services below:
# #### online security
# #### internet service
# #### tech support
# #### Online Backup
# 

# # Machine Learning

# In[ ]:


for item in columnCat:
    data[item]=label.fit_transform(data[item])


# In[ ]:


data=data.drop('customerID',axis=1)
data.iloc[:6]


# # Train Test Split

# In[ ]:


xtrain,xtes,ytrain,ytes=train_test_split(data.drop('Churn',axis=1),data['Churn'],test_size=0.30,random_state=101)


# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rfc=RandomForestClassifier(n_estimators=100,random_state=50)
rfc.fit(xtrain,ytrain)


# In[ ]:


coef1=pd.Series(rfc.feature_importances_,xtrain.columns).sort_values(ascending=False)
coef1.plot(kind='bar',title='Feature Importances')


# In[ ]:


predictTesRFC=rfc.predict(xtes)
predictProbRFC=rfc.predict_proba(xtes)


# In[ ]:


conRFC=pd.DataFrame(data=confusion_matrix(ytes,predictTesRFC),columns=['P No','P Yes'],index=['A No','A Yes']);
conRFC


# In[ ]:


print(classification_report(ytes,predictTesRFC))


# In[ ]:


preds=predictProbRFC[:,1]
fpr,tpr,threshold=roc_curve(ytes,preds)
roc_auc=auc(fpr,tpr)

plt.title('Reveiver Operating Charateristic')
plt.plot(fpr,tpr,'b',label='AUC={}'.format(round(roc_auc,2)))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


print('log_loss=',roc_auc_score(ytes,predictProbRFC[:,1]))


# # Decision Tree

# In[ ]:


from sklearn import tree


# In[ ]:


DCT=tree.DecisionTreeClassifier()
DCT.fit(xtrain,ytrain)


# In[ ]:


coef1=pd.Series(DCT.feature_importances_,xtrain.columns).sort_values(ascending=False)
coef1.plot(kind='bar',title='Feature Importances')


# In[ ]:


predictTesDCT=rfc.predict(xtes)
predictProbDCT=rfc.predict_proba(xtes)


# In[ ]:


conDCT=pd.DataFrame(data=confusion_matrix(ytes,predictTesDCT),columns=['P No','P Yes'],index=['A No','A Yes']);
conDCT


# In[ ]:


print(classification_report(ytes,predictTesDCT))


# In[ ]:


preds=predictProbDCT[:,1]
fpr,tpr,threshold=roc_curve(ytes,preds)
roc_auc=auc(fpr,tpr)

plt.title('Reveiver Operating Charateristic')
plt.plot(fpr,tpr,'b',label='AUC={}'.format(round(roc_auc,2)))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


print('log_loss=',log_loss(ytes,predictProbDCT[:,1]))


# # XG Boost

# In[ ]:


import xgboost as xgb


# In[ ]:


xgb=xgb.XGBClassifier()
xgb.fit(xtrain,ytrain)


# In[ ]:


coef1=pd.Series(xgb.feature_importances_,xtrain.columns).sort_values(ascending=False)
coef1.plot(kind='bar',title='Feature Importances')


# In[ ]:


predictTesXGB=xgb.predict(xtes)
predictProbXGB=xgb.predict_proba(xtes)


# In[ ]:


conXG=pd.DataFrame(data=confusion_matrix(ytes,predictTesXGB),columns=['P No','P Yes'],index=['A No','A Yes']);
conXG


# In[ ]:


print(classification_report(ytes,predictTesXGB))


# In[ ]:


preds=predictProbXGB[:,1]
fpr,tpr,threshold=roc_curve(ytes,preds)
roc_auc=auc(fpr,tpr)

plt.title('Reveiver Operating Charateristic')
plt.plot(fpr,tpr,'b',label='AUC={}'.format(round(roc_auc,2)))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


print('log_loss=',roc_auc_score(ytes,predictProbXGB[:,1]))


# # Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


logmodel = LogisticRegression(solver='lbfgs',max_iter=1000)
logmodel.fit(xtrain,ytrain)


# In[ ]:


predictTesLR=logmodel.predict(xtes)
predictProbLR=logmodel.predict_proba(xtes)


# In[ ]:


con=pd.DataFrame(data=confusion_matrix(ytes,predictTesLR),columns=['P No','P Yes'],index=['A No','A Yes']);
con


# In[ ]:


print(classification_report(ytes,predictTesLR))


# In[ ]:


preds=predictProbLR[:,1]
fpr,tpr,threshold=roc_curve(ytes,preds)
roc_auc=auc(fpr,tpr)

plt.title('Reveiver Operating Charateristic')
plt.plot(fpr,tpr,'b',label='AUC={}'.format(round(roc_auc,2)))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


from sklearn.model_selection import KFold

K=10
kf=KFold(n_splits=K,shuffle=True,random_state=42)
target=data['Churn']
data=data.drop('Churn',axis=1)


# In[ ]:


def calc_train_error(xtrain,ytrain,model):
    predictions=model.predict(xtrain)
    predictProba=model.predict_proba(xtrain)
    accuracy=accuracy_score(ytrain,predictions)
    f1=f1_score(ytrain,predictions,average='macro')
    roc_auc=roc_auc_score(ytrain,predictProba[:,1])
    logloss=log_loss(ytrain,predictProba[:,1])
    report=classification_report(ytrain,predictions)
    return {
        'report':report,
        'f1':f1,
        'roc':roc_auc,
        'accuracy':accuracy,
        'logloss':logloss
    }
def calc_validation_error(xtes,ytes,model):
    predictions=model.predict(xtes)
    predictProba=model.predict_proba(xtes)
    accuracy=accuracy_score(ytes,predictions)
    f1=f1_score(ytes,predictions,average='macro')
    roc_auc=roc_auc_score(ytes,predictProba[:,1])
    logloss=log_loss(ytes,predictProba[:,1])
    report=classification_report(ytes,predictions)
    return {
        'report':report,
        'f1':f1,
        'roc':roc_auc,
        'accuracy':accuracy,
        'logloss':logloss
    }
def calc_metrics(xtrain,ytrain,xtes,ytes,model):
    model.fit(xtrain,ytrain)
    train_error=calc_train_error(xtrain,ytrain,model)
    validation_error=calc_validation_error(xtes,ytes,model)
    return train_error,validation_error


# In[ ]:


train_errors=[]
validation_errors=[]
for train_index,val_index in kf.split(data,target):
    #Split Data
    xtrain,x_val=data.iloc[train_index],data.iloc[val_index]
    ytrain,y_val=target.iloc[train_index],target.iloc[val_index]
    
    #calculate errors
    train_error,val_error=calc_metrics(xtrain,ytrain,x_val,y_val,logmodel)
    
    #append to appropiate list
    train_errors.append(train_error)
    validation_errors.append(val_error)
dfLR = []
for tr,val in zip(train_errors, validation_errors):
    dfLR.append([tr['f1'], val['f1'], tr['roc'], val['roc'],
                  tr['logloss'], val['logloss'],tr['accuracy'], val['accuracy']])
dfLR = pd.DataFrame(dfLR, columns=['f1 train','f1 test','Train ROC AUC','Test ROC AUC',
                                       'Train log_loss','Test log_loss','Train accuracy',
                                       'Test accuracy'])
dfLR


# In[ ]:


print('log_loss=',roc_auc_score(ytes,predictProbLR[:,1]))


# # GridSearch and evaluate the model

# In[ ]:


from sklearn.model_selection import GridSearchCV

parameters={'class_weight':({0:1,1:3},{0:1,1:5},{0:1,1:7}),
            'min_samples_leaf':(15,20,25,30)}
rfc=RandomForestClassifier(n_estimators=100,random_state=101)
dt=GridSearchCV(rfc,parameters,
               scoring='roc_auc',cv=5)
dt.fit(xtrain,ytrain)
rfc=dt.best_estimator_
dt.best_estimator_


# In[ ]:


train_errors=[]
validation_errors=[]
for train_index,val_index in kf.split(data,target):
    #Split Data
    xtrain,x_val=data.iloc[train_index],data.iloc[val_index]
    ytrain,y_val=target.iloc[train_index],target.iloc[val_index]
    
    #calculate errors
    train_error,val_error=calc_metrics(xtrain,ytrain,x_val,y_val,rfc)
    
    #append to appropiate list
    train_errors.append(train_error)
    validation_errors.append(val_error)
dfRFC = []
for tr,val in zip(train_errors, validation_errors):
    dfRFC.append([tr['f1'], val['f1'], tr['roc'], val['roc'],
                  tr['logloss'], val['logloss'],tr['accuracy'], val['accuracy']])
dfRFC = pd.DataFrame(dfRFC, columns=['f1 train','f1 test','Train ROC AUC','Test ROC AUC',
                                       'Train log_loss','Test log_loss','Train accuracy',
                                       'Test accuracy'])
dfRFC


# In[ ]:


DCT


# In[ ]:


parameters={'class_weight':({0:1,1:3},{0:1,1:5},{0:1,1:7},{0:1,1:10}),
            'min_samples_leaf':(90,100,110)}
dt=GridSearchCV(DCT,parameters,
               scoring='roc_auc',
               cv=5)
dt.fit(xtrain,ytrain)
DCT=dt.best_estimator_
dt.best_estimator_


# In[ ]:


train_errors=[]
validation_errors=[]
for train_index,val_index in kf.split(data,target):
    #Split Data
    xtrain,x_val=data.iloc[train_index],data.iloc[val_index]
    ytrain,y_val=target.iloc[train_index],target.iloc[val_index]
    
    print(len(x_val),len(xtrain)+len(x_val))
    
    
    #calculate errors
    train_error,val_error=calc_metrics(xtrain,ytrain,x_val,y_val,DCT)
    
    #append to appropiate list
    train_errors.append(train_error)
    validation_errors.append(val_error)
dfDCT = []
for tr,val in zip(train_errors, validation_errors):
    dfDCT.append([tr['f1'], val['f1'], tr['roc'], val['roc'],
                  tr['logloss'], val['logloss'],tr['accuracy'], val['accuracy']])
dfDCT = pd.DataFrame(dfDCT, columns=['f1 train','f1 test','Train ROC AUC','Test ROC AUC',
                                       'Train log_loss','Test log_loss','Train accuracy',
                                       'Test accuracy'])
dfDCT


# In[ ]:


from sklearn.model_selection import GridSearchCV

parameters={'max_depth':(1,2,3),
            'min_child_weight':(13,15,17,20)}
dt=GridSearchCV(xgb,parameters,
               scoring='roc_auc',
               cv=5)
dt.fit(xtrain,ytrain)
xgb=dt.best_estimator_
dt.best_estimator_


# In[ ]:


train_errors=[]
validation_errors=[]
for train_index,val_index in kf.split(data,target):
    #Split Data
    xtrain,x_val=data.iloc[train_index],data.iloc[val_index]
    ytrain,y_val=target.iloc[train_index],target.iloc[val_index]
    
    print(len(x_val),len(xtrain)+len(x_val))
    
    
    #calculate errors
    train_error,val_error=calc_metrics(xtrain,ytrain,x_val,y_val,xgb)
    
    #append to appropiate list
    train_errors.append(train_error)
    validation_errors.append(val_error)
dfXGB = []
for tr,val in zip(train_errors, validation_errors):
    dfXGB.append([tr['f1'], val['f1'], tr['roc'], val['roc'],
                  tr['logloss'], val['logloss'],tr['accuracy'], val['accuracy']])
dfXGB = pd.DataFrame(dfXGB, columns=['f1 train','f1 test','Train ROC AUC','Test ROC AUC',
                                       'Train log_loss','Test log_loss','Train accuracy',
                                       'Test accuracy'])
dfXGB


# In[ ]:


outside = ['f1', 'f1', 'f1','f1', 'f1',
          'f1','f1','f1','f1','f1','f1','f1', 'ROC_AUC','ROC_AUC', 'ROC_AUC',
          'ROC_AUC','ROC_AUC','ROC_AUC', 'ROC_AUC','ROC_AUC','ROC_AUC','ROC_AUC','ROC_AUC','ROC_AUC','logloss',
          'logloss','logloss',
          'logloss','logloss','logloss','logloss','logloss','logloss','logloss','logloss','logloss','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy','accuracy']
inside = [1,2,3,4,5,6,7,8,9,10,'Avg','Std', 1,2,3,4,5,6,7,8,9,10,'Avg','Std', 1,2,3,4,5,6,7,8,9,10,'Avg','Std', 1,2,3,4,5,6,7,8,9,10,'Avg','Std']
hier_index = list(zip(outside, inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)
hier_index


# In[ ]:


f1=[]
roc=[]
logloss=[]
accuracy=[]
kol = {
    'f1' : 'f1 test',
    'ROC_AUC' : 'Test ROC AUC',
    'logloss' : 'Test log_loss',
    'accuracy' : 'Test accuracy'
}
for item1,item2,item3,item4 in zip(dfRFC.values,dfXGB.values,dfDCT.values,dfLR.values):
    f1.append([item1[1],item2[1],item3[1],item4[1]])
    roc.append([item1[3],item2[3],item3[3],item4[3]])
    logloss.append([item1[5],item2[5],item3[5],item4[5]])
    accuracy.append([item1[7],item2[7],item3[7],item4[7]])

for i,j in zip([f1,roc,logloss,accuracy], ['f1','ROC_AUC','logloss','accuracy']):
    i.append([dfRFC[kol[j]].mean(), dfXGB[kol[j]].mean(),dfDCT[kol[j]].mean(),dfLR[kol[j]].mean()])
    i.append([dfRFC[kol[j]].std(), dfXGB[kol[j]].std(),dfDCT[kol[j]].std(),dfLR[kol[j]].std()])
    
dfEval = pd.concat([pd.DataFrame(f1),pd.DataFrame(roc),pd.DataFrame(logloss),pd.DataFrame(accuracy)], axis=0)
dfEval.columns = ['RFC','XGB','DCT','LR']
dfEval.index = hier_index
dfEval


# In[ ]:


for item in ['ROC_AUC', 'accuracy', 'f1', 'logloss']:
    print('Average of {}'.format(item))
    print(dfEval.loc[item].loc['Avg'])


# #### From The Table we can see XGB perform better than RFC and DT and slightly better than Logistic Regression

# # Test Evaluation XGBOOST with GridSearch

# In[ ]:


xgb.fit(xtrain,ytrain)


# In[ ]:


coef1=pd.Series(xgb.feature_importances_,xtrain.columns).sort_values(ascending=False)
coef1.plot(kind='bar',title='Feature Importances')


# In[ ]:


predictTesXGB=xgb.predict(xtes)
predictProbXGB=xgb.predict_proba(xtes)


# In[ ]:


conXG=pd.DataFrame(data=confusion_matrix(ytes,predictTesXGB),columns=['P No','P Yes'],index=['A No','A Yes']);
conXG


# In[ ]:


print(classification_report(ytes,predictTesXGB))


# In[ ]:


preds=predictProbXGB[:,1]
fpr,tpr,threshold=roc_curve(ytes,preds)
roc_auc=auc(fpr,tpr)

plt.title('Reveiver Operating Charateristic')
plt.plot(fpr,tpr,'b',label='AUC={}'.format(round(roc_auc,2)))
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
# plt.xlim([0,1])
# plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:


print('log_loss=',roc_auc_score(ytes,predictProbXGB[:,1]))


# In[ ]:


from sklearn.model_selection import learning_curve

train_sizes,train_scores,test_scores=learning_curve(estimator=xgb,
                                                   X=data,
                                                   y=target,
                                                   train_sizes=np.linspace(0.3,1.0,5),
                                                   cv=10,
                                                   scoring='roc_auc')
print('\nTrain Scores:')
print(train_scores)
#Mean value of accuracy against training data
train_mean=np.mean(train_scores,axis=1)
print('\ntrain Mean: ')
print(train_mean)
print('\nTrain Size: ')
print(train_sizes)
#Standard deviation of training accuracy per number of training samples
train_std=np.std(train_scores,axis=1)
print('\nTrain Std: ')
print(train_std)

#Same as data above for test data
test_mean=np.mean(test_scores,axis=1)
test_std=np.std(test_scores,axis=1)
print('\nTest Scores:')
print(test_scores)
print('\nTest Mean: ')
print(test_mean)
print('\nTest Std: ')
print(test_std)

#Plot training accuracies
plt.plot(train_sizes,train_mean,color='red',marker='o',label='Training Accuracy')
#Plot the variances of training accuracies
plt.fill_between(train_sizes,
                train_mean+train_std,
                train_mean-train_std,
                alpha=0.15,color='red')
#Plot for test data as training data
plt.plot(train_sizes,test_mean,color='blue',linestyle='--',marker='s',
        label='Test Accuracy')
plt.fill_between(train_sizes,
                test_mean+test_std,
                test_mean-test_std,
                alpha=0.15,color='blue')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# # Business Solution 1
# #### Let's say that our company want to know best recall from Churned Client because the cost for promotion is near zero or zero we assume budget=50000

# In[ ]:


listItem=[]
for item1,item2,item3 in zip(tpr,fpr,threshold):
    listItem.append([item1,item2,item3])
dftpr=pd.DataFrame(columns=['TPR','FPR','Threshold'],data=listItem)
# dftpr[dftpr['Threshold']<0.0382987]
dftpr


# In[ ]:


predictTreshold=[]
for item in predictProbXGB:
    if item[1]>=0.038299:
        predictTreshold.append(1)
    else:
        predictTreshold.append(0)
predictTreshold[:5]


# In[ ]:


conXG=pd.DataFrame(data=confusion_matrix(ytes,predictTreshold),columns=['P No','P Yes'],index=['A No','A Yes']);
conXG


# In[ ]:


print(classification_report(ytes,predictTreshold))


# In[ ]:


datatest=xtes
datatest['Churn']=ytes
kerugian=datatest.groupby(by='Churn').sum()['MonthlyCharges']
print(kerugian)
sns.barplot(x=target,y=xtes['MonthlyCharges'],estimator=sum)


# In[ ]:


listItem=[]
for item in zip(predictTreshold,ytes,data['TotalCharges'],data['MonthlyCharges']):
    listItem.append([item[0],item[1],item[2],item[3]])
dfDesc=pd.DataFrame(columns=['Predict','Actual','Total Charges','Monthly Charges'],data=listItem)
dfDesc.head()


# In[ ]:


dfDesc.groupby(by=['Predict','Actual']).sum()


# In[ ]:


# Budget=(kerugian[1]/sum(predictTreshold))
Budget=50000/sum(predictTreshold)
print('Budget Promotion Per Customer: ',Budget)


# In[ ]:


sns.countplot(x=dfDesc['Predict'],hue=dfDesc['Actual'])


# In[ ]:


sns.barplot(x=dfDesc['Predict'],y=dfDesc['Monthly Charges'],estimator=sum,hue=dfDesc['Actual'])


# In[ ]:


sns.barplot(x=dfDesc['Predict'],y=dfDesc['Monthly Charges'],estimator=sum)


# In[ ]:


sns.barplot(x=dfDesc['Actual'],y=dfDesc['Monthly Charges'],estimator=sum)


# # Business Solution 2
# #### The cost for Churned Client is so high so we need precision for Churned Client

# In[ ]:


dftpr[dftpr['FPR']<0.4]


# In[ ]:


predictTreshold=[]
for item in predictProbXGB:
    if item[1]>=0.54:
        predictTreshold.append(1)
    else:
        predictTreshold.append(0)
predictTreshold[:5]


# In[ ]:


conXG=pd.DataFrame(data=confusion_matrix(ytes,predictTreshold),columns=['P No','P Yes'],index=['A No','A Yes']);
conXG


# In[ ]:


print(classification_report(ytes,predictTreshold))


# In[ ]:


listItem=[]
for item in zip(predictTreshold,ytes,data['TotalCharges'],data['MonthlyCharges']):
    listItem.append([item[0],item[1],item[2],item[3]])
dfDesc=pd.DataFrame(columns=['Predict','Actual','Total Charges','Monthly Charges'],data=listItem)
dfDesc.head()


# In[ ]:


dfDesc.groupby(by=['Predict','Actual']).sum()


# In[ ]:


# Budget=(kerugian[1]/sum(predictTreshold))
Budget=50000/sum(predictTreshold)
print('Budget Promotion per Customer: ',Budget)


# In[ ]:


sns.countplot(x=dfDesc['Predict'],hue=dfDesc['Actual'])


# In[ ]:


sns.barplot(x=dfDesc['Predict'],y=dfDesc['Monthly Charges'],estimator=sum,hue=dfDesc['Actual'])


# In[ ]:


sns.barplot(x=dfDesc['Predict'],y=dfDesc['Monthly Charges'],estimator=sum)


# In[ ]:


sns.barplot(x=dfDesc['Actual'],y=dfDesc['Monthly Charges'],estimator=sum)

