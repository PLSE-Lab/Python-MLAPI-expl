#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np        # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings                        # To ignore any warnings 
warnings.filterwarnings("ignore")


# In[ ]:


train=pd.read_csv("../input/bank-loan2/madfhantr.csv") 
test=pd.read_csv("../input/bank-loan2/madhante.csv")


# In[ ]:


train_original=train.copy() 
test_original=test.copy()


# In[ ]:


train.columns


# In[ ]:


test.columns


# In[ ]:


train


# In[ ]:


test


# In[ ]:


train.dtypes


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train['Loan_Status'].value_counts()


# In[ ]:


train['Loan_Status'].value_counts(normalize=True)


# In[ ]:


train['Loan_Status'].value_counts(normalize=True).plot.bar()


# In[ ]:


train['Gender'].value_counts(normalize=True).plot.bar(title= 'Gender')
#figsize=(20,10)


# In[ ]:


train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed') 


# In[ ]:



train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History') 


# In[ ]:


train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(24,6), title= 'Dependents') 


# In[ ]:



train['Education'].value_counts(normalize=True).plot.bar(title= 'Education')


# In[ ]:


train['Property_Area'].value_counts(normalize=True).plot.bar(title= 'Property_Area')


# In[ ]:


sns.distplot(train['ApplicantIncome'])


# In[ ]:



train['ApplicantIncome'].plot.box(figsize=(10,5))

train.boxplot(column='ApplicantIncome', by = 'Education') 


# In[ ]:


sns.distplot(train['CoapplicantIncome'])


# In[ ]:


train['CoapplicantIncome'].plot.box(figsize=(10,5))


# In[ ]:


df=train.dropna() 


# In[ ]:


train['LoanAmount'].plot.box(figsize=(16,5)) 
plt.show()


# In[ ]:


Gender=pd.crosstab(train['Gender'],train['Loan_Status']) 
Gender.div(Gender.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[ ]:


Married=pd.crosstab(train['Married'],train['Loan_Status']) 
Dependents=pd.crosstab(train['Dependents'],train['Loan_Status']) 
Education=pd.crosstab(train['Education'],train['Loan_Status']) 
Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status']) 
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.show() 
Education.div(Education.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show()


# In[ ]:



Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status']) 

Credit_History

Credit_History.sum(1)


# In[ ]:



Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status']) 
Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status']) 
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4)) 
plt.show() 
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.show()


# In[ ]:


train.groupby('Loan_Status')['ApplicantIncome'].mean().plot.bar()


# In[ ]:


bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)


# In[ ]:


Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status']) 
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('ApplicantIncome') 


# In[ ]:


bins=[0,1000,3000,42000] 
group=['Low','Average','High'] 
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)


# In[ ]:


Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status']) 
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('CoapplicantIncome') 
P = plt.ylabel('Percentage')


# In[ ]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status']) 
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('Total_Income') 
P = plt.ylabel('Percentage')


# In[ ]:


bins=[0,100,200,700] 
group=['Low','Average','High'] 
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status']) 
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('LoanAmount') 
P = plt.ylabel('Percentage')


# In[ ]:


train=train.drop(['Income_bin','Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)
train['Dependents'].replace('3+', 3,inplace=True) 
test['Dependents'].replace('3+', 3,inplace=True) 
train['Loan_Status'].replace('N', 0,inplace=True) 
train['Loan_Status'].replace('Y', 1,inplace=True)


# In[ ]:


matrix = train.corr() 
f, ax = plt.subplots(figsize=(9, 6)) 
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");


# In[ ]:


train.isnull().sum()


# In[ ]:


train['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
train['Married'].fillna(train['Married'].mode()[0], inplace=True) 
train['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
train['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
train['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True)


# In[ ]:


train['Loan_Amount_Term'].value_counts()


# In[ ]:


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True)
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)
train.isnull().sum()


# In[ ]:


test['Gender'].fillna(train['Gender'].mode()[0], inplace=True) 
test['Dependents'].fillna(train['Dependents'].mode()[0], inplace=True) 
test['Self_Employed'].fillna(train['Self_Employed'].mode()[0], inplace=True) 
test['Credit_History'].fillna(train['Credit_History'].mode()[0], inplace=True) 
test['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0], inplace=True) 
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


# In[ ]:


train['LoanAmount_log'] = np.log(train['LoanAmount']) 
train['LoanAmount_log'].hist(bins=20) 
test['LoanAmount_log'] = np.log(test['LoanAmount'])


# In[ ]:


train=train.drop('Loan_ID',axis=1) 
test=test.drop('Loan_ID',axis=1)


# In[ ]:



X = train.drop('Loan_Status',1) 
Y = train.Loan_Status


# In[ ]:


X=pd.get_dummies(X) 
train=pd.get_dummies(train) 
test=pd.get_dummies(test)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,Y, test_size =0.2)


# In[ ]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
model = LogisticRegression() 
model.fit(x_train, y_train)


# In[ ]:


pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)


# In[ ]:


pred_test = model.predict(test)


# In[ ]:


pred_test


# In[ ]:


submission = pd.read_csv("../input/bank-loan2/sample_submission_49d68Cx.csv")


# In[ ]:


submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']


# In[ ]:


submission['Loan_Status'].replace(0,'N',inplace=True)
submission['Loan_Status'].replace(1,'Y',inplace=True)


# In[ ]:


pd.DataFrame(submission,columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')


# In[ ]:


from sklearn.model_selection import StratifiedKFold
i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,Y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = Y[train_index],Y[test_index]         
    model = LogisticRegression(random_state=1)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test) 
    pred=model.predict_proba(xvl)[:,1]


# In[ ]:


from sklearn import metrics 
fpr, tpr, _ = metrics.roc_curve(yvl,  pred) 
auc = metrics.roc_auc_score(yvl, pred) 
plt.figure(figsize=(12,8)) 
plt.plot(fpr,tpr,label="validation, auc="+str(auc)) 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.legend(loc=4) 
plt.show()


# In[ ]:


submission['Loan_Status']=pred_test
submission['Loan_ID']=test_original['Loan_ID']


# In[ ]:


submission['Loan_Status'].replace(0,'N',inplace=True)
submission['Loan_Status'].replace(1,'Y',inplace=True)


# In[ ]:


pd.DataFrame(submission,columns=['Loan_ID','Loan_Status']).to_csv('Logistic.csv')


# In[ ]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome'] 
test['Total_Income']=test['ApplicantIncome']+test['CoapplicantIncome']
sns.distplot(train['Total_Income']);


# In[ ]:


train['Total_Income_log'] = np.log(train['Total_Income']) 
sns.distplot(train['Total_Income_log']); 
test['Total_Income_log'] = np.log(test['Total_Income'])


# In[ ]:


train['EMI']=train['LoanAmount']/train['Loan_Amount_Term'] 
test['EMI']=test['LoanAmount']/test['Loan_Amount_Term']


# In[ ]:


sns.distplot(train['EMI']); 


# In[ ]:


train['Balance Income']=train['Total_Income']-(train['EMI']*1000) 
test['Balance Income']=test['Total_Income']-(test['EMI']*1000)
sns.distplot(train['Balance Income']);


# In[ ]:


train=train.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1) 
test=test.drop(['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term'], axis=1)


# In[ ]:


X = train.drop('Loan_Status',1)
y = train.Loan_Status
ac = []
algo = []


# In[ ]:


#LogisticRegression
algo.append('logistic regression')
acc=[]
i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,Y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]  
    model = LogisticRegression(random_state=1)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)
    acc.append(score)
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test) 
    pred=model.predict_proba(xvl)[:,1]
ac.append(max(acc))


# In[ ]:


#DecisionTree
from sklearn import tree
algo.append('decision tree')
acc=[]
i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,Y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = tree.DecisionTreeClassifier(random_state=1)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)
    acc.append(score)
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test)
ac.append(max(acc))


# In[ ]:


submission['Loan_Status']=pred_test
submission['Loan_ID'] = test_original['Loan_ID']

submission['Loan_Status'].replace(0,'N',inplace=True)
submission['Loan_Status'].replace(1,'Y',inplace=True)

pd.DataFrame(submission,columns=['Loan_ID','Loan_Status']).to_csv('DecisionTree.csv')


# In[ ]:


#RandomForest
algo.append('random forest')
from sklearn.ensemble import RandomForestClassifier
i=1 
acc=[]
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,Y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = RandomForestClassifier(random_state=1)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)  
    acc.append(score)
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test)
ac.append(max(acc))


# In[ ]:


from sklearn.model_selection import GridSearchCV

paramgrid = {'max_depth': list(range(1,20,2)), 'n_estimators': list(range(1,200,20))}

grid_search = GridSearchCV(RandomForestClassifier(random_state = 1),paramgrid)

from sklearn.model_selection import train_test_split

x_train,x_cv,y_train,y_cv = train_test_split(X,y,test_size = 0.3,random_state = 1)

grid_search.fit(x_train,y_train)


# In[ ]:


grid_search.best_estimator_


# In[ ]:


i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,Y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = RandomForestClassifier(random_state=1, max_depth = 3, n_estimators  =41)    
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test)
    


# In[ ]:


submission['Loan_Status']=pred_test
submission['Loan_ID'] = test_original['Loan_ID']

submission['Loan_Status'].replace(0,'N',inplace=True)
submission['Loan_Status'].replace(1,'Y',inplace=True)

pd.DataFrame(submission,columns=['Loan_ID','Loan_Status']).to_csv('RandomForest.csv')


# In[ ]:


importance = pd.Series(model.feature_importances_, index = X.columns)

importance.plot(kind = 'barh', figsize=(12,6))


# In[ ]:


pip install xgboost


# In[ ]:


from xgboost import XGBClassifier
algo.append('XG boost')
acc=[]
i=1 
kf = StratifiedKFold(n_splits=5,random_state=1,shuffle=True) 
for train_index,test_index in kf.split(X,Y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = X.loc[train_index],X.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = XGBClassifier(n_estimators  =50, max_depth = 4)    
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    acc.append(score)
    print('accuracy_score',score)     
    i+=1 
    pred_test = model.predict(test)
ac.append(max(acc))
    


# In[ ]:


submission['Loan_Status']=pred_test
submission['Loan_ID'] = test_original['Loan_ID']

submission['Loan_Status'].replace(0,'N',inplace=True)
submission['Loan_Status'].replace(1,'Y',inplace=True)

pd.DataFrame(submission,columns=['Loan_ID','Loan_Status']).to_csv('xgboost.csv')


# In[ ]:


print(algo)
print(ac)


# In[ ]:


plt.plot(algo,ac)
plt.title("Accuracy of Algorithms")
plt.show()


# In[ ]:




