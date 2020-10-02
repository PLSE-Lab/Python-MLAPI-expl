#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
os.getcwd()


# In[ ]:



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df=pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# # Understanding the data with the help of visualisation

# In[ ]:


# Let's see how the data is distributed

fig=plt.figure(figsize=(25,6))
sns.countplot(x='age',data=df)


# The distribution is rightly skewed. It shows most of the people belongs to age greather than 30.

# In[ ]:



fig=plt.figure(figsize=(25,8))
sns.countplot(x='job',data=df)


# Management, technicians and blue collars jobs are highly targeted for compaign 

# In[ ]:


fig=plt.figure(figsize=(12,5))
sns.countplot(x='marital',data=df)


# Most married people were targeted 

# In[ ]:


sns.distplot(df.balance,hist=True,kde=False)


# Majority of people who were targeted, had balance less than 20000 

# In[ ]:


fig=plt.figure(figsize=(20,20))
ax1=fig.add_subplot(331)
ax2=fig.add_subplot(332)
ax3=fig.add_subplot(333)


sns.countplot(x='loan',data=df,ax=ax1)
ax1.set_title('Loan Taken ')

sns.countplot(x='contact',data=df,ax=ax2)
ax2.set_title('Contact Medium ')

sns.countplot(x='marital',data=df)
ax3.set_title('Marital Staus')


# In[ ]:


sns.pairplot(df,hue='deposit')


# In[ ]:


fig=plt.figure(1,figsize=(18,5))

sns.boxplot(x='job',y='balance',data=df,hue='deposit')


# Retired People and management subscribed to the term deposite have more balance in their account
# 
# Especially in management, people subscribed to term deposit have more than median balance

# In[ ]:


fig=plt.figure(1,figsize=(13,8))

p=sns.boxplot(x='marital',y='balance',data=df,hue='deposit')
p.set_title('Marital Status vs Subscription')


# Married people with higher balance had majority of subscription.
# Divorced people having less balance their accounts.

# In[ ]:


fig=plt.figure(1,figsize=(8,5))

p=sns.boxplot(x='housing',y='balance',data=df,hue='deposit')
p.set_title('housing loan vs Subscription')


# People without housing loan, who have subscribed to the term deposit have more balance 

# In[ ]:



p=sns.barplot(x='contact',y='balance',data=df,hue='deposit')
p.set_title('Medium vs Subscription')


# Telephone was the most influencial medium for reaching out to the customers

# In[ ]:



p=sns.barplot(x='deposit',y='campaign',data=df)
p.set_title('Contact made vs Subscribed')


# In[ ]:


fig=plt.figure(1,figsize=(13,8))

p=sns.boxplot(x='education',y='balance',data=df,hue='deposit')
p.set_title('Contact made vs Subscribed')


# In[ ]:


df['marital/education']=np.nan

lst = [df]

for col in lst:
    col.loc[(col.marital=='single') & (col.education=='primary'),'marital/education']='Single_primary'
    col.loc[(col.marital=='single') & (col.education=='secondary'),'marital/education']='Single_secondary'
    col.loc[(col.marital=='single') & (col.education=='tertiary'),'marital/education']='Single_tertiary'
    col.loc[(col.marital=='married') & (col.education=='primary'),'marital/education']='married_Primary'
    col.loc[(col.marital=='married') & (col.education=='secondary'),'marital/education']='married_secondary'
    col.loc[(col.marital=='married') & (col.education=='tertiary'),'marital/education']='married_tertiary'
    col.loc[(col.marital=='divorced') & (col.education=='primary'),'marital/education']='divorced_primary'
    col.loc[(col.marital=='divorced') & (col.education=='secondary'),'marital/education']='divorced_secondary'
    col.loc[(col.marital=='divorced') & (col.education=='tertiary'),'marital/education']='divorced_tertiary'



    


# In[ ]:



df.head()


# In[ ]:


fig=plt.figure(figsize=(20,7))
sns.barplot(x='marital/education',y='balance',data=df)


# Divorced people with tertiary education has highest balance in their accounts followed by married and single people with same education level.
# 
# 

# In[ ]:


fig=plt.figure(figsize=(20,7))
sns.barplot(x='marital/education',y='balance',data=df,hue='loan')


# Married people were amongst the ones who has taken the most loans in the dataset. 
# 
# 

# In[ ]:


fig=plt.figure(figsize=(5,4))
sns.boxplot(x='deposit',y='duration',data=df)


# In[ ]:


df.groupby(['deposit']).mean()
df.duration.describe()


# The more the call duration, more people are likely to buy the term deposit.

# In[ ]:


fig=plt.figure(figsize=(20,7))
sns.barplot(x='marital/education',y='duration',data=df,hue='loan')


# Single and divorced people have more call duration.
# 
# Specifically, single & Divorced people with primary level of education has more more call duration.

# In[ ]:


fig=plt.figure(figsize=(20,7))
sns.barplot(x='marital/education',y='duration',data=df)


# # Evaluating the correlation between various features

# In[ ]:


from sklearn.preprocessing import StandardScaler,LabelEncoder
copy=df.copy()
copy['deposit'] = LabelEncoder().fit_transform(copy['deposit'])
copy['marital'] = LabelEncoder().fit_transform(copy['marital'])
copy['job'] = LabelEncoder().fit_transform(copy['job'])

copy['education'] = LabelEncoder().fit_transform(copy['education'])
copy['loan'] = LabelEncoder().fit_transform(copy['loan'])
copy['housing'] = LabelEncoder().fit_transform(copy['housing'])


# In[ ]:



copy.head()


# In[ ]:


corr=copy.corr()


# In[ ]:


fig = plt.figure(figsize=(12,8))

sns.heatmap(corr,annot=True)


# Duration is highly corelated with deposit. 
# 
# Loan and housing has impact on subscribing to term deposit.

# # Creating the Models

# In[ ]:




df=df.drop('marital/education',axis=1)


# In[ ]:


from sklearn.preprocessing import StandardScaler,LabelEncoder
df['deposit'] = LabelEncoder().fit_transform(df['deposit'])

df.head()


# In[ ]:


from sklearn.model_selection import train_test_split

X=df.drop('deposit',axis=1)
X=pd.get_dummies(X)
y=df['deposit']


# In[ ]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=75)


# 

# In[ ]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


#Logistic Regression

logreg=LogisticRegression()
logreg.fit(X_train,y_train)
print('# Logistic Regression Models Results\n')
print('Training Score: ',logreg.score(X_train, y_train))
print('Accuracy Score: ', accuracy_score(y_test,logreg.predict(X_test)))


# Decision Tree Classifier
Decision_tree=tree.DecisionTreeClassifier()
Decision_tree.fit(X_train,y_train)
print("\n # Decision Tree Classifier Model Results\n")
print('Training Score: ',Decision_tree.score(X_train,y_train))
print('Testing Score: ',accuracy_score(y_test,Decision_tree.predict(X_test)))      

# Random Forest Classifier
rf_classifier=RandomForestClassifier()
rf_classifier.fit(X_train,y_train)
print("\n # Random Forest Classifier Model Results\n")
print('Training Score: ',rf_classifier.score(X_train,y_train))
print('Testing Score: ',accuracy_score(y_test,rf_classifier.predict(X_test)))

# KNN Classifier
knn_classifier=KNeighborsClassifier()
knn_classifier.fit(X_train,y_train)
print("\n # KNN Classifier Model Results\n")
print('Training Score: ',knn_classifier.score(X_train,y_train))
print('Testing Score: ',accuracy_score(y_test,knn_classifier.predict(X_test)))


# In[ ]:


# Hyperparameter tunning and cross validation on Logistic Regression Model to improve accuracy score 

# USing CV 

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

logreg=LogisticRegression()
logreg.fit(X_train,y_train)
log_scores=cross_val_score(logreg,X_train,y_train,cv=5)
print("Accuracy score using CV",log_scores.mean())


# Now use hyperparameters

log_reg_params={'penalty':['l1','l2'], 'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
log_reg_best=grid_log_reg.best_estimator_
accuray_score=cross_val_score(log_reg_best,X_train,y_train,cv=5)
print("Accuracy score using hyperparameters and CV: ",accuray_score.mean())


# In[ ]:


# Hyperparameter tunning and cross validation on Decision Tree Model to improve accuracy score 

from sklearn.tree import DecisionTreeClassifier

# USing CV 

Decision_tree=tree.DecisionTreeClassifier()
Decision_tree.fit(X_train,y_train)
dt_score=cross_val_score(Decision_tree,X_train,y_train,cv=5)
print("Accuray using CV: ",dt_score.mean())

# Now use hyperparameters
dec_tree_parms={"criterion":['gini','entropy'],'max_depth': np.arange(1,10,1),'min_samples_leaf':np.arange(1,10,1) }
Grid_decision_tree=GridSearchCV(DecisionTreeClassifier(),dec_tree_parms)
Grid_decision_tree.fit(X_train,y_train)
Decision_tree_best=Grid_decision_tree.best_estimator_
Grid_dt_score=cross_val_score(Decision_tree_best,X_train,y_train,cv=5)
print("Accuracy score using hyperparameters and CV: ",Grid_dt_score.mean())


# In[ ]:


# Hyperparameter tunning and cross validation on Random Forest Model to improve accuracy score 

# USing CV 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

rf_classifier=RandomForestClassifier(n_jobs=-1,n_estimators=100)
rf_classifier.fit(X_train,y_train)
rf_score=cross_val_score(rf_classifier,X_train,y_train,cv=5)
print("Accuracy using CV : ",rf_score.mean() )

# Now use hyperparameters
rf_parms={"criterion":['gini','entropy'],'max_depth': np.arange(10,100,10),'max_features': ['auto', 'sqrt', 'log2'],
          }

grid_rf=GridSearchCV(rf_classifier,rf_parms,cv=5)
grid_rf.fit(X_train,y_train)
print(grid_rf.best_estimator_)

grid_rf_score=cross_val_score(grid_rf.best_estimator_,X_train,y_train,cv=5)
print("Accuracy score using hyperparameters and CV: ",grid_rf_score.mean())


# In[ ]:


# Hyperparameter tunning and cross validation on KNN to improve accuracy score 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# USing CV 
knn_classifier=KNeighborsClassifier()
knn_classifier.fit(X_train,y_train)
knn_score=cross_val_score(knn_classifier,X_train,y_train,cv=5)
print("Accuracy score using CV: ", knn_score.mean())

# Now use hyperparameters
knn_parms={'n_neighbors':np.arange(1,5,1),'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
knn_grid_clf=GridSearchCV(knn_classifier,knn_parms,cv=5)
knn_grid_clf.fit(X_train,y_train)
print(knn_grid_clf.best_estimator_)
knn_grid_best_score=cross_val_score(knn_grid_clf.best_estimator_,X_train,y_train)
print("Accuracy score using hyperparmeters and CV: ", knn_grid_best_score.mean())


# In[ ]:


# Taking the best score resulted from our experiments

scores={"Model":["Logistic Regression","Decision Tree","Random Forest","KNN"],
       "Accuracy scores":[log_scores.mean(),Grid_dt_score.mean(),grid_rf_score.mean(),knn_score.mean()]
        
       }

scores_df=pd.DataFrame(data=scores)
scores_df


# Random forest classifier has the highest accuracy amongst the all models. Now let's validate it 

# In[ ]:


from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict


y_pred=cross_val_predict(grid_rf.best_estimator_,X_train,y_train,cv=5)

accuracy_score(y_train, y_pred)



conf_matrix=confusion_matrix(y_train,y_pred)
sns.heatmap(conf_matrix, annot=True,fmt='d')
plt.title("Confusion Matrix")


# In[ ]:


from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score, recall_score

print('Precision Score: ', precision_score(y_train, y_pred))
print('Recall Score: ', recall_score(y_train, y_pred))
print('F1 Score',f1_score(y_train,y_pred))
print('Accuracy Score',accuracy_score(y_train,y_pred))
fpr,tpr,thresholds=roc_curve(y_train,y_pred)


# In[ ]:


from sklearn.metrics import roc_auc_score


plt.plot(fpr,tpr)
plt.plot([0, 1], [0, 1],linestyle='--')
print('ROC AUC Score',roc_auc_score(y_train, y_pred))
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title("ROC Curve")


# In[ ]:


# Getting the important features

feature_importances=pd.DataFrame(grid_rf.best_estimator_.feature_importances_,
                                 index=X_train.columns,columns=["Importance"]).sort_values('Importance',ascending=False)
importances = grid_rf.best_estimator_.feature_importances_
feature_names=X_train.columns
indices = np.argsort(importances)
plt.figure(figsize=(15, 16))

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


#  Conclusion:
#         
# -Retired  and management people subscribed to the term deposite have more balance in their account. Hence they should be highly targeted in the campaign
# 
# -People with more than average call duration are more likely to subscribe to term deposit
# 
# -More number of subscriptions have been seen in oct, dec, march and sept months
# 
# - People who have high balance, are less likely to have housing loan and more likely to subscribe to a term deposit 

# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




