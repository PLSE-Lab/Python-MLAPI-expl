#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import uniform

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


nira = pd.read_csv('customers (1).csv')


# In[ ]:


nira.shape


# In[ ]:


nira.head()


# In[ ]:


#Drop Uniqiue id and test columns
# nira.drop(['RowNumber',"CustomerId"],axis=1, inplace=True)


# In[ ]:


# Check columns type
nira.info()


# In[ ]:


nira.columns.values


# In[ ]:


# Check unique column values

# for i in nira.columns.values:
#     print(i,'\t',len(nira[i].unique()))
nira.nunique()


# In[ ]:


#Check for missing/null values
print(nira.isnull().values.any())
print(nira.isna().sum())


# In[ ]:


nira.describe()


# Unique columns values

# In[ ]:


print('tenure\t\t', nira['Tenure'].unique())
print('Geography\t', nira['Geography'].unique())
print('NumOfProducts\t', nira['NumOfProducts'].unique())


# In[ ]:


nira['HasCrCard'].value_counts()


# In[ ]:


nira['Exited'].value_counts()


# In[ ]:


nira[nira['HasCrCard']==1]['Exited'].value_counts()


# In[ ]:


nira[nira['HasCrCard']==0]['Exited'].value_counts()


# In[ ]:


nira['Reason for exiting company'].value_counts()


# In[ ]:





# In[ ]:


# Behavior of Exited customers for various categorical features
fig, axarr = plt.subplots(2,2, figsize=(16,10))
# plt.figure(figsize=(15,5))
sns.countplot(x='Geography', hue = 'Exited',data = nira, ax=axarr[0][0])
sns.countplot(x='Gender', hue = 'Exited',data = nira, ax=axarr[0][1])
sns.countplot(x='HasCrCard', hue = 'Exited',data = nira, ax=axarr[1][0])
sns.countplot(x='IsActiveMember', hue = 'Exited',data = nira, ax=axarr[1][1])


# From above plots we can infer that:
#     1. Germany has higher churn ratio than Spain and France.
#     2. Females churners are more as compared to male.
#     3. Proportion of customers having credit cards is higher than non-credit card holders.
#     4. Inactive member have higher risk to churn.

# In[ ]:


# Relations based on the continuous data attributes
fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
sns.boxplot(y='CreditScore',x = 'Exited', hue = 'Exited',data = nira, ax=axarr[0][0])
sns.boxplot(y='Age',x = 'Exited', hue = 'Exited',data = nira , ax=axarr[0][1])
sns.boxplot(y='Tenure',x = 'Exited', hue = 'Exited',data = nira, ax=axarr[1][0])
sns.boxplot(y='Balance',x = 'Exited', hue = 'Exited',data = nira, ax=axarr[1][1])
sns.boxplot(y='NumOfProducts',x = 'Exited', hue = 'Exited',data = nira, ax=axarr[2][0])
sns.boxplot(y='EstimatedSalary',x = 'Exited', hue = 'Exited',data = nira, ax=axarr[2][1])


# Inferences:
#     1. There is no significant contribution of CreditScore,NumOfProducts and EstimatedSalary in retained and churned customers.
#     2. Aged customers show more tendency to churn.
#     3. Average tenure customers as safe players. low and hight tenure products are more likely to churn.
#     4. The bank is losing customers with significant bank balances which is likely to hit their available capital for lending.

# In[ ]:


sns.distplot(nira.EstimatedSalary,kde=False)


# In[ ]:


sns.distplot(nira.Age,kde=False)


# In[ ]:


sns.distplot(nira.Balance,kde=False)


# In[ ]:


sns.distplot(nira.Tenure,kde=False)


# In[ ]:


len(nira[nira['Exited']==1]['Surname'].unique())


# # Data Preprocessing

# In[ ]:


# lb = LabelEncoder()
# nira['Geography'] = lb.fit_transform(nira['Geography'])
# Need not to encode reasons as its available only for exited customers and does not contribute towards churn.
# nira['Reason for exiting company'] = lb.fit_transform(nira['Reason for exiting company'])
# nira['Gender'] = lb.fit_transform(nira['Gender'])
# nira['Surname'] = lb.fit_transform(nira['Surname'])


# In[ ]:


# nira.head()


# In[ ]:


# One hot encode the categorical variables
# lst = ['Geography', 'Gender']
# remove = list()

# for i in lst:

#     if (nira[i].dtype == np.str or nira[i].dtype == np.object):
#         for j in nira[i].unique():
#             nira[i+'_'+j] = np.where(nira[i] == j,1,-1)
#         remove.append(i)
# nira = nira.drop(remove, axis=1)
# nira.head()


# In[ ]:


# nira.columns


# In[ ]:


def prepare_data(df):
    
    #One hot encoding
    lst = ['Geography', 'Gender']
    remove = list()

    for i in lst:

        if (df[i].dtype == np.str or df[i].dtype == np.object):
            for j in df[i].unique():
                df[i+'_'+j] = np.where(df[i] == j,1,-1)
            remove.append(i)
    df = df.drop(remove, axis=1)
    
    # Create Features
    df['BalToSalRatio'] = df['Balance']/df['EstimatedSalary']
    df['TenureByAge'] = df.Tenure/(df.Age - 18)
    df['CreditScoreGivenAge'] = df.CreditScore/(df.Age - 18)
    df.loc[df.HasCrCard == 0, 'HasCrCard'] = -1
    df.loc[df.IsActiveMember == 0, 'IsActiveMember'] = -1
    
    # Arrange columns by data type for easier manipulation
    continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalToSalRatio',
                       'TenureByAge','CreditScoreGivenAge']
    cat_vars = ['HasCrCard', 'IsActiveMember', 'Geography_France', 'Geography_Spain',
           'Geography_Germany', 'Gender_Female', 'Gender_Male']
    df = df[['RowNumber', 'CustomerId','Exited'] + continuous_vars + cat_vars]
    # minMax scaling the continuous variables
    continousv = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary']
    minVec = nira[continousv].min().copy()
    maxVec = nira[continousv].max().copy()
    df[continousv] = (df[continousv]-minVec)/(maxVec-minVec)
    df[df==np.inf]=np.nan
    df.fillna(nira.mean(), inplace=True)
    return df


# # Feature Engineering

# Since we don't have prior history or bank statements. so we should create another feature with balance given.

# In[ ]:





# In[ ]:


#Create a new feature Balance to salary ratio.
# nira['BalToSalRatio'] = nira['Balance']/nira['EstimatedSalary']
# nira['TenureByAge'] = nira.Tenure/(nira.Age - 18)
# nira['CreditScoreGivenAge'] = nira.CreditScore/(nira.Age - 18)
# nira.loc[nira.HasCrCard == 0, 'HasCrCard'] = -1
# nira.loc[nira.IsActiveMember == 0, 'IsActiveMember'] = -1


# In[ ]:


nira.columns


# In[ ]:





# In[ ]:





# In[ ]:


# Arrange columns by data type for easier manipulation
# continuous_vars = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary', 'BalToSalRatio',
#                    'TenureByAge','CreditScoreGivenAge']
# cat_vars = ['HasCrCard', 'IsActiveMember', 'Geography_France', 'Geography_Spain',
#        'Geography_Germany', 'Gender_Female', 'Gender_Male']
# nira = nira[['RowNumber', 'CustomerId','Exited'] + continuous_vars + cat_vars]
# nira.head()


# In[ ]:


# minMax scaling the continuous variables
# continousv = ['CreditScore',  'Age', 'Tenure', 'Balance','NumOfProducts', 'EstimatedSalary']
# minVec = nira[continousv].min().copy()
# maxVec = nira[continousv].max().copy()
# nira[continousv] = (nira[continousv]-minVec)/(maxVec-minVec)
# nira.head()


# In[ ]:


# nira[nira==np.inf]=np.nan
# nira.fillna(nira.mean(), inplace=True)


# In[ ]:


nira_data = prepare_data(nira)


# In[ ]:


nira_data.columns


# In[ ]:


features= [i for i in nira_data.columns if i not in ['RowNumber', 'CustomerId','Surname','Reason for exiting company','Exited']]
target = 'Exited'


# In[ ]:


nira_data.head()


# In[ ]:


#split test and train set
# nira.fillna(0)
# nira.round(2)
X_train, X_test, Y_train, Y_test = train_test_split(nira_data[features],nira_data[target], test_size = 0.2, random_state = 0)


# In[ ]:


print(X_train.shape,Y_train.shape, X_test.shape, Y_test.shape)


# In[ ]:


# col_mask=nira.isnull().any(axis=0)
# col_mask
# row_mask=nira.isnull().any(axis=1)
# row_mask
# nira.loc[row_mask,col_mask]


# In[ ]:





# # Build model to predict customer attrition

# For the model fitting, I will try out the following
# 
#     Logistic regression in the primal space and with different kernels
#     SVM in the primal and with different Kernels
#     Ensemble models

# In[ ]:


print(nira.isnull().values.any())
print(nira.isna().sum())
# (np.where(np.isnan(nira)))


# In[ ]:


# Function to give best model score and parameters
def best_model(model):
    print(model.best_score_)    
    print(model.best_params_)
    print(model.best_estimator_)
def get_auc_scores(y_actual, method,method2):
    auc_score = roc_auc_score(y_actual, method); 
    fpr_df, tpr_df, _ = roc_curve(y_actual, method2); 
    return (auc_score, fpr_df, tpr_df)


# In[ ]:


# Fit primal logistic regression
param_grid = {'C': [0.1,0.5,1,10,50,100], 'max_iter': [250], 'fit_intercept':[True],'intercept_scaling':[1],
              'penalty':['l2'], 'tol':[0.00001,0.0001,0.000001]}
log_primal_Grid = GridSearchCV(LogisticRegression(),param_grid, cv=10, refit=True, verbose=0)
log_primal_Grid.fit(X_train,Y_train)
best_model(log_primal_Grid)
# lr = LogisticRegression()
# lr.fit(X_train,Y_train)


# In[ ]:


# Fit logistic regression with degree 2 polynomial kernel
param_grid = {'C': [0.1,10,50], 'max_iter': [300,500], 'fit_intercept':[True],'intercept_scaling':[1],'penalty':['l2'],
              'tol':[0.0001,0.000001]}
poly2 = PolynomialFeatures(degree=2)
df_train_pol2 = poly2.fit_transform(X_train)
log_pol2_Grid = GridSearchCV(LogisticRegression(solver = 'liblinear'),param_grid, cv=5, refit=True, verbose=0)
log_pol2_Grid.fit(X_train,Y_train)
best_model(log_pol2_Grid)


# In[ ]:





# In[ ]:


# Fit SVM with RBF Kernel
param_grid = {'C': [0.5,100,150], 'gamma': [0.1,0.01,0.001],'probability':[True],'kernel': ['rbf']}
SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
SVM_grid.fit(X_train,Y_train)
best_model(SVM_grid)


# In[ ]:


# # Fit SVM with pol kernel
# param_grid = {'C': [1,10], 'gamma': [0.1,0.01],'probability':[True],'kernel': ['poly'],'degree':[2] }
# SVM_grid = GridSearchCV(SVC(), param_grid, cv=3, refit=True, verbose=0)
# SVM_grid.fit(X_train,Y_train)
# best_model(SVM_grid)


# In[ ]:


# Fit random forest classifier
param_grid = {'max_depth': [3, 5, 6], 'max_features': [2,4,7,9],'n_estimators':[50,100],'min_samples_split': [3, 5, 6, 7]}
RanFor_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, refit=True, verbose=0)
RanFor_grid.fit(X_train,Y_train)
best_model(RanFor_grid)


# In[ ]:


# Fit Extreme Gradient boosting classifier
param_grid = {'max_depth': [5,6,7,8], 'gamma': [0.01,0.001,0.001],'min_child_weight':[1,5,10], 'learning_rate': [0.05,0.1, 0.2, 0.3], 'n_estimators':[5,10,20,100]}
xgb_grid = GridSearchCV(XGBClassifier(), param_grid, cv=5, refit=True, verbose=0)
xgb_grid.fit(X_train,Y_train)
best_model(xgb_grid)


# In[ ]:


xgb_grid.feature_importances_


# In[ ]:


# Fit primal logistic regression
log_primal = LogisticRegression(C=100, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=250, multi_class='warn',n_jobs=None, 
                                penalty='l2', random_state=None, solver='lbfgs',tol=1e-05, verbose=0, warm_start=False)
log_primal.fit(X_train,Y_train)


# In[ ]:


print(classification_report(Y_train, log_primal.predict(X_train)))


# In[ ]:


# Fit logistic regression with pol 2 kernel
poly2 = PolynomialFeatures(degree=2)
df_train_pol2 = poly2.fit_transform(X_train)
log_pol2 = LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=300, multi_class='warn', n_jobs=None, 
                              penalty='l2', random_state=None, solver='liblinear',tol=0.0001, verbose=0, warm_start=False)
log_pol2.fit(df_train_pol2,Y_train)


# In[ ]:


print(classification_report(Y_train,  log_pol2.predict(df_train_pol2)))


# In[ ]:


# Fit SVM with RBF Kernel
SVM_RBF = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.1, kernel='rbf', max_iter=-1, probability=True, 
              random_state=None, shrinking=True,tol=0.001, verbose=False)
SVM_RBF.fit(X_train,Y_train)


# In[ ]:


print(classification_report(Y_train,  SVM_RBF.predict(X_train)))


# In[ ]:


# # Fit SVM with Pol Kernel
# SVM_POL = SVC(C=100, cache_size=200, class_weight=None, coef0=0.0,  decision_function_shape='ovr', degree=2, gamma=0.1, kernel='poly',  max_iter=-1,
#               probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)
# SVM_POL.fit(X_train,Y_train)


# In[ ]:


# print(classification_report(Y_train,  SVM_POL.predict(X_train)))


# In[ ]:


# Fit Random Forest classifier
RF = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',max_depth=8, max_features=6, max_leaf_nodes=None,min_impurity_decrease=0.0,
                            min_impurity_split=None,min_samples_leaf=1, min_samples_split=3,min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
                            oob_score=False, random_state=None, verbose=0,warm_start=False)
RF.fit(X_train,Y_train)


# In[ ]:


RF.classes_


# In[ ]:


print(classification_report(Y_train,  RF.predict(X_train)))


# # Top 4 Employee Who are Likely to Leave and their probability.

# In[ ]:


predictions = pd.DataFrame(RF.predict_proba(nira[features]),columns = ['Prob_0', 'Prob_1'])


# In[ ]:


Resultant_nira = pd.concat([nira,predictions],axis=1)


# In[ ]:


Resultant_nira[['RowNumber','CustomerId','Exited','Prob_0','Prob_1']]


# In[ ]:


top_churners =Resultant_nira.sort_values(['Prob_1'],ascending=0)


# In[ ]:


#top churners
top_churners[['CustomerId','Prob_1','Exited', 'Balance', 'Age', 'CreditScoreGivenAge', 'BalToSalRatio', 'NumOfProducts', 'TenureByAge','Prob_0']].head(4)
# top_churners.columns


# In[ ]:


# Top retained customers
top_churners[['CustomerId','Prob_1','Exited','Balance', 'Age', 'CreditScoreGivenAge', 'BalToSalRatio', 'NumOfProducts', 'TenureByAge','Prob_0']].tail(4)


# In[ ]:


From above top churners and top retained list. we can conclude that:
    1. Senior citizens are more likely to churn.
    2. Exited customers have higher BalToSalRatio,CreditScoreGivenAge,TenureByAge ratio.
    3. NumOfProducts value is significantly higher for churners.


# In[ ]:


top_churners.head(4)['CustomerId'].values


# ## Top 4 customers about to churn are:
# 15700801, 15647725, 15641175, 15672056

# In[ ]:



Resultant_nira[Resultant_nira['CustomerId'].isin([75572918,23109012])]
# Resultant_nira[Resultant_nira['CustomerId']==23109012]


# In[ ]:


print('min id',Resultant_nira['CustomerId'].min())
print('max id',Resultant_nira['CustomerId'].max())


# ### Customer Id 75572918,23109012 are not present in dataset

# In[ ]:


Resultant_nira.columns


# In[ ]:


# Fit Extreme Gradient Boost Classifier
XGB = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,colsample_bytree=1, gamma=0.01, learning_rate=0.1, max_delta_step=0,max_depth=7,
                    min_child_weight=5, missing=None, n_estimators=20,n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,reg_alpha=0, 
                    reg_lambda=1, scale_pos_weight=1, seed=None, silent=True, subsample=1)
XGB.fit(X_train,Y_train)


# In[ ]:


print(classification_report(Y_train,  XGB.predict(X_train)))


# # Feature Importances

# In[ ]:


XGB.feature_importances_


# In[ ]:


from xgboost import plot_importance
plt.bar(range(len(XGB.feature_importances_)), XGB.feature_importances_)
plt.show()


# In[ ]:


plot_importance(XGB)
plt.show()


# In[ ]:





# In[ ]:


y = Y_train
X = X_train
X_pol2 = df_train_pol2
auc_log_primal, fpr_log_primal, tpr_log_primal = get_auc_scores(y, log_primal.predict(X),log_primal.predict_proba(X)[:,1])
auc_log_pol2, fpr_log_pol2, tpr_log_pol2 = get_auc_scores(y, log_pol2.predict(X_pol2),log_pol2.predict_proba(X_pol2)[:,1])
auc_SVM_RBF, fpr_SVM_RBF, tpr_SVM_RBF = get_auc_scores(y, SVM_RBF.predict(X),SVM_RBF.predict_proba(X)[:,1])
# auc_SVM_POL, fpr_SVM_POL, tpr_SVM_POL = get_auc_scores(y, SVM_POL.predict(X),SVM_POL.predict_proba(X)[:,1])
auc_RF, fpr_RF, tpr_RF = get_auc_scores(y, RF.predict(X),RF.predict_proba(X)[:,1])
auc_XGB, fpr_XGB, tpr_XGB = get_auc_scores(y, XGB.predict(X),XGB.predict_proba(X)[:,1])


# In[ ]:


plt.figure(figsize = (12,6), linewidth= 1)
plt.plot(fpr_log_primal, tpr_log_primal, label = 'log primal Score: ' + str(round(auc_log_primal, 5)))
plt.plot(fpr_log_pol2, tpr_log_pol2, label = 'log pol2 score: ' + str(round(auc_log_pol2, 5)))
plt.plot(fpr_SVM_RBF, tpr_SVM_RBF, label = 'SVM RBF Score: ' + str(round(auc_SVM_RBF, 5)))
# plt.plot(fpr_SVM_POL, tpr_SVM_POL, label = 'SVM POL Score: ' + str(round(auc_SVM_POL, 5)))
plt.plot(fpr_RF, tpr_RF, label = 'RF score: ' + str(round(auc_RF, 5)))
plt.plot(fpr_XGB, tpr_XGB, label = 'XGB score: ' + str(round(auc_XGB, 5)))
plt.plot([0,1], [0,1], 'k--', label = 'Random: 0.5')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC Curve')
plt.legend(loc='best')
#plt.savefig('roc_results_ratios.png')
plt.show()


# # Conclusions

# 1. Using Bagging Random forest classifier gives best results with best score of 88%.
# 2. Balance, Age, CreditScoreGivenAge, BalToSalRatio, NumberOfProducts, TenureByAge are the top contributers to predict customer attrition.

# In[ ]:


test_nira= pd.read_csv('test_nira.csv')


# In[ ]:


test_nira.shape


# In[ ]:


# test_nira.drop(['Exited','Reason for exiting company'],axis=1,inplace=True)


# In[ ]:


test_data = prepare_data(test_nira)


# In[ ]:


test_nira.drop(['Exited','Reason for exiting company'],axis=1,inplace=True)


# In[ ]:


test_data


# In[ ]:


predict = pd.DataFrame(RF.predict(test_data[features]),columns = ['Exit_Prediction'])


# In[ ]:


predict


# In[ ]:


preds = pd.DataFrame(RF.predict_proba(test_data[features]),columns = ['Prob_0', 'Prob_1'])


# In[ ]:


test_results = pd.concat([test_data,preds,predict],axis=1)


# In[ ]:


test_data_original = pd.concat([test_nira,preds,predict],axis=1)


# In[ ]:


# Sort the test data with probability to churn 
test_results_sorted = test_results.sort_values(['Prob_1'],ascending=0)
test_results_sorted.to_csv('test_normalised_results_sorted.csv')
test_data_original= test_data_original.sort_values(['Prob_1'],ascending=0)
test_data_original.to_csv('test_data_results.csv')


# In[ ]:


test_results_sorted


# In[ ]:


test_results_sorted.columns


# In[ ]:


test_results_sorted[['CustomerId','Prob_1','Prob_0','Exited', 'Balance', 'Age', 'CreditScoreGivenAge', 'BalToSalRatio', 'NumOfProducts', 'TenureByAge']]


# In[ ]:


75572918,23109012


# In[ ]:




