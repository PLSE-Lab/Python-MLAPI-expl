#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv("../input/Hotel Booking Cancellation.csv")
df


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[ ]:


categ_col = df.select_dtypes('object').columns.tolist()


# In[ ]:


fig, ax = plt.subplots(5,3,figsize=(20,20))
for i, j in zip(categ_col,ax.flatten()):
    sns.countplot(df[i],ax=j)
plt.show()


# In[ ]:


df.describe()


# In[ ]:


df.describe(include='object')


# In[ ]:


numeric_col = df.select_dtypes(np.number).columns.tolist()


# In[ ]:


fig, ax = plt.subplots(6,3,figsize=(40,40))
for i, j in zip(numeric_col,ax.flatten()):
    sns.countplot(df[i],ax=j)
plt.show()


# In[ ]:


fig, ax = plt.subplots(6,3,figsize=(40,40))
for i, j in zip(numeric_col,ax.flatten()):
    sns.boxplot(df[i],ax=j)
plt.show()


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(df.corr(),annot=True,cmap='Greens')


# In[ ]:


Q1 = df.quantile(0.25)

#calculate the third quartile
Q3 = df.quantile(0.75)

# The Interquartile Range (IQR) is defined as difference between the third and first quartile
# calculate IQR
IQR = Q3 - Q1
print(IQR)


# In[ ]:


df_outliers = df[((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[ ]:


df.IsCanceled.value_counts().plot(kind='bar')


# In[ ]:


print('Percentage of class 0 (Not Canceled) :',df['IsCanceled'].value_counts()[0]*100/df.shape[0])
print('Percentage of class 1 (Canceled):',df['IsCanceled'].value_counts()[1]*100/df.shape[0])


# In[ ]:


pd.crosstab(df['MarketSegment'],df['DistributionChannel'])


# In[ ]:


pd.crosstab(df['IsRepeatedGuest'],df['PreviousCancellations'])


# In[ ]:


pd.crosstab(df['IsRepeatedGuest'],df['PreviousBookingsNotCanceled'])


# In[ ]:


pd.crosstab(df['ReservationStatus'],df['IsCanceled'])


# In[ ]:


ct =pd.crosstab(df['CustomerType'],df['IsCanceled'])
ct.div(ct.sum(1), axis=0).sort_values(by=1,ascending=False)


# In[ ]:


pd.crosstab(df['ReservedRoomType'],df['AssignedRoomType'])


# In[ ]:


ct =pd.crosstab(df['ReservedRoomType'],df['AssignedRoomType'])
ct.div(ct.sum(1), axis=0)


# In[ ]:


pd.crosstab(df['DepositType'],df['AssignedRoomType'])


# In[ ]:


df['DepositType'].unique()


# In[ ]:


df[df['DepositType'] == 'Non Refund     ']['Children'].value_counts()


# In[ ]:


df['DepositType'].value_counts()


# In[ ]:


ct =pd.crosstab(df['DepositType'],df['IsCanceled'])
ct.div(ct.sum(1), axis=0).plot(kind='bar', stacked=True)


# In[ ]:


m = df['Agent'].value_counts()[(df['Agent'].value_counts()>400)].index
m


# In[ ]:


ct =pd.crosstab(df['Agent'],df['IsCanceled']).loc[m,:]
ct.div(ct.sum(1), axis=0).sort_values(by=1,ascending=False)


# In[ ]:


df1 = df[df['AssignedRoomType'] != df['ReservedRoomType']]


# In[ ]:


ct =pd.crosstab(df1['ReservedRoomType'],df1['IsCanceled'])
ct.div(ct.sum(1), axis=0).sort_values(by=1,ascending=False)


# In[ ]:


ct =pd.crosstab(df['AssignedRoomType'],df['IsCanceled'])
ct.div(ct.sum(1), axis=0).sort_values(by=1,ascending=False)


# In[ ]:


ct =pd.crosstab(df['ArrivalDateMonth'],df['IsCanceled'])
ct.div(ct.sum(1), axis=0).sort_values(by=1,ascending=False)


# In[ ]:


m = df['Company'].value_counts()[(df['Company'].value_counts()>100)].index


# In[ ]:


pd.crosstab(df['Company'],df['MarketSegment']).loc[m,:]


# In[ ]:


ct = pd.crosstab(df['Company'],df['IsCanceled']).loc[m,:]
ct.div(ct.sum(1), axis=0).sort_values(by=1,ascending=False).plot(kind='bar', stacked=True)


# In[ ]:


m = df['Country'].value_counts()[(df['Country'].value_counts()>400)].index
m


# In[ ]:


df['Country'][df['IsCanceled']==1].value_counts()[(df['Country'].value_counts()>400)].index


# In[ ]:


ct = pd.crosstab(df['Country'],df['IsCanceled']).loc[m,:]
ct.div(ct.sum(1), axis=0).sort_values(by=1,ascending=False)


# # Statistical Analysis

# In[ ]:


from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.api import anova_lm


# In[ ]:


significant_col = []
insignificant_col = []
for i in categ_col:
    for j in categ_col:
        chisq, pval, dof, exp = stats.chi2_contingency(pd.crosstab(df[j],df[i]))
        if pval < 0.00001:
            significant_col.append((i,j))
        else:
            insignificant_col.append((i,j))
print('Significant Features:\n',significant_col)
print('\nInsignificant Features:\n',insignificant_col)


# In[ ]:


significant_col = []
insignificant_col = []
for i in categ_col:
    for j in categ_col:
        chisq, pval, dof, exp = stats.chi2_contingency(pd.crosstab(df[j],df[i]))
        print(i,j)
        print(pval)
        print('\n')


# In[ ]:


significant_col = []
insignificant_col = []
p_num = []
p_cat = []
for i in numeric_col:    
    formula = i+' ~ IsCanceled'
    model = ols(formula,df).fit()
    ano = anova_lm(model, typ=2)
    p_num.append(model.pvalues['IsCanceled'])
    if model.pvalues['IsCanceled']< 0.05:
        significant_col.append(i)
    else:
        insignificant_col.append(i)
for j in categ_col:
    chisq, pval, dof, exp = stats.chi2_contingency(pd.crosstab(df[j],df.IsCanceled))
    p_cat.append(pval)
    if pval < 0.05:
        significant_col.append(j)
    else:
        insignificant_col.append(j)
print('Significant Features:\n',significant_col)
print('\nInsignificant Features:\n',insignificant_col)


# In[ ]:


df_pval_num = pd.DataFrame({'Feature':numeric_col,'P value':p_num})
df_pval_num


# In[ ]:


df_pval_cat = pd.DataFrame({'Feature':categ_col,'P value':p_cat})
df_pval_cat


# # Feature Engineering

# In[ ]:


def quarter(x):
    if x in ['January','February','March',]:
        return 1
    elif x in ['April','May','June','July']:
        return 2
    elif x in ['August','September','October']:
        return 3
    else:
        return 4


# In[ ]:


df['Quarter'] = df['ArrivalDateMonth'].apply(quarter)


# In[ ]:


df['TotalStayinNights'] = df['StaysInWeekNights']+df['StaysInWeekendNights']


# In[ ]:


def adults(x):
    if x == 0:
        return 0
    else:
        return 1


# In[ ]:


df['Adults/Children'] = df['Children']+df['Babies']


# In[ ]:


df['Family/Business'] = df['Adults/Children'].apply(adults)


# In[ ]:


def country(x):
    if x == 'PRT':
        return 'Portugal'
    elif x in ['CHE','BRA','ESP','IRL','ITA','USA']:
        return 'High'
    else:
        return 'Low'


# In[ ]:


def agent(x):
    if x == 'NULL':
        return 'No Agent'
    elif x in ['96','240','242','298']:
        return 'High'
    elif x in ['250','314','241','6','40','243']: 
        return 'Medium' 
    else:
        return 'Low'


# In[ ]:


def company(x):
    if x == '       NULL':
        return 'No Company'
    elif x in ['223','281','154']:
        return 'Significant'
    else:
        return 'Others'


# In[ ]:


df['Company'] = df['Company'].apply(company)


# In[ ]:


(df['Company'].value_counts()/df['Company'].count())*100


# In[ ]:


df['Agent'] = df['Agent'].apply(agent)


# In[ ]:


(df['Agent'].value_counts()/df['Agent'].count())*100


# In[ ]:


df['Country'] = df['Country'].apply(country)


# In[ ]:


(df['Country'].value_counts()/df['Country'].count())*100


# In[ ]:


def monthlyquarter(x):
    if x in range(1,8):
        return 1
    elif x in range(8,15):
        return 2
    elif x in range(15,22):
        return 3
    else:
        return 4


# In[ ]:


df['MonthlyQuarter'] = df['ArrivalDateDayOfMonth'].apply(monthlyquarter)


# In[ ]:


df['Country'].unique()


# In[ ]:


df['Assigned/Reserved'] = df.apply(lambda row: (row.AssignedRoomType != row.ReservedRoomType), axis = 1)


# In[ ]:


col = ['DistributionChannel','MarketSegment','ReservedRoomType','AssignedRoomType']
for j in col:
    chisq, pval, dof, exp = stats.chi2_contingency(pd.crosstab(df[j],df.IsCanceled))
    print(j)
    print(pval)


# In[ ]:


df.drop(['ReservationStatus','ReservationStatusDate','MarketSegment','Adults/Children','ReservedRoomType','ArrivalDateMonth','ArrivalDateWeekNumber','StaysInWeekNights','Children'], axis = 1,inplace=True)


# In[ ]:


cat_cols = df.select_dtypes('object').columns.tolist()


# In[ ]:


df_final = df.copy()


# In[ ]:


df_final.head()


# In[ ]:


for col in cat_cols:
    k=df_final[col].value_counts().index[:-1]
    for cat in k:
        name=col+'_'+cat
        df_final[name]=(df_final[col]==cat).astype(int)
    del df_final[col]
    print(col)


# In[ ]:


df_final['TotalPreviousBookings'] = df_final['PreviousBookingsNotCanceled']+df_final['PreviousCancellations']


# In[ ]:


df_final[df_final['LeadTime'] > df['LeadTime'].quantile([0.99]).values[0]].shape


# In[ ]:


df_final.columns


# In[ ]:


df_final.shape


# In[ ]:


from scipy.stats import zscore


# In[ ]:


X = df_final.drop('IsCanceled',axis=1)
y = df_final['IsCanceled']
X_scaled = X.apply(zscore)


# # Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.3,random_state=5)


# # Model Building

# ## Logistic Regression

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[ ]:


logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[ ]:


conf_matrix=pd.DataFrame(data=cm,columns=['P_0','P_1'],index=['A_0','A_1'])
conf_matrix


# In[ ]:


sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()


# In[ ]:


from sklearn.metrics import classification_report
result = classification_report(y_test,y_pred)

# print the result
print(result)


# In[ ]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_curve, auc


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


from sklearn import metrics
cols = ['Model', 'AUC Score', 'Precision Score', 'Recall Score','f1-score','Accuracy Score','Train Accuracy']

# creating an empty dataframe of the colums
result_tabulation = pd.DataFrame(columns = cols)

# compiling the required information
Logistic_regression_metrics = pd.Series({'Model': "Logistic regression ",
                     'AUC Score' : roc_auc_score(y_test, y_pred),
                 'Precision Score': metrics.precision_score(y_test, y_pred,average='weighted'),
                 'Recall Score': metrics.recall_score(y_test, y_pred,average='weighted'),
                 'f1-score':metrics.f1_score(y_test, y_pred,average='weighted'),
                 'Accuracy Score': metrics.accuracy_score(y_test, y_pred),
                 'Train Accuracy': logreg.score(X_train,y_train)})



# appending our result table
result_tabulation = result_tabulation.append(Logistic_regression_metrics , ignore_index = True)

# view the result table
result_tabulation


# # Decision Tree

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
decision_tree_classification = DecisionTreeClassifier(criterion='entropy')

# train model
decision_tree = decision_tree_classification.fit(X_train, y_train)

# predict the model using 'X_test'
decision_tree_pred = decision_tree.predict(X_test)


# In[ ]:


result = classification_report(y_test,decision_tree_pred)

# print the result
print(result)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test,decision_tree_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


Decision_Tree_metrics = pd.Series({'Model': "Decision Tree",
                     'AUC Score' : roc_auc_score(y_test, decision_tree_pred),
                 'Precision Score': metrics.precision_score(y_test, decision_tree_pred,average='weighted'),
                 'Recall Score': metrics.recall_score(y_test, decision_tree_pred,average='weighted'),
                 'f1-score':metrics.f1_score(y_test, decision_tree_pred,average='weighted'),
                 'Accuracy Score': metrics.accuracy_score(y_test, decision_tree_pred),
                 'Train Accuracy': decision_tree.score(X_train,y_train)})



# appending our result table
result_tabulation = result_tabulation.append(Decision_Tree_metrics , ignore_index = True)

# view the result table
result_tabulation


# # GridSearch Cross Validation

# In[ ]:


from sklearn.model_selection import GridSearchCV
max_depth = np.arange(1,5,1)
min_samples_leaf = np.arange(1,50,10)
max_leaf_nodes = np.arange(10,20,1)

# set of parameters to test
param_grid = {"criterion": ["gini", "entropy"],
              "max_depth": max_depth,
              "min_samples_leaf":min_samples_leaf
              }


# In[ ]:


decision_tree_Gridsearch = DecisionTreeClassifier()
decision_tree_Gridsearch = GridSearchCV(decision_tree_Gridsearch, param_grid, cv=10)
decision_tree_Gridsearch.fit(X_train, y_train)


# In[ ]:


decision_tree_Gridsearch.best_params_


# In[ ]:


decision_tree_classification = DecisionTreeClassifier(criterion='entropy',
 max_depth=4,
 max_leaf_nodes=4,
 min_samples_leaf=1)

# train model
decision_tree_Grid = decision_tree_classification.fit(X_train, y_train)

# predict the model using 'X_test'
decision_tree_Grid_pred = decision_tree_Grid.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, decision_tree_Grid_pred)

# set size of the plot
#plt.figure(figsize = (8,5))

# plot a heatmap
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()


# In[ ]:


result = classification_report(y_test,decision_tree_Grid_pred)

# print the result
print(result)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test,decision_tree_Grid_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


Decision_Tree_Grid_metrics = pd.Series({'Model': "Decision Tree with Grid Search",
                     'AUC Score' : roc_auc_score(y_test, decision_tree_Grid_pred),
                 'Precision Score': metrics.precision_score(y_test, decision_tree_Grid_pred,average='weighted'),
                 'Recall Score': metrics.recall_score(y_test, decision_tree_Grid_pred,average='weighted'),
                 'f1-score':metrics.f1_score(y_test, decision_tree_Grid_pred,average='weighted'),
                 'Accuracy Score': metrics.accuracy_score(y_test, decision_tree_Grid_pred),
                 'Train Accuracy': decision_tree_Grid.score(X_train,y_train)})



# appending our result table
result_tabulation = result_tabulation.append(Decision_Tree_Grid_metrics , ignore_index = True)

# view the result table
result_tabulation


# # Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, rf_pred)

# label the confusion matrix  
conf_matrix=pd.DataFrame(data=cm,columns=['P_0','P_1'],index=['A_0','A_1'])

# set size of the plot
#plt.figure(figsize = (8,5))

# plot a heatmap
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
plt.show()


# In[ ]:


result = classification_report(y_test,rf_pred)

# print the result
print(result)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test,rf_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


Random_forest_metrics = pd.Series({'Model': "Random Forest",
                     'AUC Score' : roc_auc_score(y_test, rf_pred),
                 'Precision Score': metrics.precision_score(y_test, rf_pred,average='weighted'),
                 'Recall Score': metrics.recall_score(y_test, rf_pred,average='weighted'),
                 'f1-score':metrics.f1_score(y_test, rf_pred,average='weighted'),
                 'Accuracy Score': metrics.accuracy_score(y_test, rf_pred),
                 'Train Accuracy': rf.score(X_train,y_train)})



# appending our result table
result_tabulation = result_tabulation.append(Random_forest_metrics , ignore_index = True)

# view the result table
result_tabulation


# # Feature Selection

# In[ ]:


feature_imp = pd.Series(rf.feature_importances_,index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(15,15))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")


# In[ ]:


rf_f = feature_imp[feature_imp>0.00730].index.tolist()
rf_f


# In[ ]:


len(rf_f)


# In[ ]:


c=rf_f
len(c)


# In[ ]:


c


# In[ ]:


d = {'Random Forest': sorted(rf_f), 'RFE': sorted(c)}
feat_sel = pd.DataFrame(d)
feat_sel


# In[ ]:


df_final.columns


# In[ ]:


df_final.shape


# In[ ]:


df['Meal'].value_counts()


# In[ ]:


df_feat = df_final[c]
df_feat['Meal_HB'] = df_final['Meal_HB       ']
df_feat.shape


# In[ ]:


df_feat.columns


# In[ ]:


#Scalling
X = df_feat
y = df_final['IsCanceled']
X_scaled = X.apply(zscore)


# # Train Test Split

# In[ ]:


X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=5)


# # Model Building
# 
# ## Logistic Regression

# In[ ]:


logreg = LogisticRegression()


# In[ ]:


logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)

# set size of the plot
#plt.figure(figsize = (8,5))

# plot a heatmap
sns.heatmap(cm, annot=True,fmt='d',cmap="YlGnBu")
plt.show()


# In[ ]:


result = classification_report(y_test,y_pred)

# print the result
print(result)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test,y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


cols = ['Model', 'AUC Score', 'Precision Score', 'Recall Score','f1-score','Accuracy Score','Train Accuracy']

# creating an empty dataframe of the colums
result_tabulation1 = pd.DataFrame(columns = cols)

# compiling the required information
Logistic_regression_metrics = pd.Series({'Model': "Logistic regression with feature selection",
                     'AUC Score' : roc_auc_score(y_test, y_pred),
                 'Precision Score': metrics.precision_score(y_test, y_pred,average='weighted'),
                 'Recall Score': metrics.recall_score(y_test, y_pred,average='weighted'),
                 'f1-score':metrics.f1_score(y_test, y_pred,average='weighted'),
                 'Accuracy Score': metrics.accuracy_score(y_test, y_pred),
                 'Train Accuracy': logreg.score(X_train,y_train)})

# appending our result table
result_tabulation1 = result_tabulation1.append(Logistic_regression_metrics , ignore_index = True)

# view the result table
result_tabulation1


# # Decision Tree

# In[ ]:


decision_tree_classification = DecisionTreeClassifier(criterion='entropy')

# train model
decision_tree = decision_tree_classification.fit(X_train, y_train)

# predict the model using 'X_test'
decision_tree_pred = decision_tree.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, decision_tree_pred)

# set size of the plot
#plt.figure(figsize = (8,5))

# plot a heatmap
sns.heatmap(cm, annot=True,fmt='d',cmap="YlGnBu")
plt.show()


# In[ ]:


result = classification_report(y_test,decision_tree_pred)

# print the result
print(result)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test,decision_tree_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


Decision_Tree_metrics = pd.Series({'Model': "Decision Tree with feature selection",
                     'AUC Score' : roc_auc_score(y_test, decision_tree_pred),
                 'Precision Score': metrics.precision_score(y_test, decision_tree_pred,average='weighted'),
                 'Recall Score': metrics.recall_score(y_test, decision_tree_pred,average='weighted'),
                 'f1-score':metrics.f1_score(y_test, decision_tree_pred,average='weighted'),
                 'Accuracy Score': metrics.accuracy_score(y_test, decision_tree_pred),
                 'Train Accuracy': decision_tree.score(X_train,y_train)})



# appending our result table
result_tabulation1 = result_tabulation1.append(Decision_Tree_metrics , ignore_index = True)

# view the result table
result_tabulation1


# # Grid Search Cross Validation

# In[ ]:


max_depth = np.arange(1,5,1)
min_samples_leaf = np.arange(1,50,10)
max_leaf_nodes = np.arange(10,20,1)

# set of parameters to test
param_grid = {"criterion": ["gini", "entropy"],
              "max_depth": max_depth,
              "min_samples_leaf":min_samples_leaf,
              "max_leaf_nodes": max_leaf_nodes}


# In[ ]:


decision_tree_Gridsearch = DecisionTreeClassifier()
decision_tree_Gridsearch = GridSearchCV(decision_tree_Gridsearch, param_grid, cv=10)
decision_tree_Gridsearch.fit(X_train, y_train)


# In[ ]:


decision_tree_Gridsearch.best_params_


# In[ ]:


decision_tree_classification = DecisionTreeClassifier(criterion='entropy',
 max_depth=4,
 max_leaf_nodes=10,
 min_samples_leaf=1)

# train model
decision_tree_Grid = decision_tree_classification.fit(X_train, y_train)

# predict the model using 'X_test'
decision_tree_Grid_pred = decision_tree_Grid.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, decision_tree_Grid_pred)


# set size of the plot
#plt.figure(figsize = (8,5))

# plot a heatmap
sns.heatmap(cm, annot=True,fmt='d',cmap="YlGnBu")
plt.show()


# In[ ]:


result = classification_report(y_test,decision_tree_Grid_pred)

# print the result
print(result)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test,decision_tree_Grid_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


Decision_Tree_Grid_metrics = pd.Series({'Model': "Decision Tree with grid search",
                     'AUC Score' : roc_auc_score(y_test, decision_tree_Grid_pred),
                 'Precision Score': metrics.precision_score(y_test, decision_tree_Grid_pred,average='weighted'),
                 'Recall Score': metrics.recall_score(y_test, decision_tree_Grid_pred,average='weighted'),
                 'f1-score':metrics.f1_score(y_test, decision_tree_Grid_pred,average='weighted'),
                 'Accuracy Score': metrics.accuracy_score(y_test, decision_tree_Grid_pred),
                 'Train Accuracy': decision_tree_Grid.score(X_train,y_train)})



# appending our result table
result_tabulation1 = result_tabulation1.append(Decision_Tree_Grid_metrics , ignore_index = True)

# view the result table
result_tabulation1


# # Random Forest

# In[ ]:


random_forest = RandomForestClassifier()

# train model
random_forest.fit(X_train, y_train)

# predict the model using 'X_test'
rf_pred = random_forest.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, rf_pred)



# set size of the plot
#plt.figure(figsize = (8,5))

# plot a heatmap
sns.heatmap(cm, annot=True,fmt='d',cmap="YlGnBu")
plt.show()


# In[ ]:


result = classification_report(y_test,rf_pred)

# print the result
print(result)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test,rf_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


rf_metrics = pd.Series({'Model': "Random Forest with selected features",
                     'AUC Score' : roc_auc_score(y_test, rf_pred),
                 'Precision Score': metrics.precision_score(y_test, rf_pred,average='weighted'),
                 'Recall Score': metrics.recall_score(y_test, rf_pred,average='weighted'),
                 'f1-score':metrics.f1_score(y_test, rf_pred,average='weighted'),
                 'Accuracy Score': metrics.accuracy_score(y_test, rf_pred),
                 'Train Accuracy': random_forest.score(X_train,y_train)})

# appending our result table
result_tabulation1 = result_tabulation1.append(rf_metrics , ignore_index = True)

# view the result table
result_tabulation1


# In[ ]:


max_depth = np.arange(1,5,1)
min_samples_leaf = np.arange(1,50,10)
max_leaf_nodes = np.arange(10,18,1)
n_estima = np.arange(100,200,25)

param_grid = {"criterion": ["gini", "entropy"],
              "max_depth": max_depth,
              "min_samples_leaf":min_samples_leaf,
              "max_leaf_nodes": max_leaf_nodes,
              "n_estimators":n_estima}


# In[ ]:


random_forest_Gridsearch = RandomForestClassifier()
random_forest_Gridsearch = GridSearchCV(random_forest_Gridsearch, param_grid, cv=5)
random_forest_Gridsearch.fit(X_train, y_train)


# In[ ]:


random_forest_Gridsearch.best_params_


# In[ ]:


random_forest = RandomForestClassifier(criterion= 'gini',
                                       max_depth=4,max_leaf_nodes=16,min_samples_leaf=11,n_estimators=125)

# train model
random_forest.fit(X_train, y_train)

# predict the model using 'X_test'
rf_pred = random_forest.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, rf_pred)

# plot a heatmap
sns.heatmap(cm, annot=True,fmt='d',cmap="YlGnBu")
plt.show()


# In[ ]:


result = classification_report(y_test,rf_pred)

# print the result
print(result)


# In[ ]:


roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


rf_metrics = pd.Series({'Model': "Random Forest with selected features and Grid Search",
                     'AUC Score' : roc_auc_score(y_test, rf_pred),
                 'Precision Score': metrics.precision_score(y_test, rf_pred,average='weighted'),
                 'Recall Score': metrics.recall_score(y_test, rf_pred,average='weighted'),
                 'f1-score':metrics.f1_score(y_test, rf_pred,average='weighted'),
                 'Accuracy Score': metrics.accuracy_score(y_test, rf_pred),
                 'Train Accuracy': random_forest.score(X_train,y_train)})

# appending our result table
result_tabulation1 = result_tabulation1.append(rf_metrics , ignore_index = True)

# view the result table
result_tabulation1


# # Ada Boost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


adaboost = AdaBoostClassifier()

# train model
adaboost.fit(X_train, y_train)

# predict the model using 'X_test'
ada_pred = adaboost.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, ada_pred)

# plot a heatmap
sns.heatmap(cm, annot=True,fmt='d',cmap="YlGnBu")
plt.show()


# In[ ]:


result = classification_report(y_test,ada_pred)

# print the result
print(result)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test,ada_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


rf_metrics = pd.Series({'Model': "Ada Boost with selected features",
                     'AUC Score' : roc_auc_score(y_test, ada_pred),
                 'Precision Score': metrics.precision_score(y_test, ada_pred,average='weighted'),
                 'Recall Score': metrics.recall_score(y_test, ada_pred,average='weighted'),
                 'f1-score':metrics.f1_score(y_test, ada_pred,average='weighted'),
                 'Accuracy Score': metrics.accuracy_score(y_test, ada_pred),
                 'Train Accuracy': adaboost.score(X_train,y_train)})

# appending our result table
result_tabulation1 = result_tabulation1.append(rf_metrics , ignore_index = True)

# view the result table
result_tabulation1


# # XGboost 

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


xgboost = XGBClassifier()

# train model
xgboost.fit(X_train, y_train)

# predict the model using 'X_test'
xg_pred = xgboost.predict(X_test)


# In[ ]:


cm = confusion_matrix(y_test, xg_pred)

# plot a heatmap
sns.heatmap(cm, annot=True,fmt='d',cmap="YlGnBu")
plt.show()


# In[ ]:


result = classification_report(y_test,xg_pred)

# print the result
print(result)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_test,xg_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


rf_metrics = pd.Series({'Model': "XG Boost with selected features",
                     'AUC Score' : roc_auc_score(y_test, xg_pred),
                 'Precision Score': metrics.precision_score(y_test, xg_pred,average='weighted'),
                 'Recall Score': metrics.recall_score(y_test, xg_pred,average='weighted'),
                 'f1-score':metrics.f1_score(y_test, xg_pred,average='weighted'),
                 'Accuracy Score': metrics.accuracy_score(y_test, xg_pred),
                 'Train Accuracy': xgboost.score(X_train,y_train)})

# appending our result table
result_tabulation1 = result_tabulation1.append(rf_metrics , ignore_index = True)

# view the result table
result_tabulation1


# # Stacking

# In[ ]:


from vecstack import stacking


# In[ ]:


model = [AdaBoostClassifier(),
        LogisticRegression(),
         XGBClassifier()]


# In[ ]:


S_train, S_test = stacking(model,
                           X_train, y_train, X_test,
                           
                           regression=False,

                           mode='oof_pred_bag',

                           needs_proba=False,

                           save_dir=None,

                           n_folds=4,

                           stratified=True,

                           shuffle=True,
                           
                           random_state=10)


# In[ ]:


model=XGBClassifier()
model = model.fit(S_train, y_train)
y_pred = model.predict(S_test)


# In[ ]:


cm = confusion_matrix(y_test, y_pred)

# plot a heatmap
sns.heatmap(cm, annot=True,fmt='d',cmap="YlGnBu")
plt.show()


# In[ ]:


result = classification_report(y_test,y_pred)

# print the result
print(result)


# In[ ]:


rf_metrics = pd.Series({'Model': "Stacked Final Model",
                     'AUC Score' : roc_auc_score(y_test, y_pred),
                 'Precision Score': metrics.precision_score(y_test, y_pred,average='weighted'),
                 'Recall Score': metrics.recall_score(y_test, y_pred,average='weighted'),
                 'f1-score':metrics.f1_score(y_test, y_pred,average='weighted'),
                 'Accuracy Score': metrics.accuracy_score(y_test, y_pred),
                 'Train Accuracy': model.score(S_train,y_train)})

# appending our result table
result_tabulation1 = result_tabulation1.append(rf_metrics , ignore_index = True)

# view the result table
result_tabulation1


# In[ ]:




