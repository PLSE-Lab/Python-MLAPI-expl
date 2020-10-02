#!/usr/bin/env python
# coding: utf-8

# # IBM HR Analytics
# 
# I have used this awesome dataset to compare some cool Ensemble Learning techniques like: <br>
# **Bagging <br>
# Boosting <br>
# Gradient Boosting <br>
# Extreme Gradient Boosting <br>**
# 
# To learn more about **Ensemble Techiques** visit https://priyanshu.xyz/blog/f/ensemble-learning-bagging-vs-boosting <br>
# 
# [IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df.columns


# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df['Attrition'] = df['Attrition'].map(lambda x: 1 if x=='Yes' else 0)


# ## Exploratory Data Analysis

# In[ ]:


sns.countplot(df.dtypes)


# In[ ]:


sns.pairplot(df)


# In[ ]:


sns.set(rc={'figure.figsize': (18,8)})
sns.heatmap(df.corr(),vmin=-1.0,vmax=1.0,cmap='GnBu')


# In[ ]:


df.columns


# In[ ]:


sns.set(rc={'figure.figsize': (18,8)})
sns.violinplot(df['Attrition'],df['TotalWorkingYears'],hue=df['Gender'], split=True)


# In[ ]:


sns.set(rc={'figure.figsize': (18,6)})
sns.countplot(df['YearsAtCompany'],palette='viridis')


# In[ ]:


sns.set(rc={'figure.figsize': (18,5)})
sns.countplot(df['JobRole'],palette='inferno',)


# In[ ]:


sns.set(rc={'figure.figsize': (18,8)})
sns.swarmplot(df['Attrition'], df['TotalWorkingYears'])


# In[ ]:


sns.set(rc={'figure.figsize': (18,8)})
sns.swarmplot(df['Attrition'], df['MonthlyIncome'], hue=df['MaritalStatus'], palette='inferno')


# In[ ]:


df.columns


# In[ ]:


sns.set(rc={'figure.figsize': (18,6)})
sns.kdeplot(df['YearsSinceLastPromotion'],shade=True)
sns.kdeplot(df['YearsInCurrentRole'],shade=True)
sns.kdeplot(df['YearsWithCurrManager'],shade=True)


# In[ ]:


sns.set(rc={'figure.figsize': (18,8)})
sns.lmplot('YearsAtCompany','MonthlyIncome',data=df,hue='Attrition')


# ## Summary
# 
# Most of the given features don't have a very strong correlation with Attrition. <br>
# Employees with higher experience (TotalWorkingYears) have less chance of attrition. <br>
# Most of the employees are new and in their early years at the company. Mostly 0-5 years. <br>
# New employees have higher chance of attrition. <br>
# Employees with lower income are more likely to be chosen for attrition.

# # Data Cleaning & Preparation

# In[ ]:


df.head(5)


# In[ ]:


df.info()


# In[ ]:


categorical = []
for column in df:
    if df[column].dtype == 'object':
        categorical.append(column)
categorical


# In[ ]:


cat = pd.DataFrame(df.apply(pd.Series.nunique, axis = 0))
categorical = list(cat[cat[0]<=9].sort_values(0).reset_index()['index'])
categorical


# In[ ]:


data = pd.get_dummies(df, columns=categorical, drop_first=True)


# In[ ]:


data.head(5)


# In[ ]:


sns.heatmap(data.isna())


# In[ ]:


y = data['Attrition_1']
X = data.drop(['Attrition_1'],axis=1)


# # Modeling

# # Decision Tree

# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1210)


# In[ ]:


from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


# In[ ]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

smote = SMOTE(ratio='minority', random_state=1210)
X_train_sm, y_train_sm = smote.fit_sample(X_train, y_train)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


clf = DecisionTreeClassifier(random_state=1210)


# In[ ]:


clf.fit(X_train_sm, y_train_sm)


# In[ ]:


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


# In[ ]:


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    '''
    print the accuracy score, classification report and confusion matrix of classifier
    '''
    if train:
        '''
        training performance
        '''
        pred = clf.predict(X_train)
        
        print("Train Result:\n")
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_train, pred)))
        print("Classification Report: \n {}\n".format(classification_report(y_train, pred)))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_train, pred)))

        res = cross_val_score(clf, X_train, y_train.ravel(), cv=10, scoring='accuracy')
        print("Average Accuracy: \t {0:.4f}".format(np.mean(res)))
        print("Accuracy SD: \t\t {0:.4f}".format(np.std(res)))
        print("AUROC Score: \n {}\n".format(roc_auc_score(y_train, pred)))
        
        
    elif train==False:
        '''
        test performance
        '''
        pred = clf.predict(X_test)
        
        print("Test Result:\n")        
        print("accuracy score: {0:.4f}\n".format(accuracy_score(y_test, pred)))
        print("Classification Report: \n {}\n".format(classification_report(y_test, pred)))
        print("Confusion Matrix: \n {}\n".format(confusion_matrix(y_test, pred)))
        print("AUROC Score: \n {}\n".format(roc_auc_score(y_test, pred)))    
        


# In[ ]:


print_score(clf, X_train_sm, y_train_sm, X_test, y_test, train=True)


# In[ ]:


print_score(clf, X_train, y_train, X_test, y_test, train=False)


# The result is clearly not satisfactory. We will revisit this project after we covered ensemble model.

# ****

# # Bagging

# In[ ]:


from sklearn.ensemble import BaggingClassifier


# In[ ]:


bag_clf = BaggingClassifier(base_estimator=clf, n_estimators=5000,
                            bootstrap=True, n_jobs=-1, random_state=1210)


# In[ ]:


bag_clf.fit(X_train_sm, y_train_sm.ravel())


# In[ ]:


print_score(bag_clf, X_train_sm, y_train_sm, X_test, y_test, train=True)
print_score(bag_clf, X_train_sm, y_train_sm, X_test, y_test, train=False)


# ***

# # Random Forest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


rf_clf = RandomForestClassifier(n_estimators=500,class_weight={0:1,1:5})


# In[ ]:


rf_clf.fit(X_train_sm, y_train_sm.ravel())


# In[ ]:


print_score(rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(rf_clf, X_train, y_train, X_test, y_test, train=False)


# In[ ]:


pd.Series(rf_clf.feature_importances_, 
         index=X.columns).sort_values(ascending=False).plot(kind='bar', figsize=(18,6));


# # AdaBoost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier


# In[ ]:


ada_clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.01)


# In[ ]:


ada_clf.fit(X_train_sm, y_train_sm.ravel())


# In[ ]:


print_score(ada_clf, X_train, y_train, X_test, y_test, train=True)
print()
print_score(ada_clf, X_train, y_train, X_test, y_test, train=False)


# In[ ]:





# In[ ]:





# ***

# # AdaBoost + RandomForest

# In[ ]:


ada_rf_clf = AdaBoostClassifier(RandomForestClassifier(n_estimators=500,class_weight={0:1,1:5}))
ada_rf_clf.fit(X_train_sm, y_train_sm.ravel())


# In[ ]:


print_score(ada_rf_clf, X_train, y_train, X_test, y_test, train=True)
print_score(ada_rf_clf, X_train, y_train, X_test, y_test, train=False)


# ***

# # Gradient Boosting Classifier

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier


# In[ ]:


gbc_clf = GradientBoostingClassifier(n_estimators=1000,learning_rate=0.01)
gbc_clf.fit(X_train_sm, y_train_sm)


# In[ ]:


print_score(gbc_clf, X_train, y_train, X_test, y_test, train=True)
print_score(gbc_clf, X_train, y_train, X_test, y_test, train=False)


# ***

# # XGBoost

# In[ ]:


import xgboost as xgb


# In[ ]:


xgb_clf = xgb.XGBClassifier()
xgb_clf.fit(X_train, y_train.ravel())


# In[ ]:


warnings.filterwarnings("ignore")
print_score(xgb_clf, X_train_sm, y_train_sm, X_test, y_test, train=True)
print_score(xgb_clf, X_train_sm, y_train_sm, X_test, y_test, train=False)


# ***

# # Light GBM

# In[ ]:


from lightgbm import LGBMClassifier


# In[ ]:


lgbm = LGBMClassifier(nthread=4, n_estimators=10000,
            learning_rate=0.02, num_leaves=32,
            colsample_bytree=0.9497036, subsample=0.8715623,
            max_depth=8, reg_alpha=0.04,
            reg_lambda=0.073, min_split_gain=0.0222415,
            min_child_weight=40, silent=-1,
            verbose=-1)


# In[ ]:


lgbm.fit(X_train_sm, y_train_sm, eval_set=[(X_train_sm, y_train_sm), (X_test, y_test)], 
            eval_metric= 'auc', verbose= 1000, early_stopping_rounds= 100000)


# In[ ]:


warnings.filterwarnings("ignore")
print_score(lgbm, X_train_sm, y_train_sm, X_test, y_test, train=True)
print_score(lgbm, X_train_sm, y_train_sm, X_test, y_test, train=False)


# # Results Comparison

# In[ ]:


warnings.filterwarnings("ignore")
AUROC_dict = {'Decision Tree Classifier': roc_auc_score(y_test, clf.predict(X_test))}
AUROC_dict['Random Forest Classifier'] = roc_auc_score(y_test, rf_clf.predict(X_test))
AUROC_dict['Bagging Classifier'] = roc_auc_score(y_test, bag_clf.predict(X_test))
AUROC_dict['AdaBoost Classifier'] = roc_auc_score(y_test, ada_clf.predict(X_test))
AUROC_dict['AdaBoost + RandomForest Classifier'] = roc_auc_score(y_test, ada_rf_clf.predict(X_test))
AUROC_dict['XGBoost Classifier'] = roc_auc_score(y_test, xgb_clf.predict(X_test))
AUROC_dict['LightGBM Classifier'] = roc_auc_score(y_test, lgbm.predict(X_test))
AUROC_dict['Gradient Boosting Classifier'] = roc_auc_score(y_test, gbc_clf.predict(X_test))


# In[ ]:


AUROC = pd.DataFrame(pd.Series(AUROC_dict, index=AUROC_dict.keys(), name='AUROC Score'))


# In[ ]:


AUROC.sort_values('AUROC Score', ascending= False).plot(kind='bar', title = 'AUROC Score', legend = False)


# # Best Results?

# Although **AdaBoost** offers some great results but **LightGBM Classifier** wins the race with 88.44% Accuracy Score and 87% F1-score and 70% Area under ROC Curve.
