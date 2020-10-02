#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from datetime import datetime
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = [6, 6]


# In[ ]:


features_to_select = 20


# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


train.Survived.sum()


# In[ ]:


train.Survived.count()


# In[ ]:


train.Survived.mean()


# In[ ]:


train.info()


# - Null count in Age column is significant. We can drop this field because Title is a decent proxy.
# - Null count in Embarked column is not significant, just 2. But the field will be useful, so we impute nulls with 'S'.
# - Null count in Cabin column is significant. We can drop this field.

# In[ ]:


train.drop(columns=['Age', 'Cabin'], inplace=True)


# In[ ]:


train['Title'] = pd.DataFrame(train.Name.str.split(',', expand=True).values, columns=['LN', 'TFN'])                                 ['TFN'].str.split('.', expand=True)[0].str.strip()


# In[ ]:


train.Embarked = train.Embarked.fillna('S')


# In[ ]:


train.info()


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


test['Title'] = pd.DataFrame(test.Name.str.split(',', expand=True).values, columns=['LN', 'TFN'])                               ['TFN'].str.split('.', expand=True)[0].str.strip()


# In[ ]:


test.info()


# - Null count in Age column is significant. We can drop this field because Title is a decent proxy.
# - Only one record with null in Fare field. We put zero and move on as the field will be useful.
# - Null count in Cabin column is significant. We can drop this column.

# In[ ]:


test.drop(columns=['Age', 'Cabin'], inplace=True)


# In[ ]:


test.Fare = test.Fare.fillna(0)


# In[ ]:


test.info()


# In[ ]:


# adjusting data situation so that columns after dummies match for both train and test sets
test.Parch = test.Parch.apply(lambda x: 6 if x == 9 else x)


# In[ ]:


cat_features = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Title']


# In[ ]:


# train data groups
print('Train')
for col in cat_features:
    train[col] = train[col].apply(str)
    print(train.groupby(col).count()['PassengerId'])
    print('Train\n', train.groupby(col).mean()['Survived'])    


# In[ ]:


# test data groups
for col in cat_features:
    test[col] = test[col].apply(str)
    print(test.groupby(col).count()['PassengerId'])


# In[ ]:


train = pd.concat([train.Survived, train['Fare'], pd.get_dummies(train[cat_features], drop_first=True)], axis=1)
train.head()


# In[ ]:


hold_PassengerID = test['PassengerId']


# In[ ]:


test = pd.concat([test['Fare'], pd.get_dummies(test[cat_features], drop_first=True)], axis=1)
test.head()


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


y_train = train.pop('Survived')
X_train = sm.add_constant(train)


# In[ ]:


# run recursive feature selection to pick the recommended fields
rfe = RFE(LogisticRegression(), features_to_select).fit(X_train, y_train)
df_rfe = pd.DataFrame(list(zip(X_train.columns, rfe.support_, rfe.ranking_)), columns=['feature', 'sel_flag', 'sel_rank'])
df_rfe[df_rfe.sel_flag].feature


# In[ ]:


def model():
    logmodel = sm.GLM(y_train, X_train, family=sm.families.Binomial()).fit()
    print(logmodel.summary())
    return logmodel

def validate_model(cutoff, print_flag=False):
    
    y_train_pred = round(logmodel.predict(X_train), 6)

    df_result = pd.DataFrame()
    df_result['Survived_Train'] = y_train
    df_result['Predicted Probability'] = y_train_pred
    df_result['Survived'] = y_train_pred.apply(lambda x: 1 if x > cutoff else 0)

    confusion = metrics.confusion_matrix(df_result.Survived_Train, df_result.Survived)
    accuracy = round(metrics.accuracy_score(df_result.Survived_Train, df_result.Survived), 6)
    
    TP = confusion[1,1] # true positive 
    TN = confusion[0,0] # true negatives
    FP = confusion[0,1] # false positives
    FN = confusion[1,0] # false negatives

    sensitivity = round(TP / float(TP+FN), 6)
    specificity = round(TN / float(TN+FP), 6)

    if print_flag:
        print('\nConfusion Matrix')
        print(confusion)
        print('\nAccuracy')
        print(accuracy)
        print('\nSensitivity (TP / [TP + FN])')
        print(sensitivity)
        print('\nSpecificity (TN / [TN + FP])')
        print(specificity)
        
    return accuracy, sensitivity, specificity

def find_ideal_cutoff():
    df_cutoff = pd.DataFrame(columns = ['cutoff','accuracy','sensitivity','specificity'])
    for cutoff in np.arange(0.0, 1.0, 0.05):
        cutoff = round(cutoff, 2)
        accuracy, sensitivity, specificity = validate_model(cutoff)
        df_cutoff.loc[cutoff] = [cutoff, accuracy, sensitivity, specificity]
    df_cutoff.plot.line(x='cutoff', y=['accuracy','sensitivity','specificity'], grid=True)
    plt.grid(b=True, which='minor', linestyle='-')
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.show()
    
# create a dataframe that will contain the names of all the feature variables and their respective VIFs
def performVIFanalysis():
    vif = pd.DataFrame()
    vif['Features'] = X_train.columns
    vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    print(vif)


# In[ ]:


X_train = X_train[df_rfe[df_rfe.sel_flag].feature]
logmodel = model()


# In[ ]:


find_ideal_cutoff()


# In[ ]:


df_hold = pd.DataFrame()


# In[ ]:


df_hold['const'] = X_train.pop('const') # highest p-value
logmodel = model()
find_ideal_cutoff()


# In[ ]:


df_hold['Parch_6'] = X_train.pop('Parch_6') # highest p-value
logmodel = model()
find_ideal_cutoff()


# In[ ]:


df_hold['Title_Don'] = X_train.pop('Title_Don') # highest p-value
logmodel = model()
find_ideal_cutoff()


# In[ ]:


df_hold['Title_Jonkheer'] = X_train.pop('Title_Jonkheer') # highest p-value
logmodel = model()
find_ideal_cutoff()


# In[ ]:


df_hold['SibSp_5'] = X_train.pop('SibSp_5') # highest p-value
logmodel = model()
find_ideal_cutoff()


# In[ ]:


df_hold['SibSp_8'] = X_train.pop('SibSp_8') # highest p-value
logmodel = model()
find_ideal_cutoff()


# In[ ]:


df_hold['Parch_4'] = X_train.pop('Parch_4') # highest p-value
logmodel = model()
find_ideal_cutoff()


# In[ ]:


df_hold['Title_Rev'] = X_train.pop('Title_Rev') # highest p-value
logmodel = model()
find_ideal_cutoff()


# In[ ]:


df_hold['Title_Dr'] = X_train.pop('Title_Dr') # highest p-value
logmodel = model()
find_ideal_cutoff()


# In[ ]:


df_hold['Title_Mr'] = X_train.pop('Title_Mr') # highest p-value
logmodel = model()
find_ideal_cutoff()


# In[ ]:


df_hold['Sex_male'] = X_train.pop('Sex_male') # highest p-value
logmodel = model()
find_ideal_cutoff()


# In[ ]:


df_hold['Parch_5'] = X_train.pop('Parch_5') # highest p-value
logmodel = model()
find_ideal_cutoff()


# In[ ]:


performVIFanalysis()


# In[ ]:


chosen_prob_cutoff = 0.6
validate_model(chosen_prob_cutoff, False)


# In[ ]:


X_test = test[X_train.columns]
y_test_pred = round(logmodel.predict(X_test), 6)


# In[ ]:


df_out = pd.DataFrame()
df_out['PassengerID'] = hold_PassengerID
df_out['Survived'] = y_test_pred.apply(lambda x: 1 if x > chosen_prob_cutoff else 0)


# In[ ]:


df_out['Survived'].sum()


# In[ ]:


df_out['Survived'].count()


# In[ ]:


df_out['Survived'].mean()

