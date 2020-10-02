#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV, KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, Normalizer, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# In[ ]:


train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')
sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')


# In[ ]:


train


# In[ ]:


y = train['Survived']


# In[ ]:


test


# In[ ]:


traintest = pd.concat([train.drop('Survived', axis=1),test]).reset_index()


# In[ ]:


traintest


# In[ ]:


traintest.info()


# In[ ]:


traintest[traintest['Age'].isnull()]


# In[ ]:


traintest[traintest['Age']<14]


# In[ ]:


traintest['Cabin'].describe()


# In[ ]:


traintest['Ticket'].describe()


# In[ ]:


traintest['Title'] = traintest.Name.str.extract('([A-Za-z]+)\.')


# In[ ]:


traintest_drop3 = traintest.drop(['Name', 'Ticket', 'Cabin'], axis=1)


# In[ ]:


traintest_drop3.Title.unique()


# In[ ]:


traintest_drop3[traintest_drop3['Title']=='Mrs']['Age'].sort_values().value_counts(sort=False,dropna=False) 


# In[ ]:


traintest_drop3[traintest_drop3['Title']=='Master']['Age'].value_counts(dropna=False)


# In[ ]:


traintest_drop3[traintest_drop3['Title']=='Mr']['Age'].sort_values().value_counts(sort=False,dropna=False) 


# In[ ]:


traintest_drop3[traintest_drop3['Title']=='Miss']['Age'].sort_values().value_counts(sort=False,dropna=False) 


# In[ ]:


master_nan_median = np.median(traintest_drop3[traintest_drop3['Title']=='Master']['Age'].dropna().values)
master_nan_median


# In[ ]:


traintest_drop3[traintest_drop3['Title']=='Master']['Age']


# In[ ]:


# Setting the NaN ages to the median age of the people with that title.
traintest_drop3.loc[(traintest_drop3['Title']=='Master') & (traintest_drop3['Age'].isna()),'Age']  = master_nan_median


# In[ ]:


traintest_drop3.loc[traintest_drop3['Title']=='Master', 'Age']


# In[ ]:


mr_nan_median = np.median(traintest_drop3[traintest_drop3['Title']=='Mr']['Age'].dropna().values)
mr_nan_median


# In[ ]:


traintest_drop3.loc[(traintest_drop3['Title']=='Mr') & (traintest_drop3['Age'].isna()),'Age']  = mr_nan_median


# In[ ]:


mrs_nan_median = np.median(traintest_drop3[traintest_drop3['Title']=='Mrs']['Age'].dropna().values)
mrs_nan_median


# In[ ]:


traintest_drop3.loc[(traintest_drop3['Title']=='Mrs') & (traintest_drop3['Age'].isna()),'Age']  = mrs_nan_median


# In[ ]:


miss_nan_median = np.median(traintest_drop3[traintest_drop3['Title']=='Miss']['Age'].dropna().values)
miss_nan_median


# In[ ]:


traintest_drop3.loc[(traintest_drop3['Title']=='Miss') & (traintest_drop3['Age'].isna()),'Age']  = miss_nan_median


# In[ ]:


traintest_drop3['Age'].isna().sum()


# In[ ]:


still_nan = np.where(traintest_drop3['Age'].isna())[0].tolist()
still_nan


# In[ ]:


traintest_drop3.iloc[still_nan]


# In[ ]:


traintest_drop3.loc[(traintest_drop3['Title']=='Ms') & (traintest_drop3['Age'].isna()),'Age'] = mrs_nan_median


# In[ ]:


traintest_drop3.loc[(traintest_drop3['Title']=='Dr') & (traintest_drop3['Age'].isna()),'Age']  = mr_nan_median


# In[ ]:


fare_nan = np.where(traintest_drop3['Fare'].isna())[0].tolist()


# In[ ]:


traintest_drop3.iloc[fare_nan]


# In[ ]:


pclass3_median = np.median(traintest_drop3[traintest_drop3['Pclass']==3]['Fare'].dropna().values)
pclass3_median


# In[ ]:


traintest_drop3.loc[(traintest_drop3['Fare'].isna()),'Fare'] = pclass3_median


# In[ ]:


traintest_drop3['Fare'].isna().sum()


# In[ ]:


emb_nan = np.where(traintest_drop3['Embarked'].isna())[0].tolist()
traintest_drop3.iloc[emb_nan]


# In[ ]:


traintest_drop3[traintest_drop3['Fare']==80.0]


# In[ ]:


traintest_drop3[traintest_drop3['Pclass']==1]


# In[ ]:


traintest_drop3[traintest_drop3['PassengerId']==61]


# In[ ]:


traintest_drop3[traintest_drop3['PassengerId']==63]


# In[ ]:


traintest_drop3[traintest_drop3['Pclass']==1]['Embarked'].value_counts() 


# In[ ]:


traintest_drop3.loc[(traintest_drop3['Embarked'].isna()),'Embarked'] = 'S'


# In[ ]:


traintest_drop3.info()


# In[ ]:


oenc = OrdinalEncoder()
traintest_objenc = pd.DataFrame(oenc.fit_transform(traintest_drop3.select_dtypes('object')),columns = traintest_drop3.select_dtypes('object').columns)


# In[ ]:


traintest_num = traintest_drop3.select_dtypes(['int64','float64']).drop(['index','PassengerId'], axis=1).join(traintest_objenc)


# In[ ]:


traintest_num


# In[ ]:


traintest_norm =  traintest_drop3[['index','PassengerId']].join(pd.DataFrame(StandardScaler().fit_transform(traintest_num), columns = traintest_num.columns))
#traintest_norm =  traintest_drop3[['index','PassengerId']].join(pd.DataFrame(MinMaxScaler().fit_transform(traintest_num), columns = traintest_num.columns))


# In[ ]:


traintest_norm


# In[ ]:


train_norm = traintest_norm[0:891].drop('index', axis=1)


# In[ ]:


train_norm


# In[ ]:


test_norm = traintest_norm[891::].drop('index', axis=1).reset_index().drop('index',axis=1)


# In[ ]:


test_norm


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_norm, y)


# In[ ]:




modelgbr = GradientBoostingClassifier( random_state=42)
modelgbr.fit(X_train, y_train)
y_predgbr = modelgbr.predict(X_test)
accuracy_score(y_test, y_predgbr)


# In[ ]:




pd.DataFrame(modelgbr.feature_importances_, X_train.columns).sort_values(by=0,ascending=False) 


# In[ ]:


def model_test(testmodel):
    model = testmodel()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


# In[ ]:


models = [RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, 
          BaggingClassifier, ExtraTreesClassifier, DecisionTreeClassifier, KNeighborsClassifier]


# In[ ]:


for i in models:
    print(i)
    print(model_test(i))


# In[ ]:


modelclf = GradientBoostingClassifier()
parameters = {'n_estimators':[50,200,300], 'learning_rate':[0.001,0.01,0.1],'max_depth':[1,3,6]}
clf = GridSearchCV(modelclf, parameters)
clf.fit(X_train, y_train)
y_predclf = clf.predict(X_test)
accuracy_score(y_test, y_predclf)
clf.cv_results_
clf.best_params_


# In[ ]:


modelsub = GradientBoostingClassifier(learning_rate = 0.01, max_depth = 3, n_estimators = 300)
modelsub.fit(train_norm.drop('PassengerId',axis=1), y)
preds_test = modelsub.predict(test_norm.drop('PassengerId',axis=1))


# In[ ]:





# In[ ]:




submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": preds_test
    })
submission.to_csv('submission.csv', index=False)




# In[ ]:




