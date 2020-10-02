#!/usr/bin/env python
# coding: utf-8

# # Findings

# **Following observations can be made from this dataset**
# * Number of female survivors is far more than male survivors.
# * Most people belonging to Passenger class 3 (lowest class) did not survived.
# * Most of the young passengers were travelling my third passenger class.
# * Passengers with expensive tickets had a better survival rate.
# * Number of siblings have a positive correlation with survival rate i.e people with their siblings on board had better chance of survival.

# **Main Code:**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


sns.set_style('whitegrid')


# Importing the data

# In[ ]:


train_data = pd.read_csv('/kaggle/input/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/titanic/test.csv')


# In[ ]:


train_data.head()


# In[ ]:


train_data.info()


# # Visualization

# Lets plot some graphs using seaborn library

# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(data=train_data,x='Sex',hue='Survived',palette='RdBu_r')


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='Embarked',data=train_data,hue='Survived',palette='RdBu_r')


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='Pclass',data=train_data,hue='Sex',palette='RdBu_r')


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='Pclass',data=train_data,hue='Survived',palette='RdBu_r')


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='Parch',data=train_data,hue='Sex',palette='RdBu_r')


# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(x='Pclass',y='Age',data=train_data,hue='Sex',palette='RdBu_r')


# In[ ]:


plt.figure(figsize=(12,6))
sns.countplot(x='Pclass',data=train_data,hue='Sex',palette='RdBu_r')


# In[ ]:


plt.figure(figsize=(12,6))
sns.barplot(data=train_data,y='Fare',x='Pclass',palette='RdBu_r')


# In[ ]:


plt.figure(figsize=(12,6))
sns.heatmap(data=train_data.corr(),cmap="YlGnBu")


# # Feature Engineering

# Fixing feature 'Embarked'

# In[ ]:


train_data['Embarked'] = train_data['Embarked'].fillna('S')
test_data['Embarked'] = test_data['Embarked'].fillna('S')


# Fixing feature age

# From the graphs it is clear that Age depends upon the Pclass, therefore using the median per class to fill missing age

# In[ ]:


def fix_age(blob):
    Age = blob[0]
    Pclass = blob[1]
    Sex = blob[2]
    
    if pd.isnull(Age):
        if Sex == 'male':
            if Pclass == 1:
                return 40
            elif Pclass == 2:
                return 30
            else:
                return 25
        else:
            if Pclass == 1:
                return 35

            elif Pclass == 2:
                return 27

            else:
                return 22
    else:
        return Age


# In[ ]:


train_data['Age'] = train_data[['Age','Pclass','Sex']].apply(fix_age,axis=1)
test_data['Age'] = test_data[['Age','Pclass','Sex']].apply(fix_age,axis=1)


# Fixing 'Fare' feature

# In[ ]:


train_data["Fare"] = train_data["Fare"].fillna(train_data["Fare"].median())
test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].median())


# Fixing SibSp

# In[ ]:


train_data['SibSp'].fillna(-1,inplace=True)
test_data['SibSp'].fillna(-1,inplace=True)


# Fixing Parch

# In[ ]:


train_data['Parch'].fillna(-1,inplace=True)
test_data['Parch'].fillna(-1,inplace=True)


# Fixing Sex

# In[ ]:


train_data['Sex'] = train_data['Sex'].fillna('male')
test_data['Sex'] = test_data['Sex'].fillna('male')


# Extracting features from 'Name' column

# In[ ]:


def add_family(blob):
    temp = blob.split(' ')[0]
    temp = temp[:len(temp)-1]
    return temp


# In[ ]:


train_data['FamilyName'] = train_data['Name'].apply(add_family)


# In[ ]:


test_data['FamilyName'] = test_data['Name'].apply(add_family)


# In[ ]:


train_data['FamilyName'].value_counts()


# Adding Name suffix

# In[ ]:


def apply_name_suffix(blob):
    temp = blob.split(' ')[1]
    return temp


# In[ ]:


train_data['NameSuffix'] = train_data['Name'].apply(apply_name_suffix)


# In[ ]:


test_data['NameSuffix'] = test_data['Name'].apply(apply_name_suffix)


# Fixing name suffix

# In[ ]:


def fix_name_suffix(blob):
    temp = ['Mr.','Miss.','Mrs.','Master.','Dr.','Rev.']
    if blob in temp:
        return blob[0:len(blob)-1]
    else:
        return 'no_suffix'


# In[ ]:


train_data['NameSuffix'] = train_data['NameSuffix'].apply(fix_name_suffix)
test_data['NameSuffix'] = test_data['NameSuffix'].apply(fix_name_suffix)


# Deleting Name feature

# In[ ]:


del train_data['Name']
del test_data['Name']


# Fixing ticket column

# In[ ]:


train_data.Ticket = [i[0] for i in train_data.Ticket.astype("str")]
test_data.Ticket = [i[0] for i in test_data.Ticket.astype("str")]


# In[ ]:


train_data['Ticket'] = train_data['Ticket'].apply(lambda x: x if x in ['S','P','C','A','W','F','L'] else 'G')


# In[ ]:


test_data['Ticket'] = test_data['Ticket'].apply(lambda x: x if x in ['S','P','C','A','W','F','L'] else 'G')


# Fixing cabin

# In[ ]:


train_data.Cabin = [i[0] for i in train_data.Cabin.astype("str")]
test_data.Cabin = [i[0] for i in test_data.Cabin.astype("str")]


# In[ ]:


def most_common(lst):
    return max(set(lst), key=lst.count)
def fix_cabin(blob):
    temp = blob.tolist()
    temp1 = []
    for i in temp :
        if i == 'n':
            continue
        else:
            temp1.append(i)
    if temp1 == [] :
        return 'Z'
    else:
        temp1 = most_common(temp1)
        return temp1


# In[ ]:


train_data['Cabin'] = train_data['Cabin'].groupby(train_data['FamilyName']).transform(fix_cabin)
test_data['Cabin'] = test_data['Cabin'].groupby(test_data['FamilyName']).transform(fix_cabin)


# In[ ]:


del train_data['PassengerId']


# In[ ]:


from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction import DictVectorizer


# # Pipeline

# In[ ]:


list_of_obj = []
list_of_num = []
for i in train_data.columns :
    if train_data[i].dtypes == 'object':
        list_of_obj.append(i)
    else:
        if i == 'Survived':
            continue
        else:
            list_of_num.append(i)


# In[ ]:


def return_text(df):
    return df[list_of_obj]


# In[ ]:


get_text = FunctionTransformer(func=return_text,validate=False)


# In[ ]:


def return_num(df):
    return df[list_of_num]


# In[ ]:


get_numerical = FunctionTransformer(func=return_num,validate=False)


# In[ ]:


num_pipeline = Pipeline([
    ('numerical',get_numerical),
])


# In[ ]:


def return_dict(blob):
    return blob.to_dict("records")


# In[ ]:


text_pipeline = Pipeline([
    ('textual',get_text),
    ('dictifier',FunctionTransformer(func=return_dict,validate=False)),
    ('vectorizer',DictVectorizer(sort=False)),
])


# # Model

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    train_data.drop(['Survived'],axis=1), train_data['Survived'], test_size=0.2, random_state=42)


# Lets try to use XGBClassifier with it

# In[ ]:


from sklearn.metrics import confusion_matrix , classification_report, roc_auc_score


# In[ ]:


import xgboost as xgb


# In[ ]:


xgb_model = xgb.XGBClassifier(silent=False, 
                      scale_pos_weight=1,
                      learning_rate=0.01,  
                      colsample_bytree = 0.4,
                      subsample = 0.8,
                      objective='binary:logistic', 
                      n_estimators=500, 
                      reg_alpha = 0.3,
                      gamma=10,
                      eval_metric = 'auc')


# In[ ]:


pipeline = Pipeline([
    ('union',FeatureUnion(
        transformer_list = [
            ('num',num_pipeline),
            ('text',text_pipeline)
        ])),
    ('clf',xgb_model)
])


# In[ ]:


pipeline.fit(X_train,y_train)


# In[ ]:


preds_xgb = pipeline.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,preds_xgb))
print(roc_auc_score(y_test,preds_xgb))
print(classification_report(y_test,preds_xgb))


# trying to improve upon xgbclassifier using randomizedSearch

# In[ ]:


from sklearn.model_selection import RandomizedSearchCV


# In[ ]:


params = {
    "clf__learning_rate"    : list(np.arange(0.05,0.6,0.05)) ,
    "clf__max_depth"        : list(np.arange(1,20,2)),
    "clf__min_child_weight" : list(np.arange(1,9,1)),
    "clf__gamma"            : list(np.arange(1,20,1)),
    "clf__colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
    "clf__subsample" : list(np.arange(0.1,0.9,0.1)),
    "clf__n_estimators" : [200,500,800,1000,1500,2000],
    "clf__reg_alpha" : list(np.arange(0.1,0.9,0.1))
         }


# In[ ]:


grid = RandomizedSearchCV(pipeline,
                         param_distributions=params,
                         scoring='roc_auc',cv=5,verbose=5)


# In[ ]:


grid.fit(X_train,y_train)


# In[ ]:


grid.best_params_, grid.best_score_


# In[ ]:


best_matrix = {'subsample': 0.30000000000000004,
  'reg_alpha': 0.2,
  'n_estimators': 500,
  'min_child_weight': 3,
  'max_depth': 9,
  'learning_rate': 0.15000000000000002,
  'gamma': 4,
  'colsample_bytree': 0.3}


# In[ ]:


xgb_model = xgb.XGBClassifier(**best_matrix,silent=False, 
                      eval_metric = 'auc')


# In[ ]:


finalized_model = pipeline = Pipeline([
    ('union',FeatureUnion(
        transformer_list = [
            ('num',num_pipeline),
            ('text',text_pipeline)
        ])),
    ('clf',xgb_model)
])


# # Submission

# Training on full data now

# In[ ]:


X = train_data.drop(['Survived'],axis=1)
y = train_data['Survived']


# In[ ]:


finalized_model.fit(X,y)


# In[ ]:


final_test = test_data.drop(['PassengerId'],axis=1)


# In[ ]:


PassengerId = test_data['PassengerId']


# In[ ]:


final_prediction = finalized_model.predict(final_test)


# In[ ]:


submission = pd.DataFrame({ 'PassengerId': PassengerId,
                            'Survived': final_prediction })
submission.to_csv(path_or_buf ="Titanic_Submission.csv", index=False)


# In[ ]:




