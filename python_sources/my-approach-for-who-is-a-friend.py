#!/usr/bin/env python
# coding: utf-8

# # Who is a Friend?

# ![](https://turbologo.com/articles/wp-content/uploads/2019/12/friends-logo-cover-1280x720.png)

# ### Predict Friends or Not
# * In this machine learning challenge you are to predict whether two persons are friends or not.You are given a data of recorded events from a group of people and using the given data you have to predict whether they are friends or not.

# ## **Data Dictionary**
# * **Person A**: Name of Person A
# * **Person B** : Name of Person B
# * **Years of Knowing** : Years of Person A knowing Person B
# * **Interaction Type**: The type of interaction they had
# * **Interaction Duration** : The duration of interaction they had (in minutes)
# * **Moon Phase During Interaction** : The Moon Phase during their interaction

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Import modules

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier,StackingClassifier,VotingClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier,BaggingClassifier,ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

import xgboost as xgb
import lightgbm as lgb
import catboost as cat


# ## Import Datasets

# In[ ]:


train = pd.read_csv("/kaggle/input/whoisafriend/train.csv")
test = pd.read_csv("/kaggle/input/whoisafriend/test.csv")
sample = pd.read_csv("/kaggle/input/whoisafriend/sample_submission.csv")


# In[ ]:


print(train.shape,test.shape)


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## Checking for NUll values

# In[ ]:


train.isnull().sum()


# In[ ]:


test.isnull().sum()


# ## Unique values
# 

# In[ ]:


train.nunique()


# In[ ]:


test.nunique()


# ### Value Count of target feature
# In this data, the target class is balanced, the number of class 0 and class 1 are approximately equal

# In[ ]:


train['Friends'].value_counts()


# # EDA

# In this part all the features can be analysed for getting some insights.
# 1. In Moon Phase During Interaction and Interaction Type features are having equal distribution amoung different classes.
# Now we have to check whether removing these two features makes any difference to the model or not.

# In[ ]:


plt.figure(figsize=(15, 4))
sns.countplot(train['Moon Phase During Interaction'], hue=train['Friends'])
plt.show()


# In[ ]:


plt.figure(figsize=(15, 4))
sns.countplot(train['Interaction Type'], hue=train['Friends'])
plt.show()


# In[ ]:


# Create a copy of the train and test
dtrain = train.copy()
dtest = test.copy()


# # Label Encoding

# In[ ]:


cat_feat = ['Person A','Person B','Interaction Type','Moon Phase During Interaction']


# In[ ]:


import warnings
warnings.warn('my warning')


# In[ ]:


# Combine the train and test data for label encoding
df = pd.concat([train,test])

le = LabelEncoder()
for i in cat_feat:
    df[i] = le.fit_transform(df[i])

train = df[df['Friends'].notnull()]
test = df[df['Friends'].isnull()]

del df

train['Friends']  = train['Friends'].astype(int)


# ## Features

# In[ ]:


features = list(set(train.columns)-set(['Friends','ID']))
target = 'Friends'
features


# ## Splitting the dataset

# In[ ]:


X_train, X_test, y_train, y_test  = train_test_split(train[features],train[target],test_size=0.3,random_state=45)


# In[ ]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# # Baseliner Models

# In[ ]:


def baseliner(train, features, target, cv=3, metric='accuracy'):
    """
    Function for baselining Models which return CV Score, Train Score, Valid Score
    """
    print("Baseliner Models\n")
    eval_dict = {}
    models = [lgb.LGBMClassifier(), xgb.XGBClassifier(), cat.CatBoostClassifier(verbose=0), GradientBoostingClassifier(), LogisticRegression(), 
              RandomForestClassifier(), DecisionTreeClassifier(), AdaBoostClassifier()
             ]
    print("Model Name \t |   CV")
    print("--" * 50)

    for index, model in enumerate(models, 0):
        model_name = str(model).split("(")[0]
        eval_dict[model_name] = {}

        results = cross_val_score(model, train[features], train[target], cv=cv, scoring=metric)
        eval_dict[model_name]['cv'] = results.mean()

        print("%s \t | %.4f \t" % (
            model_name[:12], eval_dict[model_name]['cv']))


# In[ ]:


baseliner(train,features,target)


# In[ ]:


def cross_validation_function(model,train,features,cv):
    results = cross_val_score(model, train[features], train[target], cv=cv, scoring='accuracy')
    return print("Cross Validation Score:",results.mean())


# In[ ]:


model = lgb.LGBMClassifier(random_state=7)
cross_validation_function(model,train,features,cv=10)


# In[ ]:


lgb_model = lgb.LGBMClassifier(random_state=7)
lgb_model.fit(X_train,y_train)
y_pred = lgb_model.predict(X_test)
accuracy_score(y_test,y_pred)


# ## Plot the Feature Importance

# In[ ]:


importances = lgb_model.feature_importances_
indices = np.argsort(importances)

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='g', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:


def fit_model(model,train,features,target,test):
    model.fit(train[features],train[target])
    y_pred_test = model.predict(test[features])
    test['Friends'] = y_pred_test
    test['Friends'] = test['Friends'].astype(int)
    return test


# In[ ]:


model = lgb.LGBMClassifier(random_state=7)
test = fit_model(model,train,features,target,test)
test[['ID','Friends']].to_csv("lgb_model.csv",index=False)


# In[ ]:


features.remove("Moon Phase During Interaction")
features


# On Further Checking all the models, by removing "moon phase during interaction" feature the model was performing well.

# # Ensembling Method

# ![](https://image.slidesharecdn.com/ensemblelearning-120730220523-phpapp02/95/ensemble-learning-the-wisdom-of-crowds-of-machines-15-728.jpg?cb=1343685993)

# In[ ]:


from sklearn.ensemble import VotingClassifier

lgb_model  = lgb.LGBMClassifier(random_state=7)
ada_model  = AdaBoostClassifier(random_state=7)
grb_model = GradientBoostingClassifier(random_state=7)


# In[ ]:


eclf1 = VotingClassifier(estimators=[
        ('lgb', lgb_model), ('ada', ada_model), ('grb', grb_model)], voting='hard')


# In[ ]:


test = fit_model(eclf1,train,features,target,test)
test[['ID','Friends']].to_csv("ens_grb_lgb_ada_model.csv",index=False)


# # Stacking

# ![](https://www.researchgate.net/publication/324552457/figure/fig3/AS:616245728645121@1523935839872/An-example-scheme-of-stacking-ensemble-learning.png)

# In[ ]:


# Baseliners Level 0
lgb_model  = lgb.LGBMClassifier(random_state=7)
lgb_model.fit(train[features],train[target])
y_pred_train_lgb = lgb_model.predict(train[features])
y_pred_test_lgb = lgb_model.predict(test[features])



cat_model  = cat.CatBoostClassifier(verbose=0,random_state=7)
cat_model.fit(train[features],train[target])
y_pred_train_cat = cat_model.predict(train[features])
y_pred_test_cat = cat_model.predict(test[features])


grb_model = GradientBoostingClassifier(random_state=7)
grb_model.fit(train[features],train[target])
y_pred_train_grb = grb_model.predict(train[features])
y_pred_test_grb = grb_model.predict(test[features])


# In[ ]:


train_pred = {
    'cat':y_pred_train_cat,
    'grb':y_pred_train_grb,
    'lgb':y_pred_train_lgb
}
train_df = pd.DataFrame(train_pred)
train_df.head()


# In[ ]:


test_pred = {
    'cat':y_pred_test_cat,
    'grb':y_pred_test_grb,
    'lgb':y_pred_test_lgb
}
test_df = pd.DataFrame(test_pred)
test_df.head()


# In[ ]:


grb_m = GradientBoostingClassifier(random_state=7)
grb_m.fit(train_df,train[target])
y_pred = grb_m.predict(test_df)
test['Friends'] = y_pred
test[['ID','Friends']].to_csv("stacking_2_level.csv",index=False)


# Stacking Model:  
# *     Level 0 : LGB,GradientBoosting, CatBoosting  
# *     Level 1 : GradientBoosting

# **Thank you**  
# **Please Upvote if u have learnt something from this**
