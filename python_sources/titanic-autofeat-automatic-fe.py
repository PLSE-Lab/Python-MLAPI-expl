#!/usr/bin/env python
# coding: utf-8

# ![](https://upload.wikimedia.org/wikipedia/commons/6/6e/St%C3%B6wer_Titanic.jpg)

# <a class="anchor" id="0.0"></a>
# # Titanic : Comparison of automatic FE efficiency with Autofeat and tradicional approaches

# My kernels outline traditional approaches to FE:
# 
# 1) the consolidated result of EDA and FE optimization from many authors:
# * https://www.kaggle.com/vbmokin/titanic-top-3-cluster-analysis
# * https://www.kaggle.com/vbmokin/titanic-top-3-one-line-of-the-prediction-code
# * https://www.kaggle.com/vbmokin/three-lines-of-code-for-titanic-top-15
# * https://www.kaggle.com/vbmokin/three-lines-of-code-for-titanic-top-20
# 
# 2) the result of the formation of many features and their processing by 20 models (boosting, regression, simple neural networks, etc.):
# 
# * [Titanic (0.83253) - Comparison 20 popular models](https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models)
# 
# 
# The kernel [Autofeat Regression](https://www.kaggle.com/lachhebo/autofeat-feature-engineering/comments#652596) provides an example of using library Autofeat for automatic FE. Let us analyze whether this application will produce comparable results.

# <a class="anchor" id="0.1"></a>
# 
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [Download datasets](#2)
# 1. [Preparing to modeling with manual FE](#3)
#  -  [Manual FE](#3.1)
#  -  [Encoding categorical features](#3.2)
# 1. [Automatic FE](#4)
#  -  [For all 16 features](#4.1)
#  -  [For optimal 3 features](#4.2)
# 1. [Modeling](#5)
#  -  [The simple rule - very accurate model](#5.1)
#  -  [The LR and BGC for all 16 features](#5.2)
#  -  [The LR and BGC for optimal 3 features](#5.3)
# 1. [Comparison of 23 models](#6)
# 1. [Conclusion](#7)

# ## 1. Import libraries <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


get_ipython().system('pip install autofeat')


# In[ ]:


import pandas as pd
import numpy as np 

from sklearn.preprocessing import LabelEncoder
from autofeat import AutoFeatRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import explained_variance_score

# model tuning
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe, space_eval

import warnings
warnings.filterwarnings("ignore")

pd.set_option('max_columns', 100)


# ## 2. Download datasets <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


traindf = pd.read_csv('../input/titanic/train.csv').set_index('PassengerId')
testdf = pd.read_csv('../input/titanic/test.csv').set_index('PassengerId')
df = pd.concat([traindf, testdf], axis=0, sort=False)


# In[ ]:


df.head(5)


# ## 3. Preparing to modeling with manual FE <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# ### 3.1. Manual FE <a class="anchor" id="3.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


#Thanks to:
# https://www.kaggle.com/mauricef/titanic
# https://www.kaggle.com/vbmokin/titanic-top-3-one-line-of-the-prediction-code
#
df = pd.concat([traindf, testdf], axis=0, sort=False)
df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
df['Title'] = df.Name.str.split(',').str[1].str.split('.').str[0].str.strip()
df['IsWomanOrBoy'] = ((df.Title == 'Master') | (df.Sex == 'female'))
df['LastName'] = df.Name.str.split(',').str[0]
family = df.groupby(df.LastName).Survived
df['WomanOrBoyCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).count())
df['WomanOrBoyCount'] = df.mask(df.IsWomanOrBoy, df.WomanOrBoyCount - 1, axis=0)
df['FamilySurvivedCount'] = family.transform(lambda s: s[df.IsWomanOrBoy].fillna(0).sum())
df['FamilySurvivedCount'] = df.mask(df.IsWomanOrBoy, df.FamilySurvivedCount -                                     df.Survived.fillna(0), axis=0)
df['WomanOrBoySurvived'] = df.FamilySurvivedCount / df.WomanOrBoyCount.replace(0, np.nan)
df.WomanOrBoyCount = df.WomanOrBoyCount.replace(np.nan, 0)
df['Alone'] = (df.WomanOrBoyCount == 0)

#Thanks to https://www.kaggle.com/kpacocha/top-6-titanic-machine-learning-from-disaster
#"Title" improvement
df['Title'] = df['Title'].replace('Ms','Miss')
df['Title'] = df['Title'].replace('Mlle','Miss')
df['Title'] = df['Title'].replace('Mme','Mrs')
# Embarked
df['Embarked'] = df['Embarked'].fillna('S')
# Cabin, Deck
df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
df.loc[(df['Deck'] == 'T'), 'Deck'] = 'A'

# Thanks to https://www.kaggle.com/erinsweet/simpledetect
# Fare
med_fare = df.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
df['Fare'] = df['Fare'].fillna(med_fare)
#Age
df['Age'] = df.groupby(['Sex', 'Pclass', 'Title'])['Age'].apply(lambda x: x.fillna(x.median()))
# Family_Size
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

# Thanks to https://www.kaggle.com/vbmokin/titanic-top-3-cluster-analysis
cols_to_drop = ['Name','Ticket','Cabin']
df = df.drop(cols_to_drop, axis=1)

df.WomanOrBoySurvived = df.WomanOrBoySurvived.fillna(0)
df.WomanOrBoyCount = df.WomanOrBoyCount.fillna(0)
df.FamilySurvivedCount = df.FamilySurvivedCount.fillna(0)
df.Alone = df.Alone.fillna(0)
df.Alone = df.Alone*1


# In[ ]:


df.head(5)


# ### 3.2. Encoding categorical features <a class="anchor" id="3.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


target = df.Survived.loc[traindf.index]
df = df.drop(['Survived'], axis=1)
train, test = df.loc[traindf.index], df.loc[testdf.index]


# In[ ]:


# Determination categorical features
numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = train.columns.values.tolist()
for col in features:
    if train[col].dtype in numerics: continue
    categorical_columns.append(col)
categorical_columns


# In[ ]:


# Encoding categorical features
for col in categorical_columns:
    if col in train.columns:
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))


# In[ ]:


train.head()


# In[ ]:


test.head()


# ## 4. Automatic FE <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


model = AutoFeatRegression()
model


# ### 4.1. For all 16 features <a class="anchor" id="4.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


X_train_feature_creation = model.fit_transform(train.to_numpy(), target.to_numpy().flatten())
X_test_feature_creation = model.transform(test.to_numpy())
X_train_feature_creation.head()


# In[ ]:


# Number of new features
print('Number of new features -',X_train_feature_creation.shape[1] - train.shape[1])


# ### 4.2. For optimal 3 features <a class="anchor" id="4.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


df2 = pd.concat([df.WomanOrBoySurvived.fillna(0), df.Alone, df.Sex.replace({'male': 0, 'female': 1})], axis=1)
train2, test2 = df2.loc[traindf.index], df2.loc[testdf.index]


# In[ ]:


X_train_feature_creation2 = model.fit_transform(train2.to_numpy(), target.to_numpy().flatten())
X_test_feature_creation2 = model.transform(test2.to_numpy())
X_train_feature_creation2.head()


# In[ ]:


# Number of new features
print('Number of new features -',X_train_feature_creation2.shape[1] - train2.shape[1])


# ## 5. Modeling and comparison <a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# ### 5.1. The simple rule - very accurate model <a class="anchor" id="5.1"></a>
# 
# [Back to Table of Contents](#0.1)

# From my kernel: [Titanic Top 3% : one line of the prediction code](https://www.kaggle.com/vbmokin/titanic-top-3-one-line-of-the-prediction-code)

# In[ ]:


test_x = df2.loc[testdf.index]

# The one line of the code for prediction : LB = 0.83253 (Titanic Top 3%) 
test_x['Survived'] = (((test_x.WomanOrBoySurvived <= 0.238) & (test_x.Sex > 0.5) & (test_x.Alone > 0.5)) |           ((test_x.WomanOrBoySurvived > 0.238) &            ~((test_x.WomanOrBoySurvived > 0.55) & (test_x.WomanOrBoySurvived <= 0.633))))

# Saving the result
pd.DataFrame({'Survived': test_x['Survived'].astype(int)},              index=testdf.index).reset_index().to_csv('survived.csv', index=False)


# In[ ]:


LB_simple_rule = 0.83253


# ### 5.2. The LR and BGC for all 16 features <a class="anchor" id="5.2"></a>
# 
# [Back to Table of Contents](#0.1)

# **Liner Regression with Autofeat** from kernel [Autofeat Regression](https://www.kaggle.com/lachhebo/autofeat-feature-engineering/comments#652596)

# In[ ]:


# Linear Regression without Autofeat
model_LR = LinearRegression().fit(train,target.to_numpy().flatten())

# Linear Regression with Autofeat
model_Autofeat = LinearRegression().fit(X_train_feature_creation, target.to_numpy().flatten())


# In[ ]:


test['Survived_LR'] = np.clip(model_LR.predict(test),0,1)
test['Survived_AF'] = np.clip(model_Autofeat.predict(X_test_feature_creation),0,1)


# **Gradient Boosting Classifier with HyperOpt tuning** - from my kernel: [Titanic (0.83253) - Comparison 20 popular models](https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models) 

# In[ ]:


def hyperopt_gb_score(params):
    clf = GradientBoostingClassifier(**params)
    current_score = cross_val_score(clf, X_train_feature_creation, target, cv=10).mean()
    print(current_score, params)
    return current_score 
 
space_gb = {
            'n_estimators': hp.choice('n_estimators', range(100, 1000)),
            'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int))            
        }
 
best = fmin(fn=hyperopt_gb_score, space=space_gb, algo=tpe.suggest, max_evals=10)
print('best:')
print(best)


# In[ ]:


params = space_eval(space_gb, best)
params


# In[ ]:


# Gradient Boosting Classifier

gradient_boosting = GradientBoostingClassifier(**params)
gradient_boosting.fit(X_train_feature_creation, target)
Y_pred = gradient_boosting.predict(X_test_feature_creation).astype(int)
gradient_boosting.score(X_train_feature_creation, target)
acc_gradient_boosting = round(gradient_boosting.score(X_train_feature_creation, target) * 100, 2)
acc_gradient_boosting


# In[ ]:


# Saving the results
pd.DataFrame({'Survived': test['Survived_LR'].astype(int)},              index=testdf.index).reset_index().to_csv('survived_LR16.csv', index=False)
pd.DataFrame({'Survived': test['Survived_AF'].astype(int)},              index=testdf.index).reset_index().to_csv('survived_Autofeat16.csv', index=False)
pd.DataFrame({'Survived': Y_pred},              index=testdf.index).reset_index().to_csv('survived_GBC16.csv', index=False)


# In[ ]:


# After download solutions in Kaggle competition:
LB_LR16 = 0.69377
LB_Autofeat16 = 0.67942
LB_GBC16 = 0.82296


# ### 5.3. The LR and BGC for optimal 3 features <a class="anchor" id="5.3"></a>
# 
# [Back to Table of Contents](#0.1)

# **Linear Regression**

# In[ ]:


# Linear Regression without Autofeat
model_LR2 = LinearRegression().fit(train2,target.to_numpy().flatten())

# Linear Regression with Autofeat
model_Autofeat2 = LinearRegression().fit(X_train_feature_creation2, target.to_numpy().flatten())


# In[ ]:


test2['Survived_LR'] = np.clip(model_LR2.predict(test2),0,1)
test2['Survived_AF'] = np.clip(model_Autofeat2.predict(X_test_feature_creation2),0,1)


# **Gradient Boosting Classifier with HyperOpt tuning** 

# In[ ]:


def hyperopt_gb_score(params):
    clf = GradientBoostingClassifier(**params)
    current_score = cross_val_score(clf, X_train_feature_creation2, target, cv=10).mean()
    print(current_score, params)
    return current_score 
 
space_gb = {
            'n_estimators': hp.choice('n_estimators', range(100, 1000)),
            'max_depth': hp.choice('max_depth', np.arange(2, 10, dtype=int))            
        }
 
best = fmin(fn=hyperopt_gb_score, space=space_gb, algo=tpe.suggest, max_evals=10)
print('best:')
print(best)


# In[ ]:


params2 = space_eval(space_gb, best)
params2


# In[ ]:


# Gradient Boosting Classifier

gradient_boosting2 = GradientBoostingClassifier(**params2)
gradient_boosting2.fit(X_train_feature_creation2, target)
Y_pred2 = gradient_boosting2.predict(X_test_feature_creation2).astype(int)
gradient_boosting2.score(X_train_feature_creation2, target)
acc_gradient_boosting2 = round(gradient_boosting2.score(X_train_feature_creation2, target) * 100, 2)
acc_gradient_boosting2


# In[ ]:


# Saving the results
pd.DataFrame({'Survived': test2['Survived_LR'].astype(int)},              index=testdf.index).reset_index().to_csv('survived_LR3.csv', index=False)
pd.DataFrame({'Survived': test2['Survived_AF'].astype(int)},              index=testdf.index).reset_index().to_csv('survived_Autofeat3.csv', index=False)
pd.DataFrame({'Survived': Y_pred2},              index=testdf.index).reset_index().to_csv('survived_GBC3.csv', index=False)


# In[ ]:


# After download solutions in Kaggle competition:
LB_LR3 = 0.67942
LB_Autofeat3 = 0.69377
LB_GBC3 = 0.83253


# ## 6. Comparison of 23 models <a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# **Comparison of 4 models, including 3 new models**

# In[ ]:


models = pd.DataFrame({
    'Model': ['Simple rule','Linear Regression without Autofeat', 'Linear Regression with Autofeat',
              'GradientBoostingClassifier with Autofeat'],
    
    'LB_for_16_features': [LB_simple_rule, LB_LR16, LB_Autofeat16, LB_GBC16],

    'LB_for_3opt_features': [LB_simple_rule, LB_LR3, LB_Autofeat3, LB_GBC3]})


# In[ ]:


models.sort_values(by=['LB_for_3opt_features', 'LB_for_16_features'], ascending=False)


# In[ ]:


models.sort_values(by=['LB_for_16_features', 'LB_for_3opt_features'], ascending=False)


# **Comparison with 20 other models**

# LB of the tradicional approaches from my kernels: [Titanic (0.83253) - Comparison 20 popular models](https://www.kaggle.com/vbmokin/titanic-0-83253-comparison-20-popular-models)
# 
# ![image.png](attachment:image.png)

# As you can see, the **Gradient Boosting Classifier** equally well extracts features without Autofeat library.

# ## 7. Conclusion <a class="anchor" id="7"></a>
# 
# [Back to Table of Contents](#0.1)

# The analysis makes the following conclusions:
# 
# - Autofeat methods should only be used to find new dependencies and features, but **this technology does not replace traditional FE methods**
# 
# - The **Gradient Boosting Classifier equally well extracts features without Autofeat library**, essentially negating the its benefits for this model (and apparently for other advanced decision tree models with hyperparameter optimization)
# 
# - the **Linear Regression model** has low accuracy and **Autofeat methods allow it to be slightly improved, but only with the condition of preliminary FE**, otherwise its application can also impair the accuracy of prediction
# 
# - The **Autofeat methods study should be repeated** if you are trying to tuning other conversion features and make the Autofeat classifier's hyperparameter optimization
# 
# - It is advisable to try to **apply Autofeat methods to the results of other methods of manual or automatic feature extraction**, such as Featurestools

# I hope you find this kernel useful and enjoyable.

# Your comments and feedback are most welcome.

# [Go to Top](#0.0)
