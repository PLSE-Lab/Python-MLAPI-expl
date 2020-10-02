#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing


# In[ ]:


# load data
df = pd.read_csv('/kaggle/input/individual-company-sales-data/sales_data.csv')
df.head()


# In[ ]:


# shape of dataframe
df.shape


# In[ ]:


# dataframe dtypes for each feature
df.dtypes


# #### Missing values analysis

# 1. Replace *'Unknown'* categories with *null* values

# In[ ]:


for cat in df.columns:
    print(cat, df[cat].unique())


# Seems like *gender*, *age* and *child* features have this kind of category

# In[ ]:


df['gender'] = df.gender.replace('U', np.NaN)
df['age'] = df.age.replace('1_Unk', np.NaN)
df['child'] = df.child.replace('U', np.NaN)
df['child'] = df.child.replace('0', np.NaN)


# In[ ]:


df.isnull().sum()


# In[ ]:


# relative
df.isnull().sum() / df.shape[0] * 100


# #### Handling missing values

# In[ ]:


def category_stackedbar(df, category):
    '''Returns stacked bar plot'''
    return pd.DataFrame(
        df.groupby(category).count()['flag'] / df.groupby(category).count()['flag'].sum() * 100).rename(columns={"flag": "%"}).T.plot(
            kind='bar', 
            stacked=True
    );


# > *house_owner*

# In[ ]:


category_stackedbar(df, 'house_owner');


# Since most of house owners are owners, I can fill *null* values with *Owner* category

# In[ ]:


df['house_owner'] = df['house_owner'].fillna(df.mode()['house_owner'][0])


# > *age*

# In[ ]:


category_stackedbar(df, 'age');


# There is no dominant category so I prefer to delete rows with *null* values in *age*

# In[ ]:


df = df.dropna(subset=['age'])


# > child

# In[ ]:


category_stackedbar(df, 'child');


# In[ ]:


# percentage of null values in *child*
(df.isnull().sum() / df.shape[0] * 100)['child']


# Same case as *age* feature, but in this case I am going to drop the column to avoid removing too many observations

# In[ ]:


df = df.drop('child', axis=1)


# > marriage

# In[ ]:


category_stackedbar(df, 'marriage');


# More than 80% are Married so I decided to fill *null* values with the mode

# In[ ]:


df['marriage'] = df['marriage'].fillna(df.mode()['marriage'][0])


# > education and gender

# *null* values from *education* and *gender* represent less than 2% of the total rows, so we can just drop them

# In[ ]:


df = df.dropna(subset=['gender', 'education'])


# In[ ]:


# checking data cleaning
df.isnull().sum()


# ### Feature Engineering

# In[ ]:


df.dtypes


# Transforming *flag* and *online* features to binary integer

# In[ ]:


df['flag'] = df['flag'].apply(lambda value: 1 if value == 'Y' else 0)
df['online'] = df['online'].apply(lambda value: 1 if value == 'Y' else 0)


# In[ ]:


df.dtypes


# From the categorical features I'm going to transform the columns *education*, *age*, *mortgage* and *fam_income* using label encoding because they have a hierarchy. For the other categories I'll treat them as dummy variables.

# In[ ]:


# explore categories of features with hierarchy
for cat in ['education', 'age', 'mortgage', 'fam_income']:
    print(cat, df[cat].unique())


# In[ ]:


# education to integer
df['education'] = df['education'].apply(lambda value: int(value[0]) + 1)


# In[ ]:


# age to integer
df['age'] = df['age'].apply(lambda value: int(value[0]) - 1)


# In[ ]:


# mortgage to integer
df['mortgage'] = df['mortgage'].apply(lambda value: int(value[0]))


# In[ ]:


#fam_income label dictionary
dict_fam_income_label = {}
for i, char in enumerate(sorted(df['fam_income'].unique().tolist())):
    dict_fam_income_label[char] = i + 1


# In[ ]:


df['fam_income'] = df['fam_income'].apply(lambda value: dict_fam_income_label[value])


# In[ ]:


dummy_features = ['gender', 'customer_psy', 'occupation', 'house_owner', 'region', 'marriage']


# In[ ]:


# explore categories of dummy features
for cat in dummy_features:
    print(cat, df[cat].unique())


# In[ ]:


def apply_dummy(df, cat, drop_first=True):
    return pd.concat([df, pd.get_dummies(df[cat], prefix=cat, drop_first=drop_first)], axis=1).drop(cat, axis=1)


# In[ ]:


for cat in dummy_features:
    df = apply_dummy(df, cat)


# In[ ]:


# dataframe with just numbers
df.head()


# ## Xgboost Model

# In[ ]:


from xgboost import XGBClassifier


# In[ ]:


X = df.drop('flag', axis=1)
y = df['flag']


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[ ]:


# scale to handle imbalanced dataset
scale = y_train[y_train == 0].count() / y_train[y_train == 1].count()


# In[ ]:


xgbmodel = XGBClassifier(max_depth=3, learning_rate=0.01, n_estimators=1000, scale_pos_weight=scale)


# In[ ]:


xgbmodel.fit(X_train, y_train)


# In[ ]:


y_pred_test = xgbmodel.predict(X_test)
y_pred_train = xgbmodel.predict(X_train)


# In[ ]:


print('Train')
print('Precision: {:.2f}% \tRecall: {:.2f}% \t\tF1 Score: {:.2f}%'.format(precision_score(y_train, y_pred_train)*100, recall_score(y_train, y_pred_train)*100, f1_score(y_train, y_pred_train)*100))


# In[ ]:


print('Test')
print('Precision: {:.2f}% \tRecall: {:.2f}% \t\tF1 Score: {:.2f}%'.format(precision_score(y_test, y_pred_test)*100, recall_score(y_test, y_pred_test)*100, f1_score(y_test, y_pred_test)*100))


# ### Explaining the output of Xgboost

# In[ ]:


import shap


# In[ ]:


# load JS visualization code to notebook
shap.initjs()


# In[ ]:


# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(xgbmodel)
shap_values = explainer.shap_values(X_train)


# In[ ]:


# summarize the effects of all the features
shap.summary_plot(shap_values, X_train)


# In[ ]:


shap.summary_plot(shap_values, X, plot_type="bar")


# In[ ]:


shap.dependence_plot("age", shap_values, X_train)


# In[ ]:


shap.dependence_plot("education", shap_values, X_train)


# ### Target Audience (most likely to buy the product)
# * Male
# * Age between 35 and 55
# * Experience shopping online
# * High house evaluation
# * High education (Bach - Grad)
# * Professional occupation
# * High Mortgage

# In[ ]:




