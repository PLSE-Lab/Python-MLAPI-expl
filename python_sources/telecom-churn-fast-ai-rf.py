#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This is my first Kaggle kernel which will help in building a predictive model which will identify the churn in Telecom industry. I have recently started with Fast ai ML course hence I will extensively make use of the code and my understandings from the course.  
# I was not sure how to install Fast ai library in Kaggle so I have copy pasted the necessatry code from the Fast ai library since the functions are independent and dont have any dependencies. Full Fast ai ML code can be found at [github](https://github.com/fastai/fastai)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
import os
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from pandas.api.types import is_string_dtype, is_numeric_dtype
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df_raw = pd.read_csv(r"../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df_raw.head()


# In[ ]:


df_raw.shape


# In[ ]:


df_raw.info()


# Total Charges columns has float values but the data type of the column is object so lets try to  convert it to float column

# In[ ]:


df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors='coerce')


# In[ ]:


df_raw.head()


# In[ ]:


sns.countplot(x='Churn', data=df_raw)


# Let's see if we have null values in any of our columns

# In[ ]:


df_raw.isnull().sum().sort_index()/len(df_raw)


# In[ ]:


#Code taken from fast ai library. Code can be found at github.
#https://github.com/fastai/fastai/blob/master/old/fastai/structured.py
def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c): df[n] = c.astype('category').cat.as_ordered()


# The categorical variables are currently stored as strings, which is inefficient, and doesn't provide the numeric coding required for a random forest. Therefore we call train_cats to convert strings to pandas categories.

# In[ ]:


train_cats(df_raw)


# In[ ]:


df_raw.PhoneService.cat.categories


# In[ ]:


df_raw.PaperlessBilling.cat.categories


# In[ ]:


#The below functions are taken from fast ai library, code for which can be found at github.
#https://github.com/fastai/fastai/blob/master/old/fastai/structured.py
def fix_missing(df, col, name, na_dict):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum() or (name in na_dict):
            df[name+'_na'] = pd.isnull(col)
            filler = na_dict[name] if name in na_dict else col.median()
            df[name] = col.fillna(filler)
            na_dict[name] = filler
    return na_dict

def numericalize(df, col, name, max_n_cat):
    if not is_numeric_dtype(col) and ( max_n_cat is None or len(col.cat.categories)>max_n_cat):
        df[name] = col.cat.codes+1
        
def proc_df(df, y_fld=None, skip_flds=None, ignore_flds=None, do_scale=False, na_dict=None,
            preproc_fn=None, max_n_cat=None, subset=None, mapper=None):
    if not ignore_flds: ignore_flds=[]
    if not skip_flds: skip_flds=[]
    if subset: df = get_sample(df,subset)
    else: df = df.copy()
    ignored_flds = df.loc[:, ignore_flds]
    df.drop(ignore_flds, axis=1, inplace=True)
    if preproc_fn: preproc_fn(df)
    if y_fld is None: y = None
    else:
        if not is_numeric_dtype(df[y_fld]): df[y_fld] = df[y_fld].cat.codes
        y = df[y_fld].values
        skip_flds += [y_fld]
    df.drop(skip_flds, axis=1, inplace=True)

    if na_dict is None: na_dict = {}
    else: na_dict = na_dict.copy()
    na_dict_initial = na_dict.copy()
    for n,c in df.items(): na_dict = fix_missing(df, c, n, na_dict)
    if len(na_dict_initial.keys()) > 0:
        df.drop([a + '_na' for a in list(set(na_dict.keys()) - set(na_dict_initial.keys()))], axis=1, inplace=True)
    if do_scale: mapper = scale_vars(df, mapper)
    for n,c in df.items(): numericalize(df, c, n, max_n_cat)
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([ignored_flds, df], axis=1)
    res = [df, y, na_dict]
    if do_scale: res = res + [mapper]
    return res


# In[ ]:


df, y, nas = proc_df(df_raw, 'Churn', skip_flds=['customerID'], max_n_cat=8)


# In[ ]:


df.head().T


# In[ ]:


def split_vals(a,n): return a[:n], a[n:]


# Split the data into Train and Test set. I will set aside 10% of the data as a test set.

# In[ ]:


n_valid = int(7043 * 0.1)
n_trn = len(df)-n_valid
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)


# In[ ]:


from sklearn import metrics


# In[ ]:


def print_score(m):
    res = [metrics.accuracy_score(m.predict(X_train), y_train), 
           metrics.accuracy_score(m.predict(X_valid), y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


m = RandomForestClassifier(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# Jeremy Howard mentions that once you have trained a Random Forest, the next thing you should do is Feature Importance. This will help you in identifying the features that played a prominent role in the predictive model.

# In[ ]:


#The below functions are taken from fast ai library, code for which can be found at github.
#https://github.com/fastai/fastai/blob/master/old/fastai/structured.py
def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)


# In[ ]:


fi = rf_feat_importance(m, df) 
fi[:10]


# In[ ]:


def plot_fi(fi): 
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[ ]:


plot_fi(fi[:30])


# As you can see the top most features for the Random Forest are Total Charges, Monthly Charges and Tenure. Let's try to analyze these features in detail.

# In[ ]:


plt.figure(figsize=(9, 4))
plt.title("KDE for Total Charges")
ax0 = sns.kdeplot(df_raw[df_raw['Churn'] == 'No']["TotalCharges"].dropna(), color= 'green', label= 'Churn: No')
ax1 = sns.kdeplot(df_raw[df_raw['Churn'] == 'Yes']["TotalCharges"].dropna(), color= 'red', label= 'Churn: Yes')


# In[ ]:


plt.figure(figsize=(9, 4))
plt.title("KDE for Monthly Charges")
ax0 = sns.kdeplot(df_raw[df_raw['Churn'] == 'No']["MonthlyCharges"].dropna(), color= 'green', label= 'Churn: No')
ax1 = sns.kdeplot(df_raw[df_raw['Churn'] == 'Yes']["MonthlyCharges"].dropna(), color= 'red', label= 'Churn: Yes')


# In[ ]:


plt.figure(figsize=(9, 4))
plt.title("KDE for Tenure")
ax0 = sns.kdeplot(df_raw[df_raw['Churn'] == 'No']["tenure"].dropna(), color= 'green', label= 'Churn: No')
ax1 = sns.kdeplot(df_raw[df_raw['Churn'] == 'Yes']["tenure"].dropna(), color= 'red', label= 'Churn: Yes')


# As you can see from the above plots, for higher monthly charges the churn is more. Also higher tenured customers are less likely to leave the telco.

# Let's try to tune some hyperparameters and see if we can increase the accuracy of the model

# In[ ]:


#Increase the number of estimator's i.e. the number of decision trees to to 40.
m = RandomForestClassifier(n_estimators=40, n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


fi


# Let's only keep the features in the model that have a feature importance of more than 0.01

# In[ ]:


to_keep = fi[fi.imp>0.01].cols
len(to_keep)


# In[ ]:


df_keep = df[to_keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)


# In[ ]:


m = RandomForestClassifier(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


fi = rf_feat_importance(m, df_keep)
plot_fi(fi);


# Let's bump the number of estimators to 100 and see if our model performance improves

# In[ ]:


m = RandomForestClassifier(n_estimators=100, n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:




