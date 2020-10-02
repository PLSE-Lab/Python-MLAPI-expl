#!/usr/bin/env python
# coding: utf-8

# # Identifying Breast Cancer using Logistic Regression with Weight of Evidence

# ![Weight Of Evidence](https://miro.medium.com/max/768/1*6Aw782wiyiFtzvK7EOY8CA.png)
# ![Information Value](https://miro.medium.com/max/1400/1*xWA7a2KsTQOhaQ9MZFJJeQ.png)

# * [Introduction](#Introduction)
# * [Import Packages](#Import-Packages)
# * [Load in Data](#Load-in-Data)
# * [EDA](#EDA)
# * [WOE and IV](#WOE-and-IV)
#     * [Weight of Evidence](#Weight-of-Evidence)
#     * [Information Value](#Information-Value)
# * [Encoding Data](#Encoding-Data)
# * [Training Model](#Training-Model)
# * [Testing and Checking Metrics](#Testing-and-Checking-Metrics)
# * [Conclusion](#Conclusion)

# ## Introduction

# The purpose of this notebook is to show how to use Weight of Evidence (WOE) and Information Value (IV) to predict whether breast tissues are benign or malignant using the Breast Cancer Wisconsin (Diagnostic) Data Set.  
# 
# Since this is a binary classification problem, Logisitic Regression will be used as the classifier. 
# 
# All information included in this notebook on WOE and IV can be found here: [Weight of evidence and Information Value using Python](https://medium.com/@sundarstyles89/weight-of-evidence-and-information-value-using-python-6f05072e83eb)
# 
# <a href="#top">Back to Top</a>

# ## Import Packages

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
import string

import warnings

import gc; gc.enable()

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="darkgrid", color_codes=True, font_scale=1.3)
warnings.filterwarnings("ignore")


# <a href="#top">Back to Top</a>

# ## Load in Data

# In[ ]:


#read in data
df = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv", index_col = 0)
df.info()


# <a href="#top">Back to Top</a>

# ## EDA

# In[ ]:


df.reset_index(inplace=True)
df.head()


# In[ ]:


df.shape


# In[ ]:


df.isna().sum()


# In[ ]:


#drop last column with no values
df.drop("Unnamed: 32", axis = 1, inplace = True)


# In[ ]:


#Setting our target and converting values to numeric
df['target'] = df['diagnosis'].apply(lambda x : 1 if x == 'M' else 0)
df.drop('diagnosis', axis = 1, inplace = True)
df.head()


# In[ ]:


# visualize distribution of classes 
plt.figure(figsize=(10, 8))
sns.countplot(df['target'], palette='RdBu')

# count number of obvs in each class
benign, malignant = df['target'].value_counts()
print('Number of cells labeled Benign: ', benign)
print('Number of cells labeled Malignant : ', malignant)
print('')
print('% of cells labeled Benign', round(benign / len(df) * 100, 2), '%')
print('% of cells labeled Malignant', round(malignant / len(df) * 100, 2), '%')


# In[ ]:


target = 'target'
_id = "id"
used_cols = [col for col in df.columns.tolist() if col not in [target, _id]]
X = df[used_cols].copy()
y = df[target].copy()
del df
gc.collect()


# In[ ]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)


# <a href="#top">Back to Top</a>

# ## WOE and IV
# ref : [Weight of evidence and Information Value using Python](https://medium.com/@sundarstyles89/weight-of-evidence-and-information-value-using-python-6f05072e83eb)

# In[ ]:


max_bin = 20
force_bin = 3

# define a binning function
def mono_bin(Y, X, n = max_bin):
    
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)})
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1 
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = force_bin         
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1]-(bins[1]/2)
        d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins),include_lowest=True)}) 
        d2 = d1.groupby('Bucket', as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3=d3.reset_index(drop=True)
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]       
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    
    return(d3)

def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def data_vars(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i            
                count = count + 1
                
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv)


# <a href="#top">Back to Top</a>

# ### Weight of Evidence

# In[ ]:


final_iv, df_IV = data_vars(X_train,y_train)


# In[ ]:


final_iv.sort_values("IV").head()


# <a href="#top">Back to Top</a>

# ### Information Value

# From the reference material an IV less than 0.02 is characterized as a useless predictor

# In[ ]:


df_IV.loc[df_IV['IV'] < .02].sort_values("IV")


# In[ ]:


useless_ = df_IV.loc[df_IV['IV'] < .02]['VAR_NAME'].tolist()


# An IV between 0.02 to 0.1 is characterized as a weak predictor

# In[ ]:


df_IV.loc[(df_IV['IV']>.02) & (df_IV['IV']<.1)].sort_values("IV")


# An IV between 0.1 to 0.3 is characterized as a medium predictor

# In[ ]:


df_IV.loc[(df_IV['IV']>.1) & (df_IV['IV']<.3)].sort_values("IV")


# An IV between 0.3 to 0.5 is characterized as a strong predictor

# In[ ]:


df_IV.loc[(df_IV['IV']>.3) & (df_IV['IV']<.5)].sort_values("IV")


# An IV is greater than 0.5 is characterized as suspicious or too good to be true.

# In[ ]:


df_IV.loc[df_IV['IV']>.5].sort_values("IV")


# <a href="#top">Back to Top</a>

# ## Encoding Data

# In[ ]:


#function that uses our information determined form WOE and applies it to our data
def transform_(df, transform_vars_list, final_iv):
    for var in transform_vars_list:
        small_df = final_iv[final_iv['VAR_NAME'] == var]
        transform_dict = dict(zip(small_df.MAX_VALUE,small_df.WOE))
        replace_cmd = ''
        replace_cmd1 = ''
        for i in sorted(transform_dict.items()):
            replace_cmd = replace_cmd + str(i[1]) + str(' if x <= ') + str(i[0]) + ' else '
            replace_cmd1 = replace_cmd1 + str(i[1]) + str(' if x == "') + str(i[0]) + '" else '
        replace_cmd = replace_cmd + '0'
        replace_cmd1 = replace_cmd1 + '0'
        if replace_cmd != '0':
            try:
                df[transform_prefix + var] = df[var].apply(lambda x: eval(replace_cmd))
            except:
                df[transform_prefix + var] = df[var].apply(lambda x: eval(replace_cmd1))
    return df


# In[ ]:


#dropping useless predictors
keep_cols = [col for col in used_cols if col not in useless_]


# In[ ]:


transform_prefix = ''
X_train = transform_(X_train, keep_cols, final_iv)


# <a href="#top">Back to Top</a>

# ## Training Model

# In[ ]:


logreg = LogisticRegression(fit_intercept = False, class_weight = 'balanced', C = 1e15)
model_log = logreg.fit(X_train, y_train)


# <a href="#top">Back to Top</a>

# In[ ]:


X_test = transform_(X_test, keep_cols, final_iv)


# In[ ]:


y_pred = model_log.predict(X_test)


# In[ ]:


print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred)) 
print("The accuracy score is" + " "+ str(accuracy_score(y_test, y_pred)))


# <a href="#top">Back to Top</a>

# ## Conclusion

# We can see from these metrics that using WOE and IV with Logistic Regression can be a strong technique to use in solving binary classification problems.

# More information on Weight of Evidence and Information Value used for this can be found in the following Notebooks:
# * [Weight of Evidence(WOE) & Information Value(IV)](https://www.kaggle.com/pavansanagapati/weight-of-evidence-woe-information-value-iv)
# * [IV + WoE Starter for Python](https://www.kaggle.com/puremath86/iv-woe-starter-for-python)

# Let me know if you have any feedback. 

# <a href="#top">Back to Top</a>
