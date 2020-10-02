#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('bmh')


# Parse the training set
# 1. Check the data types for each columns
# 2. Check the correlation of the data
# 3. Statistic (1. F-test for continous; 2. Chi-test for categorical)
# 4. Feature Selection

#  #### 1. Data Type of the data

# In[ ]:


## read data
train=pd.read_csv('../input/train.csv',index_col=0)
## check the data types for each column
print (train.columns)
print (np.unique(train.dtypes,return_counts=True))


# #### 2. Correlation of the data 
# #### 2.1 check the float type

# In[ ]:


train_float=train.select_dtypes(include=['float64'])
train_cat=train.select_dtypes(include=['int64'])

import seaborn as sns
plt.figure(figsize=(12,10))
sns.heatmap(train_float.corr(),linecolor='white',annot=True)


# #### 3 Statistics
# ##### 3.1 F-test for float

# In[ ]:


from sklearn.feature_selection import f_classif

anova=f_classif(train_float,train['target'])
anova_df=pd.DataFrame({'F':anova[0],'pval':anova[1]},index=train_float.columns)


# In[ ]:


anova_df['pval_reverse_log']=-1*np.log10(anova_df['pval'])
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
anova_df['pval_reverse_log'].plot(kind='bar')
plt.title('-Log(pval,10)')
plt.subplot(1,2,2)
anova_df['F'].plot(kind='bar')
plt.title('F')


# Conclusion of the F-test, most of the input are significant ( the bigger the value, the significant it is), ps_calc (01,02,03) related inputs are not so significant. 'ps_car_13' seems to be most important, the -log(pval,10) is not accurate becuase pval is 0, and can't be logged, use F score instead.

# In[ ]:


plt.figure(figsize=(12,8))

column_2_skip=['ps_calc_01','ps_calc_02','ps_calc_03','ps_car_14']

count=1
for cl_name in train_float.columns:
    if cl_name in column_2_skip: continue
    
    ax=plt.subplot(2,3,count)
    sns.kdeplot(train[train['target']==0][cl_name],label='0',legend=True,bw=0.05)
    sns.kdeplot(train[train['target']==1][cl_name],label='1',legend=True,bw=0.05)
    ax.set_xlabel(cl_name)
    count+=1


# #### 3.2 chisq test for categorical

# In[ ]:


from sklearn.feature_selection import chi2
chi_stat=chi2(train_cat.drop(['target'],axis=1).replace(-1,0),train['target'])


# In[ ]:


chi_stat_df=pd.DataFrame({'chi2':chi_stat[0],'pval':chi_stat[1]},index=train_cat.drop(['target'],axis=1).columns)
chi_stat_df['pval_reverse_log']=-1*np.log10(chi_stat_df['pval'])

plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
chi_stat_df['pval_reverse_log'].plot(kind='bar')
plt.title('chisq -log(pval,10)')

plt.subplot(2,1,2)
chi_stat_df['chi2'].plot(kind='bar')
plt.title('chi2')


# Conclusion of the chisq-test, ps_calc related inputs are not significant. 'ps_car_04' seems to be most important, the -log(pval,10) is not accurate becuase pval is 0, and can't be logged, use chi2 score instead.

# In[ ]:


def reframe_groupby(data_frame,col_name,target_name):
    col_table=data_frame.groupby([target_name,col_name]).size()
    col_norm=col_table.groupby(level=0).apply(lambda x: 100*x/float(x.sum()))
    col_df=pd.DataFrame({'target':col_norm.index.labels[0],col_name:col_norm.index.labels[1],
                         'freq':col_norm.values})
    return col_df

cat_col_selected=chi_stat_df[chi_stat_df['pval']<0.05].index
plt.figure(figsize=(16,16))
count=1
for cl_name in cat_col_selected:
    ax=plt.subplot(5,5,count)
    col_df=reframe_groupby(train,cl_name,'target')
    sns.barplot(x=cl_name,y='freq',hue='target',data=col_df)
    count+=1


# #### 4. Feature selection
# #### 4.1  Cross validation (with Logistic regression)

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score

## select the columns
float_column=[cl for cl in train_float.columns if cl not in column_2_skip]
selected_column=float_column+list(cat_col_selected)

## 5 fold CV
skf5=StratifiedKFold(n_splits=5,random_state=42)
for train_idx,test_idx in skf5.split(train[selected_column].values,train['target']):
    X_train=train[selected_column].values[train_idx]
    X_test=train[selected_column].values[test_idx]
    
    y_train=train['target'].values[train_idx]
    y_test=train['target'].values[test_idx]

    lm_fit=LogisticRegression(penalty='l2',C=0.1,class_weight='balanced')
    lm_fit.fit(X_train,y_train)
    pred_test=lm_fit.predict(X_test)
    print ('Average score:',average_precision_score(y_test,pred_test,average='weighted'))


# In[ ]:




