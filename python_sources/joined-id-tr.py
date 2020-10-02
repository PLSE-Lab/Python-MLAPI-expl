#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


from sklearn import preprocessing
import xgboost as xgb


# In[ ]:


train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')


# In[ ]:


train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape)
print(test.shape)


# In[ ]:


train.head()


# In[ ]:


train.shape
del train_transaction, train_identity, test_identity, test_transaction


# In[ ]:


pd.DataFrame(train.dtypes)


# In[ ]:


filter_col = [col for col in train if col.startswith(('addr', 'M', 'card', 'id_1', 'id_2', 'id_3'))]


# In[ ]:


filter_col


# In[ ]:


train[filter_col] = train[filter_col].astype('category')


# In[ ]:


# convert_dict = {
#     'DeviceType': "category",
#     'DeviceInfo': "category",
#     'ProductCD': "category",
#     'P_emaildomain':"category",
#     'R_emaildomain':"category",
#     'isFraud':"category",
#     'id_10': 'float',
#     'id_11': 'float'
# }
convert_dict = {
    'DeviceType': "category",
    'DeviceInfo': "category",
    'ProductCD': "category",
    'P_emaildomain':"category",
    'R_emaildomain':"category",
    'isFraud': "float",
    'id_10': 'float',
    'id_11': 'float'
}


# In[ ]:


tr = train.astype(convert_dict)


# In[ ]:


pd.DataFrame(tr.dtypes)


# Summary Statistics

# In[ ]:


# All Numeric Cols only
num_tr_ = tr._get_numeric_data()
num_tr_.shape


# In[ ]:


pd.set_option('display.float_format', lambda x: '%.3f' % x)
mean_summary = num_tr_.groupby(['isFraud']).mean().transpose()


# In[ ]:


mean_summary.head()


# In[ ]:


mean_summary['times_fraud'] = mean_summary[mean_summary.columns[1]]/mean_summary[mean_summary.columns[0]]


# In[ ]:


print(mean_summary.shape)
mean_summary.head()


# In[ ]:


filtr = mean_summary.loc[mean_summary['times_fraud'] > 2]
print(filtr.shape)
filtr.sort_values('times_fraud', ascending = False)


# In[ ]:


# convert is fraud to category
tr['isFraud'] = tr.isFraud.astype('category')
tr_fraud_ = tr.loc[tr['isFraud'] == 1]
tr_notfraud_ = tr.loc[tr['isFraud'] == 0]


# In[ ]:


from IPython.display import display_html
def display_side_by_side(*args):
    html_str=''
    for df in args:
        html_str+=df.to_html()
    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

    # tr_fraud_.describe(include=['category']).transpose()
    
display_side_by_side(tr_notfraud_.describe(include=['category']).transpose(), tr_fraud_.describe(include=['category']).transpose())


# In[ ]:


tr['id_10'].plot.box(vert = False)


# In[ ]:


tr.boxplot(by='isFraud', column = ['V147', 'V146'], vert = False)


# In[ ]:


tr.plot.scatter(x = 'V147', y = 'V146', c = 'isFraud')


# In[ ]:


short_tr = tr.head(1000)


# In[ ]:


import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 15, 10
plt.figure();
bp_ = tr.boxplot(by='isFraud', column = ['id_10', 'id_11'], vert = False)


# In[ ]:


tr['isFraud'].value_counts().plot.bar()


# In[ ]:


short_tr.plot.scatter(x='id_11', y='id_10', c = 'id_01');


# In[ ]:


short_tr['id_01'].plot.kde()


# In[ ]:


from pandas.plotting import parallel_coordinates


# In[ ]:


shrt_ = short_tr[['isFraud', 'TransactionDT', 'TransactionAmt', 'V147']]


# In[ ]:


shrt_.nunique()
shrt_.shape


# In[ ]:


fraud_shrt_ = shrt_[shrt_.isFraud == 1]
print(fraud_shrt_.shape)
# print(fraud_shrt_.isna().sum())
# print(fraud_shrt_.dropna().isna().sum())
fraud_shrt_ = fraud_shrt_.dropna()
print(fraud_shrt_.shape)
# print(fraud_shrt_.dropna().isna().sum())


# In[ ]:


fdad_long = train[['isFraud', 'TransactionDT', 'TransactionAmt', 'dist1']]
fdad_long_fraud_only = fdad_long[fdad_long.isFraud == 1]
print(fdad_long.shape)
print(fdad_long_fraud_only.shape)


# In[ ]:


print(fraud_shrt_.shape)
fraud_shrt_


# In[ ]:


plt.figure()
parallel_coordinates(shrt_, 'isFraud', color = 'gr')


# In[ ]:


plt.figure()
parallel_coordinates(fraud_shrt_, 'isFraud', color = 'r')


# In[ ]:


plt.figure()
parallel_coordinates(fdad_long_fraud_only, 'isFraud', color = 'r')


# In[ ]:


from pandas.plotting import radviz
radviz(shrt_, 'isFraud', color = 'gr')


# In[ ]:


radviz(fraud_shrt_, 'isFraud', color = 'r')


# In[ ]:


radviz(fdad_long_fraud_only, 'isFraud', color = 'r')

