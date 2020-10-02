#!/usr/bin/env python
# coding: utf-8

# <h1><center><font size="6">IEEE FRAUD DETECTION</font></center></h1>
# 
# 
# 
# 
# <img src="https://gdprinformer.com/wp-content/uploads/2017/09/fraud-laptop.jpg" width="800"></img>
# 
# 
# 
# <br>

# # <a id='0'>Content</a>
# 
# - <a href='#1'>Importing Libraries and Modules</a>  
# - <a href='#2'>Distribution of features across Train and Test</a>
#  - <a href='#21'>Card Features</a>   
#   - <a href='#22'>Addr,Dist and EmailDomain</a>  
#   - <a href='#23'>C Features</a> 
#   - <a href='#24'>D Features</a> 
#   - <a href='#25'>M features</a> 
#   - <a href='#26'>V Features</a>   
#   - <a href='#27'>ID features</a> 
# - <a href='#3'>Dropping Columns</a>   
# - <a href='#4'>Reducing Memory Size</a>    
# - <a href='#5'>Feature Engineering</a>     
# - <a href='#6'>Model Development</a>
# - <a href='#7'>Feature Importance</a>
# 
# 
# **Please upvote if you like the kernel . Happy Kaggling and all the best for the competition**

# # <a id='1'>Importing Libraries and Modules</a> 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from hyperopt import tpe,hp,Trials
from hyperopt.fmin import fmin
import gc
import os
print(os.listdir("../input"))
seed=5
# Any results you write to the current directory are saved as output.


# In[ ]:


train_identity=pd.read_csv('../input/train_identity.csv')
test_identity=pd.read_csv('../input/test_identity.csv')
train_transaction=pd.read_csv('../input/train_transaction.csv')
test_transaction=pd.read_csv('../input/test_transaction.csv')


# In[ ]:


train=pd.merge(train_transaction,train_identity,how='left',on='TransactionID')
test=pd.merge(test_transaction,test_identity,how='left',on='TransactionID')


# In[ ]:


del train_identity,test_identity,train_transaction,test_transaction


# In[ ]:


train.info()


# In[ ]:


# most_null_values=[col for col in train.columns if (train[col].isna().sum()/train.shape[0])>0.9]
# len(most_null_values)


# In[ ]:


# dominant_unique_values=[col for col in train.columns if (train[col].value_counts().values[0]/train.shape[0])>0.9]
# len(dominant_unique_values)


# In[ ]:


# dominant_unique_values.remove('isFraud')


# In[ ]:


# cols_to_drop=list(set(most_null_values+dominant_unique_values+['TransactionID','TransactionDT']))
# train=train.drop(cols_to_drop,axis=1)
# test=test.drop(cols_to_drop,axis=1)


# # <a id='2'>Distribution of features across Train and Test</a> 

# In[ ]:


train.columns[:50]


# In[ ]:


sns.countplot(train['isFraud'])


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,5))
sns.distplot(train[train['isFraud']==0]['TransactionAmt'],ax=ax[0],hist=False,label='NonFraud')
sns.distplot(train[train['isFraud']==1]['TransactionAmt'],ax=ax[0],hist=False,label='Fraud')
ax[0].set_title('Fraud and NonFraud TransactionAmt Distribution')

sns.distplot(np.log(test['TransactionAmt']),ax=ax[1],hist=False,label='Test')
sns.distplot(np.log(train['TransactionAmt']),ax=ax[1],hist=False,label='Train')
ax[1].set_title('Test and Train TransationAmt Distribution')


# Remember to take log of Transaction Amount. 

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(10,5))
sns.distplot(test['TransactionAmt'],hist=False,label='Test',ax=ax[0],color='orange')
ax[0].set_title('Distribution of TransactionAmt in Test')
sns.distplot(train['TransactionAmt'],hist=False,label='Train',ax=ax[1])
ax[1].set_title('Distribution of TransactionAmt in Train')
plt.tight_layout()


# Remove the Transaction amout outliers or take log.

# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(10,5))
sns.countplot(train['ProductCD'],ax=ax[0])
ax[0].set_title('Train ProductCD Distribution')

sns.countplot(test['ProductCD'],ax=ax[1],label='Test')
ax[1].set_title('Test ProductCD Distribution')

plt.tight_layout()


# ## <a id='21'>Distribution of  Card features across Train and Test</a> 

# In[ ]:


card_cols=['card1','card2','card3','card4','card5','card6']
for c in card_cols:
    print(f'Number of unique variables in {c}: ',train[c].nunique())


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,5))
sns.distplot(train[train['isFraud']==0]['card1'],ax=ax[0],hist=False,label='NonFraud')
sns.distplot(train[train['isFraud']==1]['card1'],ax=ax[0],hist=False,label='Fraud')
ax[0].set_title('Fraud and NonFraud Card1 Distribution')

sns.distplot(test['card1'],ax=ax[1],hist=False,label='Test')
sns.distplot(train['card1'],ax=ax[1],hist=False,label='Train')
ax[1].set_title('Test and Train Card1 Distribution')


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,5))
sns.distplot(train[train['isFraud']==0]['card2'],ax=ax[0],hist=False,label='NonFraud')
sns.distplot(train[train['isFraud']==1]['card2'],ax=ax[0],hist=False,label='Fraud')
ax[0].set_title('Fraud and NonFraud Card1 Distribution')

sns.distplot(test['card2'],ax=ax[1],hist=False,label='Test')
sns.distplot(train['card2'],ax=ax[1],hist=False,label='Train')
ax[1].set_title('Test and Train Card1 Distribution')


# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(15,8))
sns.distplot(train[train['isFraud']==0]['card3'],ax=ax[0,0],hist=False,label='NonFraud')
sns.distplot(train[train['isFraud']==1]['card3'],ax=ax[0,0],hist=False,label='Fraud')
ax[0,0].set_title('Fraud and NonFraud Card3 Distribution')

sns.distplot(test['card3'],ax=ax[0,1],hist=False,label='Test')
sns.distplot(train['card3'],ax=ax[0,1],hist=False,label='Train')
ax[0,1].set_title('Test and Train Card3 Distribution')

sns.distplot(train[train['isFraud']==0]['card5'],ax=ax[1,0],hist=False,label='NonFraud')
sns.distplot(train[train['isFraud']==1]['card5'],ax=ax[1,0],hist=False,label='Fraud')
ax[1,0].set_title('Fraud and NonFraud Card5 Distribution')

sns.distplot(test['card5'],ax=ax[1,1],hist=False,label='Test')
sns.distplot(train['card5'],ax=ax[1,1],hist=False,label='Train')
ax[1,1].set_title('Test and Train Card5 Distribution')

plt.tight_layout()


# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(10,8))
sns.countplot(train['card4'],ax=ax[0,0])
ax[0,0].set_title('Train Card4 Distribution')

sns.countplot(test['card4'],ax=ax[0,1])
ax[0,1].set_title('Test Card4 Distribution')

sns.countplot(train['card6'],ax=ax[1,0])
ax[1,0].set_title('Train Card6 Distribution')

sns.countplot(test['card6'],ax=ax[1,1])
ax[1,1].set_title('Test Card6 Distribution')

plt.tight_layout()


# ## <a id='22'>Distribution of Addr,Dist and Emaildomain across Train and Test</a> 

# In[ ]:


train.columns[:50]


# In[ ]:


addr=['addr1','addr2','dist1','dist2','P_emaildomain','R_emaildomain']
for c in addr:
    print(f'Number of unique variables in {c}: ',train[c].nunique())


# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(15,8))
sns.distplot(train[train['isFraud']==0]['addr1'],ax=ax[0,0],hist=False,label='NonFraud')
sns.distplot(train[train['isFraud']==1]['addr1'],ax=ax[0,0],hist=False,label='Fraud')
ax[0,0].set_title('Fraud and NonFraud addr1 Distribution')

sns.distplot(test['addr1'],ax=ax[0,1],hist=False,label='Test')
sns.distplot(train['addr1'],ax=ax[0,1],hist=False,label='Train')
ax[0,1].set_title('Test and Train addr1 Distribution')

sns.distplot(train[train['isFraud']==0]['addr2'],ax=ax[1,0],hist=False,label='NonFraud')
sns.distplot(train[train['isFraud']==1]['addr2'],ax=ax[1,0],hist=False,label='Fraud')
ax[1,0].set_title('Fraud and NonFraud addr2 Distribution')

sns.distplot(test['addr2'],ax=ax[1,1],hist=False,label='Test')
sns.distplot(train['addr2'],ax=ax[1,1],hist=False,label='Train')
ax[1,1].set_title('Test and Train addr2 Distribution')

plt.tight_layout()


# addr2 of test perfectly overlaps the addr2 of train and so it is not seen.

# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(15,8))
sns.distplot(train[train['isFraud']==0]['dist1'],ax=ax[0,0],hist=False,label='NonFraud')
sns.distplot(train[train['isFraud']==1]['dist1'],ax=ax[0,0],hist=False,label='Fraud')
ax[0,0].set_title('Fraud and NonFraud dist1 Distribution')

sns.distplot(test['dist1'],ax=ax[0,1],hist=False,label='Test')
sns.distplot(train['dist1'],ax=ax[0,1],hist=False,label='Train')
ax[0,1].set_title('Test and Train dist1 Distribution')

sns.distplot(train[train['isFraud']==0]['dist2'],ax=ax[1,0],hist=False,label='NonFraud')
sns.distplot(train[train['isFraud']==1]['dist2'],ax=ax[1,0],hist=False,label='Fraud')
ax[1,0].set_title('Fraud and NonFraud dist2 Distribution')

sns.distplot(test['dist2'],ax=ax[1,1],hist=False,label='Test')
sns.distplot(train['dist2'],ax=ax[1,1],hist=False,label='Train')
ax[1,1].set_title('Test and Train dist2 Distribution')

plt.tight_layout()


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,5))
sns.distplot(test['dist2'],hist=False,label='Test',ax=ax[0],color='orange')
ax[0].set_title('Distribution of dist2 in Test')
sns.distplot(train['dist2'],hist=False,label='Train',ax=ax[1])
ax[1].set_title('Distribution of dist2 in Train')
plt.tight_layout()


# Clearly the distribution is a lot different. I shall eliminate dist2.

# In[ ]:


fig,ax=plt.subplots(2,2,figsize=(15,8))
train['P_emaildomain'].value_counts()[:10].plot.bar(ax=ax[0,0])
ax[0,0].set_title('Train P_emaildomain Distribution')

test['P_emaildomain'].value_counts()[:10].plot.bar(ax=ax[0,1])
ax[0,1].set_title('Test P_emaildomain Distribution')

train['R_emaildomain'].value_counts()[:10].plot.bar(ax=ax[1,0])
ax[1,0].set_title('Train R_emaildomain Distribution')

test['R_emaildomain'].value_counts()[:10].plot.bar(ax=ax[1,1])
ax[1,1].set_title('Test R_emaildomain Distribution')

plt.tight_layout()


# ## <a id='23'>Distribution of C features across Train and Test</a> 

# In[ ]:


C_columns=[col for col in train.columns if 'C'==col[0]]
for c in C_columns:
    print(f'Number of unique entries in {c}:',train[c].nunique())


# In[ ]:


def get_subplots(feature1,feature2):
    fig,ax=plt.subplots(2,2,figsize=(15,8))
    sns.distplot(train[train['isFraud']==0][feature1],ax=ax[0,0],hist=False,label='NonFraud')
    sns.distplot(train[train['isFraud']==1][feature1],ax=ax[0,0],hist=False,label='Fraud')
    ax[0,0].set_title(f'Fraud and NonFraud {feature1} Distribution')

    sns.distplot(test[feature1],ax=ax[0,1],hist=False,label='Test')
    sns.distplot(train[feature1],ax=ax[0,1],hist=False,label='Train')
    ax[0,1].set_title(f'Test and Train {feature1} Distribution')

    sns.distplot(train[train['isFraud']==0][feature2],ax=ax[1,0],hist=False,label='NonFraud')
    sns.distplot(train[train['isFraud']==1][feature2],ax=ax[1,0],hist=False,label='Fraud')
    ax[1,0].set_title(f'Fraud and NonFraud {feature2} Distribution')

    sns.distplot(test[feature2],ax=ax[1,1],hist=False,label='Test')
    sns.distplot(train[feature2],ax=ax[1,1],hist=False,label='Train')
    ax[1,1].set_title(f'Test and Train {feature2} Distribution')

    plt.tight_layout()


# In[ ]:


get_subplots('C1','C2')


# In[ ]:


get_subplots('C3','C4')


# C4 seems very much useful for nonfraud and fraud distinction because its distribution is very different for those two categories.

# In[ ]:


get_subplots('C5','C6')


# In[ ]:


get_subplots('C7','C8')


# C8 and C7 have very different distribution in case of Fraud and NonFraud categories. So they might be useful. But again, C8 distribution in test and train is also very different. We will try dropping and not dropping it. 

# In[ ]:


get_subplots('C9','C10')


# C10 can be useful for distinction. But its distribution in test and train is very different as well. Same in case of C9. But I think it can be definitely dropped because its distribution in case of fraud and nonfraud is almost same.

# In[ ]:


get_subplots('C11','C12')


# In[ ]:


get_subplots('C13','C14')


# ## <a id='24'>Distribution of D features across Train and Test</a> 

# In[ ]:


train.columns[:50]


# In[ ]:


D_cols=[col for col in train.columns if col[0]=='D']
for d in D_cols:
    print(f'Number of unique entries in {d}:',train[d].nunique())


# In[ ]:


get_subplots('D1','D2')


# In[ ]:


get_subplots('D3','D4')


# In[ ]:


get_subplots('D5','D6')


# In[ ]:


def detailed_subplot(feature):
    fig,ax=plt.subplots(1,2,figsize=(15,5))
    sns.distplot(test[feature],hist=False,label='Test',ax=ax[0],color='orange')
    ax[0].set_title(f'Distribution of {feature} in Test')
    sns.distplot(train[feature],hist=False,label='Train',ax=ax[1])
    ax[1].set_title(f'Distribution of {feature} in Train')
    plt.tight_layout()


# In[ ]:


detailed_subplot('D6')


# I think we can safely drop this column

# In[ ]:


get_subplots('D7','D8')


# In[ ]:


print(train['D9'].value_counts())
print('*'*100)
print(test['D9'].value_counts())


# In[ ]:


get_subplots('D10','D11')


#  Drop some columns : From https://www.kaggle.com/jazivxt/safe-box/notebook

# In[ ]:


get_subplots('D12','D13')


# In[ ]:


detailed_subplot('D12')


# We can try dropping this column.

# In[ ]:


get_subplots('D14','D15')


# In[ ]:


detailed_subplot('D14')


# We can try dropping this.

# ## <a id='25'>Distribution of M features across Train and Test</a> 

# In[ ]:


M_cols=[col for col in train.columns if col[0]=='M']
for m in M_cols:
    print(f'Number of unique entries in {m}:',train[m].nunique())


# In[ ]:


def get_bar_subplots(feature1,feature2,top=10,incre=0):
    fig,ax=plt.subplots(2,2,figsize=(15,8))
    train[feature1].value_counts()[:top].plot.bar(ax=ax[0,0])
    ax[0,0].set_title(f'Train {feature1} Distribution')
    
    test[feature1].value_counts()[:top].plot.bar(ax=ax[0,1])
    ax[0,1].set_title(f'Test {feature1} Distribution')

    train[feature2].value_counts()[:top+incre].plot.bar(ax=ax[1,0])
    ax[1,0].set_title(f'Train {feature2} Distribution')

    test[feature2].value_counts()[:top+incre].plot.bar(ax=ax[1,1])
    ax[1,1].set_title(f'Test {feature2} Distribution')

    plt.tight_layout()


# In[ ]:


get_bar_subplots('M1','M2',2)


# In[ ]:


get_bar_subplots('M3','M4',2,1)


# In[ ]:


get_bar_subplots('M5','M6',2)


# In[ ]:


get_bar_subplots('M7','M8',2)


# In[ ]:


fig,ax=plt.subplots(1,2,figsize=(15,8))
train['M9'].value_counts().plot.bar(ax=ax[0])
ax[0].set_title('Train M9 Distribution')
    
test['M9'].value_counts().plot.bar(ax=ax[1])
ax[1].set_title('Test M9 Distribution')


# In[ ]:


train.columns[50:100]


# ## <a id='26'>Distribution of V features across Train and Test</a> 

# In[ ]:


V_cols=[col for col in train.columns if col[0]=='V']
V_type=[train[col].dtype for col in V_cols]
print(set(V_type))
print(len(V_cols))


# In[ ]:


unique=[(col,train[col].nunique()) for col in V_cols]
sorted_unique=sorted(unique,key=lambda x: x[1])
sorted_unique[:50]


# In[ ]:


def get_bar_subplots(feature1,feature2):
    fig,ax=plt.subplots(2,2,figsize=(15,8))
    train[feature1].value_counts().plot.bar(ax=ax[0,0])
    ax[0,0].set_title(f'Train {feature1} Distribution')
    
    test[feature1].value_counts().plot.bar(ax=ax[0,1])
    ax[0,1].set_title(f'Test {feature1} Distribution')

    train[feature2].value_counts().plot.bar(ax=ax[1,0])
    ax[1,0].set_title(f'Train {feature2} Distribution')

    test[feature2].value_counts().plot.bar(ax=ax[1,1])
    ax[1,1].set_title(f'Test {feature2} Distribution')

    plt.tight_layout()


# I skimmed through the distribution of every V_column and removed the columns which satisfied following criteria:
# 1. **extreme outliers in case of test or train** or 
# 2. **the Density of the major peak of test and train had very significant mismatch** or 
# 3. **test or train had lot of extra unique values compared to each other. **

# In[ ]:


get_subplots('V339','V338')


# In[ ]:


detailed_subplot('V328')


# ## <a id='27'>Distribution of ID features across Train and Test</a> 

# Same as done with other features. All the distributions are similar in test and train. So no ID column is dropped.

# In[ ]:


train.columns[350:]


# In[ ]:


i_cols=[col for col in train.columns if col[0]=='i']
unique_val=[(col,train[col].nunique()) for col in i_cols]
unique_val


# In[ ]:


get_subplots('id_01','id_02')


# In[ ]:


get_subplots('id_03','id_04')


# In[ ]:


get_subplots('id_05','id_06')


# In[ ]:


get_subplots('id_07','id_08')


# In[ ]:


get_subplots('id_09','id_10')


# In[ ]:


get_subplots('id_11','id_13')


# In[ ]:


get_subplots('id_14','id_17')


# In[ ]:


get_subplots('id_18','id_19')


# In[ ]:


get_subplots('id_20','id_21')


# In[ ]:


get_subplots('id_22','id_24')


# In[ ]:


get_subplots('id_25','id_26')


# In[ ]:


get_bar_subplots('id_38','id_37')


# In[ ]:


get_bar_subplots('id_35','id_36')


# In[ ]:


get_bar_subplots('id_32','id_34')


# In[ ]:


get_bar_subplots('id_28','id_29')


# In[ ]:


get_bar_subplots('id_23','id_27')


# In[ ]:


get_bar_subplots('id_15','id_16')


# In[ ]:


get_bar_subplots('id_12','id_15')


# # <a id='3'>Dropping Columns</a> 

# In[ ]:


drop_cols=['dist2','C8','C9','C10','D6','D12','V27','V28','V45','V70','V71','V77','V78','V86','V87','V89','V91','V92','V95','V96','V97',
          'V101','V102','V103','V107','V126','V127','V128','V130','V131','V132','V133','V134','V135','V136','V137','V143','V145','V150',
          'V159','V160','V164','V165','V166','V167','V168','V171','V176','V177','V178','V179','V181','V182','V183','V190','V199','V202',
          'V203','V204','V207','V211','V212','V213','V214','V215','V216','V221','V226','V227','V228','V230','V234','V240','V241','V245',
          'V246','V255','V256','V257','V258','V259','V261','V263','V264','V265','V270','V271','V272','V273','V274','V275','V276','V277',
          'V278','V279','V280','V291','V292','V293','V294','V295','V297','V306','V307','V308','V310','V312','V316','V317','V318','V319',
          'V320','V321','V322','V323','V324','V331','V332','V333','V338','V339','TransactionDT','TransactionID']

train=train.drop(drop_cols,1)
test=test.drop(drop_cols,1)
train['TransactionAmt']=np.log(train['TransactionAmt'])
test['TransactionAmt']=np.log(test['TransactionAmt'])


# In[ ]:



# drop_col = ['TransactionDT','TransactionID','V300', 'V309', 'V111', 'C3', 'V124', 'V106', 'V125', 'V315', 'V134', 'V102', 'V123', 
#             'V316', 'V113', 'V136', 'V305', 'V110', 'V299', 'V289', 'V286', 'V318', 'V103', 'V304', 'V116', 'V298', 
#             'V284', 'V293', 'V137', 'V295', 'V301', 'V104', 'V311', 'V115', 'V109', 'V119', 'V321', 'V114', 'V133', 
#             'V122', 'V319', 'V105', 'V112', 'V118', 'V117', 'V121', 'V108', 'V135', 'V320', 'V303', 'V297', 'V120']

# drop_col=['TransactionDT','TransactionID']
# train=train.drop(drop_col,1)
# test=test.drop(drop_col,1)


# # <a id='4'>Reducing memory size</a> 

# In[ ]:


get_ipython().run_cell_magic('time', '', '# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage\n# WARNING! THIS CAN DAMAGE THE DATA \ndef reduce_mem_usage(df):\n    """ iterate through all the columns of a dataframe and modify the data type\n        to reduce memory usage.        \n    """\n    start_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage of dataframe is {:.2f} MB\'.format(start_mem))\n    \n    for col in df.columns:\n        col_type = df[col].dtype\n        \n        if col_type != object:\n            c_min = df[col].min()\n            c_max = df[col].max()\n            if str(col_type)[:3] == \'int\':\n                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:\n                    df[col] = df[col].astype(np.int8)\n                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:\n                    df[col] = df[col].astype(np.int16)\n                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:\n                    df[col] = df[col].astype(np.int32)\n                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:\n                    df[col] = df[col].astype(np.int64)  \n            else:\n                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:\n                    df[col] = df[col].astype(np.float16)\n                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:\n                    df[col] = df[col].astype(np.float32)\n                else:\n                    df[col] = df[col].astype(np.float64)\n        else:\n            df[col] = df[col].astype(\'category\')\n\n    end_mem = df.memory_usage().sum() / 1024**2\n    print(\'Memory usage after optimization is: {:.2f} MB\'.format(end_mem))\n    print(\'Decreased by {:.1f}%\'.format(100 * (start_mem - end_mem) / start_mem))\n    \n    return df')


# In[ ]:


train=reduce_mem_usage(train)
test=reduce_mem_usage(test)


# In[ ]:


# obj_cols=[col for col in train.columns if train[col].dtype=='object']
# if 'isFraud' in obj_cols:
#     obj_cols.remove('isFraud')
    
# unique_values=sorted([(col,train[col].nunique()+test[col].nunique()) for col in obj_cols],key=lambda x: x[1],reverse=False)


# In[ ]:


# dummy_cols=[col[0] for col in unique_values[:18]]
# target=train['isFraud']
# train=train.drop('isFraud',1)
# ntrain=train.shape[0]
# print(train.shape)
# merged_data=pd.concat([train,test],axis=0,ignore_index=True)
# X=pd.get_dummies(merged_data,columns=dummy_cols)
# merged_data.drop(dummy_cols,axis=1,inplace=True)
# merged_data=pd.concat([merged_data,X],axis=1)
# del X
# train=merged_data[:ntrain]
# test=merged_data[ntrain:]
# print(train.shape)
# del merged_data
# gc.collect()


# # <a id='5'>Feature Engineering</a> 

# In[ ]:


num_cols=[col for col in train.columns if (('float' in str(train[col].dtype)) or ('int' in str(train[col].dtype)))]
num_cols.remove('isFraud')
train['mean']=train[num_cols].mean(axis=1)
test['mean']=test[num_cols].mean(axis=1)
train['std']=train[num_cols].std(axis=1)
test['std']=test[num_cols].std(axis=1)
train['max']=train[num_cols].max(axis=1)
test['max']=test[num_cols].max(axis=1)
train['min']=train[num_cols].min(axis=1)
test['min']=test[num_cols].min(axis=1)
train['median']=train[num_cols].median(axis=1)
test['median']=test[num_cols].median(axis=1)
train['skew']=train[num_cols].skew(axis=1)
test['skew']=test[num_cols].skew(axis=1)
train['kurt']=train[num_cols].kurt(axis=1)
test['kurt']=test[num_cols].kurt(axis=1)


# Randomly generating some features. Very illogical I know :P

# In[ ]:


# columns=['TransactionAmt','card1','card2','addr2','dist1','C1','C2','D1','D2','V1','V2',
#         'id_01','id_02']
# obj_cols=['DeviceInfo','card4','card6','ProductCD','DeviceType']

# for col in columns:
#     for feat in obj_cols:
#         train[f'{col}_mean_group_{feat}']=train[col]/train.groupby(feat)[col].transform('mean')
#         test[f'{col}_mean_group_{feat}']=test[col]/test.groupby(feat)[col].transform('mean')
#         train[f'{col}_max_group_{feat}']=train[col]/train.groupby(feat)[col].transform('max')
#         test[f'{col}_max_group_{feat}']=test[col]/test.groupby(feat)[col].transform('max')
#         train[f'{col}_min_group_{feat}']=train[col]/train.groupby(feat)[col].transform('min')
#         test[f'{col}_min_group_{feat}']=test[col]/test.groupby(feat)[col].transform('min')
#         train[f'{col}_skew_group_{feat}']=train[col]/train.groupby(feat)[col].transform('skew')
#         test[f'{col}_skew_group_{feat}']=test[col]/test.groupby(feat)[col].transform('skew')
#         train[f'{col}_skew_group_{feat}']=train[col]/train.groupby(feat)[col].transform('count')
#         test[f'{col}_skew_group_{feat}']=test[col]/test.groupby(feat)[col].transform('count')
   


# In[ ]:


# def fill_missing(df):
#     num_cols=[col for col in df.columns if df[col].dtype=='float64' or df[col].dtype=='int64']
#     for col in num_cols:
#         df[col]=df[col].fillna(df[col].mean())
#     obj_cols=[col for col in df.columns if df[col].dtype=='object']
#     for col in obj_cols:
#         df[col]=df[col].fillna(df[col].mode()[0])
        
#     return df


# In[ ]:


# train_df=fill_missing(train_df)
# test_df=fill_missing(test_df)


# In[ ]:


from sklearn.preprocessing import LabelEncoder

object_cols=[col for col in train.columns if (('category' in str(train[col].dtype)) or ('object' in str(train[col].dtype)))]
le=LabelEncoder()
for col in object_cols:
    le.fit(list(train[col].values)+list(test[col].values))
    train[col]=le.transform(list(train[col].values))
    test[col]=le.transform(list(test[col].values))


# In[ ]:


train.shape , test.shape


# In[ ]:


target=train['isFraud']
train=train.drop('isFraud',1)


# In[ ]:


# from sklearn.model_selection import train_test_split
# train_X,val_X,train_y,val_y=train_test_split(train,target,test_size=0.2,random_state=seed,stratify=target)


# In[ ]:


# from xgboost import XGBClassifier
# from sklearn.metrics import roc_auc_score


# def objective(params):
#     params=dict(max_depth=int(params['max_depth']),
#                subsample=np.round(params['subsample'],3),
#                colsample_bytree=np.round(params['colsample_bytree'],3),
#                learning_rate=np.round(params['learning_rate'],3),
#                verbosity=0)
    
#     clf=XGBClassifier(n_estimators=1000,random_state=seed,**params,tree_method='gpu_hist')
#     clf.fit(train_X,train_y,eval_set=[(val_X,val_y)],eval_metric='auc',early_stopping_rounds=10)
#     val_pred=clf.predict(val_X)
#     score=roc_auc_score(val_y,val_pred)
#     return score

# space={'max_depth':hp.quniform('max_depth',2,10,2),
#       'subsample':hp.uniform('subsample',0.1,1),
#       'colsample_bytree':hp.uniform('colsample_bytree',0.1,1),
#       'learning_rate':hp.uniform('learning_rate',0.01,0.1)}

# trial=Trials()
# best=fmin(fn=objective,algo=tpe.suggest,space=space,max_evals=100,trials=trial,rstate=np.random.RandomState(seed))

    


# In[ ]:


# best['max_depth']=int(best['max_depth'])
# print('Best parameters:',best)


# In[ ]:


# del train_X,val_X,train_y,val_y
# gc.collect()


# In[ ]:


# TID=[t['tid'] for t in trial.trials]
# Loss=[t['result']['loss'] for t in trial.trials]
# maxd=[t['misc']['vals']['max_depth'][0] for t in trial.trials]
# lr=[t['misc']['vals']['learning_rate'][0] for t in trial.trials]
# sub=[t['misc']['vals']['subsample'][0] for t in trial.trials]
# col_samp=[t['misc']['vals']['colsample_bytree'][0] for t in trial.trials]


# hyperopt_xgb=pd.DataFrame({'tid':TID,'loss':Loss,
#                           'max_depth':maxd,'learning_rate':lr,
#                           'subsample':sub, 'colsample_bytree':col_samp})


# In[ ]:


# plt.subplots(3,2,figsize=(10,10))
# plt.subplot(3,2,1)
# sns.scatterplot(x='tid',y='max_depth',data=hyperopt_xgb)
# plt.subplot(3,2,2)
# sns.scatterplot(x='tid',y='loss',data=hyperopt_xgb)
# plt.subplot(3,2,3)
# sns.scatterplot(x='tid',y='learning_rate',data=hyperopt_xgb)
# plt.subplot(3,2,4)
# sns.scatterplot(x='tid',y='subsample',data=hyperopt_xgb)
# plt.subplot(3,2,5)
# sns.scatterplot(x='tid',y='colsample_bytree',data=hyperopt_xgb)
# plt.subplot(3,2,6)
# sns.scatterplot(x='tid',y='loss',data=hyperopt_xgb)

# plt.tight_layout()


# # <a id='6'>Model Development and Feature Importance</a> 

# In[ ]:


from sklearn.model_selection import StratifiedKFold
import gc

nfolds=10


xgb_params=dict(n_estimators=1000,
                verbosity=0,
                tree_method='gpu_hist',
                random_state=seed,
               colsample_bytree=0.6,
               subsample=0.6,
               learning_rate=0.05,
               max_depth=9)

lgb_params=dict(objective='binary',
               num_leaves=62,
               seed=seed,
               max_depth=9,
               pos_bagging_fraction=0.5,
               neg_bagging_fraction=1.0,
               bagging_freq=5,
               feature_fraction=0.9,
                metric='auc',
               learning_rate=0.05,
               verbosity=-1,
               device='gpu')


skfold=StratifiedKFold(nfolds,random_state=seed)



def build_model(params,model='xgb',plot_feature_importance=True):
    oof=np.zeros(train.shape[0])
    pred=np.zeros(test.shape[0])
    scores=[]
    feature_importance=pd.DataFrame()
    for i,(train_index,val_index) in enumerate(skfold.split(train,target)):
        print('Fold :',i+1)

        
        if model=='xgb':
            train_X,val_X=train.iloc[train_index,:],train.iloc[val_index,:]
            train_y,val_y=target[train_index],target[val_index]
            clf=XGBClassifier(**params)
            clf.fit(train_X,train_y,eval_metric='auc',eval_set=[(val_X,val_y)],early_stopping_rounds=10,verbose=20)
            val_pred=clf.predict_proba(val_X)[:,1]
        
        
        if model=='lgb':
        
            train_d=lgb.Dataset(train.iloc[train_index,:].values,label=target[train_index].values)
            val_d=lgb.Dataset(train.iloc[val_index,:].values,label=target[val_index].values)
            clf=lgb.train(params,train_d,num_boost_round=1000,valid_sets=[val_d],verbose_eval=20,early_stopping_rounds=10)
            val_pred=clf.predict(train.iloc[val_index,:].values)
        
    
        oof[val_index]=val_pred
        val_score=roc_auc_score(target[val_index],val_pred)
        scores.append(val_score)
        print(f'Validation score using {model} for fold {i} :'+ str(val_score))
        print('-'*100)
        
        if model=='xgb':
            pred+=clf.predict_proba(test)[:,1]/nfolds
        if model=='lgb':
            pred+=clf.predict(test.values)/nfolds
            
        if model=='xgb':
            del train_X,val_X,train_y,val_y
        if model=='lgb':
            del train_d,val_d
       
        gc.collect()
        
        
        fold_importance=pd.DataFrame()
        fold_importance['feature']=train.columns
        if model=='xgb':
            fold_importance['importance']=clf.feature_importances_
        if model=='lgb':
            fold_importance['importance']=clf.feature_importance()
        fold_importance['fold']=i+1
        feature_importance=pd.concat([feature_importance,fold_importance],axis=0)
            
            
    print('Mean validation score :',np.mean(scores)) 
    
    if plot_feature_importance:
        df=feature_importance[['feature','importance']].groupby('feature').mean().sort_values(by='importance',ascending=False).reset_index()
        plt.figure(figsize=(10,10))
        sns.barplot(x='importance',y='feature',data=df.iloc[:25,:])
        plt.title('Feature Importances')
        
    return pred


# In[ ]:


pred=build_model(xgb_params,model='xgb')


# In[ ]:


sub=pd.read_csv('../input/sample_submission.csv')
sub['isFraud']=pred
sub.to_csv('submission.csv',index=False)

