#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas  as pd 
 
import matplotlib.pyplot as plt
import seaborn as sns

import os
os.chdir('..//input//ieee-fraud-detection')

import missingno as ms 

get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder,LabelBinarizer
from tqdm import tqdm


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


train_id = pd.read_csv('train_identity.csv')
train_tran = pd.read_csv('train_transaction.csv')
test_id = pd.read_csv('test_identity.csv')
test_tran = pd.read_csv('test_transaction.csv')


# In[ ]:


train = pd.merge(train_tran,train_id,on='TransactionID',how='left')
test = pd.merge(test_tran,test_id,on='TransactionID',how='left')


# In[ ]:


def null_function(data):
    null_values = pd.DataFrame((data.isnull().sum()/len(data.index)*100),columns=['Percent_Null'])
    only_missing_variables = null_values[null_values['Percent_Null'] !=0 ]
    return only_missing_variables.sort_values(by='Percent_Null', ascending=False)


# In[ ]:


def removing_missing_columns(df,percent):
    dfn = df.copy()
    a = dfn.columns
    c =[]
    for i in a:
        c.append(round(((dfn[i].isnull().sum())/((dfn.shape[0])))*100,3))
    d = pd.DataFrame([c],columns=dfn.columns)
    f = []
    for j in a:
        if (d[j].values) > percent:
            f.append(j)
    return f 


# In[ ]:


r = removing_missing_columns(train,40.0)  


# In[ ]:


train=train.drop(r,axis=1)
test = test.drop(r,axis=1) 


# In[ ]:


print(train.shape)
print(test.shape)


# In[ ]:


complete_data = pd.concat([train,test])


# In[ ]:


c_1 = train.select_dtypes(include='object')
n_1 = train.select_dtypes(exclude='object')

cat_columns = c_1.columns.values
num_columns = n_1.columns.values


# In[ ]:


mod_1 = complete_data['ProductCD'].mode()
mod_2 = complete_data['card4'].mode()
mod_3 = complete_data['card6'].mode()
mod_4 = complete_data['P_emaildomain'].mode()
mod_5 = complete_data['M6'].mode()


# In[ ]:


print(mod_1)
print(mod_2)
print(mod_3)
print(mod_4)
print(mod_5)


# In[ ]:



train['ProductCD'].fillna('W',inplace=True)
train['card4'].fillna('visa',inplace=True)
train['card6'].fillna('debit',inplace=True)
train['P_emaildomain'].fillna('gmail.com',inplace=True)
train['M6'].fillna('F',inplace=True)

test['ProductCD'].fillna('W',inplace=True)
test['card4'].fillna('visa',inplace=True)
test['card6'].fillna('debit',inplace=True)
test['P_emaildomain'].fillna('gmail.com',inplace=True)
test['M6'].fillna('F',inplace=True)


# In[ ]:


num_columns = ['TransactionID', 'TransactionDT', 'TransactionAmt',
       'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'C1', 'C2',
       'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
       'C13', 'C14', 'D1', 'D4', 'D10', 'D15', 'V12', 'V13', 'V14', 'V15',
       'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24',
       'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33',
       'V34', 'V35', 'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42',
       'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51',
       'V52', 'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60',
       'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69',
       'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78',
       'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87',
       'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96',
       'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104',
       'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112',
       'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120',
       'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128',
       'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135', 'V136',
       'V137', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285',
       'V286', 'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293',
       'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V301',
       'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309',
       'V310', 'V311', 'V312', 'V313', 'V314', 'V315', 'V316', 'V317',
       'V318', 'V319', 'V320', 'V321']


# In[ ]:


for i in tqdm(num_columns):
    med = complete_data[i].median()
    train[i].fillna(med,inplace=True)
    test[i].fillna(med,inplace=True)
    


# In[ ]:


pd.DataFrame(null_function(train))


# In[ ]:


pd.DataFrame(null_function(test))


# In[ ]:


x_train = train.drop('isFraud',axis=1)
y_train = train[['isFraud']]
x_test = test.copy() 


# In[ ]:


for i in cat_columns:
    lb = LabelBinarizer()
    lb.fit(list(x_train[i].values)+list(x_test[i].values))
    x_train[i] = lb.transform(list(x_train[i].values))
    x_test[i] = lb.transform(list(x_test[i].values)) 


# In[ ]:


x_train_1 = x_train.drop(['TransactionID','TransactionDT'],axis=1)
x_test_1 = x_test.drop(['TransactionID','TransactionDT'],axis=1)


# In[ ]:


rf = RandomForestClassifier(random_state=1,n_estimators=200,class_weight='balanced')
rf.fit(x_train_1,y_train)


# In[ ]:


y_pre = rf.predict(test)


# In[ ]:


submission_file = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


submission_file['isFraud'] = y_pre[:,1]


# In[ ]:


submission_file.to_csv('submission_file.csv', index=False)

