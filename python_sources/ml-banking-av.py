#!/usr/bin/env python
# coding: utf-8

# ### Importing Dataset and Libraries

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold,train_test_split,KFold
import re
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score,f1_score


# In[ ]:


url='/kaggle/input/janatahack-machine-learning-for-banking/'
train_df = pd.read_csv(url+'train_fNxu4vz.csv')
test_df = pd.read_csv(url+'test_fjtUOL8.csv')
submission = pd.read_csv(url+'sample_submission_HSqiq1Q.csv')


# ### Knowing about Data

# In[ ]:


train_df.head(3)


# In[ ]:


test_df.head(3)


# #### Missing Values and Data types

# In[ ]:


d1=train_df.isnull().sum().to_frame().rename(columns={0: "Train_Null_Values"})
d2=test_df.isnull().sum().to_frame().rename(columns={0: "Test_Null_Values"})
d3=train_df.dtypes.to_frame().rename(columns={0: "Data_Type"})
pd.concat([d1, d2,d3], axis=1)


# ## EDA

# ### Target Exploration

# In[ ]:


train_df['Interest_Rate'].value_counts(normalize=True)


# In[ ]:


sns.countplot(train_df["Interest_Rate"])


# ### Univariate Analysis & Missing Value_Treatment

# In[ ]:


train=train_df.append(test_df,sort=False)


# In[ ]:


train.columns


# In[ ]:


plt.figure(figsize=(24, 6))
plt.subplot(131)
sns.countplot(train['Home_Owner'],order = train['Home_Owner'].value_counts(dropna=False).index)
plt.subplot(132)
sns.countplot(train['Income_Verified'],order = train['Income_Verified'].value_counts(dropna=False).index)
plt.subplot(133)
sns.countplot(train['Gender'],order = train['Gender'].value_counts(dropna=False).index)


# In[ ]:


train['Home_Owner'].value_counts(dropna=False,normalize=True)


# Home Owner Variable contains 15% of data,Which more than other category, so it is handles as seperate Variable

# In[ ]:


train.Home_Owner.fillna('Unknown',inplace=True)


# In[ ]:


plt.figure(figsize=(24, 6))
sns.countplot(train['Purpose_Of_Loan'],order = train['Purpose_Of_Loan'].value_counts(dropna=False).index)


# In[ ]:


train['Months_Since_Deliquency'].value_counts(dropna=False,normalize=True)


# 50% Data, is Missing in *Months_Since_Deliquency* Variable, ***Model Built with avoiding this variable gave poor accuracy than***, Substuting the variable with dummy number.

# In[ ]:


train.Months_Since_Deliquency.fillna(9999,inplace=True)


# In[ ]:


plt.figure(figsize=(24, 6))
plt.subplot(121)
sns.distplot(train["Annual_Income"])
plt.subplot(122)
sns.distplot(np.log1p(train["Annual_Income"]))
plt.show()


# In[ ]:


train['Annual_Income']=np.log1p(train["Annual_Income"])
train.Annual_Income.fillna(train.Annual_Income.median(),inplace=True)


# In[ ]:


plt.figure(figsize=(24, 6))
plt.subplot(121)
sns.distplot(train["Total_Accounts"])
plt.subplot(122)
sns.distplot(np.log1p(train["Total_Accounts"]))
plt.show()


# In[ ]:


train['Total_Accounts']=np.log1p(train["Total_Accounts"])


# In[ ]:


train.Length_Employed.value_counts(dropna=False,normalize=True)


# Length_Employed is equally Distributed Among, Working years, So I have added as Seperate Variable, I Didn't tried using Mode because it cotributes only 30%

# In[ ]:


train.Length_Employed.fillna(99,inplace=True)
train.Length_Employed.replace({'10+ years':'10 years','< 1 year': '0 year'},inplace= True)
train.Length_Employed=train.Length_Employed.apply(lambda x: ''.join(re.findall('\d+',str(x))))
train.Length_Employed=train.Length_Employed.astype(int)


# Loan Amount Request contain ',', so it becames object type data, So it has to converted as Numerical Data

# In[ ]:


train.Loan_Amount_Requested=train.Loan_Amount_Requested.apply(lambda x: x.replace(',',''))
train.Loan_Amount_Requested=train.Loan_Amount_Requested.astype(int)


# In[ ]:


plt.figure(figsize=(24, 6))
plt.subplot(121)
sns.distplot(train["Loan_Amount_Requested"])
plt.subplot(122)
sns.distplot(np.log1p(train["Loan_Amount_Requested"]))
plt.show()


# In[ ]:


train['Loan_Amount_Requested']=np.log1p(train["Loan_Amount_Requested"]) #Normalizing not making much difference


# In[ ]:


train_df.shape,test_df.shape


# In[ ]:


train_df=train[:164309]
test_df=train[164309:]


# # Model Building

# In[ ]:


# Set up folds
K = 5
skf = StratifiedKFold(n_splits = K, random_state = 7, shuffle = True)


# In[ ]:


cat_columns=train_df.select_dtypes(include='object').columns.tolist()
cat_columns.extend(['Length_Employed']) #Considering the Length_Employed as Category increased the Accuracy
# To specify categorical variables indexes
cat_columns


# *Learned from AV Hackathon Janatahack Solution*

# In[ ]:


X = train_df.drop(columns=['Loan_ID','Interest_Rate'],axis=1)
y = train_df['Interest_Rate']
X_test = test_df.drop(columns=['Loan_ID'],axis=1)
y_valid_pred = 0*y
y_test_pred = 0
accuracy = 0
result={}
#fitting catboost classifier model
j=1
model = CatBoostClassifier(n_estimators=1000,verbose=False,learning_rate=0.1)
for train_index, test_index in skf.split(X, y):  
    # Create data for this fold
    y_train, y_valid = y.iloc[train_index], y.iloc[test_index]
    X_train, X_valid = X.iloc[train_index,:], X.iloc[test_index,:]
    print( "\nFold ", j)
    
    # Run model for this fold
    fit_model = model.fit( X_train, y_train, eval_set=(X_valid, y_valid),cat_features=cat_columns, use_best_model=True)
    print( "  N trees = ", model.tree_count_ )
        
    # Generate validation predictions for this fold
    pred = fit_model.predict(X_valid)
    y_valid_pred.iloc[test_index] = pred.reshape(-1)
    print(accuracy_score(y_valid,pred))
    accuracy+=accuracy_score(y_valid,pred)
    # Accumulate test set predictions
    y_test_pred += fit_model.predict(X_test)
    result[j]=fit_model.predict(X_test)
    j+=1
results = y_test_pred / K  # Average test set predictions
print(accuracy/5)


# ### Solution

# In[ ]:


prediction = pd.DataFrame()
for i in range(1, 6):
    prediction = pd.concat([prediction,pd.DataFrame(result[i])],axis=1)
prediction.columns=['Split1','Split2','Split3','Split4','Split5']


# In[ ]:


submission.Interest_Rate=prediction.mode(axis=1)[0]
submission.to_csv('CatBoost.csv',index = False)


# In[ ]:




