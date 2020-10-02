#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from matplotlib import pyplot as plt
import numpy as np # linear algebra
import category_encoders as ce
from sklearn.metrics import median_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import BaggingRegressor
import seaborn as sns
from scipy import stats
import numpy as np
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#run these 
"""run1-outlier section and check the loss and tune the threshold value if it effect the loss"""
"""run2-plot vs cc_acc(target is the leftmost column)"""
"""run3-grid search on random forest"""#plz write the code of grid search for xgb work better till now and set hyperparameters
"""there are three plz do->1,2,3 """


# Training Data

# In[ ]:



df=pd.read_csv('../input/train.csv')
print('df.shape :->'+str(df.shape))
df.describe()


# Test Data

# In[ ]:


df_test=pd.read_csv('../input/test.csv')
df_test.head()


# Sample

# In[ ]:


df_sample=pd.read_csv('../input/sample.csv')
df_sample.head()


# In[ ]:


df.head()


# In[ ]:


df.info()


# **Outlier**

# In[ ]:


import seaborn as sns
for i in range(3,44):
    x=df.iloc[:,i]
    y=df['cc_cons']
    sns.boxplot(x=x)
    plt.show()


# In[ ]:


"""plz do 1- remove outlier and then check acc is it giving some benifits or change threshold if require"""
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(df.iloc[:,1:]))
threshold = 3
df_rm = df_train[(z < threshold).all(axis=1)]
print('df_train_rm shape :'+str(df_rm.shape))
print('df_train shape :'+str(df.shape))


#  **Data plots **

# In[ ]:


"""plz do 2- iam not getting any know form data no relation so i think this is of now use plz see this ones"""
features=df.iloc[:,3:40].columns
for i in features:
    sns.lmplot(x=i, y='cc_cons', data=df,line_kws={'color': 'red'})
    text="Relation between Points and " + i 
    plt.title(text)
    plt.show()


# **Data Cleaning**

# In[ ]:


df_rem=df.iloc[:,1:]#remove id
df_test_rem=df_test.iloc[:,1:]
df_test_rem.shape


# Fill the missing value of training data
# 

# In[ ]:


df_rem.head()
for i in df_rem.columns:
    if(i!='account_type' and i!='gender' and i!='loan_enq'):
         df_rem[i]=df_rem[i].fillna(df_rem[i].median())


# Fill the missing value of test data

# In[ ]:


df_test_rem.head()
for i in df_test_rem.columns:
    if(i!='account_type' and i!='gender' and i!='loan_enq'):
         df_test_rem[i]=df_test_rem[i].fillna(df_test_rem[i].median())


# In[ ]:


df_rem_no_null_now=df_rem


# In[ ]:


df_rem


# Fill loan_enq with missing values with N

# In[ ]:


df_1=df_rem
df_1.iloc[:,40]=df_1.iloc[:,40].fillna('N')
df_1=df_1.iloc[:,:].values


# same fot test

# In[ ]:


df_1_test=df_test_rem
df_1_test.iloc[:,40]=df_1_test.iloc[:,40].fillna('N')
df_1_test=df_1_test.iloc[:,:].values


# In[ ]:


df_1[:,40]


# Encoding for gender,account and loan_enq

# In[ ]:


le = LabelEncoder()
#df_1=df_1.iloc[:,:].values
df_1[:, 0] = le.fit_transform(df_1[:, 0])
df_1[:, 1] = le.fit_transform(df_1[:, 1])
df_1[:, 40] = le.fit_transform(df_1[:, 40])


# In[ ]:


ohe = OneHotEncoder(categorical_features = [0])
X = ohe.fit_transform(df_1).toarray()
ohe = OneHotEncoder(categorical_features = [1+1])
X = ohe.fit_transform(X).toarray()
ohe = OneHotEncoder(categorical_features = [40+1+1])
X = ohe.fit_transform(X).toarray()


# In[ ]:


df_1_test.shape


# encodeing for test

# In[ ]:


le = LabelEncoder()
#df_1=df_1.iloc[:,:].values
df_1_test[:, 0] = le.fit_transform(df_1_test[:, 0])
df_1_test[:, 1] = le.fit_transform(df_1_test[:, 1])
df_1_test[:, 40] = le.fit_transform(df_1_test[:, 40])##column 40 is of loan_enq
ohe_1 = OneHotEncoder(categorical_features = [0])
X_test = ohe_1.fit_transform(df_1_test).toarray()
ohe_1 = OneHotEncoder(categorical_features = [1+1])
X_test= ohe_1.fit_transform(X_test).toarray()
ohe_1= OneHotEncoder(categorical_features = [40+1+1])
X_test = ohe_1.fit_transform(X_test).toarray()


# In[ ]:


X_test=pd.DataFrame(X_test)
X_test.head()


# In[ ]:


X.shape


# In[ ]:


X_new = pd.DataFrame(X)
"""X_new is ready to feed now"""
X_new


# In[ ]:


X_new1=shuffle(X_new)


# In[ ]:


X_load=X_new1.iloc[:,:45]
y_load=X_new1.iloc[:,45]


# In[ ]:


X_load.head()


# In[ ]:


y_load.head()


# In[ ]:


#next task divide test and cv and apply bagging , xgboost regression and test it
X_train, X_cv, y_train, y_cv = train_test_split(X_load, y_load, test_size=0.25, random_state=42)


# In[ ]:


print('X_train.shape:'+str(X_train.shape))
print('X_cv.shape:'+str(X_cv.shape))


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
min_max_scaler =MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X_train)
df_normov = pd.DataFrame(np_scaled)
np_scaled2 = min_max_scaler.fit_transform(X_cv)
df_cv=pd.DataFrame(np_scaled2)
df_normov.head()


# In[ ]:


"""plz do -3 please apply grid search so we get better acc and suggest so we can get better acc pipline? kfold?"""


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=300,max_depth=30)
rfr.fit(df_normov,y_train)
y_pred_rfr=rfr.predict(df_cv)
rfr_pred=mean_squared_error(y_cv, y_pred_rfr)
print(rfr_pred)
print('r2_score of linear regression model:->'+str(r2_score(y_cv, y_pred_rfr)))


# In[ ]:


xgb_pred=mean_absolute_error(np.array(y_cv), y_pred_rfr)
xgb_pred


# In[ ]:


model = BaggingRegressor(DecisionTreeRegressor(max_depth=20),n_estimators=100)
model.fit(df_normov, y_train)
y_pred_bg=model.predict(np.array(df_cv))
bg_pred=mean_squared_error(np.array(y_cv),y_pred_bg)
print(bg_pred)
print('r2_score of linear regression model:->'+str(r2_score(np.array(y_cv), y_pred_bg)))


# In[ ]:


from sklearn.metrics import median_absolute_error
bg_pred=median_absolute_error(np.array(y_cv),y_pred_bg)
print(bg_pred)


# In[ ]:


xgb=XGBRegressor(n_estimators=200, learning_rate=0.01, gamma=0.001, subsample=0.75,  colsample_bytree=1, max_depth=30
xgb.fit(df_normov,y_train)
y_pred_xgb=xgb.predict(df_cv)
xgb_pred=mean_squared_error(np.array(y_cv), y_pred_xgb)
print(xgb_pred)
print('r2_score of linear regression model:->'+str(r2_score(np.array(y_cv), y_pred_xgb))) 


# In[ ]:


print('r2_score of linear regression model:->'+str(r2_score(np.array(y_cv), y_pred_xgb))) 
xgb_pred=mean_squared_error(np.array(y_cv), y_pred_rfr)
print(xgb_pred)
xgb_pred2=median_absolute_error(np.array(y_cv),y_pred_xgb)
print(xgb_pred2)


# In[ ]:


#next task apply grid search on xgb and bagging at night from 11 pm to 1 am


# In[ ]:


y_pred_final=xgb.predict(X_test)


# In[ ]:


res=pd.DataFrame(y_pred_final)
df_sample.iloc[:,1]=res.iloc[:,0]


# In[ ]:


df_sample


# In[ ]:


from IPython.display import HTML

df_sample.to_csv('submission.csv')

def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)

# create a link to download the dataframe which was saved with .to_csv method
create_download_link(filename='submission.csv')

