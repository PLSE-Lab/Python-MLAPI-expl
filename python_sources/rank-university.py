#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import sklearn as sk
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split


# In[2]:


world_rank = pd.read_csv('../input/cwurData.csv')


# In[3]:


world_rank.info()
world_rank.describe([0.25,0.50,0.75,0.99])
world_rank.columns
world_rank[list(world_rank.dtypes[world_rank.dtypes=='object'].index)].head()


# In[4]:


world_rank.shape


# In[5]:


world_rank.isnull().all()
world_rank.isnull().any()
round(100*(world_rank.isnull().sum()/len(world_rank.index)),2)


# In[6]:


world_rank= world_rank[~world_rank.broad_impact.isnull()]


# In[7]:


round(100*(world_rank.isnull().sum()/len(world_rank.index)),2)


# In[8]:


world_rank.dtypes[world_rank.dtypes!='object']


# In[9]:


world_rank = world_rank.drop('year',axis='columns')


# In[10]:


sns.pairplot(world_rank,x_vars = ['national_rank','quality_of_education','alumni_employment',
                       'quality_of_faculty','publications'],y_vars = 'world_rank')
sns.pairplot(world_rank,x_vars = ['influence','citations','broad_impact',
                       'patents','score'],y_vars = 'world_rank')
plt.show()


# In[11]:


plt.figure(figsize=(10,10))
sns.heatmap(world_rank.corr(),annot = True,cmap='summer')


# In[12]:


sns.lmplot(x = 'broad_impact',y= 'world_rank',data = world_rank)
plt.title('world rank vs broad impact')
plt.show()


# In[13]:


sns.lmplot(x = 'influence',y= 'world_rank',data = world_rank)
plt.title('world rank vs broad impact')
plt.show()


# In[14]:


country_frame = pd.get_dummies(world_rank.country,drop_first=True)


# In[15]:


df_world_rank = pd.concat([world_rank,country_frame],axis = 'columns')


# In[16]:


df_world_rank = df_world_rank.drop(list(df_world_rank.dtypes[df_world_rank.dtypes=='object'].index),axis='columns')


# In[17]:


df_train,df_test = train_test_split(df_world_rank,train_size = 0.7,test_size=0.3,random_state=42)


# In[18]:


from sklearn.preprocessing import StandardScaler,MinMaxScaler


# In[19]:


vars = ['world_rank', 'national_rank', 'quality_of_education',
       'alumni_employment', 'quality_of_faculty', 'publications', 'influence',
       'citations', 'broad_impact', 'patents', 'score']
scaler = MinMaxScaler()
df_train[vars] = scaler.fit_transform(df_train[vars])


# In[20]:


df_train.shape
df_train.head()


# In[21]:


y_train = df_train.pop('world_rank')
X_train = df_train


# In[22]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_vif(df):
    vif = pd.DataFrame()
    X = df
    vif['features'] = X.columns
    vif['VIF'] = [variance_inflation_factor(X.values,i) for i in range(X.shape[1])]
    vif['VIF'] = round(vif['VIF'],2)    
    vif = vif.sort_values(by = 'VIF',ascending=False)
    return vif


# In[23]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()


# In[24]:


from sklearn.feature_selection import RFE
lm.fit(X_train,y_train)
rfe = RFE(lm,15)
rfe = rfe.fit(X_train,y_train)


# In[25]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[26]:


cols = X_train.columns[rfe.support_]
cols


# In[27]:


X_train_rfe = X_train[cols]


# In[28]:


X_train_rfe.head()


# In[29]:


# Model 1
import statsmodels.api as sm
X_train_lm1 = sm.add_constant(X_train_rfe)
lm1 = sm.OLS(y_train,X_train_lm1).fit()
print(lm1.summary())


# In[30]:


# Model 2
import statsmodels.api as sm
X_train_new2 = X_train_rfe.drop('Singapore',axis='columns')
X_train_lm2 = sm.add_constant(X_train_new2)
lm2 = sm.OLS(y_train,X_train_lm2).fit()
print(lm2.summary())


# In[31]:


# Model 3
X_train_new3 = X_train_new2.drop('Colombia',axis='columns')
X_train_lm3 = sm.add_constant(X_train_new3)
lm3 = sm.OLS(y_train,X_train_lm3).fit()
print(lm3.summary())


# In[32]:


vif1 = X_train_lm3.drop('const',axis='columns')
calculate_vif(vif1)


# In[33]:


# Model 4
import statsmodels.api as sm
X_train_new4 = X_train_new3.drop('publications',axis='columns')
X_train_lm4 = sm.add_constant(X_train_new4)
lm4 = sm.OLS(y_train,X_train_lm4).fit()
print(lm4.summary())


# In[34]:


vif2 = X_train_lm4.drop('const',axis=1)
calculate_vif(vif2)


# In[35]:


# Model 5
import statsmodels.api as sm
X_train_new5 = X_train_new4.drop('quality_of_education',axis='columns')
X_train_lm5 = sm.add_constant(X_train_new5)
lm5 = sm.OLS(y_train,X_train_lm5).fit()
print(lm5.summary())


# In[36]:


vif3 = X_train_lm5.drop('const',axis=1)
calculate_vif(vif3)


# In[37]:


# Model 6
import statsmodels.api as sm
X_train_new6 = X_train_new5.drop('patents',axis='columns')
X_train_lm6 = sm.add_constant(X_train_new6)
lm6 = sm.OLS(y_train,X_train_lm6).fit()
print(lm6.summary())


# In[38]:


# Model 7
import statsmodels.api as sm
X_train_new7 = X_train_new6.drop('score',axis='columns')
X_train_lm7 = sm.add_constant(X_train_new7)
lm7 = sm.OLS(y_train,X_train_lm7).fit()
print(lm7.summary())


# In[39]:


vif4 = X_train_lm7.drop('const',axis=1)
calculate_vif(vif4)


# In[40]:


y_train_rank = lm7.predict(X_train_lm7)


# In[41]:


from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


fig = plt.figure()
sns.distplot(y_train_rank-y_train,bins=20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)  
fig.show()


# In[43]:


vars1 = ['world_rank', 'national_rank', 'quality_of_education',
       'alumni_employment', 'quality_of_faculty', 'publications', 'influence',
       'citations', 'broad_impact', 'patents', 'score']
df_test[vars1] = scaler.transform(df_test[vars1])


# In[44]:


y_test = df_test.pop('world_rank')
X_test = df_test


# In[45]:


X_test_new = X_test[X_train_new7.columns]


# In[46]:


X_test_new = sm.add_constant(X_test_new)


# In[47]:


y_pred = lm7.predict(X_test_new)


# In[48]:


fig = plt.figure()
plt.scatter(y_pred,y_test)
fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 
plt.xlabel('y_test', fontsize=18)                          # X-label
plt.ylabel('y_pred', fontsize=16)    


# In[49]:


from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(y_test,y_pred))


# In[50]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[ ]:




