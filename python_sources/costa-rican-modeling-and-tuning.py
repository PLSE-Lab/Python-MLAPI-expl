#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  OneHotEncoder as ohe
from sklearn.preprocessing import StandardScaler as ss
from sklearn.compose import ColumnTransformer as ct
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier as rf
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier as gbm
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb


from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
import sklearn.metrics as metrics
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


from bayes_opt import BayesianOptimization
from skopt import BayesSearchCV
from eli5.sklearn import PermutationImportance


get_ipython().run_line_magic('matplotlib', 'inline')


# In[47]:


# Read in data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[48]:


train.head()


# In[49]:


train.shape, test.shape


# > Missing Values

# In[50]:


train.isnull().values.any()


# In[51]:


test.isnull().values.any()


# In[52]:


train.info()
train.isnull().values.sum(axis=0)


# In[53]:


train_describe = train.describe()
train_describe


# In[54]:


test_describe = test.describe()
test_describe


# In[55]:


test.isnull().values.sum(axis=0)


# Distribution of Target Variable

# In[56]:


plt.figure(figsize=(12, 5))
plt.hist(train.Target.values, bins=4)
plt.title('Histogram - target counts')
plt.xlabel('Count')
plt.ylabel('Target')
plt.show()


# In[57]:


plt.title("Distribution of Target")
sns.distplot(train['Target'].dropna(),color='blue', kde=True,bins=100)
plt.show()


# In[58]:


sns.set_style("whitegrid")
ax = sns.violinplot(x=train.Target.values)
plt.show()


# In[59]:


plt.title("Distribution of log(target)")
sns.distplot(np.log1p(train['Target']).dropna(),color='blue', kde=True,bins=100)
plt.show()


# In[60]:


sns.set_style("whitegrid")
ax = sns.violinplot(x=np.log(1+train.Target.values))
plt.show()


# In[61]:


yes_no_map = {'no':0,'yes':1}
train['dependency'] = train['dependency'].replace(yes_no_map).astype(np.float32)
train['edjefe'] = train['edjefe'].replace(yes_no_map).astype(np.float32)
train['edjefa'] = train['edjefa'].replace(yes_no_map).astype(np.float32)


# In[62]:


train.drop(['Id','idhogar',"dependency","edjefe","edjefa"], inplace = True, axis =1)

test.drop(['Id','idhogar',"dependency","edjefe","edjefa"], inplace = True, axis =1)


# In[63]:


y = train.iloc[:,137]
y.unique()


# In[64]:


X = train.iloc[:,1:138]
X.shape


# In[65]:


my_imputer = SimpleImputer()
X = my_imputer.fit_transform(X)
scale = ss()
X = scale.fit_transform(X)
pca = PCA(0.95)
X = pca.fit_transform(X)


# Splitting the data into train & test

# In[66]:


X_train, X_test, y_train, y_test = train_test_split(
                                                    X,
                                                    y,
                                                    test_size = 0.2)


# Random Forest

# In[67]:


modelrf = rf()


# In[68]:


import time
start = time.time()
modelrf = modelrf.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[69]:


classes = modelrf.predict(X_test)


# In[70]:


(classes == y_test).sum()/y_test.size 


# In[71]:


KNeighborsClassifier


# In[72]:


modelneigh = KNeighborsClassifier(n_neighbors=4)
start = time.time()
modelneigh = modelneigh.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[73]:


classes = modelneigh.predict(X_test)

classes
(classes == y_test).sum()/y_test.size 


# GradientBoostingClassifier

# In[74]:


modelgbm=gbm()
start = time.time()
modelgbm = modelgbm.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[75]:


classes = modelgbm.predict(X_test)

classes
(classes == y_test).sum()/y_test.size 


# Modelling with Light Gradient Booster

# In[76]:


modellgb = lgb.LGBMClassifier(max_depth=-1, learning_rate=0.1, objective='multiclass',
                             random_state=None, silent=True, metric='None', 
                             n_jobs=4, n_estimators=5000, class_weight='balanced',
                             colsample_bytree =  0.93, min_child_samples = 95, num_leaves = 14, subsample = 0.96)


# In[77]:


start = time.time()
modellgb = modellgb.fit(X_train, y_train)
end = time.time()
(end-start)/60


# In[78]:


classes = modellgb.predict(X_test)

classes
(classes == y_test).sum()/y_test.size 


# In[ ]:





# In[ ]:




