#!/usr/bin/env python
# coding: utf-8

# In[24]:


import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
import lightgbm as lgb
sns.set()


# In[25]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# PATH="../input/Santander/" 
PATH="../input/" 
print(os.listdir(PATH))
# Any results you write to the current directory are saved as output.


# In[26]:


get_ipython().run_cell_magic('time', '', 'X_train_df = pd.read_csv(PATH+"X_train.csv")\nX_test_df = pd.read_csv(PATH+"X_test.csv")\nY_train_df = pd.read_csv(PATH+"y_train.csv")\nsub = pd.read_csv(PATH+"sample_submission.csv")')


# Credits: 
# - https://www.kaggle.com/jsaguiar/surface-recognition-baseline
# - https://www.kaggle.com/gpreda/santander-eda-and-prediction

# # Data exploration

# Let's check the train and test set.

# In[27]:


X_train_df.shape, Y_train_df.shape , X_test_df.shape


# #### X_train_df have 487680 rows with 13 columns. Y_train_df have 3810 rows with 3 columns. X_test_df have 488448 rows with 13 columns. 
# 
# X_train_df and X_test_df contain:
# - series_id
# - measurement_number
# - 10 numerical variables named: orientation_X,	orientation_Y,	orientation_Z,	orientation_W,	angular_velocity_X,	angular_velocity_Y,	angular_velocity_Z,	linear_acceleration_X,	linear_acceleration_Y and linear_acceleration_Z
# 
# Y_train_df contain:
# - series_id	
# - group_id	
# - surface
# 

# In[28]:


X_train_df.head()


# In[29]:


Y_train_df.head()


# In[30]:


X_test_df.head()


# Check missing data

# In[31]:


def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


# In[32]:


missing_data(X_train_df)


# In[33]:


missing_data(X_test_df)


# In[34]:


missing_data(Y_train_df)


# There are no missing data train and test  datasets.

# In[35]:


X_train_df.describe()


# In[36]:


X_test_df.describe()


# #### Let's check the distribution of surface(target) value in train(Y_train_df) dataset.

# In[37]:


plt.figure(figsize=(10,6))
plt.title("Target labels")
sns.countplot(y='surface', data = Y_train_df, order = Y_train_df['surface'].value_counts().index,  palette="Set2")
plt.show()


# The data is unbalanced! 

# # Feature extraction

# Let's extract 10 features from time series
# 
# We will use these papers  [link 1](https://ieeexplore.ieee.org/document/8181558) [link 2](https://www.sciencedirect.com/science/article/pii/S2405896318323127). We will use the top 10 features to vibration signal. 

# In[38]:


from scipy.stats import kurtosis
from scipy.stats import skew

def _kurtosis(x):
    return kurtosis(x)

def CPT5(x):
    den = len(x)*np.exp(np.std(x))
    return sum(np.exp(x))/den

def skewness(x):
    return skew(x)

def SSC(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x,x[1])
    xn = x[1:len(x)-1]
    xn_i2 = x[2:len(x)]    # xn+1 
    xn_i1 = x[0:len(x)-2]  # xn-1
    ans = np.heaviside((xn-xn_i1)*(xn-xn_i2),0)
    return sum(ans[1:]) 

def wave_length(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x,x[1])
    xn = x[1:len(x)-1]
    xn_i2 = x[2:len(x)]    # xn+1 
    return sum(abs(xn_i2-xn))
    
def norm_entropy(x):
    tresh = 3
    return sum(np.power(abs(x),tresh))

def SRAV(x):    
    SRA = sum(np.sqrt(abs(x)))
    return np.power(SRA/len(x),2)

def mean_abs(x):
    return sum(abs(x))/len(x)

def zero_crossing(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x,x[1])
    xn = x[1:len(x)-1]
    xn_i2 = x[2:len(x)]    # xn+1
    return sum(np.heaviside(-xn*xn_i2,0))


# In[39]:


def feature_extraction(raw_frame):
    frame = pd.DataFrame()
    raw_frame['angular_velocity'] = raw_frame['angular_velocity_X'] + raw_frame['angular_velocity_Y'] + raw_frame['angular_velocity_Z']
    raw_frame['linear_acceleration'] = raw_frame['linear_acceleration_X'] + raw_frame['linear_acceleration_Y'] + raw_frame['linear_acceleration_Z']
    raw_frame['velocity_to_acceleration'] = raw_frame['angular_velocity'] / raw_frame['linear_acceleration']
    
    for col in raw_frame.columns[3:]:
        frame[col + '_mean'] = raw_frame.groupby(['series_id'])[col].mean()        
        frame[col + '_CPT5'] = raw_frame.groupby(['series_id'])[col].apply(CPT5) 
        frame[col + '_SSC'] = raw_frame.groupby(['series_id'])[col].apply(SSC) 
        frame[col + '_skewness'] = raw_frame.groupby(['series_id'])[col].apply(skewness)
        frame[col + '_wave_lenght'] = raw_frame.groupby(['series_id'])[col].apply(wave_length)
        frame[col + '_norm_entropy'] = raw_frame.groupby(['series_id'])[col].apply(norm_entropy)
        frame[col + '_SRAV'] = raw_frame.groupby(['series_id'])[col].apply(SRAV)
        frame[col + '_kurtosis'] = raw_frame.groupby(['series_id'])[col].apply(_kurtosis) 
        frame[col + '_mean_abs'] = raw_frame.groupby(['series_id'])[col].apply(mean_abs) 
        frame[col + '_zero_crossing'] = raw_frame.groupby(['series_id'])[col].apply(zero_crossing) 
    return frame


# In[40]:


train_df = feature_extraction(X_train_df)
test_df = feature_extraction(X_test_df)


# In[41]:


train_df.head()


# In[42]:


train_df.shape, test_df.shape


# Fix -inf, +inf and NaN

# In[43]:



train_df.fillna(0, inplace = True)
train_df.replace(-np.inf, 0, inplace = True)
train_df.replace(np.inf, 0, inplace = True)
test_df.fillna(0, inplace = True)
test_df.replace(-np.inf, 0, inplace = True)
test_df.replace(np.inf, 0, inplace = True)


# Standarization

# In[21]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(train_df))
X_test_scaled = pd.DataFrame(scaler.transform(test_df))


# In[ ]:





# # LightGBM classifier 

# In[22]:


le = LabelEncoder()
target = le.fit_transform(Y_train_df['surface'])


# In[45]:


params = {
    'num_leaves': 54,
    'min_data_in_leaf': 40,
    'objective': 'multiclass',
    'max_depth': 8,
    'learning_rate': 0.01,
    "boosting": "gbdt",
    "bagging_freq": 5,
    "bagging_fraction": 0.8126672064208567,
    "bagging_seed": 11,
    "verbosity": -1,
    'reg_alpha': 0.1302650970728192,
    'reg_lambda': 0.3603427518866501,
    "num_class": 9,
    'nthread': -1
}

def multiclass_accuracy(preds, train_data):
    labels = train_data.get_label()
    pred_class = np.argmax(preds.reshape(9, -1).T, axis=1)
    return 'multi_accuracy', np.mean(labels == pred_class), True

t0 = time.time()
train_set = lgb.Dataset(X_train_scaled, label=target)
eval_hist = lgb.cv(params, train_set, nfold=10, num_boost_round=9999,
                   early_stopping_rounds=100, seed=19, feval=multiclass_accuracy)
num_rounds = len(eval_hist['multi_logloss-mean'])

# retrain the model and make predictions for test set
clf = lgb.train(params, train_set, num_boost_round=num_rounds)
predictions = clf.predict(X_test_scaled, num_iteration=None)
print("Timer: {:.1f}s".format(time.time() - t0))


# The following plots show the mean logloss and accuracy at each iteration (blue line). The red lines are the standard deviation between folds.

# In[46]:


v1, v2 = eval_hist['multi_logloss-mean'][-1], eval_hist['multi_accuracy-mean'][-1]
print("Validation logloss: {:.4f}, accuracy: {:.4f}".format(v1, v2))
plt.figure(figsize=(10, 4))
plt.title("CV multiclass logloss")
num_rounds = len(eval_hist['multi_logloss-mean'])
ax = sns.lineplot(x=range(num_rounds), y=eval_hist['multi_logloss-mean'])
ax2 = ax.twinx()
p = sns.lineplot(x=range(num_rounds), y=eval_hist['multi_logloss-stdv'], ax=ax2, color='r')

plt.figure(figsize=(10, 4))
plt.title("CV multiclass accuracy")
num_rounds = len(eval_hist['multi_accuracy-mean'])
ax = sns.lineplot(x=range(num_rounds), y=eval_hist['multi_accuracy-mean'])
ax2 = ax.twinx()
p = sns.lineplot(x=range(num_rounds), y=eval_hist['multi_accuracy-stdv'], ax=ax2, color='r') 


# ### LGBM submission

# In[49]:


sub['surface'] = le.inverse_transform(predictions.argmax(axis=1))
sub.to_csv('submission_lgbm.csv', index=False)


# # Random forest

# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_train_scaled, target, test_size=0.2, random_state=23, stratify=target)


# In[58]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Number of optimal trees
# -You can test other values
optimal_k = 120 
rf_acc = []

# Optimal trees
rf_clf = RandomForestClassifier(n_estimators=optimal_k, random_state=0)        
# Train data split
rf_clf.fit(X_train, Y_train)

# Accuracy test data split
Y_pred = rf_clf.predict(X_test)
acc = accuracy_score(Y_pred,Y_test)

print('Acc_Test:')
print(float("%0.3f" % (100*acc)))


# In[59]:


#With original data
rf_clf.fit(X_train_scaled, target)


# # RF submission

# In[60]:


Y_test_sub = rf_clf.predict(X_test_scaled)
sub['surface'] = le.inverse_transform(Y_test_sub)
sub.to_csv('submission_rf.csv', index=False)


# In[ ]:




