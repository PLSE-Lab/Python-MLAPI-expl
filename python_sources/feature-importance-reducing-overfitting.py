#!/usr/bin/env python
# coding: utf-8

# ## FastAi Approach 
# This notebook will guide you through FastAI approach.
# 
# We'll use RandomForest to  find out feature important and find Partial dependence. Also we'll plot the correlation between variables.
# 
# ### Imports

# In[ ]:


get_ipython().system('pip install fastai==0.7.0')


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.imports import *
from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
from IPython.display import display


# In[ ]:


PATH = '../input/'


# In[ ]:


df_raw = pd.read_csv(PATH + 'train.csv',low_memory=False)


# In[ ]:


df_test = pd.read_csv(PATH + 'test.csv',low_memory=False)


# In[ ]:


df_raw.head(5)


# In[ ]:


df_raw.describe()


# In[ ]:


#check the missing value
df_raw.isnull().values.any()


# There are no missing values and also we have checked the mean and Standard Deviation of Target Column
# 
# Now let's use proc_df function which will handle categorical data and convert into numeric data
# 
# Note - There are no categorical data and no missing values so even if we dont do proc_df then it won't matter at all. Here proc df will just split the data into into X and Y(Predictor and Target)

# In[ ]:


df_trn, y_trn, nas = proc_df(df_raw, 'target')


# In[ ]:


def split_vals(a,n): return a[:n], a[n:]
n_valid = 30
n_trn = len(df_trn)-n_valid
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
raw_train, raw_valid = split_vals(df_raw, n_trn)


# In[ ]:


from sklearn.metrics import roc_auc_score

def auc(x,y): return roc_auc_score(x, y)#x - y_true, y = y_score

def print_score(m):
    res = [auc(y_train, m.predict(X_train)), auc(y_valid, m.predict(X_valid)),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


#This is definately overfitting
m = RandomForestRegressor(n_estimators=1000, min_samples_leaf=5, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


#a better fit
m = RandomForestRegressor(n_estimators=1000, min_samples_leaf=25, max_features=0.6, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# Let's try cross validation

# In[ ]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(m, df_trn, y_trn, cv=5, scoring='roc_auc')
scores


# So we are getting 80% AUC on crossval, let's try feature importance and then check cross val again
# ## Feature importance
# 

# In[ ]:


fi = rf_feat_importance(m, df_trn);


# In[ ]:


#top 30 features are
fi[:30]


# In[ ]:


fi.plot('cols', 'imp', figsize=(10,6), legend=False);


# In[ ]:


def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[ ]:


plot_fi(fi[:30]);


# We'll keep all those varialbes which are above 
# * 0.001
# * 0.005

# In[ ]:


to_keep = fi[fi.imp>0.005].cols; 
len_tokeep = len(to_keep)


# In[ ]:


df_keep = df_trn[to_keep].copy()
X_train, X_valid = split_vals(df_keep, 250)


# In[ ]:


X_train.shape


# In[ ]:


m = RandomForestRegressor(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1, oob_score=True)
scores = cross_val_score(m, X_train, y_trn, cv=5, scoring='roc_auc')
scores


# Here we can see that we are getting 87.5 AUC on whole set using crossval
# ### Now let's try out prediction on this

# In[ ]:


m.fit(X_train, y_trn)


# In[ ]:


df_keep = df_test[to_keep].copy()


# In[ ]:


df_keep.shape


# In[ ]:


y_preds = m.predict(df_keep)


# In[ ]:


y_preds


# In[ ]:


submission_rf = pd.read_csv(PATH + 'sample_submission.csv')


# In[ ]:


submission_rf['target'] = y_preds


# In[ ]:


submission_rf.to_csv('submission_0.005.csv', index=False)


# 
# ### Now let's try with feature importance > 0.001

# In[ ]:


to_keep = fi[fi.imp>0.001].cols; 
len(to_keep)


# In[ ]:


df_keep = df_trn[to_keep].copy()
X_train, X_valid = split_vals(df_keep, 250)


# In[ ]:


m = RandomForestRegressor(n_estimators=100, min_samples_leaf=25, max_features=0.5, n_jobs=-1, oob_score=True)
scores = cross_val_score(m, X_train, y_trn, cv=5, scoring='roc_auc')
scores


# ### We can see that  features between 0.001-0.005 are not giving great accuracy

# In[ ]:


m.fit(X_train, y_trn)
df_keep = df_test[to_keep].copy()
y_preds = m.predict(df_keep)
submission_rf['target'] = y_preds
submission_rf.to_csv('submission_0.001.csv', index=False)


# ## Let's try removing redundant features

# In[ ]:


to_keep = fi[fi.imp>0.005].cols; 
df_keep = df_trn[to_keep].copy()
len_tokeep = len(to_keep)


# In[ ]:


from scipy.cluster import hierarchy as hc


# In[ ]:


corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()


# We can clearly see that there are no features which are highly correlated with other, so don;t remove anything

# ## Thank you
# 
# Please upvote the kernel if you liked it
# 
# Connect with me on - https://www.linkedin.com/in/savannahar/
# 
# Share important tips / links / resouces in comment section because sharing is caring.
