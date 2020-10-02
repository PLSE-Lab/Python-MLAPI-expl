#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.imports import *


# In[ ]:


from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from IPython.display import display


# In[ ]:


os.getcwd()


# In[ ]:


os.listdir("/kaggle/working")


# In[ ]:


os.chdir("../")


# In[ ]:


os.makedirs("working/data")


# In[ ]:


os.listdir("working")


# In[ ]:


os.listdir("working/data")


# In[ ]:


train_file = "input/train/Train.csv"
test_file = "input/Test.csv"
valid_file = "input/Valid.csv"
req_files = [train_file,test_file,valid_file]
dest = "working/data"
def move_files(filename,destination):
    shutil.copy(filename,destination)
    
for i in req_files:
    move_files(i,dest)


# In[ ]:


os.chdir("working")


# In[ ]:


os.getcwd()


# In[ ]:


PATH = "data/"


# In[ ]:


df_raw = pd.read_csv(f"{PATH}Train.csv",low_memory=False,parse_dates=['saledate'])


# In[ ]:


df_raw.head()


# In[ ]:


df_raw.describe()


# In[ ]:


df_raw.info()


# In[ ]:


df_raw.SalePrice = np.log(df_raw.SalePrice)


# ## Pre-Processing

# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows",1000):
        with pd.option_context("display.max_columns",1000):
            display(df)


# In[ ]:


display_all(df_raw.describe(include='all').T)


# In[ ]:


add_datepart(df_raw,'saledate')
df_raw.columns


# In[ ]:


train_cats(df_raw)


# In[ ]:


df_raw.head()


# In[ ]:


df_raw.UsageBand.cat.set_categories(["High","Low","Medium"],ordered = True,inplace = True)
#df_raw.ProductSize.cat.set_categories(["High","Low","Medium"],ordered = True,inplace = True)


# In[ ]:


#Percentage of nulls in each column
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))


# In[ ]:


df_raw.UsageBand.cat.categories


# In[ ]:


os.makedirs('tmp',exist_ok=True)
df_raw.to_feather('tmp/raw')


# In[ ]:


df_raw = pd.read_feather('tmp/raw')


# In[ ]:


df,y,nas = proc_df(df_raw,'SalePrice')


# In[ ]:


m = RandomForestRegressor(n_jobs = -1)
m.fit(df,y)
m.score(df,y)


# ## Train Test Validation split

# In[ ]:


def split_vals(a,n): return a[:n].copy(),a[n:].copy()
n_valid = 12000
n_train = len(df) - n_valid
raw_train,raw_valid = split_vals(df_raw,n_train)
X_train,X_valid = split_vals(df,n_train)
y_train,y_valid = split_vals(y,n_train)
X_train.shape,y_train.shape,X_valid.shape


# In[ ]:


def rmse(x,y):return math.sqrt(((x-y)**2).mean())
def print_score(m):
    res = [rmse(m.predict(X_train),y_train),rmse(m.predict(X_valid),y_valid),
           m.score(X_train,y_train),m.score(X_valid,y_valid)]
    if hasattr(m,"oob_score_"): res.append(m.oob_score_)
    print(res)


# In[ ]:


m = RandomForestRegressor(n_jobs = -1)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# ## Faster computation

# In[ ]:


df_trn,y_trn, nas = proc_df(df_raw,"SalePrice",subset=30000,na_dict=nas)
X_train,_ = split_vals(df_trn,20000)
y_train,_ = split_vals(y_trn,20000)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1,n_estimators=1,max_depth=3,bootstrap=False)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


draw_tree(m.estimators_[0],df_trn,precision=3)


# ## Bagging

# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0],np.mean(preds[:,0]),y_valid[0]


# In[ ]:


plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)])


# In[ ]:


m = RandomForestRegressor(n_jobs=-1,n_estimators=1)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1,n_estimators=20)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1,n_estimators=80)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1,n_estimators=100,bootstrap=True)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators=80, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# ## Subsampling to reduce overfitting

# In[ ]:


df_trn,y_trn,nas = proc_df(df_raw,"SalePrice")
X_train,X_valid = split_vals(df_trn,n_train)
y_train,y_valid = split_vals(y,n_train)


# In[ ]:


set_rf_samples(20000)       ## Subsampling to get nrows from the training set for each tree


# In[ ]:


m = RandomForestRegressor(n_estimators= 40 , n_jobs = -1,oob_score=True)
m.fit(X_train,y_train)
print_score(m)


# ## Fiddling with other hyperparameters

# In[ ]:


reset_rf_samples()


# In[ ]:


m = RandomForestRegressor(n_estimators= 40 , n_jobs = -1,oob_score=True)
m.fit(X_train,y_train)
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators= 40 ,min_samples_leaf=3, n_jobs = -1,oob_score=True)
m.fit(X_train,y_train)
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators= 40 ,min_samples_leaf=3,max_features= 0.5, n_jobs = -1,oob_score=True)
m.fit(X_train,y_train)
print_score(m)


# ## Confidence Intervals

# In[ ]:


get_ipython().run_line_magic('time', 'preds = np.stack([t.predict(X_valid) for t in m.estimators_])')


# In[ ]:


preds[0:5]


# In[ ]:


np.mean(preds[:,0]),np.std(preds[:,0])


# In[ ]:


#### Parallel trees

def get_preds(t): return t.predict(X_valid)
get_ipython().run_line_magic('time', 'preds = np.stack(parallel_trees(m,get_preds))')
np.mean(preds[:,0]),np.std(preds[:,0])


# In[ ]:


x = raw_valid.copy()
x.Enclosure.value_counts()


# In[ ]:


x = raw_valid.copy()
x["pred"] = np.mean(preds,axis = 0)
x["pred_std"] = np.std(preds,axis = 0)
x.Enclosure.value_counts().plot.barh()


# In[ ]:


cols = ['Enclosure', 'SalePrice','pred', 'pred_std' ]
enc_summ = x[cols].groupby('Enclosure',as_index = False).mean()
enc_summ


# In[ ]:


enc_summ = enc_summ[~pd.isnull(enc_summ.SalePrice)]
enc_summ.plot("Enclosure", "SalePrice", 'barh',xlim = (0,11))


# In[ ]:


enc_summ.plot("Enclosure", "pred", 'barh',xlim = (0,11),xerr = "pred_std" )


# ## Feature importance

# In[ ]:


fi = rf_feat_importance(m,df_trn);fi[:10]


# In[ ]:


fi.plot('cols','imp',legend = False, figsize= (12,6))


# In[ ]:


def plot_fi(fi): return fi.plot('cols','imp','barh',figsize = (12,7))
plot_fi(fi[:30])


# In[ ]:


keep_cols = fi[fi.imp > 0.005].cols
keep_cols


# In[ ]:


df_keep = df_trn[keep_cols].copy()
X_train,X_valid = split_vals(df_keep,n_train)


# In[ ]:


m = RandomForestRegressor(n_estimators= 40 ,min_samples_leaf=3,max_features= 0.5, n_jobs = -1,oob_score=True)
m.fit(X_train,y_train)
print_score(m)


# In[ ]:


fi = rf_feat_importance(m,df_keep)
plot_fi(fi[:30])


# ## One Hot Encoding

# In[ ]:


df_trn2,y_trn,nas = proc_df(df_raw,"SalePrice" , max_n_cat= 7)
X_train,X_valid = split_vals(df_trn2,n_train)
y_train,y_valid = split_vals(y_trn,n_train)


# In[ ]:


X_valid.shape,y_valid.shape


# In[ ]:


set_rf_samples(20000)


# In[ ]:


m = RandomForestRegressor(n_estimators= 40 ,min_samples_leaf=3,max_features= 0.5, n_jobs = -1,oob_score=True)
m.fit(X_train,y_train)
print_score(m)


# In[ ]:


display_all(X_train.head())


# In[ ]:


#### Feature importance after one hot encoding
fi = rf_feat_importance(m,df_trn2)
fi[:10]


# In[ ]:


plot_fi(fi[:25])


# ## Clustering to remove redundant features

# In[ ]:


from scipy.cluster import hierarchy as hc


# In[ ]:


corr = np.round(scipy.stats.spearmanr(df_keep).correlation,4)
corr


# In[ ]:


corr_condensed = hc.distance.squareform(1-corr)
corr_condensed


# In[ ]:


z = hc.linkage(corr_condensed,method = "average")
z


# In[ ]:


fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z,labels=df_keep.columns,orientation="left",leaf_font_size=16)
plt.show()


# In[ ]:


def get_oob(df):
    x,_ = split_vals(df,n_train)
    m = RandomForestRegressor(n_estimators=40,max_features=0.5,min_samples_leaf=3,n_jobs=-1,
                              oob_score=True)
    m.fit(x,y_train)
    return m.oob_score_


# In[ ]:


get_oob(df_keep)


# In[ ]:


for c in ('saleYear', 'saleElapsed', 'Grouser_Tracks', 'Hydraulics_Flow', 'Coupler_System',
          'ProductGroupDesc', 'fiProductClassDesc','fiBaseModel', 'fiModelDesc'):
    print (c,get_oob(df_keep.drop(c,axis = 1)))


# In[ ]:


###Simpler model
to_drop = ['saleYear', 'fiBaseModel', 'Grouser_Tracks']
get_oob(df_keep.drop(to_drop,axis = 1))


# In[ ]:


df_keep.drop(to_drop,axis = 1, inplace=True)
X_train,X_valid = split_vals(df_keep,n_train)


# In[ ]:


reset_rf_samples()


# In[ ]:


m = RandomForestRegressor(n_estimators=40,max_features=0.5,min_samples_leaf=3,n_jobs=-1,
                              oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# ## More fine tuning with Partial Dependence Plot

# In[ ]:


set_rf_samples(50000)


# In[ ]:


df_trn2,y_trn,nas = proc_df(df_raw,"SalePrice",max_n_cat=7)
X_train,X_valid = split_vals(df_trn2,n_train)
m = RandomForestRegressor(n_estimators=40,max_features=0.5,min_samples_leaf=3,n_jobs=-1,
                              oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


plot_fi(rf_feat_importance(m,df_trn2)[:10])


# In[ ]:


df_trn2.plot('YearMade', 'saleElapsed','scatter',figsize=(10,8),alpha = 0.01)


# In[ ]:


x_all = get_sample(df_trn2[df_trn2["YearMade"] > 1930],500)


# In[ ]:


x = get_sample(X_train[X_train.YearMade > 1930],500)


# In[ ]:


from pdpbox import pdp


# In[ ]:


from plotnine import *


# In[ ]:


mod_features = x.columns


# In[ ]:


def pdp_plot(feat):
    p = pdp.pdp_isolate(
        model=m, dataset=x, model_features=mod_features, feature= feat
    )
    return pdp.pdp_plot(p, feat, plot_lines=True, frac_to_plot=100)


# In[ ]:


pdp_plot("YearMade")


# In[ ]:


f = ['saleElapsed','YearMade']
p = pdp.pdp_interact(model=m,features=f,dataset=x,model_features=mod_features)
pdp.pdp_interact_plot(pdp_interact_out=p,feature_names=f)


# In[ ]:




