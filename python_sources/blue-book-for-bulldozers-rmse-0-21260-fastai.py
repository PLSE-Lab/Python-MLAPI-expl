#!/usr/bin/env python
# coding: utf-8

# I will writting the description shortly. 

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


PATH = '../input/Train/Train.csv'


# In[ ]:


get_ipython().system('ls {PATH}')


# In[ ]:


df_train = pd.read_csv(f'{PATH}',low_memory=False,parse_dates=["saledate"])


# In[ ]:


df_train.head()


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows",1000):
        with pd.option_context("display.max_columns",1000):
            display(df)
    


# In[ ]:


display_all(df_train.tail())


# In[ ]:


display_all(df_train.describe(include='all').transpose())


# In[ ]:


display_all(df_train.columns)


# In[ ]:


df_train.SalePrice = np.log(df_train.SalePrice)


# In[ ]:


fld=df_train.saledate


# 
# The columns saledate is converted to the following columns with value's split.
# 
# Example: saledate	2011-11-02 00:00:00
# 
#        'saleYear', 'saleMonth',
#        'saleWeek', 'saleDay', 'saleDayofweek', 'saleDayofyear',
#        'saleIs_month_end', 'saleIs_month_start', 'saleIs_quarter_end',
#        'saleIs_quarter_start', 'saleIs_year_end', 'saleIs_year_start',
#        'saleElapsed'
#        
# The following method extracts particular date fields from a complete datetime for the purpose of constructing categoricals. You should always consider this feature extraction step when working with date-time. Without expanding your date-time into these additional fields, you can't capture any trend/cyclical behavior as a function of time at any of these granularities.       

# In[1]:


add_datepart(df_train, 'saledate')
df_train.saleYear.head()


# In[ ]:


df_train.columns


# Turning other columns into categorical 

# In[ ]:


train_cats(df_train)


# In[ ]:


df_train.UsageBand.cat.categories


# In[ ]:


df_train.UsageBand.cat.codes


# In[ ]:


df_train.UsageBand.cat.set_categories(['High', 'Low', 'Medium'],ordered=True,inplace=True)


# In[ ]:


display_all(df_train.isnull().sum().sort_index()/len(df_train))


# In[ ]:


os.makedirs('tmp',exist_ok=True)
df_train.to_feather('tmp/raw')


# In[ ]:


df_train=pd.read_feather('tmp/raw')


# In[ ]:


df, y, nas = proc_df(df_train,'SalePrice')


# In[ ]:


df.columns


# In[ ]:


m = RandomForestRegressor(n_jobs=-1)
m.fit(df,y)
m.score(df,y)


# In[ ]:


def split_vals(a,n):
    return a[:n].copy(),a[n:].copy()

n_valid = 12000
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_train, n_trn)
X_train, X_valid = split_vals(df,n_trn)
y_train, y_valid = split_vals(y,n_trn)
X_train.shape, y_train.shape, X_valid.shape


# In[ ]:


def rmse(x,y):
    return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res =[rmse(m.predict(X_train),y_train),rmse(m.predict(X_valid),y_valid),
             m.score(X_train,y_train),m.score(X_valid,y_valid)]
    if hasattr(m, 'oob_score_'):res.append(m.oob_score_)
    print(res)


# In[ ]:


m=RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


df_trn, y_trn, nas = proc_df(df_train,'SalePrice',subset=30000)
X_train,_ = split_vals(df_trn,20000)
y_train,_ = split_vals(y_trn,20000)


# In[ ]:


m=RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators=1,max_depth=3,bootstrap=False,n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators=1,bootstrap=False,n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


#bagging - bag of little bootstrap - 
m = RandomForestRegressor(n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds


# In[ ]:


y_valid


# In[ ]:


plt.plot([metrics.r2_score(y_valid,np.mean(preds[:i+1],axis=0)) for i in range(20)]);


# In[ ]:


m = RandomForestRegressor(n_estimators=40,n_jobs=-1)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


df_trn, y_trn, nas = proc_df(df_train,'SalePrice')
X_train,X_valid = split_vals(df_trn,n_trn)
y_train,y_valid = split_vals(y_trn,n_trn)


# In[ ]:


set_rf_samples(20000)


# In[ ]:


m = RandomForestRegressor(n_jobs=-1,oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators=40,min_samples_leaf=3,n_jobs=-1,oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


m = RandomForestRegressor(n_estimators=40,min_samples_leaf=3,max_features=0.5,n_jobs=-1,oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:


set_rf_samples(50000)
m = RandomForestRegressor(n_estimators=40,min_samples_leaf=3,max_features=0.5,n_jobs=-1,oob_score=True)
m.fit(X_train,y_train)
print_score(m)


# In[ ]:


def get_preds(t): 
    return t.predict(X_valid)
get_ipython().run_line_magic('time', 'preds = np.stack(parallel_trees(m, get_preds))')
np.mean(preds[:,0]), np.std(preds[:,0])


# In[ ]:


X = raw_valid.copy()
X['pred_std'] = np.std(preds, axis=0)
X['pred'] = np.mean(preds,axis=0)
X.Enclosure.value_counts().plot.barh();


# In[ ]:


flds = ['Enclosure','SalePrice','pred','pred_std']
enc_summ=X[flds].groupby('Enclosure',as_index=False).mean()
enc_summ


# In[ ]:


enc_summ = enc_summ[~pd.isnull(enc_summ.SalePrice)]
enc_summ.plot('Enclosure', 'SalePrice', 'barh', xlim=(0,11));


# In[ ]:


enc_summ.plot('Enclosure', 'pred', 'barh', xerr='pred_std', 
              alpha=0.6, xlim=(0,11));


# In[ ]:


raw_valid.ProductSize.value_counts()


# In[ ]:


raw_valid.ProductSize


# In[ ]:


raw_valid.ProductSize.value_counts().plot.barh()


# In[ ]:


flds = ['ProductSize','SalePrice','pred','pred_std']
summ = X[flds].groupby('ProductSize',as_index=False).mean()
summ


# In[ ]:


#(summ.pred_std/summ.pred).sort_values(ascending=False)
(summ.pred_std/summ.pred)


# Feature Importance : 
# 
# It gives us the columns which carries more weightage in predicting the result. In other words, it gives us important features to carry out prediction. 
# 
# It's not normally enough to just to know that a model can make accurate predictions - we also want to know how it's making predictions. The most important way to see this is with feature importance.

# In[ ]:


fi = rf_feat_importance(m,df_trn)
fi


# In[ ]:


fi.plot('cols','imp','barh',figsize=(10,60),legend=False)


# In[ ]:


to_keep = fi[fi.imp>0.005].cols; len(to_keep)
df_keep = df_trn[to_keep].copy()
(df_keep.columns)
X_train,X_valid = split_vals(df_keep,n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, 
                       max_features=0.5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


fi = rf_feat_importance(m,df_keep)
fi.plot('cols','imp','barh',figsize=(10,6),legend=False)


# In[ ]:


df_trn2, y_trn, nas = proc_df(df_train, 'SalePrice',max_n_cat=7)
X_train, X_valid = split_vals(df_trn2,n_trn)

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, 
                       max_features=0.5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


fi = rf_feat_importance(m, df_trn2)
#fi.plot('cols','imp','barh',figsize=(10,6),legend=False)[:50]
fi[:50]


# In[ ]:


from scipy.cluster import hierarchy as hc


# In[ ]:


corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()


# In[ ]:


def get_oob(df):
    m = RandomForestRegressor(n_estimators=30, min_samples_leaf=5, max_features=0.6, n_jobs=-1, oob_score=True)
    X, _ = split_vals(df, n_trn)
    m.fit(X,y_train)
    return m.oob_score_


# In[ ]:


get_oob(df_keep)


# In[ ]:


for c in ('saleYear', 'saleElapsed', 'fiModelDesc', 'fiBaseModel', 'Grouser_Tracks', 'Coupler_System'):
    print(c ,get_oob(df_keep.drop(c,axis=1)))


# In[ ]:


to_drop = ['saleYear', 'fiBaseModel', 'Grouser_Tracks']
get_oob(df_keep.drop(to_drop,axis=1))


# In[ ]:


df_keep.drop(to_drop,axis=1,inplace=True)
X_train, X_valid = split_vals(df_keep,n_trn)


# In[ ]:


np.save('tmp/keep_cols.npy', np.array(df_keep.columns))


# In[ ]:


keep_cols = np.load('tmp/keep_cols.npy')
df_keep = df_trn[keep_cols]


# In[ ]:


reset_rf_samples()


# In[ ]:


m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, 
                       max_features=0.5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train,y_train)')
print_score(m)


# In[ ]:


#from pdpbox import pdp
#from plotnine import *


# In[ ]:


set_rf_samples(50000)
df_trn2, y_trn, nas = proc_df(df_train, 'SalePrice', max_n_cat=7)
X_train, X_valid = split_vals(df_trn2, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train);


# In[ ]:


def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)


# In[ ]:


plot_fi(rf_feat_importance(m,df_trn2))


# In[ ]:


df_train.plot('YearMade', 'saleElapsed', 'scatter', alpha=0.01, figsize=(10,8));


# In[ ]:


x_all = get_sample(df_train[df_train.YearMade>1930],500)


# In[ ]:


x = get_sample(X_train[X_train.YearMade>1930],500)


# In[ ]:


#def plot_pdp(feat, clusters=None, feat_name=None):
#    feat_name = feat_name or feat
#    p = pdp.pdp_isolate(m, x, feat)
#    return pdp.pdp_plot(p, feat_name, plot_lines=True, 
#                        cluster=clusters is not None, n_cluster_centers=clusters)


# In[ ]:


#plot_pdp('YearMade')


# In[ ]:


#plot_pdp('YearMade',clusters=5)


# In[ ]:


#feats = ['saleElapsed', 'YearMade']
#p = pdp.pdp_interact(m, x, feat)
#pdp.pdp_interact_plots(p,feats)


# In[ ]:


#plot_pdp(['Enclosure_EROPS w AC', 'Enclosure_EROPS', 'Enclosure_OROPS'], 5, 'Enclosure')


# In[ ]:


df_train.YearMade[df_train.YearMade<1950] = 1950
df_keep['age'] = df_train['age'] = df_train.saleYear-df_train.YearMade


# In[ ]:


X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.6, n_jobs=-1)
m.fit(X_train, y_train)
plot_fi(rf_feat_importance(m, df_keep));


# In[ ]:


#from treeinterpreter import treeinterpreter as ti


# In[ ]:


df_train, df_valid = split_vals(df_train[df_keep.columns],n_trn)


# In[ ]:


row = X_valid.values[None,0]; row


# In[ ]:


#prediction, bias, contributions = ti.predict(m, row)


# In[ ]:


#prediction[0],bias[0]


# In[ ]:


#idxs = np.argsort(contributions[0])


# In[ ]:


#[o for o in zip(df_keep.columns[idxs], df_valid.iloc[0][idxs], contributions[0][idxs])]


# In[ ]:


#contributions[0].sum()


# In[ ]:


df_ext = df_keep.copy()
df_ext['is_valid']=1
df_ext.is_valid[:n_trn] = 0
x, y , nas = proc_df(df_ext,'is_valid') 


# In[ ]:


m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_


# In[ ]:


fi = rf_feat_importance(m,x)
fi[:10]


# In[ ]:


feats = ['SalesID', 'saleElapsed', 'MachineID']


# In[ ]:


x.drop(feats,axis=1,inplace=True)


# In[ ]:


m = RandomForestClassifier(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(x, y);
m.oob_score_


# In[ ]:


fi = rf_feat_importance(m,x)
fi[:10]


# In[ ]:


set_rf_samples(50000)


# In[ ]:


feats=['SalesID', 'saleElapsed', 'MachineID', 'age', 'YearMade', 'saleDayofyear']


# In[ ]:


X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


for f in feats:
    df_subs = df_keep.drop(f, axis=1)
    X_train, X_valid = split_vals(df_subs, n_trn)
    m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
    m.fit(X_train, y_train)
    print(f)
    print_score(m)


# In[ ]:


reset_rf_samples()
df_subs = df_keep.drop(['SalesID', 'MachineID', 'saleDayofyear'], axis=1)
X_train, X_valid = split_vals(df_subs, n_trn)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)


# In[ ]:


plot_fi(rf_feat_importance(m,X_train))


# In[ ]:


np.save('tmp/subs_cols.npy', np.array(df_subs.columns))


# In[ ]:


#final model

m = RandomForestRegressor(n_estimators=40, max_features=0.5, n_jobs=-1, oob_score=True)
get_ipython().run_line_magic('time', 'm.fit(X_train, y_train)')
print_score(m)


# In[ ]:





# In[ ]:





# In[ ]:




