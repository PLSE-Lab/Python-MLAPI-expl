#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:



from fastai.imports import *
#from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn.model_selection import StratifiedKFold
import itertools
from sklearn.metrics import *
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


# In[ ]:


PATH = "../input"
get_ipython().system('ls {PATH}')


# In[ ]:


df_raw = pd.read_csv(f'{PATH}/X_train.csv', low_memory=False)
y_raw = pd.read_csv(f'{PATH}/y_train.csv', low_memory=False)
df_test = pd.read_csv(f'{PATH}/X_test.csv', low_memory=False)
sub = pd.read_csv(f'{PATH}/sample_submission.csv')


# In[ ]:


df_raw.shape,df_test.shape


# In[ ]:


#https://www.kaggle.com/prashantkikani/help-humanity-by-helping-robots

def fe(data):
    
    df = pd.DataFrame()
    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 +
                             data['angular_velocity_Z']**2)** 0.5
    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 +
                             data['linear_acceleration_Z'])**0.5
    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 +
                             data['orientation_Z'])**0.5
   
    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']
    
    for col in data.columns:
        if col in ['row_id','series_id','measurement_number']:
            continue
        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()
        df[col + '_median'] = data.groupby(['series_id'])[col].median()
        df[col + '_max'] = data.groupby(['series_id'])[col].max()
        df[col + '_min'] = data.groupby(['series_id'])[col].min()
        df[col + '_std'] = data.groupby(['series_id'])[col].std()
        df[col + '_range'] = df[col + '_max'] - df[col + '_min']
        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']
        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))
        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))
        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'df_train = fe(df_raw)\ndf_test = fe(df_test)')


# In[ ]:


df_train.shape, df_test.shape


# In[ ]:


df_train.head()


# In[ ]:


df_train.fillna(0, inplace = True)
df_test.fillna(0, inplace = True)
df_train.replace(-np.inf, 0, inplace = True)
df_train.replace(np.inf, 0, inplace = True)
df_test.replace(-np.inf, 0, inplace = True)
df_test.replace(np.inf, 0, inplace = True)


# In[ ]:


le = LabelEncoder()
y_raw['surface'] = le.fit_transform(y_raw['surface'])


# ### Look at the data

# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000): 
        with pd.option_context("display.max_columns", 1000): 
            display(df)


# In[ ]:


from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(df_train.surface, pred)
# cm


# In[ ]:


def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 0  
n_trn = len(df_train)-n_valid
#raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df_train, n_trn)
y_train, y_valid = split_vals(y_raw, n_trn)

X_train.shape, y_train.shape, X_valid.shape,df_train.shape,y_raw.shape


# In[ ]:


X_test = df_test


# In[ ]:


X_train.shape, y_train.shape, X_valid.shape, y_valid.shape,X_test.shape, df_.shape, y.shape


# In[ ]:


def create_download_link(title = "Download CSV file", filename = "data.csv"):  
    html = '<a href={filename}>{title}</a>'
    html = html.format(title=title,filename=filename)
    return HTML(html)


# ## Base model

# In[ ]:


#def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [ m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


# In[ ]:


def k_folds(X, y, X_test, k,n_est,min_leaf,max_feat):
    folds = StratifiedKFold(n_splits = k, shuffle=True, random_state=2019)
    y_test = np.zeros((X_test.shape[0], 9))
    y_oof = np.zeros((X.shape[0]))
    score = 0
    for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):
        clf =  RandomForestClassifier(n_estimators = n_est, n_jobs = -1,min_samples_leaf=min_leaf,max_features=max_feat)
        clf.fit(X_train.iloc[train_idx], y[train_idx])
        y_oof[val_idx] = clf.predict(X.iloc[val_idx])
        y_test += clf.predict_proba(X_test) / folds.n_splits
        score += clf.score(X.iloc[val_idx], y[val_idx])
        print('Fold: {} score: {}'.format(i,clf.score(X.iloc[val_idx], y[val_idx])))
    print('Avg Accuracy', score / folds.n_splits) 
        
    return y_oof, y_test 


# In[ ]:


y_oof, y_test = k_folds(X_train, y_raw['surface]', X_test, k= 50,n_est=500,min_leaf=1,max_feat='auto')


# In[ ]:


from IPython.display import HTML
y_test = np.argmax(y_test, axis=1)
submission = pd.read_csv(os.path.join("../input/",'sample_submission.csv'))
submission['surface'] = le.inverse_transform(y_test)
submission.to_csv('submission1.csv', index=False)
submission.head(10)
create_download_link(filename='submission1.csv')


# In[ ]:


y_oof, y_test = k_folds(X_train, y_raw['surface'], X_test, k= 50,n_est=500,min_leaf=1,max_feat='sqrt')


# In[ ]:


y_test = np.argmax(y_test, axis=1)
submission = pd.read_csv(os.path.join("../input/",'sample_submission.csv'))
submission['surface'] = le.inverse_transform(y_test)
submission.to_csv('submission2.csv', index=False)
submission.head(10)
create_download_link(filename='submission2.csv')


# In[ ]:


def k_folds_gbm(X, y, X_test, k,n_est):
    folds = StratifiedKFold(n_splits = k, shuffle=True, random_state=2019)
    y_test = np.zeros((X_test.shape[0], 9))
    y_oof = np.zeros((X.shape[0]))
    score = 0
    for i, (train_idx, val_idx) in  enumerate(folds.split(X, y)):
        clf =  XGBClassifier(n_estimators = n_est, n_jobs = -1,learning_rate = 0.2,objective='multi:softmax',eval_metric='merror')
        clf.fit(X_train.iloc[train_idx], y[train_idx])
        y_oof[val_idx] = clf.predict(X.iloc[val_idx])
        y_test += clf.predict_proba(X_test) / folds.n_splits
        score += clf.score(X.iloc[val_idx], y[val_idx])
        print('Fold: {} score: {}'.format(i,clf.score(X.iloc[val_idx], y[val_idx])))
    print('Avg Accuracy', score / folds.n_splits) 
        
    return y_oof, y_test 

from xgboost import XGBClassifier
# # fit model no training data
# m = XGBClassifier(random_stae=1)
# m.fit(X_train, y_train)
# print_score(m)


# In[ ]:


y_oof, y_test = k_folds_gbm(X_train, y_raw['surface'], X_test, k= 50,n_est=500)
# m = XGBClassifier(random_stae=1,learning_rate = 0.2, n_estimators=150,n_jobs=-1,objective='multi:softmax',eval_metric='merror')
# m.fit(df, y)
# m.score(df,y)


# In[ ]:


y_test = np.argmax(y_test, axis=1)
submission = pd.read_csv(os.path.join("../input/",'sample_submission.csv'))
submission['surface'] = le.inverse_transform(y_test)
submission.to_csv('submission3.csv', index=False)
submission.head(10)
create_download_link(filename='submission3.csv')


# In[ ]:




