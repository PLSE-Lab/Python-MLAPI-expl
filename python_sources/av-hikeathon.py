#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')
from fastai.collab import *
from fastai.tabular import *
from fastai.imports import *
#from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegressionCV
#from fastai.column_data import *


# In[ ]:


PATH = "../input"
get_ipython().system('ls {PATH}')


# In[ ]:


path = '../working'
get_ipython().system('ls {path}')


# In[ ]:


dtype = {'node1_id':'int32','node2_id':'int32','is_chat':'int8','node_id':'int32','id':'int32','f1':'int8','f2':'int8','f3':'int8',
         'f4':'int8','f5':'int8','f6':'int8','f7':'int8','f8':'int8','f9':'int8','f10':'int8','f11':'int8','f12':'int8','f13':'category',}


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_train = pd.read_csv(f'{PATH}/train.csv', low_memory=True,dtype=dtype)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "user_feat = pd.read_csv(f'{PATH}/user_features.csv', low_memory=True,dtype=dtype)")


# In[ ]:


def display_all(df):
    with pd.option_context("display.max_rows", 1000): 
        with pd.option_context("display.max_columns", 1000): 
            display(df)


# In[ ]:


df_test = pd.read_csv(f'{PATH}/test.csv', low_memory=True,dtype=dtype)


# In[ ]:


# df_train.shape, user_feat.shape, df_test.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_train = df_train.merge(user_feat, left_on='node2_id', right_on='node_id').drop(['node_id'],axis=1)\nno_rename = {'node1_id', 'node2_id','is_chat'}\ndf_train.columns = ['{}{}'.format(c, '' if c in no_rename else '_2') for c in df_train.columns]\ndf_train = df_train.merge(user_feat, left_on='node1_id', right_on='node_id').drop(['node_id'],axis=1)")


# In[ ]:


# df_train.to_feather('df_train_hike')


# In[ ]:


get_ipython().run_cell_magic('time', '', "df_test = df_test.merge(user_feat, left_on='node2_id', right_on='node_id').drop(['node_id'],axis=1)\nno_rename = {'node1_id', 'node2_id','id'}\ndf_test.columns = ['{}{}'.format(c, '' if c in no_rename else '_2') for c in df_test.columns]\ndf_test = df_test.merge(user_feat, left_on='node1_id', right_on='node_id').drop(['node_id'],axis=1)")


# In[ ]:


#display_all(df_test.head().T)


# In[ ]:


#df_test.to_feather('df_test_hike')


# In[ ]:


# df_train = pd.read_feather('df_train_hike')
# df_test = pd.read_feather('df_test_hike')


# In[ ]:


df_train_samp = df_train[0:16000000]


# In[ ]:


df_test.head()


# In[ ]:


df_train.head()


# In[ ]:


df_train.shape, df_test.shape


# In[ ]:


#df_train.dtypes, df_test.dtypes


# In[ ]:


# def print_score(m):
#     res = [roc_auc_score(y_train,m.predict_proba(X_train)[:,1]), roc_auc_score(y_valid,m.predict_proba(X_valid)[:,1]),
#                 m.score(X_train, y_train), m.score(X_valid, y_valid)]
#     if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
#     print(res)


# In[ ]:


test_id = df_test.id
df_test = df_test.drop(['id'],axis =1)


# In[ ]:


dep_var = 'is_chat'
cat_vars = ['f13', 'f13_2']
cont_vars = ['node1_id','node2_id','f1_2','f2_2','f3_2','f4_2','f5_2','f6_2','f7_2','f8_2','f9_2','f10_2','f11_2','f12_2','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12']
procs = [Categorify]


# In[ ]:


data = (TabularList.from_df(df_train_samp, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs,)
                .split_by_idx(list(range(14500001,16000000)))
                .label_from_df(cols=dep_var)
                .add_test(TabularList.from_df(df_test, path=path, cat_names=cat_vars, cont_names=cont_vars))
                .databunch())


# In[ ]:


data.show_batch(rows=5)


# In[ ]:


#learn = tabular_learner(data, layers=[200,100], metrics=accuracy)


# In[ ]:


learn = tabular_learner(data, layers=[200,100], ps=[0.001,0.01], emb_drop=0.04, 
                         metrics=accuracy)


# In[ ]:


learn.model


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3, 5e-2, wd=0.2)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


# learn.lr_find()
# learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(3, 3e-2)


# In[ ]:


test_preds = learn.get_preds(DatasetType.Test)[0][:,1]
submit = pd.DataFrame({'id':test_id, 'is_chat':test_preds}, columns=['id', 'is_chat'])
submit.to_csv('submit_1.csv',index=False)


# In[ ]:


submit.shape

