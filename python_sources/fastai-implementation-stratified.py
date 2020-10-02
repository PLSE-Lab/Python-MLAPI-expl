#!/usr/bin/env python
# coding: utf-8

# # Import modules and data

# In[ ]:


from fastai.tabular import *
from fastai.callbacks import ReduceLROnPlateauCallback,EarlyStoppingCallback
from sklearn.metrics import roc_auc_score
import joblib
import gc
import os
print(os.listdir("../input"))


# In[ ]:


class roc(Callback):
    
    def on_epoch_begin(self, **kwargs):
        self.total = 0
        self.batch_count = 0
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        preds = F.softmax(last_output, dim=1)
        roc_score = roc_auc_score(to_np(last_target), to_np(preds[:,1]))
        self.total += roc_score
        self.batch_count += 1
    
    def on_epoch_end(self, num_batch, **kwargs):
        self.metric = self.total/self.batch_count


# In[ ]:


df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')


# In[ ]:


df.shape


# In[ ]:


df_test.shape


# In[ ]:


df.head()


# In[ ]:


df_test.head()


# # Balance class

# In[ ]:


from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler()
df_resampled, y_resampled = ros.fit_sample(df, df['target'])
df_resampled = pd.DataFrame(df_resampled, columns = df.columns)
df['target'].mean(), df_resampled['target'].mean()


# In[ ]:


#df_resampled['var_0'].dtype


# In[ ]:


#cols = df.columns
num_cols = df._get_numeric_data().columns
#list(set(cols) - set(num_cols))
num_cols


# In[ ]:


df_resampled.shape


# In[ ]:


dep_var = 'target'
cont_names = num_cols[1:].tolist()
procs = [FillMissing, Categorify, Normalize]


# In[ ]:


df_resampled[cont_names] = df_resampled[cont_names].apply(pd.to_numeric)
df_resampled['target'] = df_resampled['target'].astype('int')


# In[ ]:


test = TabularList.from_df(df_test, cont_names=cont_names)
data = (TabularList.from_df(df_resampled, cont_names=cont_names, procs=procs)
                           .random_split_by_pct(0.1)
                           .label_from_df(cols=dep_var)
                           .add_test(test)
                           .databunch())


# In[ ]:


data.show_batch(rows=10)


# In[ ]:


learn = tabular_learner(data, layers=[256, 128], ps=[0,0], metrics=[accuracy], y_range=(0, 1)).to_fp16()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# class CategoryList sets it to CrossEntropyFlat()
# 
# class MultiCategoryList sets it to BCEWithLogitsFlat()
# 
# class FloatList (for regression problems) sets it to MSELossFlat()

# In[ ]:


learn.fit_one_cycle(25, 1e-3)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.save('0')
#learn.load('0')


# In[ ]:


probs, _ = learn.get_preds(ds_type=DatasetType.Test) 


# In[ ]:


len(pred)


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] =  F.softmax(probs, dim=1)[:, 1].numpy()
submission.to_csv('starter_submission.csv', index=False)


# In[ ]:


#y_pred = learn.get_preds(ds_type=DatasetType.Valid) 
#roc_auc_own(y_pred[0], y_pred[1])


# In[ ]:




