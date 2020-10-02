#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai import *
from fastai.tabular import *


# In[ ]:


path = Path("/kaggle/input")
path.ls()


# In[ ]:


train_df = pd.read_csv('/kaggle/input/human-activity-recognition-with-smartphones/train.csv')
train_df.head()


# In[ ]:


train_df.info()


# In[ ]:


train_df.shape


# In[ ]:


train_df.describe()


# In[ ]:


train_df.dtypes


# In[ ]:


train_df['Activity'].unique()


# In[ ]:


test_df = pd.read_csv("/kaggle/input/human-activity-recognition-with-smartphones/test.csv")
test_df.head()


# In[ ]:


test_df.info()


# In[ ]:


test_df.shape


# In[ ]:


test_df.describe()


# In[ ]:


test_df.dtypes


# In[ ]:



dep_var = 'Activity'

cont_names = list(train_df.columns.values.tolist()) 
cont_names.remove('Activity')

# Transformations
procs = [FillMissing, Normalize]

# Test Tabular List
test_data = TabularList.from_df(test_df, cont_names=cont_names)

# Train Data Bunch


data_list = (TabularList.from_df(train_df, path='.',cont_names=cont_names)
                        .split_by_idx(list(range(1000,3000)))
                        .label_from_df(cols = dep_var)
                        .add_test(test_df)
                        .databunch())

data_list.show_batch(rows=10)


# In[ ]:


learn = tabular_learner(data_list, layers=[200,100, 15], metrics=accuracy)


# In[ ]:


# select the appropriate learning rate
learn.lr_find()


# In[ ]:



# we typically find the point where the slope is steepest
learn.recorder.plot()


# In[ ]:


# Fit the model based on selected learning rate
learn.fit_one_cycle(10, max_lr=slice((1e-01)))


# In[ ]:


# Analyse our model
learn.model


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data_list.valid_ds)==len(losses)==len(idxs)


# In[ ]:


interp.plot_confusion_matrix(figsize=(20,20), dpi=100)


# In[ ]:


interp.most_confused(min_val=1)


# In[ ]:




