#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.tabular import * 


# In[ ]:


path = Path('/kaggle/input/mushroom-classification')
path.ls()


# In[ ]:


df = pd.read_csv(path/'mushrooms.csv')
print(df.shape)
df[:20].T


# In[ ]:


# Explore the dataset
df.describe()


# In[ ]:


# Preprocessing
procs = [FillMissing, Categorify, Normalize]

# Use 1000 data points for validation
valid_idx = range(len(df) - 1000, len(df))


# In[ ]:


# Split our variables into target, categorical and continuous variables
dep_var = 'class'

# It seems like for this dataset, all columns are categorical data
cat_names = list(df.columns.values)
cat_names.remove('class')
print(cat_names)


# In[ ]:


# Create the data source
data = TabularDataBunch.from_df(path, df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)
print(data.train_ds.cont_names)  # `cont_names` defaults to: set(df)-set(cat_names)-{dep_var}


# In[ ]:


# Define the model
learn = tabular_learner(data, layers=[200,100], metrics=accuracy)


# In[ ]:


learn.model_dir = '/kaggle/working'


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


min_grad_lr = learn.recorder.min_grad_lr
learn.fit_one_cycle(5, min_grad_lr)


# * So we can easily achieve 100% accuracy with this dataset! :)
