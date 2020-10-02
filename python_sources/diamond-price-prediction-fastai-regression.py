#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.tabular import *


# In[ ]:


path = Path('/kaggle/input/diamonds')
path.ls()


# In[ ]:


df = pd.read_csv(path/'diamonds.csv')
df.head()


# In[ ]:


# As we can see from the "count" row, there is no missing data in the dataset
df.describe()


# * Let's build a model that can predict the **price** of a diamond given its **characteristics**

# In[ ]:


# Preprocessing 
procs = [FillMissing, Categorify, Normalize]


# In[ ]:


# Split our variables into target, categorical and continuous variables
dep_var = 'price'

# There were too many missing "Cabin" values, so we will ignore that
# The "Name" column has already been replaced by the "Title" column
cat_names = df.select_dtypes(exclude='number').columns.tolist()

cont_names = df.drop(['price', 'Unnamed: 0'], axis=1).select_dtypes(include='number').columns.tolist()

print(dep_var)
print(cat_names)
print(cont_names)


# In[ ]:


# Remember to add "label_cls=FloatList" to tell the model that we are doing regression
data = (TabularList.from_df(df=df, cat_names=cat_names, cont_names=cont_names, procs=procs)
                   .split_by_rand_pct(seed=42)
                   .label_from_df(cols=dep_var, label_cls=FloatList)
                   .databunch())


# In[ ]:


data.show_batch(10)


# In[ ]:


learn = tabular_learner(data, layers=[200, 100], metrics=mse)


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


learn.recorder.plot_losses()

