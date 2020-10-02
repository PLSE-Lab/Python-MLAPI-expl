#!/usr/bin/env python
# coding: utf-8

# This is my first kernel submission and just a baseline model for this competition using Fastai V1.  I'm just trying to get things working and understand the Kaggle CLI. I'll be documenting my steps, cleaning up my code, and improving my model over time.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.tabular import *


# ## Data Cleaning

# In[ ]:


os.makedirs("data/pet_finder", exist_ok=True)
path = Path('data/pet_finder'); path


# In[ ]:


train_df = pd.read_csv('../input/train/train.csv')
test_df = pd.read_csv('../input/test/test.csv')


# ### Databunch

# In[ ]:


dep_var = 'AdoptionSpeed'
cat_names = ['Type', 'Name', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3', 'MaturitySize',
             'FurLength', 'State', 'RescuerID', 'PetID']
cont_names = ['Age', 'Fee', 'VideoAmt', 'PhotoAmt']
procs = [FillMissing, Categorify, Normalize]

# Not including 'Description'


# In[ ]:


test = TabularList.from_df(test_df, path=path, cat_names=cat_names, cont_names=cont_names)


# In[ ]:


data = (TabularList.from_df(train_df, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)
        .random_split_by_pct(0.2, seed=42)
        .label_from_df(cols=dep_var)
        .add_test(test)
        .databunch()
)


# In[ ]:


data.show_batch(rows=10)


# ## Learner

# In[ ]:


learn = tabular_learner(data, layers=[200,100], metrics=accuracy)


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit(1, 1e-2)


# ### Submission

# In[ ]:


test_preds = np.argmax(learn.get_preds(DatasetType.Test)[0],axis=1)
test_preds


# In[ ]:


sub_df = pd.DataFrame(data={'PetID': pd.read_csv('../input/test/test.csv')['PetID'],
                            'AdoptionSpeed': test_preds})
sub_df.head()


# In[ ]:


sub_df.to_csv('submission.csv', index=False)

