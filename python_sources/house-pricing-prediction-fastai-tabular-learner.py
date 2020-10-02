#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.tabular import *


# In[ ]:


path = Path('/kaggle/input/house-prices-advanced-regression-techniques')
path.ls()


# In[ ]:


# Import the datasets
train_df = pd.read_csv(path/'train.csv')
test_df = pd.read_csv(path/'test.csv')


# In[ ]:


# Check the length of the dataset
print(train_df.shape)
print(test_df.shape)


# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# * Now let's explore the training set a little bit

# In[ ]:


print(train_df.columns)


# In[ ]:


train_df.describe()


# In[ ]:


# We can see that there are some missing data, let's check for any null value
train_df.isnull().sum()


# In[ ]:


# Let's check for the test set as well
test_df.isnull().sum()


# * We can see that there are 3 columns that have missing values: **MSZoning, LotFrontage, SaleType**

# * Now let's use the FastAI Tabular Learner to train our model

# In[ ]:


# Fill missing value in the test dataset first
test_df.fillna(value=test_df.mean(), inplace=True)


# In[ ]:


# Preprocessing 
procs = [FillMissing, Categorify, Normalize]


# In[ ]:


# Split our variables into target, categorical and continuous variables
dep_var = 'SalePrice'

cat_names = train_df.drop('Id', axis=1).select_dtypes(exclude='number').columns.tolist()

cont_names = train_df.drop('SalePrice', axis=1).select_dtypes(include='number').columns.tolist()

print(cat_names)
print(cont_names)


# In[ ]:


test = TabularList.from_df(df=test_df, cat_names=cat_names, cont_names=cont_names, procs=procs)


# In[ ]:


data = (TabularList.from_df(df=train_df, cat_names=cat_names, cont_names=cont_names, procs=procs)
                   .split_by_rand_pct()
                   .label_from_df(cols=dep_var, label_cls=FloatList, log=True)
                   .add_test(test)
                   .databunch())


# In[ ]:


data.show_batch(10)


# In[ ]:


learn = tabular_learner(data, layers=[10, 10, 10], metrics=rmse)


# In[ ]:


learn.model_dir = '/kaggle/working'


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[ ]:


min_grad_lr = learn.recorder.min_grad_lr
learn.fit_one_cycle(30, slice(min_grad_lr))


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


# Getting prediction
preds, targets = learn.get_preds(DatasetType.Test)
labels = [np.exp(p[0].data.item()) for p in preds]

# Create "submission.csv" file
submission = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': labels})
submission.to_csv('submission.csv', index=False)
submission.head()

