#!/usr/bin/env python
# coding: utf-8

# # Import

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

from fastai.tabular import *
from sklearn.metrics import roc_auc_score

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


torch.manual_seed(47)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(47)


# # Loading data

# In[ ]:


data_dir = '../input/'
get_ipython().system('ls {data_dir}')


# In[ ]:


train_raw = pd.read_csv(f'{data_dir}train.csv')
train_raw.head()


# In[ ]:


test_raw = pd.read_csv(f'{data_dir}test.csv')
test_raw.head()


# In[ ]:


train_raw.shape, test_raw.shape


# In[ ]:


train_raw.isnull().sum().sum(), test_raw.isnull().sum().sum()


# So there are no missing values in either training or test set.

# # Target distribution

# In[ ]:


sns.countplot(train_raw.target)
plt.show()


# In[ ]:


train_raw.target.value_counts()


# Looks like class labels are uniformly distributed in training data.

# # Validation set
# 
# We will use 20,000 records from our training set as validation data.

# In[ ]:


valid_idx = range(len(train_raw)- 20000, len(train_raw))


# # Feature Engineering

# In[ ]:


columns = train_raw.columns[1:-1]
first_name = [i.split("-")[0] for i in columns]
print(set(first_name))
print(len(first_name))
print(len(set(first_name)))


# In[ ]:


for first in first_name:
    filter_col = [col for col in train_raw if col.startswith(first)]
    test_raw[first+"-mean"] = test_raw.loc[:, filter_col].mean(axis=1)
    train_raw[first+"-mean"] = train_raw.loc[:, filter_col].mean(axis=1)
    test_raw[first+"-std"] = test_raw.loc[:, filter_col].std(axis=1)
    train_raw[first+"-std"] = train_raw.loc[:, filter_col].std(axis=1)


# In[ ]:


second_name = [i.split("-")[1] for i in columns]
print(set(second_name))
print(len(second_name))
print(len(set(second_name)))


# In[ ]:


for second in second_name:
    filter_col = [col for col in columns if second==col.split("-")[1]]
    test_raw[second+"-mean"] = test_raw.loc[:, filter_col].mean(axis=1)
    train_raw[second+"-mean"] = train_raw.loc[:, filter_col].mean(axis=1)
    test_raw[second+"-std"] = test_raw.loc[:, filter_col].std(axis=1)
    train_raw[second+"-std"] = train_raw.loc[:, filter_col].std(axis=1)


# In[ ]:


train_raw.shape


# In[ ]:


for col in train_raw.columns:
    if (train_raw[col].isnull().sum()>0):
        train_raw.drop([col], axis=1, inplace=True)
        test_raw.drop([col], axis=1, inplace=True)


# In[ ]:


train_raw.shape


# # Databunch
# 
# Let's create a databunch instance with our training, validation and test data

# In[ ]:


cont_names = train_raw.columns.tolist()
cont_names.remove('id')
cont_names.remove('target')
cont_names.remove('wheezy-copper-turtle-magic')

cat_names = ['wheezy-copper-turtle-magic']

procs = [FillMissing, Categorify, Normalize]


# In[ ]:


dep_var = 'target'

data = TabularDataBunch.from_df('.', train_raw, dep_var=dep_var, valid_idx=valid_idx, procs=procs,
                                cat_names=cat_names, cont_names=cont_names, test_df=test_raw, bs=2048)


# # Learner

# In[ ]:


learn = tabular_learner(data, layers=[1000, 750, 500, 300], emb_szs={'wheezy-copper-turtle-magic': 512}, metrics=accuracy, ps=0.65, wd=3e-1)


# Let's find optimal learning rate for our classification task -

# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# Based on above learning rate finder graph, we can choose `1e-3` as our learning rate.

# # Training

# In[ ]:


lr = 1e-3
learn.fit_one_cycle(40, lr)


# In[ ]:


learn.save("stage-1")


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


learn.lr_find()


# In[ ]:


learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(200, 1e-4)


# In[ ]:


learn.recorder.plot_losses()


# In[ ]:


learn.recorder.plot_metrics()


# In[ ]:


learn.save("model-2")


# # Calculating roc_auc_score on validation data

# In[ ]:


val_preds = learn.get_preds(DatasetType.Valid)
roc_auc_score(train_raw.iloc[valid_idx].target.values, val_preds[0][:,1].numpy())


# # Retrain on all the data

# In[ ]:


data = TabularDataBunch.from_df('.', train_raw, dep_var=dep_var, valid_idx=[], procs=procs,
                                cat_names=cat_names, cont_names=cont_names, test_df=test_raw, bs=2048)
learn.data = data


# In[ ]:


learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(200, 1e-4)


# # Prediction on test data

# In[ ]:


test_preds = learn.get_preds(DatasetType.Test)


# In[ ]:


sub_df = pd.read_csv(f'{data_dir}sample_submission.csv')
sub_df.target = test_preds[0][:,1].numpy()
sub_df.head()


# In[ ]:


sub_df.to_csv('solution.csv', index=False)


# In[ ]:


test_raw.to_csv("test_raw.csv", index=False)
train_raw.to_csv("train_raw.csv", index=False)

