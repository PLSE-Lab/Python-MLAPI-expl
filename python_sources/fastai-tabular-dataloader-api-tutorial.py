#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

from fastai.vision import *
from fastai.tabular import *
from fastai.metrics import error_rate
import csv
import numpy as np
import PIL 
import pandas as pd
#defaults.device = torch.device('cuda')


# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Overview
# In this kernel we apply the tabular component of the fastai v1 library and use only the train.csv as the basis of our predictions. We drop a few columns that seem irrelevant (name of the pet, the long description, and the supplemental breed & color info). 
# 
# ## Goal
# To show how to use the fastai databunch API on tabular data.

# The first two cells are straightforward: Load the train & test csv's into pandas dataframes, dropping superfluous columns.

# In[ ]:


train = pd.read_csv('../input/train/train.csv')
train = train.drop(['Name','Breed2','Color3', 'Description'], axis=1)
train.head(2)


# In[ ]:


test = pd.read_csv('../input/test/test.csv'); 
test = test.drop(['Name','Breed2','Color3','Description'], axis=1)
test.head(2)


# Here we create two numpy arrays. The first holds just the independent variable, the PetId, and the second will hold our predictions.

# In[ ]:


pet = test['PetID'].values
pred = []


# ## fastai
# Here is the deep learning part! The fastai library requires us to identify categorical variables, continuous variables (not done here) and the dependent variable. 

# In[ ]:


cat_names = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize', 'Vaccinated', 'Dewormed', 
             'Sterilized', 'Health', 'RescuerID','VideoAmt','PetID','PhotoAmt']
dep_var = 'AdoptionSpeed'


# The variable `valid_idx` tells fastai how many rows of our tabular dataset to hold aside as a validation set. We are keeping this relatively small.
# 
# `procs` refers to pre-processing. These are out-of-the-box.
# 
# The line `data = ` is where we create our TabularDataBunch, using a dataframe (the 'train' dataframe we created via pd.read_csv earlier) and passing the variables we set earlier..

# In[ ]:


valid_idx = range(len(train)-1000, len(train))


# In[ ]:


procs = [FillMissing, Categorify, Normalize]
data = TabularDataBunch.from_df(path='../working/', df=train, dep_var=dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)


# # Learner
# 
# Now that our data is loaded into a DataBunch, we can train. The `layers` variable is a real swag. It's worth trying other dimensions.
# 
# We fit a cycle, and then I extract the predictions (predlist is an empty array to hold these predictions). That's it! 
# 
# With no sentiment analysis, no image classification, and only a few seconds of gpu time this is a ~.28 model. 

# In[ ]:


learn = tabular_learner(data, layers=[200,100], metrics=accuracy)
learn.fit_one_cycle(1, 1e-2)


# In[ ]:


predlist = []


# In[ ]:


for x in range(0,len(test)):
    pred.append(learn.predict(test.iloc[x]))


# In[ ]:


for x in range(0,len(test)):
    preds = pred[x][0]


# In[ ]:


for x in range(0,len(test)):
    predlist.append(pred[x][1].item())


# In[ ]:


submission = pd.DataFrame({'PetID':pet, 'AdoptionSpeed':predlist})
submission.head()


# In[ ]:


submission.to_csv('submission.csv', index=False)


# In[ ]:




