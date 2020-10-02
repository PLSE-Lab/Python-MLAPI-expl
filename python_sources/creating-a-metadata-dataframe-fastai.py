#!/usr/bin/env python
# coding: utf-8

# It's really handy to have all the DICOM info available in a single DataFrame, so let's create that! In this notebook, we'll just create the DICOM DataFrames. To see how to use them to analyze the competition data, see [this followup notebook](https://www.kaggle.com/jhoward/some-dicom-gotchas-to-be-aware-of-fastai).
# 
# First, we'll install the latest versions of pytorch and fastai v2 (not officially released yet) so we can use the fastai medical imaging module.

# In[ ]:


get_ipython().system('pip install torch torchvision feather-format kornia pyarrow --upgrade   > /dev/null')
get_ipython().system('pip install git+https://github.com/fastai/fastai_dev             > /dev/null')


# In[ ]:


from fastai2.basics import *
from fastai2.medical.imaging import *


# Let's take a look at what files we have in the dataset.

# In[ ]:


path = Path('../input/rsna-intracranial-hemorrhage-detection/')


# Most lists in fastai v2, including that returned by `Path.ls`, are returned as a [fastai.core.L](http://dev.fast.ai/core.html#L), which has lots of handy methods, such as `attrgot` used here to grab file names.

# In[ ]:


path_trn = path/'stage_1_train_images'
fns_trn = path_trn.ls()
fns_trn[:5].attrgot('name')


# In[ ]:


path_tst = path/'stage_1_test_images'
fns_tst = path_tst.ls()
len(fns_trn),len(fns_tst)


# We can grab a file and take a look inside using the `dcmread` method that fastai v2 adds.

# In[ ]:


fn = fns_trn[0]
dcm = fn.dcmread()
dcm


# # Labels

# Before we pull the metadata out of the DIMCOM files, let's process the labels into a convenient format and save it for later. We'll use *feather* format because it's lightning fast!

# In[ ]:


def save_lbls():
    path_lbls = path/'stage_1_train.csv'
    lbls = pd.read_csv(path_lbls)
    lbls[["ID","htype"]] = lbls.ID.str.rsplit("_", n=1, expand=True)
    lbls.drop_duplicates(['ID','htype'], inplace=True)
    pvt = lbls.pivot('ID', 'htype', 'Label')
    pvt.reset_index(inplace=True)    
    pvt.to_feather('labels.fth')


# In[ ]:


save_lbls()


# In[ ]:


df_lbls = pd.read_feather('labels.fth').set_index('ID')
df_lbls.head(8)


# In[ ]:


df_lbls.mean()


# There's not much RAM on these kaggle kernel instances, so we'll clean up as we go.

# In[ ]:


del(df_lbls)
import gc; gc.collect();


# # DICOM Meta

# To turn the DICOM file metadata into a DataFrame we can use the `from_dicoms` function that fastai v2 adds. By passing `px_summ=True` summary statistics of the image pixels (mean/min/max/std) will be added to the DataFrame as well (although it takes much longer if you include this, since the image data has to be uncompressed).

# In[ ]:


get_ipython().run_line_magic('time', 'df_tst = pd.DataFrame.from_dicoms(fns_tst, px_summ=True)')
df_tst.to_feather('df_tst.fth')
df_tst.head()


# In[ ]:


del(df_tst)
gc.collect();


# In[ ]:


get_ipython().run_line_magic('time', 'df_trn = pd.DataFrame.from_dicoms(fns_trn, px_summ=True)')
df_trn.to_feather('df_trn.fth')


# There is one corrupted DICOM in the competition data, so the command above prints out the information about this file. Despite the error message show above, the command completes successfully, and the data from the corrupted file is not included in the output DataFrame.