#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from fastai.vision import *
from fastai import metrics


# In[ ]:


path = untar_data(URLs.MNIST_SAMPLE); path


# In[ ]:


path.ls()


# In[ ]:


tfms = get_transforms(do_flip=False)


# In[ ]:


get_ipython().set_next_input('data = ImageDataBunch.from_folder');get_ipython().run_line_magic('pinfo2', 'ImageDataBunch.from_folder')


# In[ ]:


data = ImageDataBunch.from_folder


# In[ ]:


data = ImageDataBunch.from_folder(path, ds_tfms=tfms,  size=26)


# In[ ]:


def show_data():
    return data.show_batch(3,figsize=(12,4))


# In[ ]:


show_data()


# In[ ]:


df = pd.read_csv(path/'labels.csv')


# In[ ]:


df.head()


# In[ ]:


data = ImageDataBunch.from_df(path,df,ds_tfms=tfms,size=24)


# In[ ]:


show_data()


# In[ ]:


data = ImageDataBunch.from_csv(path, ds_tfms=tfms, size=24)


# In[ ]:


show_data()


# In[ ]:


path.ls()


# In[ ]:


fnames =[path/name for name in df['name']]


# In[ ]:


fnames[:2]


# In[ ]:


pat = r'/(\d)/\d+.png$'


# In[ ]:


data = ImageDataBunch.from_name_re(path, fnames, pat, ds_tfms=tfms, size=24)


# In[ ]:


show_data()


# In[ ]:


data = ImageDataBunch.from_name_func(path,fnames,lambda x: '3' if '/3/' in str(x) else '7', ds_tfms=tfms , size=24)


# In[ ]:


show_data()


# In[ ]:


labels =[('3' if '/3/' in str(name) else '7') for name in fnames]


# In[ ]:


labels[:2]


# In[ ]:


data = ImageDataBunch.from_lists(path, fnames, labels=labels, ds_tfms=tfms, size=24)


# In[ ]:


show_data()


# In[ ]:




