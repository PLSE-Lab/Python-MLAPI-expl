#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from fastai.vision import *


# In[ ]:


path_data = untar_data(URLs.MNIST_TINY); path_data.ls()


# In[ ]:


itemlist = ItemList.from_folder(path_data/'test')
itemlist


# How does such output above is generated?
# 
# behind the scenes, executing `itemlist` calls `ItemList.__repr__` which basically prints out `itemlist[0]` to `itemlist[4]`

# In[ ]:


itemlist[0]


# In[ ]:


print(itemlist[0])


# In[ ]:


itemlist[0].__class__


# In[ ]:


itemlist[0].__repr__()


# and `itemlist[0]` basically calls `itemlist.get(0)` which returns `itemlist.items[0]`. That's why we have outputs like above.

# In[ ]:


imagelist = ImageList.from_folder(path_data/'test')
imagelist


# How does such output above is generated?
# 
# from `ItemList`, `ImageList` inherits `__repr__`. behind the scenes, executing `imagelist` calls `ImageList.__repr__` which basically prints out `imagelist[0]` to `imagelist[4]`

# In[ ]:


imagelist[0]


# In[ ]:


print(imagelist[0])


# In[ ]:


imagelist[0].__repr__()


# In[ ]:


imagelist[0]._repr_png_()


# @stas provided a very helpful advice [here](https://forums.fast.ai/t/documentation-improvements/32550/179?u=daniel).

# In[ ]:




