#!/usr/bin/env python
# coding: utf-8

# In this kernel I'll explore the changes in the DCT tables of the Cover and modified images - how big they are in terms of how many bits have been changed compared to the cover image. The number of the changed bits doesn't nescessary equals to payload size, because sometimes you don't have to change the coefficient at all when you encoding the bit. But I'd expect it to be somewhat proportional
# 
# Some key findings:
# 1. The payloads can vary quite significantly in size
# 2. JMiPOD pictures have generally mode bits altered
# 3. Higher quality images have mode bits altered
# 4. For some images there are no changes at all.

# In[ ]:


#install jpegio
get_ipython().system(' git clone https://github.com/dwgoon/jpegio')
# Once downloaded install the package
get_ipython().system('pip install jpegio/.')
import jpegio as jio


# In[ ]:


import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm


# In[ ]:


WORKDIR = "/kaggle/input/alaska2-image-steganalysis/"


# In[ ]:


subdirs = ["Cover", "JUNIWARD", "JMiPOD", "UERD"]


# In[ ]:


def getqf(image):
    qt0 =  image.quant_tables[0][0, 0]
    dic = {2:95, 3:90, 8:75}
    return dic[qt0]


# In[ ]:


qfs = []
methods = []
bits = []
fnames_ser = []
fnames = sorted(os.listdir(os.path.join(WORKDIR, subdirs[0])))
for fn in tqdm(fnames):
    cover_dct = None
    for method in subdirs:
        fulldir = os.path.join(WORKDIR, method)
        im = jio.read(os.path.join(fulldir, fn))
        qf = getqf(im)
        dct = im.coef_arrays
        if cover_dct is None:
            cover_dct = np.array(dct).copy()
            nbits = 0
        else:
            nbits = np.count_nonzero(np.array(dct) != np.array(cover_dct))
        qfs.append(qf)
        bits.append(nbits)
        methods.append(method)
        fnames_ser.append(fn)


# In[ ]:


df_stats = pd.DataFrame({"file":fnames_ser, "method": methods, "quality": qfs, "nbits": bits})
df_stats.head(10)


# In[ ]:


df_stats.to_csv("changed_bits.csv", index=False)


# In[ ]:


df_stats.groupby(["method", "quality"]).nbits.agg(['mean', 'median', 'min', 'max'])


# In[ ]:


df_stats.groupby("method").nbits.mean().plot.bar()


# In[ ]:


df_stats.groupby("method").nbits.median().plot.bar()


# In[ ]:


df_stats.groupby("quality").nbits.mean().plot.bar()


# In[ ]:


df_stats.groupby("quality").nbits.median().plot.bar()


# In[ ]:


df_stats.query("method=='JUNIWARD' and quality == 75").nbits.hist()


# In[ ]:


df_stats.query("method=='JUNIWARD' and quality == 90").nbits.hist()


# In[ ]:


df_stats.query("method=='JUNIWARD' and quality == 95").nbits.hist()


# Do we have any modified files that are actually not modified at all? Seems like we do.

# In[ ]:



df_unchanged = df_stats.query("method !='Cover' and nbits == 0")
df_unchanged.shape, df_unchanged.head(20)


# In[ ]:


df_stats.query("file=='02137.jpg'")


# Is there really no difference? Yes.

# In[ ]:


get_ipython().system('diff --report-identical /kaggle/input/alaska2-image-steganalysis/Cover/02137.jpg /kaggle/input/alaska2-image-steganalysis/UERD/02137.jpg')

