#!/usr/bin/env python
# coding: utf-8

# # Please First Upvote the kernel from where this has been forked
# 
# - **All work belongs to the original author, not to me except very few small changes to match this comp's stuff, So my thanks and Kudos to the author..**
# 
# - Original Kernel Link https://www.kaggle.com/kmader/create-a-mini-xray-dataset-standard

# In[ ]:


get_ipython().system('ls ../input/')


# In[ ]:


get_ipython().system('grep  -i pneumothorax ../input/data/BBox_List_2017.csv>pneumothorax.csv')
import glob
import zipfile
inf = open("pneumothorax.csv", "r")
myzip = zipfile.ZipFile('pneumothorax.zip', 'w')
for l in inf:
    print(l)
    r = l[:-1].split(",")
    print(r[0])
    fn = glob.glob("../input/data/images_*/images/"+r[0])
    myzip.write(fn[0], arcname=r[0])
myzip.close()


# In[ ]:


ls -l 

