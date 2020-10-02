#!/usr/bin/env python
# coding: utf-8

# This kernel will produce a subset of the NIH chest x-ray dataset, containing only the images that has bbox annotations and diagnosed as pneumonia. Great for evaluating your model before submitting!
# Commit the kernel and download the resulting output files. There are not that many, about 200.

# In[ ]:


get_ipython().system('grep  -i pneu ../input/data/BBox_List_2017.csv>pneumonia.csv')


# In[ ]:


import glob
import zipfile
inf = open("pneumonia.csv", "r")
myzip = zipfile.ZipFile('nih_pneumonia.zip', 'w')
for l in inf:
    print(l)
    r = l[:-1].split(",")
    print(r[0])
    fn = glob.glob("../input/data/images_*/images/"+r[0])
    myzip.write(fn[0], arcname=r[0])
myzip.close()
    
    


# In[ ]:


ls -l 

