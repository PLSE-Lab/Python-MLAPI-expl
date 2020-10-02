#!/usr/bin/env python
# coding: utf-8

# ## Getting the data from the file structure  
# There are plenty of ways, but here's a quick example to get you going. Maybe not very elegant, but it worked for my project.

# In[ ]:


import os
import pandas as pd
def get_set(path):
    data = pd.DataFrame()
    for dirpath, subdirs, files in os.walk(path):
        for file in files:
            filepath = os.path.abspath(os.path.join(dirpath,file))
            strpath = str(filepath).split('/')
            class_num = strpath[4]
            class_color = strpath[5]
            class_shape = strpath[6]
            class_shade = strpath[7]
            row = pd.Series({'filepath':filepath,
                             'class_color':class_color,
                             'class_num':class_num,
                             'class_shape':class_shape,
                             'class_shade':class_shade})
            data = data.append(row,ignore_index=True)
    return data

data = get_set('../input/set_dataset')
data.head()


# In[ ]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

filepath = data['filepath'].iloc[0]
print(filepath)
img = cv2.imread(filepath,1)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb)
plt.show()


# You'll have to shift the indices in `get_set()` up by one, and add another folder to the filepath, if you use this in your own kernel, because Kaggle changed the way kernels handle single datasets, but apparently the change didn't propogate to starter notebooks.
