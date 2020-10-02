#!/usr/bin/env python
# coding: utf-8

# Discovered images not of a cervix in additional data. If these images made it past human annotator, I suspect the labels may not be accurate. 
# 

# In[ ]:



get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

# Images which I found to be not of a cervix
not_cervix = [
    '../input/additional/Type_1/746.jpg',
    '../input/additional/Type_1/2030.jpg',
    '../input/additional/Type_1/4065.jpg',
    '../input/additional/Type_1/4702.jpg',
    '../input/additional/Type_1/4706.jpg',
    '../input/additional/Type_2/1813.jpg',
    '../input/additional/Type_2/3086.jpg',
]

for fname in not_cervix:
    plt.figure()
    plt.imshow(plt.imread(fname))
   

