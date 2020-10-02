#!/usr/bin/env python
# coding: utf-8

# This kernel is based on the approach from https://www.kaggle.com/matthewa313/ensembling-algorithm-for-average-precision-metric
# 
# Take predictions from LB 0.771 from https://www.kaggle.com/ateplyuk/ensembling-lb-0-771
# 
# Take predictions from LB 0.766 https://www.kaggle.com/hung96ad/ensembling-algorithm-for-average-precision-0-766
# 
# Plug your outputs to this kernel and you're good to go.

# In[ ]:


import os
os.listdir('../input')


# In[ ]:


import csv
import pandas as pd # not key to functionality of kernel

sub_files = [
                '../input/publickernel/sub_766.csv',
                '../input/publickernel/sub_771.csv'
]

# Weights of the individual subs
sub_weight = [
                0.766**2,
                0.771**2
            ]


# In[ ]:


Hlabel = 'Image' 
Htarget = 'Id'
npt = 5 # number of places in target

place_weights = {}
for i in range(npt):
    place_weights[i] = ( 1 / (i + 1) )
    
print(place_weights)

lg = len(sub_files)
sub = [None]*lg
for i, file in enumerate( sub_files ):
    ## input files ##
    print("Reading {}: w={} - {}". format(i, sub_weight[i], file))
    reader = csv.DictReader(open(file,"r"))
    sub[i] = sorted(reader, key=lambda d: str(d[Hlabel]))

## output file ##
out = open("sub_ns.csv", "w", newline='')
writer = csv.writer(out)
writer.writerow([Hlabel,Htarget])

for p, row in enumerate(sub[0]):
    target_weight = {}
    for s in range(lg):
        row1 = sub[s][p]
        for ind, trgt in enumerate(row1[Htarget].split(' ')):
            target_weight[trgt] = target_weight.get(trgt,0) + (place_weights[ind]*sub_weight[s])
    tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
    writer.writerow([row1[Hlabel], " ".join(tops_trgt)])
out.close()


# In[ ]:




