#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import pandas as pd


# In[ ]:


sub_files = [
                 '../input/vansht2/sub_ens (4).csv',
'../input/vansht/sub_ens (3).csv'            ]

sub_weight = [
                0.952**2,
                0**2
            ]


# In[ ]:


Hlabel = 'Image' 
Htarget = 'Id'
npt = 6
place_weights = {}
for i in range(npt):
    place_weights[i] = ( 1 / (i +5 ) )
    
print(place_weights)

lg = len(sub_files)
sub = [None]*lg
for i, file in enumerate( sub_files ):
   
    print("Reading {}: w={} - {}". format(i, sub_weight[i], file))
    reader = csv.DictReader(open(file,"r"))
    sub[i] = sorted(reader, key=lambda d: str(d[Hlabel]))

out = open("sub_ens.csv", "w", newline='')
writer = csv.writer(out)
writer.writerow([Hlabel,Htarget])

for p, row in enumerate(sub[0]):
    target_weight = {}
    for s in range(lg):
        row1 = sub[s][p]
        for ind, trgt in enumerate(row1[Htarget].split(' ')):
            target_weight[trgt] = target_weight.get(trgt,0) + (place_weights[ind]*2*sub_weight[s])
    tops_trgt = sorted(target_weight, key=target_weight.get, reverse=True)[:npt]
    writer.writerow([row1[Hlabel], " ".join(tops_trgt)])
out.close()

