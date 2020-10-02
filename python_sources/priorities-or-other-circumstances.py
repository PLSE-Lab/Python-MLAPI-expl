# As technology becomes more ubiquitous, are viewpoints on priorities 
# and necessities shifting?

#This file is very much a work in progress and I'm developing it locally
#in a notebook, but here's a fast start. 

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../input/pums/ss13husa.csv')

### Pare the dataset down to only those structures that have more than one person
df = df[(df['NP']>0)]

### Now, pull out rows where occupancy is vacant
df = df[df['VACS'].isnull()]

### Print total occupied
print('Total number occupied: ')
print(df.shape[0])

### See how many people had internet subscriptions and food stamps
intAndFoodStamps = df[(df['ACCESS']==1) & (df['FS']==1)]

### Number that had an internet subscription and were on food stamps
print ('Number with internet subscriptions and food stamps: ')
print (intAndFoodStamps.shape[0])

### Internet and no flush toilet
intAndNoToilet = df[(df['ACCESS']==1) & (df['TOIL']==2)]

print ('Number with internet subscriptions and no flushing toilet: ')
print (intAndNoToilet.shape[0])

### More details
intAndNoToilet[['YBL','ST','FS','LAPTOP','BATH','TYPE','REFR','RWAT','TEN','WATP','VEH','YBL','FINCP', 'HINCP','PARTNER','R18','PLM']]

