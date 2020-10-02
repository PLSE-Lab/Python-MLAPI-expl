#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 10]


# # Load the train and test sets

# In[ ]:


train = pd.read_csv( '../input/liverpool-ion-switching/train.csv' , dtype={'time':'str','signal':'float','open_channels':'int'} )
test  = pd.read_csv( '../input/liverpool-ion-switching/test.csv'  , dtype={'time':'str','signal':'float'} )

# Split trainset in groups of 100k and 500k consecutive rows
train['group']  = np.arange(train.shape[0])//100_000
train['group2'] = np.arange(train.shape[0])//500_000

# Split testset in groups of 100k consecutive rows
test['group']   = np.arange(test.shape[0])//100_000


# # Fix train and test signal levels.
# ## Train 500k groups 4 and 9 and test 100k groups 5 and 7 needs to be shifted by +2*exp(1)
# ## Other groups are shifted by +exp(1)
# ## All groups must be rescaled by 1.25 to match open_channels scale

# In[ ]:


train.loc[ (train.group2==4)|(train.group2==9) ,'signal'] += np.exp(1)
test.loc[  ( test.group ==5)|( test.group ==7) ,'signal'] += np.exp(1)

train['signal'] += np.exp(1)
train['signal'] /= 1.25

test['signal'] += np.exp(1)
test['signal'] /= 1.25


# ## Analyzing the signal and target statistics we classified each train and test groups as xA and xB, where x is the maximum number of simultaneous open_channels per group

# In[ ]:


MAP = {0:'1A', 1:'1A', 2:'1B', 3:'3B', 4:'10B', 5:'5B', 6:'1B', 7:'3B', 8:'5B', 9:'10B'}
train['type'] = train['group2'].map( MAP )

MAP = {0:'3A', 1:'3B', 2:'5B', 3:'3A', 4:'1B', 5:'10B', 6:'5B', 7:'10B', 8:'3A', 9:'3B',10:'3A', 11:'3A', 12:'4A', 13:'3A', 14:'3A', 15:'3A', 16:'3A', 17:'3A', 18:'3A', 19:'3A'}
test['type'] = test['group'].map( MAP )


# # Observe the drifts in trainset

# In[ ]:


for i in range(10):
    train.loc[train.group2==i].signal.plot()


# In[ ]:


for i in range(20):
    test.loc[test.group==i].signal.plot()


# # After deep analisys we found that all the drifts come from the same equation:
# ## 4 * sin(  2 * pi * range(500000) / 1000000  )

# In[ ]:


SINE_DRIFT = 4*np.sin(  2*np.pi*np.arange(500000)/1000000 )
plt.plot( SINE_DRIFT )


# # Remove the drift from train and test

# In[ ]:


train.loc[ train.group2==6, 'signal' ] -= SINE_DRIFT
train.loc[ train.group2==7, 'signal' ] -= SINE_DRIFT
train.loc[ train.group2==8, 'signal' ] -= SINE_DRIFT
train.loc[ train.group2==9, 'signal' ] -= SINE_DRIFT

train.signal.iloc[ 500000:600000 ]-= SINE_DRIFT[:100000]

test.signal.iloc[ :100000 ]       -= SINE_DRIFT[:100000]
test.signal.iloc[ 100000:200000 ] -= SINE_DRIFT[:100000]
test.signal.iloc[ 400000:500000 ] -= SINE_DRIFT[:100000]
test.signal.iloc[ 600000:700000 ] -= SINE_DRIFT[:100000]
test.signal.iloc[ 700000:800000 ] -= SINE_DRIFT[:100000]
test.signal.iloc[ 800000:900000 ] -= SINE_DRIFT[:100000]

test.loc[ (test.group>=10)&(test.group<=14), 'signal' ] -= SINE_DRIFT


# In[ ]:


train.groupby(['group2','type'])['signal'].agg(['mean','std'])


# In[ ]:


for i in range(50):
    train.loc[train.group==i].signal.plot()


# In[ ]:


for i in range(20):
    test.loc[test.group==i].signal.plot()


# # Remove outlier on train 100k batch group 4
# ## this outlier pattern happens only on trainset so its safe to remove it.

# In[ ]:


train.loc[train.group==4].signal.plot()

train['noise'] = train['signal'] - train['open_channels']
train.loc[ (train.group==4)&(train.noise>1.0), 'signal' ] = np.random.normal( 0,0.2,1 ) #Just add gaussian noise with std 0.20

train.loc[train.group==4].signal.plot()


# # Remove/Decrease 50Hz power line interference

# In[ ]:


# Power line 50Hz interference is that small peak at axis 500 of the plot below
# We used Fourier Transform to remove it precisely
f = np.fft.fft( train.loc[train.group==45].signal )
plt.plot( np.abs(f)[1:1000] )
plt.plot([480, 480], [0, 8000], '--', lw=1)
plt.plot([510, 510], [0, 8000], '--', lw=1)


# In[ ]:


from scipy import fftpack

# Removing 100% of power line interference can remove also usefull information that lies in the 50Hz spectrum.
# Thats why we removed only 45% of powerline interference in order to keep some useful signal present in the 50Hz.
# Remove 50Hz in batches of 100k rows.

for g in range( train.group.max() ):
    f = np.fft.fft( train.loc[train.group==g].signal )
    freq = np.abs( fftpack.fftfreq( len(f) , d=0.0001) )
    f[ (freq >= 49.8)& (freq <= 50.2) ] = 0.55*f[ (freq >= 49.8)& (freq <= 50.2) ]
    train.loc[train.group==g,'signal'] = np.fft.ifft( f ).real
    
for g in range( test.group.max() ):
    f = np.fft.fft( test.loc[test.group==g].signal )
    freq = np.abs( fftpack.fftfreq( len(f) , d=0.0001) )
    f[ (freq >= 49.8)& (freq <= 50.2) ] = 0.55*f[ (freq >= 49.8)& (freq <= 50.2) ]
    test.loc[test.group==g,'signal'] = np.fft.ifft( f ).real
    


# In[ ]:


#Note that the 50Hz frequency interference was decreased but not removed 100%.
f = np.fft.fft( train.loc[train.group==45].signal )
plt.plot( np.abs(f)[1:1000] )
plt.plot([480, 480], [0, 8000], '--', lw=1)
plt.plot([510, 510], [0, 8000], '--', lw=1)


# # Now that signal and open_channels are in the same scale, to calculate the noise added to each row it is just a mater of subtract each other.

# In[ ]:


# Calculate real noise in train set.
# This is only possible because signal and open_channels are in the same scale
train['noise']       = train['signal'] - train['open_channels']

# Subtracting signal from round(signal) is a pseudo way to calculate noise that can be applied both to train and test sets.
train['noise_round'] = train['signal'] - train['signal'].round()
test ['noise_round'] = test ['signal'] - test ['signal'].round()


# In[ ]:


#This is the bias offset and Standard deviation of the noise added to each group type.
train.groupby(['group2','type'])['noise'].agg( ['mean','std'] )


# # Note that the std of the gaussian noise above is different for each type:
# ## Type 1A sdt ~ 0.197
# ## Type 1B sdt ~ 0.197
# ## Type 3B sdt ~ 0.215
# ## Type 5B sdt ~ 0.231
# ## Type 10B sdt ~ 0.327

# In[ ]:





# # Adding 2x 5B std noise gives:

# In[ ]:


x = np.random.normal( 0, 0.231, 1000000) + np.random.normal( 0, 0.231, 1000000)
np.std(x)


# # The result above makes us think that the 10 channels groups are the sum of two 5 channels groups

# In[ ]:





# # Since we can't have the real noise in test set, we can approximate the noise in test using signal-signal.round() as a proxy for the noise.

# In[ ]:


train.groupby(['group2','type'])['noise_round'].agg( ['mean','std'] )


# In[ ]:


test.groupby(['group','type'])['noise_round'].agg( ['mean','std'] )


# In[ ]:





# # # Removing outliers from 100k groups batches 36, 37 and 38 of trainset

# ## We can cleanly see that there is an interference in the 3B groups in trainset and the corresponding 3B groups in test doesn't have such interference.

# In[ ]:


train.loc[ train.group.isin([36,37,38]) ,'signal'].plot()


# In[ ]:


test.loc[ test.type=='3B'].plot( x='time' ,y='signal' )


# In[ ]:


print( train.loc[ train.group==35 ].groupby('open_channels')['noise'].agg(['mean','std']) )
print( train.loc[ train.group==39 ].groupby('open_channels')['noise'].agg(['mean','std']) )


# ## to remove the outliers, lets just add a gaussian noise to open_channels with a specific std level for the specific batches.

# In[ ]:


tmp = train.loc[ train.group==36 ].copy()
tmp.loc[ tmp.open_channels==0 ,'signal'] = 0 + np.random.normal(  -0.0130 , 0.2342, np.sum(tmp.open_channels==0) )
tmp.loc[ tmp.open_channels==1 ,'signal'] = 1 + np.random.normal(  -0.0299 , 0.2356, np.sum(tmp.open_channels==1) )
tmp.loc[ tmp.open_channels==2 ,'signal'] = 2 + np.random.normal(  -0.0460 , 0.2368, np.sum(tmp.open_channels==2) )
tmp.loc[ tmp.open_channels==3 ,'signal'] = 3 + np.random.normal(  -0.0600 , 0.2344, np.sum(tmp.open_channels==3) )
train.loc[ train.group==36 ] = tmp.copy()

tmp = train.loc[ train.group==37 ].copy()
tmp.loc[ tmp.open_channels==0 ,'signal'] = 0 + np.random.normal(  -0.0110 , 0.2307, np.sum(tmp.open_channels==0) )
tmp.loc[ tmp.open_channels==1 ,'signal'] = 1 + np.random.normal(  -0.0299 , 0.2309, np.sum(tmp.open_channels==1) )
tmp.loc[ tmp.open_channels==2 ,'signal'] = 2 + np.random.normal(  -0.0450 , 0.2368, np.sum(tmp.open_channels==2) )
tmp.loc[ tmp.open_channels==3 ,'signal'] = 3 + np.random.normal(  -0.0580 , 0.2295, np.sum(tmp.open_channels==3) )
train.loc[ train.group==37 ] = tmp.copy()

tmp = train.loc[ train.group==38 ].copy()
tmp.loc[ tmp.open_channels==0 ,'signal'] = 0 + np.random.normal(  -0.0100 , 0.2257, np.sum(tmp.open_channels==0) )
tmp.loc[ tmp.open_channels==1 ,'signal'] = 1 + np.random.normal(  -0.0299 , 0.2275, np.sum(tmp.open_channels==1) )
tmp.loc[ tmp.open_channels==2 ,'signal'] = 2 + np.random.normal(  -0.0440 , 0.2278, np.sum(tmp.open_channels==2) )
tmp.loc[ tmp.open_channels==3 ,'signal'] = 3 + np.random.normal(  -0.0560 , 0.2245, np.sum(tmp.open_channels==3) )
train.loc[ train.group==38 ] = tmp.copy()

train['noise'] = train['signal'] - train['open_channels']


# In[ ]:


# Now all train groups looks fine
for i in range(50):
    train.loc[train.group==i].signal.plot()


# In[ ]:


# Now outlier group 7(3B) have similar statistics as group 3(3B)
train.groupby(['group2','type'])['signal'].agg(['mean','std'])


# In[ ]:





# # There still some bias to be removed per group, both in train and in test

# In[ ]:


for i in range(50):
    train.loc[train.group2==i].noise_round.rolling(10000).mean().plot()


# In[ ]:


for i in range(20):
    test.loc[test.group==i].noise_round.rolling(10000).mean().plot()


# # Remove the bias

# In[ ]:


#Remove the bias offset using an iterative loop for both train and test sets

train['noise_round'] = train['signal'] - train['signal'].round()
test ['noise_round'] = test ['signal'] - test ['signal'].round()
for i in range( 7 ):
    train['bias'] = train.groupby('group')['noise_round'].transform('mean')
    test ['bias'] = test.groupby( 'group')['noise_round'].transform('mean')
    
    train['signal'] = train['signal'] - train['bias']
    test ['signal'] = test ['signal'] - test ['bias']
    train['noise_round'] = train['signal'] - train['signal'].round()
    test ['noise_round'] = test ['signal'] - test ['signal'].round()
    
    print( i, 'acc:',np.mean( train.open_channels == train.signal.round() ) , 'f1:',f1_score( train.open_channels , np.clip(train.signal.round(),0,10), average='macro' ) )

train['noise'] = train['signal'] - train['open_channels']
train['noise_round'] = train['signal'] - train['signal'].round()
test ['noise_round'] = test ['signal'] - test ['signal'].round()


# In[ ]:


#Check bias offsets now. Must be close to zero.
train.groupby(['group','type'])['noise'].agg(['mean','std'])


# In[ ]:


train.groupby(['group','type'])['noise_round'].agg(['mean','std'])


# In[ ]:


test.groupby(['group','type'])['noise_round'].agg(['mean','std'])


# # Now the bias offset are close to zero. But still a small bias offset in groups 10B.
# 

# In[ ]:


for i in range(50):
    train.loc[train.group==i].noise_round.rolling(10000).mean().plot()


# In[ ]:


# The bias offset in groups 10B in train is at the same level as in test, so don't worry about it. 
for i in range(10):
    test.loc[test.group==i].noise_round.rolling(10000).mean().plot()


# # Accuracy and F1 using only round(signal) as a predictor.

# In[ ]:


print( 'ACC:', np.mean( train.open_channels == train.signal.round() )  )
print( 'F1:', f1_score( train.open_channels , np.clip(train.signal.round(),0,10), average='macro' ) )


# In[ ]:


#Calc ACC per group
train['hit'] = train.open_channels == train.signal.round()
train.groupby(['group','type'])['hit'].mean()


# In[ ]:


train[['time','signal','type','open_channels']].to_csv('train_clean_giba.csv', index=False )
test [['time','signal','type']].to_csv('test_clean_giba.csv', index=False )

