#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import f1_score
#%matplotlib notebook
import matplotlib.pyplot as plt
import gc


# In[ ]:


train = pd.read_csv('../input/data-without-drift/train_clean.csv', dtype={'time':'str','signal':'float','open_channels':'int'} )
test  = pd.read_csv('../input/data-without-drift/test_clean.csv' , dtype={'time':'str','signal':'float'} )

test ['open_channels'] = 0
train['group'] = np.arange(train.shape[0])//500_000
test ['group'] = np.arange( test.shape[0])//100_000
train.loc[ (train.group==4)|(train.group==9) ,'signal'] += np.exp(1)
test.loc[  ( test.group==5)|( test.group==7) ,'signal'] += np.exp(1)

train['group'] = np.arange(train.shape[0])//100_000

train['signal'] = train['signal'] + np.exp(1)
train['signal'] = train['signal'] / 1.239

test['signal'] = test['signal'] + np.exp(1)
test['signal'] = test['signal'] / 1.239


# In[ ]:


train['signal_round'] = np.clip( train['signal'].round() , 0 , 10 )
train['noise'] = train['signal'] - train['open_channels']
train['noise_round'] = train['signal'] - train['signal'].round()
train['noise_std'] = train.groupby(['group','signal_round'])['noise_round'].transform('std')
train['gaussian_noise'] = train['noise_std'].apply( lambda x: np.random.normal(0,x,1)[0]  )
train['gaussian_noise'] = train['gaussian_noise'].fillna(0.20)
print( train.head(2) )

test['signal_round'] = np.clip( test['signal'].round() , 0 , 10 )
test['noise'] = 0
test['noise_round'] = test['signal'] - test['signal'].round()
test['noise_std'] = test.groupby(['group','signal_round'])['noise_round'].transform('std')
test['gaussian_noise'] = test['noise_std'].apply( lambda x: np.random.normal(0,x,1)[0]  )
test['gaussian_noise'] = test['gaussian_noise'].fillna(0.20)
print( test.head(2) )


# In[ ]:


train.groupby('group')['noise_std'].mean()


# In[ ]:


test.groupby('group')['noise_std'].mean()


# In[ ]:


TRAIN = []
for i, dt in train.groupby( 'group' ):
    tmp = dt.copy()
    varf = np.fft.fft( tmp['gaussian_noise'].values )
    mmax = np.max( np.abs( varf ) )

    varf = np.fft.fft( tmp['noise_round'].values )
    filter_ind = np.where( np.abs(varf) >= 1.10*mmax )[0]
    print( i , dt.shape, len(filter_ind) )
    
    filter_ind = np.concatenate( [filter_ind,
                                  filter_ind+1,filter_ind-1,
                                  filter_ind+2,filter_ind-2,
                                  filter_ind+3,filter_ind-3,
                                 ] )
    filter_ind = filter_ind[filter_ind<100000]
        
    varf[filter_ind] = 0
    tmp['signal_filter'] = np.fft.ifft( varf ).real + tmp["signal"].round()
    
    TRAIN.append( tmp )
    
TRAIN = pd.concat( TRAIN, sort=True )
#TRAIN.sort_values( 'time', inplace=True )
TRAIN = TRAIN.reset_index( drop=True )
TRAIN = TRAIN[['time','signal','signal_filter','open_channels']].copy()
TRAIN.head()


# In[ ]:


print( f1_score( TRAIN.open_channels, np.clip(TRAIN.signal.round(),0,10)        , average='macro' ) )
print( f1_score( TRAIN.open_channels, np.clip(TRAIN.signal_filter.round(),0,10) , average='macro' ) )


# In[ ]:


TRAIN['group'] = np.arange(TRAIN.shape[0])//500_000
TRAIN['hit1'] = 1*(TRAIN.open_channels == TRAIN.signal.round())
TRAIN['hit2'] = 1*(TRAIN.open_channels == TRAIN.signal_filter.round())
TRAIN.groupby( 'group' )[['hit1','hit2']].mean()


# In[ ]:


TEST = []
for i, dt in test.groupby( 'group' ):
    tmp = dt.copy()
    varf = np.fft.fft( tmp['gaussian_noise'].values )
    mmax = np.max( np.abs( varf ) )

    varf = np.fft.fft( tmp['noise_round'].values )
    filter_ind = np.where( np.abs(varf)>= mmax )[0]
    print( i , dt.shape, len(filter_ind) )
    
    filter_ind = np.concatenate( [filter_ind,
                                  filter_ind+1,filter_ind-1,
                                  filter_ind+2,filter_ind-2,
                                  filter_ind+3,filter_ind-3,
                                 ] )
    filter_ind = filter_ind[filter_ind<100000]
    
    varf[filter_ind] = 0
    tmp['signal_filter'] = np.fft.ifft( varf ).real + tmp["signal"].round()
    
    TEST.append( tmp )
    
TEST = pd.concat( TEST, sort=True )
#TEST.sort_values( 'time', inplace=True )
TEST = TEST.reset_index( drop=True )
TEST = TEST[['time','signal','signal_filter']].copy()
TEST.head()


# In[ ]:


TRAIN['signal'] = TRAIN['signal_filter'].values
TEST ['signal'] = TEST ['signal_filter'].values
TRAIN[['time','signal','open_channels']].to_csv( 'train_clean_dropfreq.csv.zip', index=False, compression='zip' )
TEST[['time','signal']].to_csv( 'test_clean_dropfreq.csv.zip', index=False, compression='zip' )


# In[ ]:


get_ipython().system('ls -l ../working')


# In[ ]:





# # Now create augmented trainset using shifted original noise

# In[ ]:


train = pd.read_csv( 'train_clean_dropfreq.csv.zip', dtype={'time':'str','signal':'float','open_channels':'int'}, compression='zip' )
train['group'] = np.arange(train.shape[0])//500_000
train['noise'] = train['signal'] - train['open_channels']
train.groupby( 'group' )['noise'].std()


# In[ ]:


for N in range(5):
    np.random.seed(N*13+11)
    for i in range(10):
        noise = train.loc[ train.group==i,'noise'].values.copy()
        
        if i==7:
            noise = np.concatenate( [noise[:142000],noise[-177000:] ] )#Filter just the good part of noise for group 7
        noise = np.concatenate( [noise,noise,noise,noise] )#Augment with inverse negative noise
        window = np.random.randint(0, len(noise)-500000,1 )[0]
        noise = noise[ window:(window+500000) ]
        
        train.loc[ train.group==i,'noise'] = noise
    
    train['signal'] = train['open_channels'] + train['noise']    
    print( f1_score( train.open_channels, np.clip(train.signal.round(),0,10) , average='macro' ) )
    
    train[['time','signal','open_channels']].to_csv('train-add-noise-'+str(N)+'.csv.zip', index=False, compression='zip' )


# In[ ]:


train.groupby( 'group' )['noise'].std()


# In[ ]:


get_ipython().system('ls -l ../working')


# In[ ]:





# # Now create trainset using custom FSM

# In[ ]:


train = pd.read_csv( 'train_clean_dropfreq.csv.zip', dtype={'time':'str','signal':'float','open_channels':'int'}, compression='zip' )
train['group'] = np.arange(train.shape[0])//500_000
train['noise'] = train['signal'] - train['open_channels']
train.groupby( 'group' )['noise'].std()


# In[ ]:


class StateMachineA:
    def __init__(self):
        self.cursts = 'C1'
        
        self.outputs = {}
        self.outputs['C1'] = 0
        self.outputs['C2'] = 0
        self.outputs['C3'] = 0
        self.outputs['O1'] = 1
        self.outputs['O2'] = 1
        
        self.states = {}
        self.states['C1'] = { 'C1':0.9997786 ,'O2':0.0002214, 'O1':0.00000   , 'C2':0.00000  , 'C3':0.00000 }
        self.states['C2'] = { 'C1':0.000000  ,'O2':0.3814155, 'O1':0.00000   , 'C2':0.6185845, 'C3':0.00000 }
        self.states['C3'] = { 'C1':0.000000  ,'O2':0.000000 , 'O1':0.3237174 , 'C2':0.00000  , 'C3':0.6762826 }
        self.states['O1'] = { 'C1':0.000000  ,'O2':0.761767 , 'O1':0.0499013 , 'C2':0.00000  , 'C3':0.1883317 }
        self.states['O2'] = { 'C1':0.0133812 ,'O2':0.0014535, 'O1':0.8260250 , 'C2':0.1591403, 'C3':0.00000 }          
        
        self.choices = {}
        self.choices['C1'] = ['C1'] * int( self.states['C1']['C1']*1000000 ) + ['C2'] * int( self.states['C1']['C2']*1000000 ) + ['C3'] * int( self.states['C1']['C3']*1000000 ) + ['O1'] * int( self.states['C1']['O1']*1000000 ) + ['O2'] * int( self.states['C1']['O2']*1000000 )
        self.choices['C2'] = ['C1'] * int( self.states['C2']['C1']*1000000 ) + ['C2'] * int( self.states['C2']['C2']*1000000 ) + ['C3'] * int( self.states['C2']['C3']*1000000 ) + ['O1'] * int( self.states['C2']['O1']*1000000 ) + ['O2'] * int( self.states['C2']['O2']*1000000 )
        self.choices['C3'] = ['C1'] * int( self.states['C3']['C1']*1000000 ) + ['C2'] * int( self.states['C3']['C2']*1000000 ) + ['C3'] * int( self.states['C3']['C3']*1000000 ) + ['O1'] * int( self.states['C3']['O1']*1000000 ) + ['O2'] * int( self.states['C3']['O2']*1000000 )
        self.choices['O1'] = ['C1'] * int( self.states['O1']['C1']*1000000 ) + ['C2'] * int( self.states['O1']['C2']*1000000 ) + ['C3'] * int( self.states['O1']['C3']*1000000 ) + ['O1'] * int( self.states['O1']['O1']*1000000 ) + ['O2'] * int( self.states['O1']['O2']*1000000 )
        self.choices['O2'] = ['C1'] * int( self.states['O2']['C1']*1000000 ) + ['C2'] * int( self.states['O2']['C2']*1000000 ) + ['C3'] * int( self.states['O2']['C3']*1000000 ) + ['O1'] * int( self.states['O2']['O1']*1000000 ) + ['O2'] * int( self.states['O2']['O2']*1000000 )
        
    def run(self):
        randint = np.random.randint( 0, len(self.choices[self.cursts]), 1 )[0]
        #Next State
        self.cursts = self.choices[self.cursts][ randint ]
        return self.outputs[self.cursts]


# In[ ]:


class StateMachineB:
    def __init__(self):
        self.cursts = 'C1'

        self.outputs = {}
        self.outputs['C1'] = 0
        self.outputs['C2'] = 0
        self.outputs['C3'] = 0
        self.outputs['O1'] = 1
        self.outputs['O2'] = 1
        self.outputs['O3'] = 1

        self.states = {}
        self.states['C1'] = { 'C1':0.718306,'C2':0.0223246, 'C3':0.00000 , 'O1':0.2593693 , 'O2':0.00000  , 'O3':0.00000 }
        self.states['C2'] = { 'C1':0.172716,'C2':0.226948 , 'C3':0.600336, 'O1':0.00000   , 'O2':0.00000  , 'O3':0.00000 }
        self.states['C3'] = { 'C1':0.000000,'C2':0.317792 , 'C3':0.682208, 'O1':0.00000   , 'O2':0.00000  , 'O3':0.00000 }

        self.states['O1'] = { 'C1':0.297278,'C2':0.000000 , 'C3':0.00000 , 'O1':0.6483142 , 'O2':0.0544078, 'O3':0.00000 }
        self.states['O2'] = { 'C1':0.000000,'C2':0.000000 , 'C3':0.00000 , 'O1':0.0565354 , 'O2':0.085590 , 'O3':0.8578746}
        self.states['O3'] = { 'C1':0.000000,'C2':0.000000 , 'C3':0.00000 , 'O1':0.00000   , 'O2':0.305837 , 'O3':0.694163}

        self.choices = {}
        self.choices['C1'] = ['C1'] * int( self.states['C1']['C1']*9000000 ) + ['C2'] * int( self.states['C1']['C2']*9000000 ) + ['C3'] * int( self.states['C1']['C3']*9000000 ) + ['O1'] * int( self.states['C1']['O1']*9000000 ) + ['O2'] * int( self.states['C1']['O2']*9000000 ) + ['O3'] * int( self.states['C1']['O3']*9000000 )
        self.choices['C2'] = ['C1'] * int( self.states['C2']['C1']*9000000 ) + ['C2'] * int( self.states['C2']['C2']*9000000 ) + ['C3'] * int( self.states['C2']['C3']*9000000 ) + ['O1'] * int( self.states['C2']['O1']*9000000 ) + ['O2'] * int( self.states['C2']['O2']*9000000 ) + ['O3'] * int( self.states['C2']['O3']*9000000 )
        self.choices['C3'] = ['C1'] * int( self.states['C3']['C1']*9000000 ) + ['C2'] * int( self.states['C3']['C2']*9000000 ) + ['C3'] * int( self.states['C3']['C3']*9000000 ) + ['O1'] * int( self.states['C3']['O1']*9000000 ) + ['O2'] * int( self.states['C3']['O2']*9000000 ) + ['O3'] * int( self.states['C3']['O3']*9000000 )
        self.choices['O1'] = ['C1'] * int( self.states['O1']['C1']*9000000 ) + ['C2'] * int( self.states['O1']['C2']*9000000 ) + ['C3'] * int( self.states['O1']['C3']*9000000 ) + ['O1'] * int( self.states['O1']['O1']*9000000 ) + ['O2'] * int( self.states['O1']['O2']*9000000 ) + ['O3'] * int( self.states['O1']['O3']*9000000 )
        self.choices['O2'] = ['C1'] * int( self.states['O2']['C1']*9000000 ) + ['C2'] * int( self.states['O2']['C2']*9000000 ) + ['C3'] * int( self.states['O2']['C3']*9000000 ) + ['O1'] * int( self.states['O2']['O1']*9000000 ) + ['O2'] * int( self.states['O2']['O2']*9000000 ) + ['O3'] * int( self.states['O2']['O3']*9000000 )
        self.choices['O3'] = ['C1'] * int( self.states['O3']['C1']*9000000 ) + ['C2'] * int( self.states['O3']['C2']*9000000 ) + ['C3'] * int( self.states['O3']['C3']*9000000 ) + ['O1'] * int( self.states['O3']['O1']*9000000 ) + ['O2'] * int( self.states['O3']['O2']*9000000 ) + ['O3'] * int( self.states['O3']['O3']*9000000 )

    def run(self):
        randint = np.random.randint( 0, len(self.choices[self.cursts]), 1 )[0]
        #Next State
        self.cursts = self.choices[self.cursts][ randint ]
        return self.outputs[self.cursts]


# In[ ]:


for N in range(1):
    print( N )
    
    np.random.seed(N*17+19)
    
    print('.')
    fsm = StateMachineA()
    tmp0 = np.array( [fsm.run() for i in range(500000)] )
    train.loc[ train.group==0, 'open_channels' ] = tmp0
    
    print('.')
    fsm = StateMachineA()
    tmp1 = np.array( [fsm.run() for i in range(500000)] )
    train.loc[ train.group==1, 'open_channels' ] = tmp1
    
    print('.')
    fsm = StateMachineB()
    tmp2 = np.array( [fsm.run() for i in range(500000)] )
    train.loc[ train.group==2, 'open_channels' ] = tmp2
    
    print('.')
    fsm = StateMachineB()
    tmp3 = np.array( [fsm.run() for i in range(500000)] )
    train.loc[ train.group==6, 'open_channels' ] = tmp3
    
    print('.')
    fsm = StateMachineB()
    tmp0 = np.array( [fsm.run() for i in range(500000)] )
    tmp1 = np.array( [fsm.run() for i in range(500000)] )
    tmp2 = np.array( [fsm.run() for i in range(500000)] )
    train.loc[ train.group==3, 'open_channels' ] = tmp0+tmp1+tmp2
    
    print('.')
    fsm = StateMachineB()
    tmp0 = np.array( [fsm.run() for i in range(500000)] )
    tmp1 = np.array( [fsm.run() for i in range(500000)] )
    tmp2 = np.array( [fsm.run() for i in range(500000)] )
    train.loc[ train.group==7, 'open_channels' ] = tmp0+tmp1+tmp2
    
    print('.')
    fsm = StateMachineB()
    tmp0 = np.array( [fsm.run() for i in range(500000)] )
    tmp1 = np.array( [fsm.run() for i in range(500000)] )
    tmp2 = np.array( [fsm.run() for i in range(500000)] )
    tmp3 = np.array( [fsm.run() for i in range(500000)] )
    tmp4 = np.array( [fsm.run() for i in range(500000)] )
    train.loc[ train.group==5, 'open_channels' ] = tmp0+tmp1+tmp2+tmp3+tmp4
    
    print('.')
    fsm = StateMachineB()
    tmp0 = np.array( [fsm.run() for i in range(500000)] )
    tmp1 = np.array( [fsm.run() for i in range(500000)] )
    tmp2 = np.array( [fsm.run() for i in range(500000)] )
    tmp3 = np.array( [fsm.run() for i in range(500000)] )
    tmp4 = np.array( [fsm.run() for i in range(500000)] )
    train.loc[ train.group==8, 'open_channels' ] = tmp0+tmp1+tmp2+tmp3+tmp4
    
    print('.')
    fsm = StateMachineB()
    tmp0 = np.array( [fsm.run() for i in range(500000)] )
    tmp1 = np.array( [fsm.run() for i in range(500000)] )
    tmp2 = np.array( [fsm.run() for i in range(500000)] )
    tmp3 = np.array( [fsm.run() for i in range(500000)] )
    tmp4 = np.array( [fsm.run() for i in range(500000)] )
    tmp0 = tmp0+tmp1+tmp2+tmp3+tmp4
    
    tmp5 = np.array( [fsm.run() for i in range(500000)] )
    tmp6 = np.array( [fsm.run() for i in range(500000)] )
    tmp7 = np.array( [fsm.run() for i in range(500000)] )
    tmp8 = np.array( [fsm.run() for i in range(500000)] )
    tmp9 = np.array( [fsm.run() for i in range(500000)] )
    tmp1 = tmp5+tmp6+tmp7+tmp8+tmp9
    train.loc[ train.group==4, 'open_channels' ] = tmp0+tmp1
    
    print('.')
    fsm = StateMachineB()
    tmp0 = np.array( [fsm.run() for i in range(500000)] )
    tmp1 = np.array( [fsm.run() for i in range(500000)] )
    tmp2 = np.array( [fsm.run() for i in range(500000)] )
    tmp3 = np.array( [fsm.run() for i in range(500000)] )
    tmp4 = np.array( [fsm.run() for i in range(500000)] )
    tmp0 = tmp0+tmp1+tmp2+tmp3+tmp4
    
    tmp5 = np.array( [fsm.run() for i in range(500000)] )
    tmp6 = np.array( [fsm.run() for i in range(500000)] )
    tmp7 = np.array( [fsm.run() for i in range(500000)] )
    tmp8 = np.array( [fsm.run() for i in range(500000)] )
    tmp9 = np.array( [fsm.run() for i in range(500000)] )
    tmp1 = tmp5+tmp6+tmp7+tmp8+tmp9
    train.loc[ train.group==9, 'open_channels' ] = tmp0+tmp1
    
    
    #Now add original noise back to open_channels
    for i in range(10):
        noise = train.loc[ train.group==i,'noise'].values.copy()
        oc    = train.loc[ train.group==i,'open_channels'].values.copy()
        
        if i==7:
            noise = np.concatenate( [noise[:142000],noise[-177000:] ] )#Filter just the good part of noise for group 7
        noise = np.concatenate( [noise,noise,noise,noise] )#Augment with inverse negative noise
        window = np.random.randint(0, len(noise)-500000,1 )[0]
        noise = noise[ window:(window+500000) ]
    
        train.loc[ train.group==i,'signal'] = oc + noise    
    
    print( f1_score( train.open_channels, np.clip(train.signal.round(),0,10) , average='macro' ) )
    
    train[['time','signal','open_channels']].to_csv( 'train-fsmAB-artificial-'+str(N)+'.csv.zip' , index=False, compression='zip'  )


# In[ ]:


get_ipython().system('ls -l ../working')


# In[ ]:


i = 0
train0 = pd.read_csv( 'train_clean_dropfreq.csv.zip', compression='zip'  )
train1 = pd.read_csv( 'train-add-noise-'+str(i)+'.csv.zip', compression='zip'  )
train2 = pd.read_csv( 'train-fsmAB-artificial-'+str(i)+'.csv.zip', compression='zip'  )

train0.sort_values( 'time', inplace=True )
train1.sort_values( 'time', inplace=True )
train2.sort_values( 'time', inplace=True )

train0['group'] = np.arange(train0.shape[0])//500_000
train1['group'] = np.arange(train1.shape[0])//500_000
train2['group'] = np.arange(train2.shape[0])//500_000


# In[ ]:


plt.plot( train0.signal.values[:1000] )
plt.plot( train1.signal.values[:1000] )
plt.plot( train2.signal.values[:1000] )


# In[ ]:


train0.groupby('group')['open_channels'].agg(['mean','std','min','max'])


# In[ ]:


train1.groupby('group')['open_channels'].agg(['mean','std','min','max'])


# In[ ]:


train2.groupby('group')['open_channels'].agg(['mean','std','min','max'])


# In[ ]:


print( f1_score( train0.open_channels, np.clip(train0.signal.round(),0,10) , average='macro' ) )
print( f1_score( train1.open_channels, np.clip(train1.signal.round(),0,10) , average='macro' ) )
print( f1_score( train2.open_channels, np.clip(train2.signal.round(),0,10) , average='macro' ) )


# In[ ]:


train0['hit1'] = 1*(train0.open_channels == train0.signal.round())
train0.groupby( 'group' )['hit1'].mean()


# In[ ]:


train1['hit1'] = 1*(train1.open_channels == train1.signal.round())
train1.groupby( 'group' )['hit1'].mean()


# In[ ]:


train2['hit1'] = 1*(train2.open_channels == train2.signal.round())
train2.groupby( 'group' )['hit1'].mean()


# In[ ]:





# In[ ]:





# > # Create Artificial Public Testset

# In[ ]:


# train = pd.read_csv( 'train_clean_dropfreq.csv.zip', dtype={'time':'str','signal':'float','open_channels':'int'}, compression='zip' )
# train['group'] = np.arange(train.shape[0])//500_000
# train['noise'] = train['signal'] - train['open_channels']
# train.groupby( 'group' )['noise'].std()


# In[ ]:


# Public Test
# Group, sum
# 0, 3A (or maybe 4A)
# 1, 3B 
# 2, 5B 
# 3, 3A (or maybe 4A) 
# 4,  B
# 5,10B

# group
# 0     0.191035
# 1     0.214339
# 2     0.223478
# 3     0.190773
# 4     0.193667
# 5     0.267836

# Train
# Group, sum
# 0,  A 
# 1,  A 
# 2,  B 
# 3, 3B 
# 4,10B 
# 5, 5B 
# 6,  B 
# 7, 3B 
# 8, 5B 
# 9,10B 

# group
# 0    0.197584
# 1    0.199169
# 2    0.198097
# 3    0.215502
# 4    0.333838
# 5    0.232669
# 6    0.197967
# 7    0.231789
# 8    0.230806
# 9    0.333575


# In[ ]:


# for N in range(1):
#     print( N )
#     np.random.seed(N*171+191)

#     print('.')
#     fsm = StateMachineA()
#     tmp0 = np.array( [fsm.run() for i in range(500000)] )
#     tmp1 = np.array( [fsm.run() for i in range(500000)] )
#     tmp2 = np.array( [fsm.run() for i in range(500000)] )
#     train.loc[ train.group==0, 'open_channels' ] = tmp0 + tmp1 + tmp2

#     print('.')
#     fsm = StateMachineB()
#     tmp0 = np.array( [fsm.run() for i in range(500000)] )
#     tmp1 = np.array( [fsm.run() for i in range(500000)] )
#     tmp2 = np.array( [fsm.run() for i in range(500000)] )
#     train.loc[ train.group==1, 'open_channels' ] = tmp0 + tmp1 + tmp2

#     print('.')
#     fsm = StateMachineB()
#     tmp0 = np.array( [fsm.run() for i in range(500000)] )
#     tmp1 = np.array( [fsm.run() for i in range(500000)] )
#     tmp2 = np.array( [fsm.run() for i in range(500000)] )
#     tmp3 = np.array( [fsm.run() for i in range(500000)] )
#     tmp4 = np.array( [fsm.run() for i in range(500000)] )
#     train.loc[ train.group==2, 'open_channels' ] = tmp0+tmp1+tmp2+tmp3+tmp4

#     print('.')
#     fsm = StateMachineA()
#     tmp0 = np.array( [fsm.run() for i in range(500000)] )
#     tmp1 = np.array( [fsm.run() for i in range(500000)] )
#     tmp2 = np.array( [fsm.run() for i in range(500000)] )
#     train.loc[ train.group==3, 'open_channels' ] = tmp0 + tmp1 + tmp2

#     print('.')
#     fsm = StateMachineB()
#     tmp0 = np.array( [fsm.run() for i in range(500000)] )
#     train.loc[ train.group==4, 'open_channels' ] = tmp0

#     print('.')
#     fsm = StateMachineB()
#     tmp0 = np.array( [fsm.run() for i in range(500000)] )
#     tmp1 = np.array( [fsm.run() for i in range(500000)] )
#     tmp2 = np.array( [fsm.run() for i in range(500000)] )
#     tmp3 = np.array( [fsm.run() for i in range(500000)] )
#     tmp4 = np.array( [fsm.run() for i in range(500000)] )
#     tmp0 = tmp0+tmp1+tmp2+tmp3+tmp4

#     tmp5 = np.array( [fsm.run() for i in range(500000)] )
#     tmp6 = np.array( [fsm.run() for i in range(500000)] )
#     tmp7 = np.array( [fsm.run() for i in range(500000)] )
#     tmp8 = np.array( [fsm.run() for i in range(500000)] )
#     tmp9 = np.array( [fsm.run() for i in range(500000)] )
#     tmp1 = tmp5+tmp6+tmp7+tmp8+tmp9
#     train.loc[ train.group==5, 'open_channels' ] = tmp0+tmp1


#     #Now add original noise back to open_channels
#     mapnoise = [0,3,8,0,2,4]
#     for i in range(6):
#         noise = train.loc[ train.group==mapnoise[i],'noise'].values.copy()
#         oc    = train.loc[ train.group==i,'open_channels'].values.copy()

#         if i==7:
#             noise = np.concatenate( [noise[:142000],noise[-177000:] ] )#Pick just the good part of noise for group 7
#         noise = np.concatenate( [noise,-1*noise] )#Augment with inverse negative noise
#         noise = np.random.choice( noise, 500000, replace=False )

#         train.loc[ train.group==i,'signal'] = oc + noise    

#     py = np.where( train.group <=5 )[0]
#     print( f1_score( train.open_channels.values[py], np.clip(train.signal.values[py].round(),0,10) , average='macro' ) )

#     train[['time','signal','open_channels']].iloc[py].to_csv( 'train-PublicLB-fsmAB-artificial-'+str(N)+'.csv.zip' , index=False, compression='zip'  )


# In[ ]:





# In[ ]:





# # Create Real Public Test

# In[ ]:


# Public Test
# Group, sum
# 0, 3A (or maybe 4A)
# 1, 3B 
# 2, 5B 
# 3, 3A (or maybe 4A) 
# 4,  B
# 5,10B

# group
# 0     0.191035
# 1     0.214339
# 2     0.223478
# 3     0.190773
# 4     0.193667
# 5     0.267836


# Train
# Group, sum
# 0,  A 
# 1,  A 
# 2,  B 
# 3, 3B 
# 4,10B 
# 5, 5B 
# 6,  B 
# 7, 3B 
# 8, 5B 
# 9,10B 

# group
# 0    0.197584
# 1    0.199169
# 2    0.198097
# 3    0.215502
# 4    0.333838
# 5    0.232669
# 6    0.197967
# 7    0.231789
# 8    0.230806
# 9    0.333575


# In[ ]:


train = pd.read_csv( 'train_clean_dropfreq.csv.zip', dtype={'time':'str','signal':'float','open_channels':'int'}, compression='zip' )
train['group'] = np.arange(train.shape[0])//500_000
train['noise'] = train['signal'] - train['open_channels']

tmp = train.loc[ train.group==7 ].copy()
tmp['noise'] = tmp['signal'] - tmp['open_channels']
tmp['signal'].iloc[142000:323000] = tmp['open_channels'].iloc[142000:323000]
tmp['noise'].iloc[142000:242000] = tmp['noise'].iloc[:100000].values
tmp['noise'].iloc[242000:323000] = tmp['noise'].iloc[323000:404000].values
train.loc[ train.group==7 ] = tmp.copy()

train.groupby( 'group' )['noise'].std()


# In[ ]:


print( f1_score( train.open_channels.values, np.clip(train.signal.round().values,0,10) , average='macro' ) )


# In[ ]:


for N in range(10):
    print( N )
    np.random.seed(N*171+191)
    TRAIN = []

    A_base = pd.concat( (train.loc[ train.group==0 ].copy() , train.loc[ train.group==1 ].copy() , train.loc[ train.group==0 ].copy() , train.loc[ train.group==1 ].copy()) )
    B_base = pd.concat( (train.loc[ train.group==2 ].copy() , train.loc[ train.group==6 ].copy() , train.loc[ train.group==2 ].copy() , train.loc[ train.group==6 ].copy()) )
    B3_base = pd.concat( (train.loc[ train.group==3 ].copy() , train.loc[ train.group==7 ].copy() , train.loc[ train.group==3 ].copy() , train.loc[ train.group==7 ].copy()) )
    B5_base = pd.concat( (train.loc[ train.group==5 ].copy() , train.loc[ train.group==8 ].copy() , train.loc[ train.group==5 ].copy() , train.loc[ train.group==8 ].copy()) )
    B10_base = pd.concat( (train.loc[ train.group==4 ].copy() , train.loc[ train.group==9 ].copy() , train.loc[ train.group==4 ].copy() , train.loc[ train.group==9 ].copy()) )
    
    #B1
    n0,n1 = np.random.randint( 0, B_base.shape[0]-500000, 2 )
    B0 = B_base.iloc[ n0:(n0+500000) ].copy()
    B1 = B_base.iloc[ n1:(n1+500000) ].copy()
    B0['noise1'] = B_base['noise'].iloc[ n1:(n1+500000) ].values.copy()
    NOISE = B0[['noise','noise1']].values
    ind = np.random.randint(0,2, 500000)
    B0['signal'] = [NOISE[i,ind[i]] for i in range(500000) ]
    B0['signal'] = B0['signal'] + B0['open_channels']
    B0 = B0.reset_index(drop=True)
    TRAIN.append( B0[['signal','open_channels']] )
    del B0, B1, NOISE, ind
    gc.collect()    
    
    #A4
    n0,n1,n2,n3 = np.random.randint( 0, A_base.shape[0]-500000, 4  )
    A0 = A_base.iloc[ n0:(n0+500000) ].copy()
    A1 = A_base.iloc[ n1:(n1+500000) ].copy()
    A2 = A_base.iloc[ n2:(n2+500000) ].copy()
    A3 = A_base.iloc[ n3:(n3+500000) ].copy()
    A0['open_channels'] = A0['open_channels'].values+A1['open_channels'].values+A2['open_channels'].values+A3['open_channels'].values
    A0['noise1'] = A1['noise'].values
    A0['noise2'] = A2['noise'].values
    A0['noise3'] = A3['noise'].values
    NOISE = A0[['noise','noise1','noise2','noise3']].values
    ind = np.random.randint(0,4, 500000)
    A0['signal'] = [NOISE[i,ind[i]] for i in range(500000) ]
    A0['signal'] = A0['signal'] + A0['open_channels']
    A0 = A0.reset_index(drop=True)
    TRAIN.append( A0[['signal','open_channels']] )
    del A0, A1, A2, A3, NOISE, ind
    gc.collect()
    
    #A4
    n0,n1,n2,n3 = np.random.randint( 0, A_base.shape[0]-500000, 4  )
    A0 = A_base.iloc[ n0:(n0+500000) ].copy()
    A1 = A_base.iloc[ n1:(n1+500000) ].copy()
    A2 = A_base.iloc[ n2:(n2+500000) ].copy()
    A3 = A_base.iloc[ n3:(n3+500000) ].copy()
    A0['open_channels'] = A0['open_channels'].values+A1['open_channels'].values+A2['open_channels'].values+A3['open_channels'].values
    A0['noise1'] = A1['noise'].values
    A0['noise2'] = A2['noise'].values
    A0['noise3'] = A3['noise'].values
    NOISE = A0[['noise','noise1','noise2','noise3']].values
    ind = np.random.randint(0,4, 500000)
    A0['signal'] = [NOISE[i,ind[i]] for i in range(500000) ]
    A0['signal'] = A0['signal'] + A0['open_channels']
    A0 = A0.reset_index(drop=True)
    TRAIN.append( A0[['signal','open_channels']] )
    del A0, A1, A2, A3, NOISE, ind
    gc.collect()
    
    #B3
    n0,n1,n2 = np.random.randint( 0, A_base.shape[0]-500000, 3 )
    B0 = B_base.iloc[ n0:(n0+500000) ].copy()
    B1 = B_base.iloc[ n1:(n1+500000) ].copy()
    B2 = B_base.iloc[ n2:(n2+500000) ].copy()
    B0['open_channels'] = B0['open_channels'].values+B1['open_channels'].values+B2['open_channels'].values
    B0['noise']  = B3_base['noise'].iloc[ n0:(n0+500000) ].values.copy()
    B0['noise1'] = B3_base['noise'].iloc[ n1:(n1+500000) ].values.copy()
    B0['noise2'] = B3_base['noise'].iloc[ n2:(n2+500000) ].values.copy()
    NOISE = B0[['noise','noise1','noise2']].values
    ind = np.random.randint(0,3, 500000)
    B0['signal'] = [NOISE[i,ind[i]] for i in range(500000) ]
    B0['signal'] = B0['signal'] + B0['open_channels']
    B0 = B0.reset_index(drop=True)
    TRAIN.append( B0[['signal','open_channels']] )
    del B0, B1, B2, NOISE, ind
    gc.collect()
    
    #B5
    n0,n1,n2 = np.random.randint( 0, B_base.shape[0]-500000, 3 )
    B0 = B3_base.iloc[ n0:(n0+500000) ].copy()
    B1 = B_base.iloc[ n1:(n1+500000) ].copy()
    B2 = B_base.iloc[ n2:(n2+500000) ].copy()
    B0['open_channels'] = B0['open_channels'].values+B1['open_channels'].values+B2['open_channels'].values
    B0['noise']  = B5_base['noise'].iloc[ n0:(n0+500000) ].values.copy()
    B0['noise1'] = B5_base['noise'].iloc[ n1:(n1+500000) ].values.copy()
    B0['noise2'] = B5_base['noise'].iloc[ n2:(n2+500000) ].values.copy()
    NOISE = B0[['noise','noise1','noise2']].values
    ind = np.random.randint(0,3, 500000)
    B0['signal'] = [NOISE[i,ind[i]] for i in range(500000) ]
    B0['signal'] = B0['signal'] + B0['open_channels']
    B0 = B0.reset_index(drop=True)
    TRAIN.append( B0[['signal','open_channels']] )
    del B0, B1, B2, NOISE, ind
    gc.collect()    
    
    #B10
    n0,n1,n2,n3 = np.random.randint( 0, B_base.shape[0]-500000, 4 )
    B0 = B5_base.iloc[ n0:(n0+500000) ].copy()
    B1 = B3_base.iloc[ n1:(n1+500000) ].copy()
    B2 = B_base.iloc[ n2:(n2+500000) ].copy()
    B3 = B_base.iloc[ n3:(n3+500000) ].copy()
    B0['open_channels'] = B0['open_channels'].values+B1['open_channels'].values+B2['open_channels'].values+B3['open_channels'].values
    B0['noise']  = B10_base['noise'].iloc[ n0:(n0+500000) ].values.copy()
    B0['noise1'] = B10_base['noise'].iloc[ n1:(n1+500000) ].values.copy()
    B0['noise2'] = B10_base['noise'].iloc[ n2:(n2+500000) ].values.copy()
    B0['noise3'] = B10_base['noise'].iloc[ n3:(n3+500000) ].values.copy()
    NOISE = B0[['noise','noise1','noise2','noise3']].values
    ind = np.random.randint(0,4, 500000)
    B0['signal'] = [NOISE[i,ind[i]] for i in range(500000) ]
    B0['signal'] = B0['signal'] + B0['open_channels']
    B0 = B0.reset_index(drop=True)
    TRAIN.append( B0[['signal','open_channels']] )
    del B0, B1, B2, B3, NOISE, ind
    gc.collect()

    TRAIN = pd.concat( TRAIN )
    TRAIN['time'] = [ "{0:.4f}".format( i/10000 ) for i in range(TRAIN.shape[0]) ]

    TRAIN[['time','signal','open_channels']].to_csv( 'train-PublicLB-mix-real-data-'+str(N)+'.csv.zip' , index=False, compression='zip'  )


# In[ ]:


N = 0
train0 = pd.read_csv( 'train-PublicLB-mix-real-data-'+str(N)+'.csv.zip', compression='zip' )
print( f1_score( train0.open_channels.values, np.clip(train0.signal.round().values,0,10) , average='macro' ) )


# In[ ]:


train0['group'] = np.arange(train0.shape[0])//500_000
train0['noise'] = train0['signal'] - train0['open_channels']
train0.groupby( 'group' )['noise'].std()


# In[ ]:


train.groupby( 'group' )[['signal','open_channels']].agg(['mean','std'])


# In[ ]:


train0.groupby( 'group' )[['signal','open_channels']].agg(['mean','std'])


# In[ ]:


test.groupby( 'group' )[['signal']].agg(['mean','std'])


# In[ ]:




