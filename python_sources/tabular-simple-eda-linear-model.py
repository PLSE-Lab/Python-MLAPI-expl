#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from statsmodels.formula.api import quantreg

import os


# In[ ]:


train = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/train.csv' )
test  = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/test.csv' )

train['traintest'] = 0
test ['traintest'] = 1

sub   = pd.read_csv( '../input/osic-pulmonary-fibrosis-progression/sample_submission.csv' )
sub['Weeks']   = sub['Patient_Week'].apply( lambda x: int(x.split('_')[-1]) )
sub['Patient'] = sub['Patient_Week'].apply( lambda x: x.split('_')[0] ) 

print( train.shape, test.shape, sub.shape )

sub.head()


# In[ ]:


train.tail(10)


# In[ ]:


test.tail(10)


# # Train have 176 patients and Test only 5
# # Welcome to the uncertainty and LB shake up world!

# In[ ]:


train.Patient.nunique(), sub.Patient.nunique()


# In[ ]:


sub.Patient.isin( test.Patient.unique() ).mean()


# # Concatenate the 5 samples we have in test with the train set

# In[ ]:


train = pd.concat( (train,test) )
train.sort_values( ['Patient','Weeks'], inplace=True )
train.shape


# In[ ]:


train.describe()


# In[ ]:


train['Age'].hist( bins=100, figsize=(5, 5) )


# In[ ]:


train['Percent'].hist( bins=100, figsize=(5, 5) )


# In[ ]:


train['FVC'].hist( bins=100, figsize=(5, 5) )


# In[ ]:


train.groupby( ['Sex','SmokingStatus'] )['FVC'].agg( ['mean','std','count'] )


# In[ ]:


train.groupby( ['Sex','Age'] )['FVC'].agg( ['mean','std','count'] )


# In[ ]:





# # Label encode Strings

# In[ ]:


train['Sex']           = pd.factorize( train['Sex'] )[0]
train['SmokingStatus'] = pd.factorize( train['SmokingStatus'] )[0]
train


# # Manual mean removal + std

# In[ ]:


train['Percent']       = (train['Percent'] - train['Percent'].mean()) / train['Percent'].std()
train['Age']           = (train['Age'] - train['Age'].mean()) / train['Age'].std()
train['Sex']           = (train['Sex'] - train['Sex'].mean()) / train['Sex'].std()
train['SmokingStatus'] = (train['SmokingStatus'] - train['SmokingStatus'].mean()) / train['SmokingStatus'].std()
train.head(10)


# # Fit quantreg models

# In[ ]:


modelL = quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', train).fit( q=0.15 )
model  = quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', train).fit( q=0.50 )
modelH = quantreg('FVC ~ Weeks+Percent+Age+Sex+SmokingStatus', train).fit( q=0.85 )
print(model.summary())


# In[ ]:


train['ypredL'] = modelL.predict( train ).values
train['ypred']  = model.predict( train ).values
train['ypredH'] = modelH.predict( train ).values
train['ypredstd'] = 0.5*np.abs(train['ypredH'] - train['ypred'])+0.5*np.abs(train['ypred'] - train['ypredL'])
train.head(10)


# # Calculate competition metric

# In[ ]:


def metric( trueFVC, predFVC, predSTD ):
    
    clipSTD = np.clip( predSTD, 70 , 9e9 )  
    
    deltaFVC = np.clip( np.abs(trueFVC-predFVC), 0 , 1000 )  

    return np.mean( -1*(np.sqrt(2)*deltaFVC/clipSTD) - np.log( np.sqrt(2)*clipSTD ) )
    

print( 'Metric:', metric( train['FVC'].values, train['ypred'].values, train['ypredstd'].values  ) )


# In[ ]:





# # Merge features to test set

# In[ ]:


dt = train.loc[ train.traintest==1 ,['Patient','Percent','Age','Sex','SmokingStatus']]
test = pd.merge( sub, dt, on='Patient', how='left' )
test.sort_values( ['Patient','Weeks'], inplace=True )
test.head(10)


# In[ ]:


test['ypredL'] = modelL.predict( test ).values
test['FVC']    = model.predict( test ).values
test['ypredH'] = modelH.predict( test ).values
test['Confidence'] = np.abs(test['ypredH'] - test['ypredL']) / 2

test.head(10)


# In[ ]:


test[['Patient_Week','FVC','Confidence']].to_csv('submission.csv', index=False)
test[['Patient_Week','FVC','Confidence']].head(10)


# In[ ]:


test.loc[ test.Patient=='ID00419637202311204720264'  ].plot( x='Weeks', y='FVC' )
test.loc[ test.Patient=='ID00421637202311550012437'  ].plot( x='Weeks', y='FVC' )
test.loc[ test.Patient=='ID00422637202311677017371'  ].plot( x='Weeks', y='FVC' )
test.loc[ test.Patient=='ID00423637202312137826377'  ].plot( x='Weeks', y='FVC' )
test.loc[ test.Patient=='ID00426637202313170790466'  ].plot( x='Weeks', y='FVC' )


# In[ ]:


test.loc[ test.Patient=='ID00419637202311204720264'  ].plot( x='Weeks', y='Confidence' )
test.loc[ test.Patient=='ID00421637202311550012437'  ].plot( x='Weeks', y='Confidence' )
test.loc[ test.Patient=='ID00422637202311677017371'  ].plot( x='Weeks', y='Confidence' )
test.loc[ test.Patient=='ID00423637202312137826377'  ].plot( x='Weeks', y='Confidence' )
test.loc[ test.Patient=='ID00426637202313170790466'  ].plot( x='Weeks', y='Confidence' )

