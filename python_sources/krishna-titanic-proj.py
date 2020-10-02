import pandas as pd
import numpy as np
import csv as csv
import pylab as pl
from sklearn import svm

df=pd.read_csv('/input/train.csv',header=0)
median_ages =np.zeros((2,3))
df['gender']=df['Sex'].map({'female':0,'male':1})
for i in range (0,2):
    for j in range(0,3):
        median_ages[i,j]=df[(df['gender']==i)&(df['Pclass']==j+1)]['Age'].dropna().median()

df['Agefill']=df['Age']        
for i in range (0,2):
    for j in range(0,3):
        df.loc[(df['Age'].isnull())&(df['Pclass']==j+1)&(df['gender']==i)]['Agefill']=median_ages[i,j]

#df=df.drop(df.dtypes[df.dtypes.map(lambda x:x!='object',)],axis=1)
        
#print (df.dtypes)
df=df.drop(['Name','Age','Embarked','Cabin','Sex','Ticket'],axis=1)
print (df.dtypes)
df=df.dropna()
train_data=df.values
print (train_data)

df2=pd.read_csv('/input/test.csv',header=0)
median_ages2 =np.zeros((2,3))
df2['gender']=df2['Sex'].map({'female':0,'male':1})
for i in range (0,2):
    for j in range(0,3):
        median_ages2[i,j]=df2[(df2['gender']==i)&(df2['Pclass']==j+1)]['Age'].dropna().median()

df2['Agefill']=df2['Age']        
for i in range (0,2):
    for j in range(0,3):
        df2.loc[(df2['Age'].isnull())&(df2['Pclass']==j+1)&(df2['gender']==i)]['Agefill']=median_ages2[i,j]

#df=df.drop(df.dtypes[df.dtypes.map(lambda x:x!='object',)],axis=1)
        
#print (df.dtypes)
df2=df2.drop(['Name','Age','Embarked','Cabin','Sex','Ticket'],axis=1)
print (df2.dtypes)
df2=df2.dropna()
test_data=df2.values
print (test_data)        

        









