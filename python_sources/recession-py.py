#Regression Completed/ Classification Completed / Accuracy displayed

import numpy as np #numpy
import pandas as pd #pandas for processing data
import datetime 
import matplotlib.pyplot as plt # For plotting graphs
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score
import scipy
from sklearn import preprocessing
from sklearn.decomposition import PCA
import random
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))
files=['GOLDPMGBD228NLBM.csv', 'UNRATE.csv', 'DEXUSUK.csv', 'NASDAQCOM.csv', 'DTB3.csv', 'DGS1.csv', 'A191RL1Q225SBEA.csv', 'CPICPIAUCSL.csv']
di={'GOLDPMGBD228NLBM':'GOLD','UNRATE':'UNRATE','DEXUSUK':'EXCH',
    'NASDAQCOM':'STOCK','DTB3':'3M','DGS1':'1Y',
    'A191RL1Q225SBEA':'GDP','CPICPIAUCSL':'CPI','CPIAUCSL':'CPI'}
#renaming into easily understandable Names
p=[]
pdi={}
for i in os.listdir("../input"):
    if i!='DTB3.csv'and i!='NASDAQCOM.csv': 
        # Useless data are removed here+
        p.append(pd.read_csv("../input/"+i))

for i in range(len(p)): p[i]['DATE']=pd.to_datetime(p[i]['DATE'])
# Converting date into python datetime format
for i in range(len(p)): p[i]=p[i].set_index('DATE')
# Setting the data as the index
for i in range(len(p)): p[i]=p[i].rename(columns=di);p[i]=p[i].replace(".",np.nan).astype('float');
# Replacing missing values with NaN and converting the datatype as float
for i in range(len(p)): p[i]=p[i].interpolate();p[i]=p[i].fillna(method='bfill');
#Extrapolating to find missing values

meta_date=[]
meta_days=[]
for i in p: meta_date.append([(i[0:1].index[0].date()),i.columns[0]])
# Starting dates of all columns
for i in p: meta_days.append([(i[1:2].index[0]-i[0:1].index[0]).days,i.columns[0]]) #contains pairs of days and name of the column
# Difference between two dates in all columns
for i in p: pdi[i.columns[0]]=i

meta_days.sort()
meta_date.sort() 
#meta_date[-1][0] is the latest starting date of the dataset
order=[] 
# the oder in which data must be aligned in data frame
for i in meta_days: order.append(i[1])

for i in pdi.keys(): 
    if(i!='GDP'):
        pdi[i]=pdi[i].pct_change(); 
        pdi[i]=pdi[i].interpolate(axis=0)
        pdi[i]=pdi[i].fillna(method='bfill')
# Changing the data as Percentage change from previous year
        
for i in pdi.keys(): 
    pdi[i]=pdi[i][meta_date[-1][0]:];print(pdi[i].describe())
    if(i!='GDP'):
        pdi[i].plot()
        pdi[i]=pd.DataFrame(preprocessing.scale(pdi[i]),index=pdi[i].index,columns=pdi[i].columns)
        pdi[i].plot()
    
# Scaled everything here except GDP as it is the label.

pdi2={}
prev=0
#for all keys except GDP in pdi - dict
for j in meta_days[0:-1]:
	j[1] #contains the key value to pdi in order except GDP
	for i in range(len(pdi['GDP'])-1):
	    temp=pdi[j[1]][pdi[meta_days[-1][1]].index[i:i+1][0]+datetime.timedelta(days=1):pdi[meta_days[-1][1]].index[i+1:i+2][0]].values
	    temp=temp.reshape(temp.shape[1],temp.shape[0]);
	    temp_df=pd.DataFrame.from_records(temp,index=pdi[meta_days[-1][1]].index[i+1:i+2]) if i==0 else temp_df.append(pd.DataFrame.from_records(temp,index=pdi[meta_days[-1][1]].index[i+1:i+2]))
        # Slicing data for every quarter and reshaping the data into 1x60 for every quarter
	for i in temp_df: temp_df[i]=temp_df[i].replace(".",np.nan); temp_df[i]=temp_df[i].astype('float');
	temp_df=tmp_df=temp_df.interpolate(axis=1).fillna(method='backfill',axis=1)
	if(len(temp_df.columns)>60):temp_df=pd.DataFrame.from_records(PCA(n_components=3).fit_transform(temp_df),index=temp_df.index) # for pca
    #PCA for rows having greater than 60 columns.
	ran=temp_df.columns
	ran+=prev
	prev=ran[-1]
	cols={}
	for i in temp_df.columns:cols[i]=ran[i]	
	temp_df=temp_df.rename(columns=cols)
	pdi2[j[1]]=temp_df;print(j[1],temp_df.shape)

p2=[]
for key in pdi2.keys(): p2.append(pdi2[key])
df=pd.concat(p2,axis=1)
print (df)

x_train, x_test, y_train, y_test = train_test_split(df, pdi['GDP'][1:], test_size=0.35, random_state=int(random.random()*100))
regr = LinearRegression().fit(x_train,y_train)
# Training the regression algorithm
neg = [i for i, val in enumerate(regr.predict(x_test)) if val<0]
plt.plot(pd.DataFrame(regr.predict(x_test),index=y_test.index).sort_index(),label='P')
plt.plot(y_test.sort_index(),label='O')
plt.show()
print(np.sum(np.logical_xor(regr.predict(x_test)<0,y_test.values<0)),"are falsely predicted out of ",len(y_test))
print(mean_squared_error(regr.predict(x_test),y_test )," general time")
print(mean_squared_error(regr.predict(x_test)[neg],y_test.values[neg] )," for recession time ",len(neg))