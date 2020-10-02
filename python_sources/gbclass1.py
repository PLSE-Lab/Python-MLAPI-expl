#!/usr/bin/env python
# coding: utf-8

# In[ ]:


rom __future__ import division
import string
import numpy as np
from numpy.random import randn
from pandas import Series, DataFrame
import pandas as pd
import csv
import os
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import time
import datetime
from sklearn.ensemble import GradientBoostingClassifier
#%matplotlib inline
#voterlist:list of list which contains classifiers for one cluster
def bicluster(i,j):
    tix = np.array(trainpart['hotel_cluster'].values==i)+np.array(trainpart['hotel_cluster'].values==j)
    tGBtraintarget = (trainpart['hotel_cluster'].values==i)*1
    tGBpara = {'data':GBdata,'feature_names':featurelist,'target':tGBtraintarget,
    'target_names':np.arange(100)}
    tmp = tGBpara['target'][tix]
    if sum(tmp==0)==0:
        tmp[-1] = 0
    tclf = GradientBoostingClassifier(n_estimators=20, learning_rate=1,
    max_depth=2, random_state=0).fit(GBdata[tix], tmp)
    return tclf

def oneclus(n):
    if n<99:
        return bicluster(n,n+1)
    else:
        return bicluster(n,0)
    
def getvoter():
    voterlist = []
    for i in range(100):
        accuracy = []
        clflist = []
        clf  = oneclus(i)
        clflist.append(clf)
        for j in range(100):
            tix = np.array(testpart1['hotel_cluster'].values==i)+np.array(testpart1['hotel_cluster'].values==j)
            accuracy.append( clf.score(testdata1[tix], 1*(testpart1['hotel_cluster'][tix].values==i)) )  
            #must use a testdata that contains true clusters
        accuracy = DataFrame([accuracy],index = ['accuracy']).T
        clusix = accuracy.sort_values( by ='accuracy',ascending = True).index[:4]
        tclf = clf
        for ind in clusix:
            tclf = bicluster(i,ind)    
            clflist.append(tclf)
        voterlist.append(clflist)
    return voterlist
    
def GBvote(testdata,voterlist):
    clusprob = []
    now = datetime.datetime.now()
    path = 'submission_GB_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    out = open(path, "w")
    out.write("id,hotel_cluster\n")
    m = len(voterlist[0])
    for i in range(100):
        print('1----'+str(i))
        clflist = voterlist[i]
        tmp = np.zeros([len(testdata),2])
        for j in range(m):    #compute the probability given by evevey clf
            clf = clflist[j]
            tmp = tmp + clf.predict_proba(testdata)
        tmp = tmp/m    
        tmp = (tmp[:,1]>0.5)*tmp[:,1]    #total probability for belonging to cluster i
        clusprob.append(tmp)
    clusprob = np.array(clusprob)
    for i in range(len(testdata)):
        if i%20000 == 0:
            print('2----'+str(i))
        clus = []
        a = clusprob[:,i]
        b=np.argsort(a)[-5:]
        #clusprob.drop(i,axis = 1)
        for ind in b:
            clus.append(str(ind))
        out.write(str(i)+","+"\t".join(clus)+"\n")


# In[ ]:





# In[ ]:


file = open("../input/train.csv")
fout = open('subset_datatest.csv','w')
n = 0
for line in file:
    if n == 0:
        fout.write(line)
    if n <400000*5:
        n +=1
    elif 400000*5<=n <400000*10:
        n +=1
        fout.write(line)
    else:
        break
fout.close()
file.close()
file = open("../input/train.csv")
fout = open('subset_datatrain.csv','w')
n = 0
for line in file:
    if n <400000*5:
        n +=1
        fout.write(line)
    else:
        break
fout.close()
file.close()


# In[ ]:


dategroup = ['2013Jan','2013May','2013Sep','2014Jan','2014May','2014Sep','2015Jan','2015May','2015Sep']
chingroup = ['2013Jan','2013May','2013Sep','2014Jan','2014May','2014Sep','2015Jan','2015May','2015Sep']
dateix = [[] for i in range(9)]
chinix = [[] for i in range(9)]
timeframe = pd.read_csv('subset_datatest.csv',na_values=['--  '],usecols = ['date_time','srch_ci'])
datetime = pd.to_datetime(timeframe['date_time'].values)
datetime = Series(np.arange(len(datetime)),index = datetime)
for i in range(36):
    y = divmod(i,12)[0]
    r = divmod(i,12)[1]
    n = divmod(i,4)[0]
    if r<9:
        dateix[n].extend(datetime['201'+str(3+y)+'-0'+str(r+1)].values)
    else:
        dateix[n].extend(datetime['201'+str(3+y)+'-'+str(r+1)].values)


# In[ ]:


1


# In[ ]:


chinix = [[] for i in range(9)]
citime = pd.to_datetime(timeframe['srch_ci'].values)
citime = Series(np.arange(len(citime)),index = citime)
for i in range(36):
    y = divmod(i,12)[0]
    r = divmod(i,12)[1]
    n = divmod(i,4)[0]
    if r<9:
        chinix[n].extend(citime['201'+str(3+y)+'-0'+str(r+1)].values)
    else:
        chinix[n].extend(citime['201'+str(3+y)+'-'+str(r+1)].values)


# In[ ]:


timeframe2 = pd.read_csv('subset_datatrain.csv',na_values=['--  '],usecols = ['date_time','srch_ci'])
chinix2 = [[] for i in range(9)]
citime2 = pd.to_datetime(timeframe2['srch_ci'].values)
citime2 = Series(np.arange(len(citime2)),index = citime2)
for i in range(36):
    y = divmod(i,12)[0]
    r = divmod(i,12)[1]
    n = divmod(i,4)[0]
    if r<9:
        chinix2[n].extend(citime2['201'+str(3+y)+'-0'+str(r+1)].values)
    else:
        chinix2[n].extend(citime2['201'+str(3+y)+'-'+str(r+1)].values)


# In[ ]:


timeframe2 = pd.read_csv('subset_datatrain.csv',na_values=['--  '],usecols = ['date_time','srch_ci'])
chinix2 = [[] for i in range(9)]
citime2 = pd.to_datetime(timeframe2['srch_ci'].values)
citime2 = Series(np.arange(len(citime2)),index = citime2)
for i in range(36):
    y = divmod(i,12)[0]
    r = divmod(i,12)[1]
    n = divmod(i,4)[0]
    if r<9:
        chinix2[n].extend(citime2['201'+str(3+y)+'-0'+str(r+1)].values)
    else:
        chinix2[n].extend(citime2['201'+str(3+y)+'-'+str(r+1)].values)


# In[ ]:


1


# In[ ]:


featurelist = ['user_location_city','srch_destination_id','hotel_continent','srch_ci']
whlist = ['user_location_city','srch_destination_id','hotel_continent','srch_ci','hotel_cluster']
trainpart = pd.read_csv('subset_datatrain.csv',na_values=['--  '],usecols = whlist)
for i in range(9):
    trainpart['srch_ci'].values[chinix2[i]] = i
GBdata = trainpart[featurelist].values
GBpara = {'data':GBdata,'feature_names':featurelist,'target':trainpart['hotel_cluster'].values,
'target_names':np.arange(100)}

testpart = pd.read_csv('../input/test.csv',na_values=['--  '],usecols = featurelist)
for i in range(9):
    testpart['srch_ci'].values[chinix[i]] = i
testdata = testpart[featurelist].values
testpart1 = pd.read_csv('subset_datatest.csv',na_values=['--  '],usecols = whlist)
testdata1 = testpart1[featurelist].values


# In[ ]:


voterlist = getvoter()
del testpart1
del testdata1
del GBdata
del GBpara
del trainpart
del testdata
#GBvote(testpart.values,voterlist)


# In[ ]:


testdata = testpart.values
clusprob = []
now = datetime.datetime.now()
path = 'submission_GB_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
out = open(path, "w")
out.write("id,hotel_cluster\n")
m = len(voterlist[0])
for i in range(100):
    print('1----'+str(i))
    clflist = voterlist[i]
    tmp = np.zeros([len(testdata),2])
    for j in range(m):    #compute the probability given by evevey clf
        clf = clflist[j]
        tmp = tmp + clf.predict_proba(testdata)
    tmp = tmp/m    
    tmp = (tmp[:,0]>0.5)*tmp[:,0]    #total probability for belonging to cluster i
    clusprob.append(tmp)
clusprob = np.array(clusprob)
for i in range(len(testdata)):
    if i%1000 == 0:
        print('2----'+str(i))
    clus = []
    a = DataFrame(clusprob[:,i])
    b=a.sort_values(by = 0,ascending = False).index[:5]
    #clusprob.drop(i,axis = 1)
    for ind in b:
        clus.append(str(ind))
    out.write(str(i)+","+"\t".join(clus)+"\n")


# In[ ]:


1


# In[ ]:


for i in range(len(testdata)):
    if i%1000 == 0:
        print('2----'+str(i))
    clus = []
    a = DataFrame(clusprob[:,i])
    b=a.sort_index(by = 0,ascending = False).index[:5]
    #clusprob.drop(i,axis = 1)
    for ind in b:
        clus.append(str(ind))
    out.write(str(i)+","+"\t".join(clus)+"\n")

