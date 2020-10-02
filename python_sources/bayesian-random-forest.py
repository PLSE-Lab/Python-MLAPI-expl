#!/usr/bin/env python
# coding: utf-8

# Random Forest Model

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

X_train = pd.read_json("../input/train.json")
X_test = pd.read_json("../input/test.json")


# In[ ]:



X_train.head()


# In[ ]:


X_train.shape


# In[ ]:


X_test.head()


# In[ ]:


X_test.shape


# In[ ]:


sample=pd.read_csv("../input/sample_submission.csv")


# In[ ]:


sample.head()


# In[ ]:


print(check_output(["ls", "../input/images_sample/"]).decode("utf8"))


# In[ ]:


import os
import subprocess as sub
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('../input/images_sample/6811957/') if isfile(join('../input/images_sample/6811957/', f))]
print (onlyfiles)


# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
img=[]
for i in range (0,5):
    img.append(mpimg.imread('../input/images_sample/6811957/'+onlyfiles[i]))
    plt.imshow(img[i])
    fig = plt.figure()
    a=fig.add_subplot()
    


# In[ ]:


type(X_train)


# In[ ]:




X_train.dropna(subset = ['interest_level'])
print(X_train.shape)
print (X_train['interest_level'])


# In[ ]:



grouped = X_train.groupby(['interest_level'])
print(X_train.shape)
print(len(X_train))
print (grouped.size())
#probability
print (grouped.size()/len(X_train))

# P(H), high,low,meidum
P_H=(grouped.size()/len(X_train))
P_H


# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
X_train["num_photos"] = X_train["photos"].apply(len)
X_train["num_features"] = X_train["features"].apply(len)
X_train["num_description_words"] = X_train["description"].apply(lambda x: len(x.split(" ")))
X_train["created"] = pd.to_datetime(X_train["created"])
X_train["created_year"] = X_train["created"].dt.year
X_train["created_month"] = X_train["created"].dt.month
X_train["created_day"] = X_train["created"].dt.day
num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]
X = X_train[num_feats]
y = X_train["interest_level"]

X_train2, X_val, y_train2, y_val = train_test_split(X, y, test_size=0.1)


# In[ ]:


type(X_train['features'])


# In[ ]:



X_test["num_photos"] = X_test["photos"].apply(len)
X_test["num_features"] = X_test["features"].apply(len)
X_test["num_description_words"] = X_test["description"].apply(lambda x: len(x.split(" ")))
X_test["created"] = pd.to_datetime(X_test["created"])
X_test["created_year"] = X_test["created"].dt.year
X_test["created_month"] = X_test["created"].dt.month
X_test["created_day"] = X_test["created"].dt.day
X_test2 = X_test[num_feats]

y_orig = X_train["interest_level"]
y_high=[1 if x=='high' else 0 for x in y_orig]
y_medium=[1 if x=='medium' else 0 for x in y_orig]
y_low=[1 if x=='low' else 0 for x in y_orig]
y=pd.DataFrame({'high':y_high,'medium':y_medium,'low':y_low})
type(y['high'])


# In[ ]:



# Train uncalibrated random forest classifier on whole train and validation
# data and evaluate on test data
#rfmodel = RandomForestClassifier(n_estimators=300)
#rfmodel.fit(X_train2, y_train2)


# In[ ]:


#y_val_pred = rfmodel.predict_proba(X_val)
#log_loss(y_val, y_val_pred)


# **Follow other work of Bayesian method
# 
# 
# ----------
# 
# 
# **

# In[ ]:





# In[ ]:


#predictClasses=['high','medium','low']
#resultNormal={}
#resultBayesian={}


# In[ ]:


"""
for p in predictClasses:
    print('Training model:',p)
    model = RandomForestClassifier(n_estimators=300)
    model.fit(X, y[p])
    # P(m|H)
    P_m_given_H={}
    yHat=model.predict(X.ix[np.array(y[p])==1,:])
    P_m_given_H[0]=sum(yHat==0)*1.0/len(yHat)
    P_m_given_H[1]=sum(yHat==1)*1.0/len(yHat)
    P_m_given_H
    # P(m)
    P_m={}
    yHatAll=model.predict(X)
    P_m[0]=sum(yHatAll==0)*1.0/len(yHatAll)
    P_m[1]=sum(yHatAll==1)*1.0/len(yHatAll)
    
    Factor={}
    Factor[0]=P_m_given_H[0]/P_m[0]
    Factor[1]=P_m_given_H[1]/P_m[1]
    
    print ('Factor :', Factor)
    print ('P(m)   :', P_m)
    print ('P(m|H) :', P_m_given_H)
    
    Hi=sum(y[p]==1)*1.0/len(y[p])
    
    yHat=model.predict(X_test2)
    
    resultNormal[p]=yHat
    resultBayesian[p]=Hi*np.array([Factor[m] for m in yHat])"""


# In[ ]:


"""
resultBayesian['listing_id'] = X_test['listing_id']
resultBayesian = pd.DataFrame(resultBayesian)[['listing_id', 'high', 'medium', 'low']]

resultNormal['listing_id'] = X_test['listing_id']
resultNormal = pd.DataFrame(resultNormal)[['listing_id', 'high', 'medium', 'low']]"""


# In[ ]:


"""
print(resultBayesian.head())
print(resultNormal.head())"""


# **improvement with predict_proba instead of predict**
# -------------------------------------------------

# *try y_high with predict_proba*

# In[ ]:


"""
predictClasses=['high','medium','low']
resultNormal={}
resultBayesian={}
p='high'
print('Training model:',p)
model = RandomForestClassifier(n_estimators=300)
model.fit(X, y[p])"""


# In[ ]:


# P(m|H)
#P_m_given_H={}
#yHat=model.predict_proba(X.ix[np.array(y[p])==1,:])


# In[ ]:


#print(yHat)
#print(yHat[:,1])


# In[ ]:


"""
import scipy.stats
from matplotlib import pyplot as plt
df_0=yHat[:,0]
df_1=yHat[:,1]
pdf_high_1=scipy.stats.kde.gaussian_kde(df_1)
pdf_high_0=scipy.stats.kde.gaussian_kde(df_0)
x=np.linspace(0,1,1000)
#f=pdf(x)
plt.plot(x,pdf_high_1(x))
plt.show()"""


# In[ ]:


#P_m_given_high={}
#P_m_given_high['nonhigh']=pdf_high_0
#P_m_given_high['high']=pdf_high_1


# In[ ]:


#print(P_m_given_high['high'](0.55))
#print(P_m_given_high['nonhigh'](0.55))
#print(P_m_given_high)


# In[ ]:


"""
P_m={}
yHatAll=model.predict_proba(X)
print(yHatAll)
P_m['nonhigh']=scipy.stats.kde.gaussian_kde(yHatAll[:,0])
P_m['high']=scipy.stats.kde.gaussian_kde(yHatAll[:,1])
print(P_m)"""


# In[ ]:


"""
yHat_test=model.predict_proba(X_test2)
print(yHat_test)"""


# In[ ]:


"""
print(yHat_test[0,0])
print(yHat_test[0,1])"""


# In[ ]:


"""
[P_m_given_high['nonhigh'](yHat_test[0,0])/P_m['nonhigh'](yHat_test[0,0]),
P_m_given_high['high'](yHat_test[0,1])/P_m['high'](yHat_test[0,1])]"""


# In[ ]:


"""
predictClasses=['high','medium','low']
resultNormal={}
resultBayesian={}
for p in predictClasses:
    print('Training model:',p)
    model = RandomForestClassifier(n_estimators=300)
    model.fit(X, y[p])
    # P(m|H)
    P_m_given_H={}
    yHat=model.predict_proba(X.ix[np.array(y[p])==1,:])
    P_m_given_H[0]=np.mean(yHat[:,0])
    P_m_given_H[1]=np.mean(yHat[:,1])
    P_m_given_H
    # P(m)
    P_m={}
    yHatAll=model.predict_proba(X)
    P_m[0]=np.mean(yHatAll[:,0])
    P_m[1]=np.mean(yHatAll[:,1])
    
    Factor={}
    Factor[0]=P_m_given_H[0]/P_m[0]
    Factor[1]=P_m_given_H[1]/P_m[1]
    
    print ('Factor :', Factor)
    print ('P(m)   :', P_m)
    print ('P(m|H) :', P_m_given_H)
    
    Hi=np.mean(y[p]==1)
    
    yHat=model.predict(X_test2)
    
    resultNormal[p]=yHat
    resultBayesian[p]=Hi*np.array([Factor[m] for m in yHat])

resultBayesian['listing_id'] = X_test['listing_id']
resultBayesian = pd.DataFrame(resultBayesian)[['listing_id', 'high', 'medium', 'low']]

resultNormal['listing_id'] = X_test['listing_id']
resultNormal = pd.DataFrame(resultNormal)[['listing_id', 'high', 'medium', 'low']]

print(resultBayesian.head())
print(resultNormal.head())"""


# In[ ]:





# **some improvement with 3 clusters.**
# ---------------------------------

# $P(H_{high}|m)=P(H_{high})*P(m|H_{high})/P(m)$
# Let m=(\hat P(H_high),\hat P(H_medium))

# In[ ]:


X_train2.shape,y_train2.shape,X_val.shape,y_val.shape


# In[ ]:


resultNormal={}
resultBayesian={}
model = RandomForestClassifier(n_estimators=300)
model.fit(X_train2, y_train2)


# In[ ]:


yHat_all=model.predict_proba(X_train2)
yHat_all_nonprob=model.predict(X_train2)


# In[ ]:


yHat_all,yHat_all_nonprob


# In[ ]:


len(yHat_all_nonprob==y_train2),sum(yHat_all_nonprob==y_train2)


# In[ ]:


y_val_m=y_val_m_given_high+y_val_m_given_low+y_val_m_given_medium


# In[ ]:


(np.array(y_train2)=='high',yHat_all_nonprob=='high')


# In[ ]:


type(yHat_all_nonprob)


# In[ ]:


yHat_val=model.predict_proba(X_val)
yHat_val.shape
log_loss(y_val, yHat_val)


# ## Use predict, not predict_prob

# In[ ]:


y_val_nonprob_m_given_high=[yHat_all_nonprob[x] for x in range(len(yHat_all)) if (np.array(y_train2)[x]=='high' and yHat_all_nonprob[x]=='high')]
y_val_nonprob_m_given_low=[yHat_all_nonprob[x] for x in range(len(yHat_all)) if (np.array(y_train2)[x]=='low' and yHat_all_nonprob[x]=='low')]
y_val_nonprob_m_given_medium=[yHat_all_nonprob[x] for x in range(len(yHat_all)) if (np.array(y_train2)[x]=='medium' and yHat_all_nonprob[x]=='medium')]
y_val_nonprob_m=y_val_nonprob_m_given_high+y_val_nonprob_m_given_low+y_val_nonprob_m_given_medium
len(y_val_nonprob_m_given_high),len(y_val_nonprob_m_given_low),len(y_val_nonprob_m_given_medium),len(y_val_nonprob_m)


# In[ ]:


p_m=len(y_val_nonprob_m)/len(y_train2)
p_m_given_high=len(y_val_nonprob_m_given_high)/sum(y_train2=='high')
p_m_given_low=len(y_val_nonprob_m_given_low)/sum(y_train2=='low')
p_m_given_medium=len(y_val_nonprob_m_given_medium)/sum(y_train2=='medium')


# In[ ]:


p_high_given_m=P_H['high']*p_m_given_high/p_m
p_medium_given_m=P_H['medium']*p_m_given_medium/p_m
p_low_given_m=P_H['low']*p_m_given_low/p_m
p_high_given_m,p_medium_given_m,p_low_given_m,p_high_given_m+p_medium_given_m+p_low_given_m


# In[ ]:


resultBayesian_nonprob=[[p_high_given_m,p_low_given_m,p_medium_given_m]]*len(y_val)
log_loss(y_val,yHat_val),log_loss(y_val, resultBayesian_nonprob)


# ## Use predict_prob

# In[ ]:


y_val_m_given_high=[yHat_all[x,0] for x in range(len(yHat_all)) if (np.array(y_train2)[x]=='high' and yHat_all_nonprob[x]=='high')]
y_val_m_given_low=[yHat_all[x,1] for x in range(len(yHat_all)) if (np.array(y_train2)[x]=='low' and yHat_all_nonprob[x]=='low')]
y_val_m_given_medium=[yHat_all[x,2] for x in range(len(yHat_all)) if (np.array(y_train2)[x]=='medium' and yHat_all_nonprob[x]=='medium')]
len(y_val_m_given_high),len(y_val_m_given_low),len(y_val_m_given_medium)


# In[ ]:


len(yHat_all),sum(yHat_all_nonprob==y_train2)


# In[ ]:


np.array(y_train2=='high'),yHat_all_nonprob=='high',(np.array(y_train2=='high'))& (yHat_all_nonprob=='high')


# In[ ]:


import scipy.stats
from matplotlib import pyplot as plt
P_m=scipy.stats.kde.gaussian_kde(y_val_m)
P_m_given_high=scipy.stats.kde.gaussian_kde(y_val_m_given_high)
P_m_given_low=scipy.stats.kde.gaussian_kde(y_val_m_given_low)
P_m_given_medium=scipy.stats.kde.gaussian_kde(y_val_m_given_medium)


# In[ ]:


resultBayesian['high']


# p_high_given_HatHigh=p_high*p_HatHigh_given_high/p_HatHigh

# p_H_given_m=p_h*p_m_given_h/p_m

# In[ ]:


#test_select=range(100)


# In[ ]:


##factor_rf={}
#factor_rf['high']=[float(P_m_given_high['low'](x))*1.0/float(P_m['low'](x)) for x in yHat_val[:,0]]
#factor_rf['medium']=[float(P_m_given_medium['low'](x))*1.0/float(P_m['low'](x)) for x in yHat_val[:,2]]
#factor_rf['low']=[float(P_m_given_low['low'](x))*1.0/float(P_m['low'](x)) for x in yHat_val[:,1]]


# In[ ]:


##resultBayesian={}
#resultBayesian['high']=[P_H['high'] * x for x in factor_rf['high']]
#resultBayesian['medium']=[P_H['medium'] *x for x in factor_rf['medium']]
#resultBayesian['low']=[P_H['low'] * x for x in factor_rf['low']]
#resultBayesian['listing_id'] = X_test['listing_id']
#resultBayesian = pd.DataFrame(resultBayesian)[['listing_id', 'high', 'medium', 'low']]
#resultBayesian = pd.DataFrame(resultBayesian)[['high', 'low', 'medium']]


# In[ ]:


#round(resultBayesian.head(),2)


# In[ ]:


#log_loss(y_val, resultBayesian)


# In[ ]:


"""Factor_new={}
Bayesian_new={}
for p in predictClasses:
    print('Given H_',p)
    # P(m|H)
    P_m_given_H={}
    yHat=model.predict(X.ix[y_orig==p,:])
    P_m_given_H['high']=sum(yHat=='high')*1.0/len(yHat)
    P_m_given_H['medium']=sum(yHat=='medium')*1.0/len(yHat)
    P_m_given_H['low']=sum(yHat=='low')*1.0/len(yHat)
    print('P(m|H):',P_m_given_H)
    
    # P(m)
    P_m={}
    yHatAll=model.predict(X)
    P_m['high']=sum(yHatAll=='high')*1.0/len(yHatAll)
    P_m['medium']=sum(yHatAll=='medium')*1.0/len(yHatAll)
    P_m['low']=sum(yHatAll=='low')*1.0/len(yHatAll)
    print('P(m):',P_m)
    
    Factor={}
    Factor['high']=P_m_given_H['high']/P_m['high']
    Factor['medium']=P_m_given_H['medium']/P_m['medium']
    Factor['low']=P_m_given_H['low']/P_m['low']
    print('Factor:',Factor)
    
    yHat=model.predict(X_test2)
    resultNormal=yHat
    
    Factor_new[p]=Factor[p]
    
    Bayesian_new[p]=P_H[p]*Factor[p]
print(Factor_new)
print(Bayesian_new)"""


# In[ ]:


"""yHat=model.predict(X_test2)
resultNormal=yHat
resultBayesian=[Bayesian_new[m] for m in yHat]"""


# In[ ]:


#print(resultNormal)
#print(resultBayesian)


# In[ ]:


#plogloss=-np.mean(np.log(resultBayesian))


# In[ ]:




