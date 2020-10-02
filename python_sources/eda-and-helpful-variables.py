#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data0=pd.read_csv("../input/c_0000.csv")


# Lets add some helpful variables.  typically we'd like a "distance" and overall "velocity".  Many moons ago I taught physics labs and may or may not remember my cooridnate transforms. 
# 
# I also added in Angular momentum--"L"  L=r x p (r and p are vectors so its r cross p, p of course is mv). Brief refresher--since we already have (x,y,z) and (vx,vy,vz) and m in constant than L ~ r x v
# so Li=(rj)(vk)(eijk) ( I really need to figure out subscripts)! in markdown
# 
# CAUTION--when using numpy, the np.arctan() function is not the one you want.  When converting back to spherical you need to account for the quadrant of x/y-- aka +x,+y is the same quadrant as -x,-y but the angle should be MUCH different (0 to pi) versus (-pi to 0).  That is why we use arctan2!   Many of you may also see that we use phi on (-pi,pi) rather than (0,2*pi) this is a result of the numpy output of arctan2. 

# In[ ]:


data0['time']=0
data0['r']=np.sqrt(data0.x**2+data0.y**2+data0.z**2)
data0['theta']=np.arccos(data0.z/data0.r)
data0['phi']=np.arctan2(data0.y,data0.x)
data0['v']=np.sqrt(data0.vx**2+data0.vy**2+data0.vz**2)
data0['v_theta']=np.arccos(data0.vz/data0.v)
data0['v_phi']=np.arctan2(data0.vy,data0.vx)
data0['Lx']=data0.y*data0.vz-data0.z*data0.y
data0['Ly']=data0.z*data0.vx-data0.x*data0.vz
data0['Lz']=data0.x*data0.vy-data0.y*data0.vx
data0['L']=np.sqrt(data0.Lx**2+data0.Ly**2+data0.Lz**2)


# Lets get all the data into a big dataframe.

# In[ ]:


for i in range(1,19):
    if i<10:
        step="0"+str(i)
    else:
        step=str(i)
    file="../input/c_"+step+"00.csv"
    data1=pd.read_csv(file)
    data1['r']=np.sqrt(data1.x**2+data1.y**2+data1.z**2)
    data1['theta']=np.arccos(data1.z/data1.r)
    data1['phi']=np.arctan2(data1.y,data1.x)
    data1['v']=np.sqrt(data1.vx**2+data1.vy**2+data1.vz**2)
    data1['v_theta']=np.arccos(data1.vz/data1.v)
    data1['v_phi']=np.arctan2(data1.vy,data1.vx)
    data1['time']=i*100
    data1['Lx']=data1.y*data1.vz-data1.z*data1.y
    data1['Ly']=data1.z*data1.vx-data1.x*data1.vz
    data1['Lz']=data1.x*data1.vy-data1.y*data1.vx
    data1['L']=np.sqrt(data1.Lx**2+data1.Ly**2+data1.Lz**2)
    data0=data0.append(data1)
    


# Unofrtunatly, I did not find anything too exciting.  Both r and v stay nicely distributed (although there are a lot more outliers as time goes on--not enough to see visually--I found them because my "bins" got very thick and curves became jagged.  The direction also stay nicely distributed.  As a reminder "theta" is measured from the +z axis and goes from 0 to pi/2 and phi is measured from +X(y,z=0) counter clockwise.
# 
# The v_phi does show some evolution as it goes from a straight (-pi,pi) distribution to  a definite favor in to (0,pi) interval.  My apologies for the colors, it is late and I am wrapping up.
# 
# https://en.wikipedia.org/wiki/Spherical_coordinate_system
# 
# You may notice some "tails" where the angles exceed pi, that is an artifact of the fit/plotting function.  

# In[ ]:


import seaborn as sns #fancy graphics package
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
sample=data0[data0.r<10] #trims off outliers to give us smooth curves
time=[0,100,200,300,400,500,600,700,800,900,1000,1100,100,1300,1400,1500,1600,1700,1800]
for i in time:
    sns.kdeplot(sample[sample.time==i].r,label=str(i))
plt.xlim(0,10)
plt.show()


# In[ ]:


sample=data0[data0.v<10]
for i in time:
    sns.kdeplot(sample[sample.time==i].v,label=str(i))
plt.xlim(0,2)
plt.show()


# In[ ]:



for i in time:
    sns.kdeplot(data0[data0.time==i].theta,label=str(i))
plt.xlim(0,np.pi+.05)
plt.show()


# In[ ]:


for i in time:
    sns.kdeplot(data0[data0.time==i].v_theta,label=str(i))
plt.xlim(0,3.2)
plt.show()


# In[ ]:


for i in time:
    sns.kdeplot(data0[data0.time==i].phi,label=str(i))

plt.show()


# In[ ]:


sns.kdeplot(data0[data0.time==0].phi,label="initial")
sns.kdeplot(data0[data0.time==1800].phi,label="t=1800")
plt.show()
       


# In[ ]:


for i in time:
    sns.kdeplot(data0[data0.time==i].v_phi,label=str(i))

plt.show()


# In[ ]:


sample=data0[data0.L<5]
for i in time:
    sns.kdeplot(sample[sample.time==i].L,label=str(i))
plt.xlim(0,10)
plt.show()

