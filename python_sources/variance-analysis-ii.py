#!/usr/bin/env python
# coding: utf-8

# This is a Markowitz Lagrange relaxation solution
# ---
# i don't care the LB score, since its a pure theoretical model

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb 
from sklearn.metrics import r2_score

get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.display import display, HTML
# Shows all columns of a dataframe
def show_dataframe(X, rows = 5):
    display(HTML(X.to_html(max_rows=rows)))

# Datasets
train = pd.read_csv('../input/train.csv')


# Variance - markowitz optimisation

# In[ ]:


def vcm(X_):
    M11=np.ones((len(X_),len(X_)))
    X_a=X_-M11.dot(X_)/len(X_)
    return X_a.T.dot(X_a)/len(X_)

def vcminv(X_):
    kolom=X_.columns
    ym_=X_.mean(axis=0)
    X_vcm=pd.DataFrame( X_.cov()*2,index=kolom,columns=kolom) 
    X_vcm['y_']=ym_
    X_vcm['lamb']=1.0    
    X_vcmT=X_vcm.T
    X_vcmT['y_']=ym_.append(pd.DataFrame([0,0],index=['lamb','y_']))
    X_vcmT['lamb']=1.0    
    X_vcmT['lamb']['lamb']=0.0
    X_vcmT['y_']['lamb']=0.0
    X_vcmT['lamb']['y_']=0.0    
    X_vcmT.dropna(axis=(0,1), how='any')
    show_dataframe(X_vcmT)
    koloml=X_vcmT.columns
    return pd.DataFrame( np.linalg.pinv( X_vcmT) ,index=koloml,columns=koloml )


# X0 has low variability and explains very good the testing time
# ---
# see red+ versus the 200 cars test group

# In[ ]:


divisor=201
#create groups of divisor length
train['groep']=train.index/divisor-.499
train['groep']=train['groep'].round(0)
train['teller']=1.0

#groep mean - standarddeviation
me_st=train[['y','groep']].groupby(by='groep').mean()
me_st['v']=train[['y','groep']].groupby(by='groep').var()
#print(me_st)

#X0 mean - standarddeviation
Xvar='X0'
X0_me_st=train[['y',Xvar]].groupby(by=Xvar).mean()
X0_me_st['v']=train[['y',Xvar]].groupby(by=Xvar).var()


#print(X0_me_st)
Xvar='X1'
X8_me_st=train[['y',Xvar]].groupby(by=Xvar).mean()
X8_me_st['v']=train[['y',Xvar]].groupby(by=Xvar).var()
#print(X0_me_st)

import matplotlib.pyplot as plt
import seaborn as sns


plt.title('TESTTime Variability Group o X0 * X1 +')
plt.scatter(y=me_st['y'], x=me_st['v'], marker='o', alpha=0.5)
plt.scatter(y=X0_me_st['y'], x=X0_me_st['v'], marker='*', alpha=0.5)
plt.scatter(y=X8_me_st['y'], x=X8_me_st['v'], marker='+', alpha=0.5)
plt.xlabel('Variance'); plt.ylabel('Mean TEST Time')


# X3 explains better the variation
# ----

# In[ ]:


divisor=201
#create groups of divisor length
train['groep']=train.index/divisor-.499
train['groep']=train['groep'].round(0)
train['teller']=1.0

#groep mean - standarddeviation
me_st=train[['y','groep']].groupby(by='groep').mean()
me_st['v']=train[['y','groep']].groupby(by='groep').var()
#print(me_st)

#X0 mean - standarddeviation
Xvar='X2'
X0_me_st=train[['y',Xvar]].groupby(by=Xvar).mean()
X0_me_st['v']=train[['y',Xvar]].groupby(by=Xvar).var()

Xvar='X3'
X8_me_st=train[['y',Xvar]].groupby(by=Xvar).mean()
X8_me_st['v']=train[['y',Xvar]].groupby(by=Xvar).var()
#print(X0_me_st)

import matplotlib.pyplot as plt
import seaborn as sns


plt.title('TESTTime Variability Group o X2 * X3 + ')
#plt.scatter(y=portRet, x=portSTD, marker='.', alpha=0.5)
plt.scatter(y=me_st['y'], x=me_st['v'], marker='o', alpha=0.5)
plt.scatter(y=X0_me_st['y'], x=X0_me_st['v'], marker='*', alpha=0.5)
#plt.scatter(y=X5_me_st['y'], x=X5_me_st['v'], marker='x', alpha=0.5)
plt.scatter(y=X8_me_st['y'], x=X8_me_st['v'], marker='+', alpha=0.5)
plt.xlabel('Variance'); plt.ylabel('Mean TEST Time')


# X6 explains equally good the variation 
# ----
# see green * on graph versus 200car groups

# In[ ]:


divisor=201
#create groups of divisor length
train['groep']=train.index/divisor-.499
train['groep']=train['groep'].round(0)
train['teller']=1.0

#groep mean - standarddeviation
me_st=train[['y','groep']].groupby(by='groep').mean()
me_st['v']=train[['y','groep']].groupby(by='groep').var()
#print(me_st)

#X0 mean - standarddeviation
Xvar='X6'
X0_me_st=train[['y',Xvar]].groupby(by=Xvar).mean()
X0_me_st['v']=train[['y',Xvar]].groupby(by=Xvar).var()

Xvar='X8'
X8_me_st=train[['y',Xvar]].groupby(by=Xvar).mean()
X8_me_st['v']=train[['y',Xvar]].groupby(by=Xvar).var()
#print(X0_me_st)

import matplotlib.pyplot as plt
import seaborn as sns


plt.title('TESTTime Variability Group o X6 * X8 + ')
#plt.scatter(y=portRet, x=portSTD, marker='.', alpha=0.5)
plt.scatter(y=me_st['y'], x=me_st['v'], marker='o', alpha=0.5)
plt.scatter(y=X0_me_st['y'], x=X0_me_st['v'], marker='*', alpha=0.5)
#plt.scatter(y=X5_me_st['y'], x=X5_me_st['v'], marker='x', alpha=0.5)
plt.scatter(y=X8_me_st['y'], x=X8_me_st['v'], marker='+', alpha=0.5)
plt.xlabel('Variance'); plt.ylabel('Mean TEST Time')


# X0 efficient frontier near '0' variation
# ----

# In[ ]:


# groep  VCM with pivot and fillNA
X0_time = pd.pivot_table(train, values='y', index=['groep'],columns=['X0'], aggfunc=np.mean).fillna( method='ffill', axis=0).fillna( method='bfill', axis=0)

Am1=vcminv(X0_time)
b_=[0 for ci in X0_time.columns]+[0,0]
b_[-1]=1
X0m_=X0_time.mean(axis=0) 
X0m2_=X0m_.append(pd.DataFrame([1,1],index=['y_','lambda']))
#print(X0m_)
portRet=[]
portSTD=[]
vcm_=vcm(X0_time)
for ef in range(70,140,1):
    b_[-2]=ef
    portf=Am1.dot(b_)
    #print(portf)
    #porty=( portf.T*(X0m2_.T) ).sum(axis=1)  #exact target return
    portRet=portRet+[ef]
    
    var=(portf[:-2].T.dot(vcm_)).dot(portf[:-2]) #
    portSTD=portSTD+ [var ]

X0_ym=train[['y','X0']].groupby(by='X0').mean()
X0_yv=train[['y','X0']].groupby(by='X0').var()
#print(portSTD)
plt.title('Efficient frontier X3 versus group  ')
plt.scatter(y=portRet, x=portSTD, marker='.', alpha=0.5)
plt.scatter(y=me_st['y'], x=me_st['v'], marker='o', alpha=0.5)
plt.scatter(y=X0_ym,x=X0_yv,marker='*', alpha=0.5)
#plt.scatter(y=X5_me_st['y'], x=X5_me_st['v'], marker='x', alpha=0.5)
#plt.scatter(y=X8_me_st['y'], x=X8_me_st['v'], marker='+', alpha=0.5)
plt.xlabel('Variance'); plt.ylabel('Mean TEST Time')


# Efficient frontier X3
# ---
# nears 0, but of course its an impossible point to achieve.  Would mean you are outsourcing the cars

# In[ ]:


# groep  VCM with pivot and fillNA
X0_time = pd.pivot_table(train, values='y', index=['groep'],columns=['X3'], aggfunc=np.mean).fillna( method='ffill', axis=0).fillna( method='bfill', axis=0)

Am1=vcminv(X0_time)
b_=[0 for ci in X0_time.columns]+[0,0]
b_[-1]=1
X0m_=X0_time.mean(axis=0) 
X0m2_=X0m_.append(pd.DataFrame([1,1],index=['y_','lambda']))
#print(X0m_)
portRet=[]
portSTD=[]
vcm_=X0_time.cov()
for ef in range(70,140,1):
    b_[-2]=ef
    portf=Am1.dot(b_)
    #print(portf)
    #porty=( portf.T*(X0m2_.T) ).sum(axis=1)  #exact target return
    portRet=portRet+[ef]
    var=(portf[:-2].T.dot(vcm_)).dot(portf[:-2]) #
    portSTD=portSTD+ [var ]


X3_ym=train[['y','X3']].groupby(by='X3').mean()
X3_yv=train[['y','X3']].groupby(by='X3').var()
#print(portSTD)
plt.title('Efficient frontier X3 versus group  ')
plt.scatter(y=portRet, x=portSTD, marker='.', alpha=0.5)
plt.scatter(y=me_st['y'], x=me_st['v'], marker='o', alpha=0.5)
plt.scatter(y=X3_ym,x=X3_yv,marker='*', alpha=0.5)
#plt.scatter(y=X5_me_st['y'], x=X5_me_st['v'], marker='x', alpha=0.5)
#plt.scatter(y=X8_me_st['y'], x=X8_me_st['v'], marker='+', alpha=0.5)
plt.xlabel('Variance'); plt.ylabel('Mean TEST Time')


# X6 efficient frontier
# ---
# its also possible to reduce the variability of the production with an optimisation of X6

# In[ ]:


# groep  VCM with pivot and fillNA
X0_time = pd.pivot_table(train, values='y', index=['groep'],columns=['X6'], aggfunc=np.mean).fillna( method='ffill', axis=0).fillna( method='bfill', axis=0)

Am1=vcminv(X0_time)
b_=[0 for ci in X0_time.columns]+[0,0]
b_[-1]=1
X0m_=X0_time.mean(axis=0) 
X0m2_=X0m_.append(pd.DataFrame([1,1],index=['y_','lambda']))
#print(X0m_)
portRet=[]
portSTD=[]
vcm_=X0_time.cov()
for ef in range(70,140,1):
    b_[-2]=ef
    portf=Am1.dot(b_)
    #print(portf)
    #porty=( portf.T*(X0m2_.T) ).sum(axis=1)  #exact target return
    portRet=portRet+[ef]
    var=(portf[:-2].T.dot(vcm_)).dot(portf[:-2]) #
    portSTD=portSTD+ [var ]

X6_ym=train[['y','X6']].groupby(by='X6').mean()
X6_yv=train[['y','X6']].groupby(by='X6').var()
#print(portSTD)
plt.title('Efficient frontier X6 versus group  ')
plt.scatter(y=portRet, x=portSTD, marker='.', alpha=0.5)
plt.scatter(y=me_st['y'], x=me_st['v'], marker='o', alpha=0.5)
plt.scatter(y=X6_ym,x=X6_yv,marker='*', alpha=0.5)
plt.xlabel('Variance'); plt.ylabel('Mean TEST Time')


# Example of efficient frontier X6 optimised
# ---
# to optimize this VCM you have to constraint the fractions >0
# since this solution has no constraint it proposes an optimisation with negative numbers

# In[ ]:


b_[-2]=100
print('Percent')
print( Am1.dot(b_).round(2)*100 )
b_[-2]=103
print('Percent')
print( Am1.dot(b_).round(2)*100 )


# If you want to diminish with X3 optimisation the time to 100 seconds ?
# ---
# you have to change the  'ratio' between the X3 parameter fe
# in this simplified model...
# of course, there are still other variables like X6, so the model should add those variables too

# In[ ]:


def vcminv2(X_):
    kolom=X_.columns
    ym_=X_.mean(axis=0)
    X_vcm=pd.DataFrame( X_.cov()*2,index=kolom,columns=kolom) 
    X_vcm['y_']=ym_
    X_vcm['lamb']=1.0    
    Xm_vcm=pd.DataFrame( -X_.cov()*2,index=kolom,columns=kolom) 
    Xm_vcm['y_']=0
    Xm_vcm['lamb']=-1.0    
    Xt_vcm=X_vcm.append(Xm_vcm)
    #print(Xt_vcm)
    
    Xt_vcmT=Xt_vcm.T
    Xt_vcmT['y_']=ym_.append(pd.DataFrame([0,0],index=['lamb','y_']))
    Xt_vcmT['lamb']=1.0    
    Xt_vcmT['lamb']['lamb']=0.0
    Xt_vcmT['y_']['lamb']=0.0
    Xt_vcmT['lamb']['y_']=0.0    
    Xt_vcmT.dropna(axis=(0,1), how='any')
    X2_vcm=Xt_vcmT.T
    show_dataframe(X2_vcm)
    koloml=X2_vcm.columns
    rowi=X2_vcm.index
    return pd.DataFrame( np.linalg.pinv( X2_vcm) ,columns=rowi,index=koloml )


# In[ ]:


# groep  VCM with pivot and fillNA
X0_time = pd.pivot_table(train, values='y', index=['groep'],columns=['X3'], aggfunc=np.mean).fillna( method='ffill', axis=0).fillna( method='bfill', axis=0)

Am1=vcminv2(X0_time)
#print(Am1)
b_=[0 for ci in X0_time.columns]+[0,0]
b_[-1]=1
b_[-2]=100
optimum=pd.DataFrame( Am1.T.dot(b_) ,index=Am1.columns)

print('100 seconds optimal distribution',Am1.T.dot(b_) )
#print(optimum)


# Comparing with the optimum
# ----
# the problem is something probably you can't change, but the productmix has 21 %too much X3
#  'c' , 3x too much 'f' and not enouph the others;..
# 

# In[ ]:


# groep  VCM with pivot and fillNA
X0_time = pd.pivot_table(train, values='teller', index=['groep'],columns=['X3'], aggfunc=np.sum).fillna( method='ffill', axis=0).fillna( method='bfill', axis=0)
vergelijk = X0_time/2
optimum.columns=['optimum_100']
optimum=optimum[:7]*100
vergelijk = vergelijk.append(optimum.T)
print(vergelijk)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(vergelijk, cmap=cmap)


# In[ ]:


Same analysis X6
---
to many option 'g' and 'j'... in the mix


# In[ ]:


# groep  VCM with pivot and fillNA
X0_time = pd.pivot_table(train, values='y', index=['groep'],columns=['X6'], aggfunc=np.mean).fillna( method='ffill', axis=0).fillna( method='bfill', axis=0)

Am1=vcminv2(X0_time)
#print(Am1)
b_=[0 for ci in X0_time.columns]+[0,0]
b_[-1]=1
b_[-2]=100
optimum=pd.DataFrame( Am1.T.dot(b_) ,index=Am1.columns)

print('100 seconds optimal distribution',Am1.T.dot(b_) )
#print(optimum)


# In[ ]:


# groep  VCM with pivot and fillNA
X0_time = pd.pivot_table(train, values='teller', index=['groep'],columns=['X6'], aggfunc=np.sum).fillna( method='ffill', axis=0).fillna( method='bfill', axis=0)
vergelijk = X0_time/2
optimum.columns=['optimum_100']
optimum=optimum[:12]*100
vergelijk = vergelijk.append(optimum.T)
print(vergelijk)
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(vergelijk, cmap=cmap)

