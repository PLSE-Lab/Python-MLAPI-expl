#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
syd=pd.read_csv('../input/sym_dis_matrix.csv')
dia=pd.read_csv('../input/diagn_title.csv')
sym=pd.read_csv('../input/symptoms2.csv')
#
A=syd
B=A.drop(['eye'],axis=1)

#You can merge the symptoms and the diagnosis
A=pd.merge(sym,syd,how='inner',right_on='eye',left_on='_id')
B=pd.merge(A,dia,how='inner',left_on='_id',right_on='id')

print(A.head())
print(B.head())


C=B.T
C.index.name='eye'
#D=C.drop(['eye'],axis=1)
print(C.head())
#A.index.name='id'
#print(dia)
#print(syd.head())

#print(sym.head())
#print(dia.as_matrix())
#print(syd.rename_axis(index=str,axis=dia.as_matrix().transpose()))


# In[ ]:


#this is the core function
from scipy.cluster.vq import whiten
C=whiten(C)
#the eye column moet gedropt worden
#print(A.head())
from numpy.linalg import inv
U,s,V=np.linalg.svd(C,full_matrices=False)
# reconstruct
S=np.diag(s)
S=S[0:20,0:20]
iS=inv(S)
US=np.dot(U[:,0:20],iS)
US
# A fill up with US matrix
US_df=pd.DataFrame(data=US)
print(US_df.head())
# with this simple math i know all the relations between all the symptoms and diseases
#


# In[ ]:


#visualizing the data matrixes

import matplotlib.pylab as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(U, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(S, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()


import matplotlib.pylab as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(V, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(US, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()


import matplotlib.pylab as plt
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(iS, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(US_df, interpolation='nearest', cmap=plt.cm.ocean)
plt.colorbar()
plt.show()

