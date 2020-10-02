#!/usr/bin/env python
# coding: utf-8

# This study simply visualizes the differences between a linear and a nonlinear classifier in the case of 2-dimensional features. For this purpose, we use well-known multi-class iris data.

# In[ ]:


import numpy as num
from sklearn.datasets import load_iris
dir(load_iris())
Y = load_iris().target
Xs = load_iris().data[:,(2,3)]


# In[ ]:


x1 = num.arange( min(Xs[:,0]) - 1 , max(Xs[:,0]) + 1 , .01 )
x2 = num.arange( min(Xs[:,1]) - 1 , max(Xs[:,1]) + 1 , .01 )
X1 , X2 = num.meshgrid( x1 , x2  )
meshed_Xs = num.c_[ X1.ravel() , X2.ravel() ]


# **1. Nonlinear logistic regression**

# In[ ]:


from sklearn.preprocessing import PolynomialFeatures
PF_obj = PolynomialFeatures( degree=6, include_bias=False )
Xspoly = PF_obj . fit_transform( Xs )
meshed_Xspoly = PF_obj . fit_transform( meshed_Xs )


# In[ ]:


from sklearn.linear_model import LogisticRegression
LRre_obj = LogisticRegression( multi_class='multinomial' , solver='newton-cg' )
LRre_obj . fit( Xs , Y )
ymeshlR = LRre_obj.predict( meshed_Xs )
ymeshlR = ymeshlR . reshape ( X1.shape  )


# In[ ]:


import matplotlib.pyplot as pyp
pyp.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
pyp.contourf( X1 , X2 , ymeshlR , alpha=0.3  )
pyp.scatter ( Xs[:,0] , Xs[:,1] , c=Y , s=30 , edgecolor='k' )
pyp.xlabel ( 'Patal length [cm]' )
pyp.ylabel ( 'Patal Width [cm]' )
pyp.xlim( xmin=0 )
pyp.ylim( ymin=0 )
pyp.title( 'Decision Boundary for linear logistic regression' )
pyp.show()


# In[ ]:


LRre_obj . fit ( Xspoly , Y )
ymeshNR = LRre_obj . predict ( meshed_Xspoly  )
ymeshNR = ymeshNR . reshape ( X1 . shape)


# In[ ]:


pyp.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
pyp.contourf( X1 , X2 , ymeshNR , alpha=0.3  )
pyp.scatter ( Xs[:,0] , Xs[:,1] , c=Y , s=30 , edgecolor='k' )
pyp.xlabel ( 'Patal length [cm]' )
pyp.ylabel ( 'Patal Width [cm]' )
pyp.xlim( xmin=0 )
pyp.ylim( ymin=0 )
pyp.title( 'Decision Boundary for nonlinear logistic regression' )
pyp.show()


# **2. SVM classifier**

# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
SVML_obj = Pipeline(  [ ( 'scaler' , StandardScaler() ) , ( 'linear_svc' , LinearSVC(  C = 1 , loss = 'hinge' ,multi_class = 'ovr' ) ) ]  )
SVML_obj . fit ( Xs ,  Y)
ymeshSVM = SVML_obj.predict ( meshed_Xs  )
ymeshSVM = ymeshSVM . reshape ( X1.shape  )


# In[ ]:


pyp.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
pyp.contourf( X1 , X2 , ymeshSVM , alpha=0.3  )
pyp.scatter ( Xs[:,0] , Xs[:,1] , c=Y , s=30 , edgecolor='k' )
pyp.xlabel ( 'Patal length [cm]' )
pyp.ylabel ( 'Patal Width [cm]' )
pyp.xlim( xmin=0 )
pyp.ylim( ymin=0 )
pyp.title( 'Decision Boundary for linear SVM' )
pyp.show()


# In[ ]:


SVML_obj . fit ( Xspoly ,  Y)
ymeshNSVM = SVML_obj . predict ( meshed_Xspoly )
ymeshNSVM= ymeshNSVM.reshape ( X1.shape  )


# In[ ]:


pyp.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
pyp.contourf( X1 , X2 , ymeshNSVM , alpha=0.3  )
pyp.scatter ( Xs[:,0] , Xs[:,1] , c=Y , s=30 , edgecolor='k' )
pyp.xlabel ( 'Patal length [cm]' )
pyp.ylabel ( 'Patal Width [cm]' )
pyp.xlim( xmin=0 )
pyp.ylim( ymin=0 )
pyp.title( 'Decision Boundary for nonlinear SVM' )
pyp.show()

