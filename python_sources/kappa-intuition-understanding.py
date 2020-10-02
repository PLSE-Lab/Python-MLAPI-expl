#!/usr/bin/env python
# coding: utf-8

# ## The purpose of this kernel is to study the Kappa metrics

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# plots
import matplotlib.pyplot as plt
import seaborn as sns 
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
from numba import jit 

@jit
def qwk3(a1, a2, max_rat=3): # qwk3 CPMP


    hist1 = np.zeros((max_rat + 1, ))
    hist2 = np.zeros((max_rat + 1, ))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o +=  (i - j) * (i - j)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += hist1[i] * hist2[j] * (i - j) * (i - j)

    e = e / a1.shape[0]

    return 1 - o / e


# In[ ]:


y_true = np.array([0,1,2,3])
y_true1 = np.array([0,2,2,3])


# In[ ]:


error = y_true.copy()
error[error == 0] = 3
error1 = y_true.copy()
error1[error1 == 0] = 2
error2 = y_true.copy()
error2[error2 == 0] = 1
error3 = y_true1.copy()
error3[error3 == 0] = 1
error4 = y_true.copy()
error4[error4 == 3] = 0


# In[ ]:


a0 = qwk3(y_true, error)
b0 = mean_squared_error(y_true, error)
c0 = mean_absolute_error(y_true, error)
a1 = qwk3(y_true, error1)
b1 = mean_squared_error(y_true, error1)
c1 = mean_absolute_error(y_true, error1)
a2 = qwk3(y_true, error2)
b2 = mean_squared_error(y_true, error2)
c2 = mean_absolute_error(y_true, error2)
a3 = qwk3(y_true1, error3)
b3 = mean_squared_error(y_true1, error3)
c3 = mean_absolute_error(y_true1, error3)


# In[ ]:


print("            |  FATAL ERROR |","   Symmetric  |", "   ERROR      |", "LITTLE ERROR  |","   DISTRIBUTION                          ")
print("--------------------------------------------------------------------------------------------")
print("Truth       |\t%s"%y_true,' |', "\t%s"%y_true,' |', "\t%s"%y_true,' |', "\t%s"%y_true,' |', "\t%s"%y_true1,'          |')
print("Predicted   |\t%s"%error4,' |', "\t%s"%error,' |', "\t%s"%error1,' |'"\t%s"%error2,' |', "\t%s"%error3,'          |')
print("MAE         |\t%s"%c0,'      |', "\t%s"%mean_absolute_error(y_true, error4),'      |',"\t%s"%c1,'       |'"\t%s"%c2,'      |',"\t%s"%c3,'               |')
print("Kappa       |\t%s"%round(a0,2),'       |', "\t%s"%round(qwk3(y_true, error4),2),'       |'"\t%s"%a1,'       |'"\t%s"%round(a2,2),'      |',"\t%s"%round(a3,2),'               |')
print("RMSE        |\t%s"%b0,'      |', "\t%s"%mean_squared_error(y_true, error4),'      |',"\t%s"%b1,'       |',"\t%s"%b2,'      |',"\t%s"%b3,'               |')


# In[ ]:


res = pd.DataFrame(np.array([[0,0,0,0], [3,2,1,0], [a0,a1,a2,1],[b0,b1,b2,0],[c0,c1,c2,0]]).transpose(),
            columns=['y_tr', 'preds', 'kappa','rmse','mae'])


# In[ ]:


figure = plt.figure()
axes = figure.add_subplot(111)
axes = plt.gca()
axes.xaxis.set_ticklabels(['0', '(0,3)', '(0,2)', '(0,1)', '(0,0)','(3,2)','(3,1)','(3,0)'])
plt.plot(res['kappa'],"-*r")
plt.plot(res['rmse'],"-*b")


# In[ ]:


res[['kappa','rmse','mae']].corr()

