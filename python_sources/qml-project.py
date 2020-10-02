#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install wildqat   ')
get_ipython().system('pip install pyitlib')


# These are the necessarry imports for the kernel. pyitlib is an MIT-licensed library of information-theoretic methods for data analysis and machine learning, implemented in Python and NumPy. Wildqat is an open source Python framework for QUBO. 

# In[ ]:


import numpy as np
import pandas as pd
from pyitlib import discrete_random_variable as drv
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
import wildqat as wq


# #### Here we have developed an objective function based on mRMR(minimum Redundancy Maximum Relevance) and Mutual Information is taken as the type of correlation.
# 

# In[ ]:


def Mutual_Information(X):
    return drv.information_mutual(X.T)
    
def Identity_Matrix(NOF):
    return np.identity(NOF)
    
def Diagonal_Matrix(data,NOF):
    D = np.zeros([NOF,NOF])
    for i in range(NOF):
        corr = np.corrcoef(data.iloc[:,i], data.iloc[:,NOF])
        D[i,i]=corr[0,1]
    return D
    
def mRMR(D,M,NOF):       #-------- minimun Redundancy Maximum Relevance
    return ((M/NOF)-D)


# ## Here the operation is done with CMC data set from UCL Machine Learning repository.
# 10 datasets are added. You can try with the other datasets also but a minimal modification will be required.
# 
# 

# In[ ]:


file_path = '../input/Vehicle.csv'


# In[ ]:


data = pd.read_csv(file_path)
col_no = len(data.columns)
NOF = col_no - 1
M = data.iloc[:,0:NOF]    ##----------- Data Matrix containing feature data
M.values
#print(M)    


# In[ ]:


##########################################  Run this cell if feature scaling is required
scaler = preprocessing.MinMaxScaler()
M = scaler.fit_transform(M)
print(M)


# If you target variable then run the cell below. This assigns numrical values to catagorical data, which makes calculation possible. Otherwise we cannot get feature-class(target) **correlation values(D matrix)**.

# In[ ]:


##########################################  Run this if the Class variable is categorical
class_label_encoder = LabelEncoder()
data.iloc[:,NOF]=class_label_encoder.fit_transform(data.iloc[:,NOF])


# Mutual information will be calculated with integer values. So binning is required. Here we have hard coded the number of bins to 5. You may require to change the number for different dataset.

# In[ ]:


enc = KBinsDiscretizer(n_bins=5,encode='ordinal')   
M_binned = enc.fit_transform(M)
#print(M_binned)
    
    
D = Diagonal_Matrix(data,NOF)
#print(type(D))

MI = Mutual_Information(M_binned.astype(int))
obj = mRMR(D,MI,NOF)
#mRMR_qubo = pd.DataFrame(mRMR)
#mRMR_qubo.to_csv("CMC_mi_mRMR_qubo.csv")


# 100 iterations are performed for statistical weight measure for the sake of accuracy checking.

# In[ ]:


Result = []
for j in range(10):
    a = wq.opt()
    a.qubo = obj
    answer = a.sa()
    print('Solution of {}th iteration: '.format(j+1),answer)
    Result.append(answer)


# In[ ]:


cols = list(data.columns)
cols.pop()
features = cols



df = pd.DataFrame(Result, columns = features)
#df.to_csv('vehicle_mi_mRMR_sa_result.csv')
df


# In[ ]:




