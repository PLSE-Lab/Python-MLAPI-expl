#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('rm -rf 6dof_ctrl')
get_ipython().system('rm -rf bluerov_ffg')
get_ipython().system('rm -rf Sophus')

get_ipython().system('git clone https://github.com/slovak194/6dof_ctrl.git')
get_ipython().system('git clone https://github.com/slovak194/bluerov_ffg.git')
get_ipython().system('git clone https://github.com/slovak194/Sophus.git')


# In[ ]:


get_ipython().system('cp -R Sophus/py/sophus 6dof_ctrl/')


# In[ ]:


get_ipython().run_line_magic('cd', '6dof_ctrl')
get_ipython().run_line_magic('pwd', '')
get_ipython().system('git pull')


# In[ ]:


import sys

get_ipython().system('{sys.executable} -m pip install xmltodict')
get_ipython().system('{sys.executable} -m pip install numpy-quaternion')


# In[ ]:


get_ipython().run_line_magic('run', "-i 'motion_model.py'")


# In[ ]:


Ann = np.where(np.abs(An)<=1e-10, 0, An)
np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
U, S, Vh = np.linalg.svd(Ann)
print(U)
U @ np.diag(S)


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

mpl.rcParams['figure.dpi']= 100
plt.rcParams['figure.figsize'] = [12, 6]

plt.figure()

plt.subplot(1, 3, 1)
plt.imshow(abs(U))
plt.colorbar(cax=make_axes_locatable(plt.gca()).append_axes("right", size="5%", pad=0.05))

plt.subplot(1, 3, 2)
plt.imshow(abs(U @ np.diag(S)))
plt.colorbar(cax=make_axes_locatable(plt.gca()).append_axes("right", size="5%", pad=0.05))

plt.subplot(1, 3, 3)
plt.imshow(Ann.T @ Ann)
plt.colorbar(cax=make_axes_locatable(plt.gca()).append_axes("right", size="5%", pad=0.05))

plt.figure()

plt.rcParams['figure.figsize'] = [12, 6]

plt.subplot(1, 3, 1)
plt.imshow(abs(Ann))
plt.colorbar(cax=make_axes_locatable(plt.gca()).append_axes("right", size="5%", pad=0.05))

plt.subplot(1, 3, 2)
plt.imshow(np.log(abs(np.linalg.pinv(Ann))))
plt.colorbar(cax=make_axes_locatable(plt.gca()).append_axes("right", size="5%", pad=0.05))

plt.figure()
plt.plot(S, '*-')

