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

############################################
######     import python module  ##########
###########################################

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns



############################################
######   Read DataSet from csv file   ######
###########################################


df = pd.read_csv("../input/heart.csv")
df.head()

############################################
######   plot pairplot using seaborn   #####
############################################

sns.pairplot(df)

############################################
###### code for prediction and loc target and data   #####
############################################

data = df.iloc[:,0:-1]
target = df.iloc[:,13:14]
from sklearn.neighbors import KNeighborsClassifier
reg = KNeighborsClassifier()
reg.fit(data,target)

# dummy example for prediction
reg.predict([[56,1,1,120,236,0,1,178,0,0.8,2,0,2]])

reg.predict([[68,1,1,110,244,0,1,118,0,0.8,2,0,2]])
