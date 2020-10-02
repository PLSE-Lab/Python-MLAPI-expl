#!/usr/bin/env python
# coding: utf-8

# Here, I'm testing the hypothesys of whether the evolution of COVID19 could be predicted by the temperature. I'm using data from US average temperatures for every season in each State and fitting them with a SVM model against the COVID19 cases in the same States, as of March 21, 2020. Then, using a mock dataset of hypothetical increasing temperatures, I try to predict the evolution of the cases based on changing temperatures.
# 
# The only case predicting decrease in cases is the Winter one, which is actually the most similar to when COVID19 did spread.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/temperature-correlations'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Import Pandas and Numpy
import pandas as pd
import numpy as np


# Sets up the graphing configuration
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as graph
get_ipython().run_line_magic('matplotlib', 'inline')
graph.rcParams['figure.figsize'] = (15,5)
graph.rcParams["font.family"] = 'DejaVu Sans'
graph.rcParams["font.size"] = '12'
graph.rcParams['image.cmap'] = 'rainbow'

# Load dataset
dataset = pd.read_csv("/kaggle/input/temperature-correlations/US_cases.csv")
print(dataset.head())

# Assign variables
y = dataset["Cases"]

features = ["Winter", "Spring", "Summer", "Fall"]
X = dataset[features]
x1 = X["Winter"]
x2 = X["Spring"]
x3 = X["Summer"]
x4 = X["Fall"]

# Reshape arrays
x1R=x1.values.reshape(-1,1)
x2R=x2.values.reshape(-1,1)
x3R=x3.values.reshape(-1,1)
x4R=x4.values.reshape(-1,1)
yR=y.values.reshape(-1,1)

# Call and fit SVM model with Polynomial Function
from sklearn import svm
SVM_Model_x1 = svm.SVC(kernel='poly',degree=3, coef0=0.2, gamma = 0.2).fit(x1R,yR)
SVM_Model_x2 = svm.SVC(kernel='poly',degree=3, coef0=0.2, gamma = 0.2).fit(x2R,yR)
SVM_Model_x3 = svm.SVC(kernel='poly',degree=3, coef0=0.2, gamma = 0.2).fit(x3R,yR)
SVM_Model_x4 = svm.SVC(kernel='poly',degree=3, coef0=0.2, gamma = 0.2).fit(x4R,yR)

# Call the test data
X_test=pd.read_csv("/kaggle/input/temperature-correlations/test.csv")

# Plot

fig = graph.figure()

graph.subplot(2, 2, 1)
graph.title("COVID19 cases US vs. Temperature Winter")
graph.plot(x1,y,'ob',label='observed')
x1_predictions=SVM_Model_x1.predict(X_test)
graph.plot(X_test,x1_predictions,'Xr-',label='predicted')
graph.ylabel("Nr. of Cases")
graph.xlabel("Temperature(C)")
graph.yscale("log")

graph.subplot(2, 2, 2)
graph.title("COVID19 cases US vs. Temperature Spring")
graph.plot(x2,y,'ob',label='observed')
x2_predictions=SVM_Model_x2.predict(X_test)
graph.plot(X_test,x2_predictions,'Xr-',label='predicted')
graph.ylabel("Nr. of Cases")
graph.xlabel("Temperature(C)")
graph.yscale("log")

graph.subplot(2, 2, 3)
graph.title("COVID19 cases US vs. Temperature Summer")
graph.plot(x3,y,'ob',label='observed')
x3_predictions=SVM_Model_x3.predict(X_test)
graph.plot(X_test,x3_predictions,'Xr-',label='predicted')
graph.ylabel("Nr. of Cases")
graph.xlabel("Temperature(C)")
graph.yscale("log")

graph.subplot(2, 2, 4)
graph.title("COVID19 cases US vs. Temperature Fall")
graph.plot(x4,y,'ob',label='observed')
x4_predictions=SVM_Model_x4.predict(X_test)
graph.plot(X_test,x4_predictions,'Xr-',label='predicted')
graph.ylabel("Nr. of Cases")
graph.xlabel("Temperature(C)")
graph.yscale("log")

fig.tight_layout()
graph.legend()
graph.show()


# In[ ]:




