# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# load bosten dataset from sklearn
from sklearn.datasets import load_boston
boston = load_boston()
boston

# Lets load everything into one dataframe
df_x = pd.DataFrame(boston.data,columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)

df_x.describe()

reg = LinearRegression()

# Split data in traning and testing
x_train,x_test,y_train,y_test = train_test_split(df_x,df_y,test_size=0.2,random_state = 4)
reg.fit(x_train,y_train)

# To find the cofficient of variance.
reg.coef_

# Predict
pred = reg.predict(x_test)

# Mean square error
np.mean((pred-y_test)**2)