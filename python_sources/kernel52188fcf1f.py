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
print(filename)
# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
oldf = pd.read_csv('/kaggle/input/old-faithful/faithful.csv')
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = oldf['waiting'].values.reshape(-1, 1)
y = oldf['eruptions'].values.reshape(-1, 1)
oldf.plot(x = 'waiting',y ='eruptions', style='o')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
                                     
model = LinearRegression()
model.fit(X_train, y_train) #training the algorithm
                                     
r_sq = model.score(X,y)   
print('slope:', model.coef_)
print('intercept:', model.intercept_)
print('coefficient of determination:', r_sq)
                                            
                                            