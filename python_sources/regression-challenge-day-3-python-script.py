# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
cameras = pd.read_csv("../input/camera_dataset.csv")
print(cameras.head())
print(cameras.describe())
cameras = cameras.dropna()
from sklearn import linear_model

import matplotlib.pyplot as plt
logreg = linear_model.LogisticRegression()
x = cameras['Max resolution'][:, np.newaxis]
y = cameras['Storage included']
logreg.fit(x, y)

# and plot the result
plt.scatter(x.ravel(), y, color='black', zorder=20)
plt.plot(x, logreg.predict_proba(x)[:,1], color='blue', linewidth=3)
plt.xlabel('Max resolution')
plt.ylabel('Price')
print('coefficient = ' + str(logreg.coef_))
print('intercept = ' + str(logreg.intercept_))