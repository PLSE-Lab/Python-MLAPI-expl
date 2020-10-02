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

data = pd.read_csv("../input/diamonds.csv");
print(data.head())

cutcodes = {"Ideal":5,"Premium":4,"Very Good":3,"Good":2,"Fair":1}
data["cut"].replace(cutcodes,inplace=True)
colorcodes = {"D":7,"E":6,"F":5,"G":4,"H":3,"I":2,"J":1}
data["color"].replace(colorcodes,inplace=True)
claritycodes={"IF":8,"VVS1":7,"VVS2":6,"VS1":5,"VS2":4,"SI1":3,"SI2":2,"I1":1}
data["clarity"].replace(claritycodes,inplace=True)
print(data.head())

import matplotlib.pyplot as plt
pd.tools.plotting.scatter_matrix(data,alpha=0.5,figsize=(10,10),diagonal="kde")
plt.show()
plt.savefig("scattermatrix.png")


from sklearn.linear_model import LinearRegression

