# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model  #machine learning
import matplotlib as mpl          #config file render
mpl.use('Agg')
import matplotlib.pyplot as plt   #plot util
import unicodedata as ucd #for unicode normalization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

dataframe = pd.read_csv('../input/train.csv')
dataframe.dropna(subset=['Survived'], inplace=True)
dataframe.dropna(subset=['Age'], inplace=True)
sex = dataframe['Sex'].map(lambda x: ucd.normalize('NFKD', x))

print (sex)

pId = dataframe[['PassengerId']]
survived = dataframe[['Survived']]
age = dataframe[['Age']]

#learn
datareg = linear_model.LinearRegression()
datareg.fit(age, survived)


#visualize results
fig = plt.figure()
figplot = fig.add_subplot(111)
box = dict(facecolor='yellow', pad=5, alpha=0.2)
figplot.set_ylabel('Survived', bbox=box)
figplot.set_xlabel('Age', bbox=box)
figplot.scatter(age, survived)
figplot.plot(age, datareg.predict(age))
#error = ys - datareg.predict(xs)
#figplot.scatter(xs, error)
#print (error * error).sum()
fig.savefig('graph.png')



#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.