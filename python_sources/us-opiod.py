# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pandas as pd
%pylab inline

# Any results you write to the current directory are saved as output. lets see if this works
#here is where I will start the eda, head, shape, dtype, describe, break down a componant groupby
#It would be nice if I made a note of where I stole this from. https://www.kaggle.com/elsehow/predicting-opiate-prescribing-dentists

ods = pd.read_csv('../input/overdoses.csv')
cleanNum = lambda n: int(n.replace(',', ''))
ods['Deaths'] = ods['Deaths'].map(cleanNum)
ods['Population'] = ods['Population'].map(cleanNum)
ods.head()

ods.shape

ods.dtypes

ods.describe

#so this is all about dentists, I want to break down the data set more broadly
#Next I will break down the data set about prescribers?  Why are there 3 different csv files with this project?

%pylab inline
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import re

#Its probably uneccsary to import numpy and pandas again here.

prescribers = pd.read_csv('../input/prescriber-info.csv')
opioids = pd.read_csv('../input/opioids.csv') #I am getting an error here I don't understand.  I can delete pylad inline and the other stuff and see what happens.
ODs = pd.read_csv('../input/overdoses.csv')
import re
ops = list(re.sub(r'[-\s]','.',x) for x in opioids.values[:,0])
prescribed_ops = list(set(ops) & set(prescribers.columns))
prescribers['NumOpioids'] = prescribers.apply(lambda x: sum(x[prescribed_ops]),axis=1)
prescribers['NumPrescriptions'] = prescribers.apply(lambda x: sum(x.iloc[5:255]),axis=1)
prescribers['FracOp'] = prescribers.apply(lambda x: float(x['NumOpioids'])/x['NumPrescriptions'],axis=1)
prescribers.plot.scatter('NumOpioids','NumPrescriptions')


#I need to run this through 3-10 Algoriths Linear Regression, KNN, Logistic regression, probably not decistion tree or random forest.
#I need to access this with the laptop as well as the desktop, this is better than jupiter notebooks because it automatically synchs unlike jupilter.
#I will whiteboard the main concepts as well as write too many comments as well an not enough.
#Accuracy: 0.589000 of model Logistic Regression
#Accuracy: 0.613149 of model Naive Bayes
#Accuracy: 0.831650 of model Random Forest
#Accuracy: 0.823600 of model Gradient Boosting
#Accuracy: 0.759100 of model KNN
#Accuracy: 0.774100 of model Decision Tree
#Accuracy: 0.714900 of model LDA
#Accuracy: 0.825450 of model Ensemble
#I don't actually know what the componant parts of any of these are, but I can start to break them down and see what they mean or do.




