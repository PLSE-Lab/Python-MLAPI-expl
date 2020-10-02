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


# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 21:53:32 2018

@author: ramsey
"""

### Data preprocessing & Cleaning 
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

os.chdir(r'D:\DataMining\Linear Regression Project\set1\doe-fuel-economy-data\vehicles.csv') 

check=pd.read_csv("vehicles.csv", na_values='')

list(check.columns.values)

consumption = pd.read_csv("vehicles.csv", na_values='', usecols=[0, 22, 23, 28, 47, 58, 60, 63], 
                           header=0, names=['Consumption_Barrel', 'cylinders', 'Engine_Size', 'Fuel_Cost', 'Car_Model',
                                            'City_Consumption', 'HW_Consumption', 'Make_Year'])

list(consumption.columns.values)


consumption['Consumption_Barrel'].head()
consumption['Make_Year'].head()

consumption= consumption.fillna({'Consumption_Barrel': consumption.Consumption_Barrel.mean(),
                                 'cylinders': consumption.cylinders.mean(),
                                 'Engine_Size': consumption.Engine_Size.mean(),
                                 'Fuel_Cost': consumption.Fuel_Cost.mean(),
                                 'City_Consumption': consumption.City_Consumption.mean(),
                                 'HW_Consumption': consumption.HW_Consumption.mean(),
                                 'Make_Year': consumption.cylinders.median()})

from collections import Counter
data = Counter(consumption['Car_Model'])
data.most_common()   # Returns all unique items and their counts
data.most_common(1) 

consumption['Car_Model'] = consumption['Car_Model'].fillna('F150 Pickup 2WD') 

import statsmodels.formula.api as smf
Consumption_OLS = smf.ols('Consumption_Barrel ~ cylinders + Engine_Size+ Fuel_Cost+ City_Consumption+ HW_Consumption+ Make_Year', data=consumption).fit()
print(Consumption_OLS.summary())
plt.rc('figure', figsize=(12, 7))
#plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
plt.text(0.01, 0.05, str(Consumption_OLS.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.savefig('output.png')
plt.show()

consumption.corr()

### Checking descriptive statistics by visualizing distribution and other statistics 
from matplotlib import pyplot
pyplot.hist(consumption.Consumption_Barrel)
pyplot.ylabel('Frequency')
pyplot.xlabel('Consumption')
pyplot.title('Consumption of Fuel Barrels Per Year')
pyplot.show()

from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
qqplot(consumption.Consumption_Barrel, line='s')
pyplot.title('Consumption of Fuel Barrels Per Year')
pyplot.show()

from matplotlib import pyplot
pyplot.hist(consumption.cylinders)
pyplot.ylabel('Frequency')
pyplot.xlabel('vehicles cylinders')
pyplot.title('No. of cylinders')
pyplot.show()

from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
qqplot(consumption.cylinders, line='s')
pyplot.title('No. of cylinders')
pyplot.show()

from matplotlib import pyplot
pyplot.hist(consumption.Engine_Size)
pyplot.ylabel('Frequency')
pyplot.xlabel('vehicles Engine_Size')
pyplot.title('Engine_Size')
pyplot.show()

from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
qqplot(consumption.Engine_Size, line='s')
pyplot.title('Engine_Size')
pyplot.show()


from matplotlib import pyplot
pyplot.hist(consumption.Fuel_Cost)
pyplot.ylabel('Frequency')
pyplot.xlabel('vehicles Fuel_Cost')
pyplot.title('Fuel_Cost')
pyplot.show()

from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
qqplot(consumption.Fuel_Cost, line='s')
pyplot.title('Fuel_Cost')
pyplot.show()

from matplotlib import pyplot
pyplot.hist(consumption.City_Consumption)
pyplot.ylabel('Frequency')
pyplot.xlabel('vehicles City_Consumption')
pyplot.title('Fuel Consumption in City')
pyplot.show()

from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
qqplot(consumption.City_Consumption, line='s')
pyplot.title('Fuel Consumption in City')
pyplot.show()

from matplotlib import pyplot
pyplot.hist(consumption.HW_Consumption)
pyplot.ylabel('Frequency')
pyplot.xlabel('HW_Consumption')
pyplot.title('Consumption in Highways')
pyplot.show()

from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
qqplot(consumption.Fuel_Cost, line='s')
pyplot.title('Highway Consumption')
pyplot.show()

from matplotlib import pyplot
pyplot.hist(consumption.Make_Year)
pyplot.ylabel('Frequency')
pyplot.xlabel('Make_Year')
pyplot.title('vehicles Make_Year')
pyplot.show()

from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
qqplot(consumption.Make_Year, line='s')
pyplot.title('vehicles Make_Year')
pyplot.show()

import seaborn as sns
import matplotlib.pyplot as plt
sns.jointplot(x="Make_Year", y="Consumption_Barrel", data=consumption, kind="reg");
plt.show()

sns.jointplot(x="HW_Consumption", y="Consumption_Barrel", data=consumption, kind="reg");
plt.show()

sns.jointplot(x="cylinders", y="Consumption_Barrel", data=consumption, kind="reg");
plt.show()

### Cross-validation

X = consumption.iloc[:, [1,2,3,5,6,7]].values

y = consumption.iloc[:,[0]].values 

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler() 
scaler.fit(X[:, 0:5])
X[:, 0:5] = scaler.transform(X[:, 0:5])### this function will automatically scale the values of a variable into z-values of range (-3 : 3)

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, random_state = 0) 

from sklearn.linear_model import LinearRegression
reg = LinearRegression()

reg.fit(X,y)
reg.score(X,y)### R-squared
print(reg.coef_, reg.intercept_)


#### If passing floats to a classifier which expects categorical values as the target vector, you will get error. 
###If you convert it to int it will be accepted as input (although it will be questionable if that's the right way to do it).
#It would be better to convert your training scores by using scikit's labelEncoder function.
#The same is true for your DecisionTree and KNeighbors qualifier.

from sklearn.preprocessing import LabelEncoder

# Encode independent variables for training set
labelencoder_X = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = labelencoder_X.fit_transform(X[:, i])


# Encode dependent variable for training set
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Encode independent variables for test set
labelencoder_X_test = LabelEncoder()
for i in range(len(X_test[0])):
    X_test[:, i] = labelencoder_X_test.fit_transform(X_test[:, i])


# Encode dependent variable for test set
labelencoder_y_test = LabelEncoder()
y_test = labelencoder_y_test.fit_transform(y_test)


from sklearn.linear_model import LogisticRegression
classifier = LinearRegression()
classifier.fit(X,y)
y_pred = classifier.predict(X_test)
print("y_pred = classifier.predict(X_test): ", y_pred)

classifier.score(X_test, y_test)

### K Fold Cross Validation 

from sklearn.model_selection import KFold

kf = KFold(n_splits=2)
 kf.get_n_splits(X)
print(kf)  

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]



#stratified Kfold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)
print(skf)  
for train_index, test_index in skf.split(X, y):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

###applying K-fold 

from sklearn.model_selection import cross_val_score
accuricies = cross_val_score(estimator = classifier, X= X_train, y= y_train, cv=10)### accuricies of 10 test sets (K-fold validation)
accuricies_mean = accuricies.mean()







