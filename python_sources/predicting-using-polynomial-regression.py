import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize

#importing dataset
data = pd.read_csv('../input/concrete-compressive-strength-data-set/compresive_strength_concrete.csv')
print(data.head())

#Checking if any null values
data.isnull().values.any()

#Which column as missing values
data.isnull().sum()

#Normalize data
ndata = normalize(data,axis = 1)
ndata = pd.DataFrame(ndata)
#Fill the columns 
ndata.columns = data.columns
print(ndata.head())

#EDA 
#Histogram analysis
for i in range(len(ndata.columns)):
    plt.hist(ndata.iloc[:,i])
    plt.xlabel(ndata.columns[i])
    plt.ylabel("range")
    plt.show()
#Boxplot analysis
for i in range(len(ndata.columns)):
    plt.boxplot(ndata.iloc[:,i])
    plt.xlabel(ndata.columns[i])
    plt.ylabel("range")
    plt.show()
#violinplot analysis
for i in range(len(ndata.columns)):
    plt.violinplot(ndata.iloc[:,i])
    plt.xlabel(ndata.columns[i])
    plt.ylabel("range")
    plt.show()

#scatterplot analysis
for i in range(len(ndata.columns)):
    plt.scatter(ndata.iloc[:,i],ndata.iloc[:,8])
    plt.xlabel(ndata.columns[i])
    plt.ylabel("Concrete compressive strength(MPa, megapascals) ")
    plt.show()


#Visualize pairplot analysis
sns.pairplot(ndata)

#Correlation Heatmap plot
corr = ndata.corr()
corr

corr.style.background_gradient(cmap = 'coolwarm',axis = 1)
sns.heatmap(corr)

#subsetting the data
x = ndata.iloc[:,0:8]
y = ndata.iloc[:,8]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.3,random_state = 3)
print(ytrain.shape)
print(ytest.shape)

lm = LinearRegression()
model = lm.fit(xtrain,ytrain)
pred = model.predict(xtrain)
accuracy = r2_score(pred,ytrain)
print("Accuracy of Multiple Regression on training set:",end='')
print(accuracy)

predt = model.predict(xtest)
taccuracy = r2_score(predt,ytest)
print("Accuracy of Multiple Regression on test set:",end='')
print(taccuracy)

#P-value
print(f_regression(x,y))

#Backward elimination due to multicollinarity
bdata = ndata
bdata.drop(['Fine Aggregate (component 7)(kg in a m^3 mixture)'],axis = 1,inplace = True)
x = bdata.iloc[:,0:7]
y = bdata.iloc[:,7]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.feature_selection import f_regression
xbtrain,xbtest,ybtrain,ybtest = train_test_split(x,y,test_size = 0.3,random_state = 3)
print(ybtrain.shape)
print(ybtest.shape)

lm = LinearRegression()
model = lm.fit(xbtrain,ybtrain)
pred = model.predict(xbtrain)
accuracy = r2_score(pred,ybtrain)
print("Accuracy after removing fine aggregate for training set:",end='')
print(accuracy)

predt = model.predict(xbtest)
taccuracy = r2_score(predt,ybtest)
print("Accuracy after removing fine aggregate for test set:",end='')
print(taccuracy)


#Complexity parameter 
#Polynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 3)
xpoly = poly.fit_transform(xtrain)
model_3 = lm.fit(xpoly,ytrain)
pred = lm.predict(xpoly)
taccuracy = r2_score(pred,ytrain)
print("Accuracy of training set of degree 3 polynomial:",end='')
print(taccuracy)

xpoly = poly.fit_transform(xtest)
pred = lm.predict(xpoly)
ttaccuracy = r2_score(pred,ytest)
print("Accuracy of testing set of degree 3 polynomial:",end='')
print(ttaccuracy)

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 4)
xpoly = poly.fit_transform(xtrain)
lm.fit(xpoly,ytrain)
pred = lm.predict(xpoly)
daccuracy = r2_score(pred,ytrain)
print("Accuracy of training set of degree 4 polynomial:",end='')
print(daccuracy)

xpoly = poly.fit_transform(xtest)
pred = lm.predict(xpoly)
daccuracy = r2_score(pred,ytest)
print("Accuracy of testing set of degree 4 polynomial:",end='')
print(daccuracy)

#We can see the accuracy decreases as we increase in degree for testing set 
#Best model is of degree = 3
print("Training accuracy:",end='')
print(taccuracy)
print("Testing accuracy:",end='')
print(ttaccuracy)