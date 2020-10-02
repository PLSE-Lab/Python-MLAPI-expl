# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


#import basic libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#load dataset
dataset = pd.read_csv('../input/HousingData.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,13].values


#handling missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer = imputer.fit(X)
X = imputer.transform(X)
print(X)



#Using backward elimination, find parameters which actually have an impact
import statsmodels.formula.api as sm



def backwardElimination(x,sl):
    numVars=len(x[0])
    for i in range(0,numVars):
        regressor_OLS=sm.OLS(y,x,sl).fit()
        maxVar=max(regressor_OLS.pvalues).astype(float)
        if(maxVar>sl):
            for j in range(0,numVars-i):
                if(regressor_OLS.pvalues[j].astype(float)==maxVar):
                    x=np.delete(x,j,axis=1)
        #print(regressor_OLS.summary())
    print(regressor_OLS.summary())
    return x


SL=0.05
X=np.append(arr=np.ones((506,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13]]
X_opt=X_opt.astype(float)
X_modelled=backwardElimination(X_opt,SL)

#splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X_modelled,y,test_size=1/3,random_state=0)


#train the model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predict output of test set
y_pred=regressor.predict(X_test)


print(y_pred)

print(y_test)

#plot graph to check difference between predicted and normal outcome

fig,ax = plt.subplots()
ax.scatter(y_test,y_pred)
#ax.set_x_label('Measured')
#ax.set_y_label('Predicted')
ax.plot([y_test.min(),y_test.max()],[y_pred.min(),y_pred.max()],'k--',lw=4)
ax.set_title('Actual(X) vs Predicted(Y)')
fig.show()
plt.savefig('Actual vs Predicted.jpg')
