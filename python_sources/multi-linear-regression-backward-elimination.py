# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 10:57:16 2018

@author: kartik.sharma10
"""

import pandas as pd
import seaborn as sea
import matplotlib.pyplot as plt
import numpy as np

dataset =pd.read_csv('../input/50_Startups.csv')

#z=dataset.describe(include='all')

dataset['State'].value_counts()

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

'''
dataset.plot.scatter()

sea.barplot(x='R&D Spend',y='Profit',data=dataset,hue='State')

sea.lmplot(x='R&D Spend',y='Profit',data=dataset,hue='State')
# more r&d more profit
    
sea.barplot(x='Administration',y='Profit',data=dataset,hue='State')
# can not conclude

sea.lmplot(x='Administration',y='Profit',data=dataset,hue='State')
# more adm more profit (not for florida)

sea.barplot(x='Marketing Spend',y='Profit',data=dataset)

sea.lmplot(x='Marketing Spend',y='Profit',data=dataset,hue='State')

# more adm more profit 

sea.barplot(x='State',y='Profit',data=dataset)

sea.lmplot(x='State',y='Profit',data=dataset)
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, -1] = labelencoder_X.fit_transform(x[:, -1])
onehotencoder = OneHotEncoder(categorical_features = [-1])
x = onehotencoder.fit_transform(x).toarray()

x=x[:,1:]

import statsmodels.formula.api as sm
x = np.append(arr = np.ones((x.shape[0], 1)).astype(int), values = x, axis = 1)
'''
x_opt = x[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()  # 0.951, 0.945

x_opt = x[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()  # 0.951, 0.946

x_opt = x[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()  # 0.951, 0.948

x_opt = x[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()  # 0.950, 0.948

x_opt = x[:, [0,3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()  # 0.947, 0.945
'''
q=[a for a in range(x.shape[1])]
x_auto=x[:,q]

regressor_OLS = sm.OLS(endog = y, exog = x_auto).fit()
pv=regressor_OLS.pvalues
r2=regressor_OLS.rsquared
ar2=regressor_OLS.rsquared_adj

sl=0.05

while True:
    max_pval=max(pv)
    if max_pval>sl:
        index=int(np.where(pv == max_pval)[0])
        temp=q
        v=temp[index]
        temp.remove(temp[index])
        
        x_temp=x[:,temp]
        regressor_OLS = sm.OLS(endog = y, exog = x_temp).fit()
        pv=regressor_OLS.pvalues
        
        
        print(r2,ar2)
        
        
        if not(regressor_OLS.rsquared<=r2 and regressor_OLS.rsquared_adj>=ar2):
            temp.append(v)
            x_auto=x[:,temp]
            
            break
        else:
            print(temp)
            r2=regressor_OLS.rsquared
            ar2=regressor_OLS.rsquared_adj
            x_auto=x_temp
        
    else:
        break


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_auto,y,test_size = 0.25, random_state = 0)


from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(x_train, y_train)

y_pred=linear_regressor.predict(x_test)


