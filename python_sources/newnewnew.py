import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#MODEL
#for machine learning
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
import os

#import the 'factor_return' sheet
factor_data=pd.read_csv("../input/Factor_Returns.csv")
##print(factor_data.head(10))
##print(factor_data.info())

#import the 'stock_return' sheet
stock_data=pd.read_csv("../input/Stock_Returns.csv")
##print(stock_data.head(10))
## print(stock_data.info())

#Get the number of distinct stocks from stock returns file
stock_list=stock_data.Stock.unique()
#print(stock_list)

#FINAL TABLE RESULTS
#Create a DataFrame

d = {'StockID':[],
    'R-Square':[],
       'RMSE':[]}
 
FINAL = pd.DataFrame(d,columns=['StockID','R-Square','RMSE'])

for n in stock_list:
    #Step1: Get the factor returns of the relevant stock 
    Factors = factor_data[factor_data['Stock'] == n].set_index('Factor').transpose()
    Factors=Factors.drop(Factors.index[[0]])
   
    #Step2: Get the stock returns of the relevant stock
    Stock=stock_data[stock_data['Stock']==n].T
    Stock.columns = [ 'Stock']
    
    #Step3: Join the two dataframes
    result = pd.concat([Stock, Factors], axis=1, join='inner') #Header will print as Stock, F1, F2, F3
    
    #Step4: Run regression on the data (dep: stock returns, indep: f1,f2,f3)
    #split data
    X=result.drop(['Stock'],axis=1) #all indepdendant variables
    
    Y=result['Stock'] #target variables
    
    #splitting data as test and train, testing on 20% of the data and training on the 80.
    X_train=X.head(int(len(X)*(80.0/100))) #We need the first 80% of the data  for train as it is a time series data 
    X_test=X.head(int(len(X)*(20.0/100))) 
    Y_train=Y.head(int(len(Y)*(80.0/100)))
    Y_test=Y.head(int(len(Y)*(20.0/100)))
    
    reg_all=linear_model.LinearRegression()

    reg_all.fit(X_train,Y_train) #fitting the model for the x and y train

    y_pred=reg_all.predict(X_test) #predicting y(the target variable), on x test

    Rsquare=reg_all.score(X_test,Y_test)
    rmse=np.sqrt(mean_squared_error(Y_test,y_pred)) 
    
    #List all the indepedent variables with the coeffecient from the linear regression model
    coeff_df = pd.DataFrame(X_train.columns.delete(0))
    coeff_df.columns = ['Variable']
    coeff_df["Coeff"] = pd.Series(reg_all.coef_)
    
    #print intercept and rmse
    #print("Intercept: %f" %(reg_all.intercept_))
    FINAL = FINAL.append({'StockID': n,'R-Square': Rsquare,'RMSE': rmse}, ignore_index=True)
   
print(FINAL)


