import pandas as pd
import numpy as np
from os.path import join
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

mob_dataset = "../input/uncover/google_mobility/"
covid_tracking_proj = "../input/uncover/covid_tracking_project"

coviddata = pd.read_csv(join(covid_tracking_proj, "covid-statistics-by-us-states-daily-updates.csv"))
mobdata = pd.read_csv(join(mob_dataset, "us-mobility.csv"))

def removenan(df):
    for k in df.keys():
        df[k].fillna(value = 0)
    return df

mobdata = removenan(mobdata)

statecodes = np.unique(coviddata["state"])
states = np.unique(mobdata["state"])
coviddata["positive"][coviddata["positive"].isnull()] = 0

statewise_total_case = {}
for s in statecodes:
    t = np.asarray(coviddata["positive"][coviddata["state"]==s], dtype = int)
    statewise_total_case[s] = t[::-1]

class CovidPredictionModel():
    def __init__(self, hs = 100, alpha=0.0001):        
        self.alpha = alpha        
        self.model = MLPRegressor(hidden_layer_sizes=(hs, ))
        
    def dickeyFullerTest(self, data, printResults = False):
        adfTest = adfuller(data)
        dfResults = pd.Series(adfTest[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])
        for key,value in adfTest[4].items():
            dfResults['Critical Value (%s)'%key] = value
        if printResults:
            print('Augmented Dickey-Fuller Test Results:')
            print(dfResults)
        isStationary = False
        if dfResults['ADF Test Statistic']< dfResults['Critical Value (5%)']:
            isStationary = True
        return dfResults, isStationary

    def stationarize(self, data):
        
        logdata = [math.log(x+self.alpha) for x in data]
        stdata = []
        for i in range(0, len(logdata)-1):
            stdata.append(logdata[i+1]-logdata[i]) 
        return np.array(stdata)
        

    def destationarize(self, data):
        D = [data[0]]
        for i in range(1,data.shape[0]):
            D.append(D[i-1]+data[i])
        for i in range(0, len(D)):            
            D[i] = math.exp(D[i])-self.alpha
        return np.array(D)
        
    def createdataset(self, data, m):
        X = []
        Y = []
        for i in range(0, len(data)):
            x = np.zeros(m)
            k = 0
            for j in range(i-1,i-m-1,-1):
                if(j>=0 and k<m):
                    x[k] = data[j]
                    k = k+1                        
                else:
                    xt = np.append(x, 0)
                X.append(xt)
                Y.append(data[i])
        return np.array(X), np.array(Y)
    
    def run(self, data, printResult = False, plotResults = False):
        td = [data[0]]
        for i in range(1, data.shape[0]):
            td.append(data[i]-data[i-1])
        data = np.array(td)
        
        dfresults, isStationary = self.dickeyFullerTest(data)
        if not isStationary:            
            data = self.stationarize(data)
        X, Y = self.createdataset(data, 2)
        Xtr, Xtest, Ytr, Ytest = train_test_split(X,Y, test_size=0.3, random_state = 42)        
        #cv_score = cross_val_score(model, X, Y, cv = 10)
        #print(np.mean(cv_score),np.std(cv_score))
        self.model.fit(Xtr,Ytr)
        py = self.model.predict(Xtest)
        error = mean_squared_error(py,Ytest)
        py = self.destationarize(py)
        Ytest = self.destationarize(Ytest)
        
        if printResult:
            print(error)
        if plotResults:
            plt.plot(Ytest, "b")
            plt.plot(py, "g")
        return error

hidden_layer_sizes = [i for i in range(10,100,10)]
errors = []

for hs in hidden_layer_sizes:
    model = CovidPredictionModel(hs = hs)
    #Running the time series model for the state of California as a test case
    errors.append(model.run(statewise_total_case['CA'], printResult = False, plotResults = False))
print(errors)
plt.plot(errors)
plt.show()

fmodel = CovidPredictionModel(hs = 50)
fmodel.run(statewise_total_case['CA'], printResult = True, plotResults = True)





