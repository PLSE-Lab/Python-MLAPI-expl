#!/usr/bin/env python
# coding: utf-8

# # COVID India Forecast
# 
# 
# **TABLE OF CONTENTS**
# 
# 1. [Exploratory data analysis (EDA)](#section1)
# 
#     1.1. [COVID-19 India & Tamil Nadu tendency](#section11)
#     
#     1.2. [Kerala, Karnataka, Maharashtra and WestBengal](#section12)
#     
#     
# 
#     
# 2. [SIR model](#section2)
# 
#     2.1. [Implementing the SIR model](#section21)
#     
#     2.2. [Fit SIR parameters to real data](#section22)
#     
#     
# 3. [Predictions for the early stages of the transmission](#section3)
# 
#     3.1. [Linear Regression](#section31)
#     
# 
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import time
from datetime import datetime
from scipy import integrate, optimize
import warnings
warnings.filterwarnings('ignore')

# ML libraries
import lightgbm as lgb
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from numpy import array
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# # 1. Exploratory data analysis (EDA) <a id="section1"></a>
# 
# First of all, let's take a look on the data structure:

# In[ ]:


dataset = pd.read_csv('../input/covid-india1/covid_india.csv',encoding='cp1252')

#train.Province_State.fillna("None", inplace=True)
print(dataset.shape)
print(dataset.head(5))
print(dataset.describe())

print("Number of states: ", dataset['Detected State'].nunique())
print("Max date", dataset['Date Announced'])
print("Dates go from day", max(dataset['Date Announced']), "to day", min(dataset['Date Announced']), ", a total of", dataset['Date Announced'].nunique(), "days")
print("State with Districs informed: ", dataset.loc[dataset['Detected District']!='None']['Detected State'].unique())


# ## 1.1. COVID-19 India & Tamil Nadu tendency  <a id="section11"></a>
# 
#  

# In[ ]:


confirmed_total_date_india = dataset[['Patient Number','Date Announced']].groupby(['Date Announced']).count()
print(confirmed_total_date_india.sum)
confirmed_total_date_state= dataset[['Patient Number','Date Announced','Detected State','State Patient Number']].groupby(['Date Announced','Detected State']).count()
print(confirmed_total_date_state.head(5))

da1=dataset.loc[dataset['Detected State'] == 'Tamil Nadu']
confirmed_total_date_tamil_nadu = da1[['Patient Number','Date Announced','Detected State','State Patient Number']].groupby(['Date Announced']).count()
print(confirmed_total_date_tamil_nadu)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))
confirmed_total_date_india.plot(ax=ax1)
ax1.set_title("Indian confirmed cases", size=13)
ax1.set_ylabel("Number of cases", size=13)
ax1.set_xlabel("Date", size=13)
confirmed_total_date_tamil_nadu.plot(ax=ax2,color='orange')
ax2.set_title("Tamil Nadu confirmed cases", size=13)
ax2.set_ylabel("Number of cases", size=13)
ax2.set_xlabel("Date", size=13)


# **Observations**: The Indian curve shows that new cases are still increasing. 'Tamil Nadu' curve shows that daily cases are in down trend. 

# ## 1.2. Kerala, Karnataka, Maharashtra and WestBengal <a id="section12"></a>
# 

# In[ ]:


da1=dataset.loc[dataset['Detected State'] == 'Kerala']
confirmed_total_date_kerala = da1[['Patient Number','Date Announced','Detected State','State Patient Number']].groupby(['Date Announced']).count()
#print(confirmed_total_date_kerala)

da1=dataset.loc[dataset['Detected State'] == 'Maharashtra']
confirmed_total_date_Maharashtra = da1[['Patient Number','Date Announced','Detected State','State Patient Number']].groupby(['Date Announced']).count()
#print(confirmed_total_date_Maharashtra)

da1=dataset.loc[dataset['Detected State'] == 'Karnataka']
confirmed_total_date_Karnataka = da1[['Patient Number','Date Announced','Detected State','State Patient Number']].groupby(['Date Announced']).count()
#print(confirmed_total_date_Karnataka)

da1=dataset.loc[dataset['Detected State'] == 'West Bengal']
confirmed_total_date_westbengal = da1[['Patient Number','Date Announced','Detected State','State Patient Number']].groupby(['Date Announced']).count()
#print(confirmed_total_date_westbengal)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))
confirmed_total_date_kerala.plot(ax=ax1)
ax1.set_title("Kerala confirmed cases", size=13)
ax1.set_ylabel("Number of cases", size=13)
ax1.set_xlabel("Date", size=13)
confirmed_total_date_Maharashtra.plot(ax=ax2,color='orange')
ax2.set_title("Maharastra confirmed cases", size=13)
ax2.set_ylabel("Number of cases", size=13)
ax2.set_xlabel("Date", size=13)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))
confirmed_total_date_Karnataka.plot(ax=ax1)
ax1.set_title("karnataka confirmed cases", size=13)
ax1.set_ylabel("Number of cases", size=13)
ax1.set_xlabel("Date", size=13)
confirmed_total_date_westbengal.plot(ax=ax2,color='orange')
ax2.set_title("West Bengal confirmed cases", size=13)
ax2.set_ylabel("Number of cases", size=13)
ax2.set_xlabel("Date", size=13)


# **Observations**:
# 
# * other than Kerala, all state's cases are upward trending

# As a fraction of the total population of each state:

# In[ ]:


pop_tn = 72147030.
pop_kerala = 33406061.
pop_wb = 91276115.
pop_ma = 112374333.

total_date_tn_ConfirmedCases = confirmed_total_date_tamil_nadu/pop_tn*100.

total_date_kerala_ConfirmedCases = confirmed_total_date_kerala/pop_kerala*100.

total_date_wb_ConfirmedCases = confirmed_total_date_westbengal/pop_wb*100.

total_date_ma_ConfirmedCases = confirmed_total_date_Maharashtra/pop_ma*100.


plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
total_date_tn_ConfirmedCases.plot(ax=plt.gca(), title='Tamil Nadu')
plt.ylabel("Fraction of population infected")
plt.ylim(0, 0.0005)

plt.subplot(2, 2, 2)
total_date_kerala_ConfirmedCases.plot(ax=plt.gca(), title='Kerala')
plt.ylim(0, 0.0005)

plt.subplot(2, 2, 3)
total_date_wb_ConfirmedCases.plot(ax=plt.gca(), title='WB')
plt.ylabel("Fraction of population infected")
plt.ylim(0, 0.0005)

plt.subplot(2, 2, 4)
total_date_ma_ConfirmedCases.plot(ax=plt.gca(), title='MA')
plt.ylim(0, 0.0005)


# In order to compare with world countries, the fraction got infected is very very less

# # 2. SIR model <a id="section2"></a>
# 
#  I'll move on to one of the most famous epidemiologic models: SIR. 
# 
# SIR is a simple model that considers a population that belongs to one of the following states:
# 1. **Susceptible (S)**. The individual hasn't contracted the disease, but she can be infected due to transmisison from infected people
# 2. **Infected (I)**. This person has contracted the disease
# 3. **Recovered/Deceased (R)**. The disease may lead to one of two destinies: either the person survives, hence developing inmunity to the disease, or the person is deceased. 
# 
# <img src="https://www.lewuathe.com/assets/img/posts/2020-03-11-covid-19-dynamics-with-sir-model/sir.png" width="500px">
# Image by Kai Sasaki from [lewuathe.com](https://www.lewuathe.com/covid-19-dynamics-with-sir-model.html)
# **Assumptions**
# 1. We will consider that people develops immunity (in the long term, immunity may be lost and the COVID-19 may come back within a certain seasonality like the common flu) and there is no transition from recovered to the remaining two states. 
# 
#  2.No one is added to the susceptible group, since we are ignoring births and immigration. The only way an individual leaves the susceptible group is by becoming infected. We assume that the time-rate of change of  S(t),  the number of susceptibles
# 
# 3. Where $\beta$ is the contagion rate of the pathogen and $\gamma$ is the recovery rate. $\beta$ is the average number of contacts per person per time, multiplied by the probability of disease transmission in a contact between a susceptible and an infectious subject,
# 
# 4. N is the total population or sum of S, I, R
# 
# 4. S(t), I(t) and R(t) are precise number at a particular time
# 
# With this, the differential equations that govern the system are:
# 
# $$ {dS \over dt} = - {\beta S I \over N} $$
# 
# $$ {dI \over dt} = {\beta S I \over N} - \gamma I$$
# 
# $$ {dR \over dt} = \gamma I$$
# 
# Firstly note that from:
# 
# $$ {dS \over dt}+{dI \over dt}+{dR \over dt}=0$$
# 
# it follows that:
# 
# S(t)+I(t)+R(t)=N

# ## 2.1. Implementing the SIR model <a id="section21"></a>
# 
# SIR model can be implemented in many ways: from the differential equations governing the system, within a mean field approximation or running the dynamics in a social network (graph). For the sake of simplicity, I'vem chosen the first option, and we will simply run a numerical method (Runge-Kutta) to solve the differential equations system. 
# 
# The functions governing the dif.eqs. are:

# In[ ]:


# Susceptible equation
def fa(N, a, b, beta):
    fa = -beta*a*b
    return fa

# Infected equation
def fb(N, a, b, beta, gamma):
    fb = beta*a*b - gamma*b
    return fb

# Recovered/deceased equation
def fc(N, b, gamma):
    fc = gamma*b
    return fc


# In order to solve the differential equations system, we develop a  4rth order [Runge-Kutta](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) method:

# In[ ]:


# Runge-Kutta method of 4rth order for 3 dimensions (susceptible a, infected b and recovered r)
def rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs):
    a1 = fa(N, a, b, beta)*hs
    b1 = fb(N, a, b, beta, gamma)*hs
    c1 = fc(N, b, gamma)*hs
    ak = a + a1*0.5
    bk = b + b1*0.5
    ck = c + c1*0.5
    a2 = fa(N, ak, bk, beta)*hs
    b2 = fb(N, ak, bk, beta, gamma)*hs
    c2 = fc(N, bk, gamma)*hs
    ak = a + a2*0.5
    bk = b + b2*0.5
    ck = c + c2*0.5
    a3 = fa(N, ak, bk, beta)*hs
    b3 = fb(N, ak, bk, beta, gamma)*hs
    c3 = fc(N, bk, gamma)*hs
    ak = a + a3
    bk = b + b3
    ck = c + c3
    a4 = fa(N, ak, bk, beta)*hs
    b4 = fb(N, ak, bk, beta, gamma)*hs
    c4 = fc(N, bk, gamma)*hs
    a = a + (a1 + 2*(a2 + a3) + a4)/6
    b = b + (b1 + 2*(b2 + b3) + b4)/6
    c = c + (c1 + 2*(c2 + c3) + c4)/6
    return a, b, c


# And finally, to obtain the evolution of the disease we simply define the initial conditions and call the rk4 method:

# In[ ]:


def SIR(N, b0, beta, gamma, hs):
    
    """
    N = total number of population
    beta = transition rate S->I
    gamma = transition rate I->R
    k =  denotes the constant degree distribution of the network (average value for networks in which 
    the probability of finding a node with a different connectivity decays exponentially fast
    hs = jump step of the numerical integration
    """
    
    # Initial condition
    a = float(N-1)/N -b0
    b = float(1)/N +b0
    c = 0.

    sus, inf, rec= [],[],[]
    for i in range(10000): # Run for a certain number of time-steps
        sus.append(a)
        inf.append(b)
        rec.append(c)
        a,b,c = rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs)

    return sus, inf, rec


# Results obtained for N=Indian population, only one initial infected case, $\beta=0.7$, $\gamma=0.2$ and a leap pass $h_s = 0.1$ are shown below:

# In[ ]:


# Parameters of the model
N = 13000000
b0 = 0
beta = .7
gamma = 0.2
hs = 0.1

sus, inf, rec = SIR(N, b0, beta, gamma, hs)

f = plt.figure(figsize=(8,5)) 
plt.plot(sus, 'b.', label='susceptible');
plt.plot(inf, 'r.', label='infected');
plt.plot(rec, 'c.', label='recovered/deceased');
plt.title("SIR model")
plt.xlabel("time", fontsize=10);
plt.ylabel("Fraction of population", fontsize=10);
plt.legend(loc='best')
plt.xlim(0,1000)
plt.savefig('SIR_example.png')
plt.show()


# **Observations**: 
# * The number of infected cases increases for a certain time period, and then eventually decreases given that individuals recover/decease from the disease
# * The susceptible fraction of population decreases as the virus is transmited, to eventually drop to the absorbent state 0
# * The oposite happens for the recovered/deceased case
# 
# 

#  # 3. Predictions for the early stages of the transmission <a id="section3"></a>
# 
# 

# 1. ## 3.1. Linear Regression  <a id="section31"></a>
# Linear regression attempts to model the relationship between two variables by fitting a linear equation to observed data. One variable is considered to be an explanatory variable, and the other is considered to be a dependent variable. For example, a modeler might want to relate the weights of individuals to their heights using a linear regression model.
# A linear regression line has an equation of the form Y = a + bX, where X is the explanatory variable and Y is the dependent variable. The slope of the line is b, and a is the intercept (the value of y when x = 0).
# 
# 

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


datasetcons = pd.read_csv('../input/covidindia2/covid_india_consolidated.csv',encoding='cp1252')

#train.Province_State.fillna("None", inplace=True)
print(datasetcons.shape) #47
print(datasetcons.head(5))
print(datasetcons.describe())

array = datasetcons.values
#print(array)
X = array[:,2].reshape(-1,1)
y = array[:,1].reshape(-1,1)
#print('value of X',X)
#print('value of Y',y)



# In[ ]:


# Split-out validation dataset
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


#print(X_train)
#print(Y_train)
#print(X_validation)

#print(Y_validation)
print('Total X training data', len(X_train))
print('Total Y training data',len(Y_train))

print('Total X Test data',len(X_validation))
print('Total Y Test data',len(Y_validation))

#print(type(X_train))
#print(type(Y_train))


# In[ ]:


#try predicting
regressor = LinearRegression()  
regressor.fit(X_train, Y_train) 
predictions = regressor.predict(X_validation)
print(predictions)
print(Y_validation)
plt.scatter(X_validation, Y_validation)
plt.plot(X_validation, predictions, color='red')

predict= regressor.predict(np.array([48,49,50,51,52,53,54,55,56,57,58,59,60]).reshape(-1,1))
print(predict)
plt.scatter(np.array([48,49,50,51,52,53,54,55,56,57,58,59,60]), predict)
plt.plot(np.array([48,49,50,51,52,53,54,55,56,57,58,59,60]), predict, color='black')

#To retrieve the intercept:
print('Value of intercept a=',regressor.intercept_)
#For retrieving the slope:
print('Value of slope m=',regressor.coef_)

