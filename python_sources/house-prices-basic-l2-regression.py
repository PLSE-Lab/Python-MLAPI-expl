import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import torch.optim as optim
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
"""
corr = train.corr()
corr.sort_values('SalePrice', ascending = False, inplace = True)
possible = [1st 6 of corr.SalePrice excluding GarageArea and TotalBsmtSF]
Also, in train.isnull().sum() for missing values, 'possible' features have none missing
"""
possible = ['OverallQual', 'GrLivArea', 'GarageCars', '1stFlrSF']
train = pd.read_csv('train.csv')
train = train[(train.GrLivArea < 4000) & (train['1stFlrSF'] < 4000)]

#Scaling every factor from 1-10
train.GrLivArea/=400
train['1stFlrSF']/=400


test = pd.read_csv('test.csv')
test.GrLivArea/=400
test['1stFlrSF']/=400
test_id = test.Id
for i in test.columns:
    if i not in possible:
        test = test.drop(i, 1)
#People more likely to have 2 cars, also only one value is missing so..
test.GarageCars.fillna(2, inplace = True)

lr = 1e-4 #learning rate
wd = 1e-4 #weight decay

ids = train.Id
target = torch.Tensor(train.SalePrice)/100
inputs = np.zeros((len(target), 4))
for i in range(4):
    inputs[:,i] = train[possible[i]].astype(float)

inputs = torch.Tensor(inputs)

#4 variables + intercept = 5
#initial values take from graph comparision
weights = torch.rand(4, requires_grad = True)
bias = torch.rand(1, requires_grad = True)

optimizer = optim.SGD([weights, bias], lr = lr, weight_decay = wd) #Stochastic Gradient Descent

for i in range(10000):
    hx = inputs @ weights + bias
    loss = ( (target-hx)**2 ).mean()
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()

#Checking
print(weights, bias)
print("target, prediction")
print(target[1003])
pred = (inputs@weights + bias).detach().numpy()
print(pred[1003])


df = pd.DataFrame()
df['Id'] = test_id
actual_inputs = torch.Tensor(test.values.astype(float))
pred = (actual_inputs @ weights + bias)*100
df['SalePrice'] = pred.detach().numpy()
df.to_csv('pytorch_sub2.csv', index=False)

