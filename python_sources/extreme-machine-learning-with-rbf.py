import numpy as np
from numpy import linalg as lin
from numpy import ndarray as nd
import pandas as pd
from matplotlib import pyplot as plt
import random
import math
import sklearn
from sklearn import metrics as mt
x=[]
error=[]
actual=[]

for  i in range(5001):
    x.append(-10+i/250)
    if x[i]==0:
        error.append(1)
        actual.append(1)
    else:
        error.append((np.sin(x[i])/x[i])+random.uniform(-0.2,0.2))
        actual.append((np.sin(x[i])/x[i]))
plt.subplot(2,1,1)
plt.scatter(x,error,marker='.')
plt.plot(x,actual,marker='.',color='red')
#end of data set prep

#ELM start

def square(x):
    return (x*x)
    
def rbf(centr,spred,var):
    return (math.exp(-1*square(var-centr)/(2*square(spred))))
    
center=[]
spread=[]
output=np.zeros((5001,50))
least_error=10
final_center=[]
final_spread=[]
final_outweigh=[]

for i in range(50):
    center.append(0)
    spread.append(0)
    final_outweigh.append(0)
    final_spread.append(0)
    final_center.append(0)
    
for k in range(50):    
    for l in range(50):
        center[l]=random.uniform(-10,10)
        spread[l]=random.uniform(0.5,2)

    for i in range(5001):
        for j in range(50):
            output[i][j]=rbf(center[j],spread[j],x[i])

    outweigh=np.matmul(lin.pinv((output)),error)
    y_pred=np.matmul(output,outweigh)
    curr_error=mt.mean_squared_error(y_pred,actual)
    if curr_error<least_error:
        least_error=curr_error
        for m in range(50):
            final_center[m]=center[m]
            final_spread[m]=spread[m]
            final_outweigh[m]=outweigh[m]

print("final_error is:")
print(least_error)

final_ans=[]
output_final=np.zeros((5001,50))

for i in range(5001):
    final_ans.append(0)
    
plt.subplot(2,1,2)
for i in range(5001):
    for j in range(50):
        output_final[i][j]=rbf(final_center[j],final_spread[j],x[i])
        
final_ans=np.matmul(output_final,final_outweigh)       
plt.plot(x,actual,label="Actual Data")
plt.plot(x,final_ans,label="Neural net Approximation")
print(mt.mean_squared_error(final_ans,actual))