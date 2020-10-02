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
    error.append((np.sin(x[i]))+random.uniform(-0.2,0.2))
    actual.append((np.sin(x[i])))
plt.subplot(2,1,1)
plt.scatter(x,error,marker='.')
plt.plot(x,actual,color='red')
#end of data set prep

#ELM start

def tansig(var):
    return ((2/(1+math.exp(-2*var)))-1)
    
weight=[]
least_error=1
final_weight=[]
final_outweigh=[]
output=np.zeros((5001,20))

for i in range(20):
    weight.append(0)
    final_weight.append(0)
    final_outweigh.append(0)
    
for k in range(50):    
    for l in range(20):
        weight[l]=random.uniform(0,1)

    for i in range(5001):
        for j in range(20):
            output[i][j]=tansig(x[i]*weight[j])

    outweigh=np.matmul(lin.pinv((output)),error)
    y_pred=np.matmul(output,outweigh)
    curr_error=mt.mean_squared_error(y_pred,actual)
    if curr_error<least_error:
        least_error=curr_error
        for m in range(20):
            final_weight[m]=weight[m]
            final_outweigh[m]=outweigh[m]

print("final_error is:")
print(least_error)

final_ans=[]
output_final=np.zeros((5001,20))

for i in range(5001):
    final_ans.append(0)
    
plt.subplot(2,1,2)
for i in range(5001):
    for j in range(20):
        output_final[i][j]=tansig(x[i]*final_weight[j])
        
final_ans=np.matmul(output_final,final_outweigh)       
plt.plot(x,actual,label="Actual Data")
plt.plot(x,final_ans,label="Neural net Approximation")
print(mt.mean_squared_error(final_ans,actual))