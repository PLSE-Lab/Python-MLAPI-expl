import numpy as np
import pandas as pd

help(pd.read_csv)

arr1=np.array([0,0,0,0])
arr2=np.array([1,1,1,1])

print (arr1, arr2)


def calcDist(point1,point2): #calculates the Euclidean distance between two points represented as Numpy arrays
    diff=point2-point1
    squared=diff**2
    sum1=squared.sum()
    root=(sum1)**0.5
    return root
    
Distance=calcDist(arr1, arr2)
print(Distance)