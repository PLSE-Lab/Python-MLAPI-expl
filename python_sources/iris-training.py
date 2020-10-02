# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
arr = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])

print( arr[0:1,1:3] )
print( arr[2,1:3])
print( arr[1:2,1:3])
print( arr[:2,1:3])