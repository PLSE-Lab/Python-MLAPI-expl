# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
def AXB(a,b):
    ans = 0
    for i in range(len(a)):
        ans=ans+a[i]*b[i]
    return ans

def AXA(a):
    ans = 0
    for i in range(len(a)):
        ans=ans+a[i]*a[i]
    return np.sqrt(ans)

def main(a,b):
    return(AXB(a,b)/(AXA(a)*AXA(b)))