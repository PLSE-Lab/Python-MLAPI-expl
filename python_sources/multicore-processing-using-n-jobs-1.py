'''
Sept18-2018
Mahesh Babu Mariappan (https://www.linkedin.com/in/mahesh-babu-mariappan)
Source code for comparing multicore machine learning with single core machine learning for sklearn.cross_validation
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# Any results you write to the current directory are saved as output.

Results:
Time elapsed for machine_learning_with_single_core: 00h:00m:18.36s
Time elapsed for machine_learning_with_multi_core: 00h:00m:08.04s
'''
import numpy as np 
import pandas as pd 
import os
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import time
print(os.listdir("../input"))


if __name__ == '__main__':
    startTime_singlecore = time.time()
    digits = load_digits()
    X, y = digits.data,digits.target
    machine_learning_with_single_core = cross_val_score(SVC(), X, y, cv=20, n_jobs=1)
    endTime_singlecore = time.time()
    #print elapsed time in hh:mm:ss format
    hours, rem = divmod(endTime_singlecore-startTime_singlecore, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time elapsed for machine_learning_with_single_core: {:0>2}h:{:0>2}m:{:05.2f}s".format(int(hours),int(minutes),seconds))

    startTime_multicore = time.time()
    digits = load_digits()
    X, y = digits.data,digits.target
    machine_learning_with_multiple_cores = cross_val_score(SVC(), X, y, cv=20, n_jobs=-1)
    endTime_multicore = time.time()
    #print elapsed time in hh:mm:ss format
    hours, rem = divmod(endTime_multicore-startTime_multicore, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Time elapsed for machine_learning_with_multiple_cores: {:0>2}h:{:0>2}m:{:05.2f}s".format(int(hours),int(minutes),seconds))

    if endTime_multicore-startTime_multicore < endTime_singlecore-startTime_singlecore :
        print('Multicore is faster')
    else:
        print('Singlecore is faster')