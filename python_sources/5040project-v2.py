# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import scipy.io as sio
import sklearn
from sklearn import svm

# Any results you write to the current directory are saved as output.

# Extracting feature vectors from data
# file structure - input/patient_#/ictal train or non-ictal train or test 

ictal_training_variance = []
nonictal_training_variance= []
test_variance = []

ictal_training_linelength = []
nonictal_training_linelength = []
test_linelength = []

ictal_training_energy = []
nonictal_training_energy = []
test_energy = []

ictal_training_hfo = []
nonictal_training_hfo= []
test_hfo = []

ictal_training_beta = []
nonictal_training_beta = []
test_beta = []

file_path1 = "/kaggle/input/data/data/patient_"

for x in range(7):

    
    nonictal_files = os.listdir((str(file_path1) + str((x+1)) + "/non-ictal train/"))
    
    ictal_files = os.listdir((str(file_path1) + str((x+1)) + "/ictal train/"))
    
    test_files = os.listdir((str(file_path1) + str((x+1)) + "/test/"))
    
    print("Loading files from patient " + str(x+1))
    
    for i in range(len(nonictal_files) - 1):
        

        
        file_name = str(nonictal_files[i])
        first = file_name.split("_")
        

        
        if first[0] != 'patient':
            
            continue 
        
        temp = sio.loadmat((str(file_path1) + str((x+1)) + "/non-ictal train/" + str(nonictal_files[i])), mat_dtype=True)
        
        temp = np.array(temp['data'])
        
        length = len(temp)
        
        
        #variance
        
        variance = np.var(temp)
        nonictal_training_variance.append(variance)
        
        #energy
        
        energy = np.sum(np.square(temp))
        nonictal_training_energy.append(energy)
        
        #line length
        
        L = 0
        
        for j in range(length-1):
            L = L + (temp[j+1] - temp[j]);

        line_length = L/length;
        nonictal_training_linelength.append(line_length)
        
        #power spectrum density


        N = len(temp);
        half = round(length/2)
        Fs = length
        xdft = np.fft.fft(temp);
        xdft = xdft[1:half];
        psdx = (1/(Fs*N)) * np.abs(xdft);
        psdx[2:-1] = 2*psdx[2:-1];
        psdx = 10*np.log10(psdx);
        
        beta = sum(psdx[12:30]);
        nonictal_training_beta.append(beta)
        
        hfo = sum(psdx[100:600]);
        nonictal_training_hfo.append(hfo)
        
    for i in range(len(ictal_files) - 1):
        
        file_name = str(ictal_files[i])
        first = file_name.split("_")
        
        
        
        if first[0] != "patient":
            
            continue 
        
        temp = sio.loadmat((str(file_path1) + str((x+1)) + "/ictal train/" + str(ictal_files[i])), mat_dtype=True)
        
        temp = np.array(temp['data'])
        
        length = len(temp)
        
        #variance
        
        variance = np.var(temp)
        ictal_training_variance.append(variance)
        
        #energy
        
        energy = np.sum(np.square(temp))
        ictal_training_energy.append(energy)
        
        #line length
        
        L = 0
        
        for j in range(length-1):
            L = L + (temp[j+1] - temp[j]);

        line_length = L/length;
        ictal_training_linelength.append(line_length)
        
        #power spectrum density


        N = len(temp);
        Fs = length
        half = round(length/2)
        xdft = np.fft.fft(temp);
        xdft = xdft[1:half];
        psdx = (1/(Fs*N)) * np.abs(xdft);
        psdx[2:-1] = 2*psdx[2:-1];
        psdx = 10*np.log10(psdx);
        
        beta = sum(psdx[12:30]);
        ictal_training_beta.append(beta)
        
        hfo = sum(psdx[100:600]);
        ictal_training_hfo.append(hfo)
        
    for i in range(len(test_files) - 1):
        
        file_name = str(test_files[i])
        first = file_name.split("_")
        
        if first[0] != "patient":
            
            continue 
        
        temp = sio.loadmat((str(file_path1) + str((x+1)) + "/test/" + str(test_files[i])), mat_dtype=True)
        
        temp = np.array(temp['data'])
        
        length = len(temp)
        
        #variance
        
        variance = np.var(temp)
        test_variance.append(variance)
        
        #energy
        
        energy = np.sum(np.square(temp))
        test_energy.append(energy)
        
        #line length
        
        L = 0
        
        for j in range(length-1):
            L = L + (temp[j+1] - temp[j]);

        line_length = L/length;
        test_linelength.append(line_length)
        
        #power spectrum density


        N = len(temp);
        Fs = length
        half = round(length/2)
        xdft = np.fft.fft(temp);
        xdft = xdft[1:half];
        psdx = (1/(Fs*N)) * np.abs(xdft);
        psdx[2:-1] = 2*psdx[2:-1];
        psdx = 10*np.log10(psdx);
        
        beta = sum(psdx[12:30]);
        test_beta.append(beta)
        
        hfo = sum(psdx[100:600]);
        test_hfo.append(hfo)
        
clf = svm.SVC()

variance_training = np.append(ictal_training_variance, nonictal_training_variance)
print(variance_training.shape)

energy_training = np.append(ictal_training_energy,nonictal_training_energy)
print(energy_training.shape)

hfo_training = np.append(ictal_training_hfo,nonictal_training_hfo)
print(hfo_training.shape)

beta_training = np.append(ictal_training_beta,nonictal_training_beta)
print(beta_training.shape)

line_training = np.append(ictal_training_linelength, nonictal_training_linelength)
print(line_training.shape)

X_training = np.column_stack((variance_training, energy_training, line_training, hfo_training, beta_training))

X_training = np.array(X_training, dtype=np.float)

print(X_training.shape)

nonictal_training = np.column_stack((nonictal_training_variance, nonictal_training_energy, nonictal_training_linelength, nonictal_training_hfo, nonictal_training_beta))

test = np.column_stack((test_variance, test_energy, test_linelength, test_hfo, test_beta))

test = np.array(test, dtype=np.float)


#X = np.stack((ictal_training, nonictal_training))

ictal_labels = np.ones(len(ictal_training_variance))
nonictal_labels = np.zeros(len(nonictal_training_variance))

y = np.append(ictal_labels, nonictal_labels)
y = np.array(y, dtype=np.float)
print(y.shape)

#print(X.shape)
#print(y.shape)

clf.fit(X_training, y)  

test_results = clf.predict(test)

print(test_results)
        
    
    
    