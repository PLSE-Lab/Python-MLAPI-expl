# ---------Priliminary Checks--------------------------------------------------
#On WIN10, python version 3.5
#C:\Users\AMOD\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Python 3.5
#To check that the launcher is available, execute in Command Prompt: py
#To install numPy: C:\WINDOWS\system32>py.exe -m pip install numpy
#To install sciPy: C:\WINDOWS\system32>py.exe -m pip install scipy
#To install pandas: C:\WINDOWS\system32>py.exe -m pip install pandas
#To install matplotlib: C:\WINDOWS\system32>py.exe -m pip install matplotlib
# -----------------------------------------------------------------------------

import numpy as npy    # Remember: NumPy is zero-indexed

#import pandas as pnd
#from pandas import Series, DataFrame

import scipy 
from scipy import stats

import matplotlib.pyplot as plt

# ----------Read data from TXT/CSV/XLSX formats-------------------------------
# Get data from a SIMPLE txt fie. loadtxt does not work with mixed data type
# rawData = npy.loadtxt('statsData.txt', delimiter=' ', skiprows=1, dtype=str)
dataRaw = npy.loadtxt('statsCSV.txt', delimiter=',', skiprows=1)

# Useful for COLUMNS with STRINGS and missing data
#rawData = npy.genfromtxt("statsData.txt", dtype=None, delimiter=" ", skip_header=1, names=True)

# Get data from a CSV file
#dataRaw = pnd.read_csv('statsData.csv', sep=',', header=0)
#dataRaw = pnd.read_excel('statsData.xlsx', sheetname='inputDat')
# -----------------------------------------------------------------------------
npy.set_printoptions(precision=3)    #Precision for floats, suppresses end zeros
npy.set_printoptions(suppress=True)  #No scientific notation for small numbers

#Alternatively use formatter option, it will not suppress end ZEROS
npy.set_printoptions(formatter={'float': '{: 8.2f}'.format})

mean = npy.mean(dataRaw, axis = 0)  # axis keyword: 0 -> columns, 1 -> rows
print("Mean:     ", mean)
medn = npy.median(dataRaw, axis = 0)
print("Median:   ", medn)
sdev = npy.std(dataRaw, axis = 0)
print("SD:       ", sdev)

# Generate plot
n = dataRaw[:,1].size               #Python arrays are 0-based
x = npy.arange(1, n+1)
y = dataRaw[:, 1]                   #Read first column of the input data
plt.title("Crude Price") 
plt.xlabel("Day") 
plt.ylabel("Crude Price in US$") 
plt.plot(x,y) 
plt.show()
