# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#plots a histogram of the (numerical !!) variable
#column    -column number of the column in the csv table
#bucketSize - is the size of the buckets or bins in the histogram.
def plot_variable(filename,column,bucketSize):
    
    dataframe = pd.read_csv(filename,sep=';',encoding = "ISO-8859-1")
    
    #Achsentitel für plots
    xtitle = dataframe.columns[column]
    ytitle = dataframe.columns[1]
    
    #variable to be plotted
    variableList = dataframe[dataframe.columns[column]].tolist()
    
    #alle Einträge in Integers umwandeln
    try:
        float(variableList[1])
    except ValueError:
        print('Variablenwerte sind nicht numerisch!')
        return False
    variableList = list(map(int, variableList))
    
    #Zielvariable, welche auch geplottet wird
    target = dataframe['Zielvariable'].tolist()
    
    targetList = []
    for i in range(0,len(target)):
        t = target[i]
        if t == 'ja':
            v = 100
        elif t=='nein':
            v = 0
        else:
            print("Der Wert der Zielvariable ist weder 'ja' noch 'nein'!\n")
        targetList.append(v)
    
    
    #combine an item of variableList and targetList whose indices are equal to a new item.
    combinedList = list(zip(variableList, targetList))
    #sort by variableList entries while keeping the indices true for target
    combinedList = sorted(combinedList, key = lambda combinedList : combinedList[0])
    
    i = 0
    bucketedList = []
    #bucket the entries of combinedList
    while i <= len(combinedList)-1:
        #if the end of the list is almost reached
        if(i+bucketSize > len(combinedList)-1):
            combinedBucket = combinedList[i:len(combinedList)]
            #entries are means
            a = np.mean([combinedBucket[j][0]for j in range(0,len(combinedBucket))])
            
            b = np.mean([combinedBucket[j][1]for j in range(0,len(combinedBucket))])
            
            bucketedList.append((a,b))
        #if the end of the list is not almost reached
        else:
            combinedBucket = combinedList[i:i+bucketSize]
            a = np.mean([combinedBucket[j][0]for j in range(0,len(combinedBucket))])
            
            b = np.mean([combinedBucket[j][1]for j in range(0,len(combinedBucket))])
            
            bucketedList.append((a,b))
        i = i+bucketSize
    
    #print(bucketedList)
    n_buckets = len(bucketedList)
    
    varList = [bucketedList[i][0] for i in range(0,n_buckets)]
    targList = [bucketedList[i][1] for i in range(0,n_buckets)]
    
    #print(varList)
    #print(targList)
    

    ind = np.arange(len(varList))  # the x locations for the groups
    width = 0.35  # the width of the bars
    height = 3
    
    fig, ax = plt.subplots()
    #2 bars next to each other
    rects1 = ax.bar(ind-width/2, varList, width,color='SkyBlue',label=xtitle)
    rects2 = ax.bar(ind+width/2, targList, width,color='IndianRed',label=ytitle)
    
    #labeling
    ax.set_xlabel('Quantile von {}'.format(xtitle))
    ax.set_ylabel('%-ige Abschlusswahrscheinlichkeit/\n{}'.format(xtitle))
    ax.legend()
    
    plt.show()
            
plot_variable('../input/TrainData.csv', 2,500)
