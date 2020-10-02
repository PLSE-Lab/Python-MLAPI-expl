# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

#plots a probability distribution or the target for each category
#column    -column number of the column in the csv table
def plot_variable(filename,column):
    
    dataframe = pd.read_csv(filename,sep=';', encoding = "ISO-8859-1")
    
    #Variablenspalte
    xtitle = dataframe.columns[column]
    #Zielvariablenspalte
    ytitle = dataframe.columns[1]
    
    #variable to be plotted
    variableList = []
    #fill variableList with data
    variableList = dataframe[dataframe.columns[column]].tolist()
    
    #Zielvariable, welche auch geplottet wird
    target = dataframe['Zielvariable'].tolist()
    
    #target variable which will be plotted as well.
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
    #print(combinedList)
    
    
    #save target values as values in a dictionary. The key is the categorical variable.
    mydict={}
    for item in combinedList:
        key = item[0]
        value = item[1]
        if not(key in mydict):
            mydict[key] = [value]
        else:
            mydict[key].append(value)

    for key in mydict:
        l = mydict[key]
        mean = np.mean(l)
        mydict[key]=mean
    
    #print(mydict)
    
    #ind = np.arange(len(mydict))  # the x locations for the groups
    width = 0.35  # the width of the bars
    
    #2 bars next to each other
    rects1 = plt.bar(range(0,len(mydict)),list(mydict.values()), width,color='IndianRed',label=xtitle)
    
    #display variable names as xticks on the x axis
    plt.xticks(range(0,len(mydict)), list(mydict.keys()))
    #labeling
    plt.xlabel(xtitle)
    plt.ylabel('%-ige Wahrscheinlichkeit für Kauf')
    #plt.legend()
    
    plt.show()
    
plot_variable('../input/TrainData.csv', 6)
