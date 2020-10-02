#!/usr/bin/python3
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils.extmath import weighted_mode

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# Any results you write to the current directory are saved as output.

NUMBER_OF_WEEKS = 1099//7

def parseData(fileName):
    data = pd.read_csv(fileName)
    table = data.get('visits')
    return data, table
    
def numToDayOfWeek():
    global NUMBER_OF_WEEKS
    matrix = list()
    for i in range(NUMBER_OF_WEEKS):
        week = [j for j in range(7 * i + 1, 7 * i + 8)]
        matrix.append(week)
    return matrix

def normalize(matrix):
    global NUMBER_OF_WEEKS
    sum = 0
    for i in range(NUMBER_OF_WEEKS): #i - week
      for j in range(7):    #j - day in a week
        matrix[i][j] = (i*i/NUMBER_OF_WEEKS) #make old less important
        sum += matrix[i][j]
    for i in range(NUMBER_OF_WEEKS):
        for j in range(7):
            matrix[i][j] = matrix[i][j]/sum
    return matrix

def computeWeight(matrix, table):
    visits = list()
    weight = list()
    for i in table:
        tmp1 = list()
        tmp2 = list()
        vis = i[1:].split(' ')
        for j in vis:
            tmp1.append((int(j) - 1) % 7 + 1)
            tmp2.append(matrix[(int(j) - 1) // 7][(int(j) - 1) % 7])
        visits.append(tmp1)
        weight.append(tmp2)
    return visits, weight



def predict(visits, weight):
    res = list()
    for i in range(len(visits)):
        w = weighted_mode(visits[i], weight[i])
        res.append(' ' + str(int(w[0][0])))
    return res


data, table = parseData("../input/train.csv")
matrix = numToDayOfWeek()
matrix = normalize(matrix)
visits, weight = computeWeight(matrix, table)
prediction = predict(visits, weight)


solution = pd.DataFrame(columns = ['id', 'nextvisit'])
solution['id'] = data.get('id')
solution['nextvisit'] = prediction
solution.to_csv('solution.csv', index = False, sep = ',')

