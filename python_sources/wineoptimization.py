# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pulp #pulp is an operations rsearch library - does not come natively within Kaggle, so import using 'Settings' tab
import re
import scipy


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
### ABOUT - This model optimizes purchase of wine based on two key constraints 1.) a total limit in spend ($500) and a max amount of bottles (15)
### the objective is to maximize the points in wine (a measure of enjoyment) for the dollars and bottles available
### this model is based off of the TEDTalks optimization project (https://www.analyticsvidhya.com/blog/2017/10/linear-optimization-in-python/)

### prepare data for optimization ###
#data = pd.read_csv('../input/wine-reviews/winemag-data_first150k.csv',dtype={'price':float}) #old file path
data = pd.read_csv('../input/winesmall/winesmall.csv',dtype={'price':float}) #reduced amount of records to more managable sixe
dtyp = data.dtypes
print(dtyp) #check data types at import
df = pd.DataFrame(data)
dfss = df[['winery', 'variety','points','price']] # clean up and subset dataframe
dfss['varwine'] = df['winery'].astype(str) + df['variety'] + df['points'].astype(str) #create unique wine since no unique wine key exists
dfss[dfss.price > 0.00] #only bring in wines that have a price
dfhd = dfss.head(25) #subset to top 25 wines for testing
dfhd.reset_index(inplace=True)
dtcwinery = pd.DataFrame(dfhd.varwine.unique())
test = dfhd.groupby(['varwine'])['price'].count()

#print(dtcwinery)
#print(test)

### prepare solver ###
# create LP object,
# set up as a maximization problem --> since we want to maximize the points per purchase
prob = pulp.LpProblem('wineoptimization', pulp.LpMaximize)
#opt = scipy.optimize.linprog('wineopt',method='simplex')

# create decision - yes or no to buy the wine?
decision_variables = []
for rownum, row in dfhd.iterrows():
    # variable = set('x' + str(rownum))
    variable = str('x' + str(row['index']))
    variable = pulp.LpVariable(str(variable), lowBound = 0, upBound = 1, cat = 'Integer') # make variable binary
    decision_variables.append(variable)
    
print('Total number of decision variables: ' + str(len(decision_variables)))
print(decision_variables)

# Create optimization Function
total_bottles = ''
for rownum, row in dfhd.iterrows():
    for i, bottles in enumerate(decision_variables):
        if rownum == i:
            formula = row['points'] * bottles
            total_bottles += formula
            
prob += total_bottles
print(total_bottles)
# print('Optimization function: ' + str(total_views))

# Contraints
total_dollars_available = 500.00 # total dollars available to spend
total_bottles_available = 15 # total bottles limit

# Create Constraint 1 - total dollars available
total_dollars = ''
for rownum, row in dfhd.iterrows():
    for i,  bottles in enumerate(decision_variables):
        if rownum == i:
            formula = row['price'] * bottles
            total_dollars += formula
            
prob += (total_dollars <= total_dollars_available)

# Create Constraint 2 - Number of bottles
total_pur_bottles = ''

for rownum, row in dfhd.iterrows():
    for i, bottles in enumerate(decision_variables):
        if rownum == i:
            formula = bottles
            total_pur_bottles += formula
            
prob += (total_pur_bottles == total_bottles_available)

#review function and then write as lp problem
print(prob)
prob.writeLP('wineoptimization.lp')
print(prob.writeLP('wineoptimization.lp'))

#solve optimization and test if solution exists
optimization_result = prob.solve()
print(optimization_result)
print(pulp.LpStatusOptimal)

#print out result variables
for v in prob.variables():
    print(v.name, "=", v.varValue)

#print the objective value
print(prob.variables())
print(prob.variables()[i].varValue)
print(prob.objective.value())
3#return prob.objective.value(), prob.variables()
#res = {}
#for v in prob.variables():
#    varsdict[v.name] = v.varValue
#print(res)
