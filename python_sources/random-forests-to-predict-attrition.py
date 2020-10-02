#!/usr/bin/env python
# coding: utf-8

# I have applied randomForests in sklearn on the dataset to predict the exits. The accuracy of the trained model on test data is shocking. The reason may be it is simulated data or the variability in the data is so low that 15k data points are way ore than enough to capture the entire randomness in the behaviour of employees. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:



# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier


# In[ ]:


# R
#Read the input Data
rawDataDf = pd.read_csv('../input/HR_comma_sep.csv')

rawDataDf.head()


# In[ ]:


# See the column names to select the feature columns that we want
list(rawDataDf.columns.values)


# In[ ]:


#We split the data into training and test
rawDataDf['isTrain'] = np.random.uniform(0,1,len(rawDataDf)) <= 0.75


# In[ ]:


rawDataDf.head()


# In[ ]:


# The columns sales and salary are categorical we need to facorize them
rawDataDf['sales_fac'] = pd.factorize(rawDataDf['sales'])[0]
rawDataDf['salary_fac'] = pd.factorize(rawDataDf['salary'])[0]


# In[ ]:


# The columns sales and salary are categorical we need to facorize them
train, test = rawDataDf[rawDataDf['isTrain']==True] , rawDataDf[rawDataDf['isTrain']==False]


# In[ ]:


#
print('The training set has ',len(train),' rows and test set has' ,len(test), 'rows')


# In[ ]:


featureCols = ['satisfaction_level',
               'last_evaluation',
               'number_project',
               'average_montly_hours',
               'time_spend_company',
               'Work_accident',
               'promotion_last_5years',
               'sales_fac',
               'salary_fac']
respCol = 'left'


# Create a random forest classifier. By convention, clf means 'classifier'
# I have gone through the documentation of sklearn 
# Some parameters I want to play with are:
#     n_estimators: The number of trees  
#     criterion : The function to measure quality of split. Accepts 'gini' or 'entrophy'

# In[ ]:


effArray = []
#for n in range(3,51):
def rfClassifier(n_estimators,criterion):
    
    clf = RandomForestClassifier(n_estimators= n_estimators, criterion=criterion)
    clf.fit(train[featureCols], train[respCol])

    predPerformanceTable = pd.DataFrame(columns=['predictions','groundTruths'])
    predPerformanceTable['predictions'] = clf.predict(test[featureCols])
    predPerformanceTable['groundTruths'] = list(test[respCol])

    predPerformanceTable['success'] = predPerformanceTable['predictions'] == predPerformanceTable['groundTruths']

    hits = len(predPerformanceTable[predPerformanceTable['success']==True])
    misses = len(test) - hits
    predictionAccuracy =  hits/len(test)
    #effArray.append(predictionAccuracy)
    
    # Lets count the type I and type II errors in prediction
    
    # True Positives and True Negatives
    tp = len(predPerformanceTable[(predPerformanceTable['predictions']==1)&(predPerformanceTable['groundTruths']==1)])
    tn = len(predPerformanceTable[(predPerformanceTable['predictions']==0)&(predPerformanceTable['groundTruths']==0)])
    
    # Type I and Type II errors
    # fp: Type I error
    # fn: Type II error
    
    fp = len(predPerformanceTable[(predPerformanceTable['predictions']==1)&(predPerformanceTable['groundTruths']==0)])
    fn = len(predPerformanceTable[(predPerformanceTable['predictions']==0)&(predPerformanceTable['groundTruths']==1)])
    
    
    # calculating precision and recall
    
    # precision: tp/(tp+fp) 
    # What % of the positives that the model predicts are really positives
    
    precision = tp/(tp+fp)
    
    # recall: 
    # What % of relevant items are selected
    recall = tp/(tp+fn)
    
    
    result = { 'accuracy': predictionAccuracy,
               'performanceTable':predPerformanceTable,
               'precision':precision,
               'recall':recall,
               'typeI': fp,
               'typeII': fn}
    
    return result


# In[ ]:


criterion = 'entropy'
effArray = []
precisionArray = []
recallArray= []

for n in range(1,101):
    
    res = rfClassifier(n_estimators=n, criterion=criterion)
    
    effArray.append(res['accuracy'])
    precisionArray.append(res['precision'])
    recallArray.append(res['recall'])
    
    #print('Trees in Forest: ',n)
    #print('False Positives: ', res['typeI'], ', False Negatives: ', res['typeII'])
    
    
    
    #xAxis = np.arange(3,51)

    #plt.scatter(xAxis, effArray)


# In[ ]:


# I want to see the effect of number of trees in the forest on the prediction accuracy
# we can see that after a point the accuracy doesnt improve much
# Somewhere arounf 20 trees the three metrics stabilize and
# it is not beneficial to go for more trees as the metrics reached pleateau
xaxis = np.arange(1,101)
plt.plot(xaxis, effArray,label='accuracy')
plt.plot(xaxis, precisionArray,label='precision')
plt.plot(xaxis, recallArray,label='recall')
plt.legend()
plt.show()


# In[ ]:


# original data composition
exits = len(rawDataDf[rawDataDf['left']==1])
stays = len(rawDataDf) - exits
print( exits, ' is the number of employees who left ')


# 
# train_imp = train.drop("left", axis=1)
# 
# importances = random_forest.feature_importances_
# std = np.std([tree.feature_importances_ for tree in random_forest.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
# 
# # Print the feature ranking
# print("Feature ranking:")
# 
# for f in range(train_imp.shape[1]):
#     print("%d. %s (%f)" % (f + 1, train_imp.columns[indices[f]], importances[indices[f]]))
# 
# # Plot the feature importances of the forest
# plt.figure(figsize=(10, 5))
# plt.title("Feature importances")
# plt.bar(range(train_imp.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(train_imp.shape[1]), train_imp.columns[indices], rotation='vertical')
# plt.xlim([-1, train_imp.shape[1]])
# plt.show()

# In[ ]:


## The following analysis is adopted from notebook by Ilana Radinsky
## Thanks for the insightful work


clf = RandomForestClassifier(n_estimators= 101, criterion='entropy')
clf.fit(train[featureCols], train[respCol])
importances = clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)

indices = np.argsort(importances)[::-1]


for f in range(len(featureCols)):
    print("%d. %s (%f)" % (f + 1, featureCols[indices[f]], importances[indices[f]]))


# Plot the feature importances of the forest
plt.figure(figsize=(10, 5))
plt.title("Feature importances")
plt.bar(range(len(featureCols)), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(len(featureCols)), featureCols, rotation='vertical')
plt.xlim([-1, len(featureCols)])
plt.show()


# ## Top Factors that determine the employee exit

# 1. satisfaction_level (0.323033)
# 2. time_spend_company (0.196512)
# 3. number_project (0.160477)
# 4. average_montly_hours (0.155055)
# 5. last_evaluation (0.125988)
# 
# These factors 90% of the time determine the employee's decision to stay or leave.

# ## What more can be done?
# Seeing how these top factors affect the decision. The relation between the exit and the direction of the factors

# In[ ]:




