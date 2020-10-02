#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3
'''
Randomforest - plot: # of trees - accuracy

Modifications by: Noah Carter
Date 4/25/16
Original Author: Hideki Ikeda
Date 7/11/15
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

def main():
    # loading training data
    print('Loading training data')
    data = pd.read_csv('../input/train.csv')
    
    n_trees = [10, 15]#, 20, 25, 30, 40, 50, 70, 100, 150]

    
    #scoresByTrainingSet=np.array()
    #stdByTrainingSet=np.array()
    performTests(data,42000,12,n_trees,True,True)#,scoresByTrainingSet,stdByTrainingSet)
    
    
def plotByNumberOfTrees(n_trees,scores,sc_array,std_array,showStd):
    plt.plot(n_trees, scores)
    if (showStd):
        plt.plot(n_trees, sc_array + std_array, 'b--')
        plt.plot(n_trees, sc_array - std_array, 'b--')
    plt.ylabel('CV score')
    plt.xlabel('# of trees')
    plt.savefig('cv_trees.png')
def plotByTrainingSetSize(n_obsInTrainingSet,scores,sc_array,std_array,showStd):
    plt.plot(n_obsInTrainingSet, scores)
    if (showStd):
        plt.plot(n_obsInTrainingSet, sc_array + std_array, 'b--')
        plt.plot(n_obsInTrainingSet, sc_array - std_array, 'b--')
    plt.ylabel('CV score')
    plt.xlabel('Size of Training Set')
    plt.savefig('cv_trees2.png')
    
def trainAndTest(n_trees,X_tr,y_tr,scores,scores_std):
    for n_tree in n_trees:
        print(n_tree)
        recognizer = RandomForestClassifier(n_tree)
        score = cross_val_score(recognizer, X_tr, y_tr)
        scores.append(np.mean(score))
        scores_std.append(np.std(score))
def performTests(data,numberOfObservations,numberOfDivisions,n_trees,showStd1,showStd2):#,scoresByTrainingSet,stdByTrainingSet):
    sizeOfGap=int(numberOfObservations/numberOfDivisions)
    
    scoresListByTrainingSet=list()
    stdListByTrainingSet=list()
    n_obsInTrainingSet=list(range(sizeOfGap,numberOfObservations+1,sizeOfGap))
    
    for i in range(sizeOfGap,numberOfObservations+1,sizeOfGap):
        X_tr = data.values[:i, 1:].astype(float)
        y_tr = data.values[:i, 0]
        
        scores = list()
        scores_std = list()
        
        print('Start learning...')
                
        trainAndTest(n_trees,X_tr,y_tr,scores,scores_std)
        
        sc_array = np.array(scores)
        std_array = np.array(scores_std)
        print('Score: ', sc_array)
        print('Std  : ', std_array)
        
        plotByNumberOfTrees(n_trees,scores,sc_array,std_array,showStd1)
        
        scoresListByTrainingSet.append(scores[len(n_trees)-1])  #store the mean and std performance
        stdListByTrainingSet.append(scores_std[len(n_trees)-1]) #for the largest forest size on this
                                                                     #training set size
        
    scoresByTrainingSet=np.array(scoresListByTrainingSet)
    stdByTrainingSet=np.array(stdListByTrainingSet)
    print("Scores by Training Set Size: ",scoresByTrainingSet)
    print("Standard deviations of Scores by Training Set Size: ",stdByTrainingSet)
    plt.figure() #create a new figure
    plotByTrainingSetSize(n_obsInTrainingSet,scoresListByTrainingSet,scoresByTrainingSet,stdByTrainingSet,showStd2)
        
if __name__ == '__main__':
    main()


# In[ ]:


sdf=list(range(3500,42000+1,3

500))
print(sdf)


# In[ ]:




