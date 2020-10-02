#!/usr/bin/env python
# coding: utf-8

# Name: Jeffrey Jose, ID: 1313512 <br/>
# Name: Christian Richardson, ID: 1312908

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#import statements I think may be neccessary for this assignment
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings("ignore")
# Any results you write to the current directory are saved as output.


# In[2]:


from sklearn.ensemble import ExtraTreesClassifier

train = pd.read_csv('../input/mnist_train.csv', dtype = int)
x_train = train.drop(['label'], axis = 1)
y_train = train['label']

test = pd.read_csv('../input/mnist_test.csv', dtype = int)
x_test = test.drop(['label'], axis = 1)
y_test = test['label']

x_train = np.array(x_train)
x_test = np.array(x_test)
#Assume max_features is set to auto by default
#Method to initialise the classifier to
def direct_init_clf(clf, n_estimator, max_depth):
    return clf(n_estimators = n_estimator, random_state = 1312908, bootstrap = True, oob_score = True, n_jobs = -1, max_depth = max_depth)


# ### ExtraTreesClassifier Ensemble

# In[3]:


max_depth_range = range(10, 61, 10)
#param_grid = dict(max_depth = max_depth)
highIndexPositon = 0 #This variable holds the index position of the best max_depth
#Optional
oobArray = [] #Holds all the oob scores generated
bestDepth = [] #Holds all the max_depths used
#Finding the best depth
for i in range(len(max_depth_range)):
    clf_extraTrees = ""
    clf_extraTrees = direct_init_clf(ExtraTreesClassifier, 300, max_depth_range[i])
    clf_extraTrees.fit(x_train, y_train)
    oobArray.append(clf_extraTrees.oob_score_)
    bestDepth.append(max_depth_range[i])
    
    #print("Test Function")

currentHighScore = 0
currentHighIndexPosition = 0
highScore = 0
for k in range(len(oobArray)):
    if(k == 0):
        currentHighScore = oobArray[k]
        currentHighIndexPosition = 0
    else:
        if(oobArray[k] > currentHighScore):
            currentHighScore = oobArray[k]
            currentHighIndexPosition = k
    
highScore = currentHighScore
highIndexPosition = currentHighIndexPosition

print("The highest OOB Score is " + str(highScore) + " where the max depth is " + str(bestDepth[highIndexPosition]))


# In[4]:


from sklearn.metrics import accuracy_score

best_clf = direct_init_clf(ExtraTreesClassifier, 300, max_depth_range[highIndexPosition])

best_clf.fit(x_train, y_train)

pred = best_clf.predict(x_test)
accuracyScore = best_clf.score(x_test, y_test)
print("The prediction accuracy (tested with the best max_depth) is: {:0.2f}%".format(accuracyScore * 100))


# In[5]:


#decisionStorage = []
decisionStorage = best_clf.oob_decision_function_
print(decisionStorage)


# In[6]:


#predict will give either 0 or 1 as output
#predict_proba will give the only probability of 1
probability = best_clf.predict_proba(x_test)
print(probability)


# ### Bagged And Boosted Single Extra Trees

# In[7]:


import operator
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

def single_extra_tree(depth = None, seed = 1312908):
    return ExtraTreesClassifier(max_depth = depth, n_estimators = 1, bootstrap = False, random_state = seed)

boost = AdaBoostClassifier(base_estimator = single_extra_tree(), n_estimators = 10, random_state = 1312908)
bag = BaggingClassifier(base_estimator = boost, n_estimators = 30, bootstrap = True, oob_score = True, n_jobs = -1, random_state = 1312908)

rangeList = range(10, 61, 10)
oobArraySecond = []
bestDepthSecond = []
for i in range(len(rangeList)):
    bag_clf = ""
    boost = ""
    bag = ""
    boost = AdaBoostClassifier(base_estimator = single_extra_tree(rangeList[i], 1312908), n_estimators = 10, random_state = 1312908)
    bag = BaggingClassifier(base_estimator = boost, n_estimators = 30, bootstrap = True, oob_score = True, n_jobs = -1, random_state = 1312908)
    bag_clf = bag
    bag_clf.fit(x_train, y_train)
    oobArraySecond.append(bag_clf.oob_score_)
    bestDepthSecond.append(rangeList[i])
    
index, value = max(enumerate(oobArraySecond), key = operator.itemgetter(1))
print("The highest OOB score is " + str(value) + " where the max depth is " + str(bestDepthSecond[index]))


# In[8]:


boost1 = AdaBoostClassifier(base_estimator = single_extra_tree(bestDepthSecond[index], 1312908), n_estimators = 10)
bestSecond_clf = BaggingClassifier(base_estimator = boost1, n_estimators = 30, bootstrap = True, oob_score = True, n_jobs = -1, random_state = 1312908)
bestSecond_clf.fit(x_train, y_train)
pred_second = best_clf.predict(x_test)
print("The prediction accuracy: " + str(accuracy_score(y_test, pred_second)))
#accuracyScoreSecond = bestSecond_clf.score(x_test, y_test)
#print("The prediction accuracy (tested with the best max_depth) is: {:0.2f}%".format(accuracyScoreSecond * 100))


# In[9]:


#decisionStorageSecond = []
decisionStorageSecond = bestSecond_clf.oob_decision_function_
print(decisionStorageSecond)


# In[10]:


#predict will give either 0 or 1 as output
#predict_proba will give the only probability of 1
probabilityBag = bestSecond_clf.predict_proba(x_test)
print(probabilityBag)


# ### Bagged And Boosted Single Extra Trees On PCA

# In[11]:


from sklearn.pipeline import Pipeline
#Bag a pipeline of PCA + AdaBoost (single_extra_tree())
num_components = [20, 40, 60]

pipeLineOOBArray = []
for i in range(len(num_components)):
    pca = ""
    pip = ""
    bagPipe = ""
    pca = PCA(n_components = num_components[i], svd_solver = 'randomized', random_state = 1312908)
    pip = Pipeline([('pca', pca), ('ada_xt', AdaBoostClassifier(base_estimator = single_extra_tree(depth = bestDepthSecond[index]), n_estimators = 10, random_state = 1312908))])
    bagPipe = BaggingClassifier(base_estimator = pip, n_estimators = 30, bootstrap = True, oob_score = True, n_jobs = -1, random_state = 1312908)
    bagPipe.fit(x_train, y_train)
    pipeLineOOBArray.append(bagPipe.oob_score_)

currHigh = 0
highOOB = 0
indexer = 0
rIndexer = 0
for k in range(len(pipeLineOOBArray)):
    if(k == 0):
        currHigh = pipeLineOOBArray[k]
        indexer = 0
    else:
        if(pipeLineOOBArray[k] > currHigh):
            currHigh = pipeLineOOBArray[k]
            indexer = k

highOOB = currHigh
rIndexer = indexer

print("The best number of component: " + str(num_components[rIndexer]) + " and the OOB Score is " + str(highOOB))


# In[12]:


pca = ""
pip = ""
bagPipe = ""
pca = PCA(n_components = num_components[rIndexer], svd_solver = 'randomized', random_state = 1312908)
pip = Pipeline([('pca', pca), ('ada_xt', AdaBoostClassifier(base_estimator = single_extra_tree(depth = bestDepthSecond[index]), n_estimators = 10, random_state = 1312908))])
bagPipe = BaggingClassifier(base_estimator = pip, n_estimators = 30, bootstrap = True, oob_score = True, n_jobs = -1, random_state = 1312908)

bagPipe.fit(x_train, y_train)
bagPipePred = bagPipe.predict(x_test)
print("Accuracy Score With PipeLine: " + str(accuracy_score(y_test, bagPipePred)))


# In[13]:


#pipeDecisionStorage = []
pipeDecisionStorage = bagPipe.oob_decision_function_
print(pipeDecisionStorage)


# In[14]:


pipeProbability = bagPipe.predict_proba(x_test)
print(pipeProbability)


# ## Voting

# ### Between All Classifiers

# In[15]:


#decisionStorage, decisionStorageSecond, pipeDecisionStorage
#Estimating how good voting all three classifier might be

#Between all the classifiers
votingArray = np.sum([decisionStorage, decisionStorageSecond, pipeDecisionStorage], axis = 0)
votingArray


# In[16]:


#Accuracy for the train-set prediction
accuracy_score(y_train, np.argmax(votingArray, axis = 1))


# ### Voting For Decision Storage

# In[17]:


decisionVotingArray = np.sum([decisionStorage], axis = 0)
decisionVotingArray


# In[18]:


#Accuracy for the train-set prediction
accuracy_score(y_train, np.argmax(decisionVotingArray, axis = 1))


# ### Voting For decisionStorageSecond

# In[19]:


decisionVotingArraySecond = np.sum([decisionStorageSecond], axis = 0)
decisionVotingArraySecond


# In[20]:


accuracy_score(y_train, np.argmax(decisionVotingArraySecond, axis = 1))


# ### Voting For The Pipe Line

# In[21]:


decisionVotingArrayPipe = np.sum([pipeDecisionStorage], axis = 0)
decisionVotingArrayPipe


# In[22]:


accuracy_score(y_train, np.argmax(decisionVotingArrayPipe, axis = 1))


# ### Voting Between Pairs: decisionStorage & decisionStorageSecond

# In[23]:


votingArrayOneAndTwo = np.sum([decisionStorage, decisionStorageSecond], axis = 0)
votingArrayOneAndTwo


# In[24]:


accuracy_score(y_train, np.argmax(votingArrayOneAndTwo, axis = 1))


# ### Voting Between Pairs: decisionStorage & pipeDecisionStorage

# In[25]:


votingArrayOneAndThree = np.sum([decisionStorage, pipeDecisionStorage], axis = 0)
votingArrayOneAndThree


# In[26]:


accuracy_score(y_train, np.argmax(votingArrayOneAndThree, axis = 1))


# ### Voting Between Pair: decisionStorageSecond & pipeDecisionStorage

# In[27]:


votingArrayTwoAndThree = np.sum([decisionStorageSecond, pipeDecisionStorage], axis = 0)
votingArrayTwoAndThree


# In[28]:


accuracy_score(y_train, np.argmax(votingArrayTwoAndThree, axis = 1))


# *Which of these 7 combinations has the highest OOB accuracy estimate?* <br />
# From the output that we produced, we can see that the option which contained the ExtraTreesClassifier contained the highest OOB score as well as the highest accuracy score with 96.91%.
# 

# ## Stacking
# 

# In[29]:


from sklearn.linear_model import LogisticRegression
x_train_meta = np.concatenate([decisionStorage, decisionStorageSecond, pipeDecisionStorage], axis = 1)
x_test_meta = np.concatenate([probability, probabilityBag, pipeProbability], axis = 1)

#Logistic Regression Linear Classifier
lgr_clf = LogisticRegression(C = 50)
lgr_clf.fit(x_train_meta, y_train)
lgr_pred = lgr_clf.predict(x_test_meta)

lgr_accuracy = accuracy_score(y_test, lgr_pred)
lgr_accuracy


# *What is the test set accuracy and is it better than simple voting?* <br />
# The test set accuracy when used by the stacking method comes out to 97.25%. The test set accuracy when used by the simple voting method comes out to 96.91%. So therefore the stacking is much better than simple voting.

# ## Stacking Classifier To Detect Mismatched Images

# In[30]:


#find the index of the first missclassified item
def find_misclassified(c, true_labels, preds):
    for i in range(len(preds)):
        if c == true_labels[i]:
            if preds[i] != true_labels[i]:#IF missclassified
                return i


# In[49]:


import matplotlib.pyplot as plt
#for numbers 1 -> 10, x_test item at the missclassified index
for j in range(10):
    idx = find_misclassified(j, y_test, lgr_pred)
    display = x_test[idx].reshape(28,28)
    plt.imshow(display)
    plt.title(str(y_test[idx]) + " misclassified as " + str(lgr_pred[idx]))
    plt.show()
    plt.bar( list(range(10)), lgr_clf.predict_proba(x_test_meta[idx:idx+1])[0])   
    plt.show()
    #generate bar graphs showing the predicted class distributions
    #plt.bar(x_test, height=display, width=0.8, bottom=None, align='center', data=None)
    #plt.figure(j+1)
    #plt.bar( list(range(10)), lgr_clf.predict_proba(x_test_meta[idx:idx+1])[0])    
    #plt.figure(j+1)
    
#plt.show()


# In[ ]:




