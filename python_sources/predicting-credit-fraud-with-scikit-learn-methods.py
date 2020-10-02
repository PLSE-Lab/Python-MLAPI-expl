#!/usr/bin/env python
# coding: utf-8

# **This code tests out a number of machine learning models from scikit-learn on this dataset**

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import warnings
from tqdm import *

warnings.filterwarnings("ignore",category=DeprecationWarning)
data=pd.read_csv('../input/creditcard.csv')
data.head()
target=data['Class']
data.drop('Class',axis=1,inplace=True)
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size = 0.50)


# In[ ]:


models = {
    "Decision Tree Classifier" : DecisionTreeClassifier(),
    "Stochastic Gradient Descent Classifier" : SGDClassifier(),
    "Passive Aggressive Classifier" : PassiveAggressiveClassifier(),
    "Perceptron" : Perceptron(),
    "KNN Classifier" : KNeighborsClassifier(),
    "KNN Regression" : KNeighborsRegressor()
}


# Due to the nature of this dataset, where there are many non-fraudulent transactions and only a few fraudulent transactions (only 0.0617% of observations are fraudulent transactions), the scikit-learn *score* method is ineffective for rating the true effectiveness of the methods.
# 
# **trueScore** is the ratio of true positives to the total number of fraudulent transactions. This is much more effective at measuring model effectiveness.

# This is the list of models I tested

# In[ ]:


def trueScore(model): #Measures how effectively model detects sparse credit fraud
    total = 0
    correct = 0
    predictions = model.predict(x_test)
    for i in range(len(predictions)):
        if np.array(y_test)[i] == 1:
            total += 1
            if predictions[i] == 1:
                correct += 1
    return correct/total


# This code goes through each model, fits it to the data, and scores it using both the scikit-learn method and **trueScore**:

# In[ ]:


for m in models:
    models[m].fit(np.array(x_train), np.array(y_train))
    score = models[m].score(x_test, y_test)
    true_score = trueScore(models[m])
    #scores.append(score)
    #trueScores.append(true_score)
    print(m, "\n   score: ", score)
    print("\n    true score: ", true_score,"\n")


# The results show that while many models have over 99% efficiency, nearly all of them are ineffective at detecting credit card fraud. But the decision tree had an incredible efficiency of approximately 75%. I attribute this to the procedure that the decision tree uses to detect frauds, which categorizes observations into a number of buckets, many of which purely contain frauds. This procedure benefits from its specificity and is far more effective than other, more generalist, machine learning approaches
# 
# A visualization of the decision tree is shown below:

# In[ ]:


from sklearn.externals.six import StringIO  
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os


plt.figure(figsize=(150,200))
export_graphviz(models["Decision Tree Classifier"], out_file = 'out.dot')
os.system('dot -Tpng out.dot -o out.png')

img = mpimg.imread('out.png')

imgplot = plt.imshow(img, aspect='equal')
plt.show(block=True)

