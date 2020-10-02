#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn import metrics
from sklearn import ensemble
from sklearn import neighbors
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score as cvs
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_predict as cvp


# ### **Implementing the general-purpose utility classes which can be used by almost every major Machine Learning Project with just few minor tweaks in its implementation or directly out-of-the-box.**

# In[ ]:


class Shuffle():
    def perform(self, dataFrame):
        return dataFrame.reindex(np.random.permutation(dataFrame.index))
    
class Split():
    def __init__(self, test_size, labels, stratify=False, stratifyBy=None):
        self.test_size = test_size
        self.labels = labels
        self.stratify = stratify
        self.stratifyBy = stratifyBy
        
    def __get_tts_from_df(self, dataFrame, test_size, attrs, labels, stratifyBy=None):
        if(self.stratify):
            train, test = tts(dataFrame, test_size=test_size, stratify=stratifyBy, random_state=50)
        else:
            train, test = tts(dataFrame, test_size=test_size, random_state=50)
        Y_train = (train[labels]).values.ravel()
        Y_test = (test[labels]).values.ravel()
        return (train[attrs], test[attrs], Y_train, Y_test)
    
    def perform(self, dataFrame):
        attrs = dataFrame.drop(self.labels, axis=1).columns.values.tolist()
        if(self.stratify):
            return self.__get_tts_from_df(dataFrame, self.test_size, attrs, self.labels, dataFrame[self.stratifyBy])
        else:
            return self.__get_tts_from_df(dataFrame, self.test_size, attrs, self.labels)

class ScaleFeature():
    def perform(self, dataFrame, training=True):
        if(training):
            self.transformer = StandardScaler()
            self.transformer.fit(dataFrame)   
        npOutput = self.transformer.transform(dataFrame)
        return npOutput

class PreProcess():
    def __init__(self):
        self.scaleFeature = ScaleFeature()

    def perform(self, dataFrame, training=True):
        npOutput = self.scaleFeature.perform(dataFrame, training)
        return npOutput

class ValidateModels():
    def __init__(self, split):
        self.split = split
        
    def perform(self, model, attrSet, labelSet, classDist='uniform'):
        if(classDist == 'uniform'):
            scores = cvs(model, attrSet, labelSet, scoring="accuracy", cv=self.split)
            print("Accuracy Estimate: MEAN +/- STD = " + str(np.round(scores.mean(),3)) + " +/- " + str(np.round(scores.std(),3)))
        elif(classDist == 'skewed'):
            preds = cvp(model, attrSet, labelSet, cv=self.split)
            print("Confusion Matrix: ")
            print(metrics.confusion_matrix(labelSet, preds))
            print("Classification report: ")
            print(metrics.classification_report(labelSet, preds))

class Test():
    def perform(self, model, attrSet, labelSet):
        predictions = model.predict(attrSet)
        print()
        print("*****==========*****")
        print("Accuracy Score: " + str(metrics.accuracy_score(labelSet, predictions)))
        print("*****==========*****")
        print("Confusion Matrix: ")
        print(metrics.confusion_matrix(labelSet, predictions))
        print("*****==========*****")
        print("Classification report: ")
        print(metrics.classification_report(labelSet, predictions))
        print("*****==========*****")
        print()
        return predictions


# ### **Importing the dataset and shuffling it to ensure randomized splitting.**

# In[ ]:


dataSet = pd.read_csv("../input/iris/Iris.csv", sep=",", usecols=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species'])
shuffle = Shuffle()
dataSet = shuffle.perform(dataSet)
print(dataSet.head())


# ### **Splitting the dataset into a 80-20 (Train-Test) Ratio, while ensuring similar distribution of the labels in the Test Set using Stratification to ensure un-skewed split.**

# In[ ]:


labels = ['Species']
split = Split(0.2, labels, True, labels)
X_train, X_test, Y_train, Y_test = split.perform(dataSet)


# In[ ]:


print(X_train)


# ### **Pre-Processing the train set in the form of Fearture Scaling.**

# In[ ]:


preProcess = PreProcess()
X_train = preProcess.perform(X_train)
print(X_train)


# In[ ]:


print(Y_train)


# ### **Initializing several basic models, performing hyperparamter tuning and analyzing its performance using Cross Validation.**

# In[ ]:


validateModels = ValidateModels(10)
lr_model = linear_model.LogisticRegression(random_state=50, n_jobs=-1, C=25, solver='saga', penalty='elasticnet', max_iter=1000, l1_ratio=0.75)
validateModels.perform(lr_model, X_train, Y_train)
dt_model = tree.DecisionTreeClassifier(random_state=50, criterion='gini', splitter='random', max_depth=3)
validateModels.perform(dt_model, X_train, Y_train)
svc_model = svm.SVC(random_state=50, probability=True, kernel='rbf', C=0.5)
validateModels.perform(svc_model, X_train, Y_train)
knn_model = neighbors.KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=10)
validateModels.perform(knn_model, X_train, Y_train)
rf_model = ensemble.RandomForestClassifier(random_state=50, n_jobs=-1, n_estimators=10, max_depth=3)
validateModels.perform(rf_model, X_train, Y_train)


# ### **Initialzing a Voting Classifier Ensemble Model, analyzing its performance and fitting it to the complete training dataset.**

# In[ ]:


vc_model = ensemble.VotingClassifier(estimators=[('lr', lr_model), ('dt', dt_model), ('svc', svc_model), ('knn', knn_model), ('rf', rf_model)], voting='soft', n_jobs=-1)
validateModels.perform(vc_model, X_train, Y_train)
vc_model = vc_model.fit(X_train, Y_train)


# ### **Pre-Processing the Testing Dataset and calculating the performance of the Voting Classifier Model on it.**

# In[ ]:


X_test = preProcess.perform(X_test, training=False)

test = Test()
preds = test.perform(vc_model, X_test, Y_test)


# ## ***The End. A beginner here on Kaggle, please Upvote and Comment on this Notebook if you found it useful, help me earn some medals. Thank You !!***
