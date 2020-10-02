#!/usr/bin/env python
# coding: utf-8

# # MNIST classification
# In this notebook I tried to train and optimize a range of models on the MNIST dataset. The final result will be a stacking classifier of SVM, RF and XGB. I used other people's work and combined this into a more complete analysis. I am a Python beginner (more used to R) so bare with me..

# ### Sources (among others)
# - https://jakevdp.github.io/PythonDataScienceHandbook/05.08-random-forests.html
# - https://www.kaggle.com/archaeocharlie/a-beginner-s-approach-to-classification
# 

# # Basics

# ### Imports

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import seaborn as sns 
import time
#from scipy import stats

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Global variables

# In[ ]:


seed = 2

train_size = 0.8
test_size = 1- train_size

labeled_images = pd.read_csv('../input/train.csv')
test_data=pd.read_csv('../input/test.csv') #The kaggle test set (to get leaderboard score). For kaggle kernel add: ../input/
sample_size = len(labeled_images.index) #total number of instances for training and testing


# ### Functions

# In[ ]:


def plot_as_grid(images, labels, m):
    "Plot N x M grid of digits and labels/predictions"
    n_pixels = len(images.columns)
    dimension = int(np.sqrt(n_pixels))

    # set up the figure
    fig = plt.figure(figsize=(6, 6))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    # plot the digits: each image is max mxm pixels
    for i in range( min(m*m, len(images.index))):
        ax = fig.add_subplot(m, m, i + 1, xticks=[], yticks=[])    
    
        img=images.iloc[i].values.reshape((dimension,dimension))
    
        ax.imshow(img, cmap=plt.cm.binary, interpolation='nearest')
        ax.text(0, 7, str(labels.iloc[i,0]))


# In[ ]:


def param_selection(model, X, y, param_random, n_iter = 20, n_jobs = 2, n_folds = 3):
    "Determines the best parameters using randomSearch (slow)"
    search = RandomizedSearchCV(model, param_random, n_iter = n_iter, n_jobs = n_jobs ,
                                random_state = seed, cv=n_folds, iid=True) 
    
    search.fit(X, y)
    return search.best_params_


# In[ ]:


def inspect_performance(model, train_images, train_labels, test_images, test_labels, ypred):
    "Prints training performance, test performance and a performance report"
    print("Training error: ", model.score(train_images,train_labels))
    print("Test error: ", model.score(test_images,test_labels))
    print("Test report: ")
    print(metrics.classification_report(ypred, test_labels))


# In[ ]:


def plot_confusion_matrix(labels, predictions):
    "Plots a confusion matrix using a heatmap"
    mat = confusion_matrix(labels, predictions)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')


# In[ ]:


def plot_incorrect_classifications(ypred, test_labels, test_images):
    "Plots incorrectly classified images and corresponding prediction"
    ypred = pd.DataFrame(ypred)
    ypred = ypred.set_index(test_labels.index.values)
    ypred.columns = ['prediction']
    predict_df = pd.concat([ypred, test_labels], axis=1)
    predict_df['Incorrect'] = predict_df.prediction != predict_df.label
    idx = predict_df.index[predict_df['Incorrect']]

    plot_as_grid(test_images.loc[idx], predict_df['prediction'].loc[idx].to_frame(), 5)


# In[ ]:


def make_submission_file(model, data, name):
    "Makes a submission file by predicting on test_data and returning the predictions"
    data[data>0]=1
    results=model.predict(data) 

    #Set to proper format:
    df = pd.DataFrame(results)
    df.index.name='ImageId'
    df.index+=1
    df.columns=['Label']
    df.to_csv(name + '.csv', header=True)
    return(results)


# ### Pre-processing

# In[ ]:


labeled_images_sample = labeled_images.sample(n=sample_size, random_state= seed)
images = labeled_images_sample.iloc[:,1:] # select images (all columns except the first)

max_value = images.values.max()
images /= max_value #scale

labels = labeled_images_sample.iloc[:,:1] # select labels (only first column)

train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=train_size, 
                                                                       test_size=test_size, random_state=seed)


# ### Visualisation(s)

# In[ ]:


plot_as_grid(train_images, train_labels, 8) #digit and label


# # Model 1: SVM
# According to following steps:
# - train model
# - inspect performance
# - error analysis
# - write to .csv
# 
# **Note**: There is no parameter tuning for SVM because it is typically a slow algorithm. It takes too long to train. Also, I expect SVM not to be the best model for digit recognition so I won't spend too much time on it. 

# ### Train model

# In[ ]:


clf = svm.SVC(gamma='auto', C=1.0, kernel='rbf')
clf.fit(train_images, train_labels.values.ravel())
ypred_svm = clf.predict(test_images) 


# ### Inspect performance

# In[ ]:


#takes long, for now commented out
#inspect_performance(clf, train_images, train_labels, test_images, test_labels, ypred_svm)    


# ### Error analysis

# In[ ]:


plot_confusion_matrix(test_labels, ypred_svm)


# In[ ]:


plot_incorrect_classifications(ypred_svm, test_labels, test_images) #digit and classification  


# ### Write to .csv

# In[ ]:


ypred_svm_test = make_submission_file(clf, test_data, "results_svm")


# # Model 2: Random Forest
# According to following steps:
# - train model with optimal parameters
# - inspect performance
# - error analysis
# - write to .csv

# ### Train model with optimal parameters

# In[ ]:


start_time = time.clock()
param_random = {"n_estimators": [400, 700], # you can try multiple options, as long as you update n_iter accordingly:  "n_estimators": [400, 500, 600, 700], then n_iter should be 4*2
                'max_features' : ['auto', 'log2']} 
parameters = param_selection(RandomForestClassifier(), train_images, train_labels.values.ravel(), param_random, n_iter = 8, n_folds = 3 ) 
print(parameters)
print (time.clock() - start_time, "seconds")


# In[ ]:


rf = RandomForestClassifier(n_estimators = parameters['n_estimators'], max_features = parameters['max_features'], 
                            random_state=seed,n_jobs=2)
rf.fit(train_images, train_labels.values.ravel())
ypred_rf = rf.predict(test_images) 


# ### Inspect performance

# In[ ]:


inspect_performance(rf, train_images, train_labels, test_images, test_labels, ypred_rf)    


# ### Error analysis

# In[ ]:


plot_confusion_matrix(test_labels, ypred_rf)


# In[ ]:


plot_incorrect_classifications(ypred_rf, test_labels, test_images) #digit and classification    


# ### Write to .csv

# In[ ]:


ypred_rf_test = make_submission_file(rf, test_data, "results_rf")


# # Model 3: Xtreme Gradient Boosting
# According to following steps:
# - train model with optimal parameters
# - inspect performance
# - error analysis
# - write to .csv

# ### Train model with optimal parameters

# In[ ]:


start_time = time.clock()
param_random = {"eta": [0.3], # you can try multiple options, as long as you update n_iter accordingly:  "eta": [0.01, 0.02, 0.05, 0.1, 0.25, 0.5], then n_iter should be 6
                "early_stopping_rounds": [50],
                "n_estimators" : [100],
                "eval_metric" : ["merror"]}
parameters = param_selection(XGBClassifier(), train_images, train_labels.values.ravel(), param_random, n_iter = 1) 
print(parameters)
print (time.clock() - start_time, "seconds")


# In[ ]:


xgb = XGBClassifier(eta = parameters["eta"], early_stopping_rounds = parameters["early_stopping_rounds"], 
                    n_estimators = parameters["n_estimators"],eval_metric = parameters["eval_metric"],n_jobs=2)
xgb.fit(train_images,train_labels.values.ravel())
ypred_xgb = xgb.predict(test_images)  


# ### Inspect performance

# In[ ]:


inspect_performance(xgb, train_images, train_labels, test_images, test_labels, ypred_xgb)    


# ### Error analysis

# In[ ]:


plot_confusion_matrix(test_labels, ypred_xgb)


# In[ ]:


plot_incorrect_classifications(ypred_rf, test_labels, test_images) #digit and classification


# ### Write to .csv

# In[ ]:


ypred_xgb_test = make_submission_file(rf, test_data, "results_xgb")


# # Model 4: Stacking classifier
# Stack SVM, RF and XGB together in a stacking classifier. Parameter tuning turned out to be a computational nightmare so I will try this approach instead.

# In[ ]:


stack_train = pd.concat([pd.DataFrame(ypred_svm) , pd.DataFrame(ypred_rf), pd.DataFrame(ypred_xgb)], axis=1)
stack_test = pd.concat([pd.DataFrame(ypred_svm_test) , pd.DataFrame(ypred_rf_test), pd.DataFrame(ypred_xgb_test)], axis=1)

lr = LogisticRegression(solver ='newton-cg', multi_class="auto")
lr.fit(stack_train,test_labels.values.ravel())
ypred_lr = lr.predict(stack_test)  

results=ypred_lr
name = "results_stacking"
#Set to proper format:
df = pd.DataFrame(results)
df.index.name='ImageId'
df.index+=1
df.columns=['Label']
df.to_csv(name + '.csv', header=True)


# These different models can be submitted. Please let me know if this notebook was helpful and/or if you have feedback.
