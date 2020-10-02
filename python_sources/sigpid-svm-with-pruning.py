#!/usr/bin/env python
# coding: utf-8

# load tables:

# In[ ]:


import numpy as np # linear algebra
M = np.load('/kaggle/input/permissions-table/M.npy')
B = np.load('/kaggle/input/permissions-table/B.npy')
permissions_table = np.load('/kaggle/input/permissions-table/permissions_table.npy')


# define a model creating and evaluating funciton, we will use it later:

# In[ ]:


def create_and_evaluate_model(dataset):
    permisisons_labels = (np.sum(permissions_table,axis=1) > 10).astype(int) # as usuall all the manifests with more than 10 permisisons
    #required will be considered as maliciuos

    from sklearn.model_selection import train_test_split
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(dataset, permisisons_labels, test_size=0.3,random_state=109) # 70% training and 30% test

    #Import svm model
    from sklearn import svm
    #Create a svm Classifier
    clf = svm.SVC(kernel='linear') # Linear Kernel
    #Train the model using the training sets
    clf.fit(X_train, y_train)
    #Predict the response for test dataset
    y_pred = clf.predict(X_test)

    #Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy: how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive labels are correct?
    print("Precision:",metrics.precision_score(y_test, y_pred))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:",metrics.recall_score(y_test, y_pred))
    # Model f1 score: implemented using a formula on precision and recall
    print("F1 ScoRe:",metrics.f1_score(y_test, y_pred))
    fp = len(y_test[(y_test == 0) & (y_test == 1)])
    N = len(y_test[y_test == 0])
    # Model fall out: percent of false positive out of all negative in the data
    print("Fall out:",fp/N)


# <h1>BASIC</h1><br>
# first, try with all 121 features:

# In[ ]:


permissions_table.shape


# create the model:

# In[ ]:


create_and_evaluate_model(permissions_table)


# <h1>PRNR</h1><br>
# define the PRNR function:

# In[ ]:


def S_B(j):
    sigmaBij = np.sum(B,axis=0)[j]
    sizeBj = B.shape[0]
    sizeMj = M.shape[0]
    return (sigmaBij/sizeBj)*sizeMj

def PRNR(j):
    sigmaMij = np.sum(M,axis=0)[j]
    S_Bj = S_B(j)
    return (sigmaMij-S_Bj)/(sigmaMij+S_Bj)

def PRNR_absolute(j):
    return abs(PRNR(j))


# great, now lets filter out all the rankings which will not help us (their absolute PRNR ranking is below 0.5):

# In[ ]:


permissions_table = np.load('/kaggle/input/permissions-table/permissions_table.npy')
permissions_PRNR_ranking = np.array([PRNR_absolute(i) for i in np.arange(121)])
PRNR_reduced_permissions_table = permissions_table[ :, permissions_PRNR_ranking > 0.5]
print("number of features: " + str(PRNR_reduced_permissions_table.shape[1]))


# and we are left with 113 features. next, build a model and see how well it goes:

# In[ ]:


create_and_evaluate_model(PRNR_reduced_permissions_table)


# did just as well as the last model but with less features!

# <h1>PRNR + SPR</h1><br>
# First we look at the number of features which are used by different samples:

# In[ ]:



print("number of features used by 1 samples or more: "+str(sum(np.sum(PRNR_reduced_permissions_table,axis=0) >= 1)))
print("number of features used by 2 samples or more: "+str(sum(np.sum(PRNR_reduced_permissions_table,axis=0) >= 2)))
print("number of features used by 3 samples or more: "+str(sum(np.sum(PRNR_reduced_permissions_table,axis=0) >= 3)))
print("number of features used by 4 samples or more: "+str(sum(np.sum(PRNR_reduced_permissions_table,axis=0) >= 4)))
print("number of features used by 5 samples or more: "+str(sum(np.sum(PRNR_reduced_permissions_table,axis=0) >= 5)))


# so we will take all the features used by 2 or more samples.<br>
# here we take down all the permisisons which are used only in 1 manifest:

# In[ ]:


PRNR_SPR_reduced_permissions_table = PRNR_reduced_permissions_table[:,np.sum(PRNR_reduced_permissions_table,axis=0) > 1] # sum all the
#column thus give us the number of manifests using this feature. then we take only greater than 1



print("number of features: " + str(PRNR_SPR_reduced_permissions_table.shape[1]))


# lets make a model and test it again, this time with only 38 features:

# In[ ]:


create_and_evaluate_model(PRNR_SPR_reduced_permissions_table)


# with better accuracy, we can suggest that the other features just confused the model.

# <h1>PRNR + SPR + PMAR</h1><br>
# here we take down permisions which will always appear together (we dont take them both down, we leave just 1 of them):

# In[ ]:


#permissions which will always be together will yeild identical columns. thus we just need to remove identical columns

def unique_columns2(data): # and thank you stack overflow!
    dt = np.dtype((np.void, data.dtype.itemsize * data.shape[0]))
    dataf = np.asfortranarray(data).view(dt)
    u,uind = np.unique(dataf, return_inverse=True)
    u = u.view(data.dtype).reshape(-1,data.shape[0]).T
    return (u,uind)

PRNR_SPR_PMAR_reduced_permissions_table = unique_columns2(PRNR_SPR_reduced_permissions_table)[0]
print("number of features: " + str(PRNR_SPR_PMAR_reduced_permissions_table.shape[1]))


# from 121 features to only 34 featuers, lets check how the model will do:

# In[ ]:


create_and_evaluate_model(PRNR_SPR_PMAR_reduced_permissions_table)


# same as the last one

# <h1>Dangerous permissions</h1><br>
# Google has announced 24 dangerous permissions, we will use them to compare this tactic againt out premission reduction tactic<br>
# here we dont have 24 permissions because we dont have much apks so we dont get all of the permissions:

# In[ ]:


dangerous_permissions_labels = [12,13,16,17,23,24,43,44,59,65,55,71,82,83,92]
dangerous_permissions = permissions_table[:, dangerous_permissions_labels]
print("number of features: " + str(dangerous_permissions.shape[1]))


# with only 15 permissions, will it do as the other models?

# In[ ]:


create_and_evaluate_model(dangerous_permissions)


# not bad at all, but we did better!
