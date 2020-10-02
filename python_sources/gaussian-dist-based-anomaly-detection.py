#!/usr/bin/env python
# coding: utf-8

# Simple Multivariate gaussian distribution assumption is used to predict the probability that a given transaction is fraudulent. 
# 
#  - The raw data has ~250,000 normal transactions and ~500 fraudulent transactions
#  - This is highly imbalanced data but such is the nature of credit card data. 
#  - So regular  ML techniques like random forests and logistic regression on raw data may give high accuracy but will be not very useful in detecting the fraud transactions. Because predicting every transaction as not fraudulent still gives an accuracy of 250000/250500 i.e. 99.8% accuracy.
#  - One way to deal with this problem is resampling from the raw data to get a balanced sample and then run logistic regression or random forests
#  - Another alternative is that we can use anomaly detection algorithms. 
# 
# In this notebook I apply the simple multivariate gaussian distribution assumption to calculate the probability that a given transaction is fraudulent.
# 
# First I create a training data which contains only normal transactions. I split the transactions into normal and fraudulent transactions. 
# Train Data:   60% of the the normal transactions
# Cross Validation Data: 20% of normal transactions and 50% of the fraudulent transactions
# Test Data: 20% of normal transactions and 50% of fraudulent transactions
# 
# All the features are assumed to be independent of each other and are assumed to follow normal distribution. From the training data I calibrate the mean and variance of each feature. With this I calculate the probability of each transaction feature having a given value in the cross validation data. Now from the cross validation probabilities, I come up with an epsilon probability which can be used to classify the transaction. 
# 
# Increasing epsilon decreases false positives (normal but classified as fraud) and also the true positives (fraud classified as fraud). Using a very small epsilon increases the true positive rate but also dramatically increases the false positive rate. It is important to strike a balance here

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


rawData = pd.read_csv('../input/creditcard.csv')


# In[ ]:


rawData.head()


# In[ ]:


normals = rawData[rawData['Class']==0]
anamolies = rawData[rawData['Class']==1]


# In[ ]:


print('There are',len(normals),'normal transactions and ',len(anamolies), 'fraudulent transactions in the data')


# In[ ]:


rawData['Amount'].describe()


# In[ ]:


split = np.random.choice([1,2,3], len(normals), p=[0.6, 0.2, 0.2])
normals['split'] = split


# In[ ]:


trainData = normals[normals['split']==1]
crossVal = normals[normals['split']==2]
testData = normals[normals['split']==3]


# In[ ]:


split2 = np.random.choice([1,2], len(anamolies), p=[0.5, 0.5])
anamolies['split'] = split2


# In[ ]:


crossVal = crossVal.append(anamolies[anamolies['split']==1])
testData = testData.append(anamolies[anamolies['split']==2])


# In[ ]:


print('Length of train data', len(trainData))
print('Length of crossval data', len(crossVal))
print('Length of test data', len(testData))


# In[ ]:


np.log(10)


# In[ ]:


cols = ['V'+str(i) for i in range(1,29)]
cols.append('Amount')


# In[ ]:


featureMeans = {}
featureVars = {}


# In[ ]:


for col in cols:
    featureMeans[col] = np.mean(trainData[col])
    featureVars[col] = np.var(trainData[col])


# In[ ]:


featureMeans


# In[ ]:


featureVars


# In[ ]:


crossVal['prediction'] = ''


# In[ ]:


testData.head()


# In[ ]:


testData = testData.reset_index()
crossVal= crossVal.reset_index()


# In[ ]:


for i in range(len(crossVal)):
    
    p = 10
    
    for j in cols:
        
        p = p* (1/np.sqrt((2*np.pi)*featureVars[j]))* np.exp(-1*((crossVal.loc[i,j]-featureMeans[j])**2)/(2*featureVars[j]))
    
    crossVal.loc[i,'prediction'] = p


# In[ ]:


crossValNormals = crossVal[crossVal['Class']==0]
crossValAnams = crossVal[crossVal['Class']==1]


# In[ ]:


crossVal['classPredict'] = ''


# In[ ]:


classEpsArray = np.array([(10**(-1*i)) for i in range(10,50) ])


# In[ ]:


def classify(predictionP,eps):
    
    if predictionP < eps:
        return 1
    else:
        return 0
    


# In[ ]:


tprArray = []
fprArray = []
dtArray =[]

for classEps in classEpsArray:
    
    crossVal['classPredict'] = crossVal['prediction'].apply(lambda row: classify(row,classEps))
    effTable = crossVal[['classPredict','Class']]
    fp = len(effTable[(effTable['classPredict']==1)&(effTable['Class']==0)])
    fn = len(effTable[(effTable['classPredict']==0)&(effTable['Class']==1)])
    tp = len(effTable[(effTable['classPredict']==1)&(effTable['Class']==1)])
    tn = len(effTable[(effTable['classPredict']==0)&(effTable['Class']==0)])
    
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    
    detectRate = tp/(tp+fn)
    
    
    
    
    tprArray.append(tpr)
    fprArray.append(fpr)
    dtArray.append(detectRate)
    
        


# In[ ]:


plt.figure()
plt.plot(dtArray,label='Fraud detection Rate')
plt.plot(fprArray,label='Flase Positive Rate')
plt.legend()
plt.show()


# In[ ]:


From the above plot of fraud detection rate and flase positive rate we can see that epsilon parameter of 


# In[ ]:


plt.figure()
plt.plot(fprArray,tprArray)
plt.tight_layout()
plt.ylabel('True positive rate (TPR)')
plt.xlabel('False positive rate (FPR)')

axes = plt.gca()
axes.set_xlim([0,1])
axes.set_ylim([0,1])


# In[ ]:


classEps2 = 1.00000000e-24
crossVal['classPredict'] = crossVal['prediction'].apply(lambda row: classify(row,classEps2))
effTable = crossVal[['classPredict','Class']]


# In[ ]:


fp = len(effTable[(effTable['classPredict']==1)&(effTable['Class']==0)])
fn = len(effTable[(effTable['classPredict']==0)&(effTable['Class']==1)])
tp = len(effTable[(effTable['classPredict']==1)&(effTable['Class']==1)])
tn = len(effTable[(effTable['classPredict']==0)&(effTable['Class']==0)])


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[ ]:


from sklearn.metrics import confusion_matrix
cnf_matrix= confusion_matrix(crossVal['Class'], crossVal['classPredict'])


# In[ ]:


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['normal','anamoly'],
                      title='Confusion matrix, without normalization')


# In[ ]:


2816/(2816+53423)


# From above confusion matrix we can see that the algorithm was able to detect 223 of 250 fraudulent transactions. THat is ~89.2% of the fraud. The false positive rate is 5%

# 

# In[ ]:


testData['prediction'] = ''


# In[ ]:


for i in range(len(testData)):
    
    p = 10
    
    for j in cols:
        
        p = p* (1/np.sqrt((2*np.pi)*featureVars[j]))* np.exp(-1*((testData.loc[i,j]-featureMeans[j])**2)/(2*featureVars[j]))
    
    testData.loc[i,'prediction'] = p


# In[ ]:


classEps2 = 1.00000000e-24

testData['classPredict'] = testData['prediction'].apply(lambda row: classify(row,classEps2))
effTable2 = testData[['classPredict','Class']]

cnf_matrix2 = confusion_matrix(testData['Class'], testData['classPredict'])

plt.figure()
plot_confusion_matrix(cnf_matrix2, classes=['normal','anamoly'],
                      title='Confusion matrix, without normalization')


# On the test set the detection rate is ~ 86%

# In[ ]:




