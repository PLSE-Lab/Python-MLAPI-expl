#!/usr/bin/env python
# coding: utf-8

# ## Credit card fraud detection
# ### In this book we will be seeing how to do over sampling of the unbalanced data without adding new copies of the minority class

# In[57]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ### Loading the Dataset

# In[58]:


data = pd.read_csv("../input/creditcard.csv")
data.head()


# ### Counting the samples/class

# In[59]:


count_classes = pd.value_counts(data['Class'], sort = True).sort_index()
count_classes


# ### Clearly the data is unbalanced
# #### Thus Accuracy can't be used to evaluate any classification algorithm. 
# 
# #### There are several ways to approach this classification problem taking into consideration this unbalance.
# * Use the confusion matrix to calculate Precision, Recall. 
# * ROC curves - calculates sensitivity/specificity ratio.
# * Resampling the dataset
#  * Essentially this is a method that will process the data to have an approximate 50-50 ratio.
#  * One way to achieve this is by OVER-sampling, which is adding copies of the under-represented class (better when you have
#    little data)
#  * Another is UNDER-sampling, which deletes instances from the over-represented class (better when he have lot's of data).

# ### Proposed Approach
# 1. Recall is used for transaction systems so we'll be consindering recall score.
# 2. We'll do over sampling to add minority class copies. We're just going to replicate the minority class for a good model fitting. This technique works if the minority class have good examples. A nice approach is to find the good examples and replicating them for oversampling. Another approach is to generate the examples using the found good examples. We'll use the 2nd and 3rd approach in the later versions.
# 3. The way we will under sample the dataset will be by creating a 50/50 ratio. This will be done by randomly selecting "x" amount of sample from the majority class, being "x" the total number of records with the minority class(after oversampling).

# ### Setting input and target variables + resampling.

# In[60]:


X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']


# In[61]:


# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)
# OverSampling
fraud_indices1 = np.array(data[data.Class == 1].index)
fraud_indices2 = np.array(data[data.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, 3*number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,fraud_indices1,fraud_indices2,random_normal_indices])

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices,:]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ", len(under_sample_data[under_sample_data.Class == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ", len(under_sample_data[under_sample_data.Class == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))


# ### Splitting data into train and test set.

# In[62]:


from sklearn.cross_validation import train_test_split

# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 0)

print("Original train dataset: ", len(X_train))
print("Original test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))

# Undersampled dataset
X_train_undersample, X_test_undersample, y_train_undersample, y_test_undersample = train_test_split(X_undersample,y_undersample,test_size = 0.3,random_state = 0)
print("")
print("Undersampled train dataset: ", len(X_train_undersample))
print("Undersampled test dataset: ", len(X_test_undersample))
print("Total number of transactions: ", len(X_train_undersample)+len(X_test_undersample))  


# ### Logistic regression classifier 
# #### As mentioned above we are interested in the recall score, because this is the metric that captures the most fraudulent transactions. 
# * Accuracy = (TP+TN)/total
# * Precision = TP/(TP+FP)
# * Recall = TP/(TP+FN)
#  
#  As we know, due to the imbalacing of the data, many observations could be predicted as False Negatives, being, that we predict a normal transaction, but it is in fact a fraudulent one. Recall captures this.

# In[63]:


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 


# ### A function to plot a fancy confusion matrix

# In[64]:


import itertools

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# ### Predictions on test set using unbalnced data and plotting confusion matrix

# In[76]:


lr = LogisticRegression(C = 1, penalty = 'l1')
lr.fit(X_train,y_train.values.ravel())
y_pred = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

print("Recall metric of the unbalanced dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()


# ### Recall for unbalanced data is 0.619.
# ### Let's use Sampled data for training and use this trained model against original test data

# In[78]:


lr = LogisticRegression(C = 1, penalty = 'l1')
lr.fit(X_train_undersample,y_train_undersample.values.ravel())
y_pred = lr.predict(X_test.values)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test,y_pred)
np.set_printoptions(precision=2)

print("Recall using sampled training dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')
plt.show()


# ### Surprisingly Recall has been increased to 0.932.

# In[79]:


# ROC CURVE
lr = LogisticRegression(C = 1, penalty = 'l1')
y_pred_undersample_score = lr.fit(X_train_undersample,y_train_undersample.values.ravel()).decision_function(X_test_undersample.values)

fpr, tpr, thresholds = roc_curve(y_test_undersample.values.ravel(),y_pred_undersample_score)
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### We obtain "Area Under Curve=0.99".
