#!/usr/bin/env python
# coding: utf-8

# Lets import all the libraries we need

# In[ ]:


import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold,cross_val_score
from sklearn.metrics import accuracy_score,roc_curve,confusion_matrix,precision_recall_curve,auc,roc_auc_score,recall_score,classification_report
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict
from sklearn import svm
import random


# Lets read the data

# In[ ]:


random_seed = 72
random.seed(random_seed)
np.random.seed(random_seed)

df = pd.read_csv("../input/aps_failure_training_set.csv")
df_test = pd.read_csv("../input/aps_failure_test_set.csv")


# In[ ]:


df.head()


# Lets rename the target column to Flag and map neg to 0 and pos to 1 . Also map the na values to NULL. 

# In[ ]:


df = df.rename(columns = {'class' : 'Flag'})
df['Flag'] = df.Flag.map({'neg':0, 'pos':1})
df = df.replace(['na'],[np.NaN])


# Lets check how many columns have NAs

# In[ ]:


df.isnull().any()


# All most all the columns have NULL values. Lets check the distribution of our target variable

# In[ ]:


Count = pd.value_counts(df['Flag'], sort = True).sort_index()
Count.plot(kind = 'bar')
plt.title("Class count")
plt.xlabel("Flag")
plt.ylabel("Frequency")


# This is highly imbalanced data set. We will have to do some sampling technique here before modelling. Lets check total number of columns in our dataset

# In[ ]:


len(df.columns)


# There are 171 columns. 
# 
# I am imputing the NA values in our data with the mean value since all the fields are numeric ones 
# 
# Since i have already created a model with whole data and its runtime were on a little high side,I am going to do a PCA to the dataset to imporve its performance, 
# 
# Here i am using a mean normalization before PCA and Choosing the components which explains 95% of the variance in the data. 

# In[ ]:


df_X = df.loc[:,df.columns != 'Flag']
df_Y = df.loc[:,df.columns == 'Flag']

df_X = df_X.apply(pd.to_numeric)

df_X= df_X.fillna(df_X.mean()).dropna(axis =1 , how ='all')

scaler = StandardScaler()

scaler.fit(df_X)

df_X = scaler.transform(df_X)

pca = PCA(0.95)

pca.fit(df_X)

pca.n_components_


# We have reduced number of columns from 171 to 82 here which explains 95 % of the data. Now lets transform our data.

# In[ ]:


df_X = pca.transform(df_X)

df_X= pd.DataFrame(df_X)


# Lets do all the same process we did to our training on the test data too. 

# In[ ]:


df_test = df_test.rename(columns = {'class' : 'Flag'})
df_test = df_test.replace(['na'],[np.NaN])

Count = pd.value_counts(df_test['Flag'], sort = True).sort_index()
Count.plot(kind = 'bar')
plt.title("Class count")
plt.xlabel("Flag")
plt.ylabel("Frequency")

df_test['Flag'] = df_test.Flag.map({'neg':0, 'pos':1})

df_test_X = df_test.loc[:,df_test.columns != 'Flag']
df_test_Y = df_test.loc[:,df_test.columns == 'Flag']

df_test_X = df_test_X.apply(pd.to_numeric)

df_test_X= df_test_X.fillna(df_test_X.mean()).dropna(axis =1 , how ='all')

scaler = StandardScaler()

scaler.fit(df_test_X)

df_test_X = scaler.transform(df_test_X)

pca = PCA(82)

pca.fit(df_test_X)

pca.n_components_

df_test_X = pca.transform(df_test_X)

df_test_X= pd.DataFrame(df_test_X)


# Now we have test and training set.  Lets create a validation set to tune our parameters for the model. I am splitting our training set here to 80% for training and 20% for validation since we have a separate test data. 

# In[ ]:


X_train,X_validation,Y_train,Y_validation = train_test_split(df_X,df_Y,test_size = 0.2,random_state = 0)
DF = pd.concat([X_train,Y_train],axis = 1)

print("Percentage Neg in training: " , len(Y_train[Y_train.Flag == 0])/len(Y_train))
print("Percentage Pos in training: ", len(Y_train[Y_train.Flag == 1])/len(Y_train))
print("Total number of datapoints in training: ", len(Y_train))


print("Percentage Neg in Validation: " , len(Y_validation[Y_validation.Flag == 0])/len(Y_validation))
print("Percentage Pos in Validation: ", len(Y_validation[Y_validation.Flag == 1])/len(Y_validation))
print("Total number of datapoints in Validation: ", len(Y_validation))


# Now we have training,validation and test data sets. Since our dataset is highly imbalanced. lets do some sampling. I am going with undersampling here you can go with other sampling techniques like oversampling,SMOTE etc.

# In[ ]:


numberofrecords_pos = len(DF[DF.Flag == 1])
pos_indices = np.array(DF[DF.Flag == 1].index)

#Picking the indices of the normal class
neg_indices = DF[DF.Flag == 0].index

#out of indices selected, randomly select "x" number of records
random_neg_indices = np.random.choice(neg_indices, numberofrecords_pos, replace = False)
random_neg_indices =np.array(random_neg_indices)

#Appending the two indices
under_sample_indices = np.concatenate([pos_indices,random_neg_indices])

#Undersample dataset
under_sample_data = DF.loc[under_sample_indices,:]

X_undersample = under_sample_data.loc[:,under_sample_data.columns != 'Flag']
Y_undersample = under_sample_data.loc[:,under_sample_data.columns == 'Flag']

print("Percentage Neg: " , len(under_sample_data[under_sample_data.Flag == 0])/len(under_sample_data))
print("Percentage Pos : ", len(under_sample_data[under_sample_data.Flag == 1])/len(under_sample_data))
print("Total number of datapoints : ", len(under_sample_data))


# Now we have undersampled data. Lets try out some models. 
# 
# **Logistic Regression**
# 
# Here  we need to create a model which reduces the misclassification based on Cost 1 (10) and Cost 2 (500) and since Cost 2s multiplication factor is higher than Cost 1s i am going with recall as the performance metric here which basically gives the measure of how many positives cases did we catch(TP/TP+FN). Higher recall means lesser FNs which in turn reduces our total cost.
# 
# For Logistic regression i am here tuning only C_parameter and which regularization(L1 or L2) using the recall metric for the validation dataset. 
# 
# Lets see how it works out.

# In[ ]:


c_parameter_range = [0.0001,0.001,0.01,0.1,1,10,100]
penalty = ['l1','l2']
for penal in penalty:
    for c_param in c_parameter_range:
        
        print('------------------------')
        print("C Parameter :", c_param)
        print("Penalty: ", penal)
        print('------------------------')
        print('')
        lr = LogisticRegression(C = c_param, penalty = penal)
        lr.fit(X_undersample,Y_undersample.values.ravel())
        y_pred = lr.predict(X_validation)
        Recall = recall_score(Y_validation,y_pred)
        print ('Recall score for c param', c_param,'and penalty',penal,'=',Recall)
        print('-------------------------')
        print('')


# Here the recall score for C_Parameter = 0.001 and Penalty L2 is 0.9782 which is the highest. Lets fit the model using these and test it on our test and plot a confusion matrix. 

# In[ ]:


lr = LogisticRegression(C =0.001,penalty = 'l2')

lr.fit(X_undersample,Y_undersample.values.ravel())

y_pred = lr.predict(df_test_X)

recall_score(df_test_Y,y_pred)


# We have  a recall score of 0.978 which is quite good now lets check the confusion matrix and calculate the total cost of our model

# In[ ]:


confusion_matrix(df_test_Y,y_pred)


# So we have 8 FNs and 1296 FPs . So the total cost of our model is (8x500)+(1296x10) = 16,960 which is good. :)

# Now we can try other models. Lets check 
# 
# 
# **Random Forest**
# 
# Here i am tuning only number of trees and number of features going into each tree. 
# 
# As explained above here i am going with Recall score to tune the parameters. You can go with OOB score too.
# 
# Lets see how the model performs
# 

# In[ ]:


RANDOM_STATE = 123

import warnings
warnings.filterwarnings("ignore")
ensemble_clfs = [
    ("RandomForestClassifier, max_features='sqrt'",
        RandomForestClassifier(warm_start=True, oob_score=True,
                               max_features="sqrt",
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features='log2'",
        RandomForestClassifier(warm_start=True, max_features='log2',
                               oob_score=True,
                               random_state=RANDOM_STATE)),
    ("RandomForestClassifier, max_features=None",
        RandomForestClassifier(warm_start=True, max_features=None,
                               oob_score=True,
                               random_state=RANDOM_STATE))
]

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

# Range of `n_estimators` values to explore.
min_estimators = 15
max_estimators = 150

for label, clf in ensemble_clfs:
    for i in range(min_estimators, max_estimators + 1):
        clf.set_params(n_estimators=i)
        clf.fit(X_undersample, Y_undersample)

        # Record the OOB error for each `n_estimators=i` setting.
        y_pred = clf.predict(X_validation)
        recall = recall_score(Y_validation,y_pred)
        error = 1 - recall
        error_rate[label].append((i, error))

# Generate the "OOB error rate" vs. "n_estimators" plot.
for label, clf_err in error_rate.items():
    xs, ys = zip(*clf_err)
    plt.plot(xs, ys, label=label)

plt.xlim(min_estimators, max_estimators)
plt.xlabel("n_estimators")
plt.ylabel("Recall error rate")
plt.legend(loc="upper right")
plt.show()


# We can see that error is minimum when features are selected using Log and we have multiple options to choose number of trees for minimum error since from the graph we can see error rate is minimum for 25 trees..

#  Lets choose it and build a model

# I am choosing recall as the metric for evaluation. 

# In[ ]:



clf = RandomForestClassifier(n_estimators=25,max_features= 'log2',oob_score =True)

clf.fit(X_undersample,Y_undersample.values.ravel())

clf.oob_score_

y_pred = clf.predict(df_test_X)
recall_score(df_test_Y,y_pred)


# We have a good recall score lets check the confusion matrix and calculate the total cost. 

# In[ ]:


confusion_matrix(df_test_Y,y_pred)


# So we have 7 FNs and 1594 FPs . So the total cost of our model is (7x500)+(1594x10) = 19,440 which is greater than our logistic regression model.

# Now lets try one more model
# 
# **SVM**
# 
# Here i am tuning C_parameter,Gamma and which kernal to use. 
# 
# I tuned the gamma of the model using a grid search. Since it took some time to run the search i am not including it in the kernal. The value for gamma i got from grid search was 0.01
# 
# As explained above here i am going with Recall score to tune the other parameters.
# 
# Lets see how the model performs

# In[ ]:


c_parameter_range = [0.001,0.01,0.1,10,100]
kernel = ['linear','poly','rbf','sigmoid']
for kern in kernel:
    for c_param in c_parameter_range:
        print('------------------------')
        print("C Parameter :", c_param)
        print("Kernel: ", kern)
        print('------------------------')
        print('')
        clf = svm.SVC(C = c_param,kernel = kern,gamma = 0.01)
        clf.fit(X_undersample,Y_undersample)
        y_pred = clf.predict(X_validation)
        Recall = recall_score(Y_validation,y_pred)
        print ('Recall Score for c parameter', c_param, 'and kernel',kern,'=',Recall)
        print('-------------------------')
        print('')


# From the above results you can see that for the C_parameter 0.01 and for the Sigmoid kernal we got the maximum recall score in validation set. Now lets fit the model using these parameters and check the recall score in test set.

# In[ ]:


clf = svm.SVC(C =0.01,gamma = 0.01, kernel = 'sigmoid')

clf.fit(X_undersample,Y_undersample)
y_pred = clf.predict(df_test_X)
recall_score(df_test_Y,y_pred)


# We have a recall score of 0.98 which is pretty good. Now lets build a confusion matrix and check the total cost for the model

# In[ ]:


confusion_matrix(df_test_Y,y_pred)


# So we have 7 FNs and 2706 FPs . So the total cost of our model is (7x500)+(2706x10) = 30,560 which is greater than the other two models

# Now the models i created here are in a learning perspective  and build using very basic cleansing and parameter tunings. If spend more time in building a model by proper data cleansing,tuning other parameters will be able to improve the models significantly.
# 
# Thank You
# 
# 
# 
# note: some parts of the code were referred from scikit learn library and kernel of joparga3 :)
#  
#  Thanks
#  -----------------------------
# 
# 
