#!/usr/bin/env python
# coding: utf-8

# # **Fraud Detection using Gaussian Normal Destribution**

# # ![](https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Normal_Distribution_PDF.svg/340px-Normal_Distribution_PDF.svg.png)

# # Import the needed libraries and show the files on the directory

# In[ ]:


import numpy as np 
import pandas as pd 
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Read the data file

# In[ ]:


df = pd.read_csv("../input/creditcardfraud/creditcard.csv")


# # Show the first 5 data rows

# In[ ]:


df.head()


# # show the Histogram for each columns

# In[ ]:


hist = df.hist(bins=100, figsize = (20,20))


# # Select the data with the best Gaussian distribution

# In[ ]:


df_distributed = df[["V11","V13","V15","V18","V19","V12","Class"]]


# # divide data into positive and nigative datasets

# In[ ]:


negative_df = df_distributed.loc[df_distributed['Class'] == 0]
positive_df = df_distributed.loc[df_distributed['Class'] == 1]


# # **prepare the data by dividing them into Training, Testing and Cross validation datasets**

#  the data are very sekwed as we have almost 450 anominal samples and more than 200000 good results
#  so we will divide the data as following
#  
#  90% of good data are training data to estimate Gaussians factors (mean, Standered diviation and variance)
#  
#  5% for CV dataset and 5% for testing dataset
#  
#  while 50% the anominal data will be added to CV dataset and the other 50% will be added to Testing dataset

# In[ ]:


from sklearn.model_selection import train_test_split

y_negative = negative_df["Class"]
y_positive = positive_df["Class"]
negative_df.drop(["Class"], axis=1, inplace=True)
positive_df.drop(["Class"], axis=1, inplace=True)

# 90% of good data are training data to estimate Gaussian factors (mean, Standard deviation and variance)
negative_df_training, negative_df_testing, y_negative_training, y_negative_testing = train_test_split(negative_df,
                                                                                                      y_negative,
                                                                                                      test_size=0.1,
                                                                                                      random_state=0)
# 5% for CV dataset, 5% for testing dataset
negative_df_cv, negative_df_testing, y_negative_cv, y_negative_testing = train_test_split(negative_df_testing,
                                                                                          y_negative_testing,
                                                                                          test_size=0.5,
                                                                                          random_state=0)

# while 50% the anomalies data will be added to CV dataset and the other 50% will be added to Testing dataset
positive_df_cv, positive_df_testing, y_positive_cv, y_positive_testing = train_test_split(positive_df,
                                                                                          y_positive,
                                                                                          test_size=0.5,
                                                                                          random_state=0)

df_cv = pd.concat([positive_df_cv, negative_df_cv], ignore_index=True)
df_cv_y = pd.concat([y_positive_cv, y_negative_cv], ignore_index=True)
df_test = pd.concat([positive_df_testing, negative_df_testing], ignore_index=True)
df_test_y = pd.concat([y_positive_testing, y_negative_testing], ignore_index=True)

y_negative_training = y_negative_training.values.reshape(y_negative_training.shape[0], 1)
df_cv_y = df_cv_y.values.reshape(df_cv_y.shape[0], 1)
df_test_y = df_test_y.values.reshape(df_test_y.shape[0], 1)


# # estimate Gaussian parameters (mean, Standered diviation and variance)

# In[ ]:


def estimateGaussian(X):
    stds=[]
    mean = []
    variance =[]
    
    mean = X.mean(axis=0)
    stds =X.std(axis=0)
    variance = stds **2
    
    stds = stds.values.reshape(stds.shape[0], 1)
    mean = mean.values.reshape(mean.shape[0], 1)
    variance = variance.values.reshape(variance.shape[0], 1)
    return stds,mean,variance


# # run the estimate Gaussians and return the results (mean, Standered diviation and variance)

# In[ ]:


stds,mean,variance = estimateGaussian(negative_df_training)


# In[ ]:


print(stds.shape)
print(stds.shape)
print(stds.shape)


# # Calculate the PROBABILITY for any new data CV or Testing using the factor that we have calculated and using the GAUSSIAN NORMAL DISTRIBUTION algorithm

# In[ ]:


def multivariateGaussian(stds, mean, variance, df_cv):
    probability = []
    for i in range(df_cv.shape[0]):
        result = 1
        for j in range(df_cv.shape[1]):
            var1 = 1/(np.sqrt(2* np.pi)* stds[j])
            var2 = (df_cv.iloc[i,j]-mean[j])**2
            var3 = 2*variance[j]

            result *= (var1) * np.exp(-(var2/var3))
        result = float(result)
        probability.append(result)
    return probability


# # select the best EPSILON by calculation the F1, PRECESION and RECALL and select the beat epsilon for each using CV DATASET

# In[ ]:


def selectEpsilon(y_actual, y_probability):
    best_epi = 0
    best_F1 = 0
    best_rec = 0
    best_pre = 0
    
    stepsize = (max(y_probability) -min(y_probability))/1000000
    epi_range = np.arange(min(y_probability),max(y_probability),stepsize)
    for epi in epi_range:
        predictions = (y_probability<epi)[:,np.newaxis]
        tp = np.sum(predictions[y_actual==1]==1)
        fp = np.sum(predictions[y_actual==0]==1)
        fn = np.sum(predictions[y_actual==1]==0)
        
        prec = tp/(tp+fp)
        rec = tp/(tp+fn)
        
        if prec > best_pre:
            best_pre =prec
            best_epi_prec = epi
            
        if rec > best_rec:
            best_rec =rec
            best_epi_rec = epi
            
        F1 = (2*prec*rec)/(prec+rec)
        
        if F1 > best_F1:
            best_F1 =F1
            best_epi = epi
        
    return best_epi, best_F1,best_pre,best_epi_prec,best_rec,best_epi_rec


# # use the probability that we had to be used with the epsilon selection process
# 

# In[ ]:


probability = multivariateGaussian(stds, mean, variance, df_cv)
best_epi, best_F1,best_pre,best_epi_prec,best_rec,best_epi_rec = selectEpsilon(df_cv_y, probability)
print("The best epsilon Threshold over the croos validation set is :",best_epi)
print("The best F1 score over the croos validation set is :",best_F1)
print("The best epsilon Threshold over the croos validation set is for recall :",best_epi_rec)
print("The best Recall score over the croos validation set is :",best_rec)
print("The best epsilon Threshold over the croos validation set is for precision:",best_epi_prec)
print("The best Precision score over the croos validation set is :",best_pre)


# # F1, Percision and Recall for testing data

# In[ ]:


def prediction_scores(y_actual, y_probability, epsilon):
    predictions = (y_probability<epsilon)[:,np.newaxis]
    tp = np.sum(predictions[y_actual==1]==1)
    fp = np.sum(predictions[y_actual==0]==1)
    fn = np.sum(predictions[y_actual==1]==0)
        
    prec = tp/(tp+fp)
    rec = tp/(tp+fn) 
    F1 = (2*prec*rec)/(prec+rec)
        
    return prec,rec,F1


# RUN the code and print out the Results for the test set

# In[ ]:


epsilon = best_epi
probability = multivariateGaussian(stds, mean, variance, df_test)
prec,rec,F1 = prediction_scores(df_test_y, probability,epsilon)
print("Percision on Testing Set:",prec)
print("Recall on Testing Set:",rec)
print("F1 on Testing Set:",F1)

