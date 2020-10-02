#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Rad traiining data from train.csv
training_data = pd.read_csv("../input/train.csv", low_memory = False)


# In[ ]:


training_data.head()


# In[ ]:


#Question 1
print('Percentage of defaults in training data set: %7.4f' %(training_data.default.mean()*100)+'%')


# In[ ]:


#Question 2
number_defaults_by_ZIP = training_data.groupby('ZIP').default.mean()
print('ZIP code with highest default rate is: '+number_defaults_by_ZIP.idxmax())


# In[ ]:


#Question 3
#The rate is interpreted is fractional percentage of defaults of all loans issued in first year. 
#This can be calculated using simple average since the values are binary
first_year_default_rate = training_data.default.groupby(training_data.year).mean()[0]
print('Default rate in first year is: '+str(first_year_default_rate*100)+'%')


# In[ ]:


#Question 4
print('Correlation of age and income is:  %8.5f' %(training_data['age'].corr(training_data['income'])))


# In[ ]:


#Question 5

y_train= training_data['default']
#We have to prepare training data by replacing categorical features with dummies
dummy_list = ['ZIP','occupation']
important_features = training_data[['ZIP','occupation','rent','education','income','loan_size','payment_timing','job_stability']]
x_train_set = pd.get_dummies(data = important_features, columns= dummy_list, sparse=True)

from sklearn.ensemble import RandomForestClassifier
#Specify random forest model parameters
rf_model = RandomForestClassifier(n_estimators = 100, max_depth=4, random_state = 42, oob_score=True, n_jobs=-1)

# Train the model using training data
rf_model.fit(x_train_set, y_train)

#Run predictions on test data
predictions = rf_model.predict(x_train_set)


# In[ ]:


#Import library that help calculate accuracy score
from sklearn.metrics import accuracy_score
print("In-sample: Accuracy on data used to train is: %8.5f" %(accuracy_score(y_train, predictions)*100)+"%")


# In[ ]:


#Question 6
#Out of bag sample accuracy
print("Out of bag accuracy is:  %8.5f" %((rf_model.oob_score_)*100)+"%")


# In[ ]:


#Question 7
test_data = pd.read_csv("../input/test.csv", low_memory = False)

#Just like training set, we prepare test set categorical features to dummies
dummy_list = ['ZIP','occupation']
#Convert categorical features into dummies
important_features = test_data[['ZIP','occupation','rent','education','income','loan_size','payment_timing','job_stability']]
x_test = pd.get_dummies(data = important_features, columns= dummy_list, sparse=True)
y_test = test_data['default']

#Test trained model accuracy using test data
predictions = rf_model.predict(x_test)
print("Out of sample accuracy is:" +str((accuracy_score(y_test,predictions))*100)+"%")


# In[ ]:


#Question 8
test_data['predictions'] = predictions # Get test data predictions and append them to the test_data set
#Average predictions of non - minority
print(test_data.predictions.groupby(test_data.minority).mean())
non_minority_prob = test_data.predictions.groupby(test_data.minority).mean()[0]
print('\nNon minority probability of default is: %9.6f' %(non_minority_prob))
#This gives the average of default predictions associated with the first value which happened to be
#minorities =0: Non-minorities
#The everage equals probability of default because it gives the total defaults divided by total loans lent to minorities


# In[ ]:


#Question 9
minority_prob = test_data.predictions.groupby(test_data.minority).mean()[1]
print('Minority probability of default is: %9.6f' %(minority_prob))
#Explanations as above


# In[ ]:


#Question 10
#The scheme is group unaware because it applies 50% threshold to all applicants, irregadless of the group they belong to


# In[ ]:


#Question 11 and 12
from sklearn.metrics import confusion_matrix

#This function is designed to help answer all demographic parity question
#Instead of writing 4 chunks of similary code to compare fairness of scheme among females & males and minorities & non-minority 
#groups, a single function is written that takes parameters that indicate information we are looking for as demonstrated below

def demographic_fairness(parity_type, category):
    
    if parity_type == 'sex' and category==1:# We are are interested in getting information about how our scheme is responding to females
        ds = test_data[test_data.sex==1]
        parity_name = 'female'
    elif parity_type == 'sex' and category==0: # We are are interested in getting information about how our scheme is responding to males
        ds = test_data[test_data.sex==0]
        parity_name = 'male'
    elif parity_type == 'minority' and category==1: # Interested in non minority
        ds = test_data[test_data.minority==1]
        parity_name = 'minority'
    elif parity_type == 'minority' and category==0: # We are are interested in getting information about how our scheme is responding to males
        ds = test_data[test_data.minority==0]
        parity_name = 'non minority'

    #This line of code compares predicted defaults to ground truth defaults and generates confusion matrix that helps 
    #analyse model performance for then given feature of interest
    cm=confusion_matrix(ds.default, ds.predictions,  labels=[1,0], sample_weight=None)

    print(pd.DataFrame(cm, index=['groundtruth:1', 'groundtruth:0'],columns=['predictions:1', 'predictions:0']))

    print('Total rejected '+parity_name+' applicants: '+str(cm[0][0]+cm[1][0]))
    print('Total accepted '+parity_name+' applicants: '+str(cm[0][1]+cm[1][1]))
    print('Percentage of  '+parity_name+' applicants accepted: '+str((cm[0][1]+cm[1][1])/(cm[0][1]+cm[1][1]+cm[0][0]+cm[1][0])*100)+'%')
    print('Percentage of '+parity_name+' applicants rejected: '+str((cm[0][0]+cm[1][0])/(cm[0][1]+cm[1][1]+cm[0][0]+cm[1][0])*100)+'%')
    
    print('Share of successful '+parity_name+' applicants that defaulted : '+str((cm[0][1]/(cm[0][1]+cm[1][1]))*100)+'%')
    #print('False discovery rate: '+str((cm[1][0]/(cm[0][0]+cm[1][0]))*100)+'%')
    #print('False negative rate:  '+str((cm[0][1]/(cm[0][1]+cm[0][0]))*100)+'%')


# In[ ]:


#Question 11 and 12
#To get information about how scheme is responsing to females call the following function
print('----------scheme response to females--------------------')
demographic_fairness('sex', 1)

print('----------scheme response to males--------------------')
demographic_fairness('sex', 0)

print('----------scheme response to minorities--------------------')
demographic_fairness('minority', 1)

print('----------scheme response to non minorities--------------------')
demographic_fairness('minority', 0)

#Q11 Comments
#Sex parity
#The model accepts more females than males according to the percetages calculated. We can tell the model is biased against 
#males from the false discovery [of defaults] which is higher for males compared to that of females

#Minority/non-minority parity
#The model accepts accepts a much higher percentage of non-minorities compared to minority members. The bias is confirmed by 
#the rate discovery rate which is high for non-minorities compared to 0% for non-minorities

#The model will need improvement by perhaps getting rid of protected features so that it is gender and minority status blind

#Q12 comments
#As mentioned above, the model thinks a male is most likely to default as indicated by the false negative rate which favors 
#females. The model places higher likelihood of default on males. It is hard to conclude unfaireness on gender parity because 
#the deference in the false negative rates is not big enough.
#Likewise, the model places a much higher likelihood of default on minorities of default and by far favours non-minority members
#Based on this, we can conclude that non-minority members are more likely to secure a loan compared to minority counterparts
#The scheme is unfair

