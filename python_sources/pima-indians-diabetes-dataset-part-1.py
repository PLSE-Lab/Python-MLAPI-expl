#!/usr/bin/env python
# coding: utf-8

# # Pima Indians Diabetes Data Set 
# 
# I am new to Kaggle, Machine Learning and Python and this these are my first steps in learning. I used the [blog post of Ritchie Ng](http://www.ritchieng.com/machine-learning-evaluate-classification-model/) as a backbone and complemented it with other stuff I found useful. 
# 
# The goal is to predict the onset of diabetes in Pima Indians within five years using machine learning. A model that could do this very well, could help focusing prevention measures to the affected.
# 
# The Dataset is available [here](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes). The original paper using this dataset can be found [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/). This paper from 1984 created a model with a sensitivity and specificity of 76%. We have 2017 now, let's try to beat this!
# 
# Learn more about the Pima Indians in the video below.

# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo("pN4HqWRybwk")


# ## Loading Libraries

# In[ ]:


# show plots inside the notebook  
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ## Data Loading

# In[ ]:


diabetes_dataset = pd.read_csv("../input/diabetes.csv")


# Data Set Information:
# 
# Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. 
# 
# Attribute Information:
# 
# 1. Number of times pregnant 
# 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test 
# 3. Diastolic blood pressure (mm Hg) 
# 4. Triceps skin fold thickness (mm) 
# 5. 2-Hour serum insulin (mu U/ml) 
# 6. Body mass index (weight in kg/(height in m)^2) 
# 7. Diabetes pedigree function 
# 8. Age (years) 
# 9. Class variable (0 or 1) 

# ## Data Checking

# In[ ]:


diabetes_dataset.shape


# In[ ]:


diabetes_dataset.head()


# In[ ]:


diabetes_dataset.describe()


# In[ ]:


diabetes_dataset.groupby("Outcome").size()


# Now we know that there are 768 people with an uneven distribution of the outcome (healthy:sick = 500:268). There are also some missing values (0s) for the variables 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin' and 'BMI'.

# ## Data Cleaning
# 
# The dataset contains multiple (invalid) zero values. We are going to replace zeros with the serial mean.

# In[ ]:


# This replaces zero/invalid values with the mean in the group.
# But it does not seem to improve the results, that's why it's deactivated.
# dataset_nozeros = diabetes_dataset.copy()

# zero_fields = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] 
# diabetes_dataset[zero_fields] = diabetes_dataset[zero_fields].replace(0, np.nan)
# diabetes_dataset[zero_fields] = diabetes_dataset[zero_fields].fillna(dataset_nozeros.mean())
# diabetes_dataset.describe()  # check that there are no invalid values left


# ## Data Stratification
# 
# When we split the dataset into train and test datasets, the split is completely random. Thus the instances of each class label or outcome in the train or test datasets is random. Thus we may have many instances of class 1 in training data and less instances of class 2 in the training data. So during classification, we may have accurate predictions for class1 but not for class2. Thus we stratify the data, so that we have proportionate data for all the classes in both the training and testing data.

# In[ ]:


from sklearn.model_selection import train_test_split 

# divide into training and testing data
train,test = train_test_split(diabetes_dataset, test_size=0.25, random_state=0, stratify=diabetes_dataset['Outcome']) 

# separate the 'Outcome' column from training/testing data
train_X = train[train.columns[:8]]
test_X = test[test.columns[:8]]
train_Y = train['Outcome']
test_Y = test['Outcome']


# ## Statistical Model

# In[ ]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(train_X,train_Y)
prediction = model.predict(test_X)


# **accuracy**
# 
# ... is the percentage of correct predictions.

# In[ ]:


# calculate accuracy
from sklearn import metrics

print(metrics.accuracy_score(test_Y, prediction))


# **null accuracy**
# 
# ... is the accuracy that could be achieved by always predicting the most frequent class.

# In[ ]:


the_most_outcome = diabetes_dataset['Outcome'].median()
prediction2 = [the_most_outcome for i in range(len(test_Y))]
print(metrics.accuracy_score(test_Y, prediction2))


# This means that a dumb model that always predicts 0 would be right 65.1% of the time. This shows how classification accuracy might not be the best method to compare different models.
# 
# Accuracy just tells you the percentage of correct predictions, but it does not tell you what "types" of errors your classifier is making. The confusion matrix gives you a more complete picture of how your classifier is performing.

# **Confusion Matrix**

# In[ ]:


from sklearn import metrics

confusion_matrix = metrics.confusion_matrix(test_Y, prediction)
confusion_matrix


# In[ ]:


plt.figure()
plt.matshow(confusion_matrix, cmap='Pastel1')

for x in range(0, 2):
    for y in range(0, 2):
        plt.text(x, y, confusion_matrix[x, y])
        
plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.show()


# For beginners the visual respresentation of the confusion matrix helps a lot. Later on you probably don't need it anymore. We can see the following in the table:
# 
# 0: Person will have diabetes in 5 years<br />
# 1: Person will NOT have diabetes in 5 years
# 
# **True Positives (TP)**: (39) we correctly predicted that they do have diabetes<br />
# **True Negatives (TN)**: (110) we correctly predicted that they don't have diabetes<br />
# **False Positives (FP)**: (28) we incorrectly predicted that they do have diabetes<br />
# **False Negatives (FN)**: (15) we incorrectly predicted that they don't have diabetes
# 
# From these values we can calculate the following classification metrics:
# 
# **Sensitivity** (aka "True Positive Rate" or "Recall"): When the actual value is positive, how often is the prediction correct?
# - Something we want to maximize
# - How "sensitive" is the classifier to detecting positive instances?
# 
# **Specificity**: When the actual value is negative, how often is the prediction correct?
# - Something we want to maximize
# - How "specific" (or "selective") is the classifier in predicting positive instances?
# 
# 
# 

# In[ ]:


# [row, column]
TP = confusion_matrix[1, 1]
TN = confusion_matrix[0, 0]
FP = confusion_matrix[0, 1]
FN = confusion_matrix[1, 0]

print("Sensitivity: %.4f" % (TP / float(TP + FN)))
print("Specificy  : %.4f" % (TN / float(TN + FP)))


# Right now we have a model that is quiet good in correctly detecting if a patient is healthy in 5 years. But I think it's more important to find people, which will be sick in 5 years to apply preventive measures. We can adjust our model by changing the classification threshold.

# **Classification Threshold**

# In[ ]:


# print the first 10 predicted responses
# 1D array (vector) of binary values (0, 1)
model.predict(test_X)[0:10]


# In[ ]:


# print the first 10 predicted probabilities of class membership
model.predict_proba(test_X)[0:10]


# Observations:
# 
# 1. In each row the numbers sum up to 1
# 2. There are 2 columns for 2 classes ('Outcome' = 0 and 'Outcome' = 1)
#   - column 0: predicted probability that each observation is a member of class 0
#   - column 1: predicted probability that each observation is a member of class 1
# 3. Choose the class with the highest probability (classification threshold = 0.5)
#   - Class 1 is predicted if probability > 0.5
#   - Class 0 is predicted if probability < 0.5

# In[ ]:


# histogram of predicted probabilities

save_predictions_proba = model.predict_proba(test_X)[:, 1]  # column 1

plt.hist(save_predictions_proba, bins=10)
plt.xlim(0,1) # x-axis limit from 0 to 1
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')
plt.show()


# Just a small number of observations with probability > 0.5, most observations have a probability < 0.5 and would be predicted "no diabetes" in our case. We can increase the sensitivity (increase number of TP) of the classifier by decreasing the threshold for predicting diabetes.

# In[ ]:


# predict diabetes if the predicted probability is greater than 0.3
from sklearn.preprocessing import binarize

# it will return 1 for all values above 0.3 and 0 otherwise
# results are 2D so we slice out the first column
prediction2 = binarize(save_predictions_proba.reshape(-1, 1), 0.3)  # [0]


# In[ ]:


confusion_matrix2 = metrics.confusion_matrix(test_Y, prediction2)
confusion_matrix2

# previous confusion matrix
# array([[110,  15],
#        [ 28,  39]])


# In[ ]:


TP = confusion_matrix2[1, 1]
TN = confusion_matrix2[0, 0]
FP = confusion_matrix2[0, 1]
FN = confusion_matrix2[1, 0]

print("new Sensitivity: %.4f" % (TP / float(TP + FN)))
print("new Specificy  : %.4f" % (TN / float(TN + FP)))

# old Sensitivity: 0.5821
# old Specificy  : 0.8800


# Observations: 
# 
# - Threshold of 0.5 is used by default (for binary problems) to convert predicted probabilities into class predictions
# - Threshold can be adjusted to increase sensitivity or specificity
# - Sensitivity and specificity have an inverse relationship (Increasing one would always decrease the other)
# - Adjusting the threshold should be one of the last steps you do in the model-building process

# **Receiver Operating Characteristic (ROC) Curves**
# 
# Wouldn't it be nice if we could see how sensitivity and specificity are affected by various thresholds, without actually changing the threshold?  We can do that by plotting the ROC curve.

# In[ ]:


from sklearn.metrics import roc_curve, auc

# function roc_curve
# input: IMPORTANT: first argument is true values, second argument is predicted probabilities
#                   we do not use y_pred_class, because it will give incorrect results without 
#                   generating an error
# output: FPR, TPR, thresholds
# FPR: false positive rate
# TPR: true positive rate
FPR, TPR, thresholds = roc_curve(test_Y, save_predictions_proba)

plt.figure(figsize=(10,5))  # figsize in inches
plt.plot(FPR, TPR)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 50%  
plt.plot(FPR, TPR, lw=2, label='Logaristic Regression (AUC = %0.2f)' % auc(FPR, TPR))
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.legend(loc="lower right")


# AUC is the percentage of the ROC plot that is underneath the curve:
# 
# - AUC is useful as a single number summary of classifier performance, Higher value = better classifier
# - AUC of 0.5 is like tossing a coin
# - AUC is useful even when there is high class imbalance (unlike classification accuracy)
# like in Fraud case with a null accuracy almost 99%
# 
# A ROC curve can help you to choose a threshold that balances sensitivity and specificity in a way that makes sense for your particular context. You can't actually see the thresholds used to generate the curve on the ROC curve itself. You can use the function and plot below.

# In[ ]:


# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print("Sensitivity: %.4f" % (TPR[thresholds > threshold][-1]))
    print("Specificy  : %.4f" % (1 - FPR[thresholds > threshold][-1]))

print ('Threshold = 0.5')
evaluate_threshold(0.5)
print ()
print ('Threshold = 0.35')
evaluate_threshold(0.35)


# In[ ]:


spec = []
sens = []
thres = []

threshold = 0.00
for x in range(0, 90):
    thres.append(threshold)
    sens.append(TPR[thresholds > threshold][-1])
    spec.append(1 - FPR[thresholds > threshold][-1])
    threshold += 0.01
    
plt.plot(thres, sens, lw=2, label='Sensitivity')
plt.plot(thres, spec, lw=2, label='Specificity')
ax = plt.gca()
ax.set_xticks(np.arange(0, 1, 0.1))
ax.set_yticks(np.arange(0, 1, 0.1))
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('Sensitivity vs. Specificity')
plt.xlabel('Threshold')
plt.grid(True)
plt.legend(loc="center right")


# Conclusion: By using Logaristic Regression and a threshold of 0.35, we found an algorithm with a sensitivity of about 0.81 and a specificity of 0.81. And thereby we reached a better result than the authors of the original paper, which used the ADAP learning algorithm with a sensitivity of 0.76 and a specificity of 0.76. 
