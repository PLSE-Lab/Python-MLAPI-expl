#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Import the test and train datasets

test_data = pd.read_csv("../input/test.csv", index_col=0, low_memory = False)
train_data = pd.read_csv("../input/train.csv", index_col=0, low_memory = False)

import os
print(os.listdir("../input"))

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from string import ascii_letters
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Any results you write to the current directory are saved as output.


# In[ ]:


#Drop the 'Unnamed:0.1' column because I think it is an error

test_data=test_data.drop(['Unnamed: 0.1'],1)

print('done')


# **Question 1**
# 
# Find the default rate train_data
# 
# * I also want to do this in the test_data, just to check

# In[ ]:


#The mean of the default (0 or 1) gives the overall default rate
print(f'Default Rate in train_data: {train_data.default.mean()*100:.2f}%')
print(f'Default Rate in test_data:  {test_data.default.mean()*100:.2f}%')


# **Interpretation**
# > These two datasets have very different default rates. This is an obvious problem because the train and test should have similar characteristics

# **Question 2**
# 
# Examine each zip code's default rate
# * Again, I'll do this for both train and test, just to compare

# In[ ]:


#Group on Zip and displace the mean of default
grouped_train = train_data.groupby(['ZIP'], sort=False).mean().default #Do the function
grouped_train = grouped_train.sort_values(ascending=False) #Sort it by default rate

grouped_test = test_data.groupby(['ZIP'], sort=False).mean().default #Do the function
grouped_test = grouped_test.sort_values(ascending=False) #Sort it by default rate

#Print the output
print('Train Data')
print(grouped_train)
print()
print('Zip Code with highest default rate:', grouped_train.idxmax())
print()
print('Test Data')
print(grouped_test)
print('Zip Code with highest default rate:', grouped_test.idxmax())


# **Interpretation**
# > In the train data, two zip codes almost always default and two never default
# > The same is true in the test, but at much lower rates.
# 
# > What I expect most models to do is to predict default for anyone in the MT04PA and MT15PA zip codes.
# > There will be errors since only 30% of the time, those will actually default.

# **Question 3**
# 
# Get the default rate in the first year of the data (Year 0)
# 

# In[ ]:


#Calculate the mean of default in the data where year is 0
print(f'Train Data: {train_data.default[train_data.year==0].mean()*100:.2f}%')
print(f'Test Data: {train_data.default[test_data.year==0].mean()*100:.2f}%')


# **Further analysis**
# 
# There are not any defaults in Year 0 of the test data. Is this actually the case?

# In[ ]:


#Get the number of loans in each year of the test data
test_data.groupby('year').size()


# **Interpretation**
# 
# > The reason there are no defaults in year 0 of the test data is that the test data begins in year 30.
# > This is an indication that maybe the data was split into train and test chronologically.

# **Question 4**
# 
# What is the correlation of age and income?
# * Do train and test

# In[ ]:


#Use the Corr function
print(f'Correlation of Income and Age in Train: {train_data["income"].corr(train_data["age"]):.4f}')
print(f'Correlation of Income and Age in Test:  {test_data["income"].corr(test_data["age"]):.4f}')


# **Question 5**
# 
# Build a model on the train data and test it in-sample, that is, on the train data itself

# In[ ]:


#set X_train to include features included in assignment instructions
X_train = train_data[['ZIP','rent','education','income','loan_size','payment_timing','job_stability','occupation']]

#Turn categorical features to dummies
X_train = pd.get_dummies(X_train, columns=["ZIP", "occupation"])

#set y_train to be 'default' train_data and get dummies
y_train = train_data.default

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100,
                              random_state=42,
                               n_jobs=-1,
                              oob_score = True)
# Fit on training data
model.fit(X_train, y_train)

#Run the test
y_pred = model.predict(X_train)

print(f'In-Sample Accuracy: {metrics.accuracy_score(y_train, y_pred)*100:.4}%')


# **Interpretation**
# > An accuracy of 100% can either indicate overfitting or the use of data that is 100% correlated to actual defaults.
# > Knowing what we know about the test data, the latter is likely.

# **Question 6**
# 
# What is the Out of bag score?

# In[ ]:


print(f'Out of Bag Score: {model.oob_score_*100:.4f}%')


# **Question 7** 
# 
# Use the model created above, but run on the test data

# In[ ]:


#set X_test to be all data in test_data, but drop default
X_test = test_data[['ZIP','rent','education','income','loan_size','payment_timing','job_stability','occupation']]

#Turn categorical features to dummies
X_test = pd.get_dummies(X_test, columns=["ZIP", "occupation"])

#set y_train to be 'default' train_data and get dummies
y_test = test_data.default

#Run the test
y_pred = model.predict(X_test)

print(f' Out-of-Sample Accuracy: {metrics.accuracy_score(y_test, y_pred)*100:.4f}%')


# **Interpretation**
# > Still high, but not 100%.
# > Since the two datasets are not similar, the model is applying what it learned on a dataset that follows different rules.
# 
# Looking at the confusion matrix...

# In[ ]:


#confusion matrix
def print_confusion_matrix(test_default,predictions_test):
    cm = confusion_matrix(test_default,predictions_test)
    print('Paying Applicant Approved     :', cm[0][0], '---', cm[0][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100,"%")
    print('Paying Applicant Rejected     :', cm[0][1], '---', cm[0][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100,"%")
    print('Defaulting Applicant Approved :', cm[1][0], '---', cm[1][0]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100,"%")
    print('Defaulting Applicant Rejected :', cm[1][1], '---', cm[1][1]/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])*100,"%")
print_confusion_matrix(y_test,y_pred)


# > The model only rejected 3,010 applicants that would have paid.
# > The model approved 16,294 applicants that defaulted. This is 68% of all the defaults.
# > 
# > The model is overpredicting repayment.

# **Question 8,9**
# 
# What is the predicted average default probability for all non-minority members in the test set?  Minorities?
# 
# 

# In[ ]:


pred_proba = model.predict_proba(X_test) #Get the actual probabilities from the model, not just binary
pred_proba = pd.DataFrame(pred_proba, columns=["proba_pay","proba_default"]) #rename those columns to make it easier to understand
test_data['proba_pay']= pred_proba.proba_pay #initiate a column on the test_data for the model's predicted probability of pay
test_data['proba_default']=pred_proba.proba_default #initiate a column on the test_data for the model's predicted probability of default
test_data['y_pred'] = y_pred #initiate a column on the test_data for the model prediction (binary)

#Print predicted default rate by minority status
print("Predicted Default Probability by Minority Status (%):")
print(test_data.groupby(["minority"]).mean().proba_default*100)

#Look at some other demographics
print()
print("Predicted Default Probability by Gender (%):")
print(test_data.groupby(["sex"]).mean().proba_default*100)
print()
print("Predicted Default Probability by Status and Gender (%):")
print(test_data.groupby(["minority","sex"]).mean().proba_default*100)


# **Interpretation**
# >The model treats genders roughly the same
# >The model is defaulting more minorities than non-minorities
# >The model thinks the mean probability of default is about the same for all people

# **Question 10**
# 
# Is the loan granting scheme (the cutoff, not the model) group unaware? (This question does not require calculation as the cutoff is given in the introduction to this assignment)
# > 
# > Yes, the scheme is unaware.
# > All groups are held to the same standards
# > Maybe the groups aren't equal. So far I haven't seen that, but there is plenty more that could be investigated
# > If groups are unequal in their default rates, maybe it should be aware, but that has trade-offs

# **Question 11**
# 
# Has the loan granting scheme achieved demographic parity? Compare the share of approved female applicants to the share of rejected female applicants. Do the same for minority applicants. Are the shares roughly similar between approved and rejected applicants? What does this indicate about the demographic parity achieved by the model?

# In[ ]:


#Print rate of acceptance rate by minority status
print("Acceptance rate by Minority Status (%):")
print(100-test_data.groupby(["minority"]).mean().y_pred*100)

#Look at some other demographics
print()
print("Acceptance Rate of Defaults by Gender (%):")
print(100-test_data.groupby(["sex"]).mean().y_pred*100)
print()
print("Acceptance Rates of Default by Minority Status and Gender (%):")
print(100-test_data.groupby(["minority","sex"]).mean().y_pred*100)


# > From the analysis above, we can see that there is not demographic parity based on minority status.
# > The acceptance rate of Female (sex=1) is 92.99% and Male (sex=0) is 93.69%. The difference is insignificant (i.e. parity is achieved)
# > The acceptance rate of Minorities (minority = 1) is 89.6%, but the acceptance rate for non-minorities (minority=0) is 97.1%.
# > This difference is likely significant (i.e. parity is not achieved)
# > There are no significant differences of gender within each minority status. 

# **Question 12**
# 
# Is the loan granting scheme equal opportunity? Compare the share of successful non-minority applicants that defaulted to the share of successful minority applicants that defaulted. Do the same comparison of the share of successful female applicants that default versus successful male applicants that default. What do these shares indicate about the likelihood of default in different population groups that secured loans?

# In[ ]:


print('Confusion Matrix of Minorities')
print_confusion_matrix(test_data.default[test_data.minority==1],test_data.y_pred[test_data.minority==1])

print()
print('Confusion Matrix of Non-Minorities')
print_confusion_matrix(test_data.default[test_data.minority==0],test_data.y_pred[test_data.minority==0])


# * The model is equally accurate at accepting paying minority applicants as paying non-minority applicants.
# * The model rejects paying minority applicants at the same rate as it rejects paying non-minority applicants.
# * The model accepts twice as many defaulting non-minorities as defaulting minorities. 
# * The model only rejects 1% of defaulting non-minorities, but 8.5% of defaulting minorities
# 
# 
# * These figures indicate that the default rates for minorities and non-minorities are about equal in the test_data
# * 14.9% of non-minorities vs 15.0% of minorities
# 

# In[ ]:


print('Confusion Matrix of Female')
print_confusion_matrix(test_data.default[test_data.sex==1],test_data.y_pred[test_data.sex==1])

print()
print('Confusion Matrix of Male')
print_confusion_matrix(test_data.default[test_data.sex==0],test_data.y_pred[test_data.sex==0])


# No significant differences.

# +++++++++++++++++++++++++++++++++++
# 
# **SUPPORTING ANALYSIS**
# 
# +++++++++++++++++++++++++++++++++++
# 
# First, how does the training data and the test data compare? They should be fairly indistinguishable.

# In[ ]:


features = ['education', 'age', 'income', 'loan_size', 'payment_timing', 'job_stability']

for x in features:
    plt.hist(train_data[x], bins=100, alpha=0.5, label='train')
    plt.hist(test_data[x], bins=100, alpha=0.5, label='test')
    plt.legend(loc='upper right')
    print(x)
    print(plt.show())


# Age, Income, and Job Stability are very different between the two datasets.
# 
# Check the default rates in these features.

# In[ ]:


print("++++++++++++++++++")
print("Job Stability")
print("++++++++++++++++++")

bins = pd.qcut(train_data['job_stability'], 10) #Cut the data into deciles
print("Train Data")
print(train_data.groupby(bins)['default'].mean())
print("")
bins = pd.qcut(test_data['job_stability'], 10) #Cut the data into deciles
print("Test Data")
print(test_data.groupby(bins)['default'].mean())

print("")
print("++++++++++++++++++")
print("Age")
print("++++++++++++++++++")

bins = pd.qcut(train_data['age'], 10) #Cut the data into deciles
print("Train Data")
print(train_data.groupby(bins)['default'].mean())
print("")
bins = pd.qcut(test_data['age'], 10) #Cut the data into deciles
print("Test Data")
print(test_data.groupby(bins)['default'].mean())

print("")
print("++++++++++++++++++")
print("Income")
print("++++++++++++++++++")

bins = pd.qcut(train_data['income'], 10) #Cut the data into deciles
print("Train Data")
print(train_data.groupby(bins)['default'].mean())
print("")
bins = pd.qcut(test_data['income'], 10) #Cut the data into deciles
print("Test Data")
print(test_data.groupby(bins)['default'].mean())


# How do the Zip codes, Minorities, and Sex compare between datasets?

# In[ ]:


#Pivot tables for train and test of default rate by group. 
#'len default' is a bit of a misnomer, it counts how many people exist in the group.

print('+++++++++++++++++++++++++++')
print('Train')
print('+++++++++++++++++++++++++++')
print(pd.pivot_table(train_data,index=["ZIP"],values=["default"],aggfunc=[np.mean,len]))
print('')

print('+++++++++++++++++++++++++++')
print('Test')
print('+++++++++++++++++++++++++++')
print(pd.pivot_table(test_data,index=["ZIP"],values=["default"],aggfunc=[np.mean,len]))


# In[ ]:


#Pivot tables for train and test of default rate by group. 
#'len default' is a bit of a misnomer, it counts how many people exist in the group.

print('+++++++++++++++++++++++++++')
print('Train')
print('+++++++++++++++++++++++++++')
print(pd.pivot_table(train_data,index=["minority"],values=["default"],aggfunc=[np.mean,len]))
print('')

print('+++++++++++++++++++++++++++')
print('Test')
print('+++++++++++++++++++++++++++')
print(pd.pivot_table(test_data,index=["minority"],values=["default"],aggfunc=[np.mean,len]))


# In[ ]:


#Pivot tables for train and test of default rate by group. 
#'len default' is a bit of a misnomer, it counts how many people exist in the group.

print('+++++++++++++++++++++++++++')
print('Train')
print('+++++++++++++++++++++++++++')
print(pd.pivot_table(train_data,index=["sex"],values=["default"],aggfunc=[np.mean,len]))
print('')

print('+++++++++++++++++++++++++++')
print('Test')
print('+++++++++++++++++++++++++++')
print(pd.pivot_table(test_data,index=["sex"],values=["default"],aggfunc=[np.mean,len]))


# In[ ]:


#Pivot tables for train and test of default rate by group. 
#'len default' is a bit of a misnomer, it counts how many people exist in the group.

print('+++++++++++++++++++++++++++')
print('Train')
print('+++++++++++++++++++++++++++')
print(pd.pivot_table(train_data,index=["ZIP","minority"],values=["default"],aggfunc=[np.mean,len]))
print('')

print('+++++++++++++++++++++++++++')
print('Test')
print('+++++++++++++++++++++++++++')
print(pd.pivot_table(test_data,index=["ZIP","minority"],values=["default"],aggfunc=[np.mean,len]))


# In[ ]:


#Pivot tables for train and test of default rate by group. 
#'len default' is a bit of a misnomer, it counts how many people exist in the group.

print('+++++++++++++++++++++++++++')
print('Train')
print('+++++++++++++++++++++++++++')
print(pd.pivot_table(train_data,index=["sex","minority"],values=["default"],aggfunc=[np.mean,len]))
print('')

print('+++++++++++++++++++++++++++')
print('Test')
print('+++++++++++++++++++++++++++')
print(pd.pivot_table(test_data,index=["sex","minority"],values=["default"],aggfunc=[np.mean,len]))


# In[ ]:


train_data=pd.get_dummies(train_data, columns=["ZIP", "occupation"])


# In[ ]:


#What is the correlation of features in training data?

#produce a correlation matrix

sns.set(style="white")

# Compute the correlation matrix
corr = train_data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[ ]:


#What is feature importance in the model?

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh')


# In[ ]:


#Let's produce a correlation matrix

sns.set(style="white")

# Compute the correlation matrix
corr = X_test.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# The splitting of the training and test data was bad. The two sets look completely different. This is especially true in the default rates of minorities and zip codes. Job stability is completely different between the two datasets and was the feature with the most importance in the model. This means the model is predicting based on a feature with bad data.
# 
# Next, I want to examine the confusion matrix of the model.

# In[ ]:


#How often did the model predict a default for zip code?
test_data_dummies = pd.get_dummies(test_data, columns=["ZIP", "occupation"])

#0,1 predictions
print('Calculated based on default/no default')
print('ZIP_MT01RA:', test_data_dummies.y_pred[test_data_dummies.ZIP_MT01RA==1].mean())
print('ZIP_MT04PA:', test_data_dummies.y_pred[test_data_dummies.ZIP_MT04PA==1].mean())
print('ZIP_MT12RA:', test_data_dummies.y_pred[test_data_dummies.ZIP_MT12RA==1].mean())
print('ZIP_MT15PA:', test_data_dummies.y_pred[test_data_dummies.ZIP_MT15PA==1].mean())

#prob predictions
print()
print('calculated based on model probability')
print('ZIP_MT01RA:', test_data_dummies.proba_default[test_data_dummies.ZIP_MT01RA==1].mean())
print('ZIP_MT04PA:', test_data_dummies.proba_default[test_data_dummies.ZIP_MT04PA==1].mean())
print('ZIP_MT12RA:', test_data_dummies.proba_default[test_data_dummies.ZIP_MT12RA==1].mean())
print('ZIP_MT15PA:', test_data_dummies.proba_default[test_data_dummies.ZIP_MT15PA==1].mean())


# In[ ]:


print('train')
print(pd.pivot_table(train_data,index=["rent"],values=["minority"],aggfunc=[np.mean,len]))
print()
print('test')
print(pd.pivot_table(test_data,index=["rent"],values=["minority"],aggfunc=[np.mean,len]))


# In[ ]:


bins = pd.qcut(train_data['job_stability'], 10) #Cut the data into deciles
print("Train Data")
print(train_data.groupby(bins)['minority'].mean())
print("")
bins = pd.qcut(test_data['job_stability'], 10) #Cut the data into deciles
print("Test Data")
print(test_data.groupby(bins)['minority'].mean())


# In[ ]:


print('Job Stability and Occupation_MZ01CD:', test_data_dummies['job_stability'].corr(test_data_dummies['occupation_MZ01CD']))
print('Job Stability and Occupation_MZ10CD:', test_data_dummies['job_stability'].corr(test_data_dummies['occupation_MZ10CD']))
print('Job Stability and Occupation_MZ11CD:', test_data_dummies['job_stability'].corr(test_data_dummies['occupation_MZ11CD']))
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Default and Occupation_MZ01CD:', test_data_dummies['default'].corr(test_data_dummies['occupation_MZ01CD']))
print('Default and Occupation_MZ10CD:', test_data_dummies['default'].corr(test_data_dummies['occupation_MZ10CD']))
print('Default and Occupation_MZ11CD:', test_data_dummies['default'].corr(test_data_dummies['occupation_MZ11CD']))

