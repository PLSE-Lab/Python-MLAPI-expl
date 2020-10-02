#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction

# **Problem statement** <br>
# Using the UCI PIMA Indian Diabetes dataset to predict a person has diabetes or not using the medical attributes provided.
# 
# ** Assumptions **
# 1. This is enough data to split and reliably predict if the patient has diabetes, the dataset has only 786 data points
# 2. Just these attributes are enough to diagnose the ailment
# 
# ** Similar Problems ** <br> 
# This is very much like some common 2 class classification problems like classifying mail into spam and ham based on the contents of the email. Obviously the attributes there would be strings and not numbers like this dataset, therefore the way in which we process at least some of the features will be different.
# 
# ** Why am I solving this problem and what do I want the next steps to be? **
# 1. I am a beginner at data science & ML, I want to use this problem to learn more about 2 class classification algorithms and how they work
# 2. I hope to use this as a way to get a better understanding of the various 2 class classification algorithms
# 3. ***Most Importantly*** I want to get some valuable feedback on how I have tried to solve the problem and how it can be improved
# 
# So lets get started...
# 

# In[ ]:


# Importing required libraries to get started
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

get_ipython().run_line_magic('matplotlib', 'inline')


# # 2. Data Exploration

# Lets pull in the data and see what's in it, Here is what we already know about this data <br>
# 
# ## 2.1 Data Overview
# 
# <br>
# ** Columns **
# 1. pregnancies - Number of times pregnant 
# 2. Glucose - Plasma glucose concentration a 2 hours in an oral glucose tolerance test 
# 3. BloodPressure - Diastolic blood pressure (mm Hg) 
# 4. SkinThickness - Triceps skin fold thickness (mm) 
# 5. Insulin - 2-Hour serum insulin (mu U/ml) 
# 6. BMI - Body mass index (weight in kg/(height in m)^2) 
# 7. DiabetesPedigreeFunction - Diabetes pedigree function 
# 8. Age - Age (years) 
# 9. Outcome - Class variable (0 or 1) class value 1 is interpreted as "tested positive for
#    diabetes
#    
# **Class distribution:** <br>
# 0 : 500 <br>
# 1 :  268 <br>
# 
# **Data characteristics:**
# * The database contains only data about **female** patients who are of **Pima Indian heritage** are **21 or older**
# * All the attributes are numeric
# * The data may contain invalid or null values
# * Total number of cases presented are 786

# In[ ]:


data = pd.read_csv('../input/diabetes.csv')
data.head()


# In[ ]:


data.describe()


# In[ ]:


sns.set(style='ticks')
plt.figure(figsize=(20,20))
sns.pairplot(data, hue='Outcome')


# ## 2.2 Invalid data
# Based on the description above, Plasma glucose levels, blood pressure, skin thickness, insulin and BMI all have min values at 0 which does not pass my smoke test, especially blood pressure, since the diastolic blood pressure most likely cannot be 0 for a living person, I think...

# 1. Blood Pressure: <br>
# Based on the information provided for [blood pressure in adults on wikipedia](https://en.wikipedia.org/wiki/Blood_pressure#Classification), any diastolic blood pressure under 60 is considered hypotension which needs to be treated immideately as it indicates not enough blood is reaching the person's organs, the person is considered to be in "shock". Browsing more on this subject there are cases where the diastolic bp is read even lower while not exhibiting signs of hypotension but that's mostly rare. the distolic blood pressure can be low in case the person is sleeping too.
# **Assuming** these are normal healthy women who are currently not suffering from hypotension or are in the ER currently being treated for hypotension and they are awake at the time of the data collection, the 0 values are clearly invalid. Lets find out how many cases we have of this

# In[ ]:


print(data[data.BloodPressure == 0].shape[0])
print(data[data.BloodPressure == 0].index.tolist())
print(data[data.BloodPressure == 0].groupby('Outcome')['Age'].count())


# 2.Plasma glucose levels: <br>
# the range is normally 3.9 to 7.2 for non-diabetic patients even after fasting [[Source]](https://en.wikipedia.org/wiki/Blood_sugar_level). Given this, the 0s here are not valid either. Number of cases of this:

# In[ ]:


print(data[data.Glucose == 0].shape[0])
print(data[data.Glucose == 0].index.tolist())
print(data[data.Glucose == 0].groupby('Outcome')['Age'].count())


# 3.Skin Fold Thickness: <br>
#     for normal healthy adults the skin fold thinkness is not less than 10mm even for girls [[source]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5083983/)

# In[ ]:


print(data[data.SkinThickness == 0].shape[0])
print(data[data.SkinThickness == 0].index.tolist())
print(data[data.SkinThickness == 0].groupby('Outcome')['Age'].count())


# 4.BMI: <br>
# Based on WebMD data BMI among adults range from 18.5 to 30.0 or higher. Assuming none of these women are extremely short or extremely underweight the BMI should not be 0 or close to 0 [[Source]](https://www.webmd.com/a-to-z-guides/body-mass-index-bmi-for-adults)

# In[ ]:


print(data[data.BMI == 0].shape[0])
print(data[data.BMI == 0].index.tolist())
print(data[data.BMI == 0].groupby('Outcome')['Age'].count())


# 5.Insulin:
# ok so turns out in some rare cases a person can have zero insulin but they almost definitely have diabetes, which doesn't seem to be the case as per the data since 236 cases have insulin value 0 but are classified to not having diabetes

# In[ ]:


print(data[data.Insulin == 0].shape[0])
print(data[data.Insulin == 0].index.tolist())
print(data[data.Insulin == 0].groupby('Outcome')['Age'].count())


# Skin thickness level and Insulin values have a large number of cases with 0 values whereas Glucose, BP and BMI have much fewer cases with 0 values

# There are a couple of ways I can handle these invalid data values:
# 
# 1. Ignore/remove these casses - this may not work in the Skin Thickness and Insulin levels have large number of such invalid data points, removing those would leave me with very little data in an already small dataset. This may work for bmi, glucose and BP invalid data points
# 2. put average/mean values - this may not work out in all cases either, e.g. blood pressure, the blood pressure may be correlated to the diabetes therefore putting an average value for BP may provide a wrong signal to the model or reduce its predictive value
# 3. not using those features for the classification algorthm - this may work, perhaps in the case of skin thickness.
# 
# Lets also visualize these features to see how the Outcome is related to each of them

# ### Visualizing the different columns wrt the classes

# 1. Blood Pressure

# In[ ]:


plt.figure(figsize=(14,3))
bp_pivot = data.groupby('BloodPressure').Outcome.mean().reset_index()
sns.barplot(bp_pivot.BloodPressure, bp_pivot.Outcome)
plt.title('% chance of being diagnosed with diabetes by blood pressure reading')
plt.show()

plt.figure(figsize=(14,3))
bp_pivot = data.groupby('BloodPressure').Outcome.count().reset_index()
sns.distplot(data[data.Outcome == 0]['BloodPressure'], color='turquoise', kde=False, label='0 Class')
sns.distplot(data[data.Outcome == 1]['BloodPressure'], color='coral', kde=False, label='1 Class')
plt.legend()
plt.title('count # of people with blood pressure values')
plt.show()


# 2.Plasma Glucose Level

# In[ ]:


plt.figure(figsize=(20,5))
glucose_pivot = data.groupby('Glucose').Outcome.mean().reset_index()
sns.barplot(glucose_pivot.Glucose, glucose_pivot.Outcome)
plt.title('% chance of being diagnosed with diabetes by Glucose reading')
plt.show()

plt.figure(figsize=(14,3))
glucose_pivot = data.groupby('Glucose').Outcome.count().reset_index()
sns.distplot(data[data.Outcome == 0]['Glucose'], color='turquoise', kde=False, label='0 Class')
sns.distplot(data[data.Outcome == 1]['Glucose'], color='coral', kde=False, label='1 class')
plt.legend()
plt.title('count # of people with Glucose values')
plt.show()


# As expected the glucose feature seems to be highly correlated to the chances of getting diabetes

# In[ ]:


plt.figure(figsize=(20,5))
BMI_pivot = data.groupby('BMI').Outcome.mean().reset_index()
sns.barplot(BMI_pivot.BMI, BMI_pivot.Outcome)
plt.title('% chance of being diagnosed with diabetes by BMI reading')
plt.show()

plt.figure(figsize=(14,3))
BMI_pivot = data.groupby('BMI').Outcome.count().reset_index()
sns.distplot(data[data.Outcome == 0]['BMI'], color='turquoise', kde=False, label='Class 0')
sns.distplot(data[data.Outcome == 1]['BMI'], color='coral', kde=False, label='Class 1')
plt.legend()
plt.title('count # of people with BMI values')
plt.show()


# In[ ]:


plt.figure(figsize=(14,3))
Insulin_pivot = data.groupby('Insulin').Outcome.mean().reset_index()
sns.barplot(Insulin_pivot.Insulin, Insulin_pivot.Outcome)
plt.title('% chance of being diagnosed with diabetes by Insulin reading')
plt.show()

plt.figure(figsize=(14,3))
Insulin_pivot = data.groupby('Insulin').Outcome.count().reset_index()
sns.distplot(data[data.Outcome == 0]['Insulin'], color='turquoise', kde=False, label='Class 0')
sns.distplot(data[data.Outcome == 1]['Insulin'], color='coral', kde=False, label='Class 1')
plt.legend()
plt.title('count # of people with Insulin values')
plt.show()


# In[ ]:


plt.figure(figsize=(14,3))
SkinThickness_pivot = data.groupby('SkinThickness').Outcome.mean().reset_index()
sns.barplot(SkinThickness_pivot.SkinThickness, SkinThickness_pivot.Outcome)
plt.title('% chance of being diagnosed with diabetes by skin thickness reading')
plt.show()

plt.figure(figsize=(14,3))
SkinThickness_pivot = data.groupby('SkinThickness').Outcome.count().reset_index()
sns.distplot(data[data.Outcome == 0]['SkinThickness'], color='turquoise', kde=False, label='Class 0')
sns.distplot(data[data.Outcome == 1]['SkinThickness'], color='coral', kde=False, label='Class 1')
plt.legend()
plt.title('count # of people with Skin thickness values')
plt.show()


# ## 2.3 Data Selection and Model Fitting

# for the time being I am not removing the outlier in the data specifically BP, insulin and glucose because they are reletively few of them in the data. I see that these are the 3 features which seem to be effecting the classification the most, given this I will use them for the initial models, without any regularization. I will also be normalizing all the values since they are numeric and vary in their min and max values. Finally I am going to split the data into an 80-20 split for train and test sets and perform 5-fold cross validation of the training data to pic the best classifier for the job which will then be used for getting test predictions. I am using 5 classifiers that I know the theory behind.

# In[ ]:


from sklearn import linear_model as lm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.neighbors import KNeighborsClassifier as knnc
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, confusion_matrix


# In[ ]:


data_mod = data[(data.BloodPressure != 0) & (data.BMI != 0) & (data.Glucose != 0)]
train, test = train_test_split(data_mod, test_size=0.2)
print(data_mod.shape)
print(train.shape)
print(test.shape)


# In[ ]:


features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness'            , 'BMI', 'Age', 'Insulin', 'DiabetesPedigreeFunction']
target = 'Outcome'
classifiers = [
    knnc(),
    dtc(),
    SVC(),
    SVC(kernel='linear'),
    gnb()
]
classifier_names = [
    'K nearest neighbors',
    'Decision Tree Classifier',
    'SVM classifier with RBF kernel',
    'SVM classifier with linear kernel',
    'Gaussian Naive Bayes'
]


# In[ ]:


for clf, clf_name in zip(classifiers, classifier_names):
    cv_scores = cross_val_score(clf, train[features], train[target], cv=5)
    
    print(clf_name, ' mean accuracy: ', round(cv_scores.mean()*100, 3), '% std: ', round(cv_scores.var()*100, 3),'%')


# In[ ]:


final_model_smv_lin = SVC(kernel='linear').fit(train[features], train[target])
final_model_gnb = gnb().fit(train[features], train[target])


# In[ ]:


y_hat_svm = final_model_smv_lin.predict(test[features])
y_hat_gnb = final_model_gnb.predict(test[features])

print('test accuracy for SVM classifier with a linear kernel:'      , round(accuracy_score(test[target], y_hat_svm)*100, 2), '%')
plt.title('Confusion matrix for SVM classifier with a linear kernel')
sns.heatmap(confusion_matrix(test[target], y_hat_svm), annot=True, cmap="YlGn")
plt.xlabel('Predicted classes')
plt.ylabel('True Classes')
plt.show()

print('test accuracy for Gaussian naive bayes classifier:',       round(accuracy_score(test[target], y_hat_gnb)*100, 2),'%')
plt.title('confusion matrix for Gaussian naive bayes classifier')
sns.heatmap(confusion_matrix(test[target], y_hat_gnb), annot=True, cmap="YlGn")
plt.show()


# ## 3. Result
# 
# As shown above, the SVM linear model does seem to do much better from an accuracy perspective for the data. It also has fewer false positives than the Naive Bayes model although higher true negative predictions. <br>
# We could also choose based on if we want the model to err on the side of caution when predicting someone will have diabetes so that they take preventive care or be more cautious when predicting diabetes diagnosis. This will depend on what application this classifier.

# In[ ]:




