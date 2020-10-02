#!/usr/bin/env python
# coding: utf-8

# <h1 align='center'>CREDIT CARD FRAUD DETECTION</h1>
# <h3>Date: 21-05-2019</h3>
# 
# Being a part of the Kaggle community has boosted my enthusiasm for DataScience. I have learned a lot through Kaggle in recent times. This is my <b>first</b> kernel on Kaggle, and I want to work on a <font color='green'>Credit Card Fraud Detection problem</font>.
# ***
# 
# If there are any recommendations you would like to see in this notebook of mine, please drop it in the comment section below, as it will be greatly appreciated. If you like this notebook, please feel free to <font color='blue'>UPVOTE</font> and/or leave a <font color='blue'>COMMENT</font> below.
# 
# ***
# 
# 
# <h2>Overview</h2>
# In this kernel, we will deal with the <font color='green'>Credit Card Fraud Detection</font> dataset which is very popular for its <b>high imbalance</b>. We will use various algorithms such as <font color='red'>NearMiss</font>, <font color='red'>SMOTE</font> to deal with the <b>high imbalance</b> in the dataset. 
# 
# 
# <h2>Aim</h2>
# <ul>
# <li>Check for the extent of imbalance in our dataset.</li>
# <li>Use several sampling algorithms like <font color='red'>SMOTE</font> and <font color='red'>NearMiss</font> to deal with the imbalance.</li>
# <li>Use machine learning classification models to predict whether a transcation was <b>fradulent</b> or <b>non-fraudulent</b></li>
# </ul>

# ## Table of contents
# ***
# - [Part 1: Problem Definition](#problem_definition)
# - [Part 2: Importing Necessary Modules](#import_libraries)
#     - [2a. Libraries](#import_libraries)
#     - [2b. Load datasets](#load_data)
#     - [2c. A Glance at the dataset](#glance_dataset)
# - [Part 3: Mini EDA](#eda)
#     - [3a. Class variable](#class)
#     - [3b. Time and Amount](#time_amount)
# - [Part 4: Preprocessing](#preprocess)
#     - [4a. Scaling](#scale)
#     - [4b. Splitting of Data](#split_data)
#       -  [i Seperating the Independent variables from the Dependent variable](#seperate)
#       -  [ii Splitting the Dataset](#split)
#       -  [iii Resampling](#resample)
# - [Part 5: Ways of Combating Imbalance in Classes](#resample)
# - [Part 6: Using the Imbalanced Dataset on Classification Models](#imbalance_data_on_classifier)
# - [Part 7: Classification Metrics to be used](#metrics)
# - [Part 8: Applying Sampling Techniques to the Dataset](#sampling_techniques)
#     - [8a. Random Undersampling](#random_undersampling)
#       -  [i Before Cross Validation](#before_cv)
#       -  [ii During Cross Validation](#during_cv)
#     - [8b. NearMiss](#nearmiss)
#       -  [i Before Cross Validation](#nearmiss_before_cv)
#       -  [ii During Cross Validation](#nearmiss_during_cv)
#     - [8c. SMOTE](#smote)  
#       -  [i Before Cross Validation](#smote_before_cv)
#       -  [ii During Cross Validation](#smote_during_cv)

# <a id='problem_definition'></a>
# <h1>1. Problem Definition</h1>
# 
# The datasets contains transactions made by credit cards by European cardholders. Pricipal Component Analysis was performed on the original dataset, and features V1, V2, ..., V28 are the first 28 principal components obtained from the dataset. We know nothing about what features V1,..,V28 represent due to privacy reasons. The Time and the Amount features are the only features which are yet to be transformed by the PCA. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# 
# Most of the transcations were <b>non-fraudulent</b>, with the <b>fraudulent</b> ones being a miniature of the total transactions.

# <a id='import_libraries'></a>
# <h1>Part 2: Import Necessary Libraries and Dataset</h1>
# 
# ***
# 
# <h2>2a. Import the libaries</h2>

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import time
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score, auc, f1_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id='load_dataset'></a>
# <h2>2b. Loading the Dataset</h2>

# In[3]:


data = pd.read_csv("../input/creditcard.csv")


# The above is the dataset we are required to work on. It is required of us to use this dataset to create a model which will predict whether a transcation was fraudulent or not.

# <a id='glance_dataset'></a>
# <h2>2c. Glance at the Dataset</h2>

# In[4]:


#view the columns of the dataset
data.columns


# In[5]:


#view the first 5 samples of the dataset
data.head()


# In[6]:


#view the last 4 samples of the dataset
data.tail(4)


# In[7]:


#pick any 3 samples of the dataset
data.sample(3)


# In[8]:


data.info()


# We can see that the dataset has 284807 samples and 31 columns (30 features & 1 target variable), with each feature having all its values as floats; the target variable - Class - has its values as integers only, as we are to predict whether a transcation was non-fraudulent 0 or fraudlent 1.
# 
# We can also see that there are no missing values in any of the columns.

# In[9]:


#no missing values
max(data.isnull().sum())


# If we had missing values in our dataset, we colud have replaced them by any of the following techniques:
# 
# 1. Dropping the samples which had missing values in any of their entries.
# 2. Replacing by 0.
# 3. Replacing by the mean, median or mode of the feature in which the missing value fall into.
# 4. Forward/Backward filling using the pandas library.
# 5. Using regression models such as Linear Regression or K-Nearest Neighbors to predict the missing values.

# <a id='eda'></a>
# <h1>Part 3: Mini EDA</h1>

# In[10]:


#have a view of the features in the dataset
data.columns


# <a id='class'></a>
# <h2>3a. Class variable</h2>

# Since non-fraudlent transcations were labeled 0, and fraudlent ones 1, we can group the transactions that were fradulent as 'fraud', and those that were non-fraudlent as 'non_fraud'.

# In[11]:


fraud = data[data.Class==1]
non_fraud = data[data.Class==0]

#fraction of frauds
frac_of_fraud = len(fraud)/len(data)
#fraction of non-frauds
frac_of_non_fraud = len(non_fraud)/len(data)

print('Percentage of Frauds: {}%'.format(round(frac_of_fraud*100, 2)))
print('Percentage of Non-Frauds: {}%'.format(round(frac_of_non_fraud*100, 2)))


# From the percentages calculated, we can see that there is a very large disparity between the number of fraudulent transactions and non-fraudulent ones. If we feed this dataset into our predictive models, we will get errors as our algorithms will be bound to classify most transactions as non-fraudulent.

# In[12]:


#visualize the imbalance with a bar chart
plt.title('Distribution of Frauds', fontdict={'size' : 16, 'color':'brown'})
sns.countplot(x='Class', data=data)
labels = ['Non-Fraud', 'Fraud']   #to label the plot
vals = [0, 1]   #to put the labels right

plt.xticks(vals, labels)
plt.xlabel('Class', fontdict={'size' : 14, 'color' : 'green'})
plt.ylabel('Number of transactions', fontdict={'size' : 12, 'color':'green'})


# <a id='time_amount'></a>
# <h2 id=''>3b. Time and Amount features</h2>

# In[13]:


fig, ax = plt.subplots(figsize=(16, 6), nrows=1, ncols=2)

ax1 = sns.distplot(data['Time'], color='brown', ax=ax[0])
ax2 = sns.distplot(data['Amount'], ax=ax[1])

ax1.set_title('Distribution of Transaction Time')
ax2.set_title('Distribution of Transaction Amount')


# From the above plots, we can see that the Time, the Amount and the Class columns are skewed because they are all imbalanced.
# 
# ***
# 
# <h3><b>Skewness</b></h3>
# 
# ***
# <b>Skewness</b> in the field of statistics can be said to be a measure of the imbalance and asymmetry of a data from its mean of distribution. Data is said to be skewed when its mean is not equal to its mode i.e when <b>mean > median</b> or when <b>median > mean</b>. In contrast, whenever data has its mean <b>equal</b> to its median, data is said to be <b>normally distributed</b>. Therefore, <b>skewed</b> data is one which is not <b>normally distributed</b>.
# <ul>
#                     <p><b>Skewness = 3 (mean - median) / standard deviation</b></p>
#     </ul>
# <h5>Types of Skew</h5>
# <p><b>Positive Skew</b>: Mean > median</p>
# <p><b>Negative Skew</b>: Median > mean</p>
# <p><b>Zero Skew</b>: Mean = median (normal distribution)</p>

# In[14]:


#skewness of the Time column
print('The Transaction Time has a skewness of {}'.format(round(data.Time.skew(), 4)))

#skewness of the Amount column
print('The Transaction Amount has a skewness of {}'.format(round(data.Amount.skew(), 4)))

#skewness of the Class column
print('The Class - Target Variable - has a skewness of {}'.format(round(data.Class.skew(), 4)))


# We see that Time is negatively skewed(approximately 0), while Amount and Class are highly postively skewed.

# <a id='preprocess'></a>
# <h1>Part 4: Preprocessing</h1>
# 
# <a id='scale'></a>
# <h2>4a. Scaling</h2>
# 
# ***
# 
# Feature scaling is a key concept of machine learning models. Most datasets contain features often varying in magnitude and unit. Some machine learning models are not affected by this varying magnitudes and units; however, most machine learning models are affected by this varaiation since they make use of several distance metrics to calculate the distance between data points. 
# 
# Here, the Time and the Amount features are not in scale with columns V1, V2, ..., V28, and so we have to scale them both.
# 
# There are multiple ways of doing feature scaling. 
# <ul>
#     <li type=1><b>StandardScaler</b>
#         <ul>
#             <li>Assumes the data is normally distributed within each feature.</li>
#             <li>A normally distributed feature have zero skewness, its mean always equals its median, and if the feature is unimodal, then its mode equals the mean and median. </li>
#             <li>Scales the data so that it has mean 0 and variance of 1.</li>
#         </ul>
#         
#    </li>
#     <li type=1><b>MinMaxScaler</b>
#         <ul>
#             <li>Scales the data so that it ranges from 0 to 1 (-1 to 1 if there are negative values).</li>
#             <li>Works better if the data is not Gaussian, or the standard deviation is small. </li>
#             <li>Sensitive to outliers.</li>
#         </ul>
#    </li>
#     <li type=1><b>RobustScaler</b>
#         <ul>
#             <li>Similar to the MinMaxScaler, but uses the interquartile range for scaling.</li>
#             <li>Its use of the interquartile range makes it robust to outliers. </li>
#             <li>This means it uses less of the data for scaling, and as such, is more suitable when there are outliers in the data.</li>
#         </ul>
#    </li>
#     <li type=1><b>Unit Vector</b>
#         <ul>
#             <li>Ranges from 0 to 1.</li>
#             <li>Makes each feature to be a vector of unit length.</li>
#         </ul>
#    </li>
#  </ul>
#  
#  ***
#  I will make use of the RobustScaler to scale both the Time feature and the Amount feature. This is because the RobustScaler uses less of the data for scaling, and is more suitable when there are outliers in the data.

# In[15]:


#import the RobustScaler estimator class
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()

#apply RobustScaler to the Time and the Amount columns
data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])

#view the dataset to see the new look of the Time and Amount columns
data.head(5)


# <a id='split_data'></a>
# <h2>4b. Splitting of Data</h2>
# 
# <a id='seperate'></a>
# <h3>i. Seperating the Independent variables from the Dependent variable</h3>
# 
# Independent variables are the features in a dataset which are used to obtain the dependent variable. The dependent variable is the target variable which we are trying to find. Seperating the independent variables from the dependent variable in machine learning is a very <b>crucial</b> task.

# In[16]:


X = data.drop('Class', axis=1)  #independent variables
y = data['Class']   #dependent variable


# <a id='independent_dependent'></a>
# <h3>ii. Splitting the Dataset</h3>

# Before oversampling or undersmapling the data by using algorithms such as SMOTE and NearMiss, I will split the dataset into a training set and a test set. The purpose of this is to <b>train our model with the oversampled/undersampled training set, and test it on the original testing set</b>.

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, 
                                                    random_state=0)


# <a id='resample'></a>
# <h2>Part 5: Ways of Combating Imbalanced Classes in Machine Learning</h2>.
# 
# <ul>
#     <li type=1><b>Collecting more data</b>
#         <ul>
#             <li>A larger dataset might expose a different and perhaps more balanced perspective on the classes.</li>
#             <li>Never underestimate the power of more data. </li>
#         </ul>
#         
#    </li>
#     <li type=1><b>Undersampling</b>
#         <ul>
#             <li>Deletes copies of the <b>over-represented</b> class in the dataset.</li>
#         </ul>
#    </li>
#     <li type=1><b>Oversampling</b>
#         <ul>
#             <li>Adds copies of the <b>under-represented</b> class to the dataset.</li>
#         </ul>
#    </li>
#     <li type=1><b>Use of the NearMiss Algorithm</b>
#         <ul>
#             <li>NearMiss is a special undersampling technique.</li>
#             <li>Makes the majority class equal to the minority class.</li>
#         </ul>
#    </li>
#    <li type=1><b>Use of SMOTE</b>
#         <ul>
#             <li><b>SMOTE</b> stands for <b>Synthetic Minority Oversampling technique</b>.</li>
#             <li>Creates <b>synthetic(not duplicates)</b> samples of the minority class in the dataset.</li>
#         </ul>
#    </li>
#    <li type=1><b>Anomaly Detection</b>
#         <ul>
#             <li>This is the detection of rare events in a dataset.</li>
#             <li>Considers the minor class as the outliers class which might help you think of new ways to separate and classify samples.</li>
#         </ul>
#    </li>
#  </ul>
#  
#  
#  Further Readings: 8 Tactics to Combat Imbalanced Classes in Your Machine Learning Dataset by Jason Brownlee from Machine Learning Mastery https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/

# Since we have been given the dataset we are to work on, we need not collect more data. In this kernel, I will:
# 
# 1. Perform Random undersampling.
# 2. Use the NearMiss algorithm to perform undersampling.
# 3. Use SMOTE to perform oversampling -- create synthetics samples of the minority class.
# 
# <b>In future updates, I will use Anomaly Detection on this dataset</b>.
# 
# 
# 
# But before performing any form of undersampling and oversampling, I will train our classifiers with the imbalanced dataset in order to see how poorly machine learning models perform when fed with imbalanced data.

# #classification models that I will use in this dataset
# <ul>
# <li>Logistic Regression</li>
# <li>Decision Trees Classifier</li>
# <li>Random Forest Classifier</li>
# <li>Support Vector Classifier</li>
# </ul>

# In[18]:


#importing the classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# <a id='imbalance_data_on_classifier'></a>
# <h2>6. Using the Imbalanced Dataset on the Classifiers</h2>
# 
# Here, I will train the classifiers with the imbalanced dataset.

# In[19]:


models = {
    'Logistic Regression' : LogisticRegression(), 
    'Naive Bayes' : GaussianNB(),
     #'Support Vector Classifier' : SVC(),  computationally expensive to run on the whole dataset
    #'Decision Tree Classifier' : DecisionTreeClassifier()   computationally expensive
    
}


# In[20]:


for name, model in models.items():
    t0 = time.time()  #start time
    model.fit(X_train, y_train)
    accuracy = cross_val_score(model, X_train, y_train, cv=5).mean()
    t1 = time.time()  #stop time
    print(name, 'Score: {}%'.format(round(accuracy, 2)*100))
    print('Computation Time: {}s\n'.format(round(t1-t0, 2)))
    print('*'*80)
    print()


# From the above, we see that we get unexpectedly high accuracies. This is due to the high imbalance in the classes. Since there is a very large imbalance between fraud 1(0.17%) and non-fraud 0(99.83%), the models above will tend to classify all the transcations as non-fraud, and as such, give us way too high accuracies on the training set and the test set.

# <a id='metrics'></a>
# <h2>Part 7: Metrics</h2>
# 
# There are a number of metrics that can be used to check the performance of classification problems;
# 
# <h4>1. Accuracy</h4>
# <p>Accuracy is the total number of correct predictions made by the model all over the total number of predicted labels. Accuracy <b>should not</b> be used when there is a high imbalance in the dataset, as the model will tend to classify all the data points in the test data as the majority class.</p>
#     <ul>
#     <p>Accuracy = (Total number of correct predictions) / (Total number of predicted labels)</p>
#     <p>Accuracy = (TP + TN) / (Total)</p>
#     </ul>
#     
#  
# <h4>2. Precision</h4>
# <p>Precision is the proportion of the fradulent transactions to the total number of transactions that were predicted to be fraudulent.</p>
#     <ul>
#     <p>Precision = TP / (TP + FP)</p>
#     </ul>
# 
# 
# <h4>3. Recall (Sensitivity / True Postive Rate)</h4>
# <p>Recall is the proportion of transactions that were predicted correctly to be fradulent of the total number of transactions that were actually fraudulent.</p>
#     <ul>
#     <p>Recall = TP / (TP + FN)</p>
#     </ul>
#     
#     
# <h4>4. Specificity (False Postive Rate)</h4>
# <p>This is the converse of Recall/Sensitivity. Specificity is the proportion of transactions that were predicted correctly to not be fradulent of the total number of transactions that were actually not fraudulent.</p>
#     <ul>
#     <p>Specificity = TN / (TN + FP)</p>
#     </ul>
#     
#     
# <b>Precision, Recall and Specifcity are one of the best metrics to consider when dealing with imbalanced data.</b>
# 
# It is always better to classify a non-fraudlent transaction as fradulent, than classify a fraudlent transaction as non-fraudlent...
# 
# 1. the reason is simply because on calling a non-fraud transaction fraud, you only make security agents investigate an individual who is innocent of fraud. However, after much investigation, the individual will be vindicated because he did not actually commit fraud, it was only your model that predicted it as fraud.
# 
# 2. on the other hand, it will not be good enough if our model classify a fraudulent transaction as non-fraudulent because this will make fraudsters get away with their crime, thereby increasing thye rate at which credit card fraud is committed. 
# 
# 
# From the look of the two points above, we can see that Recall is the best metric to consider. This is because a higher Recall indicates that the number of FN (transaction considered non-fraudulent, but is actually fraudulent) is being reduced to the minimum, which is our goal - try as much as possible not to classify non-fraudulent activities as fraudulent!!! 
# 
# However, on increasing recall, precision decreases. The metric we will consider here is <b>Recall</b>...we need a high recall as much as possible.
# 
# 
# Resources: 
# * https://medium.com/@ogundareoluwafemi2001/how-to-measure-the-performance-of-machine-learning-classification-models-8994ccf28047
# * <a href="https://towardsdatascience.com/precision-vs-recall-386cf9f89488">Precision vs Recall</a> 
# * https://en.wikipedia.org/wiki/Precision_and_recall

# <a id='sampling_techniques'></a>
# <h2>Part 8: Sampling the Dataset</h2>
# 
# One major misconception about sampling datasets with imbalanced classes is that sampling must be done before splitting the dataset into training and test set, or before cross validation. <b>Sampling before cross validation</b> is <b>wrong</b> because it could cause <b>data leakage</b> - making some copies of the same points ending up in both the training set and the test set. This would make the machine learning classifier <b>cheat</b> because the classifier will see some data points in the test set being identical to some other data points in the training set, and as such, will lead to overfitting, which will eventually cause inaccurate results.
# 
# Data sampling is advised to always be done <b>during splitting of data and/or cross validation</b>.

# <h2>8a. Undersampling</h2>
# 
# In this section of the kernel, I will perform an undersampling data technique known as <b>Random Undersampling</b>. The <b>Random Undersampling Technique</b> involves the removal of data points from the majority class so as to make the number of samples in the majority class to be <b>equal</b> to the number of samples in the minority class i.e a 50/50 ratio of both classes.

# <a id='before_cv'></a>
# <h3>i. Undersampling Before Cross Validation</h3>
# 
# Here, we will undersample our dataset before splitting it into training part and test part. This leads to data leakage, as the Random Sampling technique creates new copies of original data points. When this data is split, we end up having some data points in the training set being in the test set too. This makes our model(s) to cheat. 

# In[21]:


#since I do not know whether the classes are distributed in any unique pattern, I will shuffle the whole dataset
data = data.sample(frac=1, random_state=42)

#pick out the fraud and the non-fraud samples from the shuffled dataset
fraud = data[data.Class==1]
non_fraud = data[data.Class==0]

#print out the number of samples in fraud and non_fraud
print('Before Undersampling: ')
print('Number of Fraudulent Transactions: {}'.format(len(fraud)))
print('Number of Non-Fraudulent Transactions: {}'.format(len(non_fraud)))
print()

#making the non_fraud transactions(majority class) equal to the fraud transactions(minority class)
non_fraud = non_fraud.sample(frac=1)
non_fraud = non_fraud[:len(fraud)]

#the non_fraud transactions are now equal to the fraud transaction --- let's visualize
print('After Undersampling: ')
print('Number of Fraudulent Transactions: {}'.format(len(fraud)))
print('Number of Non-Fraudulent Transactions: {}'.format(len(non_fraud)))


#now join the fraud dataset to the non_fraud dataset
sampled_data = pd.concat([fraud, non_fraud], axis=0)
#shuffle the sampled_data to allow for random distribution of classes in the dataset
sampled_data = sampled_data.sample(frac=1, random_state=42)


# In[22]:


fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)

ax1 = sns.countplot(x='Class', data=data, ax=ax[0])
ax1.set_title('Distribution of Classes before Undersampling', color='brown')

ax2 = sns.countplot(x='Class', data=sampled_data, ax=ax[1])
ax2.set_title('Distribution of Classes after Undersampling', color='red')


# In[23]:


#a pie chart will also indicate a 50/50 ratio of fraud to non_fraud
patches, texts, autotexts = plt.pie(
    x=[len(fraud), len(non_fraud)], 
    labels=['Fraud', 'Non-Fraud'],
    explode = [0.012, 0.012],
    shadow=True,
    autopct='%.1f%%',
    radius=1.2,
    startangle=30
)

for text in texts:
    text.set_color('#22AA11')
    text.set_size(14)
for autotext in autotexts:
    autotext.set_color('red')


# From the visualizations, we can see that before undersampling, there was a large imbalance between the number of fradulent transactions and the number of non-fraudulent transactions.

# In[24]:


X_undersampled = sampled_data.drop('Class', axis=1)
y_undersampled = sampled_data['Class']


#splitting the dataset
X_train_undersampled, X_test_undersampled, y_train_undersampled, y_test_undersampled = train_test_split(X_undersampled, 
                                                                                                        y_undersampled, 
                                                                                                        stratify=y_undersampled, 
                                                                                                        test_size=0.25, 
                                                                                                        random_state=0)


# Looking at what we have done above, we can vividly see that we undersampled our data before splitting it into training set and test set. This is Sampling done wrong (cross validation done wrong too!!!). It leads to <b>overfitting</b>.

# In[25]:


#use the cross validation technique - StratifiedShuffleSplit - to perform GridSearch and to calculate the train and test scores
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=0)


#using the Gaussian Naive Bayes on the dataset
bayes = GaussianNB()
bayes.fit(X_train_undersampled, y_train_undersampled)
bayes_score = cross_val_score(bayes, X_train_undersampled, y_train_undersampled, cv=cv).mean()
bayes_predictions = bayes.predict(X_test_undersampled)
bayes_precision_score = precision_score(y_test_undersampled, bayes_predictions)
bayes_recall_score = recall_score(y_test_undersampled, bayes_predictions)
bayes_auc = roc_auc_score(y_test_undersampled, bayes_predictions)





#parameters to search for using GridSearchCV

#Logistic Regression
logReg_params = {'penalty' : ['l1', 'l2'], 'C' : [0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.5, 1.7, 2.0]}

#Support Vector Classifier
svc_params = {'C' : [0.1, 1, 10], 'gamma' : [1, 0.1, 0.01, 0.001], 'kernel' : ['linear', 'rbf'] }

#Decision Tree Classifier
dtree_params = {'criterion' : ['gini', 'entropy'], 'max_depth' : [3, 4, 5]}




#GridSearch on Logistic Regression
logReg_grid = GridSearchCV(LogisticRegression(), logReg_params, refit=True, verbose=0, cv=cv, scoring='accuracy')
logReg_grid.fit(X_train_undersampled, y_train_undersampled)
logReg = logReg_grid.best_estimator_

logReg_score = cross_val_score(logReg, X_train_undersampled, y_train_undersampled, cv=cv).mean()
logReg_predictions = logReg.predict(X_test_undersampled)
logReg_precision_score = precision_score(y_test_undersampled, logReg_predictions)
logReg_recall_score = recall_score(y_test_undersampled, logReg_predictions)
logReg_auc = roc_auc_score(y_test_undersampled, logReg_predictions)


#GridSearch on Support Vector Classifier
svc_grid = GridSearchCV(SVC(), svc_params, refit=True, verbose=0, cv=cv, scoring='accuracy')
svc_grid.fit(X_train_undersampled, y_train_undersampled)
svc = svc_grid.best_estimator_

svc_score = cross_val_score(svc, X_train_undersampled, y_train_undersampled, cv=cv).mean()
svc_predictions = svc.predict(X_test_undersampled)
svc_precision_score = precision_score(y_test_undersampled, svc_predictions)
svc_recall_score = recall_score(y_test_undersampled, svc_predictions)
svc_auc = roc_auc_score(y_test_undersampled, svc_predictions)


#GridSearch on Decision Tree Classifier
dtree_grid = GridSearchCV(DecisionTreeClassifier(), dtree_params, refit=True, verbose=0, cv=cv, scoring='accuracy')
dtree_grid.fit(X_train_undersampled, y_train_undersampled)
dtree = dtree_grid.best_estimator_

dtree_score = cross_val_score(dtree, X_train_undersampled, y_train_undersampled, cv=cv).mean()
dtree_predictions = dtree.predict(X_test_undersampled)
dtree_precision_score = precision_score(y_test_undersampled, dtree_predictions)
dtree_recall_score = recall_score(y_test_undersampled, dtree_predictions)
dtree_auc = roc_auc_score(y_test_undersampled, dtree_predictions)


# In[26]:


model_names = ['Logistic Regression', 'Gaussian Naive Bayes', 'Support Vector Classifier', 'Decision Tree Classifier']


# In[27]:


for name in model_names:
    if name == 'Logistic Regression':
        print(name, 'Scores: \n')
        print('Accuracy: {}'.format(round(logReg_score, 2)))
        print('Precision: {}'.format(round(logReg_precision_score, 2)))
        print('Recall: {}'.format(round(logReg_recall_score, 2)))
        print('AUC: {}\n'.format(round(logReg_auc, 2)))
        print('*'*90)
    elif name == 'Gaussian Naive Bayes':
        print(name, 'Scores: \n')
        print('Accuracy: {}'.format(round(bayes_score, 2)))
        print('Precision: {}'.format(round(bayes_precision_score, 2)))
        print('Recall: {}'.format(round(bayes_recall_score, 2)))
        print('AUC: {}\n'.format(round(bayes_auc, 2)))
        print('*'*90)
    elif name == 'Support Vector Classifier':
        print(name, 'Scores: \n')
        print('Accuracy: {}'.format(round(svc_score, 2)))
        print('Precision: {}'.format(round(svc_precision_score, 2)))
        print('Recall: {}'.format(round(svc_recall_score, 2)))
        print('AUC: {}\n'.format(round(svc_auc, 2)))
        print('*'*90)
    elif name == 'Decision Tree Classifier':
        print(name, 'Scores: \n')
        print('Accuracy: {}'.format(round(dtree_score, 2)))
        print('Precision: {}'.format(round(dtree_precision_score, 2)))
        print('Recall: {}'.format(round(dtree_recall_score, 2)))
        print('AUC: {}\n'.format(round(dtree_auc, 2)))
        print('*'*90)


# In[28]:


print('Random Undersampling Before Data Split')
print('.'*60)
print()

logReg_conf_matrix = confusion_matrix(y_test_undersampled, logReg_predictions)
bayes_conf_matrix = confusion_matrix(y_test_undersampled, bayes_predictions)
svc_conf_matrix = confusion_matrix(y_test_undersampled, svc_predictions)
dtree_conf_matrix = confusion_matrix(y_test_undersampled, dtree_predictions)


fig, ax = plt.subplots(figsize=(10, 8), nrows=2, ncols=2)

fig.subplots_adjust(hspace=0.6, wspace=0.4)  #adjust the spaces between the subplots
tick_labels = ['non_fraud', 'fraud']  #labels for the xticks and yticks


#Logistic Regression Confusion Matrix
ax1 = sns.heatmap(logReg_conf_matrix, cbar=False, yticklabels=tick_labels, xticklabels=tick_labels, annot=True, ax=ax[0][0])
ax1.set_title('Logistic Regression Confusion Matrix', color='red')
ax1.set_ylabel('Actual Labels', size=8)
ax1.set_xlabel('Predicted Labels', size=8)


#Gaussian Naive Bayes Confusion Matrix
ax2 = sns.heatmap(bayes_conf_matrix, cbar=False, yticklabels=tick_labels, xticklabels=tick_labels, annot=True, ax=ax[0][1])
ax2.set_title('Gaussian Naive Bayes Confusion Matrix', color='red')
ax2.set_ylabel('Actual Labels', size=8)
ax2.set_xlabel('Predicted Labels', size=8)


#Support Vector Classifier Confusion Matrix
ax3 = sns.heatmap(svc_conf_matrix, cbar=False, yticklabels=tick_labels, xticklabels=tick_labels, annot=True, ax=ax[1][0])
ax3.set_title('Support Vector Classifier Confusion Matrix', color='red')
ax3.set_ylabel('Actual Labels', size=8)
ax3.set_xlabel('Predicted Labels', size=8)


#Decision Tree Confusion Matrix
ax4 = sns.heatmap(dtree_conf_matrix, cbar=False, yticklabels=tick_labels, xticklabels=tick_labels, annot=True, ax=ax[1][1])
ax4.set_title('Decision Tree Classifier Confusion Matrix', color='red')
ax4.set_ylabel('Actual Labels', size=8)
ax4.set_xlabel('Predicted Labels', size=8)


# In order to make this kernel short and straight-forward, I will make use of only <b>Logistic Regression</b> as my model till the end of the kernel. I will also apply the NearMiss and SMOTE algorithms on the dataset. In future updates, I will do for the Gaussian Naive Bayes, Support Vector Classifier and the Decision Tree Classifier respectively.

# <a id='during_cv'></a>
# <h3>ii. Undersampling During Cross Validation (Undersampling Done Right)</h3>
# 
# Here, we will undersample our dataset after splitting it into training data and test data. This is the right way to do random undersampling.

# In[29]:


cv = StratifiedShuffleSplit(n_splits=10, test_size=0.25, random_state=0)

accuracy = []
precision = []
recall = []
auc = []



#remember X_train and y_train are from the original data
#join X_train and y_train; join X_test and y_test
#then undersample only the resulting dataframe of X_train and y_train; and then test using the dataframe resulting from X_test and y_test
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)


#pick out fraud and non_fraud from the train_data
fraud = train_data[train_data.Class==1]
non_fraud = train_data[train_data.Class==0]

#make fraud equal to non_fraud by picking out random samples from non_fraud and making it equal to fraud
non_fraud = non_fraud.sample(n=len(fraud))

#number of fraud and non_fraud
print('Number of Fraudulent Transactions: {}'.format(len(fraud)))
print('Number of Non-Fraudulent Transactions: {}'.format(len(non_fraud)))

#concatenate fraud and non_fraud and call the dataframe undersampled_train_data
undersampled_train_data = pd.concat([fraud, non_fraud], axis=0)
undersampled_train_data = undersampled_train_data.sample(frac=1)  #resample your data to allow for a kind of random/even distribution

#we will use the whole undersampled_train_data to train our model, and test it using the test_data that isn't undersampled!
train_X = undersampled_train_data.drop('Class', axis=1)
train_y = undersampled_train_data['Class']

#split the test_data
test_X = test_data.drop('Class', axis=1)
test_y = test_data['Class']


#use gridsearch to search for parameters
logReg_params = {'penalty' : ['l1', 'l2'], 'C' : [0.4, 0.6, 0.8, 1.0, 1.3, 1.5, 1.7, 2.0]}

log_grid = GridSearchCV(LogisticRegression(random_state=0), logReg_params, refit=True, verbose=0)
log_grid.fit(train_X, train_y)   #training with the undersampled data
logReg = log_grid.best_estimator_
#testing with the original dataset which was not undersampled
predict = log_grid.predict(test_X)


#scores
logReg_score = cross_val_score(logReg, train_X, train_y, cv=cv).mean()

logReg_precision_score = precision_score(test_y, predict)
logReg_recall_score = recall_score(test_y, predict)
logReg_auc = roc_auc_score(test_y, predict)


# In[30]:


print('Logistic Regression Results for Random Undersampling After Data Split\n')
print('*'*78)
print()
print('Accuracy: {}'.format(round(logReg_score, 2)))
print('Precision: {}'.format(round(logReg_precision_score, 2)))
print('Recall: {}'.format(round(logReg_recall_score, 2)))
print('AUC: {}'.format(round(logReg_auc, 2)))


# In[31]:


print('Random Undersampling After Data Split')
print('.'*60)
print()

ax1 = sns.heatmap(confusion_matrix(test_y, predict), cbar=False, yticklabels=tick_labels, xticklabels=tick_labels, annot=True)
ax1.set_title('Logistic Regression Confusion Matrix', color='red')
ax1.set_ylabel('Actual Labels', size=8)
ax1.set_xlabel('Predicted Labels', size=8)


# On comapring the results we just obtained with the results of Linear Regression when Random Undersampling was performed before the split of our dataset, we see that the results of Random Undersampling after the split of our data is more impressive than that of Random Undersampling before the data split - the precision score of the Logistic Regression classifier reduced while the high recall value was still maintained. There was also an increase in the AUC.

# In[ ]:





# In[ ]:





# In[32]:


#function to display the results for the results of NearMiss and SMOTE before cv and during cv
def show_metrics(title, accuracy, precision, recall, f1, auc):
    print(title, 'Results:\n')
    print('Accuracy: {}'.format(round(np.mean(accuracy), 2)))
    print('Precision: {}'.format(round(np.mean(precision), 2)))
    print('Recall: {}'.format(round(np.mean(recall), 2)))
    print('F1-Score: {}'.format(round(np.mean(f1), 2)))
    print('AUC: {}'.format(round(np.mean(auc), 2)))
    print()
    print('*'*80)
    print()


# <a id='nearmiss'></a>
# <h2>8b. NearMiss</h2>
# 
# NearMiss is an undersampling technique used to deal with imbalanced dataset. It resamples the majority class and makes it equal to the number of samples in the minority class. It uses the <font color='green'>Nearest Neigbors</font> technique to do this.

# <a id='nearmiss_before_cv'></a>
# <h3>i. Before Cross Validation</h3>
# 
# This is <font color='red'>NearMiss done wrong</font>. Here, the NearMiss algorithm is applied to the dataset before Cross Validation is carried out.

# In[33]:


#cross validation technique to use during NearMiss
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.20, random_state=0)


# In[34]:


nearmiss_before_cv_accuracy = []         #accuracy score
nearmiss_before_cv_precision = []        #precision score
nearmiss_before_cv_recall = []           #recall
nearmiss_before_cv_f1 = []               #f1 score
nearmiss_before_cv_auc = []              #auc --- area under curve

nearmiss_before_cv_conf_matrices = []    #list to append the confusion matrices obtained during CV


print('Distribution of Target Variable in Original Data: {}\n'.format(Counter(y)))   #distribution of fraud and non_fraud in original dataset
print('Distribution of Target Variable in Training Set of Original Data: {}\n'.format(Counter(y_train)))   #distribution of fraud and non_fraud in the training part of the original dataset


#remember that X_train and y_train are a split from the original dataset
X_nearmiss, y_nearmiss = NearMiss(random_state=0).fit_sample(X_train, y_train)   #applying NearMiss before CV, WRONG!!!

print('Distribution of Target Variable in Training Set of Original Data after NearMiss: {}\n'.format(Counter(y_nearmiss)))  #distribution of fraud and non_fraud must be equal in the training part of the original dataset after NearMiss is applied

for train, test in cv.split(X_nearmiss, y_nearmiss):   #CV begins here!
    #pipeline for imbalanced data -- from imblearn.pipeline import make_pipeline as make_make_pipeline_imb
    pipeline = make_pipeline_imb(LogisticRegression(random_state=42)) 
    nearmiss_before_cv_model = pipeline.fit(X_nearmiss[train], y_nearmiss[train])
    nearmiss_before_cv_predictions = nearmiss_before_cv_model.predict(X_nearmiss[test])
    
    #scoring metrics
    nearmiss_before_cv_accuracy.append(pipeline.score(X_nearmiss[test], y_nearmiss[test]))
    nearmiss_before_cv_precision.append(precision_score(y_nearmiss[test], nearmiss_before_cv_predictions))
    nearmiss_before_cv_recall.append(recall_score(y_nearmiss[test], nearmiss_before_cv_predictions))
    nearmiss_before_cv_f1.append(f1_score(y_nearmiss[test], nearmiss_before_cv_predictions))
    nearmiss_before_cv_auc.append(roc_auc_score(y_nearmiss[test], nearmiss_before_cv_predictions))
    
    #confusion matrices --- since the no. of splits is 10, I have 10 different confusion matrices
    nearmiss_before_cv_conf_matrices.append(confusion_matrix(y_nearmiss[test], nearmiss_before_cv_predictions))


# We see that the number of fraudulent transcations is now equal to the number of non-fraudulent transactions [369 : 369] after the NearMiss algorithm was applied on the training set of the original data.

# <a id='nearmiss_during_cv'></a>
# <h3>ii. During Cross Validation</h3>
# 
# This is <font color='red'>NearMiss done right</font>. Here, the NearMiss algorithm is applied to the dataset during Cross Validation.

# In[35]:


nearmiss_during_cv_accuracy = []
nearmiss_during_cv_precision = []
nearmiss_during_cv_recall = []
nearmiss_during_cv_f1 = []
nearmiss_during_cv_auc = []


nearmiss_during_cv_conf_matrices = []

#in order to get the values of the indices obtained during cv easily, I'll convert X_train and y_train into numpy arrays; I can also do this by using the pandas' 'iloc' method
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values


for train, test in cv.split(X_train, y_train):   #CV begins here!
    pipeline = make_pipeline_imb(NearMiss(random_state=0), LogisticRegression(random_state=42)) #NearMiss during CV, CORRECT!!!
    nearmiss_during_cv_model = pipeline.fit(X_train[train], y_train[train])
    nearmiss_during_cv_predictions = nearmiss_during_cv_model.predict(X_train[test])
    
    #scoring metrics
    nearmiss_during_cv_accuracy.append(pipeline.score(X_train[test], y_train[test]))
    nearmiss_during_cv_precision.append(precision_score(y_train[test], nearmiss_during_cv_predictions))
    nearmiss_during_cv_recall.append(recall_score(y_train[test], nearmiss_during_cv_predictions))
    nearmiss_during_cv_f1.append(f1_score(y_train[test], nearmiss_during_cv_predictions))
    nearmiss_during_cv_auc.append(roc_auc_score(y_train[test], nearmiss_during_cv_predictions))
    
    #confusion matrices --- since the no. of splits is 10, I must have 10 confusion matrices
    nearmiss_during_cv_conf_matrices.append(confusion_matrix(y_train[test], nearmiss_during_cv_predictions))


# In[36]:


print('RESULTS for the NearMiss Algorithm')
print('.'*80)
print()

show_metrics('NearMiss Before Cross Validation', nearmiss_before_cv_accuracy, nearmiss_before_cv_precision, nearmiss_before_cv_recall, nearmiss_before_cv_f1, nearmiss_before_cv_auc)
show_metrics('NearMiss During Cross Validation', nearmiss_during_cv_accuracy, nearmiss_during_cv_precision, nearmiss_during_cv_recall, nearmiss_during_cv_f1, nearmiss_during_cv_auc)


# From the above, we can see that running the NearMiss algorithm on our dataset before cross validation leads to high scores than running the algorithm during cross validation. This is due to the fact that there is always data leakage from the training set to the test set when sampling is done before cross validation, which makes our Logistic Regression Classifier to cheat, leading to overfitting, and hence, high scores. When you compare the scores gotten when NearMiss was performed before cross validation to when NearMiss was performed during cross validation, we see that the scores of NearMiss during cross validation was lower than the former.
# 
# Notice that when NearMiss was performed during cross validation, we get a very high Recall value to Precision value, which is still very much okay. The accuracy score we then obtain is the true accuracy of the Logistic Regression classifier.

# In[37]:


fig, ax = plt.subplots(figsize=(20, 3), nrows=1, ncols=5)

i = 0
print('NearMiss Before CV Confusion Matrices')
print('.'*60)
#the first 5 confusion matrices of the 10
for nearmiss_before_cv_conf_matrix in nearmiss_before_cv_conf_matrices[:5]:
    ax0 = sns.heatmap(nearmiss_before_cv_conf_matrix, cbar=False,xticklabels=tick_labels, yticklabels=tick_labels, ax=ax[i], annot=True, annot_kws={'size' : 15})
    #ax0.set_ylabel('Actual Labels')
    #ax0.set_xlabel('Predicted Labels')
    i = i + 1


# In[38]:


fig, ax = plt.subplots(figsize=(20, 3), nrows=1, ncols=5)

k = 0
print('NearMiss During CV Confusion Matrices')
print('.'*60)
#the first 5 confusion matrices of the 10
for nearmiss_before_cv_conf_matrix in nearmiss_during_cv_conf_matrices[:5]:
    ax1 = sns.heatmap(nearmiss_before_cv_conf_matrix, cbar=False, xticklabels=tick_labels, yticklabels=tick_labels, ax=ax[k], annot=True, annot_kws={'size' : 15})
    #ax1.set_ylabel('Actual Labels')
    #ax1.set_xlabel('Predicted Labels')
    k= k + 1


# In[ ]:





# In[ ]:





# <a id='smote'></a>
# <h2>8c. SYNTHETIC MINORITY OVERSAMPLING TECHNIQUE - SMOTE</h2>
# 
# <b>SMOTE</b> is an <b>oversampling</b> technique used for dealing with imbalanced data. <b>SMOTE</b> synthesises new minority instances between existing(real) minority instances. <font color='brown'>The new minority instances are not just copies of existing minority cases; instead, the algorithm takes samples of the feature space for each target class and its nearest neighbors, and generates new examples that combine features of the target case with features of its neighbors. This approach increases the features available to each class and makes the samples more general</font>.
# 
# Further Readings: https://docs.microsoft.com/en-us/azure/machine-learning/studio-module-reference/smote

# <a id='smote_before_cv'></a>
# <h3>i. SMOTE Before Cross Validation</h3>
# 
# This is <font color='red'>SMOTE done wrong</font>. Here, SMOTE is applied on the dataset before Cross Validation is carried out.

# In[39]:


#cross validation technique to use during SMOTE
kf = StratifiedKFold(n_splits=3, shuffle=False, random_state=0)


# In[40]:


smote_before_cv_accuracy = []         
smote_before_cv_precision = []        
smote_before_cv_recall = []           
smote_before_cv_f1 = []               
smote_before_cv_auc = []              

smote_before_cv_conf_matrices = []  


print('Distribution of Target Variable in Original Data: {}\n'.format(Counter(y)))   #distribution of fraud and non_fraud in original dataset
print('Distribution of Target Variable in Training Set of Original Data: {}\n'.format(Counter(y_train)))   #distribution of fraud and non_fraud in the training part of the original dataset


smote_X, smote_y = SMOTE(random_state=0).fit_sample(X_train, y_train)   #applying SMOTE before CV, WRONG!!!

print('Distribution of Target Variable in Training Set of Original Data after NearMiss: {}\n'.format(Counter(smote_y)))  #distribution of fraud and non_fraud must be equal in the training part of the original dataset after SMOTE is applied


for train, test in kf.split(smote_X, smote_y):   #CV begins here!
    pipeline = make_pipeline_imb(LogisticRegression(random_state=42)) 
    smote_before_cv_model = pipeline.fit(smote_X[train], smote_y[train])
    smote_before_cv_predictions = smote_before_cv_model.predict(smote_X[test])
    
    #classification metrics
    smote_before_cv_accuracy.append(pipeline.score(smote_X[test], smote_y[test]))
    smote_before_cv_precision.append(precision_score(smote_y[test], smote_before_cv_predictions))
    smote_before_cv_recall.append(recall_score(smote_y[test], smote_before_cv_predictions))
    smote_before_cv_f1.append(f1_score(smote_y[test], smote_before_cv_predictions))
    smote_before_cv_auc.append(roc_auc_score(smote_y[test], smote_before_cv_predictions))
    
    #confusion matrices --- since the no. of splits is 10, I have 10 different confusion matrices
    smote_before_cv_conf_matrices.append(confusion_matrix(smote_y[test], smote_before_cv_predictions))


# We see that the number of fraudulent transcations is now equal to the number of non-fraudlent transactions [213236 : 213236] after SMOTE was applied on the training set of the original data.

# <a id='smote_during_cv'></a>
# <h3>ii. During Cross Validation</h3>
# 
# This is <font color='red'>SMOTE done right</font>. Here, SMOTE is applied to the dataset during Cross Validation.

# In[41]:


smote_during_cv_accuracy = []
smote_during_cv_precision = []
smote_during_cv_recall = []
smote_during_cv_f1 = []
smote_during_cv_auc = []


smote_during_cv_conf_matrices = []

#no need to convert X_train, X_test, y_train and y_test to numpy arrays, as I have done that before
X_train = X_train
X_test = X_test
y_train = y_train
y_test = y_test


for train, test in kf.split(X_train, y_train):   #CV begins here!
    pipeline = make_pipeline_imb(SMOTE(random_state=0), LogisticRegression(random_state=42)) #SMOTE during CV, CORRECT!!!
    smote_during_cv_model = pipeline.fit(X_train[train], y_train[train])
    smote_during_cv_predictions = smote_during_cv_model.predict(X_train[test])
    
    #scoring metrics
    smote_during_cv_accuracy.append(pipeline.score(X_train[test], y_train[test]))
    smote_during_cv_precision.append(precision_score(y_train[test], smote_during_cv_predictions))
    smote_during_cv_recall.append(recall_score(y_train[test], smote_during_cv_predictions))
    smote_during_cv_f1.append(f1_score(y_train[test], smote_during_cv_predictions))
    smote_during_cv_auc.append(roc_auc_score(y_train[test], smote_during_cv_predictions))
    
    
    #confusion matrices --- since the no. of splits is 10, I must have 10 confusion matrices
    smote_during_cv_conf_matrices.append(confusion_matrix(y_train[test], smote_during_cv_predictions))


# In[42]:


print('RESULTS for SMOTE')
print('.'*80)
print()

show_metrics('SMOTE Before Cross Validation', smote_before_cv_accuracy, smote_before_cv_precision, smote_before_cv_recall, smote_before_cv_f1, smote_before_cv_auc)
show_metrics('SMOTE During Cross Validation', smote_during_cv_accuracy, smote_during_cv_precision, smote_during_cv_recall, smote_during_cv_f1, smote_during_cv_auc)


# Notice that Recall became higher than Precision when SMOTE was applied during Cross Validation.

# In[43]:


fig, ax = plt.subplots(figsize=(9, 3), nrows=1, ncols=3)

x = 0
print('NearMiss Before CV Confusion Matrices')
print('.'*60)
#the first 5 confusion matrices of the 10
for smote_before_cv_conf_matrix in smote_before_cv_conf_matrices[:5]:
    ax0 = sns.heatmap(smote_before_cv_conf_matrix, cbar=False, xticklabels=tick_labels, yticklabels=tick_labels, ax=ax[x], annot=True, annot_kws={'size' : 10})
    #ax0.set_ylabel('Actual Labels')
    #ax0.set_xlabel('Predicted Labels')
    x = x + 1


# In[44]:


fig, ax = plt.subplots(figsize=(9, 3), nrows=1, ncols=3)

y = 0
print('NearMiss Before CV Confusion Matrices')
print('.'*60)
#the first 5 confusion matrices of the 10
for smote_during_cv_conf_matrix in smote_during_cv_conf_matrices[:5]:
    ax1 = sns.heatmap(nearmiss_before_cv_conf_matrix, cbar=False, xticklabels=tick_labels, yticklabels=tick_labels, ax=ax[y], annot=True, annot_kws={'size' : 10})
    #ax0.set_ylabel('Actual Labels')
    #ax0.set_xlabel('Predicted Labels')
    y = y + 1


# In[ ]:





# It is worthy to note that <b>you cannot say which is better between NearMiss and SMOTE</b>.  The reason is because <b>NearMiss</b> brings about removal of data points which could bring about <b>Underfitting</b>, while <b>SMOTE</b> brings about addition of new synthetic points which could bring about <b>Overfitting</b>. Maybe the idea that SMOTE can lead to overfitting can be traced to the high accuracies we obtained.

# In[ ]:





# In[ ]:





# <h3>Features to expect in Future Updates</h3>:
# 
# 1. Other models: Gaussian Naive Bayes, Support Vector Classifier and Decision Tree Classifier.
# 2. Precision-Recall Curves to show how our model distinguishes the two classes from each other.
# 3. Learning Curves to show each of the models' performance over experience and time.
