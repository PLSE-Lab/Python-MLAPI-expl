#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection
# #### A Case Study
# ***
# 

# The aim is to identify a fraudulent credit card transaction from a non-fraudulent one, and subsequently build a model to predict the same.
# The dataset provided is for transactions made by credit cards in September 2013 by european cardholders, for a time period of two days.

# From the source of this dataset, we have the information that there are over 2 lakh data points, out of which less than one percent is classified as fraud. It is a clear **Class Imbalance** scenario. As an initial thought, we will have to keep that in mind during our model building phase. Let's dive into it.

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import time
import os
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[ ]:


get_ipython().system('pwd')


# In[ ]:


#Change directory to where dataset is - not required on kaggle
#os.chdir("X:\\Datasets\creditcardfraud")


# In[ ]:


#Reading the dataset
master_df=pd.read_csv('../input/creditcardfraud/creditcard.csv')


# In[ ]:


#Keeping our original dataset safe :)
transaction_data = master_df.copy()


# In[ ]:


display(transaction_data.head())
print("Dimensions of the dataset are:", transaction_data.shape)


# In[ ]:


#Data points for Fraudulent transactions
fraud = len(transaction_data[transaction_data["Class"]==1])
print("Total Fraud transactions are", fraud, ", which is") 
print(round(fraud/len(transaction_data) * 100, 2), "% of the dataset")


# In[ ]:


#Checking for null values (all values are numerical)
transaction_data.isnull().sum()

# We have no null values


# In[ ]:


#Let's look at the summary of the data
transaction_data.describe()


# At first glance, we can easily see that features V1-28 have been transformed, as mentioned, using PCA. We can't really gather anything from them.<br>
# Also, looks like 'Amount' might turn out to be a parameter to predict outliers(frauds), as there is a huge difference between Maximum amount (25691.16) and Mean amount (88.349619). Let's check.

# In[ ]:


#Let's convert the time variable into hours from seconds before moving forward

transaction_data['Time'] = transaction_data['Time']/3600


# In[ ]:


#Distribution of Amount
figure, a = plt.subplots(1, 2, figsize=(15, 4))
amount = transaction_data['Amount'].values

sns.boxplot(amount, ax = a[0])
a[0].set_title('Distribution of Amount')

fraud_df = transaction_data[transaction_data['Class']==1]
amount_fraud = fraud_df['Amount'].values

sns.boxplot(amount_fraud, ax = a[1])
a[1].set_title('Distribution of Amount w.r.t. Fraud Transactions')
#Keeping a similar scale as previous graph for comparison
#a[1].set_xlim([min(amount), max(amount)])
plt.show()


# As we can see, the outliers in *fraud transactions* are not very significantly high, as compared to the whole dataset. So amount doesn't really look like a direct predictor. 

# In[ ]:


#Checking distribution with time

figure, a = plt.subplots(1, 2, figsize=(15, 4))
time = transaction_data['Time'].values

sns.distplot(time, ax = a[0])
a[0].set_title('Distribution of Transactions with Time')
a[0].set_xlim([min(time), max(time)])
#fraud_df_time = transaction_data[transaction_data['Class']==1]

time_fraud = fraud_df['Time'].values

sns.distplot(time_fraud, ax = a[1])
a[1].set_title('Distribution with Time for Fraud Transactions')
#Keeping a similar scale as previous graph for comparison
a[1].set_xlim([min(time_fraud), max(time_fraud)])
plt.show()


# From this visualisation, we can see that there is not much difference in the trend with time as well. Though an interesting thing to note will be the higher number of fraud transactions near the 10th hour. The total transactions are higher around the 20th hour. But we can't say for sure if this assumption is believable as data points for fraud transactions are too less.

# It follows from this, that our analysis will be severely affected by the huge class imbalance that we have observed. Even during model building, our model might tend to overfit the data, and we might see very high accuracy scores. Hence, we will have to keep in mind the following things:
# 
# - Dataset needs to be modified in a way, to provide some balance between both classes. We can generally do this through **Under Sampling or Over Sampling** the data.
# - Whichever model we use for prediction, we will have to use appropriate metrics to check our model performance, as simple scores such as Accuracy can be very misleading. Best approach would be to use Precision-Recall scores as well as **Area Under Precision-Recall Curve.**

# ***

# #### Let's go ahead with data manipulation.

# In[ ]:


#Scaling time and amount values, as all other predictors are already scaled through PCA
#Using min-max scaler for higher efficiency 

scaler = MinMaxScaler()
transaction_data['Time'] = scaler.fit_transform(transaction_data["Time"].values.reshape(-1,1))
transaction_data['Amount'] = scaler.fit_transform(transaction_data["Amount"].values.reshape(-1,1))


# In[ ]:


display(transaction_data.head())


# Now, coming to the first problem of class imbalance, let's go forward with a technique called SMOTE, or Synthetic Minority Oversampling Technique. As the name suggests, it is an oversampling technique which will increase the minority class data points in proportion to the majority class.<br>
# 
# It should be noted that we are not going with UnderSampling, as it poses the risk of losing out on important information. With SMOTE (oversampling), more information is available.

# In[ ]:


#Splitting into values and class labels 
target = 'Class'
labels = transaction_data[target]
values = transaction_data.copy()
values = values.drop('Class', axis =1 )
display(values.head())
print("Dataset dimension :", values.shape)
print("Labels dimension :", labels.shape)


# Before applying any sampling technique, it should be remembered that we are going to fit our model, using the *sampled training set*, but we will be predicting using only the *original test set*. So we will have to split the data into training and testing sets, before we can apply any sort of sampling.

# Also, I wish to use SelectKBest algorithm, to use the best features for classification. But since as of now, there is a huge class imbalance, the same will be represented through SelectKBest. So I will use it, after oversampling my dataset.

# In[ ]:


# We can use Stratified Shuffle Split or StratifiedKFold
strat_split = StratifiedShuffleSplit(n_splits=10, random_state=42)
for train_index, test_index in strat_split.split(values, labels):
    values_train, values_test = values.iloc[train_index], values.iloc[test_index]
    labels_train, labels_test = labels.iloc[train_index], labels.iloc[test_index]
print("Percentage of Fraud Transactions in Training Set using Stratified Shuffle Split:", 
      round(labels_train.value_counts()[1]/len(labels_train)*100, 4), "%")

kfold_split = StratifiedKFold(n_splits=10, random_state=47)
for train_index, test_index in kfold_split.split(values, labels):
    values_train1, values_test1 = values.iloc[train_index], values.iloc[test_index]
    labels_train1, labels_test1 = labels.iloc[train_index], labels.iloc[test_index]
print("Percentage of Fraud Transactions in Training Set using Stratified KFold:", 
      round(labels_train1.value_counts()[1]/len(labels_train1)*100, 4), "%")


# We are getting similar results with both. Basically, our class imbalance percentage is maintained in our training sets, for both the cases. We will go ahead with StratifiedShuffleSplit.<br>

# In[ ]:


#Applying SMOTE on the training dataset
smote = SMOTE(sampling_strategy='minority', random_state=47)
os_values, os_labels = smote.fit_sample(values_train, labels_train)

os_values = pd.DataFrame(os_values)
os_labels = pd.DataFrame(os_labels)

plt.figure(figsize=(5, 5))
sns.countplot(data = os_labels, x = 0 )
plt.title("Distribution of Oversampled Training Set")
plt.xlabel("Fraud")


print("Dimensions of Oversampled dataset is :", os_values.shape)


# Now, since our dataset is equally balanced, let's apply selectKbest for attaining the best features as per feature scores.<br> Note that we will be using SKB on the oversampled dataset, as that is the one which will be used to fit the model. Hence, transforming the complete dataset would be unneccesary.

# In[ ]:


#SelectKBest on our oversampled training dataset
os_values_skb = os_values.copy()
skb = SelectKBest(k=15)
os_values_skb = skb.fit_transform(os_values_skb, os_labels[0].ravel())
display(pd.DataFrame(os_values_skb).head())


# In[ ]:


# Getting feature scores from SelectKBest after fitting
feature_list = values.columns
unsorted_list = zip(feature_list, skb.scores_)

sorted_features = sorted(unsorted_list, key=lambda x: x[1], reverse=True)
print(len(sorted_features))
print("Feature Scores:\n")
pprint(sorted_features[:15])


# In[ ]:


selected_features = [i[0] for i in sorted_features[:15]]
print(selected_features)
#Transforming test according to Feature Selection

#values_test = values_test[selected_features]
os_values_skb = os_values.copy()
display(values_test.head())


# Unfortunately, after testing SelectKBest, our predictions were getting only worse, so we will not be using SelectKBest. For model building purpose, all features will be used, as is.

# ***
# #### Model Building
# Let's now test this oversampled dataset on a basic Logistic Regression model, as this is a binary classification problem

# In[ ]:


#Using this Oversampled data to fit a Logistic Regression model
logr = LogisticRegression().fit(os_values_skb, os_labels[0].ravel())

#Predicting on original test set
predictions = logr.predict(values_test)

print("Accuracy Score: ", round(accuracy_score(labels_test, predictions)*100, 2), '%')
print(classification_report(labels_test, predictions, target_names=["No Fraud", "Fraud"]))


# In[ ]:


#Computing major metrics
y_score = logr.decision_function(values_test)
precision, recall, _ = precision_recall_curve(labels_test, y_score)
fig = plt.figure(figsize=(10,5))
sns.lineplot(recall, precision, drawstyle='steps-post')
plt.fill_between(recall, precision, alpha=0.1, color='#78a2e2')
plt.title('Precision-Recall Curve for Logistic Regression')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.1])
plt.xlim([0.0, 1.1])


# As we can see, our Logistic Regression model, is not proving to be as good as model which the accuracy is suggesting. The precision score is too low. The Precision-Recall curve though, depicts a different story. The documentation states the following:
# 
# > <font color="#686a6d">The precision-recall curve shows the tradeoff between precision and recall for different threshold. A high area under the curve represents both high recall and high precision, where high precision relates to a low false positive rate, and high recall relates to a low false negative rate. High scores for both show that the classifier is returning accurate results (high precision), as well as returning a majority of all positive results (high recall)"</font>

# The area under the P-R curve seems to depict that our model is working moderately well. We could expect a better performance though, since we have not even tuned this model as of now. We will have to tune our model with various parameters, for achieving a better performance. Before we do that, let's look at the performance with some other models.

# In[ ]:


t0 = time.time()

DT = DecisionTreeClassifier()
y_pred_DT = DT.fit(os_values_skb, os_labels[0].ravel()).predict(values_test)

t1 = time.time()
print("Fitting Decision Tree model took", t1-t0, "secs.")
print("."*10)
ada = AdaBoostClassifier()
y_pred_ada = ada.fit(os_values_skb, os_labels[0].ravel()).predict(values_test)

t2 = time.time()
print("Fitting Adaboost model took", t2-t1, "secs.")
print("."*10)
xgb = GradientBoostingClassifier()
y_pred_xgb = xgb.fit(os_values_skb, os_labels[0].ravel()).predict(values_test)

t3 = time.time()
print("Fitting XGBoost model took", t3-t2, "secs.")
print("."*10)


# In[ ]:


# Precision, Recall, and actual prediction numbers
print("Accuracy Score for DT: ", round(accuracy_score(labels_test, y_pred_DT)*100, 2), '%')
print(classification_report(labels_test, y_pred_DT))
print("-" * 40)
print("Accuracy Score for AdaBoost: ", round(accuracy_score(labels_test, y_pred_ada)*100, 2), '%')
print(classification_report(labels_test, y_pred_ada))
print("-" * 40)
print("Accuracy Score for XGB: ", round(accuracy_score(labels_test, y_pred_xgb)*100, 2), '%')
print(classification_report(labels_test, y_pred_xgb))


# We are achieving similar results from this. Let's move on to tuning our parameters for our models, using GridSearchCV.
# ***

# #### Tuning Our Models
# 
# We will be tuning our basic Logistic Regression model, using various parameters, to obtain a better performance.

# In[ ]:


#Trying below three models:
# Logistic Regression Classifier
def tune_LogR():
    print("------------------ Using Logistic Regression --------------------")
    clf = LogisticRegression()
    param_grid = {
        'clf__penalty': ['l1', 'l2'],
        'clf__C' : [1.0, 10.0, 25.0, 50.0, 100.0, 500.0, 1000.0],
        'clf__solver' : ['liblinear']
    }

    return clf, param_grid


# In[ ]:


# Create pipeline
clf, params = tune_LogR()
estimators = [('clf', clf)]
pipe = Pipeline(estimators)

# Create GridSearchCV Instance
grid = GridSearchCV(pipe, params)
grid.fit(os_values_skb, os_labels[0].ravel())

# Final classifier
clf = grid.best_estimator_

print('\n=> Chosen parameters :')
print(grid.best_params_)

predictions = clf.predict(values_test)
print("Accuracy Score: ", round(accuracy_score(labels_test, predictions)*100, 2), '%')
print("Classification Report:\n", classification_report(labels_test, predictions, target_names = ['Non-Fraud', 'Fraud']))


# In[ ]:


#Plotting the P-R curve for this tuned model
y_score = clf.decision_function(values_test)
precision, recall, _ = precision_recall_curve(labels_test, y_score)
fig = plt.figure(figsize=(10,5))
sns.lineplot(recall, precision, drawstyle='steps-post')
plt.fill_between(recall, precision, alpha=0.1, color='#78a2e2')
plt.title('Precision-Recall Curve for Logistic Regression - After Tuning')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.1])
plt.xlim([0.0, 1.1])


# In[ ]:


#Plotting confusion matrix for this
tn, fp, fn, tp = confusion_matrix(labels_test, predictions).ravel()
print("True Negatives:", tn)
print("False Positives:", fp)
print('False Negatives:', fn)
print('True Positives:', tp)


# Let's test out our model's performance on an balanced dataset, for validation purposes.
# We will split the data with a ratio of 70:30, keeping the percentage equal, as in using the oversampled dataset.

# In[ ]:


splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=1)
for train_index, test_index in splitter.split(os_values, os_labels):
    _, os_values_test = os_values.iloc[train_index], os_values.iloc[test_index]
    _, os_labels_test = os_labels.iloc[train_index], os_labels.iloc[test_index]


# In[ ]:


predictions = clf.predict(os_values_test)
print("Accuracy Score: ", round(accuracy_score(os_labels_test, predictions)*100, 2), '%')
print("Classification Report:\n", classification_report(os_labels_test, predictions, target_names = ['Non-Fraud', 'Fraud']))


# As is expected, we are getting amazing scores all over for this, as we have a balanced dataset, plus the model has been trained on it before.

# ***

# ### Summary

# In conclusion, we have understood that this dataset has a huge class imbalance, that will definitely affect our analysis as well as model building approach. We have to keep in mind some important points, which are stated below:
# 
# - The relationships between our target variable 'Class' and our predictor variables, is difficult to estimate if we just look at the original dataset. It is highly skewed.
# - To establish a clear understanding, we need to remove the class imbalance present in our dataset. We have used SMOTE as an oversampling technique for the minority class. We have not gone with undersampling as it may lead to information loss.
# - Since this is a binary classification problem, we have chosen Logistic Regression as our model. After oversampling and tuning our model, we have achieved an accuracy of **97.34%**. Our main metric, **Area Under the Precision Recall Curve**, seems to show that model performance is moderate. We have high recall scores, but lower precision scores.
# - Testing it on a balanced dataset, we have seen that both precision and recall prove to be very high.
# 
# 

# Our main concern, is to point out if a transaction can be safely classified as fraud or not. This model might not be highly accurate in predicting frauds, but at the same time the number of correctly predicted 'Non-Fraud' Transactions is a positive point. We don't want actual, genuine transactions to be classified as Fraud. The case presented in this study is definitely a real world scenario, where the number of fake transactions would be quite low, when compared to genuine transactions.

# Lastly, as part of future additions to this study, I would like to go ahead with in depth analysis of the PCA scaled features provided to us, through Hypothesis testing. It might lead to some idea of how they are affecting the target class. Also, using unsupervised learning techniques such as K-Means Clustering, on the oversampled data, might prove to be beneficial, as SMOTE works on the principle of nearest neighbours as well.
