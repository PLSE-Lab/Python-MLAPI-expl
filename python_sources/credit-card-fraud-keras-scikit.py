#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# This is a notebook to find an algorithm to predict credit card frauds. The main focus is on the classification of an imbalanced dataset.

# In[ ]:


#import all packages
import numpy as np 
import pandas as pd 

import os
print(os.listdir("../input"))


import seaborn as sns
import scikitplot as skplt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler

from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline
from imblearn.metrics import specificity_score


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, roc_auc_score, roc_curve, accuracy_score, precision_score, make_scorer


import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dropout
from keras.layers.core import Dense
from keras.optimizers import Adamax
from keras.metrics import categorical_crossentropy


# **Data Exploration**

# In[ ]:


#read data and take first look at it
df = pd.read_csv('../input/creditcard.csv')
df.head()


# In[ ]:


#get information provided by pandas
df.info()


# In[ ]:


df.describe()


# From this first overview we can retrieve a lot of information about feature extraction and how to start to treat the dataset.
# * there are no missing data
# * all columns are of a numeric type
# * the columns from V1 to V28 are the result of Dimensionality Reduction with Principal Component Analyisis, so scaling was already done (mean of nearly 0)
# * the columns time and amount have not been scaled
# * the column class (1==fraud, 0==non fraud) has a mean of 0.001727, which implies a small amount of frauds in the dataset
# * the max of "Time" is 172792 which is nearly equal to 172800, which is the sum of seconds of 2 days

# In[ ]:


# graph to show imbalance of the dataset

sns.countplot(df["Class"])
plt.title("Fraud==1 vs. Non fraud==0")
plt.show()


# The classes are very imbalanced. This is a problem when predicting because the algorithms to predict will "learn" on nearly only non fraudulent activities. This issue has to be treated for a good prediction.

# In[ ]:


#plot "Amount" column

fig, ax = plt.subplots(1,3, figsize=(20, 8))

df['Amount'].plot(ax=ax[0])
ax[0].set_title("Amount per Transaction")
ax[0].set_xlabel("Transaction Number")
ax[0].set_ylabel("Amount in Dollar")

df['Amount'].plot.hist(ax=ax[1], bins=200, color="r")
ax[1].set_title("Distribution of Amounts")
ax[1].set_xlabel("Amount in Dollar")

df['Amount'].plot.hist(ax=ax[2], bins=200, color="g")
ax[2].set_title("Distribution of Amounts closup")
ax[2].set_xlabel("Amount in Dollar")
ax[2].set_ylim([0,50])
plt.show()


# The distribution of the amount is highly skewed with most of the transaction under 5000.

# In[ ]:


#plot "Time" distribution

df["Time"].plot.hist(bins=50)
plt.title("Distribution of time since first transaction")
plt.xlabel("Time in seconds")
plt.axvline(x=12500, color='r')
plt.axvline(x=12500+86400, color='r')
plt.show()


# The distribution of time is time a cycle of days with 86400 seconds. A random period of a day is marked with red lines.

# In[ ]:


#check the distributions of the pca-features for fraud and non fraud and compare them

pca_features = df.iloc[0:-1,0:28].columns
plt.figure(figsize=(35,30*4))
grid = gridspec.GridSpec(28, 1)
for i, feat in enumerate(df[pca_features]):
    ax = plt.subplot(grid[i])
    sns.distplot(df[feat][df.Class == 1], bins=200)
    sns.distplot(df[feat][df.Class == 0], bins=200)
    ax.set_xlabel('')
    ax.set_title('Feature : ' + str(feat))
plt.show()


# In the diagrams above for each feauture V1-V28 you can see the distribution of the values for the feature. There are two graphs in each diagram, one for each classe. So for every feature the two classes can be compared. If the distribution is too similar they are not useful for the algorithm because the feature should be different for both classes.

# **Feature Transformation**

# In[ ]:


# remove features where the distributions are for both classes are too similar
df = df.drop(['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8'], axis =1)


# In[ ]:


# converting "Time" to a time period of one day from 0 to 86400 seconds

second_day_indices = df[df["Time"]>86400].index
df.loc[second_day_indices, "Time"] = df.loc[second_day_indices, "Time"] - 86400

print("Minimum time is now {}".format(df["Time"].min()))
print("Maximum time is now {}".format(df["Time"].max()))


# The columns amount and time have to be scaled to be used with algorithms. For example for all algorithms which use the Euclidian distance as a measurement it makes a difference if the scale of the value is in a similar proportion. The choice falls on the RobustScaler from scikit learn because it is less prone against outliers like in the amount column.

# In[ ]:


# scale "Time" and "Drop", delete original columns
robust_scaler = RobustScaler()

df["time_scaled"] = robust_scaler.fit_transform(df['Time'].values.reshape(-1,1))
df["amount_scaled"] = robust_scaler.fit_transform(df["Amount"].values.reshape(-1,1))
df.drop(["Time", "Amount"], inplace=True, axis=1)
df.head()


# In[ ]:


# create feature and target dataset
X = df.drop(["Class"], axis=1)
y = df["Class"]


# **Methods of sampling **
# 
# The motivation for sampling is to give the classifier a better ratio of fraudulent activity when it is training, so it will not be trained only on non fraudulent activity and will not only predict these.

# **Undersampling**

# The first method is called random undersampling. This means to change the data to an equal (50/50) distribution of classes by randomly selecting as many non fraudulent transactions as fraudulent and put them in one dataset.

# In[ ]:


rus = RandomUnderSampler(random_state=42)
X_ramdom_undersampled, y_random_undersampled = rus.fit_resample(X, y)


sns.countplot(y_random_undersampled)
plt.title("Distribution of Class for random undersampling")
plt.show()


# A second method called "Instance hardness threshold" can be used for undersampling. Here a classifier is trained on the data and the data with lesser probabilities are not in the sample. The result looks similar but with another less random choice of the non fraudulent data for the sample.

# In[ ]:


iht = InstanceHardnessThreshold(random_state=42, estimator=LogisticRegression(
                                    solver='liblinear', multi_class='auto'))
X_iht_undersampled, y_iht_undersampled = iht.fit_resample(X, y)


sns.countplot(y_iht_undersampled)
plt.title("Distribution of Class for IHT undersampling")
plt.show()


# **Oversampling**

# First variant of oversampling is random oversampling which just randomly replicates the underrepresented class until both classes are equal.

# In[ ]:


ros = RandomOverSampler(random_state=42)
X_ramdom_oversampled, y_random_oversampled = ros.fit_resample(X, y)


sns.countplot(y_random_oversampled)
plt.title("Distribution of Class for random oversampling")
plt.show()


# The second variant of undersampling is "Synthetic Minority Over-sampling Technique" that generates new samples of the minority class which are close to the original samples using K-Nearest-Neighbors.

# In[ ]:


smote = SMOTE(random_state=42)
X_smote_oversampled, y_smote_oversampled = smote.fit_resample(X, y)


sns.countplot(y_smote_oversampled)
plt.title("Distribution of Class for smote oversampling")
plt.show()


# **Sampling and crossvalidation**

# Now we have mentioned four different sampling methods and seen the results. One could assume to use all of these for feeding the algorithms. But I also want to use cross-validation. The problem is that by using the sampling the test set will be affected because it is part of the sampling result. As we know the test set should be untouched. So the solution is to just sample the trainingset.

# **Metrics for evaluation**

# The metrics for the classification evaluation which are used are the roc-curve, confusion-matrix, accuracy, recall, roc-auc-score,  specificity and the precision-recall-curve. I choose the RandomForestClassifer without sampling to show an example of the metrics.

# In[ ]:


# splitting in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[ ]:


# make a first prediction with RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators=10)
clf_rf.fit(X_train, y_train)
prediction = clf_rf.predict(X_test)


# In[ ]:


# print roc curve
y_true = y_test
y_probas = clf_rf.predict_proba(X_test)
skplt.metrics.plot_roc(y_true, y_probas)
plt.show()


# The ROC-curve tells us how good the algorithm can distinguish between the two classes of frauds and non frauds. The bigger the area under the curve is the better.

# In[ ]:


#roc-auc-score
roc_auc_score(y_true,y_probas[:,1])


# The roc-auc-score is the numeric representation of the area under the roc-curve with 1 as the perfect score.

# In[ ]:


#print confusion matrix
skplt.metrics.plot_confusion_matrix(y_true, prediction)
plt.show()


# The confusion matrix gives us the exact numbers of the false and true predicted transactions.

# In[ ]:


#accuracy score
accuracy_score(y_test, prediction)


# The accuracy score gives us the ratio of how many transactions where labeled right.

# In[ ]:


#recall score
recall_score(y_test.values, prediction)


# The recall score gives us the ratio of how many of the fraudulent transactions where found

# In[ ]:


# precision-recall curve
skplt.metrics.plot_precision_recall(y_test, y_probas)
plt.show()


# The precision-recall-curve shows us the tradeoff between finding all frauds(recall) and the accuracy of finding them(precision).

# In[ ]:


# specificity score
specificity_score(y_test.values, prediction)


# The specificity gives us the ratio of correct classification of non fraudulent transactions

# In[ ]:


# build custom scorer for finding best hyperparameters
def custom_score(y_true, y_pred):
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    #define measures
    recall = 0.75 * recall_score(y_true, y_pred) 
    specificy = 0.25 * conf_matrix[0,0]/conf_matrix[0,:].sum() 
    #punish low recall scores
    if recall < 0.75:
        recall -= 0.2
    return recall + specificy 
    
#initialize make_scorer
optimized_score = make_scorer(custom_score)


#  I want to use GridSearchCV and to make it find the best algorithm I created  a custom scorer which takes recall and specificity into considiration when checking for the ideal parameters. After finding the best estimator for the classifier algorithm this estimator will be evaluated with the other metrics on the test set.

# The most important metrics are the recall and the specificity. The recall is important because I want to find all fraudulent activities. In production this algorithm will only stop the transaction but further investigation will be done for every single as fraudulent labeled transaction. Thatswhy the specificity has to be good too, so that not many non fraudulent transactions labeled wrong which will reduce the cost of further investigations.
# 
# **Testing algorithms**

# 1. To find the best algorithm I choose 4 to test them with the following parameters to test

# In[ ]:


clf_rf = RandomForestClassifier(n_estimators=10)
rf_params = {"max_depth" : [5,7,10], 'criterion':['gini','entropy']}

clf_lr = LogisticRegression(solver='liblinear')
lr_params = {"C" : [0.001,0.01,0.1,1,10,100], "warm_start" :[True, False]}

clf_gb = GradientBoostingClassifier()
gb_params = {"learning_rate" : [0.001,0.01,0.1], 'criterion' : ['friedman_mse', 'mse']}

clf_svc = SVC(gamma='scale', probability=True)
svc_params = {'kernel' : ['linear', 'poly', 'rbf'], "C" : [0.001,0.01,0.1,1,10,100]}


# * Due to computational limitations only RandomUnderSampling as sampling method will be used
# * training, validation and sampling will be done only on X_train to keep X_test untouched before testing
# * StratifiedShuffleSplit is used for cross validation and sampling is done during cross validation
# * the score on which GridSearchCV decides the best scorer is the custom scorer shown in the metrics
# * sampling is done during cross validation.

# **Detailed look into the finding of the best estimator for each classifier **

# 1. StratifiedShuffleSplit.split  of X_train and y_train creates 5 train and test sets (X_test stays untouched during the whole process)
# 2. On each of these train sets RandomUnderSampling and GridSearchCV fitting is done to get the best estimator
# 3. On each of the corresponding test sets the best estimator is tested with the scores
# 4. The average score of all the best estimator results is calculated

# In[ ]:


# splitting in training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

X_train = X_train.values
y_train = y_train.values
    
def find_best_estimator(estimator, params):
    gridsearch_cv = GridSearchCV(estimator, param_grid=params, cv=5, iid=False, scoring=optimized_score)

    sss = StratifiedShuffleSplit(n_splits = 5, test_size = 0.2, random_state=42)

    rus = RandomUnderSampler()
    

    recall_list = []
    specificity_list = []
    i=0
    print("")
    print( type(estimator).__name__)
    
    for train_index, test_index in sss.split(X_train, y_train):
        pipeline = make_pipeline(rus, gridsearch_cv)
        model = pipeline.fit(X_train[train_index], y_train[train_index])
        best_estimator = gridsearch_cv.best_estimator_
        prediction = best_estimator.predict(X_train[test_index])

        recall_list.append(recall_score(y_train[test_index], prediction))
        specificity_list.append(specificity_score(y_train[test_index], prediction))
        i=i+1
        
        print("Iteration {} out of 5 is finished".format(i))
      
    print("")
    print("recall on X_train split in train and test: {}".format(np.mean(recall_list)))
    print("Specificy on X_train split in train and test: {}".format(np.mean(specificity_list)))
    return best_estimator  


# In[ ]:


best_est_gb = find_best_estimator(clf_gb, gb_params)
best_est_lr = find_best_estimator(clf_lr, lr_params)
best_est_rf = find_best_estimator(clf_rf, rf_params)
best_est_svc = find_best_estimator(clf_svc, svc_params)


# Now that all the estimators have a decent score on there test sets I can build an evaluation report for all of them with the metrics mentioned at the beginning and predicting on X_test.

# In[ ]:


#change fond size for all plots
plt.rcParams.update({'font.size': 16})


# In[ ]:


# function to display all the metrics on the estimators
def evaluation_report(estimator):

    prediction = estimator.predict(X_test)
    prediction_proba = estimator.predict_proba(X_test)
       
    fig, ax = plt.subplots(1,3, figsize=(30, 8))

    fig.suptitle('Evaluation report for '+type(estimator).__name__, fontsize=16)

    skplt.metrics.plot_precision_recall(y_test, prediction_proba, ax=ax[0])
    ax[0].set_title("Precision-recall-curve")

    skplt.metrics.plot_roc(y_test, prediction_proba, ax=ax[1])
    ax[1].set_title("ROC-curve")

    skplt.metrics.plot_confusion_matrix(y_test, prediction, ax=ax[2])
    ax[2].set_title("Confusion-matrix")
    plt.show()
    print('The recall is {}'.format(recall_score(y_test.values, prediction)))
    print('The specificity is {}'.format(specificity_score(y_test.values, prediction)))
    print('The accuracy is {}'.format(accuracy_score(y_test, prediction)))
    print('The AUC-score is {}'.format(roc_auc_score(y_test,prediction_proba[:,1])))


# In[ ]:


evaluation_report(best_est_svc)


# In[ ]:


evaluation_report(best_est_lr)


# In[ ]:


evaluation_report(best_est_gb)


# In[ ]:


evaluation_report(best_est_rf)


# When looking at the metrics one can see that they are all close together. For example if the specificity is 0.95 and the recall 0.93 that means that 93 % of the frauds are detected and 5 % of the non fraudulent transaction are classified as fraud, which is still a high number. So around 25 transactions are labeled as fraud for further investigation and only one out of them is a real fraud.

# **Prediction with keras**

# Now I will use Keras with a simple neural network with dropouts to predict. First I will use the network with oversampling SMOTE and after that I will compare it with RandomUnderSampling.

# In[ ]:


# oversampling model with SMOTE
X_smote_oversampled, y_smote_oversampled = SMOTE().fit_resample(X_train, y_train)
n_inputs = X_smote_oversampled.shape[1]

oversample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),    
    Dense(2, activation='softmax')
])

oversample_model.summary()


# In[ ]:


# compile model
oversample_model.compile(Adamax(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# fit model
oversample_model.fit(X_smote_oversampled, y_smote_oversampled, validation_split=0.2, batch_size=25, epochs=15, shuffle=True, verbose=2)


# In[ ]:


# create same model with RandomUnderSampler
X_rus_undersampled, y_rus_undersampled = RandomUnderSampler().fit_resample(X_train, y_train)
n_inputs = X_rus_undersampled.shape[1]

undersample_model = Sequential([
    Dense(n_inputs, input_shape=(n_inputs, ), activation='relu'),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),    
    Dense(2, activation='softmax')
])


# In[ ]:


# compile model
undersample_model.compile(Adamax(lr=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


# fit model
undersample_model.fit(X_rus_undersampled, y_rus_undersampled, validation_split=0.2, batch_size=25, epochs=15, shuffle=True, verbose=2)


# In[ ]:


# the prediction and probas in the evaluation report have to be adjusted slightly
def evaluation_report_keras(estimator):

    prediction = estimator.predict_classes(X_test, batch_size=200, verbose=0)
    prediction_proba = estimator.predict(X_test, batch_size=200, verbose=0)
       
    fig, ax = plt.subplots(1,3, figsize=(30, 8))

    
    skplt.metrics.plot_precision_recall(y_test, prediction_proba, ax=ax[0])
    ax[0].set_title("Precision-recall-curve")

    skplt.metrics.plot_roc(y_test, prediction_proba, ax=ax[1])
    ax[1].set_title("ROC-curve")

    skplt.metrics.plot_confusion_matrix(y_test, prediction, ax=ax[2])
    ax[2].set_title("Confusion-matrix")
    plt.show()
    print('The recall is {}'.format(recall_score(y_test.values, prediction)))
    print('The specificity is {}'.format(specificity_score(y_test.values, prediction)))
    print('The accuracy is {}'.format(accuracy_score(y_test, prediction)))
    print('The AUC-score is {}'.format(roc_auc_score(y_test,prediction_proba[:,1])))


# In[ ]:


# show evaluation report SMOTE-Keras-model
evaluation_report_keras(oversample_model)


# In[ ]:


# show evaluation report RandomUnderSampler-Keras-model
evaluation_report_keras(undersample_model)


# **Conclusion**
# 
# Astonishingly undersampling with ca. 750 and oversampling with 450,000 transactions gave nearly the same result in the neural network. The specificity for the neural network is sligthly better with SMOTE but it is very close. In comparison to the other non neural network classifiers the specificity is much better but the important recall is lower. For further investigation with more computing power the oversampling methods could be investigated in more detail and other classifiers and neural networks could be used.
