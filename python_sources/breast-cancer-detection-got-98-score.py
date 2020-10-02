#!/usr/bin/env python
# coding: utf-8

# # **Inroduction**
# The goal of this problem is to use the power of Machine Learning algorithms to take the dataset of past measurements of Breast Cancer and apply some analysis to understand which most features that can be asign of a Breast Cancer? Also, to predict the likelihood of future patients to be diagnosed as sick.
# So given important measurements of a future patient we can train a ML algorithm to predict if he/she carries a Breast Cancer easily and accurately.

# # Getting Started 
# Import the basic liberaries used in this project
# * Pandas
# * Numpy
# * seaborn
# * Matplotlib 
# 
# for this project I have build a python file called `functions` and will use it in `Model Selection` section

# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

import warnings

warnings.simplefilter('ignore')


# In[ ]:


# load the datset into data 
data = pd.read_csv('../input/data.csv')

# drop uncessary columns 
data.drop(['id', 'Unnamed: 32'], axis = 1, inplace = True)

#print the number of columns and rows
print("This dataset contains {} rows and {} columns".format(data.shape[0], data.shape[1]))

# change the target to numerical to help us in statistics
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

data.head()


# # Data Exploration
# In this section of this project, I will make a cursory investigation about the dataset and provide some observations. Also I'm gooing to Familiarize myself with the data through an `explorative` process and it is a fundamental process to help us better understand the data and justify the final results. So we are going to walk through the following
# * Calculate `Statistics` for numerical features
# * Get information about the dataset and its dtypes to detect null values
# * Plot Histograms to `Visualize Feature Distributions` in the dataset(Detect Skewness)

# ### - Calculate Statistics for numerical features

# In[ ]:


# Calculate Statistics for numerical features
data.describe()


# ##### From the difference between the `median` and `mean` it seems there are some features that have `skewness` that need to be transformed 

# ### - Get information about the dataset and its dtypes to detect null values

# In[ ]:


data.info()


# ##### CONCLUSION: the dataset contains only 1 categorical column and the rest are numericals, also the dataset is clean and have no null values and ready for the preprocessing stage

# ### - Plot Histograms to Visualize Feature Distributions in the dataset and Detect Skewness 

# In[ ]:


# Split the dataset to target and features
features = data.drop('diagnosis', axis = 1)
target = data['diagnosis']


# In[ ]:


plt.figure(figsize = (15, 38))
plt.suptitle('Histograms for Numeric Features in the dataset', fontsize = 20)
L = list(data)
for i in range(data.shape[1]):
    plt.subplot(11, 3, i + 1)
    sns.distplot(data[L[i]])


# ##### Calculate the `skewness` of each feature to make sure the features are normally or sub-normally distributed and fix them if not normally distributed

# In[ ]:


from scipy.stats import skew

features_list = list(features)
Skew_D = {}

# Claculate the skewness of each feature and store them in Skew_D
for f in features_list:
    Skew_D[f] = skew(features[f], bias = False)
    
# Store the features that have high skewned
High_skewed_features = []
for i in Skew_D:
    if (Skew_D[i] > 1) or (Skew_D[i] < -1):
        High_skewed_features.append(i)


# ##### Plot the features that have high skewness to visualize the skewness using histograms 

# In[ ]:


features[High_skewed_features].hist(figsize = (15, 15))
plt.show()


# # Correlation Testing
# In this section of this project I'm going to find the correlations in the dataset. Because detecting these correlations can help us in `Feature Selection` process in order to minimize the number of features in the dataset.
# 
# #### NOTE: I will calculate the correlation between each pair of attributes (correlation matrix). Then I'll plot the correlation matrix to get an idea of which variables have a high correlation with each other.

# ### Calculate coffecients of corelations between each pair of input features

# In[ ]:


data_corr = data.corr()

plt.figure(figsize = (18, 18))
sns.heatmap(data_corr, annot = True)
plt.show()


# In[ ]:


highest_corr = data_corr['diagnosis'].sort_values(ascending = False)[1: 20]
highest_corr


# #### Now I'm going to plot barplot (diagnosis VS highest_corr features) to visualize the correlation between the target and the most effective features. 

# In[ ]:


plt.figure(figsize = (20, 35))
plt.suptitle('Barplots for Diagnosis versus highest correlated features in the dataset, B: 0 , M: 1', fontsize = 25)

L = highest_corr.index
for i in range(len(L)):
    plt.subplot(8, 4, i + 1)
    sns.barplot(data = data, x = 'diagnosis', y = L[i])


# # Outlier Detection
# In this section I'm going to apply some techniques to detect the outliers in the dataset. Outlier can be source of information in the dataset on the other hand it maybe lead to bad results or biased result. So we need to detect them and treat them in the `Preprocessing` stage. 

# The presence of outliers can often skew results. There are many techniques for how to detect and deal with the outliers in a dataset. *outlier step* is calculated as `factor` multiplied the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.

# In[ ]:


# hold all indices of outliers
outliers_index = set()

# factor for calculating the step
factor = 4.5 


for f in list(features): 
    Q1 = np.percentile(data[f], q = 25)
    Q3 = np.percentile(data[f], q = 75)
    step = (Q3 - Q1) * factor
    
    for i in range(len(data)):
        if (data[f].loc[i] > (Q3 + step)) | (data[f].loc[i] < (Q1 - step)):
            outliers_index.add(i)
            
print("There {} detected outliers".format(len(outliers_index)))


# ### Delete the outliers
# Because the number of outliers is not high we can drop them 

# In[ ]:


clean_data = data.drop(list(outliers_index), axis = 0)

#Define features and Target again
features = clean_data.drop('diagnosis', axis = 1)
target = clean_data['diagnosis']

print("The number of rows after deleting outliers is: {}".format(len(clean_data)))


# # Data Preprocessing
# #### In this section of this project I'm going to apply:
# * Apply `Feature Scaling` (Transformation). This will fix the skewness of the data 
# * Split the dataset into `Trainset` and `Testset`

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# prepare the final data for the model
final_features = scaler.fit_transform(features)


# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(final_features, target, test_size = 0.2, 
                                                    shuffle = True, random_state = 0)

print("training set size: {}, testing set size: {}".format(len(X_train), len(X_test)))


# # Model Selection
# #### In this Process I'm going to apply the following:
# * Pick up the most `appropriate models` that best-fit the problem I'm trying to solve.
# * Define the required metrics that need to evaluate the model performance.
# * Filter these models by testing them by ploting `learing curve` for each one.

# In this section I planned to list some of the best Machine Learning Classifiers and see which ones are appropriate for this problem, and after that I will filter these models to pick the best one that give me the best accuracy.
# ##### These models are:
# 1. Support Vector Machine (SVM)
# 2. Logistic Regression
# 3. Random Forest Classifier
# 4. Multi-Layer Perceptron (MLP)

# In[ ]:


from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier 

# Build these models
svc_model = SVC(random_state= 40) 
logistic_model = LogisticRegression(random_state= 40)
random_model = RandomForestClassifier(random_state= 40)
mlp_model = MLPClassifier(random_state= 40)


# ### Now I'm going to plot `learning curves` and apply `Cross Validation` to help me filter 
# these models and pick the most appropriate one.
# learning curve give initial intution about which model will overfit and which will underfit and which one will do good job.

# ### Note:
# The following cell contains some functions that have built personally to assest me when ploting `Learning Curves` and applying `GridSearchCV`, `Cross-Validation` or calculating `confusion matrix`.

# In[ ]:


from sklearn.metrics import f1_score, accuracy_score
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import learning_curve, cross_val_score, GridSearchCV


def plot_learning_curve(estimators, X, y, train_sizes, scorer, cv):
    
    #calculate required number of rows in the figure 
    n_rows = np.ceil(len(estimators) / 2)
    
    #calculate the width of the figure
    y_length = n_rows * 5 + 5
    
    # Create the figure window
    fig = plt.figure(figsize=(10, y_length))
    
    for i, est in enumerate(estimators):
        sizes, train_scores, test_scores = learning_curve(est, X, y, 
                                                          cv = cv, train_sizes = train_sizes, scoring = scorer)
        
        #print the done precentage
        print("Precentage of work done: {}%".format((i + 1) * 100 / len(estimators)))
        
        #get estimator name for title setting
        est_name = est.__class__.__name__
        
        # average train_scores and test_scores
        train_mean = np.mean(train_scores, axis = 1)
        test_mean = np.mean(test_scores, axis = 1)
        
        #Create subplots
        ax = fig.add_subplot(n_rows, 2, i + 1)
        ax.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
        ax.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
        
        #add texts 
        ax.set_title(est_name)
        ax.set_xlabel('Number of Training Points')
        ax.set_ylabel('Score')
       
    # Visual aesthetics
    ax.legend(bbox_to_anchor=(1.05, 1.8), loc='lower left', borderaxespad = 0.)
    fig.suptitle('Learning Performances for Multiple Models', fontsize = 16, y = 1.03)
    fig.show()

def multi_cross_val(estimators, X, y, cv, scoring):
    
    scores = []
    
    for est in estimators:
        S = cross_val_score(est, X, y, cv =cv, scoring = scoring)
        scores.append(S)
        
    return scores

def cal_confusion_matrix(y_true, pred):
    POS = 0
    true_pos = 0
    
    NEG = 0
    true_neg = 0
    for i, element in enumerate(y_true):

        if element == 1:
            POS += 1
            if pred[i] == 1:
                true_pos += 1
        else:
            NEG += 1
            if pred[i] == 0 :
                true_neg += 1

    false_neg = POS - true_pos
    false_pos = NEG - true_neg
    
    return ['True Positive', 'False Positive', 'False Negative','True Negative'], [true_pos, false_pos, false_neg, true_neg]
    
def multi_grid_search(estimators, X, y, params, cv, scoring):
    
    grids = []
    
    for i, est in enumerate(estimators):
        
        #Define the grid search object
        grid_obj = GridSearchCV(est, param_grid = params[i], cv = cv, scoring = scoring)
        grid_obj.fit(X, y)
        grids.append(grid_obj)
        #print the done precentage
        print("Precentage of work done: {}%".format((i + 1) * 100 / len(estimators)))
   
    #return grid_obj
    return grids


# In[ ]:


from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, f1_score


cv= KFold(n_splits = 10, shuffle = True, random_state = 0) 
train_sizes= [20, 40, 80, 180, 300, 390]
scorer= make_scorer(f1_score)

plot_learning_curve([svc_model, logistic_model, random_model, mlp_model], X_train, y_train, train_sizes, scorer, cv)


# In[ ]:


# apply cross_val_score for SVC and MLPClassifier
scores = multi_cross_val([svc_model, logistic_model, random_model, mlp_model], X_train, y_train, cv, scorer)

#get the average of the scores
scores = np.mean(scores, axis = 1)

print("The average scores for SVC is: {} and for LogisticRegression is: {}".format(scores[0], scores[1]))
print("The average score for RandomForest is: {} and for MLPClassifier is: {}".format(scores[2], scores[3]))


# From the above learning curves and Cross Validation I will pick `SVC` and `LogisticRegression` because they both gives high accuracy in learning curves and Cross Validation.
# So I'm going to take these two models and filter them later in the project.

# # Fine Tune the Model
# In this sectoin of this project I'm going to fine tune the model's hyperparameters using `Grid Search` Technique in order to improve the performance of the model.
# For this purpose I have built a method called `multi_grid_search` in functions file in order to apply GrideSearchCV for multiple models to apply it for the filtered models.

# In[ ]:


#prepare svc's paramters
svc_params = {'C': [1, 2, 2.5, 3], 'kernel': ['linear', 'poly', 'rbf']}

#prepare mlp's prarmeters
logistic_params = {'penalty': ['l1','l2'], 'C': [0.09, 0.1, 0.5, 1, 2], 'max_iter': [75, 100, 200, 500]}

#apply GrideSearchCv for both models
grids = multi_grid_search([svc_model, logistic_model], X_train, y_train, [svc_params, logistic_params], cv, scorer)


# In[ ]:


# get the best svc
best_svc = grids[0].best_estimator_

# get the best logistic model 
best_logistic = grids[1].best_estimator_


# # Model Evaluation
# In this section of the project I'm going to measure the performance of the final model(s) to make sure the final one can go to the light.

# In this section I'm going to implement the following:
# * Implement cross validation and test the models on the testset in order to pick the best one of them.
# * Plot confusion Matrix to detect `false positive` and `false_negative`

# In[ ]:


# apply cross_val_score for SVC and MLPClassifier
scores = multi_cross_val([best_svc, best_logistic], X_train, y_train, cv, scorer)

#get the average of the scores
scores = np.mean(scores, axis = 1)

print("The average score for SVC is: {} and for LogisticRegression is: {}".format(scores[0], scores[1]))


# As you noticed the SVC beats the LogisticRegression model by 1%, but further I'm going to test both on testset to make sure that SVC still generalize well.

# In[ ]:


# predict on testset 
svc_pred = best_svc.predict(X_test)
logistic_pred = best_logistic.predict(X_test)

#calculate f1_score for both predictions to decide the winner
svc_score = f1_score(y_test, svc_pred)
logistic_score = f1_score(y_test, logistic_pred)

print("The test score for SVC is: {} and for LogisticRegression is: {}".format(svc_score, logistic_score))


# ### From the above evaluations on trainset and testset, `SVC` have won on trainset, but LogisticRegression beats it on testset. That's make it hard to decide which is better so we can use both algorithms.

# **(Optional)** Now I'm going to try eliminate the less important featreus and keep the highest coorelated ones, trying to simplify the model. This setp will lead to less acurate model but if we came to trade-off between accuracy and speed of the model we may choose to reduce the number of features. So i'll give it a try.

# ## Refine the model input features
# In this section I will remove the least important features trying to speed-up the model perormance, this setp might seems not necessary in my case because I have a small dataset. But I'll do in the seek of figure out what would be the result.

# In[ ]:


import copy 

scaler = StandardScaler()

# prepare the final data for the model
final_features_reduced = scaler.fit_transform(features[highest_corr.index])


X_train_reduced, X_test_reduced, y_train, y_test = train_test_split(final_features_reduced,
                                                                                    target, test_size = 0.2, 
                                                                                    shuffle = True, random_state = 0)

svc = copy.copy(best_svc)
logistic = copy.copy(best_logistic)
svc.fit(X_train_reduced, y_train)
logistic.fit(X_train_reduced, y_train)


# In[ ]:


# apply cross_val_score for SVC and MLPClassifier
scores = multi_cross_val([svc, logistic], X_train_reduced, y_train, cv, scorer)

#get the average of the scores
scores = np.mean(scores, axis = 1)

print("The average score for SVC is: {}, and for Logistic is: {} ".format(scores[0], scores[1]))


# In[ ]:


# predict on testset 
svc_pred_reduced = svc.predict(X_test_reduced)
logistic_pred_reduced = logistic.predict(X_test_reduced)

# Calculate f1_score for both predictions to decide the winner
svc_score = f1_score(y_test, svc_pred_reduced)
logistic_score = f1_score(y_test, logistic_pred_reduced)

print("The test score for SVC is: {} , and for Logistic is: {}".format(svc_score, logistic_score))


# ### That's Good! 
# As we see after removing the less important features the model's accuracy decreased a little on both trainset and testset.
# This trade-off in this size of the dataset not important because we have small dataset, but if we increased the number of rows to be high (like 20,000 - 50,000) we may prefere to eliminate these less important features
# ---

# # Accuracy Visualization
# ### Final Step
# In this final step I'm going to calculate the `confusion matrix` in order to detect the two types of errors `False Positive` and `False Negative`. 
# For sure we care much about `False Negatives` because it's more risky as it sends the patient to home pretending he/she is safe and don't suffer anything, and because of this this type of error more risky so we wish to cut it out.

# In[ ]:



# apply first on trainset
elements, train_confusion = cal_confusion_matrix(y_train, best_svc.predict(X_train))

# apply first on trainset
elements, test_confusion = cal_confusion_matrix(y_test, best_svc.predict(X_test))


# #### Plot the confution Matrix for trainset and testset using seaborn

# In[ ]:


#Build dataframe for train_confusion
confusion_train = pd.DataFrame(index = ['Predict Positive', 'Predict Negative'], 
                          columns = ['Actual Positive', 'Actual Negative'])

#Assign values for corresponding rows and columns
confusion_train['Actual Positive'] = train_confusion[0], train_confusion[2]
confusion_train['Actual Negative'] = train_confusion[1], train_confusion[3]

#Build dataframe for test_confusion
confusion_test = pd.DataFrame(index = ['Predict Positive', 'Predict Negative'], 
                          columns = ['Actual Positive', 'Actual Negative'])

#Assign values for corresponding rows and columns
confusion_test['Actual Positive'] = test_confusion[0], test_confusion[2]
confusion_test['Actual Negative'] = test_confusion[1], test_confusion[3]


# In[ ]:


#plot confusion for trainset
plt.figure(figsize = (15, 8))
plt.suptitle("Confusion Matrix for Trainset", fontsize = 30)
sns.heatmap(confusion_train, annot = True, fmt = 'd')
plt.show()


# In[ ]:


#plot confusion for trainset
plt.figure(figsize = (15, 8))
plt.suptitle("Confusion Matrix for Testset", fontsize = 30)
sns.heatmap(confusion_test, annot = True, fmt = 'd')
plt.show()


# ## Conclusion
# ##### The total `False Postives` for trainset and testset is `7`  and got zero `False Negatives` so that's great result.
# ---
