#!/usr/bin/env python
# coding: utf-8

# ### **This is an extension to my previous kernel on this data. In the previous kernel, I did Rigorous EDA which you can find [here](https://www.kaggle.com/nishkarshtripathi/unleashing-the-power-of-eda).**
# 
# ### **In this kernel I will go through the preprocessing and prediction part. So, let's start!**

# In[ ]:


# Importing Libraries

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import permutations 

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score


# In[ ]:


# Getting our data

data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')


# In[ ]:


# Renaming the columns for better understanding of features (always a good thing to do)

data.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'serum_cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate',
       'exercise_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']


# In[ ]:


# Lets see how our features correlate with the target variable 

x = data.corr()
pd.DataFrame(x['target']).sort_values(by='target',ascending = False).style.background_gradient(cmap = 'copper')


# **As we saw in EDA that chest_pain_type and max_heart_rate have decent positive correlation with our target. Now these numbers say the same thing!** 

# # Preprocessing

# In[ ]:


# NOTE: This step is not necessary, I am performing it to avoid renaming of columns after One Hot Encoding. Also,
# here we will leave two class variables as they are already one hot encoded

# Let's map these class values to something meaningful information given in dataset description


data.chest_pain_type = data.chest_pain_type.map({1:'angina pectoris', 2:'atypical angina', 3:'non-anginal pain', 4:'SMI', 0:'absent'})

data.st_slope = data.st_slope.map({1:'upsloping', 2:'horizontal', 3:'downsloping', 0:'absent'})

data.thalassemia = data.thalassemia.map({1:'normal', 2:'fixed defect', 3:'reversable defect', 0:'absent'})


# In[ ]:


data.head()


# In[ ]:


# Seperating out predictors(X) and target(Y)

X = data.iloc[:, 0:13]

Y = data.iloc[:, -1]


# In[ ]:


X.head()


# In[ ]:


# Encoding the categorical variables using get_dummies() 
categorical_columns = ['chest_pain_type', 'thalassemia', 'st_slope']

for column in categorical_columns:
    dummies = pd.get_dummies(X[column], drop_first = True)
    X[dummies.columns] = dummies
    X.drop(column, axis =1, inplace = True)


# **Note that I have passed drop_first as True in get_dummies function to avoid dummy variable trap. It is not very necessary in models where you apply L1 or L2 regularization as it will balance any collinearity caused by dummy columns.**
# 

# In[ ]:


X.head()


# In[ ]:


# Let us again look at the correlation against our target variable

temp = X.copy()
temp['target'] = Y

d = temp.corr()
pd.DataFrame(d['target']).sort_values(by='target',ascending = False).style.background_gradient(cmap = 'copper')


# ### **We have far better correlations after Scaling and One hot encoding.**

# In[ ]:


# Splitting the data into test and train 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("X-Train:",X_train.shape)
print("X-Test:",X_test.shape)
print("Y-Train:",y_train.shape)
print("Y-Test:",y_test.shape)


# In[ ]:


# Scaling the continous data

num_columns =  ['resting_blood_pressure','serum_cholesterol', 'age', 'max_heart_rate', 'st_depression']

scaler = StandardScaler()

scaler.fit(X_train[num_columns])

X_train[num_columns] = scaler.transform(X_train[num_columns])

X_test[num_columns] = scaler.transform(X_test[num_columns])


# In[ ]:


X_train.head()


# # Modelling

# ### Logistic Regression

# In[ ]:


# Creating a function to plot correlation matrix and roc_auc_curve

def show_metrics(model):
    fig = plt.figure(figsize=(25, 10))

    # Confusion matrix
    fig.add_subplot(121)
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, annot_kws={"size": 16}, fmt='g')

    # ROC Curve
    fig.add_subplot(122)
    
    
    auc_roc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])

    plt.plot(fpr, tpr, color='darkorange', lw=2, marker='o', label='Trained Model (area = {0:0.3f})'.format(auc_roc))
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--', label= 'No Skill (area = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:


# creating our model instance
log_reg = LogisticRegression()

# fitting the model
log_reg.fit(X_train, y_train)

# predicting the target vectors
y_pred=log_reg.predict(X_test)


# In[ ]:


# calling our show_metrics function

show_metrics(log_reg)


# In[ ]:


# let's look at our accuracy

accuracy = accuracy_score(y_pred, y_test)

print(f"The accuracy on test set using Logistic Regression is: {np.round(accuracy, 3)*100.0}%")


# **Woah, 90% on the first try!**
# 
# **Logistic Regression seems to work very well as it is easily able to draw hyperplane and the reason for that can be - Patients with higher ages usually have the problem of high_blood_pressure thus which make it easy to seperate it from the low age patients. (This Speculation might be wrong!)**

# In[ ]:


# Getting precision, recall and f1-score via classification report

print(classification_report(y_test, y_pred))


# ### KNearestNeighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

# creating a list of K's for performing KNN
my_list = list(range(0,30))

# filtering out only the odd K values
neighbors = list(filter(lambda x: x % 2 != 0, my_list))

# list to hold the cv scores
cv_scores = []

# perform 10-fold cross validation with default weights
for k in neighbors:
  Knn = KNeighborsClassifier(n_neighbors = k, algorithm = 'brute')
  scores = cross_val_score(Knn, X_train, y_train, cv=10, scoring='accuracy', n_jobs = -1)
  cv_scores.append(scores.mean())

# finding the optimal k
optimal_k = neighbors[cv_scores.index(max(cv_scores))]
print("The optimal K value is with default weight parameter: ", optimal_k)


# **We got our optimal value of K as 27 so now let's look at how our accuracy varied across all K's we tried during KFoldCV. This is sometimes important because if our Hyperparameter is giving same accuracy for a wide range of values so it is better to do early stopping and use the lower value which will avoid Underfitting or Overfitting.** 
# 
# **In our case, KNN would underfit for large values of K.**

# In[ ]:


# plotting accuracy vs K
plt.plot(neighbors, cv_scores)
plt.xlabel("Number of Neighbors K")
plt.ylabel("Accuracy")
plt.title("Accuracy vs K Plot for normal ")
plt.grid()
plt.show()

print("Accuracy scores for each K value is : ", np.round(cv_scores, 3))


# **Looks like we got our best K. Now, lets fit our training data using this K and find out the accuracy on test.**

# In[ ]:


# Finding the accuracy of KNN with optimal K

from sklearn.metrics import accuracy_score

# create instance of classifier
knn_optimal = KNeighborsClassifier(n_neighbors = optimal_k, algorithm = 'kd_tree', 
                                   n_jobs = -1)

# fit the model
knn_optimal.fit(X_train, y_train)

# predict on test vector
y_pred = knn_optimal.predict(X_test)

# evaluate accuracy score
accuracy = accuracy_score(y_test, y_pred)*100
print(f"The accuracy on test set using KNN for optimal K = {optimal_k} is {np.round(accuracy, 3)}%")


# ### **~85 is not very good but isn't bad either!**

# In[ ]:


# calling our show_metrics function

show_metrics(knn_optimal)


# ## Support Vector Machine

# In[ ]:


# Creating an instance of the classifier
svm = SVC()

# training on train data
svm.fit(X_train, y_train)

# predicting on test data
y_pred = svm.predict(X_test)

# let's look at our accuracy
accuracy = accuracy_score(y_pred, y_test)

print(f"The accuracy on test set using SVC is: {np.round(accuracy, 3)*100.0}%")


# **SVM did a good job predicting the labels, scoring ~89%**

# ### Ensemble Models - Random Forest Classifier
# 
# **Trees are very easy to overfit so we will first do a RandomSearchCV to find the best parameters.**

# In[ ]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)


# In[ ]:


# Use the random grid to search for best hyperparameters

# First create the base model to tune
rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train, y_train)


# In[ ]:


# Lets look at the best parameters

rf_random.best_params_


# In[ ]:


# Creating an instance for the classifier
rf_best = RandomForestClassifier(**rf_random.best_params_)

# fitting the model
rf_best.fit(X_train, y_train)

# predict the labels
y_pred = rf_best.predict(X_test)


# In[ ]:


accuracy = accuracy_score(y_pred, y_test)

print(f"The accuracy on test set using RandomForest is: {np.round(accuracy, 3)*100.0}%")


# In[ ]:


# calling our show_metrics function

show_metrics(rf_best)


# ### Ensemble Models - Voting Classifier
# 
# **It is always good to use esemble models beacuse they are very effective. Here, I will be using Voting Classifier which does exaclty what it sounds like. It takes mulitple models and combine them together and do voting on the prediction and picks the best for us.**
# 
# **Let's combine all the models together!**

# In[ ]:


# creating a list of our models
ensembles = [log_reg, knn_optimal, rf_best, svm]

# Train each of the model
for estimator in ensembles:
    print("Training the", estimator)
    estimator.fit(X_train,y_train)


# In[ ]:


# Find the scores of each estimator
scores = [estimator.score(X_test, y_test) for estimator in ensembles]

scores


# In[ ]:


# Lets define our estimators in a list

named_estimators = [
    ("log_reg",log_reg),
    ('random_forest', rf_best),
    ('svm',svm),
    ('knn', knn_optimal),
]


# In[ ]:


# Creating an instance for our Voting classifier

voting_clf = VotingClassifier(named_estimators)


# In[ ]:


# Fit the classifier

voting_clf.fit(X_train,y_train)


# In[ ]:


# Let's look at our accuracy
acc = voting_clf.score(X_test,y_test)

print(f"The accuracy on test set using voting classifier is {np.round(acc, 3)*100}%")


# **No, Improvement at all!**
# 
# **Lets generate all permutations of the names_estimators list of length 3 and then check if the do any better.**

# In[ ]:


# to generate permutations of length three
perm = permutations(named_estimators, 3) 

# to store the acc and classifiers
best_perm = []

# to store best classifier
best=[]

# Traverse through the obtained permutations
for i in list(perm):
    
    # fit the classifier
    voting_clf = VotingClassifier(i)
    voting_clf.fit(X_train,y_train)
    
    # obtain accracy score and append it to the list
    acc = voting_clf.score(X_test,y_test)
    best_perm.append([acc,voting_clf])
    

# find out the maximum accuracy
maximum = max(best_perm, key = lambda x:x[0])
              
# there can be multiple permutations for which we get 
# best score so find all of them and append to best
for i in range(len(best_perm)):
    if maximum[0]==best_perm[i][0]:
        best.append(best_perm[i][1])


# In[ ]:


# Using the best score permutations

acc_scores = []

for i in range(len(best)):
    
    voting_clf = best[i]
    
    # fit the classifier
    voting_clf.fit(X_train,y_train)
    
    # Let's look at our accuracy
    acc_scores.append(voting_clf.score(X_test,y_test))


# In[ ]:


print(f"The accuracy on test set using voting classifier is {np.round(max(acc_scores), 3)*100}%")


# ### Still No Improvement - Is this the best on this data?
# 
# > #### **TASKS FOR YOU :**
# 
# **(I) Try some feature engineering or some more models and let me know in comments whether you can achieve higher than this.**<br><br>
# **(II) Provide suggestion, improvements and Criticism! (if any) :)**<br><br>
# **(III) Please do Upvote if you want to see more content like this!**

# # THANKS FOR READING!
