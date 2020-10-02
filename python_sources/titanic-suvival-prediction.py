#!/usr/bin/env python
# coding: utf-8

# Titanic: Machine Learning Problem to predict the survival on the titanic

# **Competition Description** : *The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.*
# 
# *One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.*
# 
# *In this challenge, we tend to complete the analysis of what sorts of people were likely to survive. In particular, we try to apply the tools of machine learning to predict which passengers survived the tragedy.*

# **Objective**:
# *Predict the Survival of the onboard passengers. How much likely they can survive.*

# PLEASE UPVOTE IF YOU FIND THIS KERNEL HELPFUL FOR YOUR ANALYSIS IN ANY WAY.

# > Loading the important libraries

# In[25]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os
import tensorflow as tf
import re as re

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB 
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import StackingCVClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
print("# Imported the libaries")


# Imported the libaries.. Now check the input files and their sizes

# In[2]:


dir_contents = os.listdir('../input/')
print("Dataset contents : {}".format(dir_contents))
print("Size of Training dataset : " + str(round(os.path.getsize('../input/train.csv')/(1024*2), 2)) +" KB")
print("Size of Testing dataset : " + str(round(os.path.getsize('../input/test.csv')/(1024*2), 2)) +" KB")


# Pretty small dataset with size less than 30Kilobytes for train set and less than 15KB for testing set. Considering the size of dataset, it is going to be really tough to fetch appropriate and great results.
# 
# Now that we have keys to open the locks of the door, let's get our hands dirty to post mortem the dataset and solve this god damn problem.

# In[3]:


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

passengerIds = test_data['PassengerId']
print("Shape of Train dataset : {}".format(train_data.shape))
print("Shape of Test dataset : {}".format(test_data.shape))


# This is really a small dataset with only ~900 observations for training set and ~400 observations for testing set.

# In[4]:


train_data.head(4)


# **This dataset contains minimal features.**
# Features/Attributes - 
# 1. PassengerId - Id 
# 2. Survived - Survival Status (0 -> Died 1 -> Survived)
# 3. PClass - Ticket Class
# 4. Name - Name of the Passenger
# 5. Sex - Gender of the Passenger
# 6. Age - Age of the Passenger in years
# 7. SibSp - # of siblings / spouses aboard
# 8. Parch - # of parents / children aboard the Titanic
# 9. Ticket - Ticket Number
# 10. fare - Ticket Fare
# 11. Cabin - Cabin Number
# 12. Embarked - Port of Embarkation
# 
# Here, the Survived Column would be the Classification Column/Prediction Column/Output Column

# > Now let's do some **EXPLORATORY DATA ANALYSIS** on the dataset(Train)

# In[5]:


train_data['Survived'].value_counts(normalize=True) * 100


# **Observation**
# Around 38% of the passengers only survived. Thus it is an imbalanced dataset but not too much. If necessary, we can also try upsampling the dataset. But that's not necessary right now.

# In[6]:


print(train_data['Age'].describe())
print("***************************************")

plt.close()
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.title("Distribution Plot for Age")
sns.distplot(train_data[train_data['Age'].notnull()]['Age'], bins=50)
plt.grid()  

plt.subplot(1, 3, 2)
plt.title("PDF-CDF Plot for Age")
counts, bin_edges = np.histogram(train_data[train_data['Age'].notnull()]['Age'], bins = 50, density = True)
pdf = counts / sum(counts)
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:], pdf, label="PDF")
plt.plot(bin_edges[1:], cdf, label="CDF")
plt.legend()
plt.grid()  

plt.subplot(1, 3, 3)
plt.title("Violin Plot for Age")
sns.violinplot(x="Survived", y="Age", data=train_data)

plt.grid()    
plt.show()


# **Observation:**
# 1. As we can see, The mean 'Age' of the passengers is about 30. 
# 2. About 90% of passengers have age less than 50 years.
# 3. About 10% of passengers have age less than or equal to 10.
# 4. From the violin plots we cannot infer anything as both the plots are highly overlapping. However, if we look closely, childrens/infants which have age less than 10 have a high survival rate.

# In[7]:


train_data[(train_data['Age'] < 10) & (train_data['SibSp'] == 0) & (train_data['Parch'] == 0)]


# I seriously do not know how this is possible. This passenger is 5 year old female and do not have a sibling or even a parent on board. This is quite strange as she survived the catastrophe.
# 
# My first guess would be this observation would be a typo/error in reporting in Age or SibSp or Parch.

# In[8]:


print("# Gender counts")
print(train_data['Sex'].value_counts())
print("*************************")
print("Percentage of Males who survived   : {}%".format(round(train_data[(train_data['Sex']=='male') & 
                                                               (train_data['Survived']==1)].shape[0] * 100 
                                                           / train_data.shape[0], 2)))
print("Percentage of Females who survived : {}%".format(round(train_data[(train_data['Sex']=='female') & 
                                                               (train_data['Survived']==1)].shape[0] * 100 
                                                           / train_data.shape[0], 2)))


# **Observation:** Females survival rate from the catastrophe was more than that of males.

# **> Data Preprocessing and Feature Engineering**

# In[9]:


full_data = [train_data , test_data]
len(full_data)


# Checking for NaN, null values in whole dataset

# In[10]:


for data in full_data:
    print("**************************")
    print(data.info())
    print("**************************")


# As we can see,
# 1. Age & Cabin attributes in both dataset contains NaN/null values to a much greater extent.
# 2. Embarked attribute in Train data has 2 missing values. 
# 3. Fare attribute in Test data has one missing value.
# 
# We need to pre process such attributes.

# Let's process **Name** attribute/feature to fetch the Title

# In[11]:


train_data['Name']


# In[12]:


def getTitleFromName(nameText):
    title = str(nameText.split(', ')[1])
    title_search = re.search('([A-Za-z]+)\.', title).group(1)
    return title_search

for data in full_data:
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    
    # Introducting a new feature - Family Size
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    # Introducing a new feature - IsAlone
    data['IsAlone'] = 0
    data.loc[data['FamilySize']==1, 'IsAlone'] = 1
        
    # Introducing a new Feature - CabinAlloted
    data['CabinAllotment'] = 0
    data.loc[data['Cabin'].notnull(), 'CabinAllotment'] = 1
    
    # If data has null values in Embarked, just replace it with 'S' as 'S' is quite frequent
    data['Embarked'] = data['Embarked'].fillna('S')
    
    # Mapping Sex
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1}).astype(int)
    
    # Mapping Embarked
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
    
    # Since Age has many null/NA values, we will process it with Median values
    data['Age'] = imp.fit_transform(data[['Age', 'Sex', 'FamilySize']])[:,0].astype(int)
    
    # Since Fare has also some null/Na Values, we will process it with Median values
    data['Fare'] = imp.fit_transform(data[['Fare', 'FamilySize', 'CabinAllotment']])
    
    # Introducing a new feature - PerTicket
    data['PerTicket'] = data['Fare'] / data['FamilySize']
    data.loc[ data['PerTicket'] <= 7.25, 'PerTicket'] = 0
    data.loc[(data['PerTicket'] > 7.25) & (data['PerTicket'] <= 8.3), 'PerTicket'] = 1
    data.loc[(data['PerTicket'] > 8.3) & (data['PerTicket'] <= 23.667), 'PerTicket'] = 2
    data.loc[ data['PerTicket'] > 23.667, 'PerTicket'] = 3
    data['PerTicket'] = data['PerTicket'].astype(int)
    
    # Mapping the Age Values to Categories (Children, Youth, Adults, Senior)
    data.loc[(data['Age'] >=0) & (data['Age'] <= 14), 'Age'] = 0       #Children
    data.loc[(data['Age'] >=15) & (data['Age'] <= 24), 'Age'] = 1      #Youth
    data.loc[(data['Age'] >=25) & (data['Age'] <= 64), 'Age'] = 2      #Adults
    data.loc[data['Age'] >=65, 'Age'] = 3    #Senior
    data['Age'] = data['Age'].astype(int)
    
    #Name Feature Engineering
    data['Title'] = data['Name'].apply(getTitleFromName)
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
    data['Status'] = "General"
    data.loc[data['Title'] == 'Capt','Status'] = 'Military'
    data.loc[data['Title'] == 'Col','Status'] = 'Military'
    data.loc[data['Title'] == 'Countess','Status'] = 'Political'
    data.loc[data['Title'] == 'Don','Status'] = 'Military'
    data.loc[data['Title'] == 'Dr','Status'] = 'General'
    data.loc[data['Title'] == 'Jonkheer','Status'] = 'Political'
    data.loc[data['Title'] == 'Lady','Status'] = 'Political'
    data.loc[data['Title'] == 'Major','Status'] = 'Military'
    data.loc[data['Title'] == 'Master','Status'] = 'General'
    data.loc[data['Title'] == 'Rev','Status'] = 'Political'
    data.loc[data['Title'] == 'Sir','Status'] = 'Military'
    
    data['Rank'] = 0
    data.loc[data['Title'] == 'Capt', 'Rank'] = 1
    data.loc[data['Title'] == 'Col', 'Rank'] = 1
    data.loc[data['Title'] == 'Major', 'Rank'] = 2
    data.loc[data['Title'] == 'Don', 'Rank'] = 2
    data.loc[data['Title'] == 'Sir', 'Rank'] = 0
    data.loc[data['Title'] == 'Dr', 'Rank'] = 1
    data.loc[data['Title'] == 'Master', 'Rank'] = 0
    data.loc[data['Title'] == 'Miss', 'Rank'] = 0
    data.loc[data['Title'] == 'Mr', 'Rank'] = 0
    data.loc[data['Title'] == 'Mrs', 'Rank'] = 0
    data.loc[data['Title'] == 'Countess', 'Rank'] = 2
    data.loc[data['Title'] == 'Jonkheer', 'Rank'] = 0
    data.loc[data['Title'] == 'Lady', 'Rank'] = 1
    data.loc[data['Title'] == 'Rev', 'Rank'] = 1
    
    data['Status'] = data['Status'].map({ 'General': 0, 'Military': 1, 'Political': 2})

# Feature Selection
remove_features = ['PassengerId', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Name', 'Title', 'Fare']
train_data = train_data.drop(remove_features, axis=1)
test_data = test_data.drop(remove_features, axis=1)

train_data.head(5)


# In[15]:


y_train = train_data['Survived'].values
X_train = train_data.drop('Survived', axis=1)
X_test = test_data.values


# In[20]:


encoding_clf = OneHotEncoder()
train_data_new = encoding_clf.fit_transform(X_train).astype('int')
test_data_new = encoding_clf.transform(X_test).astype('int')


# In[30]:


print(train_data_new.shape)
print(train_data_new.shape)


# In[29]:


y_train_new = tf.keras.utils.to_categorical(y_train, num_classes=2)


# Pearson Correlation Heatmap

# In[ ]:


colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train_data.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)


# One thing that that the Pearson Correlation plot can tell us is that there are not too many features strongly correlated with one another. This is good from a point of view of feeding these features into your learning model because this means that there isn't much redundant or superfluous data in our training set and we are happy that each feature carries with it some unique information. 

# ## Neural Network on Titanic Dataset

#  

# In[32]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(35, input_shape=(35,), activation=tf.nn.relu, kernel_initializer='he_uniform'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu, kernel_initializer='he_uniform'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu,kernel_initializer='he_uniform'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu,kernel_initializer='he_uniform'))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

relu_model = model.fit(train_data_new, y_train_new, epochs=500, batch_size=7, verbose=1, validation_split=0.20)


# In[35]:


pred = model.predict_classes(test_data_new)
pred


# In[ ]:


DL_Submission = pd.DataFrame({ 'PassengerId': passengerIds,
                            'Survived': pred })
DL_Submission.to_csv("DL_Submission.csv", index=False)


# # StackingCVClassifier
# [](http://)
# *An ensemble-learning meta-classifier for stacking using cross-validation to prepare the inputs for the level-2 classifier to prevent overfitting.*
# 
# **Stacking** is an *ensemble* learning technique to combine multiple classification models via a meta-classifier. The StackingCVClassifier extends the standard stacking algorithm (implemented as StackingClassifier) using cross-validation to prepare the input data for the level-2 classifier.
# 
# In the standard stacking procedure, the first-level classifiers are fit to the same training set that is used prepare the inputs for the second-level classifier, which may lead to overfitting. The StackingCVClassifier, however, uses the concept of cross-validation: the dataset is split into k folds, and in k successive rounds, k-1 folds are used to fit the first level classifier; in each round, the first-level classifiers are then applied to the remaining 1 subset that was not used for model fitting in each iteration. The resulting predictions are then stacked and provided -- as input data -- to the second-level classifier. After the training of the StackingCVClassifier, the first-level classifiers are fit to the entire dataset as illustrated in the figure below.
# 
# <img src='http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier_files/stacking_cv_classification_overview.png'>
# 
# *More formally, the Stacking Cross-Validation algorithm can be summarized as follows :*
# 
# <img src='http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier_files/stacking_cv_algorithm.png'>[](http://)

# In[ ]:


# mask = ['Pclass', 'Sex', 'Age', 'Embarked', 'FamilySize', 'IsAlone', 'CabinAllotment', 'PerTicket', 'Status', 'Rank']
# X = train_data[mask]
# y = train_data['Survived']


# In[ ]:


# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.metrics import roc_curve, f1_score, confusion_matrix, auc
# g_xtrain, g_xtest, g_ytrain, g_ytest = train_test_split(X, y, test_size=0.30)


# Lets apply SGDClassfier(With Log loss) i.e Logistic Regression.

# In[ ]:


from sklearn.linear_model import SGDClassifier

alpha_range = list([10** i for i in range(-10, 6, 1)] + [2**i for i in range(-5, -1, 1)])
loss_range = list(['hinge', 'log', 'modified_huber', 'squared_hinge'])
penalty_range = list(['l1', 'l2', 'elasticnet'])
parameters = {'loss': loss_range, 'penalty': penalty_range, 'alpha': alpha_range}
model = SGDClassifier()
g_clf = GridSearchCV(model, parameters, cv=10, n_jobs=-1, scoring="accuracy")
g_clf.fit(g_xtrain, g_ytrain)

print("Model fitted perfectly.")
print("Best Score (TRAIN): {}".format(g_clf.best_score_))
print("Best Params       : {}".format(g_clf.best_params_))


# In[ ]:


# optimal_alpha = g_clf.best_params_['alpha']
# optimal_loss = g_clf.best_params_['loss']
# optimal_penalty = g_clf.best_params_['penalty']

# clf_model = SGDClassifier(alpha=optimal_alpha, loss=optimal_loss, penalty=optimal_penalty)
# ccv_clf = CalibratedClassifierCV(clf_model, cv=10)
# ccv_clf.fit(g_xtrain, g_ytrain)

# # Get predicted values for test data
# pred_train = ccv_clf.predict(g_xtrain)
# pred_test = ccv_clf.predict(g_xtest)
# pred_proba_train = ccv_clf.predict_proba(g_xtrain)[:,1]
# pred_proba_test = ccv_clf.predict_proba(g_xtest)[:,1]

# fpr_train, tpr_train, thresholds_train = roc_curve(g_ytrain, pred_proba_train, pos_label=1)
# fpr_test, tpr_test, thresholds_test = roc_curve(g_ytest, pred_proba_test, pos_label=1)
# conf_mat_train = confusion_matrix(g_ytrain, pred_train, labels=[0, 1])
# conf_mat_test = confusion_matrix(g_ytest, pred_test, labels=[0, 1])
# f1_sc = f1_score(g_ytest, pred_test, average='binary', pos_label=1)
# auc_sc_train = auc(fpr_train, tpr_train)
# auc_sc = auc(fpr_test, tpr_test)

# print("Optimal Alpha: {} with Penalty: {} with AUC: {:.2f}%".format(optimal_alpha, optimal_penalty, float(auc_sc*100)))



# plt.figure(figsize=(13,7))
# # Plot ROC curve for training set
# plt.subplot(2, 2, 1)
# plt.title('Receiver Operating Characteristic - TRAIN SET')
# plt.plot(fpr_train, tpr_train, color='red', label='AUC - Train - {:.2f}'.format(float(auc_sc_train * 100)))
# plt.plot([0, 1], ls="--")
# plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.grid()
# plt.legend(loc='best')

# # Plot ROC curve for test set
# plt.subplot(2, 2, 2)
# plt.title('Receiver Operating Characteristic - TEST SET')
# plt.plot(fpr_test, tpr_test, color='blue', label='AUC - Test - {:.2f}'.format(float(auc_sc * 100)))
# plt.plot([0, 1], ls="--")
# plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.grid()
# plt.legend(loc='best')

# #Plotting the confusion matrix for train
# plt.subplot(2, 2, 3)
# plt.title('Confusion Matrix for Training set')
# df_cm = pd.DataFrame(conf_mat_train, index = ["Negative", "Positive"],
#                   columns = ["Negative", "Positive"])
# sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')

# #Plotting the confusion matrix for test
# plt.subplot(2, 2, 4)
# plt.title('Confusion Matrix for Testing set')
# df_cm = pd.DataFrame(conf_mat_test, index = ["Negative", "Positive"],
#                   columns = ["Negative", "Positive"])
# sns.heatmap(df_cm, annot=True,cmap='Blues', fmt='g')

# plt.tight_layout()
# plt.show()


# In[ ]:


# final_clf = SGDClassifier(alpha=optimal_alpha, loss=optimal_loss, penalty=optimal_penalty)
# final_clf.fit(X,y)

# pred = final_clf.predict(test_data)

# SGDSubmission = pd.DataFrame({ 'PassengerId': passengerIds,
#                             'Survived': pred })
# SGDSubmission.to_csv("SGDSubmission.csv", index=False)
# os.listdir()


# In[ ]:


# # Initializing models
# clf1 = SGDClassifier(n_jobs=-1)
# clf2 = GradientBoostingClassifier(n_estimators=50)
# clf3 = RandomForestClassifier(n_estimators=50, n_jobs=-1)
# clf4 = AdaBoostClassifier(n_estimators=50)
# meta_clf = XGBClassifier(n_estimators=100, n_jobs=-1)

# sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3, clf4], 
#                             meta_classifier=meta_clf)

# params = {
#             'sgdclassifier__alpha': alpha_range,
#             'sgdclassifier__loss': loss_range,
#             'sgdclassifier__penalty': penalty_range,
#             'randomforestclassifier__max_depth': list([5, 10, 15]),
#             'meta-xgbclassifier__max_depth': list([3, 7, 11])
#         }

# grid = GridSearchCV(estimator=sclf, 
#                     param_grid=params, 
#                     cv=10, n_jobs=-1, scoring='accuracy',
#                     refit=True)
# grid.fit(X, y)

# print('Best parameters: %s' % grid.best_params_)
# print('Accuracy: %.2f' % grid.best_score_)


# In[ ]:


# # Initializing models
# clf1 = KNeighborsClassifier()
# clf2 = RandomForestClassifier()
# clf3 = MultinomialNB()
# clf4 = SVC(kernel='rbf')
# clf5 = LogisticRegression()
# clf6 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME")
# meta_clf = XGBClassifier(n_jobs=-1)

# sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5, clf6], 
#                             meta_classifier=meta_clf)

# params = {
#             'kneighborsclassifier__n_neighbors': range(1, 50, 5),
#             'multinomialnb__alpha': [10**i for i in range(-5, 4, 1)],
#             'svc__C': [10**i for i in range(-5, 4, 1)],
#             'logisticregression__C': [10**i for i in range(-5, 4, 1)],
#         }

# grid = GridSearchCV(estimator=sclf, 
#                     param_grid=params, 
#                     cv=10, n_jobs=-1, scoring='accuracy',
#                     refit=True)
# grid.fit(X, y)

# print('Best parameters: %s' % grid.best_params_)
# print('Accuracy: %.2f' % grid.best_score_)


# In[ ]:


# #Predict from the Test Data
# pred = grid.predict(test_data)


# In[ ]:


# StackingSubmission = pd.DataFrame({ 'PassengerId': passengerIds,
#                             'Survived': pred })
# StackingSubmission.to_csv("StackingSubmission.csv", index=False)
# os.listdir()

