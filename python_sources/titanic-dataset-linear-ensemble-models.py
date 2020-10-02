#!/usr/bin/env python
# coding: utf-8

# # **Titanic Dataset: Linear & Ensemble Models**

# This is my first published notebook. In this notebook I focused on trying to find generate some new features and ultimately finding the best fitting model using **Linear classifiers (Logistic Regression and Linear SVC) and Ensemble models (Random Forest classifier and Voting classifier)** by finding the best hyper parameters using **Grid Search CV**. 
# 
# This notebook does not focus on EDA because I wanted to keed things simple and specific to the target I wanted to achieve with this notebook. I have tried my best to present the things in simplest way and to avoid errors. But in case anyone finds some error or some points by which I can improve my model I would love to know it in the comments.
# 
# I have taken some suggestions from this [notebook](https://www.kaggle.com/startupsci/titanic-data-science-solutions) which were very helpful.

# # Importing Libraries

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12,8


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report


# # Importing Data

# In[ ]:


train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')

train_df.head()


# In[ ]:


test_df.head()


# In[ ]:


print(train_df.info())
print('_'*40 + '\n')
print(test_df.info())


# In[ ]:


print('Train dataset has only {} unique tickets'.format(len(train_df['Ticket'].unique())))
print('-'*40)
print('Test dataset has only {} unique tickets'.format(len(test_df['Ticket'].unique())))


# In[ ]:


new_train_df = train_df.copy()
new_test_df = test_df.copy()


# # Custom Fuctions

# We can see that in the name column, the name of the passengers follows their title. Titles can be very useful in predicting age of the people as we will see later. So here we create a custom function which will help in extracting the titles from names.

# In[ ]:


def total_famil_members(df):
    return df['SibSp']+df['Parch']+1


# In[ ]:


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    else:
        return ""


# # Creating Features

# So now we start creating some new features which will be helpful later on. The thought behind the need of these features is as follows:
# 1. **Members** - We can see that we have two columns as (Parch and SibSp) which represent the number of parents and children and (Siblings and spouse) respectively. So we can create a new feature as Members which represent total number of members of same family traveling together.
# 2. **Adjusted_Fare** - As we saw before there are only a few unique ticket numbers amongst all the tickets. From this it is reasonable to say that the members from the same family are traveling with the same ticket. So the ticket probably is for a family rather than individuals. This means that ticket fare is also for family rather than individuals and thereby to save our model from this bias we need to create a new feature to account for the fare per family member.

# In[ ]:


for data in [new_train_df, new_test_df]:
    
    data['Members'] = total_famil_members(data)
    data['Adjusted_Fare'] = data['Fare']/data['Members']

    data['Title'] = data['Name'].apply(get_title)


# In[ ]:


for data in [new_train_df, new_test_df]:
    
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')


# We can easily visualize that most of the people embarked from Southampton(S) to it is safe to fill the missing embarked values with the same.

# In[ ]:


sns.countplot('Embarked', data=new_train_df)


# In[ ]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for data in [new_train_df, new_test_df]:   
    
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)
    
    data['Embarked'] = data['Embarked'].fillna('S')    
    data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


# In[ ]:


for data in [new_train_df, new_test_df]:
    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)


# In[ ]:


new_train_df.head()


# In[ ]:


new_test_df.head()


# In[ ]:


new_train_df.info()


# In[ ]:


new_test_df.info()


# # Model to predict age

# In[ ]:


fig,ax = plt.subplots(2,2)
sns.boxplot(x='Title', y='Age', data=new_train_df[new_train_df['Sex']==0], ax=ax[0][0])
sns.boxplot(x='Title', y='Age', data=new_train_df[new_train_df['Sex']==1], ax=ax[0][1])
sns.boxplot(x='Title', y='Age', data=new_test_df[new_test_df['Sex']==0], ax=ax[1][0])
sns.boxplot(x='Title', y='Age', data=new_test_df[new_test_df['Sex']==1], ax=ax[1][1])
ax[0][0].set_title(label='Female Age distribution(Train)')
ax[0][1].set_title(label='Male Age distribution(Train)')
ax[1][0].set_title(label='Female Age distribution(Test)')
ax[1][1].set_title(label='Male Age distribution(Test)')
plt.tight_layout()
plt.show()


# **We can see the distributions of age according to titles in both test and train datasets. Also we can see that distributions are also similar in most datasets.**
# 
# **Now we know that the age column has a lot of missing values. We can fill these values with the median age values of the corresponding title and sex. This seems intuitive but we can see that we also have a lot of features (Like no of parents, children, siblings and spouse) available which can help is in building a regression model to predict the age for a given person.**
# 
# **So we choose some features which can help in predicting a person's age and build regression model on it. Also since the distribution of age in train and test datasets is almost similar. We can use non-null age values of train dataset (as it has almost twice non null values as compared to test data) as training data for one and non-null age values of test dataset as testing data for our regression model.**
#     
# **This saves us the labour of concatenating the non-null values first and then applying the train-test-split to achieve almost similar result**

# In[ ]:


data_train = new_train_df[~new_train_df['Age'].isnull()]
data_test = new_test_df[~new_test_df['Age'].isnull()]


# In[ ]:


data_train.head()


# In[ ]:


X_train = data_train[['Pclass','Sex','SibSp','Parch','Title']]
y_train = data_train['Age']

X_test = data_test[['Pclass','Sex','SibSp','Parch','Title']]
y_test = data_test['Age']

X_train.head()


# In[ ]:


model_age_prediction = RandomForestRegressor(n_estimators=900, max_depth=6, min_samples_leaf=0.001, random_state=100)
model_age_prediction.fit(X_train, y_train)


# In[ ]:


y_predict = model_age_prediction.predict(X_test)


# In[ ]:


fig1,ax1 = plt.subplots(1,2)
ax1[0].scatter(y_train, model_age_prediction.predict(X_train))
ax1[1].scatter(y_test, y_predict)

ax1[0].set_title('Train data vs predictions train data')
ax1[1].set_title('Test data vs predictions test data')


# **Here we can see that the predictions are very close to the actual values. Although the predictions are not too good as clear with the r2 score but still it makes much more sense to use this model to fill missing values insted of randomly filling the values or using the summary statistics to accomplish the same task.**

# In[ ]:


print('test score is: {}'.format(r2_score(y_test, y_predict)))
print('training score is: {}'.format(r2_score(y_train, model_age_prediction.predict(X_train))))


# In[ ]:


train_missing_predicted = model_age_prediction.predict(new_train_df[new_train_df['Age'].isnull()][['Pclass','Sex','SibSp','Parch','Title']])
test_missing_predicted = model_age_prediction.predict(new_test_df[new_test_df['Age'].isnull()][['Pclass','Sex','SibSp','Parch','Title']])


# In[ ]:


new_train_df['Age'][np.isnan(new_train_df['Age'])] = train_missing_predicted
new_test_df['Age'][np.isnan(new_test_df['Age'])] = test_missing_predicted


# In[ ]:


print(new_train_df.info())
print('_'*40)
print(new_test_df.info())


# In[ ]:


new_test_df[new_test_df['Adjusted_Fare'].isna()]


# In[ ]:


new_test_df['Adjusted_Fare'] = new_test_df.groupby('Pclass')['Adjusted_Fare'].transform(lambda x: x.fillna(x.median()))
new_test_df.info()


# In[ ]:


new_test_df['Fare'] = new_test_df['Adjusted_Fare']*new_test_df['Members']
new_test_df.info()


# # Models

# In[ ]:


new_train_df.columns


# **Since we have created some features from the existing features. Using the same features again will just add complexity to our model. So we drop these features.**

# In[ ]:


data_model = new_train_df[['Survived', 'Pclass', 'Sex', 'Age', 'Embarked', 'Members', 'Adjusted_Fare', 'Title']]
data_model.head()


# In[ ]:


data_model.info()


# In[ ]:


sns.heatmap(data_model.corr(), annot=True, cmap='viridis')


# In[ ]:


train_X, test_X, train_y, test_y = train_test_split(data_model.drop(['Survived'], axis=1), data_model['Survived'],                                                    stratify=data_model['Survived'], random_state=123, test_size=0.25)


# **Now we start building the models and it is very important to keep a track of the train and test scores as too much difference between the two will mean that the model is overfitting. We will keep track of both the scores along with cross validation score using pandas DataFrames.**

# ### Logistic Regression

# In[ ]:


lr = LogisticRegression()
params_lr = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}


# In[ ]:


lr_cv = GridSearchCV(lr, params_lr, n_jobs=-1, cv=5)


# In[ ]:


lr_cv.fit(train_X, train_y)


# In[ ]:


lr_cv_results = pd.DataFrame(lr_cv.cv_results_)
lr_cv_results.head()


# In[ ]:


lr_cv_best_model = pd.DataFrame({'Params':lr_cv.best_params_.values(), 'best_score':lr_cv.best_score_,                                'Train Score':accuracy_score(train_y, lr_cv.best_estimator_.predict(train_X)),                                'Test Score': accuracy_score(test_y, lr_cv.best_estimator_.predict(test_X))})


# In[ ]:


lr_cv_best_model.head()


# ### Linear SVC

# In[ ]:


svc = LinearSVC()
params_svc = params_lr


# In[ ]:


svc_cv = GridSearchCV(svc, params_svc, n_jobs=-1)


# In[ ]:


svc_cv.fit(train_X, train_y)


# In[ ]:


svc_cv_results = pd.DataFrame(svc_cv.cv_results_)
svc_cv_results.head()


# In[ ]:


svc_cv_best_model = pd.DataFrame({'Params':svc_cv.best_params_.values(), 'best_score':svc_cv.best_score_,                                'Train Score':accuracy_score(train_y, svc_cv.best_estimator_.predict(train_X)),                                'Test Score': accuracy_score(test_y, svc_cv.best_estimator_.predict(test_X))})


# In[ ]:


svc_cv_best_model.head()


# ### Random Forest

# **Here in case of Random Forest I have not used GridSearchCv because CART models are prone to overfitting. So there is a huge chance that GridSearchCV will be returning a model with good cv score but surely wont be the best model. To tackle this we run loops over different values of hyperparameters and keep track of both train and test scores to find out the best model**

# In[ ]:


max_depth =[]
min_samples_leaf = []
cv_rf_scores = []
test_roc_scores = []
train_roc_scores = []
test_acc_scores = []
train_acc_scores = []
for depth in [5,6,7,8,9]:
    for samples_leaf in [0.009, 0.01]:
        rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, max_features='log2', max_depth=depth, min_samples_leaf=samples_leaf, random_state=100)
        rf.fit(train_X, train_y)
        cv_rf_scores.append(cross_val_score(rf, train_X, train_y, cv=10).mean())
        max_depth.append(depth)
        min_samples_leaf.append(samples_leaf)
        test_roc_scores.append(roc_auc_score(test_y, rf.predict(test_X)))
        test_acc_scores.append(accuracy_score(test_y, rf.predict(test_X)))
        train_roc_scores.append(roc_auc_score(train_y, rf.predict(train_X)))
        train_acc_scores.append(accuracy_score(train_y, rf.predict(train_X)))


# In[ ]:


rf_cv_scores = pd.DataFrame({'Max_depth':max_depth, 'Min_samples_leaf':min_samples_leaf, 'CV_Scores':cv_rf_scores,                             'Test_roc_score':test_roc_scores, 'Train_roc_scores':train_roc_scores,                             'Test_acc_score':test_acc_scores, 'Train_acc_scores':train_acc_scores})


# In[ ]:


rf_cv_scores_sorted = rf_cv_scores.sort_values(by='CV_Scores').reset_index()
rf_cv_scores_sorted.head(18)


# In[ ]:


plt.plot(rf_cv_scores_sorted.sort_values(by='CV_Scores')['Test_acc_score'], label='Test accuracy score')
plt.plot(rf_cv_scores_sorted.sort_values(by='CV_Scores')['Train_acc_scores'], label='Train accuracy score')

plt.axvline(x=7, ls='--', c='k', label='best-model')
plt.legend()
plt.show()


# Here I can make my point clear. If we would have used GridSearchCV it would have returned the model with best CV_score as in the table. But we can see that the best model will be the one with least difference in the test and traing scores and both should be the maximum possible which in the above case is achieved at the index no 7 of the sorted table

# ### Voting classifier

# Now voting classifiers help in finding the predicted value by taking the most votes for a given value from all the classifiers specified which might help in improving the accuracy score.

# In[ ]:


classifiers = [('Logistic Regression',LogisticRegression(C=0.1)),               ('SVC', LinearSVC(C=0.01)),               ('Random Forest', RandomForestClassifier(n_estimators=400, n_jobs=-1, max_features='log2', max_depth=7, min_samples_leaf=0.01, random_state=100))]


# In[ ]:


vc = VotingClassifier(estimators=classifiers, n_jobs=-1)


# In[ ]:


vc.fit(train_X, train_y)


# In[ ]:


vc_scores = pd.DataFrame({'Best_score':cross_val_score(vc, train_X, train_y, cv=5).mean(),                          'Train Score':accuracy_score(train_y, vc.predict(train_X)),                          'Test Score': accuracy_score(test_y, vc.predict(test_X))}, index=[0])


# In[ ]:


vc_scores


# # Model Performance

# ***But here in this case we can see that the voting classifier is performing poorer as compared to Random Forest alone. This might be due to the fact that we are using two linear classifiers and one non linear and both the linear classifiers might be performing bad in some cases (some of the values might not be classified properly by the linear classifier) and thereby giving the majority votes the incorrect classifications. So for final modeling we will be using the Random Forest classifier instead of Voting classifier to make predictions.***

# In[ ]:


sns.heatmap(confusion_matrix(test_y, vc.predict(test_X)), annot=True, fmt='.0f')


# In[ ]:


print('ROC_AUC score for VC is {}'.format(roc_auc_score(test_y, vc.predict(test_X))))
print('Accuracy score for VC is {}'.format(accuracy_score(test_y, vc.predict(test_X))))


# In[ ]:


rf1 = RandomForestClassifier(n_estimators=400, n_jobs=-1, max_features='log2', max_depth=7, min_samples_leaf=0.01, random_state=100)
rf1.fit(train_X, train_y)


# In[ ]:


sns.heatmap(confusion_matrix(test_y, rf1.predict(test_X)), annot=True, fmt='.0f')


# In[ ]:


print('ROC_AUC score for RF is {}'.format(roc_auc_score(test_y, rf1.predict(test_X))))
print('Accuracy score for RF is {}'.format(accuracy_score(test_y, rf1.predict(test_X))))


# # Feature Importances

# Now we can see that the Random Forest classifier's feature importances. And if we do basic EDA **(or is someone has even watched the movie titanic :P)** its not difficult to see that sex is one of the major factor to decide if the person would survive or not. But its astonishing that the model is giving more preference to title over other factors like adjusted fare or others. But lets keep faith in our model and continue.
# 
# **P.S. I built a model using Fare instead of Adjusted_Fare and the model using Adjusted_Fare was performing better. So maybe our initial hypothesis was correct.**

# In[ ]:


feature_importances = pd.Series(index=train_X.columns, data=rf1.feature_importances_)
sorted_feature_importances = feature_importances.sort_values()


# In[ ]:


sns.set()
sorted_feature_importances.plot(kind='barh')


# In[ ]:


results = rf1.predict(new_test_df[['Pclass', 'Sex', 'Age', 'Embarked', 'Members', 'Adjusted_Fare', 'Title']])


# In[ ]:


submission = pd.DataFrame({'PassengerId':new_test_df.PassengerId, 'Survived':results})


# In[ ]:


submission.to_csv('submission.csv', index=False)


# **So its the end of this notebook. Any suggestions are welcome. Thanks a lot:).**

# 
