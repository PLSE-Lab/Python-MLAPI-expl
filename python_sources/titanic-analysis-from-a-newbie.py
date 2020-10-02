#!/usr/bin/env python
# coding: utf-8

# Hi there, I'm a newbie learning and experimenting with the Titanic dataset, I'm creating this Kernel for easy running and for future reference and improvement. If you happen to note anything wrong or you are curious about any decision I took  in this kernel feel free to comment.

# In[ ]:


# Needed imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


# Load the models
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Get info about our test set
train.info()


# By analyzing the dataset we can see that there are some missing elements, specially the Cabin data, so what we'll do first is to drop that feature from both the test and the train datasets. This will reduce the noise while we analyze more in depth the train dataset so we can have a better understanding at the data and relate it to the problem.

# In[ ]:


train = train.drop('Cabin',axis=1)
test = test.drop('Cabin',axis=1)


# Now I think it would be the best to analyze the object type features, in order to simplify them.

# In[ ]:


train["Ticket"].value_counts()


# We can clearly see that the Ticket data is way too sparse, considering the dataset is small we should try to keep only the most meaningful features so we'll be dropping this one as well.

# In[ ]:


train = train.drop('Ticket',axis=1)
test = test.drop('Ticket',axis=1)
# To reduce space we'll check Name here as well
train["Name"].value_counts()


# As we can see, this one is as sparse as the Ticket feature, however, there seems to be a pattern with contains a 'title' to each passenger. We could actually use that information to relate the title to the surviving feature, but I guess that it could just lead to adding noise to our model, which could be bad since the training set is small, so we'll be dropping it.

# In[ ]:


train = train.drop('Name',axis=1)
test = test.drop('Name',axis=1)
# To reduce space we'll check Sex here as well
train["Sex"].value_counts()


# The Sex feature could be helpful for our model prediction since the possible values are just male or female and there could be a stretch relation between Sex and the passenger chances to survive. We'll keep this one and we'll transform it into a numeric type, so it's easier for the model to interpret it.

# In[ ]:


# Simple Label encoder would work
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
train["Sex"] = encoder.fit_transform(train["Sex"])
test["Sex"] = encoder.fit_transform(test["Sex"])
# We confirm that the Sex feature got encoded
train.head()


# Now we can check the embarked feature.

# In[ ]:


train["Embarked"].value_counts()


# It seems Embarked is another categorized feature, it could be meaningful so we'll just label encode the Embarked feature and we'll visualize how it relates to the passenger survival.

# In[ ]:


# We have to fill the 2 null values or else it'll throw an exception
# We are going to fill them with the most common values
encoder2 = LabelEncoder()
train["Embarked"].fillna(train["Embarked"].mode()[0], inplace=True)
test["Embarked"].fillna(test["Embarked"].mode()[0], inplace=True)
train["Embarked"] = encoder2.fit_transform(train["Embarked"])
test["Embarked"] = encoder2.fit_transform(test["Embarked"])
# We confirm that the Embarked feature got encoded
train.head()


# Now let's take a look again at the bigger picture to see how it looks now.

# In[ ]:


train.info()


# We can see that we have successfully encoded our object types to numbers so now is the time we'll take a look at how our features relate with the passenger survival, for this we'll use the correlation matrix

# In[ ]:


# We remove the passenger id since that won't help our model
train = train.drop('PassengerId',axis=1)
corr_matrix = train.corr()
corr_matrix["Survived"].sort_values(ascending=False)


# With this data we can see the most important features we have: Sex, Pclass and Fare would be our top 3 features with Embarked as a close fourth. We also can see that Parch, SibSp and Age are almost irrelevant as they are right now, so we could do some feature engineering with these three, but first let's take a closer look at them. 

# In[ ]:


from pandas.plotting import scatter_matrix

attributes = ["Survived", "Age", "SibSp", "Parch"]
scatter_matrix(train[attributes], figsize=(16, 16))


# With this we can get a somewhat better visualization at the Age vs Survive since it looks a little bit "fragmented" and we could actually use that to our advantage and transform the data into maybe age ranges, which actually would make sense since it's possible that children where saved the most,  so we'll work with that. The other two features still doesn't seem to be really having a pattern as they are right now, so we have to find a way to make them meaningful or else drop them since they are just adding unnecessary noise to our model.

# In[ ]:


# We'll get more insights about age first to define appropiate ranges

train["Age"].value_counts()


# With this we can easily see how sparse the data is and also that there were mostly people between 18 and 36 so we can begin our transformation by dividing the dataset in the ranges that separate those from the rest. Please note that there are a lot of ways to define these ranges, we can adjust those as we see fit. So now our groups this time will be:  
# ( age <= 17), ( age >17 && <=36) and (age > 36)

# In[ ]:


train.loc[train['Age'] <= 17, 'Age'] = 0
train.loc[(train['Age'] > 17) & (train['Age'] <= 36), 'Age'] = 1
train.loc[train['Age'] > 36, 'Age'] = 2

test.loc[test['Age'] <= 17, 'Age'] = 0
test.loc[(test['Age'] > 17) & (test['Age'] <= 36), 'Age'] = 1
test.loc[test['Age'] > 36, 'Age'] = 2


# As for the other two, I think we could actually merge them into a HasRelative feature which definitely could be easier to process and may be meaningful information in that way, so now we are going to add that to our data

# In[ ]:


# Please note that the eval function is not optimized for tiny data sets
# I used it because I think is easier to understand what exactly we are
# doing to the data
train["HasRelative"] = np.where(train.eval("SibSp > 0 or Parch > 0"), 1, 0)
test["HasRelative"] = np.where(test.eval("SibSp > 0 or Parch > 0"), 1, 0)


# Now let's look again at our correlation matrix to see if we actually did an improvement at how they relate to Survived:

# In[ ]:


corr_matrix = train.corr()
corr_matrix["Survived"].sort_values(ascending=False)


# We can see that with the change we got a huge improvement with the HasRealtive engineered feature, but, the Age one didn't really improved that much, however it's still better than having all the sparse data, let's take a look at it again to get a better picture of the ages of the passengers.

# In[ ]:


train["Age"].value_counts()


# We can clearly see now that the most passengers where young, hopefully our model will use this info to gain even more insights about the passengers. Now let's drop the Parch and SibSp features and look again how our data looks

# In[ ]:


train = train.drop('Parch',axis=1)
train = train.drop('SibSp',axis=1)
train.info()


# Now our model looks more simple and the features it has are meaningful to solve the problem we have at hand. BUT we still have some missing data in the Age feature, here we have two main options: drop all the rows that don't have an Age value specified or fill the missing data like we did with the Embarked feature. Since Age is the less correlated to Survived and it has been transformed to only hold a range of Ages we can fill it with the most common class and it probably won't add noise to our model. If the null values where in a more important feature like Sex, we should either drop the rows with null values or use other feature to try to deduce it like reading the name of the passenger. So now let's go ahead and fill the missing Age with 1 which is the most common class.

# In[ ]:


train['Age'].fillna(1, inplace=True)
train.info()


# Now we have our dataset complete, so we can start training a model for it. Since it's a Binary classification issue( our model has to tell whether the passenger survived or not) we can use a model that performs well with that type of task, so I'll give it a try to a RandomForest.

# In[ ]:


X_train = train.drop('Survived',axis=1)
Y_train = train['Survived']

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train, Y_train)


# Now that we have trained it let's look at our test set and look at it's info searching for null values so we can make predictions on it.

# In[ ]:


test.info()


# It looks like we have a null value in Fare and a lot of Age ones, also we haven't removed the SibSp and Parch features from this one. So let's take care of that by removing the unused features and fill the Fare with the median and the Age with the mode.

# In[ ]:


test = test.drop('Parch',axis=1)
test = test.drop('SibSp',axis=1)
test['Fare'].fillna(test['Fare'].dropna().median(), inplace=True)
test['Age'].fillna(test['Age'].mode()[0], inplace=True)
test.info()


# Now we make our predictions and save them for download and submission.

# In[ ]:


predictions = random_forest.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('./predictions.csv', index=False)


# With this model I got a prediction score of 0.75120 which right now is my personal best, I think I might have fine tune the random forest so it performs better than this. Any comments or corrections in this one will be very appreciated.

# ##Second Part - Improving our score  ##

# Now that I have learned some techniques to improve our model performance I'll apply them to our current model.
# The first thing that I'll do is to fine tune our random forest classifier to find out the best hyper parameters for it.

# In[ ]:


from sklearn.model_selection import GridSearchCV
param_grid = {"n_estimators": [ 10, 20, 30, 50, 100, 200],
              "max_features": [3, 4, 5, 6],
              "max_depth": [ 6, 9, 12, 15],
              "bootstrap": [True, False]}    

optimal_forest = RandomForestClassifier()
# run grid search
grid_search = GridSearchCV(optimal_forest, param_grid=param_grid,scoring="average_precision")
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
print(grid_search)


# Now that could help us to see how can we change the parameters for our RandomForest to make it work better. I'll fine tune it's parameters again.

# In[ ]:


param_grid = {"n_estimators": [ 25, 28, 30, 33, 36, 40],
              "max_features": [2, 3, 4, 5],
              "max_depth": [ 4, 5, 6, 7],
              "bootstrap": [False]}
optimal_forest = RandomForestClassifier()
# run grid search
grid_search_rfo = GridSearchCV(optimal_forest, param_grid=param_grid,scoring="average_precision")
grid_search_rfo.fit(X_train, Y_train)
print(grid_search_rfo.best_params_)
print(grid_search_rfo)


# I think we can assume these are the best parameters for our Random Forest, so now let's use that classifier and now let's   use that to generate another set of predictions with that one. 

# In[ ]:


predictions_rfo = grid_search_rfo.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_rfo
    })
submission.to_csv('./predictions_rf_optimized.csv', index=False)


# However we aren't done with just those predictions, we'll train and fine tune other classifiers. 
# 

# Let's now use another classifier, this time we'll use a Support Vector Machine more specifically a Linear SVM classifier:

# In[ ]:


from sklearn.svm import LinearSVC
param_grid = {"C": [ 1, 10, 30, 50, 100],
              "loss": ["hinge", "squared_hinge"]}
linear_svc = LinearSVC()
grid_search_lsvc = GridSearchCV(linear_svc, param_grid=param_grid,scoring="average_precision")
grid_search_lsvc.fit(X_train, Y_train)
print(grid_search_lsvc.best_params_)
print(grid_search_lsvc)


# In[ ]:


predictions_lsvc = grid_search_lsvc.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_lsvc
    })
submission.to_csv('./predictions_lsvc_optimized.csv', index=False)


# Now let's give it a try to another SVC, this time we'll go with a Radial Basis Function Kernel:

# In[ ]:


from sklearn.svm import SVC
param_grid = {"kernel":["rbf"],
              "gamma": [ 0.1, 1, 3, 5],
              "C": [ 0.001 , 0.1, 1, 10, 100]}
rbf_svc = SVC()
grid_search_rbf_svc = GridSearchCV(rbf_svc, param_grid=param_grid,scoring="average_precision")
grid_search_rbf_svc.fit(X_train, Y_train)
print(grid_search_rbf_svc.best_params_)
print(grid_search_rbf_svc)


# In[ ]:


#Another grid search to find even better parameters
param_grid = {"kernel":["rbf"],
              "gamma": [ 0.01, 0.05, 0.1, 0.15, 0.5],
              "C": [ 8 , 10, 15, 20, 40]}
rbf_svc = SVC()
grid_search_rbf_svc = GridSearchCV(rbf_svc, param_grid=param_grid,scoring="average_precision")
grid_search_rbf_svc.fit(X_train, Y_train)
print(grid_search_rbf_svc.best_params_)
print(grid_search_rbf_svc)


# In[ ]:


#Yet another grid search to find even better parameters
param_grid = {"kernel":["rbf"],
              "gamma": [ 0.001, 0.005, 0.01, 0.05],
              "C": [ 20, 40, 50, 60, 70]}
rbf_svc = SVC()
grid_search_rbf_svc = GridSearchCV(rbf_svc, param_grid=param_grid,scoring="average_precision")
grid_search_rbf_svc.fit(X_train, Y_train)
print(grid_search_rbf_svc.best_params_)
print(grid_search_rbf_svc)


# Now let's also store the predictions made from this one just to see later how it does by itself:

# In[ ]:


predictions_rbf_svc = grid_search_rbf_svc.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_rbf_svc
    })
submission.to_csv('./predictions_rbf_svc.csv', index=False)


# Now let's give it a try to an Stochastic Gradient Descent Classifier:

# In[ ]:


from sklearn.linear_model import SGDClassifier

param_grid = {"loss":["hinge","modified_huber"],
              "penalty":["l2","l1","elasticnet"]}
sgdc = SGDClassifier()
grid_search_sgdc = GridSearchCV(sgdc, param_grid=param_grid,scoring="average_precision")
grid_search_sgdc.fit(X_train, Y_train)
print(grid_search_sgdc.best_params_)
print(grid_search_sgdc)


# Again let's store the predictions

# In[ ]:


predictions_sgdc = grid_search_sgdc.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_sgdc
    })
submission.to_csv('./predictions_sgdc.csv', index=False)


# And finally a Nearest K-Neighboors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
param_grid = {"weights":["uniform","distance"],
              "n_neighbors":[5,7,9,12,15,18]}
nbrs = KNeighborsClassifier()
grid_search_nbrs = GridSearchCV(nbrs, param_grid=param_grid,scoring="average_precision")
grid_search_nbrs.fit(X_train, Y_train)
print(grid_search_nbrs.best_params_)
print(grid_search_nbrs)


# Now that we got our 5 classifiers with the best parameters for getting the best average precision it's time to assemble all of them in a voting classifier:

# In[ ]:


from sklearn.ensemble import VotingClassifier

nbrs = KNeighborsClassifier(18, weights='distance')
sgdc = SGDClassifier(loss='hinge', penalty= 'l1')
rbf_svc = SVC(kernel="rbf",C=10, gamma= 0.1)
linear_svc = LinearSVC(loss="hinge",C=1)
optimal_forest = RandomForestClassifier(bootstrap= False, max_depth=6, max_features= 4, n_estimators= 28)
voting_clf = VotingClassifier(
            estimators=[('rf', optimal_forest), ('lsvc', linear_svc), ('rbfsvc', rbf_svc), ('sgdc', sgdc), ('nbrs',nbrs)],
            voting='hard'
        )
voting_clf.fit(X_train, Y_train)

predictions = voting_clf.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('./predictions_voting.csv', index=False)


# Now I've checked our classifiers score against the test set and their results are these ones:
# 
# ---
# + `RandomForest(bootstrap= False, max_depth=6, max_features= 4, n_estimators= 28)` = **0.78947**
# + `VotingClassifier(voting="hard")` = **0.77990**
# + `SVC(kernel="rbf",C=10, gamma= 0.1)` = **0.76077**
# + `LinearSVC(loss="hinge",C=1)` = **0.74163**
# + `SGDClassifier(loss='hinge', penalty= 'l1')` = **0.71292**
# 
# ---
# As you can see our Voting Classifier is performing worse than our RandomForest, that probably has something to do with what type of classifiers we are using, since we have 2 Support Vector Machines Classifiers we could assume that we are having voting 'issues' for that matter so I'll give it a try again to our voting classifier but this time I'll only use the RandomForest, SVC with 'rbf' kernel and the SGD Classifier and see how much that one improves or gets worse.

# In[ ]:


sgdc = SGDClassifier(loss='hinge', penalty= 'l1')
rbf_svc = SVC(kernel="rbf",C=10, gamma= 0.1)
optimal_forest = RandomForestClassifier(bootstrap= False, max_depth=6, max_features= 4, n_estimators= 28)
voting_clf = VotingClassifier(
            estimators=[('rf', optimal_forest), ('rbfsvc', rbf_svc), ('sgdc', sgdc)],
            voting='hard'
        )
voting_clf.fit(X_train, Y_train)
#Save predictions
predictions = voting_clf.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('./predictions_voting_2.csv', index=False)


# Now I've got to check how this voting classifier performs, hopefully it'll do better this time.
# Unfortunately it performed a little bit worse. It got a score of **0.76077**.

# am I stopping here?
# ---
# Short answer: no, there's still a few things I would like to try and since all of this is just me messing around with the dataset I'll try some interesting things here. I'll try to find the best parameters for getting the best accuracy, for our RandomForest, our SVMC with RBF kernel and the nearest K Neighbours, which for some reason forgot to measure before, also I'll be using the _probability=True_ parameter to help me go also for a soft voting classifier and then measure all of them to later on use them in another voting classifier. 
# 
# So first things first, let's go and add that _probability=True_ parameter to our SVM Classifier and the _loss=modified_huber_ to the SDGClassifier. Now let see how the voting classifier performs by using soft voting.

# In[ ]:


sgdc = SGDClassifier(loss='modified_huber', penalty= 'l1')
rbf_svc = SVC(kernel="rbf",C=10, gamma= 0.1, probability=True)
optimal_forest = RandomForestClassifier(bootstrap= False, max_depth=6, max_features= 4, n_estimators= 28)
voting_clf = VotingClassifier(
            estimators=[('rf', optimal_forest), ('rbfsvc', rbf_svc), ('sgdc', sgdc)],
            voting='soft'
        )
voting_clf.fit(X_train, Y_train)
#Save predictions
predictions = voting_clf.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('./predictions_voting_2_soft.csv', index=False)


# For my next experiment I would like to see how a voting classifier performs when entering classifiers with the best hyper parameters to score best accuracy, F1(F1 = 2 * (precision * recall) / (precision + recall)) and average_precision. So now I should redo my grid searches now looking for accuracy.

# In[ ]:


# LinearSVC
param_grid = {"C": [ 30, 50, 100],
              "loss": ["hinge", "squared_hinge"]}
linear_svc = LinearSVC()
grid_search_lsvc = GridSearchCV(linear_svc, param_grid=param_grid,scoring="accuracy")
grid_search_lsvc.fit(X_train, Y_train)
print(grid_search_lsvc.best_params_)
predictions_lsvc = grid_search_lsvc.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_lsvc
    })
linear_svc = LinearSVC()
submission.to_csv('./predictions_lsvc_accuracy.csv', index=False)
grid_search_lsvc = GridSearchCV(linear_svc, param_grid=param_grid,scoring="f1")
grid_search_lsvc.fit(X_train, Y_train)
print(grid_search_lsvc.best_params_)
predictions_lsvc = grid_search_lsvc.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_lsvc
    })
submission.to_csv('./predictions_lsvc_f1.csv', index=False)


# In[ ]:


#Random Forest
param_grid = {"n_estimators": [ 10, 20, 35, 50, 100],
              "max_features": [3, 4, 5, 6],
              "max_depth": [ 6, 9, 12, 15],
              "bootstrap": [True, False]}    

optimal_forest = RandomForestClassifier()
grid_search = GridSearchCV(optimal_forest, param_grid=param_grid,scoring="accuracy")
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
predictions_rfo = grid_search_rfo.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_rfo
    })
submission.to_csv('./predictions_rf_accuracy.csv', index=False)

optimal_forest = RandomForestClassifier()
grid_search = GridSearchCV(optimal_forest, param_grid=param_grid,scoring="f1")
grid_search.fit(X_train, Y_train)
print(grid_search.best_params_)
predictions_rfo = grid_search_rfo.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_rfo
    })
submission.to_csv('./predictions_rf_f1.csv', index=False)


# In[ ]:


#K-Nearest
param_grid = {"weights":["uniform","distance"],
              "n_neighbors":[4,5,6,7,8,10]}
nbrs = KNeighborsClassifier()
grid_search_nbrs = GridSearchCV(nbrs, param_grid=param_grid,scoring="accuracy")
grid_search_nbrs.fit(X_train, Y_train)
print(grid_search_nbrs.best_params_)
predictions_nbrs = grid_search_nbrs.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_nbrs
    })
submission.to_csv('./predictions_knbrs_accuracy.csv', index=False)
nbrs = KNeighborsClassifier()
grid_search_nbrs = GridSearchCV(nbrs, param_grid=param_grid,scoring="f1")
grid_search_nbrs.fit(X_train, Y_train)
print(grid_search_nbrs.best_params_)
grid_search_nbrs = grid_search_nbrs.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": grid_search_nbrs
    })
submission.to_csv('./predictions_knbrs_f1.csv', index=False)


# In[ ]:


param_grid = {"loss":["hinge","modified_huber"],
              "penalty":["l2","l1","elasticnet"]}
sgdc = SGDClassifier()
grid_search_sgdc = GridSearchCV(sgdc, param_grid=param_grid,scoring="accuracy")
grid_search_sgdc.fit(X_train, Y_train)
print(grid_search_sgdc.best_params_)
predictions_sgdc = grid_search_sgdc.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_sgdc
    })
submission.to_csv('./predictions_sgdc_accuracy.csv', index=False)
sgdc = SGDClassifier()
grid_search_sgdc = GridSearchCV(sgdc, param_grid=param_grid,scoring="f1")
grid_search_sgdc.fit(X_train, Y_train)
print(grid_search_sgdc.best_params_)
predictions_sgdc = grid_search_sgdc.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_sgdc
    })
submission.to_csv('./predictions_sgdc_f1.csv', index=False)


# In[ ]:


param_grid = {"kernel":["rbf"],
              "gamma": [ 0.001, 0.005,0.01, 0.10],
              "C": [  55,60, 62, 65, 67]}
rbf_svc = SVC()
grid_search_rbf_svc = GridSearchCV(rbf_svc, param_grid=param_grid,scoring="accuracy")
grid_search_rbf_svc.fit(X_train, Y_train)
print(grid_search_rbf_svc.best_params_)
predictions_rbf_svc = grid_search_rbf_svc.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_rbf_svc
    })
submission.to_csv('./predictions_rbf_svc_accuracy.csv', index=False)
rbf_svc = SVC()
grid_search_rbf_svc = GridSearchCV(rbf_svc, param_grid=param_grid,scoring="f1")
grid_search_rbf_svc.fit(X_train, Y_train)
print(grid_search_rbf_svc.best_params_)
predictions_rbf_svc = grid_search_rbf_svc.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions_rbf_svc
    })
submission.to_csv('./predictions_rbf_svc_f1.csv', index=False)


# Wow that was a lot of code, I should have created a helper function to make it a lot less verbose, however, right now I rather get accustomed first to this workflow to GridSearch and I want to visualize it even at the cost of more lines of code, if this were actually code for a project this would be a very bad practice, mostly because if I would like to change something to this code I'll have to do that change so many times this code could become a hell to maintain.
# 
# That aside is time to finally make our last voting classifier for this experiment. It'll contain all the entire bunch of classifiers that I've made up to this point and then it'll make predictions according to the hard voting of all those classifiers. However there's also a significant issue that I might have not considered, by having a lot of classifiers in our voting classifier we are prone to get more errors if these do not perform well by themselves, so I might train more than just one voting classifier hoping to achieve good results. Here goes the code:

# In[ ]:


nbrs_1 = KNeighborsClassifier(18, weights='distance')
nbrs_2 = KNeighborsClassifier(8, weights='distance')
nbrs_3 = KNeighborsClassifier(4, weights='distance')
sgdc_1 = SGDClassifier(loss='hinge', penalty= 'l1')
sgdc_2 = SGDClassifier(loss='modified_huber', penalty= 'l1')
sgdc_3 = SGDClassifier(loss='modified_huber', penalty= 'elasticnet')
rbf_svc_1 = SVC(kernel="rbf",C=10, gamma= 0.1)
rbf_svc_2 = SVC(kernel="rbf",C=65, gamma= 0.01)
rbf_svc_3 = SVC(kernel="rbf",C=67, gamma= 0.01)
linear_svc_1 = LinearSVC(loss="hinge",C=1)
linear_svc_2 = LinearSVC(loss="squared_hinge",C=30)
linear_svc_3 = LinearSVC(loss="hinge",C=50)
optimal_forest_1 = RandomForestClassifier(bootstrap= False, max_depth=6, max_features= 4, n_estimators= 28)
optimal_forest_2 = RandomForestClassifier(bootstrap= True, max_depth=9, max_features= 5, n_estimators= 35)
optimal_forest_3 = RandomForestClassifier(bootstrap= True, max_depth=6, max_features= 5, n_estimators= 20)
voting_clf = VotingClassifier(
            estimators=[('rf_1', optimal_forest_1), 
                        ('rf_2', optimal_forest_2), 
                        ('rf_3', optimal_forest_3), 
                        ('lsvc_1', linear_svc_1),
                        ('lsvc_2', linear_svc_2),
                        ('lsvc_3', linear_svc_3),
                        ('rbfsvc_1', rbf_svc_1),
                        ('rbfsvc_2', rbf_svc_2),
                        ('rbfsvc_3', rbf_svc_3),
                        ('sgdc_1', sgdc_1),
                        ('sgdc_2', sgdc_2),
                        ('sgdc_3', sgdc_3),
                        ('nbrs_1',nbrs_1),
                        ('nbrs_2',nbrs_2),
                        ('nbrs_3',nbrs_3)],
            voting='hard'
        )
voting_clf.fit(X_train, Y_train)

predictions = voting_clf.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('./predictions_voting_many.csv', index=False)


# In[ ]:


#removed some SDG Classifiers
nbrs_1 = KNeighborsClassifier(18, weights='distance')
nbrs_2 = KNeighborsClassifier(8, weights='distance')
nbrs_3 = KNeighborsClassifier(4, weights='distance')
sgdc_1 = SGDClassifier(loss='hinge', penalty= 'l1')
rbf_svc_1 = SVC(kernel="rbf",C=10, gamma= 0.1)
rbf_svc_2 = SVC(kernel="rbf",C=65, gamma= 0.01)
rbf_svc_3 = SVC(kernel="rbf",C=67, gamma= 0.01)
linear_svc_1 = LinearSVC(loss="hinge",C=1)
linear_svc_2 = LinearSVC(loss="squared_hinge",C=30)
linear_svc_3 = LinearSVC(loss="hinge",C=50)
optimal_forest_1 = RandomForestClassifier(bootstrap= False, max_depth=6, max_features= 4, n_estimators= 28)
optimal_forest_2 = RandomForestClassifier(bootstrap= True, max_depth=9, max_features= 5, n_estimators= 35)
optimal_forest_3 = RandomForestClassifier(bootstrap= True, max_depth=6, max_features= 5, n_estimators= 20)
voting_clf = VotingClassifier(
            estimators=[('rf_1', optimal_forest_1), 
                        ('rf_2', optimal_forest_2), 
                        ('rf_3', optimal_forest_3), 
                        ('lsvc_1', linear_svc_1),
                        ('lsvc_2', linear_svc_2),
                        ('lsvc_3', linear_svc_3),
                        ('rbfsvc_1', rbf_svc_1),
                        ('rbfsvc_2', rbf_svc_2),
                        ('rbfsvc_3', rbf_svc_3),
                        ('sgdc_1', sgdc_1),
                        ('nbrs_1',nbrs_1),
                        ('nbrs_2',nbrs_2),
                        ('nbrs_3',nbrs_3)],
            voting='hard'
        )
voting_clf.fit(X_train, Y_train)

predictions = voting_clf.predict(test.drop('PassengerId',axis=1))
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('./predictions_voting_many_one_sdg.csv', index=False)

