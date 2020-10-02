#!/usr/bin/env python
# coding: utf-8

# # Machine Learning 2020 Course Projects
# 
# ## Project Schedule
# 
# In this project, you will solve a real-life problem with a dataset. The project will be separated into two phases:
# 
# 27th May - 10th June: We will give you a training set with target values and a testing set without target. You predict the target of the testing set by trying different machine learning models and submit your best result to us and we will evaluate your results first time at the end of phase 1.
# 
# 9th June - 24th June: Students stand high in the leader board will briefly explain  their submission in a proseminar. We will also release some general advice to improve the result. You try to improve your prediction and submit final results in the end. We will again ask random group to present and show their implementation.
# The project shall be finished by a team of two people. Please find your teammate and REGISTER via [here](https://docs.google.com/forms/d/e/1FAIpQLSf4uAQwBkTbN12E0akQdxfXLgUQLObAVDRjqJHcNAUFwvRTsg/alreadyresponded).
# 
# The submission and evaluation is processed by [Kaggle](https://www.kaggle.com/t/b3dc81e90d32436d93d2b509c98d0d71).  In order to submit, you need to create an account, please use your team name in the `team tag` on the [kaggle page](https://www.kaggle.com/t/b3dc81e90d32436d93d2b509c98d0d71). Two people can submit as a team in Kaggle.
# 
# You can submit and test your result on the test set 2 times a day, you will be able to upload your predicted value in a CSV file and your result will be shown on a leaderboard. We collect data for grading at 22:00 on the **last day of each phase**. Please secure your best results before this time.
# 
# 

# ## Project Description
# 
# Car insurance companies are always trying to come up with a fair insurance plan for customers. They would like to offer a lower price to the careful and safe driver while the careless drivers who file claims in the past will pay more. In addition, more safe drivers mean that the company will spend less in operation. However, for new customers, it is difficult for the company to know who the safe driver is. As a result, if a company offers a low price, it bears a high risk of cost. If not, the company loses competitiveness and encourage new customers to choose its competitors.
# 
# 
# Your task is to create a machine learning model to mitigate this problem by identifying the safe drivers in new customers based on their profiles. The company then offers them a low price to boost safe customer acquirement and reduce risks of costs. We provide you with a dataset (train_set.csv) regarding the profile (columns starting with ps_*) of customers. You will be asked to predict whether a customer will file a claim (`target`) in the next year with the test_set.csv 
# 
# ~~You can find the dataset in the `project` folders in the jupyter hub.~~ We also upload dataset to Kaggle and will test your result and offer you a leaderboard in Kaggle. Please find them under the Data tag on the following page:
# https://www.kaggle.com/t/b3dc81e90d32436d93d2b509c98d0d71

# ## Phase 1: 26th May - 9th June
# 
# ### Data Description
# 
# In order to take a look at the data, you can use the `describe()` method. As you can see in the result, each row has a unique `id`. `Target` $\in \{0, 1\}$ is whether a user will file a claim in his insurance period. The rest of the 57 columns are features regarding customers' profiles. You might also notice that some of the features have minimum values of `-1`. This indicates that the actual value is missing or inaccessible.
# 

# In[ ]:


# Quick load dataset and check
import pandas as pd
import os
running_local = True if os.getenv('JUPYTERHUB_USER') is None else False
os.listdir('data')
if ~running_local:
    path = "data/final-project-dataset/"
else:
    path = "./"


# In[ ]:


filename = "train_set.csv"
data_train = pd.read_csv("data/final-project-dataset/train_set.csv")
filename = "test_set.csv"
data_test = pd.read_csv("data/final-project-dataset/test_set.csv")


# In[ ]:


data_train.describe()


# The prefix, e.g. `ind` and `calc`, indicate the feature belongs to similiar groupings. The postfix `bin` indicates binary features and `cat` indicates categorical features. The features without postfix are ordinal or continuous. Similarly, you can check the statistics for testing data:

# In[ ]:


data_test.describe()


# 
# ### Example
# 
# We will use the decision tree classifier as an example.

# In[ ]:



from sklearn.tree import DecisionTreeClassifier

## Select target and features
fea_col = data_train.columns[2:]

data_Y = data_train['target']
data_X = data_train[fea_col]

clf = DecisionTreeClassifier()
clf = clf.fit(data_X,data_Y)
y_pred = clf.predict(data_X)


# In[ ]:


sum(y_pred==data_Y)/len(data_Y)


# 
# 
# **The decision tree has 100 percent accurate rate!**
# 
# It is unbelievable! What went wrong?
# 
# Hint: What is validation?
# 
# After fixing the problem, you may start here and try to improve the accurate rates.

# In[ ]:


from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)
clf = DecisionTreeClassifier(min_impurity_decrease = 0.001)
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_val)


# In[ ]:


sum(y_pred==y_val)/len(y_val)


# ### Information Beyond above Accuracy 
# 
# The result looks promising. **Let us take a look into the results further.**
# 
# We now make a prediction for the valid set with label-1 data only:

# In[ ]:


def extrac_one_label(x_val, y_val, label):
    X_pos = x_val[y_val == label]
    y_pos = y_val[y_val == label]
    return X_pos, y_pos

X_pos, y_pos = extrac_one_label(x_val, y_val, 1)
y_pospred = clf.predict(X_pos)
sum(y_pospred==y_pos)/len(y_pos)
#print(x_val.shape)
#print(X_pos.shape)
#print(y_val.shape)
#print(y_pos)
#print(y_pospred)


# **None of the label is detected!** Now with label-0 data only:

# In[ ]:


X_neg, y_neg = extrac_one_label(x_val, y_val, 0)
y_negpred = clf.predict(X_neg)
sum(y_negpred==y_neg)/len(y_neg)
#print(sum(y_negpred==y_neg))


# ### Your Turns:
# 
# What does it mean? Why does it look like that? How do you overcome it and get the better results? Hint :

# In[ ]:


print(sum(data_Y==0)/len(data_Y), sum(data_Y==1))


# ### My Answer:
# The above block shows how many people are classified with 0. Of our data we have 16271 people with label 1, and 430000 with 0 -> with simple math just calculating the probability we get that 96% are with label 1. We can keep this value in mind because from a statistical point theoretically we should get a similar percentage if we apply it to new data too. 
# But the classifier from further above also said that it was 96% of the time correct.
# This seems suspicious

# ### Validation metric
# Think about what the proper metric is to train your model and how you should  change your training procedure to aviod this problem?

# In[ ]:


## Your work
from sklearn.model_selection import train_test_split
# our data to train
data_train.head()

#X_train = x_train
#y_train = y_train
#X_test = x_val
#y_test = y_val

X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)

# selecting only a subset of the actual data since computing could take much longer with the entire set 
print('X_train shape: ', X_train.shape)
X_train_2 = X_train[0:2000]
print('X_train_2 shape: ', X_train_2.shape)

print('y_train shape: ', y_train.shape)
y_train_2 = y_train[0:2000]
print('y_train_2 shape: ', y_train_2.shape)

print('X_test shape: ', X_test.shape)
X_test_2 = X_test[0:600]
print('X_test_2 shape: ', X_test_2.shape)

print('y_test shape: ', y_test.shape)
y_test_2 = y_test[0:600]
print('y_test_2 shape: ', y_test_2.shape)



# In[ ]:


print(X_test_2.shape)


# In[ ]:


import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split


# ### svm
# lets try it again with svm instead of decision tree from above

# In[ ]:


# creating a svm classifier
clf = svm.SVC(kernel='linear', C=1)


# In[ ]:


# training said classifies
clf.fit(X_train_2, y_train_2)


# In[ ]:


#clf.score(X_test_2, y_test_2)

# predict response
y_pred_SVM = clf.predict(X_test_2)


# In[ ]:


from sklearn import metrics
# checking how accurately the prediction was
accuracy = metrics.accuracy_score(y_test_2, y_pred_SVM)
print('Accuracy of :',accuracy)


# ### again 96% accuracy??
# once again we get 96% accuracy. Lets take a look at what our predictions are:

# In[ ]:


print('our predicted values: ' ,y_pred_SVM)


# Intesting! All of our predicted values are 0????

# In[ ]:


print('# of y_test values which are 1:', sum(y_test_2[0:]))


# The block above tells us that our actual test results have a sum of 16 (could change at rerunning). This means there were 16 ones in the results... But we got zero ones? Since we have a total proportion of 96% zeros to 4% ones, it makes sense that with all ones we get a accuracy of 96%...
# There must be something off with our training.

# #### lets take a look at our data
# here we can see that we actually have a few features which dont give us any information at all. For example the ones where only one spike is in the graph... We also have a lot of features with the -1 value. That -1 value means we dont have a data for this point. 

# In[ ]:


all_hists = X_train.hist(bins=20, figsize=(50,25))


# #### maybe our prediction is of because of the values that are -1 
# aka not available. They could throw the set off. in the next block we'll see that quite a few features have loads of -1. ps_car_03_cat for example has 308336 empty values.

# In[ ]:


data_X[data_X == -1].count()


# ### dropping N/A 
# ps_reg_03, ps_car_03_cat, ps_car_05_cat and ps_car_14 all have many not available values. Might make sense to drop those altogether since they wont give much info about the results anyway

# In[ ]:


data_X = data_X.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)


# In[ ]:


data_X[data_X == -1].count()


# now the dataset is not crowded with features who are somewhat 'useless'. We also see that more graphy are kind of linearably seperable. There are quite a few graphs with values at the left and also values at the right side

# In[ ]:


all_hists = X_train.hist(bins=20, figsize=(50,25))


# we see here well that there aren't many classes which have a huge spike at -1. That means most of the classes are distinguishable at 0 or 1, a few have gaussian curves in them a well. But makes most of the data clearly seperable

# ### trying svm again
# lets try svm once again, maybe this fixed the problem

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)

# selecting only a subset of the actual data since computing could take much longer with the entire set 
print('X_train shape: ', X_train.shape)
X_train_2 = X_train[0:2000]
print('X_train_2 shape: ', X_train_2.shape)

print('y_train shape: ', y_train.shape)
y_train_2 = y_train[0:2000]
print('y_train_2 shape: ', y_train_2.shape)

print('X_test shape: ', X_test.shape)
X_test_2 = X_test[0:600]
print('X_test_2 shape: ', X_test_2.shape)

print('y_test shape: ', y_test.shape)
y_test_2 = y_test[0:600]
print('y_test_2 shape: ', y_test_2.shape)


# In[ ]:


from sklearn import metrics
clf_no_neg = svm.SVC(kernel='linear', C=1)
# training said classifies
clf_no_neg.fit(X_train_2, y_train_2)
#prediction
y_pred_SVM_no_neg = clf_no_neg.predict(X_test_2)
# checking how accurately the prediction was
accuracy_no_neg = metrics.accuracy_score(y_test_2, y_pred_SVM_no_neg)
print('Accuracy of :',accuracy_no_neg)
print('# of y_test values which are 1:', sum(y_test_2[0:]))
print('# of our predicted values: ' ,sum(y_pred_SVM_no_neg))


# In[ ]:


from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

conf_mat = confusion_matrix(y_true=y_test_2, y_pred=y_pred_SVM_no_neg)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()


# #### again.. 95ish percent only because values are all 0
# the confusion matrix here shows the correct and incorrect predictions for each class.
# have to come up with something else. Lets think about why that could be for a second. Our Dataset has loads of 0 and few 1. Having such an imbalaced dataset makes it hard for the training algorithm. Maybe we can adjust the weights somehow to fit better.
# Let's try again with SVM and balanced class weights:

# In[ ]:


from sklearn import metrics
clf = svm.SVC(kernel='linear', class_weight='balanced', C=1.0)
# training said classifies
clf.fit(X_train_2, y_train_2)
#prediction
y_pred_SVM = clf.predict(X_test_2)
# checking how accurately the prediction was
accuracy = metrics.accuracy_score(y_test_2, y_pred_SVM)
print('Accuracy of :',accuracy)


# #### huh.. our accuracy is different now!
# but also way worse. Let's check our the prediction data:

# In[ ]:


print('# of y_test values which are 1:', sum(y_test_2[0:]))
print('# of our predicted values: ' , sum(y_pred_SVM))


# so we got 247 ones predicted... our of 600. This is a change... although a rather bad one. This shows that the weights definitely have something to do with it. We now gotta find out how to adjust them to get a better result with our predictions.
# lets try with a different kernel:

# In[ ]:


clf = svm.SVC(kernel='rbf', class_weight='balanced', C=1.0)
# training said classifies
clf.fit(X_train_2, y_train_2)
#prediction
y_pred_SVM = clf.predict(X_test_2)
# checking how accurately the prediction was
accuracy = metrics.accuracy_score(y_test_2, y_pred_SVM)
print('Accuracy of :',accuracy)
print('# of y_test values which are 1:', sum(y_test_2[0:]))
print('# of our predicted values: ' , sum(y_pred_SVM))


# a tiny bit better but still rather bad?

# In[ ]:


conf_mat = confusion_matrix(y_true=y_test_2, y_pred=y_pred_SVM)
print('Confusion matrix:\n', conf_mat)

labels = ['Class 0', 'Class 1']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()


# The confusion matrix here just shows that we dont have many matches with 0 0 but quite many where we predicted 1 but the class is actually zero...

# In[ ]:


get_ipython().system('pip install imblearn')


# In[ ]:


import imblearn


# In[ ]:


def plot_2d_space(X, y, label='Classes'):   
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y==l, 0],
            X[y==l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


# In[ ]:


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_2)

plot_2d_space(X_train_pca, y_train_2, 'Imbalanced dataset (2 PCA components)')


# abstraction onto 2 dimensions... doesn't really show much distinction
# might need to think about resampling the data to get better more data around the few 1 to train with and in turn better results...
# https://heartbeat.fritz.ai/resampling-to-properly-handle-imbalanced-datasets-in-machine-learning-64d82c16ceaa
# tried a few of those resampling methods

# In[ ]:


# import the SMOTETomek
from imblearn.over_sampling import SMOTE

# create the  object with the desired sampling strategy.
smote = SMOTE(sampling_strategy='minority')

# fit the object to our training data
x_train_smote, y_train_smote = smote.fit_sample(X_train_2, y_train_2)

clf = svm.SVC(kernel='linear', C=1.0)
# training said classifies
clf.fit(x_train_smote, y_train_smote)
#prediction
y_predict_smote = clf.predict(X_test_2)
# checking how accurately the prediction was
accuracy = metrics.accuracy_score(y_test_2, y_predict_smote)
print('Accuracy of :',accuracy)
print('# of y_test values which are 1:', sum(y_test_2[0:]))
print('# of our predicted values: ' , sum(y_predict_smote))


# #### interesting??
# we got a accuracy of 90%, but we also predicted around as many ones as there really are! This time we were using the SMOTE Synthetic Minority Oversampling Technique. Breakthrough for once? let's check our the prediction values:

# In[ ]:


print(y_predict_smote)


# looks not too bad! let's try it with more than just 2000 training and 600 testing values. 

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)

# selecting only a subset of the actual data since computing could take much longer with the entire set 
print('X_train shape: ', X_train.shape)
X_train_2 = X_train[0:15000]
print('X_train_2 shape: ', X_train_2.shape)

print('y_train shape: ', y_train.shape)
y_train_2 = y_train[0:15000]
print('y_train_2 shape: ', y_train_2.shape)

print('X_test shape: ', X_test.shape)
X_test_2 = X_test[0:3500]
print('X_test_2 shape: ', X_test_2.shape)

print('y_test shape: ', y_test.shape)
y_test_2 = y_test[0:3500]
print('y_test_2 shape: ', y_test_2.shape)


# In[ ]:


smote = SMOTE(sampling_strategy='minority')
x_train_smote, y_train_smote = smote.fit_sample(X_train_2, y_train_2)

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(x_train_smote, y_train_smote)
#prediction
y_predict_smote = clf.predict(X_test_2)
# checking how accurately the prediction was
accuracy = metrics.accuracy_score(y_test_2, y_predict_smote)
print('Accuracy of :',accuracy)
print('# of y_test values which are 1:', sum(y_test_2[0:]))
print('# of our predicted values: ' , sum(y_predict_smote))


# For 15000 training we still get around 90% accuracy which seems pretty good. The overall accuracy is of course still worse than if we just assign 0 to everyone, but we have to think of it in terms of what our algorithm is used for. If we just assign 0 everywhere, then the insurance company would have to take every single customer and then would get 100% insurance claims for around 4% of them. If we can filter out the ones where we're fairly sure that they would claim the insurance, then the company does not have to get them their insurance and won't have to pay them in case they have an accident. Of course there are some false negatives, but in terms of overall profit, having a few less people in the insurance programm who might be save drivers and won't have an accident, is probably better than having a few where the insurance company has to pay a lot of money for if they have an accident.

# trying the training and testing with all of the data takes probably forever, leaving it for 30'+ didn't finish. 15000 for training already takes quit long, can't imagine the runtime for 300k. Since more training data generally means better results, it might be worth training with more data. But on the other hand, more data could also lead to overfitting and then in the end decrease our result. We'll try with a few more sizes and then decide on one for which we'll predict with on the data_test

# In[ ]:


data_test.shape


# gotta get rid of the extra colums in our test datafile first though:

# In[ ]:


data_test = data_test.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)


# In[ ]:


data_test.shape


# In[ ]:


data_test.describe()


# In[ ]:


data_train.shape


# In[ ]:


data_train = data_train.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)


# In[ ]:


data_train.shape


# also dropping the id fields from the two sets

# In[ ]:


data_test_2 = data_test.drop(columns=['id'])
data_train_X = data_train.drop(columns=['id'])
print('train shape:', data_train_X.shape)
print('test shape:', data_test_2.shape)


# In[ ]:


#select data and targets
fea_col = data_train.columns[2:]
data_Y = data_train_X['target']
data_X = data_train_X[fea_col]


# In[ ]:


# will take only 30k for training since it takes sooo long

print('X_train shape: ', data_X.shape)
X_train_END = data_X[0:100000]
print('X_train_2 shape: ', X_train_END.shape)

print('y_train shape: ', data_Y.shape)
y_train_END = data_Y[0:100000]
print('y_train_2 shape: ', y_train_END.shape)


#
print('test values ', data_test_2.shape)


# In[ ]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')


# In[ ]:


X_train_smote_END, y_train_smote_END = smote.fit_sample(X_train_END, y_train_END)


# In[ ]:


X_train_smote_END.shape


# In[ ]:


clf = svm.SVC(kernel='linear', C=1.0)


# In[ ]:


#fitting
clf.fit(X_train_smote_END, y_train_smote_END)


# In[ ]:


# now predict the data_test values
y_predict_smote = clf.predict(data_test_2)


# we of course cannot check the accuracy since we dont have any values to compare them with, but we can check how many ones we got to make sure the values are not all over the place.
# Lets quickly remember: 96% of our training data had zeros and 4% were ones.
# We now predicted a total of 148800 values.

# In[ ]:


y_predict_smote.shape


# In[ ]:


four_percent_of_all = 144880 * 0.04
print('4 % of all points would be:', four_percent_of_all)


# In[ ]:


print('# of our predicted values: ' , sum(y_predict_smote))


# #### 7500 ones.
# Looks alright. If we assume a accuracy of 90% that seems to be quite a successful prediction.

# In[ ]:


data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_predict_smote, True) 
data_out.to_csv("/Users/hercules/ml/data/final-project-dataset/submission.csv",index=False)


# In[ ]:


from io import StringIO


# In[ ]:


output = StringIO()


# In[ ]:


data_out.to_csv(output)


# In[ ]:


output.seek(0)


# In[ ]:


print(output.read())


# ### similar approach but with random forest:

# In[ ]:


filename = "train_set.csv"
data_train = pd.read_csv("data/final-project-dataset/train_set.csv")
filename = "test_set.csv"
data_test = pd.read_csv("data/final-project-dataset/test_set.csv")

data_train = data_train.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)
data_train_X = data_train.drop(columns=['id'])
data_test = data_test.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)
data_test_2 = data_test.drop(columns=['id'])

#select data and targets
fea_col = data_train.columns[2:]
data_Y = data_train_X['target']
data_X = data_train_X[fea_col]

# will take only 30k for training since it takes sooo long

print('X_train shape: ', data_X.shape)
X_train_END = data_X[0:10000]
print('X_train_END shape: ', X_train_END.shape)

print('y_train shape: ', data_Y.shape)
y_train_END = data_Y[0:10000]
print('y_train_END shape: ', y_train_END.shape)


#
print('test values ', data_test_2.shape)


# In[ ]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(sampling_strategy='minority')


# In[ ]:


X_train_smote_END, y_train_smote_END = smote.fit_sample(X_train_END, y_train_END)


# In[ ]:


print('X_train shape: ', X_train_smote_END.shape)


# In[ ]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train_smote_END, y_train_smote_END)


# In[ ]:


test_10k = data_test_2[0:10000]


# In[ ]:


predictions = rf.predict(test_10k)


# In[ ]:


print(10000*0.04)
print('# of our predicted values: ' , sum(predictions))
print('without smote sampleing it was at 478')


# In[ ]:


predict_all = rf.predict(data_test_2)


# In[ ]:


print(148000*0.04)
print('# of our predicted values: ' , sum(predict_all))
print('without smote sampleing it was at 7169')


# ### so looks like smote upsampleing doubles the number of predicted ones...
# not really what we are looking for
# will try again with more training values than 10k:

# In[ ]:


filename = "train_set.csv"
data_train = pd.read_csv("data/final-project-dataset/train_set.csv")
filename = "test_set.csv"
data_test = pd.read_csv("data/final-project-dataset/test_set.csv")

data_train = data_train.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)
data_train_X = data_train.drop(columns=['id'])
data_test = data_test.drop(["ps_reg_03", "ps_car_03_cat", "ps_car_05_cat", "ps_car_14"], axis = 1)
data_test_2 = data_test.drop(columns=['id'])

#select data and targets
fea_col = data_train.columns[2:]
data_Y = data_train_X['target']
data_X = data_train_X[fea_col]

# will take only 30k for training since it takes sooo long

print('X_train shape: ', data_X.shape)
X_train_END = data_X[0:30000]
print('X_train_END shape: ', X_train_END.shape)

print('y_train shape: ', data_Y.shape)
y_train_END = data_Y[0:30000]
print('y_train_END shape: ', y_train_END.shape)


#
print('test values ', data_test_2.shape)


# In[ ]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(X_train_END, y_train_END)


# In[ ]:


# create prediction of our test set:
predict_all = rf.predict(data_test_2)


# In[ ]:


print(148000*0.04)
print('# of our predicted values: ' , sum(predict_all))


# In[ ]:


cnt = 0
for i in range(predict_all.size):
    if predict_all[i] > 0.5:
        predict_all[i] = 1
        cnt = cnt + 1
    else:
        predict_all[i] = 0
cnt


# # ---------- end ----------

# ### Submission
# 
# Please only submit the csv files with predicted outcome with its id and target [here](https://www.kaggle.com/t/b3dc81e90d32436d93d2b509c98d0d71). Your column should only contain `0` and `1`.

# In[ ]:


data_out.describe()


# In[ ]:





# In[ ]:




