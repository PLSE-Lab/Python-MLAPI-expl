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


# In[ ]:


filename = "train_set.csv"
data_train = pd.read_csv(filename)
filename = "test_set.csv"
data_test = pd.read_csv(filename)


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


# **None of the label is detected!** Now with label-0 data only:

# In[ ]:


X_neg, y_neg = extrac_one_label(x_val, y_val, 0)
y_negpred = clf.predict(X_neg)
sum(y_negpred==y_neg)/len(y_neg)


# ### Your Turns:
# 
# What does it mean? Why does it look like that? How do you overcome it and get the better results? Hint :

# In[ ]:


print(sum(data_Y==0)/len(data_Y), sum(data_Y==1))


# ### Validation metric
# Think about what the proper metric is to train your model and how you should  change your training procedure to aviod this problem?

# In[ ]:


## Your work
#Please run the full provided project for necessary imports and variables
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

x_train, x_val, y_train, y_val = train_test_split(data_X, data_Y, test_size = 0.3, shuffle = True)

rus = RandomUnderSampler(random_state=0)
x_train, y_train = rus.fit_sample(x_train, y_train)

clf = LogisticRegression(solver='liblinear', random_state=0)
clf = clf.fit(x_train, y_train)
#test evaluation

#threshold
#need to do this because some 1 are wrong predicted
#probabilty 0.4 got the best results
probability = clf.predict_proba(x_val)
y_pred=list()
for proba in probability:
    if proba[0]<0.4: 
        y_pred.append(1)
    else:
        y_pred.append(0)

print(classification_report(y_val, y_pred))


# ### Submission
# 
# Please only submit the csv files with predicted outcome with its id and target [here](https://www.kaggle.com/t/b3dc81e90d32436d93d2b509c98d0d71). Your column should only contain `0` and `1`.

# In[ ]:


data_test_X = data_test.drop(columns=['id'])
probability = clf.predict_proba(data_test_X)
y_target=list()
for proba in probability:
    if proba[0]<0.4: 
        y_target.append(1)
    else:
        y_target.append(0)


# In[ ]:


data_out = pd.DataFrame(data_test['id'].copy())
data_out.insert(1, "target", y_target, True) 
data_out.to_csv('submission.csv',index=False)


# In[ ]:


data_out


# In[ ]:




