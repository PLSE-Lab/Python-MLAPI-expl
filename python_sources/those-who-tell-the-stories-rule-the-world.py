#!/usr/bin/env python
# coding: utf-8

# <center><img src="https://i.ibb.co/KhvCRfb/mining-rules.png"></center>

# <center><h3>TLDR: using bayesian rule mining to explain parts of the data <br> Use case: users who use Bayesian Approaches</h3></center>
# 
# <br>
# Visualisations are great, and scientists use them in many ways to start exploring a dataset.
# However, the power of visualisation lies in how they are interpreted. Therefore, in this notebook: **can we use ML to generate interpretations directly from the data itself? **

# In[ ]:


import pandas as pd
from sklearn import preprocessing
import random
import numpy
import warnings
import ruleset as rs


warnings.filterwarnings('ignore')

random.seed(42)
numpy.random.seed(42)


# To find interpretations in the data, an exciting dataset must be provided. The *multiple_choice_responses* table was used in this perspective because: 
# * Each row is similar to a single user, answering multiple questions.
# If we want to learn interpretations over multiple users, this table format is desirable.
# * Almost all values in the cells are categorical, which is preferable to discriminate.

# In[ ]:


df = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv')
#example data
df[['Q1','Q2','Q3','Q4','Q5','Q6','Q7', 'Q8']].head()


# While it is not impossible, generating explanations for this whole dataset would be a little bit harsh. Therefore, we will select a specific column from which we want to discriminate the users. <br>
# 
# <center>**More in general, we will try to find rules which can discriminate users based on a single column value given the information in the other columns. <br>Column Q24_Part_4 will be used, because I want to know how you specify people using Bayesian approaches regularly ;)** </center><br>
# 
# 

# In[ ]:


df['Q24_Part_4'].value_counts().sort_index(ascending=False)


# Dataset loaded, the target column is chosen, time to define our methodology:
# <h3> Association rule mining </h3>
# Applying association rule mining is possible here due to the nature of the data.<br>
# Given a set of answers on questions, association rule mining will try to find rules that will predict the occurrence of an answer on a question based on the answers of the other questions in the set. While this is maybe rather hard to understand in the setting of this particular dataset with answers and questions, it is more clear in the context of basket analysis where your grocery store tries to find patterns in customer behaviour.
# <img src="https://pbs.twimg.com/media/CSS1Q6TUwAAdGnX?format=png&name=small">
# <h3> Bayesian association rule mining </h3>
# 
# Ok, one extra element: some Bayesian knowledge <br>
# Association rule mining is useful when you want many rules over the whole dataset. I want only rules for column Q24_Part_4.
#     Whether or not a rule is beneficial for our case can be verified using some sort of a naive Bayes classifier. When a rule is mined, each row is checked given that rule using Bayes rule:
# $$P(\text{A}\text{ }|\text{ }B) = \frac{P(B\text{ }|\text{ }\text{A})\text{ }x\text{ }P(\text{A})}{P(B)}$$
# where we try to classify people using Bayesian Approaches (A) given the data in our rule (B)
# <center><img src="https://i.pinimg.com/originals/d8/fe/9d/d8fe9dd4f6e81aee6bb56ed1a234dd0d.png"></center>

# <h2> May i have your attention please.. </h2>
# 
# We've used this standard code to perform Bayesian rule mining: https://github.com/zli37/bayesianRuleSet

# In[ ]:


# load dataset, remove first row (=question text)
df = pd.read_csv('../input/kaggle-survey-2019/multiple_choice_responses.csv').iloc[1:]
# replace the underscore in the dataset, just for to get a better output
df.columns = df.columns.str.replace('_', '')
# unknown for all NaN values, we will discriminate between 'Bayesian approach' & 'Unknown' in column Q24_Part_4
df.fillna('unknown', inplace=True)

# X: training set, 
# !!!! we use a predefined set of columns to speed up the execution time !!!! (normally use .drop(['Q24Part4', , axis=1))
X = df[['Q9Part5', 'Q18OTHERTEXT', 'Q19', 'Q24Part5', 'Q18Part2', 'Q8', 'Q24Part9', 'Q29Part10']]
# change dtype of columns to string (categorical)
for col in X.columns:
    X[col] = X[col].apply(str)

# encode label
le = preprocessing.LabelEncoder()
# we want rules specifying the 'Bayesian approach' community
y = 1-le.fit_transform(df.Q24Part4)

# let's mine some rules, limit to max 5 conjunctive rules of max length 5
# https://github.com/zli37/bayesianRuleSet
model = rs.BayesianRuleSet(method='forest', max_iter=10000, maxlen=5, max_rules=5)
model.fit(X, y)


# The output shows some iterations, but we are only interested in the last one. It states: <br>
# <center> Bayesian approaches ->  not Q18OTHERTEXT(174) ^ Q24Part5(Evolutionary Approaches) ^ not Q8(I do not know) ^ Q24Part9(Recurrent Neural Networks)</center> <br>
# Alternatively, the rule states that **people who use Bayesian approaches frequently do not use 174 as a programming language (regularly), follows updates on evolutionary approaches and neural nets, knows that their current employer incorporates ML**. 

# Remarks:
# Although we have an accuracy of 81.7% keep in mind that this is an imbalanced problem...
# 
# |                     | True positives | True negatives |
# |---------------------|:--------------:|:--------------:|
# | Predicted positives |       253      |       118      |
# | Predicted negatives |      3472      |      15874     |
# 
# So this rule, found above,will only hold for 253 Bayesian users...
# 
# | Measure                          | Value  | Derivations                                           |
# |----------------------------------|--------|-------------------------------------------------------|
# | Sensitivity                      | 0.0679 | TPR = TP / (TP + FN)                                  |
# | Specificity                      | 0.9926 | SPC = TN / (FP + TN)                                  |
# | Precision                        | 0.6819 | PPV = TP / (TP + FP)                                  |
# | Negative Predictive Value        | 0.8205 | NPV = TN / (TN + FN)                                  |
# | False Positive Rate              | 0.0074 | FPR = FP / (FP + TN)                                  |
# | False Discovery Rate             | 0.3181 | FDR = FP / (FP + TP)                                  |
# | False Negative Rate              | 0.9321 | FNR = FN / (FN + TP)                                  |
# | Accuracy                         | 0.8179 | ACC = (TP + TN) / (P + N)                             |
# | F1 Score                         | 0.1235 | F1 = 2TP / (2TP + FP + FN)                            |
# | Matthews Correlation Coefficient | 0.1744 | TP*TN - FP*FN / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)) |
# 
# <br>
# More advanced iterations will probably increase the precision of the found rules. <br>
# Here, we only wanted to show the usefulness of rule mining in the area of explaining parts of the data. <br>
# 
# <center>**Maybe, more interesting rules can be found for other columns?**</center>
# 
# 
# 

# <center><img src="https://www.invespcro.com/blog/images/blog-images/Featured.1.jpg"></center>
