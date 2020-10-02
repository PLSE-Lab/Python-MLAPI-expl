#!/usr/bin/env python
# coding: utf-8

# ## Titanic - Simple Gender + Family Group Model

# Greetings! This notebook outlines a somewhat different approach to creating a predictive model using the Titanic dataset.
# 
# After trying (and failing) to score higher than ~78% using numerous variations of stacked ensembles, I decided to take a step back. Increasing complexity was obviously not increasing model accuracy. Perhaps if I better understood the problem and increased my domain knowledge, I would have a better chance of moving up the leaderboard.
# 
# I stumbled across an interesting notebook by [Chris Deotte](https://www.kaggle.com/cdeotte) in which he creates a relatively high-scoring model using the Name feature alone: [Titanic using Name only](https://www.kaggle.com/cdeotte/titanic-using-name-only-0-81818). Although parsing his R code was a bit of a challenge, Chris' commentary was extremely clear and logical. The basic premise is that we can infer family groups based on passenger last name, which is extracted from the Name field. Survival within these so-called "woman-child-groups" is almost always binary - either all members of the WCG die or all survive. Therefore, if a test case is presented for which we can determine family group status, we should predict survival based on the survival of the family group. Using this group-based strategy in conjunction with the baseline gender-based model in which all men die and all females survive results in an accuracy in the low 80%s. Having spent ages trying to perfect a more complicated stacked ensemble with no improvement, I decided to explore Chris' ideas and implement them in Python. Who knows, maybe I could find an enhancement or two!

# ### Setup

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold, KFold, RandomizedSearchCV, GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
import xgboost as xgb

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

SEED = 42
NFOLDS = 10

train_ = pd.read_csv("../input/titanic/train.csv")
test_ = pd.read_csv("../input/titanic/test.csv")


# In[ ]:


test = test_.copy()
train = train_.copy()
test_train = pd.concat([test, train], sort=False)
train.head()


# ### Data Preprocessing

# **Title**

# In[ ]:


def extract_title(x):
    return x.split(', ')[1].split(". ")[0].strip()


# In[ ]:


for dataset in [train, test, test_train]:
    dataset["Title"] = dataset["Name"].apply(extract_title)


# **Last Name**

# In[ ]:


def extract_last_name(x):
    return x.split(",")[0].strip()


# In[ ]:


for dataset in [train, test, test_train]:
    dataset["LastName"] = dataset["Name"].apply(extract_last_name)


# **Age**

# In[ ]:


for dataset in [train, test, test_train]:
    dataset["Age"] = dataset["Age"].fillna(99)


# **Fare**

# In[ ]:


for dataset in [train, test, test_train]:
    dataset["Fare"] = dataset["Fare"].fillna(0)


# **Embarked**

# In[ ]:


for dataset in [train, test, test_train]:
    dataset["Embarked"] = dataset["Embarked"].fillna(dataset["Embarked"].mode()[0])


# I decided to add several new features to the dataset to use as diagnostics when grouping families. I've noticed that many notebooks mention how SibSp and Parch are inaccurate, so we'll take any metrics based on these features with a grain of salt.

# **Family Size**

# In[ ]:


for dataset in [train, test, test_train]:
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
    
sns.catplot(x="FamilySize", y="Survived", data=train, kind="bar")
plt.show()


# **Solo Travelers**

# In[ ]:


for dataset in [train, test, test_train]:
    dataset["IsSolo"] = dataset["FamilySize"] == 1

sns.catplot(x="IsSolo", y="Survived", data=train, kind="bar")
plt.show()


# **Family Groups**
# 
# We can observe that, in general, family groups share a last name, a ticket number and fare amount. However, in some cases, the last digit of the ticket number varies within family groups (ex: Vander Planke family) and occasionally the fare amounts don't match. Deciding how to group families is critical for this strategy to be effective.
# 
# On my first pass, I constructed FamilyGroup using Last Name + Ticket[:-1] (ignoring Fare). This model produced a LB score in the low 80%s. I've left the code in, below, for clarity - labeled FamilyGroupOld.
# 
# On my second pass, I constructed FamilyGroup using Pclass + Ticket[:-1] + Embarked + Fare. This model produced a LB score in the low 81%s. More analysis is needed to determine whether this feature misses any of the information of the original last name-based version. Perhaps both methods should be used in conjuction?

# In[ ]:


for dataset in [train, test, test_train]:
    dataset["FamilyGroup"] = dataset["Pclass"].astype(str) + " - " + dataset["Ticket"].str[:-1] + " - " + dataset["Embarked"] + " - " + dataset["Fare"].astype(str)
    dataset["FamilyGroupOld"] = dataset["LastName"] + " - " + dataset["Ticket"].str[:-1]


# We can look at passengers by last name to verify our FamilyGroup feature.

# In[ ]:


train.loc[train["LastName"] == "Andersson"]


# In[ ]:


train.loc[train["FamilyGroup"] == "3 - 34708 - S - 31.275"]


# In[ ]:


train.loc[train["LastName"] == "Vander Planke"]


# **Masters**

# An important part of the grouping strategy is identifying the young boys who traveled with female passengers. These individuals are given the title "Master".

# In[ ]:


train[train["Title"] == "Master"].head()


# The first thing I wondered is whether the title "Master" was only given to young boys. What is the distribution of ages of passengers with the title "Master"?

# In[ ]:


masters = train.loc[(train["Title"] == "Master") & (train["Age"] != 99)]
sns.distplot(masters["Age"].dropna(), bins=7)
plt.ylabel("Density")
plt.show()


# The maximum age of passengers with the title "Master" is 14 (in the training dataset).

# In[ ]:


masters["Age"].describe()


# Are there any young men that do not have the title "Master"?

# In[ ]:


boys_without_master = train.loc[(train["Age"] < 18) & (train["Sex"] == "male") & (train["Title"] != "Master")]
boys_without_master


# In[ ]:


len(boys_without_master)


# As we see, there are 29 young males in the training set who are under 18 years of age but do not have the title "Master". We might consider broadening our definition of "boy" to include these individuals.

# **Potential Families**

# In our model, families consist of 2 or more females + masters who share the same FamilyGroup. Let's count the number of such individuals in our training, test and combined dataset. We can then calculate the training survival rate.

# In[ ]:


test_train_group_count = test_train.loc[(test_train["Sex"] == "female") | (test_train["Title"] == "Master")].groupby("FamilyGroup").count()[["PassengerId"]].sort_index()
test_train_group_count.columns = ["Train + Test Count"]

train_group_count = pd.merge(train.loc[(train["Sex"] == "female") | (train["Title"] == "Master")].groupby("FamilyGroup").count()[["PassengerId"]], 
                             train.loc[(train["Sex"] == "female") | (train["Title"] == "Master")].groupby("FamilyGroup").sum()[["Survived"]], how="inner", on="FamilyGroup")
train_group_count.columns = ["Train Count", "Survived"]

test_group_count = test.loc[(test["Sex"] == "female") | (test["Title"] == "Master")].groupby("FamilyGroup").count()[["PassengerId"]].sort_index()
test_group_count.columns = ["Test Count"]


# In[ ]:


groups = pd.merge(pd.merge(test_train_group_count, train_group_count, how="left", on="FamilyGroup"), test_group_count, how="left", on="FamilyGroup")
groups["Train Survival Rate"] = groups["Survived"] / groups["Train Count"]
groups = groups.reset_index()
groups.head()


# As mentioned above, I originally constructed FamilyGroup using passenger last name. However there were a handful of occasions when this resulted in individuals being left out of groups because they had a different last name. For this reason, I selected a different grouping method using Ticket, Embarked and Fare. Below is a comparison of the number of passengers in each new FamilyGroup compared to the original method.

# In[ ]:


temp = test_train.sort_values(by="Ticket")
fg_counts1 = temp.groupby("FamilyGroup").count().iloc[:,0]
fg_counts2 = temp.groupby("FamilyGroupOld").count().iloc[:,0]

fg_comparison = pd.merge(pd.merge(temp[["PassengerId", "FamilyGroup", "FamilyGroupOld"]], fg_counts1, on="FamilyGroup", how="left"), fg_counts2, on="FamilyGroupOld", how="left")
fg_comparison.columns = ["PassengerId", "FamilyGroup", "FamilyGroupOld", "CountFamilyGroup", "CountFamilyGroupOld"]
fg_comparison["CountFamilyGroup"] = fg_comparison["CountFamilyGroup"].astype(int)
fg_comparison = fg_comparison.sort_values(by="FamilyGroup")
fg_comparison[fg_comparison["CountFamilyGroup"] != fg_comparison["CountFamilyGroupOld"]].head()


# Let's double check a few family groups to make sure our counting is correct.
# 
# Filtering for FamilyGroup "1 - 11081 - C - 75.25" - there should be 1 passenger in the training set who survived:

# In[ ]:


train[train["FamilyGroup"] == "1 - 11081 - C - 75.25"]


# Now let's look at FamilyGroup "1 - 11378 - S - 151.55" - there should be 4 passengers in the training set, 2 of whom survived:

# In[ ]:


train[train["FamilyGroup"] == "1 - 11378 - S - 151.55"]


# It looks like our counts are correct. Now we can filter for groups of 2 or more passengers.

# In[ ]:


familygroups = groups[groups["Train + Test Count"] > 1][["FamilyGroup"]]
familygroups.head()


# In[ ]:


families = groups[groups["FamilyGroup"].isin(familygroups["FamilyGroup"])]
families.head()


# ### Generating Predictions

# We're ready to make predictions using our model.
# 
# First, all males without the title "Master" are predicted to die.

# In[ ]:


test_males_xmasters_ids = test.loc[(test["Sex"] == "male") & (test["Title"] != "Master")]["PassengerId"]
test_females_masters_ids = test.loc[(test["Sex"] == "female") | (test["Title"] == "Master")]["PassengerId"]


# In[ ]:


test_males_xmasters_preds = pd.DataFrame({"PassengerId": test_males_xmasters_ids, "Survived": np.zeros(len(test_males_xmasters_ids), dtype=int)})
test_males_xmasters_preds.head()


# In[ ]:


len(test_males_xmasters_preds)


# Next, all females and males with the title "Master" follow the fortunes of their group members. If survival rate for the group is greater than 50%, test cases are predicted to survive.

# In[ ]:


test_females_masters = test.loc[test["PassengerId"].isin(test_females_masters_ids)]
test_females_masters_rates = pd.merge(test_females_masters, groups, how="left", on="FamilyGroup")[["PassengerId", "FamilyGroup", "Sex", "Age", "Train + Test Count", "Train Count", "Survived", "Train Survival Rate"]]
test_females_masters_rates["Train Survival Rate"].fillna(-1, inplace=True)
test_females_masters_rates.head()


# In[ ]:


test_females_masters_rates["Prediction"] = np.zeros(len(test_females_masters_rates), dtype=int)
test_females_masters_rates.loc[(test_females_masters_rates["Sex"] == "female"), "Prediction"] = 1
test_females_masters_rates.loc[(test_females_masters_rates["Train Survival Rate"] >= 0.5), "Prediction"] = 1
test_females_masters_rates.loc[(test_females_masters_rates["Train Survival Rate"] < 0.5) & (test_females_masters_rates["Train Survival Rate"] != -1), "Prediction"] = 0
test_females_masters_rates.head()


# ### Creating a Submission Output

# That's it! Let's generate a submission output.

# In[ ]:


test_females_masters_preds = test_females_masters_rates[["PassengerId", "Prediction"]]
test_females_masters_preds.columns = ["PassengerId", "Survived"]
output = pd.concat([test_males_xmasters_preds, test_females_masters_preds]).sort_values(by="PassengerId")
output.to_csv("submission.csv", index=False)
output.head()


# We're predicting that 150 individuals survived. Note that this is not the same number of total females in the test dataset, hence our group model has deviated slightly from the baseline gender model.

# In[ ]:


np.sum(output["Survived"])


# In[ ]:


len(test[test["Sex"] == "female"])


# ### Conclusion

# Thanks so much for reading. I had a lot of fun exploring this strategy and have a few ideas for using a simple decision tree model to generate predictions for the female + master population. If I have any success, I will create a new notebook :)
# 
# Please leave a comment below with your thoughts and ideas to improve the women-child-group strategy.
# 
# Until next time, happy coding.
