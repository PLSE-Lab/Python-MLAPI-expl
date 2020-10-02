#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# This is a very basic EDA for the titanic dataset. It provides some quick insights into the data for people who just want to get a general feel for it. I didn't include much on feature engineering or preprocessing. If you have any advice/thoughts, please leave a comment!
# 
# Also if you found it helpful some **upvotes** are greatly appreciated!

# # Read data

# In[ ]:


data_df = pd.read_csv("../input/titanic/train.csv")
data_df.head()


# In[ ]:


data_df.describe(include="all")


# Here are some initial observations from the two tables:
# 1. There are quite a few categorical features (nominal, ordinal, and binary). Some of the ordinal features are disguised as discrete numerical features (Pclass).
# 2. The feature Name has 891 unique values for 891 values total, so there's no point in including it as is. However, we might be able to find some way to engineer other features from name, such as the prefix, family information etc.
# 3. The feature ticket has some repeating values, but I'm not too sure what that might mean.

# # Univariate Analysis

# In[ ]:


sns.countplot(x="Survived", data=data_df)


# Looks like most people didn't survive, which is to be expected. There might be some dataset inbalance here.

# In[ ]:


sns.countplot(x="Pclass", data=data_df)


# As expected, amount of people in class three is the greatest. It's interesting how in our sample there are more people in class one than two though.

# In[ ]:


sns.countplot(x="Sex", data=data_df)


# There's significantly more male than female people in our sample.

# In[ ]:


sns.distplot(a=data_df["Age"].dropna())


# Looks like most people are between 20 and 40 years old. It will be interesting to explore the relationship between age and gender in the next section.

# In[ ]:


sns.countplot(x="SibSp", data=data_df)


# In[ ]:


sns.countplot(x="Parch", data=data_df)


# Most people are traveling alone.

# In[ ]:


sns.countplot(x="Embarked", data=data_df)


# In[ ]:


sns.distplot(a=data_df["Fare"])


# In[ ]:


sns.boxplot(x="Fare", data=data_df)


# Looks like there are a lot of outliers for the fare feature.

# # Bivariate Analysis

# In[ ]:


sns.heatmap(data_df.corr(), annot=True)


# Before seeing how each feature impacts survived, there are a couple relations from the heatmap that might be interesting to explore. First up, plotting sex, pclass and age to get a better idea of the demographics in our dataset.

# In[ ]:


sns.violinplot(x="Pclass", y="Age", hue="Sex", data=data_df)


# Overall they are distributed pretty evenly, with females being slightly younger overall. We'll look at if Pclass is strongly correlated with fare next.

# In[ ]:


sns.swarmplot(x="Pclass", y="Fare", data=data_df)


# It is indeed. It's natural that people in higher Pclasses pay more. Though the few outliers who paid more than 500 is interesting. Also as the classes go up, the fares are becomming more spread out.
# 
# Now let's examine how each feature relates to Survived. First up is Pclass.

# In[ ]:


sns.countplot(x="Pclass", hue="Survived", data=data_df)


# As expected, the survived to died ratio grows as the pclass increases. Higher class passengers are probably given priority when boarding lifeboats.

# In[ ]:


sns.countplot(x="Sex", hue="Survived", data=data_df)


# Surprisingly over half of the women survived. On the other hand, the majority of men died. Women might also have been given priority during the evacuation.

# In[ ]:


grouped_df = data_df.groupby("Survived")
survived_df = grouped_df.get_group(1)

age_survived = survived_df["Age"].dropna()
sns.distplot(a=age_survived, label="Survived", bins=range(0, 80, 5), kde=False)
sns.distplot(a=data_df["Age"].dropna(), label="Total", bins=range(0, 80, 5), kde=False)
plt.legend()


# I chose to overlay two distplots because doing a bar plot like before will make the graph look too crowded. This graph effectively shows the same information of the previous bar charts. The higher the blue bar is inside the yellow bar, the higher the survival rate.
# 
# It seems like, in general, older people tend to survive a lot less, and very young children tend to have an extremely high survival rate. This might be that young children are given priority when evacuating just as women might have been.
# 
# Next we'll look at SibSp (number of siblings and spouses)

# In[ ]:


sibsp_survived = survived_df["SibSp"]
sns.distplot(a=sibsp_survived, label="Survived", bins=range(0, 8, 1), kde=False)
sns.distplot(a=data_df["SibSp"], label="Total", bins=range(0, 8, 1), kde=False)
plt.legend()


# Here the results are surprising, as people with 1 sibling/spouse seems to have the highest survival rate. I originally expected people with 0 siblings/spouses to survive best, as they don't have to worry about anyone else, but thinking about it more those siblings/spouses will also look out for you and help you survive.
# 
# Next we'll look at Parch, before combining these two to look at total family number on board.

# In[ ]:


parch_survived = survived_df["Parch"]
sns.distplot(a=parch_survived, label="Survived", bins=range(0, 6, 1), kde=False)
sns.distplot(a=data_df["Parch"], label="Total", bins=range(0, 6, 1), kde=False)
plt.legend()


# Again the survival rate seems to be higher for people with parents/children. I really wish this column is separated into number of parents and number of children, though, because parents are garenteed to care about children while children are the ones being cared about. So a high number of children might mean less survival rate, but high number of parents might mean higher chance etc.
# This can actually be done by using the name prefix or age combined with Parch, but feature engineering is out of the scope of this EDA.
# 
# Next we'll combine SibSp and Parch to examine total family member number on board.

# In[ ]:


family_survived = survived_df["Parch"] + survived_df["SibSp"]
sns.distplot(a=family_survived, label="Survived", bins=range(0, 11, 1), kde=False)
sns.distplot(a=data_df["Parch"]+data_df["SibSp"], label="Total", bins=range(0, 11, 1), kde=False)
plt.legend()


# This plot further confirms the fact that having someone to look out for you is nice. However, as the relationship of looking out for someone and having someone look out for you is combined in this feature, there is an optimal point. There are very few survivers past the point of having more than 4 family members on board, as too large a group is less likely to survive.
# 
# Next we'll look at fare.

# In[ ]:


fare_survived = survived_df["Fare"]
sns.distplot(a=fare_survived, label="Survived", bins=range(0, 200, 10), kde=False)
sns.distplot(a=data_df["Fare"], label="Total", bins=range(0, 200, 10), kde=False)
plt.legend()


# The trend here is pretty clear. In general, the more fare money a passenger paid, the more likely they are to survive. I cut the graph window at $200 because past that point it's hard to see more bars because the count is so small.
# 
# Finally for the bivariate section we'll look at placed embarked.

# In[ ]:


sns.countplot(x="Embarked", hue="Survived", data=data_df)


# Surprisingly, most people who embarked at C survived, while for S and Q much more died than lived. Let's explore why this is the case.

# In[ ]:


sns.countplot(x="Embarked", hue="Pclass", data=data_df)


# Ah, the reason is that a lot of people who embarked at C are class 1, meaning they are more likely to survive.

# # Missing Values

# In[ ]:


data_df.isnull().sum()


# We can see that age has 177 missing values, but it is quite a helpful predictor. We'll use mean value to impute as the purpose of this kernel is not preprocessing. Alternative strategies include grouping people by features like name prefix (title), parch, etc. and using the mean values for each group to impute.
# 
# We'll also fill in embarked with S, because it is the most common embarked location.
# 
# As for Cabin, the information it contains might help because some cabins might be closer to exits etc. So we'll impute the missing values with another category: Other. It might help to add anotehr feature: "HasImputedCabin", but from my experiences it didn't really help much.

# In[ ]:


data_df["Age"] = data_df["Age"].fillna(np.mean(data_df["Age"]))
data_df["Embarked"] = data_df["Embarked"].fillna("S")
data_df["Cabin"] = data_df["Cabin"].fillna("Other")
data_df.isnull().sum().sum()


# # Feature Importances

# Nice! All the missing values are gone. Now let's look at some feature importances. The models I'm using here are all tree-based, so I won't be using one-hot encoding at all. Take this section with a grain of salt, as feature importances are heavily model dependent. Tree based models perform this step internally when looking for the splits that would result in the highest purity, so I used them here.

# In[ ]:


data_df


# In[ ]:


data_df.drop(["Name", "Ticket", "PassengerId"], axis=1, inplace=True)
data_df["Sex"] = LabelEncoder().fit_transform(data_df["Sex"])
data_df["Cabin"] = LabelEncoder().fit_transform(data_df["Cabin"])
data_df["Embarked"] = LabelEncoder().fit_transform(data_df["Embarked"])


# In[ ]:


rf = RandomForestClassifier(n_estimators=500)
et = ExtraTreesClassifier(n_estimators=500)

rf.fit(data_df.drop("Survived", axis=1), data_df["Survived"])
et.fit(data_df.drop("Survived", axis=1), data_df["Survived"])


# In[ ]:


figures, axes = plt.subplots(1, 2, figsize=(24, 12))
print(rf.feature_importances_)
axes[0].bar(height=rf.feature_importances_, x=range(1, 9), tick_label=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"])
axes[0].set_title("Random Forest Feature Importances")
axes[1].bar(height=et.feature_importances_, x=range(1, 9), tick_label=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"])
axes[1].set_title("Extra Trees Feature Importances")


# That all for a quick overview of this dataset.
# Hope you liked this kernel!
