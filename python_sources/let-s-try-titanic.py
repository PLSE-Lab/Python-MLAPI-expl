#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
feature_list = []
df.head()


# In[ ]:


df.drop("PassengerId", 1, inplace=True)
df.head()


# In[ ]:


df.info()


# ## Pclass
# The objectives of this section includes:
# 
# 1. To check the relationship between *Survived* and *Pclass*.
# 2. To decide whether or not to put the *Pclass* to final features list.

# In[ ]:


plt.figure(figsize=(6, 6))
plt.pie(df["Pclass"].value_counts(), labels=df["Pclass"].value_counts().index, autopct='%1.1f%%')
plt.title("Pclass Percentages")
plt.show()


# Let's check the survival rates of passenger classes.

# In[ ]:


print(df.groupby(by='Pclass').mean()["Survived"])


# The results say that majority of people who did survive belongs to the 1st class and 2nd class. We will calculate the correlation between them to see how stronger is this relation.

# In[ ]:


df["Pclass"].corr(df["Survived"])


# There is a negative correlation between *Pclass* and *Survived* and it is significant. As the *Pclass* increases *Survival* is decreases. This statement supports the previous ratios. So, we definitely put *Pclass* into our final feature list.
# 
# Thus, our final feature list now includes: `['Pclass']`

# In[ ]:


feature_list.append('Pclass')


# ## Age
# Let's check how age affects the survival rate. The steps are similar to the *Pclass*. But, first we need to validate our data.
# 
# * Are there any NaN values?
# * Are there any non-logical values (such as input errors, etc.)?
# * How is age distributed along victims?

# In[ ]:


df["Age"].isna().sum()


# In[ ]:


print("{:.2f}%".format((df["Age"].isna().sum() / len(df.index)) * 100))


# There are 177 missing values. That is 19.87% of the whole data. This may be an important issue. But, let's continue with the data at hand.

# In[ ]:


age_description = df["Age"].describe()
age_description


# In[ ]:


plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.hist(x=df["Age"].dropna())
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Histogram")

plt.subplot(1, 2, 2)
sns.boxplot(x=df["Age"])
plt.title("Age Boxplot")
plt.show()


# In[ ]:


df["Age"].corr(df["Survived"])


# You can see that the plot is right skewed and there is no abnormal value in the histogram. The mean of age is 29.7 and the median is 28. You can see there are some outliers in the data but we know that this is normal since people can live 100 years or more in some rare cases. You can also see that there is negative correlation between age and survival rate.
# 
# Let's fill the data with the mean and see how correlation changes.

# In[ ]:


temp_age = df["Age"].fillna(value=df["Age"].mean())


# Now, we will check the NaN count again and plot the previous graphs once more to check the changes are negligible.

# In[ ]:


print("NaN count:", temp_age.isna().sum())

plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
plt.hist(x=temp_age.dropna())
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Histogram")

plt.subplot(1, 2, 2)
sns.boxplot(x=temp_age)
plt.title("Age Boxplot")
plt.show()


# There is quite a bit of difference. This is because we fill NaN values with the mean. Let's look at to the correlation.

# In[ ]:


temp_age.corr(df["Survived"])


# We know that if you are a child you have a higher chance to survive. We can validate this deduction by splitting the data into 3 groups: child, adult, and old. These groups are put into a column named *Cat_Age* after categorical age.

# In[ ]:


df["Cat_Age"] = pd.cut(temp_age, bins=[0, 18, 60, 100], labels=[0, 1, 2]) # 0:child, 1:adult, 2:old
df.drop("Age", axis=1, inplace=True)


# Now, check for the correlation between *Cat_Age* and *Survived*.

# In[ ]:


sns.barplot(x=df["Cat_Age"], y=df["Survived"])
plt.xticks(np.arange(3), ("Child", "Adult", "Old"))
plt.show()


# In[ ]:


df["Cat_Age"].corr(df["Survived"])


# This is much better, right? We definitely add *Cat_Age* into our feature list. Our feature list now includes `[Pclass, Cat_Age]`. The latest snapshot of our dataset is like that:

# In[ ]:


feature_list.append('Cat_Age')
feature_list


# ## Sex
# The same questions as before apply for sex as well. Let's look our population in terms of *Sex*.

# In[ ]:


plt.figure(figsize=(6, 6))
plt.pie(df["Sex"].value_counts(), labels=df["Sex"].value_counts().index, autopct='%1.1f%%')
plt.title("Sex Percentages")
plt.show()


# Ok. Now, check for correlation. But, we need to encode it to numerical values since they are nominal strings.

# In[ ]:


df["Sex"] = df["Sex"].apply(lambda x: 0 if x == "male" else 1) # 0: male, 1: female


# In[ ]:


df["Sex"].corr(df["Survived"])


# A very high positive correlation. This means that as the gender changes from male to female to survivability increases. This is a very good feature and will be definitely added to our feature list. Now the feature list updated as `['Pclass', 'Cat_Age', 'Sex']`.

# In[ ]:


feature_list.append('Sex')
feature_list


# ## Siblings / Spouses & Parents / Children
# These two features can be considered as a single one. Because they both refer to family relationships. We can extract a new feature that states if the passanger travels alone or not according to family member counts.

# In[ ]:


df["Alone"] = df.apply(lambda row: 1 if row["SibSp"] + row["Parch"] == 0 else 0, axis=1) # 1: Alone, 0: Not alone


# Now, we can check the correlation between *Alone* and *Survived*.

# In[ ]:


df["Alone"].corr(df["Survived"])


# That's a sweet negative correlation. This means that if you are alone it is likely that you cannot survive. This is an important feature for us. Thus, we will add it to our feature list. The list is now `['Pclass', 'Cat_Age', 'Sex', 'Alone']`.

# In[ ]:


feature_list.append('Alone')
feature_list


# ## Ticket
# I think that *Ticket* is a rough one. Because it is not standard through passangers. So, I will write all of my trials and not delete any of them. **Be ready for lots of crappy things in this section. So, DON'T JUDGE ME!**
# 
# Let's start with how many NaNs and unique tickets are there.

# In[ ]:


df["Ticket"].isna().sum()


# In[ ]:


df["Ticket"].nunique()


# This number tells us there are 681 unique ticket numbers. What about the remaining 210? They may be the ones that traveled together. So, this can support the *Alone* feature! Also, some ticket numbers alphabetic prefixes. We should also consider extracting them from the number.

# In[ ]:


df.drop("Cabin", 1, inplace=True)
test_df.drop("Cabin", 1, inplace=True)
df.head()


# In[ ]:


df["Fare"] = pd.cut(df["Fare"], 3, labels=[0, 1, 2]) # 0: Low, 1: Medium, 2: High

test_df["Fare"].fillna(test_df["Fare"].mean(), inplace=True)
test_df["Fare"] = pd.cut(test_df["Fare"], 3, labels=["Low", "Medium", "High"])
df.head()


# In[ ]:


df["Fare"].corr(df["Survived"])


# In[ ]:


feature_list.append("Fare")
feature_list


# In[ ]:


df.head()


# In[ ]:


import sklearn.preprocessing


# In[ ]:


embarked_encoder = sklearn.preprocessing.LabelEncoder()

df["Embarked"].fillna("S", inplace=True)
df["Embarked"] = embarked_encoder.fit_transform(df["Embarked"])
df.head()


# In[ ]:


df["Embarked"].corr(df["Survived"])


# In[ ]:


feature_list.append("Embarked")
feature_list


# In[ ]:


import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics


# In[ ]:


X, y = df[feature_list], df["Survived"]


# In[ ]:


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)


# In[ ]:


model_rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100)
model_gb = sklearn.ensemble.GradientBoostingClassifier()


# In[ ]:


model_rf.fit(X_train, y_train)
model_gb.fit(X_train, y_train)


# In[ ]:


yhat_rf = model_rf.predict(X_test)
yhat_gb = model_gb.predict(X_test)


# In[ ]:


accuracy_rf = sklearn.metrics.accuracy_score(y_test, yhat_rf)
recall_rf = sklearn.metrics.recall_score(y_test, yhat_rf)
precision_rf = sklearn.metrics.precision_score(y_test, yhat_rf)

print("Random Forest Accuracy:", accuracy_rf)
print("Random Forest Recall:", recall_rf)
print("Random Forest Precision:", precision_rf)
print()

accuracy_gb = sklearn.metrics.accuracy_score(y_test, yhat_gb)
recall_gb = sklearn.metrics.recall_score(y_test, yhat_gb)
precision_gb = sklearn.metrics.precision_score(y_test, yhat_gb)

print("Gradient Boosting Accuracy:", accuracy_gb)
print("Gradient Boosting Recall:", recall_gb)
print("Gradient Boosting Precision:", precision_gb)
print()

