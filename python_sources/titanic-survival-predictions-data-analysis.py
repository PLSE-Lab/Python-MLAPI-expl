#!/usr/bin/env python
# coding: utf-8

# # Titanic Kaggle Kernel

# Hello there. This is my first kernel submission to Kaggle. Your feedback and comments are welcomed. 

# ## Table of Contents:
# - [Get the Data](#data)
# - [Exploring the Data](#explore)
#      * [Usefulness for the task](#usefulness)
#      * [Studying the Survivors](#survivors)
#          * [Correlation](#correlation)
#          * [Age](#age)
#          * [Fare](#fare)
#          * [Parch](#parch)
#          * [Pclass](#pclass)
#          * [SibSp](#sibsp)
#          * [Survived & Gender](#survivedandgender)
#          * [Survived, Gender & Age](#survivedandgenderandage)
# - [Prepare the Data](#prepare)
#     - [Handling Missing Data](#missingdata)
#     - [Feature Engineering](#features)
#         - [Family Size](#size)
#         - [Title](#title)
#         - [Family Members](#members)
#         - [Single Women](#single)
# - [ML](#model)
#     - [Logistic Regression Model](#logistic)

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import seaborn as sns


# ## Get the Data <a class="anchor" id="data"></a>

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/train.csv')


# In[ ]:


train.head()


# In[ ]:


train = train.drop(["PassengerId"], axis=1)


# ## Exploring the Data <a class="anchor" id="explore"></a>

# In[ ]:


# Before we start, let's copy the data.
train_copy = train.copy()
test_copy = test.copy()

train_copy.describe()


# In[ ]:


train_copy.info()


# The `Age`, `Cabin` and `Embarked` columns are missing some data. We'll come back to those categories later.

# ## Usefulness of a particular category for this task (A few ideas to start off with) <a class="anchor" id="usefulness"></a>

# ### KEY: NSY = "Note Sure Yet."
# 
# - PassengerId: As of now I don't see much use for the passenger id.
# - Survived: This is what we are trying to predict.
# - Pclass: I've been doing some research on the Titanic and I've found that **class** might actually play a key role in survival. The reason I say this is becuase the upper class had access to the upper decks of the titanic. After looking at a few diagrams of the decks of the titanic I found that the lifeboats were situated at the upper decks of the ship.
# - Name: I noticed that a lot of the names, just by looking through the dataframe above, have different titles (Mr, Mrs, etc..). I'm interested in seeing how those could influence our predictive model.
# - Sex: NSY
# - Age: NSY
# - SibSp: NSY
# - Parch: NSY
# - Ticket: At the moment I'm not really sure how knowing the ticket number will help me here. We'll explore this later.
# - Fare: Closely related with an individuals class so this one seems to be important as well.
# - Cabin: Closely related with an individuals class. 
# - Embarked: After doing a little research on the embarkation of the Titanic I found that this attribute has some significance in the likelyhood that a passenger survived the titantic. Earlier I mentioned that class could potentially play a key role in surviving the titanic due to where the lifeboats were situated in relation to the decks below. Most of the upperlcass came from Cherbourg and Southampton with only a very small percentage of upperclass people coming from Queenstown.
# 

# ## Studying the Survivors <a class="anchor" id="survivors"></a>

# ## Correlation <a class="anchor" id="correlation"></a>

# In[ ]:


corr = train_copy.corr()


# In[ ]:


corr["Survived"].sort_values(ascending=False)


# In[ ]:


import seaborn as sns

colormap = sns.cubehelix_palette(light=1, as_cmap=True)
a4_dims = (10, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(corr ,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# There seems to be a positive correlation between survival and fare. This would imply that, for example, if fare increases so does the chance of survival. This makes sense considering the upper class passengers were located in decks that were situated near the lifeboats. There is also a positive correlation between the number of parents/children an individual had and survival, this could be a good area where feature engineering might come in. 
# 
# There seems to also be a negative correlation between survival and age this implies that if, for example, age decreases the likelyhood of survival increase (and vice versa). There is also a negative correlation between survival and class which is quite interesting. Let's keep exploring.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
train_copy.hist(bins=50, figsize=(20,15)) 
plt.show()


# Let's look at the individual attributes against the `survived` column.

# ## Age <a class="anchor" id="age"></a>

# In[ ]:


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.violinplot("Survived", "Age", data=train_copy,
                   palette=["#E4421A", "#06EFA0"]);


# The violin plot above seems to suggest that most people who didn't survive were between the ages of 20 and 30-ish years old and that children had a decent chance of surviving. This would suggest, naturally, that the parents of the children aboard the Titanic would have done anything to keep them safe. This may help with giving use ideas for new features. One feature that comes to mind is looking at the size of a family. Since the probability of survival was fairly high if you were young then this may suggest that those who had a fairly good amount of members in their family would have tried to help each other survive. 

# ## Fare <a class="anchor" id="fare"></a>

# In[ ]:


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.violinplot("Survived", "Fare", data=train_copy,
                   palette=["#E4421A", "#06EFA0"]);


# The majority of those who survived didn't pay a high fare. Which suggests that class may not play a role despite my previous assumptions, but exploring the data a bit more will give us more insight.

# ## Parch <a class="anchor" id="parch"></a>

# In[ ]:


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
with sns.axes_style(style=None):
    sns.violinplot("Parch", "Age", hue="Survived", data=train_copy,
                      split=True, inner="quartile",
                      palette=["#E4421A", "#06EFA0"]);


# By observing this violin plot we learn the following attributes of the data:
# 
# 1. Those who had between 1 or 3 children had a decent survival rate.
# 2. Those who had more than 3 children did not survive.
# 3. And individual with 1-3 parents/children had a higher chance of survival than those who went on the Titanic alone or had a family size greater than 3. 
# 

# ## PClass <a class="anchor" id="pclass"></a>

# In[ ]:


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.violinplot("Survived", "Pclass", data=train_copy,
                   palette=["#E4421A", "#06EFA0"]);


# I recently studied the distance between the cabins and the lifeboats and found that those who were in the upperclass were located on decks that were situated near the lifeboats. This gave me the idea of creating a new feature that measures the proximity between passengers the lifeboats. This new feature might not have to do with Pclass, but rather the Cabin attribute. Another piece of insight is the fact that the distributions of those who survived and were in the upper class as well as the lower class are closely similar.

# Speaking of the Cabin attribute. This attribute is not a numerical attribute.

# In[ ]:


train_copy["Cabin"].describe().top


# We can't compare the `Cabin` attribute with `Survived` since we have missing values. We'll handle this in the `Handeling Missing Values` section.

# ## SibSp <a class="anchor" id="sibsp"></a>

# In[ ]:


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
with sns.axes_style(style=None):
    sns.violinplot("SibSp", "Age", hue="Survived", data=train_copy,
                      split=True, inner="quartile",
                      palette=["#E4421A", "#06EFA0"]);


# By observing this violin plot we learn the following attributes of the data:
# 
# 1. Those who had more than 4 siblings had a lower survival rate.
# 2. Those who were young and had between 0 and 2 siblings had a high survival rate. 
# 
# 
# By observing both the attributes found within SibSp and Parch I'm finding that having a certain family size played a role in the survival rate of an individual on the Titanic. Therefore, I'll create a **family size** feature to highlight this in the feature engineering section.

# One other new observation we can do is comparing those who surived based on their gender. 

# ## Survived & Gender <a class="anchor" id="survivedandgender"></a>

# In[ ]:


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
sns.violinplot("Sex", "Survived", data=train_copy,
                   palette=["#4F56CE", "#FF4365"]);


# The majority of survivors were female. Let's explore the relationship between those who survived, their gender and their age.

# ## Survived, Gender, & Age <a class="anchor" id="survivedandgenderandage"></a>

# In[ ]:


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
# Age by decade (for easier viewability)
train_copy['age_dec'] = train_copy.Age.map(lambda age: 10 * (age // 10))

with sns.axes_style(style=None):
    sns.violinplot("age_dec", "Survived", hue="Sex", data=train_copy,
                      split=True, inner="quartile",
                      palette=["#4F56CE", "#FF4365"]);


# There's a lot of information we can obtain just from looking at this violin plot.
# 
# 1. The group of individuals that were most likely to have survived (by age in decades) were children.
# 2. Females had the highest probability of survival.
# 3. Males had the lowest probability of survival.
# 
# 
# This gives me the idea of creating two new features: `Mother` & `Child`. Since these two groups of people had the most likely chance of surviving. This ties in with the `family size` feature. Since family members would have tried to have save one another. 

# ## Name 

# Earlier I mentioned that I was interested in seeing how the title of individuals on the Titanic effected their survival. In the following cells we'll be extracting the titles of various individuals and analyzing them.

# In[ ]:


def return_title(x):
    title = x.split(',')[1].split('.')
    return title[0].strip()

titles_train = train_copy["Name"].transform(return_title)
titles_test = test_copy["Name"].transform(return_title)


# This seems like the perfect opportunity to use one-hot encoding for these titles, but first we need to handle the sheer number of titles we have. We can condense a lot of these titles under specific categories (Ex: Mme = Miss)

# In[ ]:


def categorize_title(title):
    officer = ['Capt','Col','Major']
    miss = ['Miss', 'Mlle']
    mrs = ['Mrs', 'Ms', 'Mme']
    mr = ['Mr']
    other = ['Master', 'Rev', 'Dr', 'Jonkheer', 'Don', 'the Countess', 'Lady', 'Sir']
    
    if title in officer:
        return 'Officer'
    elif title in miss:
        return 'Miss'
    elif title in mrs:
        return 'Mrs'
    elif title in mr:
        return "Mr"
    elif title in other:
        return 'Other'

# We will save these for the feature engineering section 
saved_titles_train = titles_train.transform(categorize_title)
saved_titles_test = titles_test.transform(categorize_title)

saved_titles_test.unique()


# # Prepare the Data <a class="anchor" id="prepare"></a>

# ## Handling Missing Data <a class="anchor" id="missingdata"></a>

# In[ ]:


# Let's take a look at what's missing here
train_copy.info()


# In[ ]:


# Let's handle the Age column first. Let's use the mean for the missing values. 
mean_age_train = np.round(train_copy["Age"].mean())
mean_age_test = np.round(test_copy["Age"].mean())

train_copy["Age"] = train_copy["Age"].fillna(mean_age_train)
test_copy["Age"] = test_copy["Age"].fillna(mean_age_test)


# In[ ]:


train_copy.info()


# Next we need to handle the `Cabin` attribute. There are only 204 cabin numbers available out of a total of 889 passengers. We can't just randomly assign an individual a cabin since the cabin letter (A-F) might play a key role in suvival since the upper decks were closer to the life boats. For this reason I'll drop this attribute. 

# In[ ]:


train_copy = train_copy.drop("Cabin",axis=1)
test_copy = test_copy.drop("Cabin",axis=1)


# Now to deal with the `Embarked` attribute. There are a few NA data found in this attribute. First let's see how much the individuals with the NA values payed for their trip.

# In[ ]:


unknown_embarkment = train_copy[train_copy["Embarked"].isnull()]

print(unknown_embarkment["Fare"])


# Both passengers payed $80 to board the Titanic. Let's check the central tendency between fare and embarkment to see where these values would fall into.

# In[ ]:


grid = sns.factorplot("Embarked","Fare", hue="Pclass", data=train_copy, kind="box",size=9) 
grid.axes[0][0].hlines(80,-1000,1000)
grid.set_axis_labels("Embarked","Fare");


# The most probable area of embarkment for the two passengers based on what they have payed and what the average fare is for the different classes. It is reasonable to infer that both passengers must have embarked from Cherbourg (C).

# In[ ]:


unknown_embarkment


# In[ ]:


train_copy.set_value(61, 'Embarked', 'C')
train_copy.set_value(829,'Embarked' ,'C')


# In[ ]:


train_copy.info()


# ## Feature Engineering <a class="anchor" id="features"></a>

# Now that we have cleaned our data we can now move on to creating new features for our dataset. The first two new features, from the insight we gained from various visualizations, were family size and 

# ### Embarkment One-Hot encoding

# Let's start off by representing the `Embarked` column as integers (only 1 and 0 using one-hot encoding).

# In[ ]:


encoded_embark_train = pd.get_dummies(train_copy["Embarked"])
encoded_embark_test = pd.get_dummies(test_copy["Embarked"])

train_copy = train_copy.join(encoded_embark_train)
test_copy = test_copy.join(encoded_embark_test)

train_copy = train_copy.drop(["Embarked"], axis=1)
test_copy = test_copy.drop(["Embarked"], axis=1)


# ### Gender One-Hot encoding

# In[ ]:


encoded_gender_train = pd.get_dummies(train_copy["Sex"])
encoded_gender_test = pd.get_dummies(test_copy["Sex"])

train_copy = train_copy.join(encoded_gender_train)
test_copy = test_copy.join(encoded_gender_test)


# ### Family Size <a class="anchor" id="size"></a>

# In[ ]:


# Passenger + SibSP + Parch
train_copy["Family Size"] = 1 + train_copy["SibSp"] + train_copy["Parch"]
test_copy["Family Size"] = 1 + test_copy["SibSp"] + test_copy["Parch"]


# In[ ]:


a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
with sns.axes_style(style=None):
    sns.violinplot("Family Size", "Age", hue="Survived", data=train_copy,
                      split=True, inner="quartile",
                      palette=["#E4421A", "#06EFA0"]);


# Here's the insight we gain from the violin plot above:
# 
# 1. If a passenger had a family size greater than 7 they did not survive.
# 2. The majority of survivors had family sizes greater than or equal to 2 as well as those who survived on their own (family size of 1).
# 

# ### Title <a class="anchor" id="title"></a>

# In[ ]:


train_copy["Title"] = saved_titles_train
test_copy["Title"] = saved_titles_test


# In[ ]:


encoded_titles_train = pd.get_dummies(train_copy["Title"])
encoded_titles_test = pd.get_dummies(test_copy["Title"])

encoded_titles_train.head()


# In[ ]:


train_copy = train_copy.join(encoded_titles_train)
test_copy = test_copy.join(encoded_titles_test)


# In[ ]:


saved_train_titles = train_copy["Title"]
saved_test_titles = test_copy["Title"]

train_copy = train_copy.drop(["Title"], axis=1)
test_copy = test_copy.drop(["Title"], axis=1)


# In[ ]:


train_copy.head()


# In[ ]:


import seaborn as sns
corr = train_copy.corr()
colormap = sns.cubehelix_palette(light=1, as_cmap=True)
a4_dims = (10, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(corr ,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# We see here that the titles `Miss` and `Mrs` are highly correlated with survival compared to the other titles. What were their **classes**?

# In[ ]:


# Bring back original Title column
train_copy["Title"] = saved_titles_train
test_copy["Title"] = saved_titles_test

# Miss
train_copy[train_copy["Title"] == 'Miss']["Pclass"].value_counts()


# In[ ]:


# Mrs 
train_copy[train_copy["Title"] == 'Mrs']["Pclass"].value_counts()


# In[ ]:


train_copy[train_copy["Title"] == 'Mrs']["Family Size"].value_counts()


# In[ ]:


train_copy[train_copy["Title"] == 'Miss']["Family Size"].value_counts()


# ## Family Members <a class="anchor" id="members"></a>

# We saw earlier that titles such as Mrs and Miss are highly correlated with survival. I'm thinking that we need a feature where we tag individuals who are associated with other individuals who have the title Miss or Mrs. Let's create 3 categories. **Mother**, **Child**, and **Is Child of Mother with Title Mrs or Miss**.

# In[ ]:


def is_child(age):
    if age < 18:
        return 1
    else:
        return 0 
    
train_copy["Child"] = train_copy["Age"].transform(is_child)
test_copy["Child"] = test_copy["Age"].transform(is_child)

train_copy["Mother"] = 0
test_copy["Mother"] = 0

train_copy.loc[(train_copy["Sex"] == 'female') & (train_copy['Age'] > 18), 'Mother'] = 1
test_copy.loc[(train_copy["Sex"] == 'female') & (train_copy['Age'] > 18), 'Mother'] = 1


# In[ ]:


duplicate_tickets_train = train_copy[train_copy.duplicated("Ticket")]
tickets_of_mothers_with_duplicate_ticket_train = list(duplicate_tickets_train [(duplicate_tickets_train["Mother"] == 1) & ( (duplicate_tickets_train["Miss"] == 1) | (duplicate_tickets_train["Mrs"] == 1) )]["Ticket"])

train_copy["Child of Mrs or Miss"] = 0 
train_copy.loc[(train_copy["Child"] == 1) & (train_copy["Ticket"].isin(tickets_of_mothers_with_duplicate_ticket_train)), "Child of Mrs or Miss"] = 1  


duplicate_tickets_test = test_copy[test_copy.duplicated("Ticket")]
tickets_of_mothers_with_duplicate_ticket_test = list(duplicate_tickets_test [(duplicate_tickets_train["Mother"] == 1) & ( (duplicate_tickets_test["Miss"] == 1) | (duplicate_tickets_test["Mrs"] == 1) )]["Ticket"])

test_copy["Child of Mrs or Miss"] = 0 
test_copy.loc[(test_copy["Child"] == 1) & (test_copy["Ticket"].isin(tickets_of_mothers_with_duplicate_ticket_test)), "Child of Mrs or Miss"] = 1    


# In[ ]:


train_copy = train_copy.drop(["Title"], axis=1)
test_copy = test_copy.drop(["Title"], axis=1)


# In[ ]:


import seaborn as sns
corr = train_copy.corr()
colormap = sns.cubehelix_palette(light=1, as_cmap=True)
a4_dims = (10, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(corr ,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# Ok, the `Child of Mrs or Miss` didn't work out as well as I thought it would, but the `Mother` column is highly correlated with survival. What about single women?

# ## Single Women <a class="anchor" id="single"></a>

# In[ ]:


train_copy["Single Women"] = 0 
train_copy.loc[(train_copy["Mother"] == 1) & (train_copy["SibSp"] == 0) & (train_copy["Parch"] == 0), "Single Women"] = 1

test_copy["Single Women"] = 0 
test_copy.loc[(test_copy["Mother"] == 1) & (test_copy["SibSp"] == 0) & (test_copy["Parch"] == 0), "Single Women"] = 1


# In[ ]:


import seaborn as sns
corr = train_copy.corr()
colormap = sns.cubehelix_palette(light=1, as_cmap=True)
a4_dims = (10, 10)
fig, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(corr ,linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)


# It seems that the `Single Women` column is highly correlated with survival. 

# ## Machine Learning <a class="anchor" id="model"></a>

# In[ ]:


# Drop Uneccessary Columns 
train_copy.columns


# In[ ]:


train_copy = train_copy.drop(["Ticket", "age_dec", "Sex"], axis=1)
test_copy = test_copy.drop(["PassengerId", "Sex"], axis=1)


# In[ ]:


train_copy.columns


# In[ ]:


test_copy.columns


# In[ ]:


# Setup Training Data 
train_X = train_copy[['Pclass','male', 'female', 'Age', 'SibSp', 'Parch', 'Fare','C', 'Q', 'S', 'Family Size', 'Miss', 'Mr', 'Mrs', 'Officer', 'Other', 'Child', 'Mother', 'Child of Mrs or Miss', 'Single Women']]
test_X = test_copy[['Pclass','male', 'female', 'Age', 'SibSp', 'Parch', 'Fare','C', 'Q', 'S', 'Family Size', 'Miss', 'Mr', 'Mrs', 'Officer', 'Other', 'Child', 'Mother', 'Child of Mrs or Miss', 'Single Women']]

# Setup Target Data 
train_Y = train_copy[['Survived']]


# In[ ]:


# One 'Fare' value is NaN, let's fix that.
test_X["Fare"] = test_X["Fare"].fillna(test_X["Fare"].mean())


# ### Logistic Regression Model <a class="anchor" id="logistic"></a>

# In[ ]:


from sklearn.linear_model import LogisticRegression 
from sklearn import metrics
# Logistic Regression
model = LogisticRegression()
model.fit(train_X,train_Y)
prediction=model.predict(test_X)


# In[ ]:


prediction


# In[ ]:




