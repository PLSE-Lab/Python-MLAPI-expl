#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import math as math
# Checking for kaggle/input/titanic/****.csv
import os
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv");
train_data.name = "Training Set"
test_data.name = "Test Set"
train_data.sample(10)


# In[ ]:


# Column datatypes
train_data.info()
print('+'*40)
test_data.info()


# In[ ]:


# Columns
columns = train_data.columns
print(columns)


# In[ ]:


def concat_df(train_data, test_data):
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)

def divide_df(data):
    return data.loc[:890], data.loc[891:].drop(['Survived'], axis=1)


# In[ ]:


# Combine dataset for exploratory analytics
df = concat_df(train_data, test_data)
df.name = "Total Data"
df.info()


# # Exploratory Data analytics

# ## Understanding Features
# 
# ### Data Types
# - Survived, Sex: Binary
# - Pclass: Ordinal (categories with an order included)
# - Embark: Nominal (categorical)
# - Name, Cabin, Ticket: Text (string)
# - Age, Fare: Continuous numerical data (measure)
# - SibSp, Parch (Siblings, Spouces & Parents, Childrens): Discrete Numerical (count)

# In[ ]:


embarked_labels={"S": "Southampton", "C": "Chernboug", "Q": "Queenstown"}


# In[ ]:


# Change Pclass, Sex, Embarked to type category
for col in ['Pclass', 'Sex', 'Embarked']:
    train_data[col] = train_data[col].astype('category')
    test_data[col] = test_data[col].astype('category')
train_data.info()
print("-"*40)
test_data.info()


# In[ ]:


# Check for null count
print("Check for nulls in train data: ")
print(train_data.isnull().sum())
print("+"*40)
print("Check for nulls in test data: ")
print(test_data.isnull().sum())


# Records with unfilled data is not good for modelling. We should address these records one way or the other to get better results. There are lot of ways to achive it, we will revisit this point again.
# 
# __Age, Cabin, Embarked, Fare__ have N/A values

# In[ ]:


# Describe gives a very quick brief 5-point summary of the data. Take a peek look into numerical data.
train_data.describe(include="all").transpose()


# "Describe" on dataset gives a lot of info. Take time to understand what it is trying to tell you. Here is what I understood.
# 
# - **count** of all features is not same, there are some missing values
# - **top & frequency** are not applicable for numerical data and is applicable for only categorical data.
#     - __Pclass__ has 3 unique apperances __*'3'*__ being appeared most number of times (491). So 3rd class passengers travelling in titanic are more. Is it also possible that most of them not survived? Let's see further down what data tells?
#     - __Sex__ has 2 unique apperances __*'male'*__ being appeared most number of times (577). Again male passengers are more compared to female passengers. Is it also possible that most of them not survived? Let's see further down what data tells?
#     - __Embarked__ has 3 unique apperances __*'S'*__ being appeared most number of times (644). So most people who boarded the ship is from a station called **Southampton**
# - Age (numerical): mean > median, so the values are right skewed. min & max values having so much differences, so there can be potential outliers.
# - Fare (numerical): mean > median, so the values are right skewed. There is huge difference b/w min and max fares. This also tells there might be a lot of outliers for Fare. We can also link fare difference to passenger class and they are correlated..

# In[ ]:


# changing back to original data types.
train_data["Pclass"] = train_data["Pclass"].astype("int")
train_data["Sex"] = train_data["Sex"].astype("object")
train_data["Embarked"] = train_data["Embarked"].astype("object")
test_data["Pclass"] = test_data["Pclass"].astype("int")
test_data["Sex"] = test_data["Sex"].astype("object")
test_data["Embarked"] = test_data["Embarked"].astype("object")
train_data.info()
print("+"*40)
test_data.info()


# In[ ]:


_ = len(train_data[train_data['Survived'] == 1])/len(train_data) * 100
print('Survived %: ', _)


# ## Plots
# Plots are the most important aspect of EDA. This is such a powerful tool for drawing metrics from data and visually represent the facts. I'm going to spend a lot of time here to put in as many plots as possible just to get a feel of it and ofcourse to have a future reference.
# 
# I'm going to use Seaborn for plotting. It has wide variety of plots, will go through one by one and cover as many as possible.
# 
# - BarPlots
# - BoxPlots
# - DistPlots
# - FacetGrid
# - Factorplot
# - Catplots

# ### Bar plots
# 
# Barcharts are used to represent the frequency counts of a feature (categorical) or probability distribution if we choose to use y as another feature which is like yes/no. We can also use barplots for discrete numerical features as well like (SibSp, Parch) in our case.
# Example: we can choose bar plots to show total count on y-axis or show the probability of survivers in each nominal variable. Let's see both of them.
# 
# 1. To display total counts in a categorical variable (x=indexes, y=count) => count plot (x=variable, data)
# 2. Sum/Average of numerical variable against categorical variable (x=Pclass, y=Age)
# 3. Binary (true) against categorical variable (x=Pclass, y=Survived)
# 4. Group by one categorical variable (x=Pclass, y=Survived, hue=Sex) This is like percentage of passengers survived in each class by Sex. 
# 
# > Note: Use bar plots to show if there are less categories. having more categories will make the graph conjusted.
# 
# > Note: One of the axis should be a number either x or y doesn't matter

# #### <u>Using barplot for counts:</u>

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.color_palette(flatui)
# Categorical plots (Bar)
fig, ax = plt.subplots(2, 3, figsize=(20,15))
ax = ax.flatten()
unique_sibsp = train_data["SibSp"].unique()
unique_sibsp.sort()
unique_parch = train_data["Parch"].unique()
unique_parch.sort()

cols = [
    {"col": "Pclass", "x_labels": ["First", "Second", "Thrid"], "title": "Class of Passenger"},
    {"col": "Embarked", "x_labels": ["Chernboug", "Queenstown", "Southampton"], "title": "Where Passenger Boarded?"},
    {"col": "Survived", "x_labels": ["Not Survived", "Survived"], "title": "Who many passengers survived?"},
    {"col": "Sex", "x_labels": ["Male", "Female"], "title": "Sex of person"},
    {"col": "SibSp", "x_labels": unique_sibsp, "title": "Siblings & Spouses count"},
    {"col": "Parch", "x_labels": unique_parch, "title": "Parents & Children count"}
]
for i in range(0, 6):
    _ = train_data[cols[i]['col']].value_counts()
    _ = sns.barplot(x=_.index, y=_, ax=ax[i]) # returns ax of matplotlib
    _.set_xticklabels(cols[i]['x_labels'])
    _.set_ylabel('Count')
    _.set_title(cols[i]['title'])
    for patch in _.patches:
        label_x = patch.get_x() + patch.get_width()/2 # Mid point in x
        label_y = patch.get_y() + patch.get_height() + 10 # Mid point in y
        _.text(label_x,
               label_y,
               "{} ({:.1%})".format(int(patch.get_height()), patch.get_height()/len(train_data[cols[i]['col']])),
               horizontalalignment='center',
               verticalalignment='center')


# <u>**Observations**:</u>
# - ~55% of passengers on board are from 3rd class.
# - ~72% of passengers onborded from Southampton
# - ~38% of people survived in accident
# - ~65% of passengers are male
# - ~92% of passengers are either single or with one sibling/spouse.
# - ~89% of passengers are either single or with one parent/child.
# 
# These kind of conclusions will help us to better understand data, whether they are related or not and also might help us to impute the data when necessary

# #### <u>Using barplot for average age in each category:</u>

# In[ ]:


fig, ax = plt.subplots(2, 3, figsize=(20,15))
ax = ax.flatten()

cols = [
    {"col": "Pclass", "x_labels": ["First", "Second", "Thrid"], "title": "Average age of passengers in each class"},
    {"col": "Embarked", "x_labels": ["Chernboug", "Queenstown", "Southampton"], "title": "Average age of passengers based on boarding location"},
    {"col": "Sex", "x_labels": ["Male", "Female"], "title": "Average age of person based on gender"},
    {"col": "SibSp", "x_labels": unique_sibsp, "title": "Average age of Siblings & Spouses"},
    {"col": "Parch", "x_labels": unique_parch, "title": "Average age of Parents & Children"}
]

for i in range(0, 5):
    ax_ = sns.barplot(x=cols[i]["col"], y="Age", ax=ax[i], data=train_data) # returns ax of matplotlib
    ax_.set_xticklabels(cols[i]["x_labels"])
    ax_.set_ylabel("Average Age")
    ax_.set_title(cols[i]["title"])
    for patch in ax_.patches:
        ax_.text(
            patch.get_x() + patch.get_width()/2,
            patch.get_y() + (0 if math.isnan(patch.get_height()) else patch.get_height()/2),
            "{:.2f}".format(patch.get_height()),
            horizontalalignment="center",
            verticalalignment="center")


# <u>**Observations:**</u>
# We are plotting age against each categorical variable. This might not a big detail, but it can used as a crutial element for filling in missing ages.
# 
# For example: if a person age is missing and he belongs to first class, we can fill person's age with mean of that group rather than mean of entire set.

# #### <u>Using barplot to show % of survivers in each category</u>

# In[ ]:


fig, ax = plt.subplots(2, 3, figsize=(20,15))
ax = ax.flatten()

cols = [
    {"col": "Pclass", "x_labels": ["First", "Second", "Thrid"], "title": "% of survivors in each passenger class"},
    {"col": "Embarked", "x_labels": ["Chernboug", "Queenstown", "Southampton"], "title": "% of survivors based on boarding location"},
    {"col": "Sex", "x_labels": ["Male", "Female"], "title": "% of survivors based on gender"},
    {"col": "SibSp", "x_labels": unique_sibsp, "title": "% of survivors in siblings & spouses"},
    {"col": "Parch", "x_labels": unique_parch, "title": "% of surviviors in parent & children"}
]

for i in range(0, 5):
    ax_ = sns.barplot(x=cols[i]["col"], y="Survived", ax=ax[i], data=train_data) # returns ax of matplotlib
    ax_.set_xticklabels(cols[i]["x_labels"])
    ax_.set_ylabel("Average Age")
    ax_.set_title(cols[i]["title"])
    for patch in ax_.patches:
        ax_.text(
            patch.get_x() + patch.get_width()/2,
            patch.get_y() + (0 if math.isnan(patch.get_height()) else patch.get_height()/2),
            "{:.2f}%".format(patch.get_height() * 100),
            horizontalalignment="center",
            verticalalignment="center")


# <u>**Observations:**</u>
# This is a much better example of bar plot. This plot is a special plot which shows the % of people survived based on which category they belong to. There are two important take aways.
# - People belonging to 1st class have high chance of surviving where as if you are thrid class person there is less chance of being saved.
# - Women have a highest chance of surviving the incident compared to men.
# 
# This is a very important observation because these two features have a significat impact on whether a person is saved or not.
# 
# <u>For Example:</u> A women who belongs to a first class has more chance of survival than anyone else on the board.

# #### <u>Using barplot to show % of survivors in each category grouped by gender:</u>

# In[ ]:


fig, ax = plt.subplots(2, 2, figsize=(20,15))
ax = ax.flatten()

cols = [
    {"col": "Pclass", "x_labels": ["First", "Second", "Thrid"], "title": "% of survivors grouped by gender in each passenger class"},
    {"col": "Embarked", "x_labels": ["Chernboug", "Queenstown", "Southampton"], "title": "% of survivors grouped by gender in each boarding class"},
    {"col": "SibSp", "x_labels": unique_sibsp, "title": "% of survivors grouped by gender in siblings and spouses"},
    {"col": "Parch", "x_labels": unique_parch, "title": "% of survivors grouped by gender in parents and children"}
]

for i in range(0, 4):
    ax_ = sns.barplot(x=cols[i]["col"], y="Survived", hue="Sex", ax=ax[i], data=train_data) # returns ax of matplotlib
    ax_.set_xticklabels(cols[i]["x_labels"])
    ax_.set_ylabel("Average Age")
    ax_.set_title(cols[i]["title"])
    for patch in ax_.patches:
        ax_.text(
            patch.get_x() + patch.get_width()/2,
            patch.get_y() + (0 if math.isnan(patch.get_height()) else patch.get_height()/2),
            "{:.2f}%".format(patch.get_height() * 100),
            horizontalalignment="center",
            verticalalignment="center")


# <u>**Observations:**</u>
# 
# As we expected this gives even a much better insight on survival rate.
# - a woophing ~97% & ~92% of women from 1st & 2nd class passengers are saved and ~50% of women from 3rd class passengers are saved.
# - a female passenger boarded from queens have 88% of surviving rate.

# ### Distribution Plots
# Distribution plots or histograms are used to see the spread of data. These plots are used for numerical data. data is said to be perfect when it represents a symmetrical bell curve, which is not the case in most of the practical senarios. Age, Fare are numerical features in our dataset. Let's plot them
# 
# > Distribution plots shouldn't have any null values while plotting them, so remove them before plotting
# 
# <p style="color:red;text-decoration:underline;font-weight:600">This graph requires a revist once data is imputated (means fill the null values)</p>

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(20, 8))
ax = ax.flatten()
cols = [
    {"col": "Age", "x_label": "Age", "title": "Age Distribution", "color": "green"},
    {"col": "Fare", "x_label": "Fare", "title": "Fare Distribution", "color": "violet"}
]
for i in range(0, len(cols)):
    _col = cols[i]["col"]
    data_ = train_data[_col][pd.notnull(train_data[_col])]
    _ = sns.distplot(data_, kde=True, hist=False, ax=ax[i], color=cols[i]["color"])
    _.set_title(cols[i]["title"])
    _.set_xlabel(cols[i]["x_label"])
    mean_ = data_.mean()
    median_ = data_.median()
    _.axvline(mean_, linestyle="--", color="red")
    _.axvline(median_, linestyle="--", color="orange")
    _.legend({"Mean": mean_, "Median": median_})


# <u>**Observations:**</u>
# - Age & Fare both are right skewed meaning __mean > median__ & might have potential outliers in the data. Note this is not a complete distribution of the set. Have to fill in the missed values and revisit the distribution again for more insights.
# - Most of the fare distribution lies within the area indicating outliers.

# ### Box Plots
# Box plots are used to check the 5 point summary data and any outliers in the data.

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax = ax.flatten()
cols = [
    {"col": "Age", "x_label": "Age", "title": "Age Distribution"},
    {"col": "Fare", "x_label": "Fare", "title": "Fare Distribution"}
]
for i in range(0, len(cols)):
    _col = cols[i]["col"]
    data_ = train_data[_col][pd.notnull(train_data[_col])]
    _ = sns.boxplot(data_, ax=ax[i])
    _.set_title(cols[i]["title"])
    _.set_xlabel(cols[i]["x_label"])


# **Observations:**
# - There are lot of outliers in both age and fare distribution
# - These outliers will have lot of impact on models. This is more visible for regression models. We have to address these outliers while preparing the model.

#  <u>**Using box plots with category**</u>

# In[ ]:


fig, ax = plt.subplots(3, 2, figsize=(20, 15))
ax = ax.flatten()
cols = [
    {'x': 'Pclass', 'y': 'Age', 'x_label': 'Pclass', 'title': 'Age Distribution based on Pclass'},
    {'x': 'Pclass', 'y': 'Fare', 'x_label': 'Pclass', 'title': 'Fare Distribution based on Pclass'},
    {'x': 'Embarked', 'y': 'Age', 'x_label': 'Embarked', 'title': 'Age Distribution based on Embarked'},
    {'x': 'Embarked', 'y': 'Fare', 'x_label': 'Embarked', 'title': 'Fare Distribution based on Embarked'},
    {'x': 'Sex', 'y': 'Age', 'x_label': 'Sex', 'title': 'Age Distribution based on Sex'},
    {'x': 'Sex', 'y': 'Fare', 'x_label': 'Sex', 'title': 'Fare Distribution based on Sex'}
]
for i in range(0, len(cols)):
    data_ = train_data[train_data[cols[i]['y']].notnull()]
    _ = sns.boxplot(x=cols[i]['x'], y=cols[i]['y'], data=data_, ax=ax[i])
    _.set_title(cols[i]['title'])
    _.set_xlabel(cols[i]['x_label'])


# <u>**Observations:**</u>
# - there are no outliers in first class passgeners with more symmetrical distribution, where as there are too many outliers for 2nd and thrid class passengers
# - Fare is very sparsly distributed among 2nd and 3rd class passengers.

# <u>**Using box plot to group distribution based on category and then by another category as hue (Survived)**</u>

# In[ ]:


fig, ax = plt.subplots(3, 2, figsize=(20, 15))
ax = ax.flatten()
cols = [
    {'x': 'Pclass', 'y': 'Age', 'x_label': 'Pclass', 'title': 'Age Distribution based on Pclass'},
    {'x': 'Pclass', 'y': 'Fare', 'x_label': 'Pclass', 'title': 'Fare Distribution based on Pclass'},
    {'x': 'Embarked', 'y': 'Age', 'x_label': 'Embarked', 'title': 'Age Distribution based on Embarked'},
    {'x': 'Embarked', 'y': 'Fare', 'x_label': 'Embarked', 'title': 'Fare Distribution based on Embarked'},
    {'x': 'Sex', 'y': 'Age', 'x_label': 'Sex', 'title': 'Age Distribution based on Sex'},
    {'x': 'Sex', 'y': 'Fare', 'x_label': 'Sex', 'title': 'Fare Distribution based on Sex'}
]
for i in range(0, len(cols)):
    data_ = train_data[train_data[cols[i]['y']].notnull()]
    _ = sns.boxplot(x=cols[i]['x'], y=cols[i]['y'], hue='Survived', data=data_, ax=ax[i])
    _.set_title(cols[i]['title'])
    _.set_xlabel(cols[i]['x_label'])


# ### FacetGrid
# - Facet grid is very helpful in visualzing multiple features in a single graph
# We can use row, col, hue to represent three different features and facet grid object will help you to map a plot against those selected features, this mapper can be a plot of any kind like kde, dist, scatter etc..

# In[ ]:


ax = sns.FacetGrid(train_data, col="Sex", row="Pclass", legend_out=True)
ax.map(plt.hist, "Age", color="blue")


# In[ ]:


ax = sns.FacetGrid(train_data, col="Sex", row="Pclass", hue="Survived", legend_out=True)
ax.map(sns.kdeplot, "Age")


# In[ ]:


ax = sns.FacetGrid(train_data, col="Pclass", hue="Survived")
ax.map(sns.scatterplot, "Age", "Fare")


# ### Factorplot or Catplot
# - Factor plot returns a facetgrid, it can be treated as a wrapper around facet grid.
# - Catplot returns a facetgrid and is a wrapper around facet grid
# 
# Both these graphs are used to visualize the data of more than 2 features. There are x, y, hue, col, row - These many data points to visualize the data.

# <u>**Factor Plots:**<u>

# In[ ]:


# ax = sns.FacetGrid(train_data, col="Sex", row="Pclass", hue="Survived", legend_out=True)
# ax.map(sns.kdeplot, "Age")
sns.factorplot(x="Sex", y="Survived", col="Pclass", data=train_data, saturation=.5, kind="bar")


# In[ ]:


_ = sns.factorplot(x='Pclass', y='Survived', hue='Sex', data=train_data, kind='bar')
for patch in _.ax.patches:
    label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle
    label_y = patch.get_y() + patch.get_height()/2
    _.ax.text(label_x,
              label_y,
              '{:.3%}'.format(patch.get_height()),
              horizontalalignment='center',
              verticalalignment='center')


# In[ ]:


_ = sns.factorplot(x='Embarked', y='Survived', hue='Sex', data=train_data, kind='bar')
for patch in _.ax.patches:
    label_x = patch.get_x() + patch.get_width()/2  # find midpoint of rectangle
    label_y = patch.get_y() + patch.get_height()/2
    _.ax.text(label_x,
              label_y,
              '{:.3%}'.format(patch.get_height()),
              horizontalalignment='center',
              verticalalignment='center')


# <u>**Catplots**</u>

# In[ ]:


sns.catplot(data=train_data, x="Sex", y="Age", col="Pclass", hue="Survived", kind="swarm")


# In[ ]:


_ = sns.catplot(x="Sex", y="Survived", col="Pclass", data=train_data, kind='bar')
_.set_axis_labels("", "Survival Rate")
_.set_titles("{col_name}-{col_var}")
for ax in _.axes[0]:
    for patch in ax.patches:
        label_x = patch.get_x() + patch.get_width()/2
        label_y = patch.get_y() + patch.get_height()/2
        ax.text(label_x,
                label_y,
                "{0:.2f}%".format(patch.get_height()*100),
                horizontalalignment='center',
                verticalalignment='center')


# In[ ]:


_ = sns.catplot(x="Sex", y="Survived", col="Embarked", data=train_data, kind='bar')
_.set_axis_labels("", "Survival Rate")
_.set_titles("{col_name}")
for ax in _.axes[0]:
    for patch in ax.patches:
        label_x = patch.get_x() + patch.get_width()/2
        label_y = patch.get_y() + patch.get_height()/2
        ax.text(label_x,
                label_y,
                "{0:.2f}%".format(patch.get_height()*100),
                horizontalalignment='center',
                verticalalignment='center')


# In[ ]:


sns.catplot(x="Parch", y="Survived", col="Sex", data=train_data, kind="bar")


# ### Count Plots
# countplots are used to plot the feature counts if mixed with hue will give more insight on data

# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(20, 6))
ax = ax.flatten()
cols = [
    {"col": "Pclass", "title": "Passenger Class", "x_labels": ["1", "2", "3"]},
    {"col": "Embarked", "title": "Boarded from", "x_labels": ["S", "Q", "C"]},
    {"col": "Sex", "title": "Gender", "x_labels": ["M", "F"]}
]
for i in range(0, len(cols)):
    _ = sns.countplot(x=cols[i]['col'], data=train_data, hue="Survived", ax=ax[i])
    _.set_title(cols[i]['title'])
    _.set_xticklabels(cols[i]['x_labels'])
    for patch in _.patches:
        x_label = patch.get_x() + patch.get_width()/2
        y_label = patch.get_y() + patch.get_height() + 7
        
        _.text(x_label,
               y_label,
               "{0}".format(patch.get_height()),
               horizontalalignment='center',
               verticalalignment='center')


# In[ ]:


sns.countplot(x="Parch", hue="Sex", data=train_data)


# <u>**Crosstab data for categorical variables**</u>
# Crosstab is used to get the count of records against categorical variables like `class vs survived` etc..

# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(20, 5))
ax = ax.flatten()
pd.crosstab(train_data['Survived'], train_data['Pclass']).plot.bar(stacked=True, ax=ax[0])
pd.crosstab(train_data['Survived'], train_data['Embarked']).plot.bar(stacked=True, ax=ax[1])
pd.crosstab(train_data['Survived'], train_data['Sex']).plot.bar(stacked=True, ax=ax[2])


# ### Heatmap
# Heatmap plot is very useful in understanding how features interact with each other and how strongly they are correlated.
# - rA,B -> 1; then they are positively correlated
# - rA,B -> -1; then they are negatively correlated.

# ### Overlapped Distributions

# In[ ]:


fig, ax = plt.subplots(figsize=(10, 5))
sns.kdeplot(
    data=train_data['Age'][(train_data['Survived'] == 0) & (train_data['Age'].notnull())],
    ax=ax,
    color='Red',
    shade=True)
sns.kdeplot(
    data=train_data['Age'][(train_data['Survived'] == 1) & (train_data['Age'].notnull())],
    ax=ax,
    color='Blue',
    shade=True)
ax.legend(["Not Survived", "Survived"])
ax.set_title("Superimposed KDE plot for age of Survived and Not Survived")


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 5))
sns.kdeplot(data=train_data.loc[train_data['Survived'] == 0, 'Fare'], color='Red', shade=True, legend=True)
sns.kdeplot(data=train_data.loc[train_data['Survived'] == 1, 'Fare'], color='Blue', shade=True, legend=True)
ax.legend(["Not Survived", "Survived"])
ax.set_title("Superimposed KDE plot for fare of Survived and Not Survived")


# ### Heat Map (Correleation Matrix)

# In[ ]:


corr = train_data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr, annot=True)


# In[ ]:


# Check for survival rate against each category
def survival_rate(data, column):
    categories_ = data[column].unique()
    categories_.sort()
    print('{0} based survival rate:'.format(column))
    print('+'*40)
    for cat in categories_:
        _ = data.loc[data[column] == cat]
        print('{0} - {1} survival rate: {2:.3f}'.format(column, cat, (_['Survived'].sum()/len(_)) * 100))


# ## Imputing missing values
# 
# "Feature Engineering" is the term used to preparing data for model. It involves the following
# - What to do with missing data
#     - Remove them
#     - Impute them (If replaced with new values, what kind of values to consider?)
# - Is all features makes sense for model?
# - Extracting Features to reduce correlation b/w features

# ### Dealing with missing values

# In[ ]:


for d_ in [train_data, test_data]:
    print(d_.name)
    print("+"*40)
    for col in d_.columns:
        missing_ = d_[col].isnull().sum()
        if missing_ > 0:
            print("'{0}' column missing value count {1}({2:.2f}%)".format(col, missing_, missing_/len(d_)*100))
    print("+"*40)


# <u>**Observations**</u>
# 
# Age, Embarked, Cabin and Fare are the columns that have missing values. `Embarked` and `Fare` have relatively very small number of missing values, whereas `Age` has moderately missing values and `Cabin` has almost 80% of missing values.
# 
# There are lot of kernals which explain lot of interesting things on which we can fill the missing values. I have gone through couple of kernels and I'm following a kernal from [gunesevitan](https://www.kaggle.com/gunesevitan/advanced-feature-engineering-tutorial-with-titanic)
# 
# Missing values for Age, Embarked, Fare can be filled with descriptive statistical measures like mean, median, grouping etc... whereas for Cabin the same approach is not applicable.

# <u>**Dealing with Age**</u> Age is a numerical number so we can simply use mean/median to replace the age. However if you have observed the above descriptive statistics. Age can be mean/median of class wise passenger age or can be grouped by Sex and then fill them.

# In[ ]:


# Correlation Coefficents
_ = df.corr().abs().unstack().sort_values().reset_index().rename(
    columns={"level_0": "F1", "level_1": "F2", 0: "Coef_"})
_ = _[_["F1"] == "Age"]
print(_)


# In[ ]:


# _ = df.groupby(["Sex", "Pclass"]).describe()
_ = df.groupby(["Sex", "Pclass", "SibSp"]).describe()
_ = _["Age"].loc[:, ["mean", "50%"]]
print(_)
# When grouped by sex and by pclass mean and median are most consistent and they values are pretty much closer.
# So replacing the values based on these stats should be easy.


# In[ ]:


df["Age"] = df.groupby(["Sex", "Pclass"])["Age"].apply(lambda x: x.fillna(x.median()))


# In[ ]:


# Comparing Age distributions before and after imputing the values.
df_age_old = concat_df(train_data, test_data)
df_age_old = df_age_old["Age"][pd.notnull(df_age_old["Age"])]
plt.figure(figsize=(10, 5))
sns.kdeplot(df_age_old, color="red", shade=True)
sns.kdeplot(df["Age"], color="green", shade=True)
# Data distribution remains almost same before and after imputing values.


# <u>**Dealing with missing Embarked values: **</u>

# In[ ]:


df[df['Embarked'].isnull()]


# - only two records are missing for Embarked values
# - Interestingly see the values of Cabin, Fare, Ticket, Sex, Passenger class..
# - Both of them might have boarded from the same station, so how do we know from which station?
# 
# According to @gunesevitan which I liked very much is going beyond what was present inside data. He googled the actual name, meaning he looked for the facts beyond data and found that the passengers are boarded from Southampton, without any hesitation he filled the values and said <i>__"Case Closed !!"__</i>. wow !!
# 
# Let's do the same..

# In[ ]:


df["Embarked"] = df["Embarked"].fillna("S")


# <u>__Dealing with missing fare values:__</u>
# 
# - Only one person has missing fare value
# - Fare value is highly correlated to the class of person and also with how many people he/she trave

# In[ ]:


df[df["Fare"].isnull()]
_ = df.groupby(["Pclass", "Parch", "SibSp"]).describe()
_ = _["Fare"].loc[:, ["mean", "50%"]]
print(_)
df["Fare"] = df.groupby(["Pclass", "Parch", "SibSp"])["Fare"].apply(lambda x: x.fillna(x.mean()))


# <u>**Dealing with missing deck values:**</u>
# 
# This is where I liked the author @gunesevitan. He went out of box to get for the facts. This is something to remember to become a good data scientist. Let me run through steps
# 
# - 

# In[ ]:


print("Passengers without deck: ", df["Cabin"].isnull().sum())
df["Deck"] = df["Cabin"].apply(lambda x: x[0] if pd.notnull(x) else "M")
df_decks = df.groupby(["Deck", "Pclass"]).count().drop(
    columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch','Fare', 'Embarked', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={"Name": "Count"}).transpose()
df_decks


# In[ ]:


deck_x_pclass = pd.crosstab(df["Deck"], df["Pclass"])
plt.title = "passengers in each deck"
# deck_x_pclass.plot.bar(stacked=True)
plt.figure(figsize=(20, 10))
# now stack and reset
stacked = deck_x_pclass.stack().reset_index().rename(columns={0:'value'})
stacked["percent"] = 0
plt.figure(figsize=(20, 8))
# plot grouped bar chart
sns.barplot(x="Deck", y="value", hue="Pclass", data=stacked)
# Plot percentage of passengers
for index in deck_x_pclass.index:
    total_ = deck_x_pclass.loc[index].sum()
    for pclass in deck_x_pclass.columns:
        val_ = stacked.loc[(stacked["Deck"] == index) & (stacked["Pclass"] == pclass), 'value']
        stacked.loc[(stacked["Deck"] == index) & (stacked["Pclass"] == pclass), 'percent'] = val_/total_*100


# In[ ]:


stacked[stacked["percent"] > 0]


# <u>**Observations:**</u>
# - Deck `A`, `B` & `C` are completely occupied by 1st class passengers
# - Deck `T` has only one passenger is occupied by 1st class
# - Deck `D` is occupied by both 1st(86.95%) and 2nd(13.04) class passengers
# - Deck `E` is occupied by all class passengers 1st(83%), 2nd(9.7%) & 3rd(7.31%)
# - Deck `F` is occupied by 2nd(62%) and 3rd(38%) class passengers
# - Deck `G` is occupied by 3rd class passengers
# - Deck `M` unknown has all kinds of passengers (Let this be that category only)

# In[ ]:


# Replace Deck T with A as it has only one passenger
_ = df[df["Deck"] == "T"].index
if (_.size > 0):
    df.loc[_, "Deck"] = "A"


# <u>**Passenger Survival Rate in each deck**</u>

# In[ ]:


df_survived = df.groupby(["Deck", "Survived"]).count().drop(
    columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']
    ).rename(columns={'Name':'Count'}).transpose()
print(df_survived)
surv_counts = {'A':{}, 'B':{}, 'C':{}, 'D':{}, 'E':{}, 'F':{}, 'G':{}, 'M':{}}
decks = df_survived.columns.levels[0]
for deck in decks:
    for survive in range(0, 2):
        surv_counts[deck][survive] = df_survived[deck][survive][0]
df_surv_counts = pd.DataFrame(surv_counts)
surv_percentages = {}
for col in df_surv_counts:
    surv_percentages[col] = [(count / df_surv_counts[col].sum()) * 100 for count in df_surv_counts[col]]
print(surv_percentages)
df_survived_percentages = pd.DataFrame(surv_percentages).transpose()
deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')
bar_count = np.arange(len(deck_names))  
bar_width = 0.85
not_survived = df_survived_percentages[0]
survived = df_survived_percentages[1]
plt.figure(figsize=(20, 10))
plt.bar(bar_count, not_survived, color='#b5ffb9', edgecolor='white', width=bar_width, label="Not Survived")
plt.bar(bar_count, survived, bottom=not_survived, color='#f9bc86', edgecolor='white', width=bar_width, label="Survived")
plt.xlabel('Deck', size=15, labelpad=20)
plt.ylabel('Survival Percentage', size=15, labelpad=20)
plt.xticks(bar_count, deck_names)    
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})
# plt.title('Survival Percentage in Decks')


# <u>**Obseravtions:**</u>
# - Deck `B, C, D & E` has highest survival
# - Deck `M` has lowest survival rate
# 
# Right now `Deck` has highest cardinality so we can reduce this cardinality by combining certain groups
# - Combine `A, B & C` decks as `ABC`
# - Combine `D, E` as `DE`
# - Combine `F, G` as `FG`

# In[ ]:


df["Deck"] = df["Deck"].replace(["A", "B", "C"], "ABC").replace(["D", "E"], "DE").replace(["F", "G"], "FG")
df["Deck"].value_counts()


# In[ ]:


df.drop(["Cabin"], inplace=True, axis=1)


# In[ ]:


train, test = divide_df(df)
print(train.info())
print("+"*40)
print(test.info())


# > # Feature Engineering

# ## Binning the continuous features

# In[ ]:


df["Fare"] = pd.qcut(df["Fare"], 13)


# In[ ]:


plt.figure(figsize=(20, 5))
sns.countplot(x="Fare", hue="Survived", data=df)
plt.xlabel('Fare')
plt.ylabel('Passenger Count')
plt.tick_params(axis='x')
plt.tick_params(axis='y')

plt.legend(['Not Survived', 'Survived'], loc='upper right')


# In[ ]:


df["Age"] = pd.qcut(df["Age"], 10)


# In[ ]:


plt.figure(figsize=(20, 5))
sns.countplot(x="Age", hue="Survived", data=df)
plt.xlabel('Age')
plt.ylabel('Passenger Count')
plt.tick_params(axis='x')
plt.tick_params(axis='y')

plt.legend(['Not Survived', 'Survived'], loc='upper right')


# ## Grouping Families

# In[ ]:


df["FamilySize"] = df["SibSp"] + df["Parch"] + 1


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(20, 5))
sns.countplot(x="FamilySize", data=df, ax=ax[0])
sns.countplot(x="FamilySize", hue="Survived", data=df, ax=ax[1])


# 

# In[ ]:


family_map = {1: 'Alone', 2: 'Small', 3: 'Small', 4: 'Small', 5: 'Medium', 6: 'Medium', 7: 'Large', 8: 'Large', 11: 'Large'}
df['FamilySizeGrouped'] = df['FamilySize'].map(family_map)


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(20, 5))
sns.countplot(x="FamilySizeGrouped", data=df, ax=ax[0])
sns.countplot(x="FamilySizeGrouped", hue="Survived", data=df, ax=ax[1])


# ## Ticket Frequency

# In[ ]:


df["TicketFrequency"] = df.groupby("Ticket")["Ticket"].transform("count")


# In[ ]:


fig, axs = plt.subplots(figsize=(12, 5))
sns.countplot(x='TicketFrequency', hue='Survived', data=df)

plt.xlabel('Ticket Frequency', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})


# ## Title Extraction

# In[ ]:


df["Title"] = df["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]


# In[ ]:


df["IsMarried"] = 0
df.loc[df["Title"] == 'Mrs', "IsMarried"] = 1


# In[ ]:


plt.figure(figsize=(20, 7))
ax_ = sns.countplot(x="Title", data=df)
for patch in ax_.patches:
    label_x = patch.get_x() + patch.get_width()/2
    label_y = patch.get_y() + patch.get_height()+ 10
    ax_.text(label_x, label_y, patch.get_height(), horizontalalignment='center', verticalalignment='center')
sns.countplot(x="Title", hue="Survived", data=df)


# In[ ]:


df['Title'] = df['Title'].replace(
    ['Miss', 'Mrs','Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df['Title'] = df['Title'].replace(
    ['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')


# In[ ]:


plt.figure(figsize=(20, 7))
ax_ = sns.countplot(x="Title", data=df)
for patch in ax_.patches:
    label_x = patch.get_x() + patch.get_width()/2
    label_y = patch.get_y() + patch.get_height()+ 10
    ax_.text(label_x, label_y, patch.get_height(), horizontalalignment='center', verticalalignment='center')
plt.figure(figsize=(20, 7))
sns.countplot(x="Title", hue="Survived", data=df)


# ## Feature Encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
train, test = divide_df(df)
dfs = [train, test]


# In[ ]:


non_numeric_features = ['Embarked', 'Sex', 'Deck', 'Title', 'FamilySizeGrouped', 'Age', 'Fare']
for df_ in dfs:
    for feature in non_numeric_features:        
        df_[feature] = LabelEncoder().fit_transform(df_[feature])


# In[ ]:


cat_features = ['Pclass', 'Sex', 'Deck', 'Embarked', 'Title', 'FamilySizeGrouped']
encoded_features = []

for df in dfs:
    for feature in cat_features:
        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
        n = df[feature].nunique()
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        encoded_df = pd.DataFrame(encoded_feat, columns=cols)
        encoded_df.index = df.index
        encoded_features.append(encoded_df)

train = pd.concat([train, *encoded_features[:6]], axis=1)
test = pd.concat([test, *encoded_features[6:]], axis=1)


# In[ ]:


drop_cols = ['Deck', 'Embarked', 'FamilySize', 'FamilySizeGrouped', 'Name', 'Parch', 'PassengerId', 'Pclass', 'Sex', 'SibSp', 'Ticket', 'Title', 'TicketFrequency', 'IsMarried']
train = train.drop(columns=drop_cols)
test = test.drop(columns=drop_cols)


# # Machine Learning

# In[ ]:


X_train = train.loc[:, train.columns != "Survived"]
y_train = train.loc[:, "Survived"]
X_test = test


# In[ ]:


from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
X_train_scaled = std_scaler.fit_transform(X_train)
X_test_scaled = std_scaler.fit_transform(X_test)
print(X_train.shape, X_test.shape)


# ## Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(criterion="gini",
                                  n_estimators=1750,
                                  max_depth=7,
                                  min_samples_split=6,
                                  min_samples_leaf=6,
                                  max_features="auto",
                                  oob_score=True,
                                  random_state=40,
                                  n_jobs=-1)


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
N = 5
oob = 0
prob_df = pd.DataFrame(np.zeros((len(X_test), N*2)))
prob_df.columns = ["KFold{}_{}".format(i, j) for i in range(1, N+1) for j in range(2)]
prob_df
imp_df = pd.DataFrame(np.zeros((X_train.shape[1], N)))
imp_df.columns = ["KFold{}".format(i) for i in range(1, N+1)]
imp_df.index = X_train.columns


# In[ ]:


fprs, tprs, scores = [], [], []
skf = StratifiedKFold(n_splits=N, random_state=N, shuffle=True)
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print("Fold {}".format(fold))
    print("+"*40)
    rfc_model.fit(X_train.loc[trn_idx], y_train.loc[trn_idx])
    predict_proba_ = rfc_model.predict_proba(X_train.loc[trn_idx])
    trn_fpr, trn_tpr, trn_thresholds = roc_curve(
        y_train[trn_idx],
        predict_proba_[:, 1])
    trn_auc_score = auc(trn_fpr, trn_tpr)
    predict_proba_val_ = rfc_model.predict_proba(X_train.loc[val_idx])
    val_fpr, val_tpr, val_thresholds = roc_curve(
        y_train.loc[val_idx],
        predict_proba_val_[:, 1])
    val_auc_score = auc(val_fpr, val_tpr)
    scores.append((trn_auc_score, val_auc_score))
    fprs.append(val_fpr)
    tprs.append(val_tpr)
    
    prob_df.loc[:, "KFold{}_0".format(fold)] = rfc_model.predict_proba(X_test)[:, 0]
    prob_df.loc[:, "KFold{}_1".format(fold)] = rfc_model.predict_proba(X_test)[:, 1]
    
    imp_df.iloc[:, fold - 1] = rfc_model.feature_importances_
    oob += rfc_model.oob_score_ / N
    print('Fold {} OOB Score: {}\n'.format(fold, rfc_model.oob_score_)) 

print('Average OOB Score: {}'.format(oob))


# In[ ]:


imp_df["Mean"] = imp_df.mean(axis = 1)


# In[ ]:


imp_df.sort_values(by='Mean', inplace=True, ascending=False)
plt.figure(figsize=(15, 10))
sns.barplot(x='Mean', y=imp_df.index, data=imp_df)

plt.xlabel('')
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)
# plt.title('Random Forest Classifier Mean Feature Importance Between Folds')


# In[ ]:


class_survived = [col for col in prob_df.columns if col.endswith('1')]
prob_df['1'] = prob_df[class_survived].sum(axis=1) / N
prob_df['0'] = prob_df.drop(columns=class_survived).sum(axis=1) / N
prob_df['pred'] = 0
pos = prob_df[prob_df['1'] >= 0.5].index
prob_df.loc[pos, 'pred'] = 1

y_pred = prob_df['pred'].astype(int)


# In[ ]:


submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission_df['PassengerId'] = test_data['PassengerId']
submission_df['Survived'] = y_pred.values
submission_df.to_csv('submissions.csv', header=True, index=False)
submission_df.head(10)


# [Download File](./submissions.csv)

# In[ ]:




