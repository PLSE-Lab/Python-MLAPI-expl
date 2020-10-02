#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

# Pandas
import pandas as pd
pd.set_option('display.max_colwidth', None)

# Plotly
import plotly.graph_objects as go

# Markdown print
from IPython.display import Markdown, display

# Sklearn split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
    
# Functions
def printmd(string):
    display(Markdown(string))

def mycatplot(data, name):
    fig = go.Figure()
    fig.add_traces([go.Bar(x=data.index, y=data.total, name="Total", visible=False),
                    go.Bar(x=data.index, y=data["<=50K"], name="< $50K"),
                    go.Bar(x=data.index, y=data[">50K"], name="> $50K"),
                    go.Bar(x=data.index, y=data.less_ratio, name="< $50K", visible=False),
                    go.Bar(x=data.index, y=1 - data.less_ratio, name="> $50K", visible=False)])
    fig.update_layout(title=name, xaxis_title=name, yaxis_title="Count", legend_title_text="Income", showlegend=True, updatemenus=[
        dict(type="buttons", direction="right", active=1, x=1, y=1.25, buttons=list([
            dict(label="Total", method="update", args=[{"visible": [True, False, False, False, False]}, {"barmode": "group"}]),
            dict(label="Per income", method="update", args=[{"visible": [False, True, True, False, False]}, {"barmode": "group"}]),
            dict(label="Per income ratio", method="update", args=[{"visible": [False, False, False, True, True]}, {"barmode": "stack"}]),
        ]))
    ])
    return fig

def mycatpivot(data, name):
    res = data.pivot_table(index=name, columns="income", values="age", aggfunc="count", fill_value=0)
    res["total"] = res["<=50K"] + res[">50K"]
    res["less_ratio"] = (res["<=50K"] / (res["<=50K"] + res[">50K"])).round(2)
    return res.sort_values("total")


# # Adult Census Income
# 
# ## Table of Contents
# 
# 1. [Description](#Description)
#     - [Features](#Features)
# 2. [Data Preparation](#Data-Preparation)
#     - [Missing values](#Missing-values)
#     - [Duplicated rows](#Duplicated-rows)
# 3. [Data Exploration](#Data-Exploration)
#     - [Income](#Income)
#     - [Age](#Age)
#     - [Workclass](#Workclass)
#     - [Occupation](#Occupation)
#     - [Workclass - Occupation](#Workclass---Occupation)
#     - [Education - Education number](#Education---Education-number)
#     - [Marital status](#Marital-status)
#     - [Relationship](#Relationship)
#     - [Marital status - Relationship](#Marital-status---Relationship)
#     - [Race](#Race)
#     - [Sex](#Sex)
#     - [Country](#Country)
#     - [Capital](#Capital)
#     - [Work hours](#Work-hours)
# 4. [Feature Engineering](#Feature-Engineering)
#     - [Missing values](#Missing-values-2)
#     - [Outliers](#Outliers)
#     - [Transformation](#Transformation)
#     - [Selection](#Selection)
# 5. [Datasets generation](#Datasets-generation)
#     - [Splitting](#Splitting)
#     - [Encoding](#Encoding)
#     - [Summary](#Summary-of-the-generated-datasets)

# ## Description
# The *Adult Census Income* dataset has been extracted from the federal census database in 1994 by Barry Becker. It contains information about people and the prediction **task** is to determine whether they make over \$50k a year (**Supervised** (binary) **classification problem**). The dataset can be found at the following link https://archive.ics.uci.edu/ml/datasets/Census+Income.

# In[ ]:


# Read the full dataset
census = pd.read_csv("/kaggle/input/adult-census-income/adult.csv", header=0, names=["age", "workclass", "final_weight", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital", "cap_loss", "work_hours", "country", "income"])

# Merge capital columns
census.capital = census.capital - census.cap_loss
census.drop(columns=["cap_loss"], inplace=True)

# Info
printmd(f"The dataset is characterized by **{census.shape[0]}** rows and **{census.shape[1]}** features.")
census.head(10)


# ### Features
# 
# Below you can find the description of the features.
# 
# | Feature | Description | Variable type |
# | --- | --- | --- |
# | **[Age](#Age)** | Age of the individual | Numerical |
# | **[Workclass](#Workclass)** | Job sector of the individual | Categorical |
# | **[Final weight](#final-weight)** | Final weight calculated by the CPS<sup style="color: red; font-weight: bold">1</sup> | Numerical |
# | **[Education](#Education---Education-number)** | Highest education of the individual | Categorical - ordinal |\n"
# | **[Education number](#Education---Education-number)** | Number corresponding to the education<sup style="color:red; font-weight: bold">2</sup> | Numerical |
# | **[Marital Status](#Marital-status)** | Marital condition of the individual | Categorical |
# | **[Occupation](#Occupation)** | Job of the individual | Categorical |
# | **[Relationship](#Relationship)** | Social relationship of the individual | Categorical |
# | **[Race](#Race)** | Race of the individual | Categorical |
# | **[Sex](#Sex)** | Sex of the individual | Categorical |
# | **[Capital](#Capital)** | Investment gains or losses of the individual | Numerical |
# | **[Work hours](#Work-hours)** | Work hours per week | Numerical |
# | **[Country](#Country)** | Native country of the individual | Categorical |
# | **[Income](#Income)** (**target**) | Income of the individual | Categorical |
# 
# ><sup style="color:red; font-weight: bold">1</sup> <span style="font-size: 12px">The weights on the Current Population Survey (CPS) files are controlled to independent estimates of the civilian noninstitutional population of the US. These are prepared monthly for us by Population Division here at the Census Bureau.</span>
# 
# ><sup style="color:red; font-weight: bold">2</sup> <span style="font-size: 12px">The relationship between the *education* and the *education number* is reported [here](#Education---Education-number).</span>

# ## Data Preparation
# 
# ### Missing values
# 
# There are no null (`Nan`) values in the dataset. Nevertheless, we will see that there are some features that assume an **unknown** value (*?*).

# In[ ]:


printmd("### Duplicated rows\n\n"
        "As you can see from the following table, there are some rows that are **duplicated**. All the duplicated rows appear only twice (except for one appearing three times). "
        "Of course, the duplicated rows have been **removed** (keeping only one copy).")

if len(census[census.duplicated()]) != 0:
    duplicated = census[census.duplicated(keep=False)].sort_values(by=list(census.columns)).pivot_table(index=list(census.columns), aggfunc="size").reset_index(name="repetitions")
    census.drop_duplicates(ignore_index=True, inplace=True)
    
duplicated.head(len(duplicated))


# ## Data Exploration

# In[ ]:


# Income
printmd("### Income\n\n"
        "Income is our **binary target variable** that indicates whether a person makes over \$50K per year or not. The first thing to notice is that the dataset is a little bit *unbalanced*: "
        f"most of the records belong to the `<$50K` class ({census.income.value_counts()[0] * 100.0 / len(census):.2f}% -> baseline accuracy for the models).")

# Plot
fig = go.Figure(go.Histogram(x=census.income[census.income == "<=50K"], name="< $50K"))
fig.add_trace(go.Histogram(x=census.income[census.income == ">50K"], name="> $50K"))
fig.update_layout(title="Income", xaxis_title="Income", yaxis_title= "Count", legend_title_text="Income", showlegend=True)
fig.show()


# In[ ]:


# Age
printmd("### Age\n\n"
        "Age is a discrete *numerical* feature that indicates the age of the individuals. The boxplot shows that:\n"
        " - most of the individuals are less than 50 years old\n"
        " - older individuals tend to make more money")

# Plot
fig = go.Figure()
fig.add_traces([go.Box(x=census.age, name="Total", visible=False),
                go.Box(x=census.age[census.income == "<=50K"], name="< $50K"),
                go.Box(x=census.age[census.income == ">50K"], name="> $50K")])
fig.update_layout(title="Age", xaxis_title="Age", yaxis_title= "Income", showlegend=False, updatemenus=[
    dict(type="buttons", direction="right", active=1, x=1, y=1.25, buttons=list([
        dict(label="Total", method="update", args=[{"visible": [True, False, False]}]),
        dict(label="Per income", method="update", args=[{"visible": [False, True, True]}]),
    ]))
])
fig.show()


# In[ ]:


# Workclass
workclass = mycatpivot(census, "workclass")

printmd("### Workclass\n\n"
        "Workclass is a *categorical* feature indicating the job sector of the individuals. The barplot shows that:\n"
        f" - most of the individuals work in the *Private* sector ({workclass.total.loc['Private'] * 100.0 / workclass.total.sum():.2f}%)\n"
        f" - for a lot of individuals the workclass is unknown *?* ({workclass.total.loc['?'] * 100.0 / workclass.total.sum():.2f}%) (addressed [here](#Workclass---Occupation))\n"
        " - the classes *Never-worked* and *Without-pay* count a very small amount of records and are all related to income `<$50K` (addressed [here](#Workclass---Occupation))")

mycatplot(workclass, "Workclass").show()


# In[ ]:


# Occupation
occupation = mycatpivot(census, "occupation")

printmd("### Occupation\n\n"
        "Occupation is a *categorical* feature indicating the specific occupation of the individual. The barplot shows that:\n"
        " - there is not a predominant occupation\n"
        f" - for a lot of individuals the occupation is unknown *?* ({occupation.total.loc['?'] * 100.0 / occupation.total.sum():.2f}%) (addressed [here](#Workclass---Occupation))")

mycatplot(occupation, "Occupation").show()


# In[ ]:


# Workclass - Occupation
printmd("### Workclass - Occupation\n\n"
        "Analysing the features workclass and occupation together, it is possible to notice that:\n"
        " - both have unknown values (*?*) with almost a 1-to-1 relationship between them\n"
        " - the workclass classes *Never-worked* and *Without-pay* are always related to an income `<$50K` (addressed [here](#Outliers))")

workclass_occupation = census.pivot_table(index="workclass", columns="occupation", values="age", aggfunc="count", fill_value=0)
workclass_occupation.head(len(workclass_occupation))


# In[ ]:


# Education - Education num
education = mycatpivot(census, "education")
education["education_num"] = census.pivot_table(index="education", values="education_num").sort_values(by="education_num").education_num
education.sort_values("education_num", inplace=True)

printmd("### Education - Education number\n\n"
        "Education is a *categorical* feature indicating the heighest education achieved by the individuals. "
        "Each education is associated with an ordinal number going from the lowest level of education to the heighest. The barplot shows that:\n"
        " - most individuals have at least an high-school degree\n"
        " - individuals with an higher level of education tend to make more money")

mycatplot(education, "Education").show()


# In[ ]:


# Marital status
marital_status = mycatpivot(census, "marital_status")

printmd("### Marital status\n\n"
        "Marital status is a *categorical* feature indicating the marital status of the individual. The barplot shows that:\n"
        " - married individuals tend to make more money")

mycatplot(marital_status, "Marital status").show()


# In[ ]:


# Relationship
relationship = mycatpivot(census, "relationship")

printmd("### Relationship\n\n"
        "Relationship is a *categorical* feature indicating the relationship status of the individual. As seen before, the barplot shows that:\n"
        " - married individuals tend to make more money")

mycatplot(relationship, "Relationship").show()


# In[ ]:


# Marital status - Relationship
printmd("### Marital status - Relationship\n\n"
        "Analysing marital status and relationship together, the most important thing to notice is that, if you are an husband or a wife, you of course are married. "
        "On the other hand, if you are unmarried, you cannot be married. Also, notice how differentiating between husband and wife is redundant with the *Sex* feature.")
marital_relationship = census.pivot_table(index="marital_status", columns="relationship", values="age", aggfunc="count", fill_value=0)
marital_relationship


# In[ ]:


# Race
race = mycatpivot(census, "race")

printmd("### Race\n\n"
        "Race is a *categorical* feature indicating the race of the individual.")

mycatplot(race, "Race").show()


# In[ ]:


# Sex
sex = mycatpivot(census, "sex")

printmd("### Sex\n\n"
        "Sex is a *categorical* feature indicating the sex of the individual. The barplot shows that:\n"
        " - male individuals tend to make more money")

mycatplot(sex, "Sex").show()


# In[ ]:


# Country
country = mycatpivot(census, "country")

printmd("### Country\n\n"
        "Country is a *categorical* feature indicating the country of the individual.")

mycatplot(country, "Country").show()


# In[ ]:


# Capital
printmd("### Capital\n\n"
        "Capital gain and capital loss are *numerical* features that indicate how much an individual has gained or lost through investing. "
        "For simplifying the data, I have reduced the two features to a single column that is the difference of the two. (There were no records with both loss and gain different than 0). "
        " The distribution plot shows that:\n"
        f" - most of the individuals do not invest ({len(census.capital[census.capital == 0])*100.0/len(census):.2f}%)\n"
        " - if you earn from investments, you tend to earn more")

# Plot
fig = go.Figure()
fig.add_traces([go.Box(x=census.capital[census.capital != 0], visible=False),
                go.Box(x=census.capital[(census.income == "<=50K") & (census.capital != 0)], name="< $50K"),
                go.Box(x=census.capital[(census.income == ">50K") & (census.capital != 0)], name="> $50K")])
fig.update_layout(title="Capital", xaxis_title="Capital gain", yaxis_title= "Income", showlegend=False, updatemenus=[
    dict(type="buttons", direction="right", active=1, x=1, y=1.25, buttons=list([
        dict(label="Total (!=0)", method="update", args=[{"visible": [True, False, False]}]),
        dict(label="Per income (!=0)", method="update", args=[{"visible": [False, True, True]}]),
    ]))
])
fig.show()


# In[ ]:


# Work hours
printmd("### Work hours\n\n"
        "Work hours is a *numerical* feature that indicates the number of work hours per week of the individuals. The distribution plot shows that:\n"
        " - most of the individuals work 40 hours per week (25% and 50% quartiles coincide on 40: at least 25% of the individauls work 40h/week)\n"
        " - individuals that work more tend to make more money")

# Plot
fig = go.Figure()
fig.add_traces([go.Box(x=census.work_hours, name="Total", boxpoints=False, visible=False),
                go.Box(x=census.work_hours[census.income == "<=50K"], name="< $50K", boxpoints=False),
                go.Box(x=census.work_hours[census.income == ">50K"], name="> $50K", boxpoints=False)])
fig.update_layout(title="Work hours", xaxis_title="Work hours", yaxis_title= "Income", showlegend=False, updatemenus=[
    dict(type="buttons", direction="right", active=1, x=1, y=1.25, buttons=list([
        dict(label="Total", method="update", args=[{"visible": [True, False, False]}]),
        dict(label="Per income", method="update", args=[{"visible": [False, True, True]}]),
    ]))
])
fig.show()


# ## Feature Engineering
# 
# Since the dataset has a lot of features and most of them are categorical, we cannot expect models to work directly with the data we have, we need to perform some kind of *feature engineering* to simplify the structure of the data.
# 
# <span id="Missing-values-2"></span>
# 
# ### Missing values
# 
# Until now, I have only removed the duplicated rows, but there are other things that need to be analysed. To begin with, the unknown values (*?*) assumed by the features *Workclass*, *Occupation* and *Country* need to be removed; there are two ways to proceed:
# 
# - **drop** the rows that contain an unknown value: this is sub-optimal because we are both dropping important information and reducing the size of the dataset
# - **impute** the missing values: this has been proven to be a good idea even if this process involves guessing the unknown value
# 
# One of the most common imputation techniques for categorical values is to impute the most frequent class. Nevertheless, when there is not a predominant class (as in occupation), a good solution is to create an *Other* class. The imputations that I have performed are:
# 
# - impute *Private* for workclass
# - impute *Other* for occupation
# - impute *United-States* for country
# 
# In the case of workclass and occupation, it was also possible to think that *?* meant "unemployed" but there are some of these rows that are related to an income greater than \\$50K. Therefore I have rejected this hypothesis.
# 
# ### Outliers
# 
# From the data exploration, no particular evidence of outliers has been detected. Nevertheless, I think that this is the right place to point out the following thing: the workclass feature contains two particular values *Never-worked* and *Without-pay*; these two values are always related to an income less than 50K but I would say that, for domain knowledge, they must *always* be related to 0 income. Therefore it doesn't make sense to have them in the dataset.
# 
# ### Transformation
# 
# Since we can't perform the standard dimensionality reduction on one hot encoded variables, we can try to decrease the number of classes inside each categorical feature, so that the generated features are less numerous. This technique is called **binning**, and I have grouped together the following values:
# 
# - *Workclass*:
#   - **Private**: [Private]
#   - **Gov**: [Federal-gov, Local-gov, State-gov]
#   - **Self**: [Self-emp-inc, Self-emp-not-inc]
# - *Marital status*:
#   - **Married**: [Married-AF-spouse, Married-civ-spouse, Married-spouse-absent, Separated]
#   - **Single**: [Widowed, Divorced, Never-married]
# - *Relationship*:
#   - **Spouse**: [Husband, Wife]
#   - **Relative**: [Own-child, Other-relative]
#   - **Other**: [Unmarried, Not-in-family]
# - *Race*:
#   - **White**: [White]
#   - **Other**: [Black, Asian-Pac-Islander, Other, Amer-Indian-Eskimo]
# - *Country*:
#   - **US**: [United-States]
#   - **Other**: [*All the others*]
# 
# Another possible technique for continuous variables like *Work hours* and *Age* is **discretization**: the continuous values are divided into ranges/groups of values so that the resulting feature is an ordinal feature with few categorical values. Discretization helps to reduce noise and some kind of algorithms are more efficient. Specifically, I formed the following groups:
# 
# - *Age*: ranges of 10 ([0,9], [10,19], ...)
# - *Work hours*: ranges of 10 ([0,9], [10,19], ...)
# 
# Another variable that may be discretized is *Capital*: most of the individuals in the dataset do not invest. Therefore, it may make sense to convert this variable into a categorical one:
# 
# - 0 if the individual did not invest or if the result of his investments is zero
# - 1 if the investments brought to a gain
# - 2 if the investments brought to a loss
# 
# These three features have been scaled using a *StandardScaler*.
# 
# ### Selection
# 
# One last step that can be conducted, is to manually reduce the dimensionality of the dataset by selecting some features to train the model on (remember that using PCA is not optimal with one hot encoded variables). Of course, doing so requires domain knowledge; nevertheless, the data exploration has given us some ideas on which features may be good predictors for the target variable:
# 
# - *Age*: as shown [here](#Age), older individuals tend to make more money
# - *Education*: as shown [here](#Education---Education-number), higher level of education often imply higher income
# - *Sex*: as shown [here](#Sex), male individuals tend to make more money than females
# - *Work hours*: as shown [here](#Work-hours), individuals that work more hours per week tend to make more money

# ## Datasets generation
# 
# To feed the datasets into the models, there are two things that still need to be performed:
# 
# - splitting the original dataset into train and test set
# - encode all the categorical variables
# 
# ### Splitting
# 
# Since we are also comparing how the different feature engineering techniques affect the results, the split is performed only on the original dataset. In this way, the two splits are the same for all the generated datasets. Another thing to keep in mind is that our dataset is unbalanced (76% of `<$50K`): hence, it is appropriate to stratify the split (keep the proportion of the 2 classes in the 2 splits). I opted for a 3:1 split.
# 
# ### Encoding
# 
# As mentioned before, the models cannot work directly with categorical variables, therefore we need to encode them:
# 
# - the first option is to One Hot Encode them: generate a binary column for each class of a categorical feature to indicate whether the row had that value or not.
# - the second option is to Label Encode them: assign a number to each class (this is not ideal because we are implicitly generating an order between the classes and also giving them different magnitude)
# 
# I tried both options and compared the results.
# 
# ### Summary of the generated datasets
# 
# - adult.csv: unmodified dataset
# 
# **Preparation:**
# - **clean**: clean dataset (no duplicated rows, no fnlwgt, only the ordinal version of education, no never-worked and without-pay records, difference of capitals) (this is the base for the other datasets)
# 
# **Missing values**
# 
# *Must implement one or another*
# 
# - **impute**: dataset with imputed missing values
# - **drop**: dataset with dropped missing values
# 
# **Feature engineering**
# 
# *Choose whether to implement or not*
# 
# - **discr**: dataset with age, capital and work hours discretized
# - **bin**: dataset with binned features
# 
# **Combinations**
# - **impute_discr**
# - **impute_bin**
# - **impute_bin_discr**
# - **drop_discr**
# - **drop_bin**
# - **drop_bin_discr**

# In[ ]:


def clean(df):
    # Cap
    df.capital = df.capital - df.cap_loss
    df.drop(columns=["cap_loss"], inplace=True)

    # Duplicates
    df.drop_duplicates(ignore_index=True, inplace=True)

    # No fnlwgt and only one education
    df.drop(columns=["final_weight", "education"], inplace=True)
    df.rename(columns={"education_num": "education"}, inplace=True)

    # No never-without workclass
    df = df[~((df.workclass == "Never-worked") | (df.workclass == "Without-pay"))]
    return df

def impute(df):
    df.workclass = df.workclass.map(lambda x: "Private" if x == "?" else x)
    df.occupation = df.occupation.map(lambda x: "Other" if x == "?" else x)
    df.country = df.country.map(lambda x: "United-States" if x == "?" else x)
    return df
    
def drop(df):
    return df[(df.workclass != "?") & (df.occupation != "?") & (df.country != "?")]

def binning(df):
    df.workclass = df.workclass.map(lambda x: "Private" if x == "Private" else "Gov" if x in ["Federal-gov", "Local-gov", "State-gov"] else "Self")
    df.marital_status = df.marital_status.map(lambda x: "Single" if x in ["Widowed", "Divorced", "Never-married"] else "Married")
    df.relationship = df.relationship.map(lambda x: "Spouse" if x in ["Husband", "Wife"] else "Other" if x in ["Unmarried", "Not-in-family"] else "Relative")
    df.race = df.race.map(lambda x: "White" if x == "White" else "Other")
    df.country = df.country.map(lambda x: "US" if x == "United-States" else "Other")
    return df

def discretize(df):
    df.age = df.age // 10
    df.work_hours = df.work_hours // 10
    df.capital = df.capital.map(lambda x: -1 if x < 0 else 0 if x == 0 else 1)
    return df

# Read again the original dataset
census = pd.read_csv("/kaggle/input/adult-census-income/adult.csv", header=0, names=["age", "workclass", "final_weight", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital", "cap_loss", "work_hours", "country", "income"])
census_train, census_test = train_test_split(census, train_size=0.75, random_state=0, stratify=census.income)

# Save original splits
folder = os.path.join("/kaggle/working/original")
if not os.path.isdir(folder):
    os.mkdir(folder)
census_train.to_csv("/kaggle/working/original/train.csv", index=False)
census_test.to_csv("/kaggle/working/original/test.csv", index=False)

datasets = {
    "clean": {"operations": [clean]},
    "drop": {"operations": [clean, drop]},
    "drop_bin": {"operations": [clean, drop, binning]},
    "drop_discr": {"operations": [clean, drop, discretize]},
    "drop_bin_discr": {"operations": [clean, drop, binning, discretize]},
    "impute": {"operations": [clean, impute]},
    "impute_bin": {"operations": [clean, impute, binning]},
    "impute_discr": {"operations": [clean, impute, discretize]},
    "impute_bin_discr": {"operations": [clean, impute, binning, discretize]},
}

# Generation of datasets
for key in datasets:
    temp_train = census_train.copy()
    temp_test = census_test.copy()
    for op in datasets[key]["operations"]:
        temp_train = op(temp_train)
        temp_test = op(temp_test)
    folder = os.path.join("/kaggle/working", key)
    if not os.path.isdir(folder):
        os.mkdir(folder)
    scaler = StandardScaler().fit(temp_train[["age", "education", "capital", "work_hours"]])
    temp_train[["age", "education", "capital", "work_hours"]] = scaler.transform(temp_train[["age", "education", "capital", "work_hours"]])
    temp_test[["age", "education", "capital", "work_hours"]] = scaler.transform(temp_test[["age", "education", "capital", "work_hours"]])
    temp_train.to_csv(os.path.join(folder, "train.csv"), index=False)
    temp_test.to_csv(os.path.join(folder, "test.csv"), index=False)


# In[ ]:


# Pandas
import pandas as pd
pd.set_option('display.max_colwidth', None)

# Numpy
import numpy as np

# Plotly
import plotly.graph_objects as go

# Scikit learn
from sklearn import set_config
set_config(display='diagram')
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text

# Functions
def train_and_test(datasets, classifier, param_grid, scoring='accuracy', n_jobs=-1, return_train_score=True):
    model = {}
    results = {}
    
    for key in datasets:
        model[key] = {}
        results[key] = {}
        
        # Grid search
        model[key]["model_ohe_grid"] = GridSearchCV(classifier, param_grid, scoring=scoring, n_jobs=n_jobs, return_train_score=return_train_score).fit(datasets[key]["X_train_ohe"], datasets[key]["Y_train"])
        model[key]["model_le_grid"] = GridSearchCV(classifier, param_grid, scoring=scoring, n_jobs=n_jobs, return_train_score=return_train_score).fit(datasets[key]["X_train_le"], datasets[key]["Y_train"])
        model[key]["model_ohe"] = model[key]["model_ohe_grid"].best_estimator_
        model[key]["model_le"] = model[key]["model_le_grid"].best_estimator_

        # Test
        results[key]["ohe"] = model[key]['model_ohe'].score(datasets[key]['X_test_ohe'], datasets[key]['Y_test'])
        results[key]["le"] = model[key]['model_le'].score(datasets[key]['X_test_le'], datasets[key]['Y_test'])
    
    return model, results

def plot_results(results, title):
    fig = go.Figure()
    results = pd.DataFrame(results)
    for col in results:
        fig.add_traces(go.Bar(y=results.index, x=results[col], orientation="h", name=col))
    fig.update_layout(title=title, xaxis = dict(range=[0.5, 0.9]))
    return fig 

def acc_analysis(datasets, model_dict, param_name):
    analysis = {param_name: [], 'enc': [], 'test': [], 'train': []}
    for key in datasets:
        for param in model_dict[key]["model_ohe_grid"].cv_results_["params"]:
            analysis[param_name].append(param[param_name])
            analysis['enc'].append('ohe')
            analysis['test'].append(model_dict[key]["model_ohe_grid"].cv_results_["mean_test_score"][model_dict[key]["model_ohe_grid"].cv_results_["params"].index(param)])
            analysis['train'].append(model_dict[key]["model_ohe_grid"].cv_results_["mean_train_score"][model_dict[key]["model_ohe_grid"].cv_results_["params"].index(param)])
        for param in model_dict[key]["model_le_grid"].cv_results_["params"]:
            analysis[param_name].append(param[param_name])
            analysis['enc'].append('le')
            analysis['test'].append(model_dict[key]["model_le_grid"].cv_results_["mean_test_score"][model_dict[key]["model_le_grid"].cv_results_["params"].index(param)])
            analysis['train'].append(model_dict[key]["model_le_grid"].cv_results_["mean_train_score"][model_dict[key]["model_le_grid"].cv_results_["params"].index(param)])

    analysis_train = pd.pivot_table(pd.DataFrame(analysis), index=param_name, columns="enc", values="train", aggfunc=np.mean)
    analysis_test = pd.pivot_table(pd.DataFrame(analysis), index=param_name, columns="enc", values="test", aggfunc=np.mean)

    fig = go.Figure()
    for enc in analysis_train:
        fig.add_traces(go.Scatter(x=analysis_train.index, y=analysis_train[enc], name=f"Train - {enc}"))

    for enc in analysis_test:
        fig.add_traces(go.Scatter(x=analysis_test.index, y=analysis_test[enc], name=f"Test - {enc}"))

    return fig
    
def my_print_tree(tree, feature_names):
    print(export_text(tree, feature_names=feature_names))


# # Adult Census Income - Classification
# 
# ## Table of Contents
# 
# 1. [Introduction](#Introduction)
#   - [Reading the data](#Reading-the-data)
#   - [Training and testing process](#Training-and-testing-process)
# 2. [Logistic Regression](#Logistic-Regression)
#   - [Theory recall](#logreg-theory)
#   - [Parameters](#logreg-parameters)
#   - [Testing](#logreg-testing)
#   - [Conclusions](#logreg-conclusions)
# 3. [Linear Discriminant Analysis](#Linear-Discriminant-Analysis)
#   - [Theory recall](#lda-theory)
#   - [Parameters](#lda-parameters)
#   - [Testing](#lda-testing)
#   - [Conclusions](#lda-conclusions)
# 4. [K-Nearest Neighbors](#K-Nearest-Neighbors)
#   - [Theory recall](#knn-theory)
#   - [Parameters](#knn-parameters)
#   - [Testing](#knn-testing)
#   - [Conclusions](#knn-conclusions)
# 5. [Decision Trees](#Decision-Trees)
#   - [Theory recall](#dt-theory)
#   - [Parameters](#dt-parameters)
#   - [Testing](#dt-testing)
#   - [Conclusions](#dt-conclusions)
# 6. [Random Forest](#Random-Forest)
#   - [Theory recall](#rf-theory)
#   - [Parameters (trees)](#rf-parameters-trees)
#   - [Testing (trees)](#rf-testing-trees)
#   - [Parameters (depth)](#rf-parameters-depth)
#   - [Testing (depth)](#rf-testing-depth)
#   - [Conclusions](#rf-conclusions)
# 7. [Future Work](#Future-Work)

# ## Introduction
# 
# ### Reading the data
# 
# In the previous notebook, we generated different datasets according to the different strategies of feature engineering we are trying to compare. The only step we still haven't performed is encoding; therefore, we can now read the data and encode it using both a label encoder and a one hot encoder. In this step, Xs and ys has also been separated for the next phases.

# In[ ]:


# Read the datasets
datasets = {}
datasets_keys = ["original", "clean", "drop", "drop_bin", "drop_discr", "drop_bin_discr", "impute", "impute_bin", "impute_discr", "impute_bin_discr"]

# Read and encode
for key in datasets_keys:
    datasets[key] = {}
    datasets[key]["X_train"] = pd.read_csv(f"/kaggle/working/{key}/train.csv")
    datasets[key]["X_test"] = pd.read_csv(f"/kaggle/working/{key}/test.csv")
    
    # Save target variable as 0 / 1 codes
    datasets[key]["Y_train"] = datasets[key]["X_train"].income.astype("category").cat.codes
    datasets[key]["Y_test"] = datasets[key]["X_test"].income.astype("category").cat.codes
    
    # One Hot Encoding
    datasets[key]["X_train_ohe"] = datasets[key]["X_train"].copy().drop(columns=["income"])
    datasets[key]["X_test_ohe"] = datasets[key]["X_test"].copy().drop(columns=["income"])
    for col in datasets[key]["X_train_ohe"].select_dtypes("object").columns:
        if len(datasets[key]["X_train_ohe"][col].unique()) == 2:
            datasets[key]["X_train_ohe"][col] = datasets[key]["X_train_ohe"][col].astype("category").cat.codes
            datasets[key]["X_test_ohe"][col] = datasets[key]["X_test_ohe"][col].astype("category").cat.codes
    datasets[key]["X_train_ohe"] = pd.get_dummies(datasets[key]["X_train_ohe"])
    datasets[key]["X_test_ohe"] = pd.get_dummies(datasets[key]["X_test_ohe"])
    
    # Label Encoding
    datasets[key]["X_train_le"] = datasets[key]["X_train"].copy().drop(columns=["income"])
    datasets[key]["X_test_le"] = datasets[key]["X_test"].copy().drop(columns=["income"])
    for col in datasets[key]["X_train_le"].select_dtypes("object").columns:
        datasets[key]["X_train_le"][col] = datasets[key]["X_train_le"][col].astype("category").cat.codes
        datasets[key]["X_test_le"][col] = datasets[key]["X_test_le"][col].astype("category").cat.codes
    
    del datasets[key]["X_train"]
    del datasets[key]["X_test"]


# ### Training and testing process
# 
# The general process that I have followed to find the best model for the given data is the following:
# 
# - Load the data (already split into train and test sets in the previous notebook)
# - Encode all the splits (label encoder and one hot encoder to compare them)
# - Define different parameters to be tuned for each model
# - Conduct a **GridSearchCV** to find the best params for each kind of feature engineering dataset ('best' evaluated in terms of *accuracy*)
# - Evaluate the best models, on the test set, and compare the accuracies

# ## Logistic Regression
# 
# <span id="logreg-theory" />
# 
# ### Theory recall
# 
# Since our target variable is categorical with only two classes, linear regression doesn't work very well (because the generated output isn't between 0 and 1) and logistic regression is more appropriate; this algorithm, rather than modeling the response directly, it models the *probability* that the target variable belongs to a particular class:
# 
# $$p(X) = \frac{e^{\left(\beta_{0}+\beta_{1}X\right)}}{1+e^{\left(\beta_{0}+\beta_{1}X\right)}} = \frac{1}{1+e^{-\left(\beta_{0}+\beta_{1}X\right)}}$$
# 
# If you look closely, you can see that this model is a linear function plugged into the sigmoid function (also called logistic function). In fact, the main difference between linear regression and logistic regression is that the output of the latter will always take values between 0 and 1: when the output is higher than 0.5, class 1 will be assigned, otherwise class 0 is assigned.
# 
# Manipulating the model equation, it is possible to obtain the *logit* transformation:
# 
# $$log\left(\frac{p(X)}{1-p(X)}\right) = \beta_{0}+\beta_{1}X$$
# 
# The regression coefficients are estimated maximizing a conditional ($P\left(y|X\right)$ *discriminative learning*) likelihood function: scikit learn's implementation minimizes the negative log likelihood function, which is a function called *cross-entropy*. Of course, with multiple predictors like in our case, the model is constructed on a linear combination of coefficients and predictors $\beta_{0}+\beta_{1}X_{1}+\beta_{2}X_{2}+...+\beta_{n}X_{n}$. (The likelihood function doesn't have a closed form; therefore an iterative process must be used.)
# 
# <span id="logreg-parameters" />
# 
# ### Parameters
# 
# With the grid search, the following parameters have been optimized:
# 
# - *C*: this is the inverse of the regularization strength (sklearn models use regularization by default (better generalization))
# - *solver* (check documentation [here](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression))

# In[ ]:


# Logistic Regression
logistic_regression, logistic_regression_results = train_and_test(
    datasets,
    LogisticRegression(max_iter=500),
    [{'C': [0.001, 0.01, 0.1, 1, 2, 5, 10, 100, 1000], 'solver': ['liblinear', 'lbfgs']}]
)


# If we analyse how the accuracy evolves changing the *C* parameter, we can notice the following things:
# 
# - when C is too small, the model underfits (too much regularization)
# - increasing C allows the model to be more accurate
# - if C is too big the test accuracy decreases a little bit (overfitting, because almost no regularization) (slightly noticeable in one hot encoding when the two curves diverge)

# In[ ]:


acc_analysis(datasets, logistic_regression, 'C').update_layout(xaxis_type="log", xaxis_title="C", yaxis_title="Balanced accuracy", title="Logistic regression accuracy against C")


# <span id="logreg-testing" />
# 
# ### Testing
# 
# Testing the best models of each *\"feature engineering type\"*, we obtain the accuracies reported in the plot below.

# In[ ]:


# Plot
plot_results(logistic_regression_results, "Logistic regression test accuracy").show()
print("Best model:")
logistic_regression["clean"]["model_ohe"]


# <span id="logreg-conclusions" />
# 
# ### Conclusions
# 
# As we can see from the previous plot:
# 
# - the best model has achieved an accuracy of 84,5% on the *Clean* and *Impute* one-hot-encoded datasets
# - one-hot-encoded datasets have performed, on average, better than label encoded ones (as we sad before, label encoding creates a magnitude relationship between classes that does not represent reality, hence performes poorly)
# - the technique of imputing brought generally better results than dropping
# - binning and discretizing don't seem to improve the overall model performance with one hot encoding

# ## Linear Discriminant Analysis
# 
# <span id="lda-theory" />
# 
# ### Theory recall
# 
# *Discriminant analysis* is a probabilistic model whose approach is to estimate the distribution of each feature in the dataset, and then compute the posterior probability of the target class exploiting the Bayes theorem (which makes the assumption of independent features). Therefore, the predictions are made looking to the feature with the highest density. Bayes recall:
# 
# $$P\left(Y=k|X=x\right) = \frac{P\left(X=x|Y=k\right)*P\left(Y=k\right)}{P\left(X=x\right)} = \frac{\pi_{k}*f_{k}\left(x\right)}{\sum_{l=1}^{K}\pi_{l}*f_{l}\left(x\right)}$$
# 
# On the right part of the equation $f_{k}\left(x\right)$ is the *density* of $x$ when the class is $k$ and $\pi_{k}$ is the *marginal* probability of class $k$.
# 
# With multiple predictors like in our case, the model assumes that the input $X$ has a multivariate normal distribution, with distinct means and common covariance matrix; hence, the discriminant function has the form:
# 
# $$\delta_{k}\left(x\right) = x^{T}\Sigma^{-1}\mu_{k}-\frac{1}{2}\mu_{k}^{T}\Sigma^{-1}\mu_{k}+log\left(\pi_{k}\right)$$
# 
# To an observation $x$ will be assigned the class with the largest *discriminant score*:
# 
# $$\delta_{k}\left(x\right) = x*\frac{\mu_{k}}{\sigma^{2}} - \frac{\mu_{k}^{2}}{2\sigma^{2}} + log\left(\pi_{k}\right)$$
# 
# Linear discriminant analysis parameters are estimated using the full likelihood function ($P\left(X,y\right)$ *generative learning*).
# 
# <span id="lda-parameters" />
# 
# ### Parameters
# 
# No parameters optimized with this model.

# In[ ]:


# Linear Discriminant Analysis
lda, lda_results = train_and_test(
    datasets,
    LinearDiscriminantAnalysis(),
    [{'solver': ['svd']}]
)


# <span id="lda-testing" />
# 
# ### Testing
# 
# Testing the best models of each *\"feature engineering type\"*, we obtain the accuracies reported in the plot below.

# In[ ]:


# Plot
plot_results(lda_results, "Linear discriminant analysis test accuracy").show()
print("Best model: (default params)")
lda["original"]["model_ohe"]


# <span id="lda-conclusions" />
# 
# ### Conclusions
# 
# As we can see from the plot above:
# 
# - the best model has achieved an accuracy of 84% on the *Original* one-hot-encoded dataset
# - as in logistic regression, on average, one hot encoding has performed better than label encoding

# ## K-Nearest Neighbors
# 
# <span id="knn-theory" />
# 
# ### Theory recall
# 
# KNN is a non-parametric model that exploits the available data to make predictions; specifically, after having defined a proper distance measure, the model looks at the $K$ *nearest* points and then assigns the class by majority voting. The main drawback of this model, apart from the infering slowness, is that highdimensional space is empty, therefore, with a lot of variables, we cannot expect the model to perform very well. The main advantage is that the model is the training data.
# 
# <span id="knn-parameters" />
# 
# ### Parameters
# 
# The only parameter to optimize is $K$.

# In[ ]:


# KNN
knn, knn_results = train_and_test(
    datasets,
    KNeighborsClassifier(),
    [{'n_neighbors': [3, 5, 7, 9, 13, 17, 21, 25, 30, 40]}]
)


# If we analyse how the accuracy evolves changing the *n_neighbors* parameter, we can notice the following things:
# 
# - when K is too small the model overfits (lack of generalization because looking at too few points)
# - when K is too big the model underfits (not visible in our testing because not so big Ks used)

# In[ ]:


acc_analysis(datasets, knn, 'n_neighbors').update_layout(xaxis_title="K", yaxis_title="Balanced accuracy", title="K-Nearest Neighbors accuracy against K")


# <span id="knn-testing" />
# 
# ### Testing
# 
# Testing the best models of each *\"feature engineering type\"*, we obtain the accuracies reported in the plot below.

# In[ ]:


# Plot
plot_results(knn_results, "K-Nearest Neighbors test accuracy").show()
print("Best model:")
knn["impute"]["model_ohe"]


# <span id="knn-conclusions" />
# 
# ### Conclusions
# 
# Even though our dataset has a lot of features, KNN managed to achieve good accuracies thanks to the sklearn implementation.

# ## Decision Trees
# 
# <span id="dt-theory" />
# 
# ### Theory recall
# 
# Decision tree models stratify the predictors space into simpler regions in order to make predictions. The assigned class is simply the most probable inside that region.
# 
# The most difficult part is therefore the definition of the tree and the *split criterions*. The main idea behind the definition of a *split criterion* is to generate splits where there is a low *mixing* of class variables. Hence, we need to define a measure for this *mixing* and then generate all the splits recursively.
# 
# Of course, the model stops when the level of *mixing* inside a branch is below a given threshold. In this way, the risk of overfitting is reduced. On the other hand, stopping too soon may result in underfitting.
# 
# The two most common measures of impurity are:
# 
# - Gini index: $G(node)=1-\sum\limits_{k=1}^{K}p\left(k|node\right)^{2}$ which is a measure of total variance across the $K$ classes in the node
# - Cross-entropy: $D=-\sum\limits_{k=1}^{K}p\left(k|node\right)log\left(p\left(k|node\right)\right)$
# 
# The main advantages of this model are:
# 
# - Simple and interpretable
# - Almost no need of data preparation (can handle categorical variables)
# 
# On the other hand, the main disadvantages are:
# 
# - Can overfit pretty easily
# - Not very accurate with respect to other supervised models (however, despite decreasing the interpretability, aggregating trees can solve this problem; see [Random Forest](#Random-Forest))
# - Affected by unbalanced target classes
# 
# <span id="dt-parameters" />
# 
# ### Parameters
# 
# With the grid search, the following parameters have been optimized:
# 
# - *criterion*: the split criterion used to measure the level of impurity of a split
# - *max_depth*: the maximum depth of the tree (to control overfitting)

# In[ ]:


# Decision trees
dt, dt_results = train_and_test(
    datasets,
    DecisionTreeClassifier(),
    [{'criterion': ['gini', 'entropy'], 'max_depth': [1, 2, 3, 5, 6, 8, 12, 15, 18, 22, 26]}]
)


# If we analyse how the accuracy evolves changing the *max_depth* parameter, we can notice the following things:
# 
# - when the parameter is too small, the tree underfits (only one split)
# - when the parameter is around 8, the accuracy on the test set is around his maximum
# - increasing too much the parameter makes the tree overfitting

# In[ ]:


acc_analysis(datasets, dt, 'max_depth').update_layout(xaxis_title="max_depth", yaxis_title="Balanced accuracy", title="Decision tree accuracy against max_depth")


# <span id="knn-testing" />
# 
# ### Testing
# 
# Testing the best models of each *\"feature engineering type\"*, we obtain the accuracies reported in the plot below.

# In[ ]:


# Plot
plot_results(dt_results, "Decision tree test accuracy").show()
print("Best model:")
dt["impute_bin"]["model_le"]


# <span id="dt-conclusions" />
# 
# ### Conclusions
# 
# As we can see from the following plot:
# 
# - since the decision trees can handle categorical features, the best performing datasets have been the label encoded ones
# - it is interesting how discretizing variables decreases the performance of the trees

# ## Random Forest
# 
# <span id="rf-theory" />
# 
# ### Theory recall
# 
# Random forest models are ensemble of decision trees: the main idea is to generate multiple decision trees and take as output the average of the output of all the trees. The tree algorithm of random forests is a little bit different: at each candidate split in the learning process, only a random subset of the features is selected (usually $\sqrt p$ where $p$ is the number of predictors).
# 
# The main advantage of this algorithm is that the different trees are decorrelated, thus reducing the mean variance of the observed data.
# 
# <span id="rf-parameters-trees" />
# 
# ### Parameters (trees)
# 
# To begin with, I have looked for the best number of candidate trees using the best max_depth found with a single decision tree. Then, using the best number of trees, I have optimized the max depth of the trees. 

# In[ ]:


# Random forest
rf, rf_results = train_and_test(
    datasets,
    RandomForestClassifier(max_depth=8),
    [{'n_estimators': [1, 10, 50, 100, 150], 'criterion': ['gini', 'entropy']}]
)


# If we analyse how the accuracy evolves changing the *n_estimators* parameter, we can notice the following things:
# 
# - using too few trees, the real potential of random forests is not exploited
# - when the number of trees increases, the effects of random forests start to be noticed

# In[ ]:


acc_analysis(datasets, rf, 'n_estimators').update_layout(xaxis_title="n_estimators", yaxis_title="Balanced accuracy", title="Random forest accuracy against n_estimators")


# <span id="rf-testing-trees" />
# 
# ### Testing (trees)
# 
# Testing the best models of each *\"feature engineering type\"*, we obtain the accuracies reported in the plot below.

# In[ ]:


# Plot
plot_results(rf_results, "Random forest test accuracy").show()
print("Best model:")
dt["impute_bin"]["model_ohe"]


# <span id="rf-parameters-depth" />
# 
# ### Parameters (depth)
# 
# We can now try to optimize the depth of the trees.

# In[ ]:


# Random forest depth
rf_depth, rf_results_depth = train_and_test(
    datasets,
    RandomForestClassifier(n_estimators=100),
    [{'max_depth': [1, 3, 6, 8, 12, 15], 'criterion': ['gini', 'entropy']}]
)


# If we analyse how the accuracy evolves changing the *max_depth* parameter, we can notice the following things:
# 
# - when the parameter is too small, the tree underfits (only one split)
# - when the parameter is around 12, the accuracy on the test set is around his maximum
# - increasing too much the parameter makes the tree overfitting

# In[ ]:


acc_analysis(datasets, rf_depth, "max_depth").update_layout(xaxis_title="max_depth", yaxis_title="Balanced accuracy", title="Random forest accuracy against max_depth")


# <span id="rf-testing-depth" />
# 
# ### Testing (depth)
# 
# Testing the best models of each *\"feature engineering type\"*, we obtain the accuracies reported in the plot below.

# In[ ]:


# Plot
plot_results(rf_results_depth, "Random forest").show()
print("Best model:")
dt["clean"]["model_le"]


# <span id="rf-conclusions" />
# 
# ### Conclusions
# 
# The Random Forest algorithm, with 8 maximum depth and 100 candidate tress, is the best model for the given dataset. In fact, it has achieved an accuracy of 86,5% on the test set using label encoding and imputation.

# ## Future Work
# 
# I think that achieving around 86,5% accuracy can be considered pretty good given the difficulty of dataset. Nonetheless, I am aware the there are multiple ways to maybe improve the current models. The following possibilities of improving the models are left for the future:
# 
# - try different metrics to bettere evaluate the parameters
# - try other combinations of feature engineering and maybe also different techniques
# - try to implement a feature selection or dimensionality reduction algorithm

# In[ ]:




