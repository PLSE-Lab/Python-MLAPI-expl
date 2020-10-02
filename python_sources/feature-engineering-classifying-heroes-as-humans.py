#!/usr/bin/env python
# coding: utf-8

# # Super Heroes Dataset
# 
# The goal of the task is to **predict whether a superhero is Human or not** based on their characteristics and super powers.

# ## Outline
# 
# - [Import Libraries and Data](#Import-Libraries-and-Data) 
# - [Feature Engineering](#Feature-Engineering) 
#     - [Data Overview](#Data-Overview) 
#     - [Repeated Heroes](#Repeated-Heroes) 
#     - [Handling Null Values](#Handling-Null-values) 
#     - [Categorical Variables and One-hot encoding](#Categorical-Variables-and-One-Hot-Encoding) 
#     - [All data together](#All-data-together) 
# - [Modeling](#Modeling) 
#     - [Plan](#Plan)
#     - [Training Classifiers](#Training-classifiers) 
#     - [Feature Importance](#Feature-Importance)
#     - [Dimensionality Reduction](#Dimensionality-Reduction)

# ----------------
# 
# ## Import Libraries and Data

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib import cm
import seaborn as sns
import tqdm
import warnings

warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(rc={"figure.figsize": (10, 12)})
np.random.seed(sum(map(ord, "palettes")))


# Let's call the datasets in the following way throughout the Notebook:
# 
# - Heroes Information : **`metadata`**
# - Heroes Superpowers : **`powers`**

# In[ ]:


metadata = pd.read_csv("../input/heroes_information.csv", index_col=0)
powers = pd.read_csv("../input/super_hero_powers.csv")


# -------------------
# 
# ## Feature Engineering
# 
# ### Data Overview

# In[ ]:


print("Heroes information data shape: ", metadata.shape)
print("Hero super powers data shape: ", powers.shape)


# In[ ]:


metadata.head()


# In[ ]:


powers.head()


# ### Repeated Heroes 

# As the process of dealing with Repeated Heroes is quite extensive, I created a function in the cell below that does all the work in a single step. If you would like to know the why of each action, please continue scrolling down, if not, jump to the [next section](#Handling-Null-values) .

# In[ ]:


def clean_repeated_heroes(metadata, powers):
    
    print("Initial shape of metadata and powers: ")
    print("Powers:", powers.shape)
    print("Metadata", metadata.shape)
    
    print("\nStart cleaning...")
    
    powers.drop_duplicates(inplace=True)
    metadata.drop_duplicates(inplace=True)
    
    # Handle Goliath
    goliath_idxs_to_drop = [100, 289, 290] # not dropping Goliath IV, it will be used to join powers
    metadata.drop(goliath_idxs_to_drop, inplace=True)
    metadata.loc[metadata.name == "Goliath IV", "Race"] = "Human"
    
    # Avoid outersected entries. i.e. appearing in metadata, but not in powers. And viceversa.
    metadata = metadata[metadata.name.isin(powers.hero_names)]
    powers = powers[powers.hero_names.isin(metadata.name)]
    
    # Spider-Man
    metadata.loc[metadata.name.str.contains("Spider-Man")] = metadata[metadata.name.str.contains("Spider-Man")].mode().values[0]
    metadata.drop(623, inplace=True)
    metadata.drop(624, inplace=True)

    # Nova
    metadata.drop(497, inplace=True)

    # Angel
    metadata.loc[metadata.name == "Angel", "Race"] = "Vampire"
    metadata.drop(23, inplace=True)

    # Blizzard
    metadata.loc[metadata.name == "Blizzard"] = metadata.loc[metadata.name == "Blizzard II"].values
    metadata.at[115, 'name'] = "Blizzard"
    metadata.at[116, 'Race'] = "Human"
    metadata.at[115, 'Race'] = "Human"
    metadata.drop(117, inplace=True)

    # Black Canary
    metadata.drop(97, inplace=True)

    # Captain Marvel
    metadata.at[156, 'Race'] = "Human"
    metadata.drop(155, inplace=True)

    # Blue Beettle
    metadata.at[122, 'Race'] = "Human"
    metadata.at[124, 'Race'] = "Human"
    metadata.at[122, 'Height'] = 183.0
    metadata.at[125, 'Height'] = 183.0
    metadata.at[122, 'Weight'] = 86.0
    metadata.at[125, 'Weight'] = 86.0
    metadata.drop(123, inplace=True)

    # Vindicator
    metadata.drop(696, inplace=True)

    # Atlas
    metadata.drop(48, inplace=True)

    # Speedy
    metadata.drop(617, inplace=True)

    # Firestorm
    metadata.drop(260, inplace=True)

    # Atom
    metadata.drop(50, inplace=True)
    metadata.at[49, 'Race'] = "Human"
    metadata.at[53, 'Race'] = "Human"
    metadata.at[54, 'Race'] = "Human"
    metadata.at[49, 'Race'] = "Human"
    metadata.at[54, 'Height'] = 183.0
    metadata.at[49, 'Height'] = 183.0
    metadata.at[53, "Weight"] = 72.0

    # Batman
    metadata.drop(69, inplace=True)

    # Toxin
    metadata.drop(673, inplace=True)

    # Namor
    metadata.drop(481, inplace=True)

    # Batgirl
    metadata.drop(62, inplace=True)
    
    print("Final shape of metadata and powers: ")
    print("Powers:", powers.shape)
    print("Metadata", metadata.shape)
    
    print("\nCleaning done")
    
    return metadata, powers


# In[ ]:


# if you run it twice, it won't work due to hard-coded indexers won't match.
# you need to get the data and run it again

# metadata, powers = clean_repeated_heroes(metadata, powers)


# -------------
# 
# #### Explanation

# As mentioned in the exercise wording, both tables do not have a 1-1 mapping between them. Logically this would not make sense, assuming that the super heroes between two tables have same names. This 1-\* correspondance (being 1 `powers` and \* `metadata`) might be caused by `metadata` containing repeated name entries for several heroes, or some data disturbances such as errors in the scraping, etc. Let's try to quickly get a glance of this by 
# 
# - Getting repeated values from `metadata`, or `powers`, in case of existing.
# - Try to make sense of the difference between `metadata` and `powers` number of rows by summing the entries of the repeated rows from `metadata`.

# In[ ]:


powers.drop_duplicates(inplace=True)
metadata.drop_duplicates(inplace=True)


# In[ ]:


print("Number of rows with more than 1 entry per hero name in metadata ", (metadata.name.value_counts() > 1).sum()  )
print("Number of rows with more than 1 entry per hero name in powers ", (powers.hero_names.value_counts() > 1).sum() )


# In[ ]:


mask = metadata.name.value_counts() > 1
metadata.name.value_counts()[mask].sum() - mask.sum() # get excessive number of rows from repeated names


# In[ ]:


# Does it match with difference in table length?
metadata.shape[0] - powers.shape[0]


# Looks like there are still 48 heroes that either do not have an entry in powers but in metadata, and viceversa. Let's figure them out. But first let's check if all repeated heroes are in powers dataset to avoid bias in the result.

# In[ ]:


repeated_heroes = mask.index[mask]
repeated_heroes[~repeated_heroes.isin(powers.hero_names)]


# In[ ]:


powers[powers.hero_names.str.contains("Goliath")]


# In[ ]:


metadata[metadata.name.str.contains("Goliath")]


# It seems like `Goliath` is a bit messed up in `metadata` and it can be cleaned up. `Goliath` and `Black Goliath` are the same superhero. `Goliath IV` seems to be an evolution or something from `Goliath`, but for simplicity we will say that all `Goliaths` appearing in `metadata` are the same one, hence it allows it to be merged with `powers`.
# 
# Therefore, let's unify `Goliath` into one. We will take mostly the characteristics from `Goliath IV` but the Race from the others (`Human`).

# In[ ]:


goliath_idxs_to_drop = [100, 289, 290] # not dropping Goliath IV, it will be used to join powers
metadata.drop(goliath_idxs_to_drop, inplace=True)

# modify Goliath IV row
metadata.loc[metadata.name == "Goliath IV", "Race"] = "Human"


# In[ ]:


metadata[metadata.name.str.contains("Goliath")]


# This looks better. Let's now check how many heroes do not have an entry in `metadata` but in `powers`, and viceversa.

# In[ ]:


# How many superheroes that appear in metadata, do not have an entry in powers?
(~metadata.name.isin(powers.hero_names)).sum()


# In[ ]:


# How many superheroes that appear in powers, do not have an entry in metadata?
(~powers.hero_names.isin(metadata.name)).sum()


# Looks like there are some superheroes that need to be removed from both tables as we won't have their entire information.

# In[ ]:


metadata = metadata[metadata.name.isin(powers.hero_names)]
powers = powers[powers.hero_names.isin(metadata.name)]


# In[ ]:


metadata.shape


# In[ ]:


powers.shape


# Going in the good direction, but it still looks like there are more entries in metadata, this might be related to the number of repeated entries that we did not handle yet. Let's find out.

# In[ ]:


(metadata.name.value_counts() > 1).sum() 


# #### Merge repeated heroes into one

# In[ ]:


repeated_heroes = repeated_heroes.drop("Goliath")

# let's go 1 by 1
for rh in repeated_heroes:
    print("*********** ", rh, " **************")
    print(metadata[metadata.name.str.contains(rh)])
    print("\n\n")


# In[ ]:


# Spider-Man
# As all three instances seem similar, let's just take the mode of each column as new values
metadata.loc[metadata.name.str.contains("Spider-Man")] = metadata[metadata.name.str.contains("Spider-Man")].mode().values[0]
metadata.drop(623, inplace=True)
metadata.drop(624, inplace=True)

# Nova
# They are different superheroes, but with the same name. It is not clear which one is represented in powers df
# We cannot keep both as both would have same superpowers and thus confuse the classifier. For simplicity,
# let's only keep the human Nova. And to choose between male or female, let's keep the female.
metadata.drop(497, inplace=True)

# Angel
# There are 2 Angels. Both rows seemed to have been split from one single row. Let's merge it back.
metadata.loc[metadata.name == "Angel", "Race"] = "Vampire"
metadata.drop(23, inplace=True)

# Blizzard
# Blizzard II has only 1 difference in superpower, therefore we claim both Blizzard and Blizzard II will have same 
# characteristics. And it's considered Human, according to Wikipedia.
metadata.loc[metadata.name == "Blizzard"] = metadata.loc[metadata.name == "Blizzard II"].values
metadata.at[115, 'name'] = "Blizzard"
metadata.at[116, 'Race'] = "Human"
metadata.at[115, 'Race'] = "Human"
metadata.drop(117, inplace=True)

# Black Canary
# Let's take only one of them, as they're practically similar
metadata.drop(97, inplace=True)

# Captain Marvel
# All are same, in exception of Captain Marvel II, which is showed in powers df. Let's keep CM and CM II, but in case 
# of Captain Marvel we will keep the original one by DC (the one from Marvel is copied)
metadata.at[156, 'Race'] = "Human"
metadata.drop(155, inplace=True)

# Blue Beettle
# In powers df, there are the three blue beetles, and they are indeed different in terms of powers.
# Let's keep all of them but they will all have the same characteristics as they are really similar.
metadata.at[122, 'Race'] = "Human"
metadata.at[124, 'Race'] = "Human"
metadata.at[122, 'Height'] = 183.0
metadata.at[125, 'Height'] = 183.0
metadata.at[122, 'Weight'] = 86.0
metadata.at[125, 'Weight'] = 86.0
metadata.drop(123, inplace=True)

# Vindicator
# keep only the one that does not have null values
metadata.drop(696, inplace=True)

# Atlas
# Keep only one of them
metadata.drop(48, inplace=True)

# Speedy
# Searched in Google, they are mainly the same, but male version introduced 1941 and female on 2001. Let's 
# just keep the female as it has more characteristics
metadata.drop(617, inplace=True)

# Firestorm
# keep the one that doesn't have null values
metadata.drop(260, inplace=True)

# Atom
# All atoms shown there are covered in powers df. Let's drop the row that has Atom and few null values. And add Human
# as race, plus other characteristics (all will have similar ones)
metadata.drop(50, inplace=True)
metadata.at[49, 'Race'] = "Human"
metadata.at[53, 'Race'] = "Human"
metadata.at[54, 'Race'] = "Human"
metadata.at[49, 'Race'] = "Human"
metadata.at[54, 'Height'] = 183.0
metadata.at[49, 'Height'] = 183.0
metadata.at[53, "Weight"] = 72.0

# Batman
# let's only drop the short and skinny Batman. Because he is just not.
metadata.drop(69, inplace=True)

# Toxin
# let's just keep one of them, as the second, for example
metadata.drop(673, inplace=True)

# Namor
# keep the one without null values
metadata.drop(481, inplace=True)

# Batgirl
# drop the one with null values
metadata.drop(62, inplace=True)


# In[ ]:


metadata.shape, powers.shape


# *Voila!*, 1-1 correspondence between both dataframes.

# ### Handling Null values

# `metadata` has null values, represented as either '-' for all columns except for height and weight, which is `-99.0`. 
# 
# Let's take a look at how many null values there are per each column.

# In[ ]:


metadata = metadata.replace('-', np.nan) 
metadata = metadata.replace(-99, np.nan)

metadata.isnull().sum()


# Race, which is our target in this exercise, has 232 (out of 634 rows) rows with nulls. But as for the exercise, we will ignore these. Therefore, let's drop them and then see how many null values are there still in the dataframe.

# In[ ]:


metadata.dropna(subset=['Race'], inplace=True)


# In[ ]:


metadata.isnull().sum() 


# It decreased drastically. But
# 
# - Eye color and hair color still have some percentage of values missing.
# - Skin color has too many values being null. It could be a potential feature to be removed. So let's just remove it.
# - Height and weight have quite some null values, but they can be filled by different techniques, like the median/mean of same race and gender.
# 
# Next steps:
# 
# - Look at feature distribution of each variable to see if there can be applied any quick wins for the features with many null values.

# In[ ]:


# drop Skin color because it has too many null values
metadata.drop("Skin color", axis=1, inplace=True)


# #### Handle Weight and Height null values
# 
# The idea is to set, for those rows which height and weight are null, the mean of the same gender and race (human / no-human). That way we will provide a good value yet still having more training data.
# 
# But first, **convert Race into label column with -> Human / No-Human**. The steps followed are:
# 
# - Everything that is Human, will be considered human. 
# - Those Races that are Human-\* will also be considered human.
# - All the rest, No-Human.

# In[ ]:


# transform Human- race into Human (as they are not mutations)
metadata.loc[:, "Race"] = metadata.apply(lambda x: "Human" if(x.Race.startswith("Human-")) else x.Race, axis=1)
# add label for modeling
metadata['label'] = metadata.apply(lambda x: "No-Human" if(x.Race != "Human") else x.Race, axis=1)


# #### Checking distribution of height and weight per gender and race

# In[ ]:


fig, ax = pyplot.subplots(figsize=(14,8))
sns.boxplot(x="Weight", y="label", hue="Gender", data=metadata, ax=ax)

fig, ax = pyplot.subplots(figsize=(14,8))
sns.boxplot(x="Height", y="label",  hue="Gender",data=metadata, ax=ax)


# There is a difference indeed between both genders and race. Therefore it makes sense to use such granularity. One could argue that for no-human and male, we should use median as the height and weight are quite spread. But no-human males are higher and heavier, therefore by using mean we would take outliers into account and hence preserve this relationship between both groups. 
# 
# To completely prove this, let's look at the height and weight boxplot of all No-Human races, to see if the values are more or less well distributed or they are indeed different:

# In[ ]:


fig, ax = pyplot.subplots(figsize=(12,14))
sns.boxplot(x="Height", y="Race", data=metadata[metadata.Gender == 'Male'])


# In[ ]:


fig, ax = pyplot.subplots(figsize=(12,14))
sns.boxplot(x="Weight", y="Race", data=metadata[metadata.Gender == 'Male'])


# As seen, most values lie on the same range for height, for weight the deviation is a bit higher, but I would say that it is still valid to use the mean.

# In[ ]:


# height and weight can be replaced by the mean of the same gender and race

w_means = metadata.groupby(["label", "Gender"])["Weight"].mean().unstack()
h_means = metadata.groupby(["label", "Gender"])["Height"].mean().unstack()

w_fh = w_means.loc["Human","Female"]
w_mh = w_means.loc["Human","Male"]
w_fn = w_means.loc["No-Human","Female"]
w_mn = w_means.loc["No-Human","Male"]

h_fh = h_means.loc["Human","Female"]
h_mh = h_means.loc["Human","Male"]
h_fn = h_means.loc["No-Human","Female"]
h_mn = h_means.loc["No-Human","Male"]

# Fill null values with means
metadata.loc[(metadata.label == "Human") & (metadata.Gender == "Female") & (metadata.Weight.isnull()), "Weight"] = w_fh
metadata.loc[(metadata.label == "Human") & (metadata.Gender == "Male") & (metadata.Weight.isnull()), "Weight"] = w_mh
metadata.loc[(metadata.label == "No-Human") & (metadata.Gender == "Female") & (metadata.Weight.isnull()), "Weight"] = w_fn
metadata.loc[(metadata.label == "No-Human") & (metadata.Gender == "Male") & (metadata.Weight.isnull()), "Weight"] = w_mn

metadata.loc[(metadata.label == "Human") & (metadata.Gender == "Female") & (metadata.Height.isnull()), "Height"] = h_fh
metadata.loc[(metadata.label == "Human") & (metadata.Gender == "Male") & (metadata.Height.isnull()), "Height"] = h_mh
metadata.loc[(metadata.label == "No-Human") & (metadata.Gender == "Female") & (metadata.Height.isnull()), "Height"] = h_fn
metadata.loc[(metadata.label == "No-Human") & (metadata.Gender == "Male") & (metadata.Height.isnull()), "Height"] = h_mn

# plot to see clearer differences
fig, (ax1,ax2) = pyplot.subplots(1,2, figsize=(12,6))
ax1.set_title("Weight")
ax2.set_title("Height")
w_means.plot(kind="bar", ax=ax1)
h_means.plot(kind="bar", ax=ax2)


# Check if we made a difference in terms of null values.

# In[ ]:


metadata.isnull().sum()


# As we can see, we went from 72 and 89 values for height and weight, respectivelly, to 5 and 6. And, most likely, those 5 and 6 values are due to missing Gender.
# 
# For now we could say that we can drop those rows from Gender, Height, and Weight that are null, as it will most likely be the same. Let's check it out quickly:

# In[ ]:


metadata[metadata.Gender.isnull()]


# Indeed, as I said. Let's then delete those rows.

# In[ ]:


metadata.drop(metadata[metadata.Gender.isnull()].index, axis=0, inplace=True)


# Let's check how many null values do we have missing:

# In[ ]:


metadata.isnull().sum()


# Let's get a final check of how the values of height and weight correlate to each other, to see if there are still some weird things happening:

# In[ ]:


sns.pairplot(x_vars=["Height"], y_vars=["Weight"], data=metadata, hue="label", height=10)


# In[ ]:


metadata[(metadata.Height > 400)]


# Some non-humans are really tall, but weight not too much. As they are only non-humans, let's just leave them because it won't confuse the classifier. 
# 
# What might confuse the classifier, though, it's the weight to be more than 450kg, as there are the same number of human and non-human instances. Let's take them out of the data.

# In[ ]:


metadata = metadata[(metadata.Weight < 450)]


# #### Handling Eye and Hair color
# 
# The idea would be similar as of in Height and Weight. Let's take the most common eye and hair color

# In[ ]:


for col in metadata.columns:
    if col in ("Eye color", "Hair color"):
        fig, ax = pyplot.subplots(figsize=(12,10))
        values = metadata.groupby([col, "label"]).count()['name'].unstack().sort_values(by="Human", ascending=False)
        values.plot(kind='barh', stacked=True, ax=ax)
        plt.show()


# Looking that the values are quite balanced between human and no-human between the most popular eye colors and hair colors, it will be fair to sample those null values with a random choice of the 3 most popular eye colors (green, brown, blue) and the 5 most popular hair colors (no-hair, red, blond, brown, black). 
# 
# The method would be the following
# 
# - For each entry in Eye color that is null in metadata, set a value from a random choice of an array of values (green, brown, blue). 
# - Similarly, for Hair color, but for array (no-hair, red, blond, brown, black).
# 

# In[ ]:


aux_eyes_colors = ["blue", "brown", "green"]
aux_hair_colors = ["Black", "Brown", "Blond", "Red", "No Hair"]
len_aux_eyes_colors = len(aux_eyes_colors)
len_aux_hair_colors = len(aux_hair_colors)

for ix in metadata[metadata["Eye color"].isnull()].index:
    metadata.at[ix, "Eye color"] = aux_eyes_colors[np.random.choice(len_aux_eyes_colors)]
    
for ix in metadata[metadata["Hair color"].isnull()].index:
    metadata.at[ix, "Hair color"] = aux_hair_colors[np.random.choice(len_aux_hair_colors)]


# In[ ]:


for col in metadata.columns:
    if col in ("Eye color", "Hair color"):
        fig, ax = pyplot.subplots(figsize=(12,10))
        values = metadata.groupby([col, "label"]).count()['name'].unstack().sort_values(by="Human", ascending=False)
        values.plot(kind='barh', stacked=True, ax=ax)
        plt.show()


# Practically no difference between the barplots shown before the methodology.
# 
# Let's now finally see what is left to handle in terms of null values:

# In[ ]:


metadata.isnull().sum()


# Let's just forget about the remaining rows as the number is insignificant.

# In[ ]:


metadata.drop(metadata[metadata.Publisher.isnull()].index, axis=0, inplace=True)
metadata.drop(metadata[metadata.Alignment.isnull()].index, axis=0, inplace=True)


# In[ ]:


metadata.isnull().sum()


# **Done!** Let's now continue by merging `powers` Dataframe and our brand new cleaned `metadata` dataframe.

# ### Categorical Variables and One-Hot Encoding
# 
# High cardinality categorical features, we will just convert them to their corresponding code (0,1,2,...).
# 
# - Gender
# - Alignment
# 
# For low cardinality features, one-hot encoding will be implemented.
# 
# - Eye color
# - Hair color
# - Publisher
# 

# In[ ]:


metadata = metadata.drop(['Race'], axis=1)

high_card = ["Gender", "Alignment"]
low_card = ["Eye color", "Hair color", "Publisher"]

for hc in high_card:
    one_hot = pd.get_dummies(metadata[hc])
    metadata.drop(hc, axis=1, inplace=True)
    metadata = metadata.join(one_hot)

for lc in low_card:
    metadata[lc] = metadata[lc].astype('category').cat.codes

# transform label into 0 (Human) or 1 (No-Human)
metadata['label'] = metadata['label'].astype('category').cat.codes


# In[ ]:


# transform powers data into 0,1 binary features
cols = powers.select_dtypes(['bool']).columns
for col in cols:
    powers[col] = powers[col].astype(int)


# In[ ]:


metadata.head()


# In[ ]:


powers.head()


# ### All data together

# In[ ]:


heroes = pd.merge(metadata, powers, how='inner', left_on = 'name', right_on = 'hero_names')

heroes.drop(["hero_names","name"], axis=1, inplace=True)

powers_cols = powers.columns.drop("hero_names")
metadata_cols = metadata.columns.drop("name")


# In[ ]:


heroes.shape


# In[ ]:


heroes.head()


# In[ ]:


# store dataframe
metadata.to_pickle("metadata.pkl")
powers.to_pickle("powers.pkl")
heroes.to_pickle("heroes.pkl")


# In[ ]:


# load back again
metadata = pd.read_pickle("metadata.pkl")
powers = pd.read_pickle("powers.pkl")
heroes = pd.read_pickle("heroes.pkl")


# ## Modeling
# 
# ### Plan
# 
# #### Target
# 
# The idea is to use several classifiers to predict whether the hero is Human or No-Human. 
# 
# #### Input Data
# 
# As **input data**, I will use the processed and cleaned `heroes` dataframe, which contains 388 superheroes and 177 different characteristics of each. The target is balanced between rows, but the main problem I expect to happen is that the data is highly complex (high dimensional) in comparison with the amount of examples we possess. We will tackle that issue later on.
# 
# We also standardize the data as it is shown that helps the model optimization in terms of faster convergence due to better gradient flow, and a possible increase in performance. Standardizing data means scaling the feature values so that the resulting data has zero mean and unit variance.
# 
# #### Evaluation
# 
# As the target is quite balanced, it is okay to use actual accurcacy as a main way to evaluate our models through cross validation. As a further way of evaluation, it would be interesting to look at the confusion matrix to understand a bit better what is our model doing.
# 
# #### Models
# 
# As a baseline, we use **Logistic Regression** as it is easy to implement, understand, and should provide an already decent predictive power.
# 
# We will continue with using **SVM**, **Random Forest Classifier**, and **XGBoost**. The reason behind using those is because they are known and proven to be the most powerful conventional machine learning algorithms. We won't use neural networks in this dataset due to the obvious little amount of data we have.
# 
# #### Feature Reduction
# 
# As our dataset has little amount of examples in comparison with the amount of features, it would be interesting to try to reduce the complexity of the dataset, to see if we predict better or not. One of the main assumptions behind dimensionality reduction is that the high complex datasets capture a lot of unnecessary information which can be reduced into a arbitrable amount of principal components that explain the variance of the underlying data, with the ideal case that the model will be trained with more rellevant information. To get a glampse of whether the dimensionality reduction will work or not, we can use t-SNE algorithm to plot high dimensionality data into two dimensions, colored by the label value (Human / Not-Human) and see if we can already infer a way to discriminate our dataset.
# 
# For the sake of reducing dimensionality, we will use the classic PCA with different amount of components and study what is the impact that it does to our dataset.

# In[ ]:


from sklearn import preprocessing

X = heroes.drop(["label"], axis=1).values
y = heroes["label"].values

X = preprocessing.scale(X)

print( "X - training data shape ", X.shape)
print( "y - label ", y.shape )


# ### Training classifiers

# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier


# In[ ]:


# Initialize a stratified split of our dataset for the validation process
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# In[ ]:


models = ["LogReg", "SVM", "RF", "XGB"]

for model in models:
    print( "Training ", model)
    if model == "LogReg":
        clf = LogisticRegression(random_state=0, solver='liblinear')
    elif model == "SVM":
        clf = svm.SVC(kernel='linear',C=1)
    elif model == 'RF':
        clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=1)
    else:
        warnings.filterwarnings(action='ignore', category=DeprecationWarning)
        clf = XGBClassifier(n_estimators=50, max_depth=6)
    
    results = cross_val_score(clf, X, y, cv=5).mean()
    print( model, " CV accuracy score: {:.2f}%".format(results.mean()*100) )


# As a first round for classifers, we already get a 76.77% accuracy. It is not bad, but we can do better. Let's grid search best combination of parameters for Random Forest and XGBoost as those classifiers showed the best performance so far.

# In[ ]:


# Random Forest optimization
clf_rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=1)

rf_params = {'max_features': [4, 7, 10, 13], 
             'min_samples_leaf': [1, 3, 5, 7], 
             'max_depth': [5,8,10,15], 
             "n_estimators": [50, 100] }

gcv = GridSearchCV(clf_rf, rf_params, n_jobs=-1, cv=skf, verbose=1)
gcv.fit(X, y)


# In[ ]:


gcv.best_estimator_, gcv.best_score_


# In[ ]:


# XGBoost optimization
clf_xgb = XGBClassifier(n_estimators=50, max_depth=8, random_state=1)

xgb_params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 8, 10],
        'n_estimators' : [50,100]
        }

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
gcv_xgb = GridSearchCV(clf_xgb, xgb_params, n_jobs=-1, cv=skf, verbose=1)
gcv_xgb.fit(X, y)


# In[ ]:


gcv_xgb.best_estimator_, gcv_xgb.best_score_


# By fine tuning the parameters of both models we have got a really nice improvement in both, RF and XGB with a curiosly same performance of **78.86%**.

# ### Feature Importance
# 
# Of the best model seen, which is XGBoost.
# 
# Code obtained in kernel published by sendohchang https://www.kaggle.com/sendohchang/classify-superhero-is-a-human 

# In[ ]:


importances = gcv_xgb.best_estimator_.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importancies
features = dict()
count = 1
for col in heroes.drop("label",axis=1).columns:
    index = "f"+str(count)
    features[index] = col
    count+=1

num_to_plot = 20
feature_indices = [ind+1 for ind in indices[:num_to_plot]]
top_features = list()
# Print the feature ranking
print("Feature ranking:")
  
for f in range(num_to_plot):
    print("%d. %s %f " % (f + 1, 
            features["f"+str(feature_indices[f])], 
            importances[indices[f]]))
    top_features.append(features["f"+str(feature_indices[f])])
plt.figure(figsize=(15,5))
plt.title(u"Feature Importance")
bars = plt.bar(range(num_to_plot), 
               importances[indices[:num_to_plot]],
       color=([str(i/float(num_to_plot+1)) 
               for i in range(num_to_plot)]),
               align="center")
ticks = plt.xticks(range(num_to_plot), 
                   feature_indices)
plt.xlim([-1, num_to_plot])
plt.legend(bars, [u''.join(features["f"+str(i)]) 
                  for i in feature_indices]);


# ### Dimensionality Reduction

# In[ ]:


from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, n_iter=1000)
tsne_results = tsne.fit_transform(X)


# In[ ]:


df_tsne = pd.DataFrame(data=tsne_results, columns=["tsne1", "tsne2"])
df_tsne['label'] = y

sns.pairplot(x_vars=["tsne1"], y_vars=["tsne2"], data=df_tsne, hue="label", height=10)


# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


models = ["LogReg", "SVM", "RF", "XGB"]
reductions = [20, 30, 40, 50, 70, 100, 150]

for red in reductions:
    
    print( "Applying PCA on ", red, " components")
    pca = PCA(n_components=red)
    X_reduced = pca.fit_transform(X)

    for model in models:
        print( "Training ", model )
        if model == "LogReg":
            clf = LogisticRegression(random_state=0, solver='liblinear')
        elif model == "SVM":
            clf = svm.SVC(kernel='linear',C=1)
        elif model == 'RF':
            clf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=1)
        else:
            clf = XGBClassifier(n_estimators=50, max_depth=6)

        results = cross_val_score(clf, X_reduced, y, cv=5).mean()
        print( model, " CV accuracy score: {:.2f}%".format(results.mean()*100) )
        
    print( "\n\n" )


# In[ ]:


# let's try to train xgboost with 50 PCA component

pca_50 = PCA(n_components=50)
X_reduced_p50 = pca_50.fit_transform(X)

# XGBoost optimization
clf_xgb = XGBClassifier(n_estimators=50, max_depth=8, random_state=1)

xgb_params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 8, 10],
        'n_estimators' : [50]
        }

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
gcv_xgb = GridSearchCV(clf_xgb, xgb_params, n_jobs=-1, cv=skf, verbose=1)
gcv_xgb.fit(X_reduced_p50, y)


# In[ ]:


gcv_xgb.best_estimator_, gcv_xgb.best_score_

