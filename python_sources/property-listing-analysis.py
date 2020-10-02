#!/usr/bin/env python
# coding: utf-8

# The goal of this project is to scrape and analyze property listing information from the web. I am interested in residential property prices in Kuala Lumpur, Malaysia, as listed on [iProperty](https://www.iProperty.com.my). The focus is on identifying how properties vary between neighborhoods, i.e. descriptive statistics. I will also attempt to build a predictive model for prices, but this will most likely fail due to the limited number of features I scrape.

# # Imports

# In[ ]:


import numpy as np
np.random.seed(101)
import requests
import time
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model
import sklearn.feature_selection
import sklearn.preprocessing
import sklearn.metrics
import keras.models
import keras.layers


# # Data Preprocessing
# The raw data that I collected needs to be preprocessed and the columns cleaned.

# In[ ]:


properties = pd.read_csv("../input/data_kaggle.csv")


# In[ ]:


properties.head()


# ## Price
# As a first step, I'll remove all entries with missing prices as the price will be one of the target features to predict down the road.

# In[ ]:


properties = properties.loc[~properties["Price"].isna()]


# Next, I want to make sure that every remaining entries matches the same format of "RM" followed by commas and digits.

# In[ ]:


incorrect_entries = np.sum(~properties["Price"].str.match(r"RM [0-9,]*$"))
print("There are {} entries in the wrong format.".format(incorrect_entries))


# In[ ]:


# Strip the price of the "RM" as well as commas
def strip_price(text):
    text = text.replace("RM", "")
    text = text.replace(",", "")
    text = text.strip()
    return int(text)
    
properties["Price"] = properties["Price"].apply(strip_price)


# ## Location
# 
# A common issue is capitalization, so let's make everything lower case. Additionally, let's remove the city name as I explicitly only scraped entries from Kuala Lumpur.

# In[ ]:


properties["Location"] = properties["Location"].str.lower()
properties["Location"] = properties["Location"].str.replace(r", kuala lumpur$", "")


# A look at the unique location values shows that they are reasonably standardized. 

# In[ ]:


sorted(properties["Location"].unique())


# A bigger issue are regions with very few properties posted. The frequency distribution, plotted logarithmically, looks as follows. A significant number of regions have very few properties listed, making them difficult to work with. I would need to have relative geographical locations of each neighborhood to properly clean the location data. As an initial step, I simply remove entries in locations with fewer than $100$ properties listed.

# In[ ]:


properties["Location"].value_counts().plot(logy=True);


# In[ ]:


significant_locations = properties["Location"].value_counts()[
    properties["Location"].value_counts() >= 100].index

properties = properties.loc[np.isin(properties["Location"], significant_locations)]


# A new look at the locations shows that there are no more ambiguous names.

# In[ ]:


sorted(properties["Location"].unique())


# ## Number of Rooms
# The number of rooms contains some irregularities. For example, it is common for rooms to be listed as N+M instead of the total number of rooms. I want to clean the `Rooms` column and introduce a numerical equivalent.

# In[ ]:


sorted(properties["Rooms"].unique().astype(str))


# In[ ]:


def convert_room_num(rooms):
    try:
        if rooms.endswith("+"):
            return int(rooms[:-1])
        if re.search("[0-9]+\+[0-9]+", rooms) is not None:
            tmp = rooms.split("+")
            return int(tmp[0]) + int(tmp[1])
        if rooms == "20 Above":
            return 20
        if rooms == "Studio":
            return 1
        return int(rooms)
    except AttributeError:
        return rooms

properties["Rooms Num"] = properties["Rooms"].apply(convert_room_num)


# In[ ]:


properties["Rooms Num"].value_counts(dropna=False)


# # Property Type
# There are several different property types that are typical for Malaysia. A brief glance at the full list  of property types seems overwhelming at first.

# In[ ]:


properties["Property Type"].value_counts()


# I can greatly simplify the property types, however, by trimming information. For example, there are many different variations of the Terrace/Link houses that can be grouped together. I create a new category of only the property type "super groups".

# In[ ]:


def simplify_property_type(prop_type):
    super_types = [
        "Terrace/Link House", "Serviced Residence", "Condominium", 
        "Semi-detached House", "Bungalow", "Apartment", "Townhouse", 
        "Flat", "Residential Land", "Cluster House"]
    for super_type in super_types:
        if re.search(super_type, prop_type, flags=re.IGNORECASE) is not None:
            return super_type
    
    return prop_type

properties["Property Type Supergroup"] = properties["Property Type"].apply(simplify_property_type)


# In[ ]:


properties["Property Type Supergroup"].value_counts(dropna=False)


# ## Furnishing
# The furnishing column thankfully doesn't require any cleaning.

# In[ ]:


properties["Furnishing"].value_counts(dropna=False)


# ## Size
# The size apparently always has the same structure:
# 
#     [Built-up/Land area] : [Value] sq. ft.

# In[ ]:


properties[["Size"]].sample(25)


# So I will create two new column that indicate whether this is built-up or land area and store the actual square footage.

# In[ ]:


def split_size(val, index=0):
    try:
        return val.split(":")[index].strip()
    except AttributeError:
        return val
    
properties["Size Type"] = properties["Size"].apply(split_size, index=0)
properties["Size Num"] = properties["Size"].apply(split_size, index=1)


# In[ ]:


properties["Size Type"].value_counts(dropna=False)


# Now I need to strip the new `Size_num` column and convert it to a numerical value.

# In[ ]:


def convert_size_num(size):
    # Attempt to trim the numbers down. Most of this is done explicitly without
    # regex to avoid incorrect trimming, which would lead to the concatenation
    # of numbers. I would rather have missing values than incorrectly cleaned
    # numbers.
    try:
        # If it's not in square feet then I don't want to deal with all
        # possible conversions for now.
        if re.search(r"sq\.*\s*ft\.*", size) is None:
            return None
    
        size = size.replace(",", "")
        size = size.replace("'", "")
        size = size.replace("sq. ft.", "")
        size = size.replace("sf", "")
        size = size.strip()
        size = size.lower()
        
        add_mult_match = re.search(r"(\d+)\s*\+\s*(\d+)\s*(?:x|\*)\s*(\d+)", size)
        if add_mult_match is not None:
            return int(add_mult_match.groups()[0]) + (
                int(add_mult_match.groups()[1]) * 
                int(add_mult_match.groups()[2]))
        
        mult_match = re.search(r"(\d+)\s*(?:x|\*)\s*(\d+)", size)
        if mult_match is not None:
            return int(mult_match.groups()[0]) * int(mult_match.groups()[1])
        
        return int(size)
    # If any of the above doesn't work, just turn it into None/NaN
    # We want to guarantee this column is numeric
    except:
        return None
        
properties["Size Num"] = properties["Size Num"].apply(convert_size_num)


# Cleaning the property sizes introduced only a small number of additional missing values.

# In[ ]:


print("Properties with missing raw size data: {}".format(properties["Size"].isna().sum()))
print("Properties with missing size type data: {}".format(properties["Size Type"].isna().sum()))
print("Properties with missing size num data: {}".format(properties["Size Num"].isna().sum()))


# I will synchronize the missing values between the `Size Type` and `Size Num` columns.

# In[ ]:


properties.loc[properties["Size Num"].isna(), "Size Type"] = None


# In[ ]:


properties.loc[:, "Size Type"].value_counts(dropna=False)


# ## Other columns
# The number of bathrooms and car parks are standardized and do not require any further cleaning.

# In[ ]:


properties["Bathrooms"].value_counts(dropna=False)


# In[ ]:


properties["Car Parks"].value_counts(dropna=False)


# ## Price per Area / Room
# As a last step, I want to introduce the price per area and price per rooms as features

# In[ ]:


properties["Price per Area"] = properties["Price"] / properties["Size Num"]
properties["Price per Room"] = properties["Price"] / properties["Rooms Num"]


# ## Save Preprocessed Data

# In[ ]:


properties.to_csv("Properties_preprocessed.csv")


# # Data Exploration
# 
# The most immediate question will be how properties differ between neighborhoods in their characteristics.

# In[ ]:


def plot_by_neighborhood(feature, formatting, factor=1):
    df = properties.groupby("Location")[feature].median().sort_values(ascending=False).reset_index()
    shift = 0.1 * (df[feature].max() - df[feature].min())
    df_sizes = properties.groupby("Location").size()[df["Location"]]

    fig = sns.catplot(
        data=df, x=feature, y="Location", kind="bar", 
        color="darkgrey", height=10, aspect=0.8)

    for index, row in df.iterrows():
        fig.ax.text(
            row[feature] + shift, row.name, formatting.format(row[feature] / factor), 
            color='black', ha="center", va="center")

    fig.ax.get_xaxis().set_visible(False);
    fig.despine(left=True, bottom=True)
    fig.ax.tick_params(left=False, bottom=False);
    fig.set_ylabels("");


# ## Total Prices per Neighborhood

# In[ ]:


plot_by_neighborhood(feature="Price", formatting="RM {:.2f}m", factor = 1e6)


# ## Price per Square Foot per Neighborhood

# In[ ]:


plot_by_neighborhood(feature="Price per Area", formatting="RM {:.2f}k", factor = 1e3)


# ## Price per Room per Neighborhood

# In[ ]:


plot_by_neighborhood(feature="Price per Room", formatting="RM {:.2f}k", factor = 1e3)


# ## Property Size per Neighborhood

# In[ ]:


plot_by_neighborhood(feature="Size Num", formatting="{:.2f}k sq. ft.", factor = 1e3)


# ## Rooms per Neighborhood

# In[ ]:


plot_by_neighborhood(feature="Rooms Num", formatting="{:.2f}", factor = 1)


# ## Number of Properties per Neighborhood

# In[ ]:


df = properties.groupby("Location").size().sort_values(ascending=False).reset_index()
shift = 0.05 * (df[0].max() - df[0].min())
df_sizes = properties.groupby("Location").size()[df["Location"]]

fig = sns.catplot(
    data=df, x=0, y="Location", kind="bar", 
    color="darkgrey", height=10, aspect=0.8)

for index, row in df.iterrows():
    fig.ax.text(
        row[0] + shift, row.name, row[0], 
        color='black', ha="center", va="center")

fig.ax.get_xaxis().set_visible(False);
fig.despine(left=True, bottom=True)
fig.ax.tick_params(left=False, bottom=False);
fig.set_ylabels("");


# ## Most common Property Type per Neighborhood

# In[ ]:


# Extract property type and turn it into a two-column data frame
df = properties.loc[~properties["Property Type Supergroup"].isna()].groupby(
    "Location")["Property Type Supergroup"].value_counts()
df.name = "Value"
df = df.reset_index().pivot(index="Location", columns="Property Type Supergroup")
df.columns = df.columns.droplevel(0)
df = df.fillna(0)

# normalize rows to see relative amount of properties in each neighborhood 
df_norm = df.apply(lambda x: x / x.sum(), axis=1)

fix, ax = plt.subplots(figsize=(12, 12))
hmap = sns.heatmap(
    df_norm, square=True, vmin=0, cmap="Reds", ax=ax, cbar=False)
hmap.set_ylabel(None);
hmap.set_xlabel(None);


# ## Land vs. Built-Up Area per Neighborhood

# In[ ]:


df = properties[["Location", "Size Type", "Size Num"]].groupby(
    ["Location", "Size Type"]).median().reset_index()
fig = sns.catplot(
    data=df, x="Size Num", y="Location", kind="bar", 
    hue="Size Type", height=20, aspect=0.4);

fig.despine(left=True)
fig.ax.tick_params(left=False);
fig.set_ylabels("");


# # Predictive Modelling

# ## Preparing the Data
# As a short exercise in predictive modelling, I want to try to predict the price of a property based on the characteristics listed here. Due to the heterogeneity of the data, I will only look at a subset of the property listings to reduce the number of potentially confounding factors. In particular, I will:
# 
# - Look only at entries with "built-up" area listed. This is because built-up size and land area are, strictly speaking, two different features.
# - Look only at entries without missing values for features (see below for a detailed description of which features I use).
# 
# I will also be selective about the features I include in the model. As categorical features would have to be converted to dummy features, e.g. the `Rooms` feature would be converted to boolean "has_3_rooms", "has_3+1_rooms", etc., I will try to use numerical versions of the features where possible. Specifically, the following features will _not_ be used:
# 
# - `Rooms`, which will be replaced with `Rooms Num`
# - `Size`, which will be replaced with `Size Num`
# - `Size Type`, as this will always be "built-up" in the reduced data frame)
# - `Property Type`, as there are simply too many variants. I instead use `Property Type Supergroup`.
# 
# This means our model will consider the following features:
# 
# - `Location` (converted to binary dummy features)
# - `Bathrooms`
# - `Car Parks`
# - `Furnishing`
# - `Rooms Num`
# - `Property Type Supergroup`
# - `Size Num`
# 
# And the model will be trained to predict any of the three price columns, `Price`, `Price per Area`, and `Price per Room`.
# 
# Lastly, I will make the assumption that a missing entries for `Car Parks` is 0. While not necessarily true, it is likely to be the case for many entries. However, I will not make the same assumption for `Bathrooms`, as a (built up) property will have at least one bathroom.

# In[ ]:


# Remove entries with "land area" in the "Size Type" column
Xy = properties.loc[properties["Size Type"] == "Built-up"]

# Keep only the relevant features
Xy = Xy.loc[:, [
    "Location", "Bathrooms", "Car Parks", "Furnishing", 
    "Rooms Num", "Property Type Supergroup", "Size Num", 
    "Price", "Price per Area", "Price per Room"]]

# Fill missing Car Parks feature values
Xy.loc[:, "Car Parks"] = Xy["Car Parks"].fillna(0)

# Remove entries with missing values
Xy = Xy.loc[Xy.isna().sum(axis=1) == 0]

# Specifically remove entries with "Unknown" furnishing status
Xy = Xy.loc[Xy["Furnishing"] != "Unknown"]

# Convert to dummy features
Xy = pd.get_dummies(Xy)


# In[ ]:


print("Shape of data frame: {}".format(Xy.shape))


# The data frame now consists of only numerical features:

# In[ ]:


print("Data frame DTYPES:")
for dtype in Xy.dtypes.unique():
    print(" - {}".format(dtype))


# ## Feature Selection
# Beyond the intial preprocessing, I obviously want to perform feature selection as well. Some features may be heavily correlated.

# ### Outlier removal
# The first step is to remove outliers from the original numerical features. Until now, I've used a robust aggregator (the median) and outliers have been irrelevant, but they can become a thorn in our side for predictive modelling.
# 
# I remove `Size Num` outliers heuristically. The smallest reasonable value, as can be seen from the data, corresponds to $250$ square feet. This corresponds to a small studio apartment. Ergo I use this as the lower threshold for potential values.

# In[ ]:


Xy["Size Num"].sort_values().head(10)


# On the opposite end of the spectrum, there appear to be several unreasonably large properties ($820000$ square feet corresponds to the approximate size of the Louvre museum in Paris, France). I heuristically set the cutoff at $20000$ square feet for the maximum size of a property.

# In[ ]:


Xy["Size Num"].sort_values(ascending=False).head(20)


# In[ ]:


Xy = Xy.loc[Xy["Size Num"].between(250, 20000)]


# From the remaining three originally numerical columns, `Bathrooms`, `Car Parks`, and `Rooms Num`, I trim the top and bottom $0.1\%$ of all entries.

# In[ ]:


selectors = []
for feature in ["Bathrooms", "Car Parks", "Rooms Num"]:
    selectors.append(Xy[feature].between(
        Xy[feature].quantile(0.001), 
        Xy[feature].quantile(0.999)))

Xy = Xy.loc[(~pd.DataFrame(selectors).T).sum(axis=1) == 0]


# All further feature selection will be performed on a subset of the data that will _not_ be used for training the model itself to avoid overfitting.

# In[ ]:


Xy, Xy_feature_selection = sklearn.model_selection.train_test_split(
    Xy, test_size=0.25, random_state=101)


# In[ ]:


Xy.shape


# In[ ]:


Xy_feature_selection.shape


# ### Feature Scaling
# First, the original numerical features must be scaled (the binary dummy features don't need to be scaled).

# In[ ]:


fig, ax = plt.subplots(2, 2, figsize=(10, 10));
sns.countplot(data=Xy_feature_selection, x="Bathrooms", ax=ax[0, 0], color="darkgrey");
ax[0, 0].set_title("Bathrooms");
sns.countplot(data=Xy_feature_selection, x="Car Parks", ax=ax[0, 1], color="darkgrey");
ax[0, 1].set_title("Car Parks");
sns.countplot(data=Xy_feature_selection, x="Rooms Num", ax=ax[1, 0], color="darkgrey");
ax[1, 0].set_title("Rooms Num");
sns.distplot(a=Xy_feature_selection["Size Num"], bins=50, ax=ax[1, 1], color="darkgrey");
ax[1, 1].set_title("Size Num");


# As none of the features seem to be normally distributed, I will simply scale them to lie between 0 and 1. Note that the data sets for training and feature selection are scaled separately!

# In[ ]:


cols = ["Bathrooms", "Car Parks", "Rooms Num", "Size Num"]
Xy_feature_selection[cols] = sklearn.preprocessing.MinMaxScaler().fit_transform(
    Xy_feature_selection[cols])
Xy[cols] = sklearn.preprocessing.MinMaxScaler().fit_transform(Xy[cols])


# ### Feature Correlation
# I look at the correlation between the initial numerical features to determine if they can be pruned.

# In[ ]:


hm_cmap = sns.diverging_palette(240, 0, s=99, l=50, as_cmap=True)
df = Xy_feature_selection[["Bathrooms", "Car Parks", "Rooms Num", "Size Num"]].corr()
sns.heatmap(data=df, vmin=-1, vmax=1, cmap=hm_cmap, annot=df, annot_kws={"size": 20});


# Based on the above correlation matrix, the features `Bathrooms` and `Rooms Num` both correlate very strongly with `Size Num` and can be safely removed.

# Remove the actual features from the dataset(s)

# In[ ]:


Xy = Xy.drop(["Bathrooms", "Rooms Num"], axis=1)
Xy_feature_selection = Xy_feature_selection.drop(["Bathrooms", "Rooms Num"], axis=1)


# In addition to the features, I also want to look at the potential target variables and how they correlate.

# In[ ]:


df = Xy_feature_selection[["Price", "Price per Area", "Price per Room"]].corr()
sns.heatmap(
    df, vmin=-1, vmax=1, cmap=hm_cmap, 
    annot=np.round(df, 2), annot_kws={"size": 20})


# `Price per Area` and `Price per Room` correlate very strongly so that it makes little sense to retain both. I consequently remove `Price per Room`.

# In[ ]:


Xy = Xy.drop("Price per Room", axis=1)
Xy_feature_selection = Xy_feature_selection.drop("Price per Room", axis=1)


# ## Modelling

# Split data into training and test set

# In[ ]:


Xy_train, Xy_test = sklearn.model_selection.train_test_split(Xy, test_size=0.2, random_state=101)
X_train = Xy_train.drop(["Price", "Price per Area"], axis=1)
y_train = Xy_train[["Price", "Price per Area"]]
X_test = Xy_test.drop(["Price", "Price per Area"], axis=1)
y_test = Xy_test[["Price", "Price per Area"]]


# Define convenience function to train and test a `scikit-learn` model.

# In[ ]:


def train_and_test_model(
        model, X_train=X_train, y_train=y_train, 
        X_test=X_test, y_test=y_test, **kwargs):
    model.fit(X_train, y_train, **kwargs)
    y_pred = model.predict(X_test)
    r2 = sklearn.metrics.r2_score(y_true=y_test, y_pred=y_pred)
    return model, r2


# In[ ]:


model, r2 = train_and_test_model(
    model = sklearn.linear_model.LinearRegression(), 
    X_train=X_train, y_train=y_train["Price"], 
    X_test=X_test, y_test=y_test["Price"])
print("R^2 for prediction of 'Price': {:.2f}".format(r2))

model, r2 = train_and_test_model(
    model = sklearn.linear_model.LinearRegression(), 
    X_train=X_train, y_train=y_train["Price per Area"], 
    X_test=X_test, y_test=y_test["Price per Area"])
print("R^2 for prediction of 'Price per Area': {:.2f}".format(r2))


# Neither of the targets can be predicted with a satisfying accuracy. This is most likely due to the overwhelming number of sparse binary features. A neural network, unfortunately, also does not perform satisfactorily.

# In[ ]:


def make_fcn_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=32, activation="relu", input_shape=(X_train.shape[1],)))
    model.add(keras.layers.Dense(units=32, activation="relu"))
    model.add(keras.layers.Dense(units=32, activation="relu"))
    model.add(keras.layers.Dense(units=1, activation="relu"))
    model.compile(loss="mse", optimizer="Adam")
    return model


# In[ ]:


model, r2 = train_and_test_model(
    model = make_fcn_model(), 
    X_train=X_train, y_train=y_train["Price"], 
    X_test=X_test, y_test=y_test["Price"], 
    batch_size=8, epochs=10, verbose=0)
print("R^2 for prediction of 'Price': {:.2f}".format(r2))

model, r2 = train_and_test_model(
    model = make_fcn_model(), 
    X_train=X_train, y_train=y_train["Price per Area"], 
    X_test=X_test, y_test=y_test["Price per Area"], 
    batch_size=8, epochs=10, verbose=0)
print("R^2 for prediction of 'Price per Area': {:.2f}".format(r2))


# # Conclusion
# Neither a linear model nor a neural network perform sufficiently well in predicting property prices. This is unsurprising, of course, as properties are much more complex than the features captured here indicate. In particular, the summary statistics seen in the EDA show that the neighborhood alone accounts for massive differences in property prices. The remaining features used here, e.g. the property size or the number of rooms, cannot account for all variance within. A proper price prediction model for new properties would therefore require more detailed features as scraped from the property listing itself rather than just the overview page as I've done here.
