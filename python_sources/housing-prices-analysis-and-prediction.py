#!/usr/bin/env python
# coding: utf-8

# # Housing \ School \ Venues Analysis and Prediction of GTA

# ## Table of contents
# * [1. Introduction: Business Problem](#introduction)
# * [2. Data](#data)
# * [3. Methodology](#methodology)
# * [4. Data Analysis & Visualization](#Analysis&Visualization)
# * [5. Modeling](#Modeling)
# * [6. Results and Discussion](#results)
# * [7. Conclusion](#conclusion)

# 
# 
# ## 1. Introduction: Business Problem <a name="introduction"></a>

# As a resident of the Greater Toronto Area (GTA) for more than 10 years, I'd like to choose GTA data as my Applied Data Science Capstone Project. The basic idea is using the **Venues Data** and **School Rankings** to predict **Housing Price**. Also, look into some **relationship** or **patterns** between different factors.
# 
# Toronto is the largest city in Canada, however, compare with those largest cities in the world, we have only 6 million people live in it, most of which lives and works in the south area. The population density and diversity are our unique features, that makes some of the parameters in our project needs to be customized.
# 
# When we have all of the data and the models, we can create some visualization graphs to check those patterns and compare the performance among different models. We can also create a map to map each district is clustered according to the features like school ranking and venue density.

# ## 2. Data <a name="data"></a>

# At first, I took me hours to build a quick draft model: with only the data I gathered from some realtor website using web scraping method, then use foursquare API to combine each of the house records with Venues Data. However, the accuracy of the performance does not look great. 
# 
# The biggest issue is unlike in the US, here in Canada all of the sold price and most of the important detail information such as build year, sqrt, land size, property tax, are **not opened to the public**, so there's no good solution to get all of the data I required through open source.
# 
# So the next step is to try to find other related could affect the housing price, in order to do so, I went through other websites including official school ranking for 2017-2018 provided by Fraser Institute, since just as many of other parents, the ranking of the school is one of the biggest concerns when we purchased our own house. Also, I chose a listing website for an approximate size and type for each of the property.
# 
# Please note there's two way to get latitude and longitude information: I tried **Google API** and web scrapping, both worked.
# 
# Eventually, here the key datasets I used:
# * **School Ranking** Dataset - Obtained from the **Fraser Institute** website through **web scraping** methods
#     * The name of schools
#     * Cities of schools located
#     * 2017-18 Rating of Elementary Schools
#     * Postcodes of schools
#     * Addresses of schools
# 
# * **Housing Info** Dataset - Obtained from **residences listing websites** through **web scraping** methods
#     * Addresses of houses
#     * Number of bedrooms for each house
#     * Number of bathrooms for each house
#     * Postcodes of houses
#     * Latitudes
#     * Longitudes
#     * The names of Neighborhood
#     * The type of the residences: House/Townhouse/Condo
#     * Listing prices
#     * Approximate sizes: Better than nothing, since I couldn't find an easy way to get the exact size of each residence
# 
# * **Venues Data** Dataset - Obtained through **Foursquare API**
#     * Counts of Venues closed to the Neighborhood
#     * The frequency of each Venues Category such as Office, Bus Stop, Pizza Place, Coffee, Chinese Restaurant, Italian Restaurant, etc.

# ### 2.1 Extract Data

# #### Import libraries

# In[ ]:


import numpy as np  # library to handle data in a vectorized manner
import pandas as pd  # library for data analsysis

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

from bs4 import BeautifulSoup

import json  # library to handle JSON files

#!conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import (
    Nominatim,
)  # convert an address into latitude and longitude values

import requests  # library to handle requests
from urllib import request

from pandas.io.json import json_normalize  # tranform JSON file into a pandas dataframe

# Matplotlib/Seaborn and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import pyplot as plt
import seaborn as sns

# import sklearn functions
from sklearn.cluster import KMeans

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold,
    GridSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn import neighbors

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# create a Graphviz file
from sklearn.tree import export_graphviz
from IPython.display import Image

# import StringIO, pydot
# import pylab, pydotplus
from sklearn.externals.six import StringIO

# import xgboost
import xgboost as xgb

# import keras functions
import keras
from keras import Sequential, regularizers
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.utils import np_utils

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium  # map rendering library

import time

print("Libraries imported.")


# #### Load School Dataset from CSV file

# In[ ]:


# The CSV files is exported from my Web Scraping of School informations.ipynb file
school = pd.read_csv("../input/school_scores.csv")
school = school.iloc[:, 1:]

# Since the origin csv file includes all of the schools in Ontario, here we need to define the Cities belongs to GTA
# Due to project objective, I didn't selected all of the cities in GTA
GTA = [
    "Toronto",
    "Mississauga",
    "Ajax",
    "Brampton",
    "Unionville",
    "Oakville",
    "Richmond Hill",
    "Markham",
    "Oshawa",
    "Whitby",
    "Caledon",
    "Pickering",
    "Maple",
    "Thornhill",
    "Woodbridge",
    "Aurora",
    "Vaughan",
    "Concord",
    "Stouffville",
    "Milton",
    "Newmarket",
    "King City",
    "Durham",
]
school = school[school["school_city"].isin(GTA)]
school.head()


# In[ ]:


# check the size of dataset
school.shape  # totally 1030 schools and five columns


# #### Load House Dataset from CSV file

# In[ ]:


# The CSV files is exported from my Web Scraping of House informations.ipynb file
house = pd.read_csv("../input/house.csv")
house = house.iloc[:, 1:]
house.head()


# In[ ]:


# Check the size of the dataset and those doesn't have the size data
print("shape of table house", house.shape)
print("shape of na values", house[house["size"].isna()].shape)
print("shape of not na values", house[house["size"].notna()].shape)


# ### 2.2 Transform Data

# #### Manipulate House Dataset

# In[ ]:


house["size"].unique()[:5]


# In[ ]:


# Manipulate size column
house2 = house["size"].str.strip("sqft").str.strip("<").str.strip().str.split("-")

for i in range(len(house2)):
    try:
        house2[i] = np.mean([float(i) for i in house2[i]])
    except TypeError:
        house2[i] = float(house2[i])
    except ValueError:
        house2[i] = float(house2[i][0].strip("+"))

house["size"] = house2


# In[ ]:


# drop outliers
house.drop(house[house["size"] < 300].index, inplace=True)


# In[ ]:


# Replace missing values of size column to the mean of the sizes with the same type and number of bedroom
housefillna = pd.Series()
ptypes = house.ptype.unique()
num_bedrooms = house.num_bedroom.unique()
for ptype in ptypes:
    for num_bedroom in num_bedrooms:
        try:
            batch = (house.ptype == ptype) & (house.num_bedroom == num_bedroom)
            h = house[batch]["size"].fillna(house[batch]["size"].mean())
            housefillna = housefillna.append(h)
        except:
            housefillna = housefillna.append(house[batch]["size"])

print(len(housefillna))

house["size"] = housefillna


# In[ ]:


# Drop the rest 13 null rows
house.drop(house[house["size"].isna()].index, inplace=True)


# In[ ]:


print("shape of table house", house.shape)
print("shape of na values", house[house["size"].isna()].shape)
print("shape of not na values", house[house["size"].notna()].shape)


# In[ ]:


# Manipulate num_bedroom column
house.num_bedroom = house.num_bedroom.str.strip(" bd")
house.num_bedroom = house.num_bedroom.str.strip("-")
house = house[house.num_bedroom != ""]
house.num_bedroom = house.num_bedroom.astype(int)


# In[ ]:


# Manipulate num_bathroom column
house.num_bathroom = house.num_bathroom.str.strip(" ba")
house.num_bathroom = house.num_bathroom.str.strip("-")
house = house[house.num_bathroom != ""]
house.num_bathroom = house.num_bathroom.astype(int)

# drop outliers
house = house[house.num_bathroom <= 9]


# In[ ]:


# Manipulate price column
house.price = house.price.str.replace(",", "")
house.price = house.price.astype(float)


# In[ ]:


house.info()


# In[ ]:


house.ptype.unique()


# In[ ]:


# Map categories data into numeric
house.ptype = house.ptype.map({"condo": 1, "townhouse": 2, "house": 3})
# Create price_per_room column
house["price_per_room"] = house["price"] / house["num_bedroom"]
# Drop address column
house.drop("address", axis=1, inplace=True)


# In[ ]:


# Group data by location parameters
house = house.groupby(["post_code", "neighborhood"]).mean()
house.reset_index(inplace=True)
house.head()


# #### Change Column names of School Dataset

# In[ ]:


school.columns = ["name", "city", "school_score", "post_code", "address"]


# #### Merge School Table and House Table

# In[ ]:


df = school.merge(house, on="post_code")


# In[ ]:


df.columns


# In[ ]:


df.head()


# In[ ]:


df = df.loc[
    :,
    [
        "city",
        "neighborhood",
        "post_code",
        "school_score",
        "num_bedroom",
        "num_bathroom",
        "price",
        "price_per_room",
        "latitude",
        "longitude",
        "ptype",
        "size",
    ],
]
df.head()


# In[ ]:


df.shape


# In[ ]:


# Create new columns for analysis usage
df["type_muti_price_per_room"] = df.ptype * df.price_per_room
df["tot_rooms"] = df.num_bathroom + df.num_bedroom
df.head()


# In[ ]:


# Create Score Bins
df["score_binned"] = pd.qcut(df["school_score"], 5)
df.score_binned.value_counts()


# In[ ]:


# Plot Hist Map
axes = df.tot_rooms.plot.hist(alpha=0.7)
axes = df.school_score.plot.hist(alpha=0.7)
plt.legend(labels=["tot_rooms", "school_score"])


# In[ ]:


for feature in ["price", "type_muti_price_per_room"]:
    df.loc[:, ["score_binned", feature]].groupby(
        "score_binned"
    ).mean().reset_index().plot.bar(x="score_binned", y=feature)


# ### Optional: Acquire latitude/longitude coordinates of each neighborhood through Google API

# In[ ]:


# lat = []
# lng = []
# post_code_responsed = []
# postcode_no_response = []
# key = "" #please input your google api key
# for postcode in df.post_code:
#     URL = "https://maps.googleapis.com/maps/api/geocode/json?address={},+Toronto,+CA&key={}".format(
#     postcode,key
# )

#     # Do the request and get the response data
#     try:
#         req = requests.get(URL)
#         res = req.json()
#         result = res['results'][0]
#         lat.append(result['geometry']['location']['lat'])
#         lng.append(result['geometry']['location']['lng'])
#         post_code_responsed.append(postcode)
#         print(postcode, "finished")
#     except:
#         print(postcode, "google API response no result")
#         postcode_no_response.append(0)
    


# In[ ]:


# lat_lng = pd.DataFrame([post_code_responsed, lat, lng]).T
# lat_lng.columns = ['post_code', 'lat', 'lng']
# lat_lng.to_csv("lat_lng.csv", header=['post_code', 'lat', 'lng'])
# lat_lng.head()


# ### 2.3 Acquire Venues data through Foursquare

# Use geopy library to get the latitude and longitude values of Toronto.

# In[ ]:


# address = "Toronto, Canada"

# geolocator = Nominatim(user_agent="tor_explorer")
# location = geolocator.geocode(address)
# latitude = location.latitude
# longitude = location.longitude
# print("The geograpical coordinate of Toronto are {}, {}.".format(latitude, longitude))

latitude = 43.653963
longitude = -79.387207


# Define Foursquare Credentials and Version

# In[ ]:


CLIENT_ID = ""  # please input your Foursquare ID
CLIENT_SECRET = ""  # please input your Foursquare Secret
VERSION = "20180605"  # Foursquare API version


# In[ ]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row["categories"]
    except:
        categories_list = row["venue.categories"]

    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]["name"]


# Acquire the top 300 venues that are in this location within a radius of 1000 meters.

# In[ ]:


LIMIT = 300  # limit of number of venues returned by Foursquare API

def getNearbyVenues(names, latitudes, longitudes, radius=1000):

    venues_list = []
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)

        # create the API request URL
        url = "https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}".format(
            CLIENT_ID, CLIENT_SECRET, VERSION, lat, lng, radius, LIMIT
        )

        # make the GET request
        try:
            results = requests.get(url).json()["response"]["groups"][0]["items"]
        except:
            print("can't get request results from url")

        # return only relevant information for each nearby venue
        venues_list.append(
            [
                (
                    name,
                    lat,
                    lng,
                    v["venue"]["name"],
                    v["venue"]["location"]["lat"],
                    v["venue"]["location"]["lng"],
                    v["venue"]["categories"][0]["name"],
                )
                for v in results
            ]
        )

    nearby_venues = pd.DataFrame(
        [item for venue_list in venues_list for item in venue_list]
    )
    nearby_venues.columns = [
        "Neighborhood",
        "Neighborhood Latitude",
        "Neighborhood Longitude",
        "Venue",
        "Venue Latitude",
        "Venue Longitude",
        "Venue Category",
    ]

    return nearby_venues


# In[ ]:


# # Run the above function on each post code and create a new dataframe called toronto_venues.
# toronto_venues = getNearbyVenues(
#     names=neighborhoods["post_code"],
#     latitudes=neighborhoods["latitude"],
#     longitudes=neighborhoods["longitude"],
#     radius=1000,
# )

# toronto_venues.to_csv("toronto_venues_1000.csv")


# #### Read toronto_venues_1000 dataset from csv

# In[ ]:


toronto_venues = pd.read_csv("../input/toronto_venues_1000.csv", encoding="ISO-8859-1").iloc[
    :, 1:
]


# #### Manipulate Venues Data

# In[ ]:


toronto_venues["post_code"] = toronto_venues.Neighborhood
venues_cnt = toronto_venues.groupby("post_code").count().Venue.reset_index()
venues_cnt.columns = ["post_code", "Venue_cnt"]
toronto_venues = toronto_venues.merge(venues_cnt, on="post_code")

print(
    "There are {} uniques categories.".format(
        len(toronto_venues["Venue Category"].unique())
    )
)

s = (
    toronto_venues["Venue Category"].value_counts() >= 3
)  # Drop Categories with only 1-2 records
i = s[s == True].index
toronto_venues = toronto_venues[toronto_venues["Venue Category"].isin(i)]

print(
    "There are {} uniques categories after filter.".format(
        len(toronto_venues["Venue Category"].unique())
    )
)


# In[ ]:


toronto_venues.head()


# In[ ]:


# one hot encoding
toronto_onehot = pd.get_dummies(
    toronto_venues[["Venue Category"]], prefix="", prefix_sep=""
)

# add neighborhood column back to dataframe
toronto_onehot["Neighborhood"] = toronto_venues["post_code"]

# move neighborhood column to the first column
fixed_columns = [toronto_onehot.columns[-1]] + list(toronto_onehot.columns[:-1])
toronto_onehot = toronto_onehot[fixed_columns]

toronto_onehot.head()


# In[ ]:


toronto_onehot.shape


# ### 2.4 Group all data into one dataset

# group rows by neighborhood and by taking the mean of the frequency of occurrence of each category

# In[ ]:


toronto_grouped = toronto_onehot.groupby("Neighborhood").mean().reset_index()
toronto_grouped.shape


# In[ ]:


toronto_grouped.rename({"Neighborhood": "post_code"}, axis=1, inplace=True)
toronto_grouped.head()


# In[ ]:


df_total = df.merge(toronto_grouped, on="post_code")
cnt = toronto_venues.loc[:, ["post_code", "Venue_cnt"]]
cnt.drop_duplicates(inplace=True)
df_total = df_total.merge(cnt, on="post_code")
print(df_total.shape)
df_total.head()


# ## 3. Methodology <a name="methodology"></a>

# In this project, we will use regression methods to **predict the housing price of different zones**. Since there's only 700 examples but 329 features. Here's the methodologies going through:
# 
# First, **dimensionality reduction** using the **correlation matrix** to filter features to avoid the sparse issue of the training set.
# 
# Second, **remove the outliers** due to the typo of the website, or the special case, which will have negative impacts on our model performance.
# 
# The third step, I'd like to use the K-Means clustering method and folium create a map of the GTA area to explore the basic **clusters distribution** of zones.
# 
# The fourth step, **Data Analysis & Visualization**, so we can picture the impacts of different features on the prices.
# 
# The fifth step, Machine Learning Modeling, I'll use different **regression models** and **grid search** methods to find the best hyper-parameters of each model and looking into their performance and confirm if that works well.
# 
# At last, I'll **compare the performances** among different methods and drop some ideas on **how to improve the accuracy** of our prediction in the future.

# ### 3.1 Create correlation matrix

# In[ ]:


cols = list(df_total.columns)
cols = [
    x
    for x in cols
    if x
    not in [
        "city",
        "neighborhood",
        "post_code",
        "latitude",
        "longitude",
        "score_binned",
    ]
]
coef = np.corrcoef(df_total[cols].values.T)
coef_matrix = pd.DataFrame(coef, columns=cols, index=cols)
coef_matrix.head()


# ### 3.2 Remove the outliers 

# In[ ]:


print(len(coef_matrix))
print(len(coef_matrix[abs(coef_matrix.price) > 0.1]))


# In[ ]:


useful_features = coef_matrix[abs(coef_matrix.price) > 0.1].index
useful_features = list(useful_features)


# In[ ]:


# drop outliers
print(len(df_total))
df_total = df_total[(df_total.price > 2 * 10 ** 5) & (df_total.price < 2.8 * 10 ** 6)]
df_total = df_total[df_total.post_code != "L1K0S1"]
print(len(df_total))


# ### 3.3 Explore the clusters distribution of different zones

# In[ ]:


df_clusters = df_total.loc[:, useful_features + ["latitude", "longitude"]]
# df_clusters.drop('score_binned',axis=1,inplace=True)
df_clusters.head()

# Standard processing
sc = StandardScaler()
df_clusters_standard = sc.fit_transform(df_clusters)
df_clusters_standard = pd.DataFrame(df_clusters_standard)
df_clusters_standard.columns = df_clusters.columns
df_clusters_standard.head()


# In[ ]:


# set number of clusters
kclusters = 6

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(df_clusters_standard)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10]

# add clustering labels
df_clusters_standard.insert(0, "Cluster Labels", kmeans.labels_)
df_clusters_standard.head()


# In[ ]:


# import matplotlib.cm as cm
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i * x) ** 2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(
    df_clusters["latitude"],
    df_clusters["longitude"],
    df_clusters_standard["school_score"],
    df_clusters_standard["Cluster Labels"],
):
    label = folium.Popup(str(poi) + " Cluster " + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster - 1],
        fill=True,
        fill_color=rainbow[cluster - 1],
        fill_opacity=0.7,
    ).add_to(map_clusters)

map_clusters


# ## 4. Data Analysis & Visualization <a name="Analysis&Visualization"></a>

# Now, let's perform some basic explanatory data analysis and use Data Visualization method to determine some relationships between cleaned features and prices.

# ### 4.1 Use seaborn to plot heatmap

# In[ ]:


cols = coef_matrix.iloc[(-np.abs(coef_matrix["price"].values)).argsort()].index[:12]

coef = np.corrcoef(df_total[cols].values.T)
fig, ax = plt.subplots(figsize=(12, 12))  # Sample figsize in inches
hm = sns.heatmap(
    coef,
    cbar=True,
    annot=True,
    square=True,
    fmt=".2f",
    annot_kws={"size": 15},
    yticklabels=cols,
    xticklabels=cols,
    ax=ax,
)


# ### 4.2 Plot scatter_matrix

# In[ ]:


from pandas.plotting import scatter_matrix

cols = coef_matrix.iloc[(-np.abs(coef_matrix["price"].values)).argsort()].index[:11]
scatter_matrix = scatter_matrix(
    df_total[cols], alpha=0.2, figsize=(17, 17), diagonal="kde"
)


# ### 4.3 Use binning method and matplotlib to plot some bar graphs

# In[ ]:


temp = df_total
temp["price_binned"] = pd.qcut(
    temp["price"], 5, labels=["very low", "low", "medium", "high", "very high"]
)
temp.price_binned.value_counts()


# In[ ]:


for feature in cols[3:]:
    temp.loc[:, ["price_binned", feature]].groupby(
        "price_binned"
    ).mean().reset_index().plot.bar(x="price_binned", y=feature, logy=True)


# ## 5. Modeling <a name="Modeling"></a>

# ### 5.1 Create Training/Testing Datasets and Performance Record Dataframe

# In[ ]:


# Create features and label
features = [
    "school_score",
    "num_bedroom",
    "num_bathroom",
    "ptype",
    "size",
] + useful_features[9:]
X = df_total.loc[:, features]
y = df_total.loc[:, "price"]

# Split data into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create a dataframe to store the performance of each models
scores = pd.DataFrame()


# ### 5.2 Create Predict and Plot function for different Machine Learning Methods

# In[ ]:


# Create Predict and Plot function for ML methods
def try_different_method(method):
    method.fit(X_train, y_train)
    y_pred = method.predict(X_test)

    y_test_temp = y_test.reset_index(drop=True)
    order = y_pred.argsort(axis=0)
    y_pred = y_pred[order]
    y_test_temp = y_test_temp[order]

    #     maer = np.mean(abs(y_pred - y_test_temp) / y_test_temp)
    mse = metrics.mean_squared_error(y_test_temp, y_pred)
    r2 = metrics.r2_score(y_test_temp, y_pred)

    plt.figure(figsize=(10, 6))
    plt.plot(
        np.arange(len(y_pred)),
        y_test_temp,
        "ro",
        markersize=4,
        label="list price",
        alpha=0.5,
    )
    plt.plot(
        np.arange(len(y_pred)),
        y_pred,
        "bo-",
        markersize=4,
        label="predict price",
        alpha=0.9,
    )

    plt.grid()
    plt.title("MSE: %f" % mse)
    print("mean_squared_error: %f" % mse)
    print("r2: %f" % r2)
    #     print('mean_abs_error_rate: %f' % maer)
    plt.legend()
    return (r2, mse)


# ### 5.3 Random Forest algorithm

# #### Use GridSearch method for Random Forest to figure the best hyper-parameters

# In[ ]:


# parameters to search over with cross-validation
grid_params = [
    {
        "n_estimators": [10, 50, 100],
        "max_depth": [3, 6, 8, 10, None],
        "min_samples_leaf": [1, 2, 5],
    }
]

clf = GridSearchCV(RandomForestRegressor(), grid_params, cv=5, scoring="r2", n_jobs=2)
clf.fit(X_train, y_train)

print("Best parameter values: %r\n" % clf.best_params_)


# #### Use try_different_method function to plot prediction graph

# In[ ]:


# RandomForestRegressor
rf = RandomForestRegressor(
    n_estimators=clf.best_params_["n_estimators"],
    criterion="mse",
    max_depth=clf.best_params_["max_depth"],
    min_samples_leaf=clf.best_params_["min_samples_leaf"],
    n_jobs=2,
    random_state=None,
)

performance_rf = try_different_method(rf)

scores.loc[0, "Random Forest"] = performance_rf[0]
scores.loc[1, "Random Forest"] = performance_rf[1]


# #### Plot Feature Importance graph

# In[ ]:


importance = pd.DataFrame({"feature": features, "importance": rf.feature_importances_})
importance.sort_values(by="importance", axis=0, ascending=False, inplace=True)
importance[:18].plot(
    x="feature",
    y="importance",
    kind="bar",
    figsize=(8, 4),
    title="Feature Importance",
    logy=True,
)


# ### 5.4 Decision Tree algorithm

# #### Use GridSearch method to figure the best hyper-parameters

# In[ ]:


# parameters to search over with cross-validation
grid_params = [{"max_depth": [3, 4, 5, 6, 8, None], "min_samples_leaf": [1, 2, 5, 7]}]

tree = GridSearchCV(DecisionTreeRegressor(), grid_params, cv=5, scoring="r2", n_jobs=2)
tree.fit(X_train, y_train)

print("Best parameter values: %r\n" % tree.best_params_)


# #### Use try_different_method function to plot prediction graph

# In[ ]:


# RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(
    max_depth=tree.best_params_["max_depth"],
    min_samples_leaf=tree.best_params_["min_samples_leaf"],
    random_state=None,
)

performance_tree = try_different_method(tree)
scores.loc[0, "Decistion Tree"] = performance_tree[0]
scores.loc[1, "Decistion Tree"] = performance_tree[1]


# #### Use Graphviz Create a Decision Tree Graph

# In[ ]:


# dot_data = StringIO()
# export_graphviz(
#     tree,
#     out_file=dot_data,
#     feature_names=features,
#     #                 class_names=['Churn'],
#     filled=True,
#     rounded=True,
#     leaves_parallel=False,
#     rotate=False,
#     special_characters=True,
# )
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# # graph.write_pdf("tree_vehicles.pdf")  # this line saves the diagram to a pdf file
# Image(graph.create_png())


# ### 5.5 XGBoost Regressor algorithm

# #### Use GridSearch method to figure the best hyper-parameters

# In[ ]:


# parameters to search over with cross-validation
grid_params = [
    {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.1, 1],
        "n_estimators": [10, 50],
        "reg_lambda": [10, 1, 0.1, 0.01],
        "objective": ["reg:linear"],
    }
]

xgbr = GridSearchCV(xgb.XGBRegressor(), grid_params, cv=5, scoring="r2", n_jobs=2)
xgbr.fit(X_train, y_train)

print("Best parameter values: %r\n" % xgbr.best_params_)


# #### Use try_different_method function to plot prediction graph

# In[ ]:


xgbr = xgb.XGBRegressor(
    max_depth=xgbr.best_params_["max_depth"],
    learning_rate=xgbr.best_params_["learning_rate"],
    n_estimators=xgbr.best_params_["n_estimators"],
    reg_lambda=xgbr.best_params_["reg_lambda"],
    n_jobs=2,
)

performance_XGB = try_different_method(xgbr)
scores.loc[0, "XGBoost"] = performance_XGB[0]
scores.loc[1, "XGBoost"] = performance_XGB[1]


# #### Plot Feature Importance graph

# In[ ]:


importance = pd.DataFrame(
    {"feature": features, "importance": xgbr.feature_importances_}
)
importance.sort_values(by="importance", axis=0, ascending=False, inplace=True)
importance[:18].plot(
    x="feature",
    y="importance",
    kind="bar",
    figsize=(8, 4),
    title="Feature Importance",
    logy=True,
)


# ### 5.5 KNN algorithm

# In[ ]:


# Standard processing
sc = StandardScaler()
X_standard = sc.fit_transform(X)
X_standard = pd.DataFrame(X_standard)
X_standard.columns = X.columns
X_standard.head()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)  # , random_state=42


# #### Use GridSearch method to figure the best hyper-parameters

# In[ ]:


# parameters to search over with cross-validation
grid_params = [{"n_neighbors": [i for i in range(1, 10)]}]

knn = GridSearchCV(
    neighbors.KNeighborsRegressor(), grid_params, cv=5, scoring="r2", n_jobs=2
)
knn.fit(X_train, y_train)

print("Best parameter values: %r\n" % knn.best_params_)


# #### Use try_different_method function to plot prediction graph

# In[ ]:


knn = neighbors.KNeighborsRegressor(
    n_neighbors=knn.best_params_["n_neighbors"], n_jobs=2
)

performance_KNN = try_different_method(knn)
scores.loc[0, "KNN"] = performance_KNN[0]
scores.loc[1, "KNN"] = performance_KNN[1]


# ### 5.6 Use Keras deploy Nural networks

# In[ ]:


# Standard processing
sc = StandardScaler()
X_standard = sc.fit_transform(X)
X_standard = pd.DataFrame(X_standard)
X_standard.columns = X.columns
X_standard.head()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)  # , random_state=42


# In[ ]:


a = 100
drop = 0.1
n = len(features)

model = Sequential(
    [
        Dense(
            int(n * 1.2), input_dim=n, kernel_initializer="normal", activation="relu"
        ),
        Dropout(drop),
        Dense(int(n), kernel_initializer="normal", activation="linear"),
        Dropout(drop),
        Dense(
            int(n * 0.5),
            activation="linear",
            kernel_regularizer=regularizers.l1_l2(l1=a, l2=a),
        ),
        Dropout(drop),
        Dense(1, kernel_initializer="normal"),
    ]
)


# In[ ]:


model.summary()


# In[ ]:


model.compile(
    loss="mean_squared_error",
    optimizer="adam",
    metrics=["mean_squared_error", "mae", "mape"],
)
history = model.fit(
    X_train, y_train, validation_split=0.2, epochs=50, verbose=0, shuffle=True
)


# In[ ]:


# plot metrics
ax1 = plt.plot(history.history["mean_squared_error"])
ax2 = plt.plot(history.history["val_mean_squared_error"])
plt.legend(["mean_squared_error", "val_mean_squared_error"])


# In[ ]:


y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
y_test = np.array(y_test)


# In[ ]:


def my_r2_score(v_true, v_pred):
    ssres = np.sum(np.square(v_true - v_pred))
    sstot = np.sum(np.square(v_true - np.mean(v_true)))
    return 1 - ssres / (sstot)


# In[ ]:


print("r2 on Test Set:", r2_score(y_test, y_pred))
print("r2 on Train Set:", r2_score(y_train, y_pred_train))
print("MSE on Test Set:", mean_squared_error(y_test, y_pred))
print("MSE on Train Set:", mean_squared_error(y_train, y_pred_train))
print("MAE on TestSet:", mean_absolute_error(y_test, y_pred))
print("MAE on TrainSet:", mean_absolute_error(y_train, y_pred_train))


# In[ ]:


y_pred_temp = np.array([i[0] for i in y_pred])
y_test_temp = y_test  # .reset_index(drop=True)
order = y_pred_temp.argsort(axis=0)
y_pred_temp = y_pred_temp[order]
y_test_temp = y_test_temp[order]


mse_NN = metrics.mean_squared_error(y_test_temp, y_pred_temp)
r2_NN = metrics.r2_score(y_test_temp, y_pred_temp)

plt.figure(figsize=(10, 6))
plt.plot(
    np.arange(len(y_pred_temp)), y_test_temp, "ro-", markersize=4, label="school score", alpha=0.8
)
plt.plot(
    np.arange(len(y_pred_temp)),
    y_pred_temp,
    "bo-",
    markersize=4,
    label="predict score",
    alpha=0.5,
)

plt.grid()
plt.title("MSE: %f" % mse_NN)
print("mean_squared_error: %f" % mse_NN)
print("r2: %f" % r2_NN)
# print('mean_abs_error_rate: %f' % maer)
plt.legend()

scores.loc[0, "Neural network"] = r2_NN
scores.loc[1, "Neural network"] = mse_NN
scores


# ## 6. Results and Discussion <a name="results"></a>

# Based on the R2 and MSE peformace of different methods, we can tell that **Random Forest** and **XGBoost** are the best choices of our case and dataset, althrough the score is limited due to lack of exact size, built year, and other important features.
# 
# For the method of the Neural network method, in this project, I've just used some basic functions by Keras for practice and comparison usage. Since the dataset itself is very small, can't work its advantages, the performance only similar to what we have with Decision Tree algorithms. In the real industry, it's only required when we need huge dataset or special business cases.

# In[ ]:


scores = scores.T
scores.columns = ["R2", "Mean Squared Error"]
scores


# In[ ]:


ax1 = scores.plot.bar(y="R2")
ax2 = scores.plot.bar(y="Mean Squared Error")
plt.legend()


# ## 7. Conclusion <a name="conclusion"></a>

# The purpose of this project was using web scraping method and API to collect the related data of GTA including housing, school, Venus. Then use ETL method to get a clean version for analysis/visualization, eventually deploy different machine learning methods both using Skitlearn and Keras.
# 
# As we can figure from the performance and analyst. Most of the features do prove there are some correlations between itself and price, especially put them together to get a decent prediction performance. However, as we mentioned, there are so many candidate features I couldn't get so far. Basic on my personal experience and relator business experience, I'm pretty sure we will have much better performance with more official data especially like housing size, land size, built year, management fee, security ranking, income, ages, rental rate, etc.
# 
