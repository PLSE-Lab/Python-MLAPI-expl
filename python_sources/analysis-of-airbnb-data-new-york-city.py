#!/usr/bin/env python
# coding: utf-8

# # New York City Airbnb Open Data
# #### Airbnb listings and metrics in NYC, NY, USA (2019)
# ![35072368-31872050.jpg](attachment:35072368-31872050.jpg)
# 
# Airbnb is a paid community platform for renting and booking private accommodation founded in 2008. Airbnb allows individuals to rent all or part of their own home as extra accommodation. The site offers a search and booking platform between the person offering their accommodation and the vacationer who wishes to rent it. It covers more than 1.5 million advertisements in more than 34,000 cities and 191 countries. From creation, inaugust 2008, until June 2012, more than 10 million nights have been booked on Airbnb.
# 
# 
# ### **I hope you find this kernel useful and your <font color="red"><b>UPVOTES</b></font> would be highly appreciated**

# ## Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")


# ## Loading the data

# In[ ]:


df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
# df.index.name = None
df.head()


# # 1. Exploratory Data Analysis (EDA)
# 
# EDA allows us to:
# 
# - **Better understand the data:** Getting domain knowledge by reading some articles about the topic you are working on. You don't need to go to deep.
# - **Build intuition about the data:** Check if the data agree with the our domain knowledge.
# - **Generate hypotheses:** Understand how the data was generated, Find insights, and try to predict the output.
# - **Exploring anonymized data:** Explore individual features, check if the values match with our domain knowledge. Explore features relations.

# In[ ]:


df.info()


# In[ ]:


# Checking for missing values
df.isnull().sum()


# In[ ]:


for column in df.columns:
    if df[column].isnull().sum() != 0:
        print("=======================================================")
        print(f"{column} ==> Missing Values : {df[column].isnull().sum()}, dtypes : {df[column].dtypes}")


# For the `float` dtypes we are going to fill the missing values by `mean()`, for `object` we are going to fill missing values by `mode()`. `last_review` is a date, so we need to convert it, then fill missing values from previous values.

# In[ ]:


df["last_review"] = pd.to_datetime(df.last_review)


# In[ ]:


df.last_review.isnull().sum()


# In[ ]:


df["reviews_per_month"] = df["reviews_per_month"].fillna(df["reviews_per_month"].mean())
df.tail()


# In[ ]:


df.last_review.fillna(method="ffill", inplace=True)


# In[ ]:


for column in df.columns:
    if df[column].isnull().sum() != 0:
        print("=======================================================")
        print(f"{column} ==> Missing Values : {df[column].isnull().sum()}, dtypes : {df[column].dtypes}")


# In[ ]:


for column in df.columns:
    if df[column].isnull().sum() != 0:
        df[column] = df[column].fillna(df[column].mode()[0])


# In[ ]:


df.isnull().sum()


# In[ ]:


pd.options.display.float_format = "{:.2f}".format
df.describe()


# In[ ]:


categorical_col = []
for column in df.columns:
    if len(df[column].unique()) <= 10:
        print("===============================================================================")
        print(f"{column} : {df[column].unique()}")
        categorical_col.append(column)


# In[ ]:


# Drop ["id", "host_name"] because it is insignificant and also for ethical reasons.
df.drop(["id", "host_name"], axis="columns", inplace=True)
df.head()


# In[ ]:


df.last_review.isnull().sum()


# # 2. Data Visualization
# 
# - Visualize your data and search for pattern that can help you solve your problem.
# - Correlation analysis helps us to see features relatations.

# In[ ]:


# Visualizing the distribution for every "feature"
df.hist(edgecolor="black", linewidth=1.2, figsize=(30, 30));


# In[ ]:


plt.figure(figsize=(30, 30))
sns.pairplot(df, height=3, diag_kind="hist")


# **We notice from the graphs that :**
# - `latitude` and `longitude` have a normal distribution, most of the hosts are concetrated in specific area.
# - `reviews_per_month` has a lot of outlayers, because of the missing values filled by `mean()` and `mode()`
# - `availability_365` the most of the hosts are not available all the year.
# - `price` most the host has a price under $1000 

# In[ ]:


col = list(df.columns)
col.remove("latitude")
col.remove("longitude")


# In[ ]:


print(col)


# In[ ]:


categorical_col


# In[ ]:


sns.catplot("neighbourhood_group", data=df, kind="count", height=8)


# In[ ]:


# Another way to visualize the data is to use FacetGrid to plot multiple kedplots on one plot

fig = sns.FacetGrid(df, hue="neighbourhood_group", aspect=4, height=10)
fig.map(sns.kdeplot, 'host_id', shade=True)
oldest = df['host_id'].max()
fig.set(xlim=(0, oldest))
sns.set(font_scale=5)
fig.add_legend()


# In[ ]:


sns.set(font_scale=1.5)
sns.catplot("room_type", data=df, kind="count", height=8)


# In[ ]:


# Another way to visualize the data is to use FacetGrid to plot multiple kedplots on one plot

fig = sns.FacetGrid(df, hue="room_type", aspect=4, height=10)
fig.map(sns.kdeplot, 'host_id', shade=True)
oldest = df['host_id'].max()
fig.set(xlim=(0, oldest))
sns.set(font_scale=5)
fig.add_legend()


# In[ ]:


sns.set(font_scale=1.5)
plt.figure(figsize=(12, 8))
df.host_id.hist(bins=100)


# In[ ]:


# df.neighbourhood.hist(bins=100)


# In[ ]:


data = df.neighbourhood.value_counts()[:10]
plt.figure(figsize=(12, 8))
x = list(data.index)
y = list(data.values)
x.reverse()
y.reverse()

plt.title("Most Popular Neighbourhood")
plt.ylabel("Neighbourhood Area")
plt.xlabel("Number of guest Who host in this Area")

plt.barh(x, y)


# In[ ]:


plt.figure(figsize=(12, 8))
plt.scatter(df.longitude, df.latitude, c=df.availability_365, cmap='summer', edgecolor='black', linewidth=1, alpha=0.75)

cbar = plt.colorbar()
cbar.set_label('availability_365')


# In[ ]:


plt.figure(figsize=(12, 8))
plt.scatter(df.longitude, df.latitude, c=df.price, cmap='summer', edgecolor='black', linewidth=1, alpha=0.75)

cbar = plt.colorbar()
cbar.set_label('Price $')


# In[ ]:


print(f"Average of price per night : ${df.price.mean():.2f}")
print(f"Maximum price per night : ${df.price.max()}")
print(f"Minimum price per night : ${df.price.min()}")


# Wow there are some free houses

# In[ ]:


df[df.price == 0]


# In[ ]:


plt.figure(figsize=(12, 8))
plt.xscale('log')
plt.yscale('log')

df.price.hist(bins=100)


# # 3. correlation matrix

# In[ ]:


# correlation matrix
sns.set(font_scale=3)
plt.figure(figsize=(30, 20))
sns.heatmap(df.corr(), annot=True)


# In[ ]:


plt.figure(figsize=(30, 30))
sns.set(font_scale=1.5)
i = 1
for column in df.columns:
    if df[column].dtype == "float64" or df[column].dtype == "int64":
        plt.subplot(3, 3, i)
        df.corr()[column].sort_values().plot(kind="barh")
        i += 1


# In[ ]:


df.drop('price', axis=1).corrwith(df.price).plot.barh(figsize=(10, 8), 
                                                        title='Correlation with Response Variable',
                                                        fontsize=15, grid=True)


# # 4. Handle categorical features
# 
# - Ordinal feature: (Ticket class 1-2-3, Driver's licence A- B -C,...)
#     - Alphabetical (sorted): [B, A, C] ==> [2, 1, 3], sklearn.preprocessing.LabelEncoder
#     - Order of apperance: [B, A, C] ==> [1, 2, 3], pandas.factorize
#     - Frequency encoding: [B, A, C] ==> [0.5, 0.3, 0.2]
#     - One-Hot Oncoding: pandas.get_dummies, sklearn.preprocessing.OneHotEncoder
# - Values in ordinal features are sorted in some meaningful order.
# - Label encoding maps categories to numbers.
# - Frequency encoding maps categories to their frequencies.
# - Label and Frequency encoding are often used for tree based models.
# - One-Hot Encoding is often used for non-tree based models.
# - Interaction of categorical features can help linear and KNN models

# In[ ]:


categorical_col


# In[ ]:


dataset = pd.get_dummies(df, columns=categorical_col)
dataset.head()


# In[ ]:


print(df.columns)
print(dataset.columns)


# In[ ]:


print(dataset.describe().loc["mean", :])
print("====================================")
print(dataset.describe().loc["std", :])


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

col_to_scale = ['host_id', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 
                'calculated_host_listings_count', 'availability_365']

s_sc = StandardScaler()
dataset[col_to_scale] = s_sc.fit_transform(dataset[col_to_scale])

label = LabelEncoder()
dataset['neighbourhood'] = label.fit_transform(dataset['neighbourhood'])


# In[ ]:


print(dataset.describe().loc["mean", :])
print("====================================")
print(dataset.describe().loc["std", :])


# In[ ]:


# plt.figure(figsize=(20, 40))

# columns = list(dataset.drop(['name', 'host_id', 'price', 'last_review', 'neighbourhood_group', 'neighbourhood'], axis=1).columns)

# for i, column in enumerate(columns, 1):
#     plt.subplot(6, 3, i)
#     plt.scatter(df.longitude, df.latitude, c=df[column], cmap='summer', edgecolor='black', linewidth=1, alpha=0.75)

#     cbar = plt.colorbar()
#     cbar.set_label(column)


# In[ ]:


dataset.head()


# In[ ]:


dataset.name.nunique()


# # 5. Model Building

# In[ ]:


from sklearn.model_selection import train_test_split

X = dataset.drop(['name', 'price', 'last_review'], axis=1)
y = dataset.price

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[ ]:


from sklearn import metrics

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)


# In[ ]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression(normalize=True)
lin_reg.fit(X_train,y_train)


# In[ ]:


print_evaluate(y_test, lin_reg.predict(X_test))


# In[ ]:


y.mean()

