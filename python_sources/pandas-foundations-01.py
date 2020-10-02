#!/usr/bin/env python
# coding: utf-8

# ## Summary of my work on the 1st chapter of the book: "Pandas 1.x Cookbook" (Harrison & Petrou).
# ### In this notebook I'll cover:
# 1. Reading the dataset and storing it in a DataFrame;
# 2. Acessing the 3 components of a DataFrame: index, columns and data;
# 3. Understanding data types;
# 4. Selecting a Column;
# 5. Calling Series Methods;
# 6. Series Operations;
# 7. Chaining Series Methods;
# 8. Renaming Column Names;
# 9. Creating and Deleting Columns.

# In[ ]:


import os
os.listdir('../input/pandas-cookbook-data')


# In[ ]:


import pandas as pd
import numpy as np


# ### 1 - Reading the dataset and storing it in a DataFrame:

# In[ ]:


movies = pd.read_csv("../input/pandas-cookbook-data/data/movie.csv") # optional attribute: index_col="movie_title"
movies


# ### 2 - Acessing the 3 components of a DataFrame: `index`, `columns` and `data`:

# In[ ]:


columns = movies.columns # accessing the columns
index = movies.index # accessing the index
data = movies.values # data stores a numpy.ndarray


# In[ ]:


type(movies)


# In[ ]:


print(type(columns)) # `columns` is of the `Index` type.
columns


# In[ ]:


print(type(index)) # `index` is of the `RangeIndex` type
index
# the RangeIndex is similar to the python's built-in range()


# In[ ]:


print(type(data)) # `data` is an numpy.ndarray
data


# "The index and the columns are closely related. Both of them are subclasses of `Index`. This allows me to perform similar operations on both the `index` and the `columns`":

# In[ ]:


print(issubclass(pd.RangeIndex, pd.Index))
print(issubclass(columns.__class__, pd.Index))


# In[ ]:


index.to_numpy()


# In[ ]:


columns.to_numpy()


# In[ ]:


movies.to_numpy()


# ### 3 - Understanding data types:

# In[ ]:


# The `.dtypes` attribute returns a pandas Series if I need to use the data
print(type(movies.dtypes))
movies.dtypes # shows the column names along with its data type


# In[ ]:


movies.dtypes.value_counts()


# In[ ]:


movies.info() # provides more information than the previous ones


# In[ ]:


type(np.nan) # NaN is float for pandas and NumPy, pandas rely heavily on NumPy...


# In[ ]:


pd.Series(["1",np.nan,"3"]).dtype # 'O' is relative to `object`, if the comlumns has mixed values, its type is `object`


# #### Almost all pandas data types are built from NumPy

# ### 4 - Selecting a Column:
# Selecting a column returns a `Series`:

# In[ ]:


print(type(movies['director_name']))
movies['director_name']


# ... alternative way ...

# In[ ]:


print(type(movies.director_name))
movies.director_name


# We can also index off the `.loc`(by column name) and `.iloc`(by position) attributes to pull out a `Series`:

# In[ ]:


movies.loc[0:3, "director_name"] # if I want to get all the rows I should put just `:` (a Colon)


# In[ ]:


movies.iloc[0:3, 1]


# In[ ]:


# the RangeIndex is similar to the python's built-in range()
movies["director_name"].index # returns a 'pandas.core.indexes.range.RangeIndex'


# In[ ]:


movies["director_name"].dtype # returns a 'numpy.dtype'


# In[ ]:


movies["director_name"].size # returns a python `int`


# In[ ]:


movies["director_name"].name #returns a python `str`


# In[ ]:


print(type(movies["director_name"].index))
print(type(movies["director_name"].dtype))
print(type(movies["director_name"].size))
print(type(movies["director_name"].name))


# We can use the `.apply` method with the type function to get back a `Series` that has the type of every member. Rather than looking at the whole `Series` result, we will chain the `.unique` method onto the result, to look at just the unique types that are found in the `director_name` column:

# In[ ]:


#apply() is used for applying a function on the values of a Series, or applying a function along the axis of a DataFrame.
print(type(movies["director_name"].apply(type)))
movies["director_name"].apply(type) # this way we are applying the type function along the values of the series...


# In[ ]:


# .unique() Return unique values of Series object.
movies["director_name"].apply(type).unique()


# ### 5 - Calling Series Methods:
# We can use the built-in `dir()` function to uncover all the attributes and methods of a `Series`:

# In[ ]:


dir(pd.Series) ## returns a `list` with all the Series attributes and methods
series_attributes_and_methods = set(dir(pd.Series)) ## returns a `set` with all the Series attributes and methods
print("Quantity of Series Attributes and Methods:", len(series_attributes_and_methods))


# In[ ]:


dir(pd.DataFrame)
dataframe_attributes_and_methods = set(dir(pd.DataFrame))
print("Quantity of DataFrame Attributes and Methods:", len(dataframe_attributes_and_methods))


# In[ ]:


s = series_attributes_and_methods
d = dataframe_attributes_and_methods
print("How many Attributes and Methods they both have in common:", len(s & d))


# `.dtype`:

# In[ ]:


director = movies["director_name"]
fb_likes = movies["actor_1_facebook_likes"]


# In[ ]:


director.dtype


# In[ ]:


fb_likes.dtype


# `.head()` and `sample()`:

# In[ ]:


director.head()


# In[ ]:


director.sample(n=5, random_state=42)


# `.value_counts()`:

# In[ ]:


director.value_counts()


# In[ ]:


director.value_counts(normalize=True)


# In[ ]:


fb_likes.value_counts()


# `.size`, `.shape` and `.unique()`:

# In[ ]:


director.size


# In[ ]:


director.shape


# In[ ]:


director.unique()


# In[ ]:


director.unique().size


# In[ ]:


type(director.unique())


# `.count()`

# In[ ]:


director.count()


# In[ ]:


fb_likes.count()


# `.min()`, `max()`, `mean()`, `.median()` and `std()`:

# In[ ]:


print("Min:", fb_likes.min())
print("Max:", fb_likes.max())
print("Mean:", fb_likes.mean())
print("Median:", fb_likes.median())
print("Std:", fb_likes.std())


# .... or just use `.describe()`:

# In[ ]:


fb_likes.describe()


# In[ ]:


director.describe()


# `.quantile()`:

# In[ ]:


fb_likes.quantile(0.5)


# In[ ]:


fb_likes.quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])


# `.isna()`:

# In[ ]:


director.isna()


# `fillna()`:

# In[ ]:


fb_likes_filled = director.fillna(0)
fb_likes_filled


# In[ ]:


print(fb_likes.count())
print(fb_likes_filled.count())


# `.dropna()`:

# In[ ]:


fb_likes_dropped = fb_likes.dropna()
print(fb_likes.size)
print(fb_likes_dropped.size)


# `.hasnans`

# In[ ]:


print(director.hasnans)
print(fb_likes.hasnans)
print(fb_likes_filled.hasnans)
print(fb_likes_dropped.hasnans)


# `.notna()`, which is the compliment of `.isna()`:

# In[ ]:


director.notna()


# ### 6 - Series Operations:

# In[ ]:


imdb_score = movies['imdb_score']
imdb_score


# In[ ]:


imdb_score + 10


# The other basic arithmetic operators, minus (-), multiplication (*), division (/), and exponentiation (***) work similarly with scalar values.

# In[ ]:


imdb_score > 7


# In[ ]:


director == "James Cameron"


# In[ ]:


(director == "James Cameron").to_numpy().sum()


# Each operator has an equivalent method(`.add()`, `.sub()`, `.mul()`, `.div()`, `.florrdiv()`, `.mod()`, `.pow()` and `.lt()`, `.gt()`, `.le()`, `.ge()`, `.eq()`, `.ne()`), for example:

# In[ ]:


imdb_score.add(10) # plus 1


# In[ ]:


imdb_score.gt(7) # greater than 7


# ### 7 - Chaining Series Methods:

# In[ ]:


director.value_counts().head(3)


# A common way to count the number of missing values:

# In[ ]:


fb_likes.isna().sum()


# In[ ]:


fb_likes.fillna(0).astype(int).head(4)


# But chaining can make it difficult to debug.. but I can chain like this:

# In[ ]:


(fb_likes
 .fillna(0)
 .astype(int).head(4))


# ... or ...

# In[ ]:


fb_likes.fillna(0).astype(int).head(4)


# Another option for debugging a chain is to use the `.pipe()` method to show an intermediate valuee:

# In[ ]:


def print_bosta(series):
    print("bosta") #this is just for me to see that the chain has come to this point
    print(series)
    return series

fb_likes.fillna(0).pipe(print_bosta).astype(int).pipe(print_bosta).head(4)


# ### 8 - Renaming Column Names:

# In[ ]:


col_map = {
    "director_name": "director",
    "num_critic_for_reviews": "critic_reviews",
}

movies.rename(columns=col_map).head() # it returns a DataFrame with its columns renamed, but it doesn't change the original DataFrame


# In[ ]:


movies.head()


# .. We can change the index also..

# In[ ]:


idx_map = {
    "Avatar": "Ratava",
    "Spectre": "Ertceps",
    "Pirates of the Caribbean: At World's End": "POC", 
}
col_map = {
    "aspect_ratio": "aspect",
    "movie_facebook_likes": "fblikes",
    "director_name": "director",
    "num_critic_for_reviews": "critic_reviews",
    "director_facebook_likes": "director_fblikes",
    "actor_3_facebook_likes": "actor_3_fblikes",
    "actor_1_facebook_likes": "actor_1_fblikes",
}

(
    movies.set_index("movie_title")
    .rename(index=idx_map, columns=col_map)
    .head(7)
)


# In[ ]:


movies_idx = pd.read_csv("../input/pandas-cookbook-data/data/movie.csv", index_col="movie_title")
movies_idx


# ... pulling the `index` and `columns` then putting in a `list`:

# In[ ]:


ids_list = movies_idx.index.to_list()
ids_list[1:5]


# In[ ]:


cols_list = movies_idx.columns.to_list()
cols_list


# ... modifying them and putting them back to the DataFrame:

# In[ ]:


ids_list[0] = "Ratava"
ids_list[1] = "POC"
ids_list[2] = "Ertceps"

cols_list[1] = "d i r e c t o r"
cols_list[-2] = "aspect"
cols_list[-1] = "fblikes"

movies_idx.index = ids_list
movies_idx.columns = cols_list

movies_idx.head(3)


# **it is possible to pass a function to the `.rename()` method:**

# In[ ]:


def to_clean(string):
    return string.strip().lower().replace(" ", "_") ## this is a good way to put all the names more in accordance with python attributes requirements..
movies_idx.rename(columns=to_clean)


# ... or I can use comprehensions:

# In[ ]:


cols = [
    col.strip().lower().replace(" ", "_")
    for col in movies_idx.columns
]
cols
movies_idx.columns = cols
movies_idx.head()


# ### 9 - Creating and Deleting Columns:

# In[ ]:


movies["has_seen"] = 0
movies.head(3)


# ... or use the `.assign()` method to create a new column:

# In[ ]:


movies.assign(bosta=1) ## it does not mutate the original DataFrame...


# adding up all columns that has facebook likes count:

# In[ ]:


total = ( # the plus operator does not ignores missing numbers, it will return NaN if it has to sum NaN with some valid number...
    movies.actor_1_facebook_likes
    + movies.actor_2_facebook_likes
    + movies.actor_3_facebook_likes
    + movies.director_facebook_likes
)
print(type(total))
total.head() ## Has missing numbers: NaN


# In[ ]:


movies.assign(total_likes=total) # this is one way of adding that Series to the DataFrame


# ... another way ...

# In[ ]:


cols = [
    "actor_1_facebook_likes",
    "actor_2_facebook_likes",
    "actor_3_facebook_likes",
    "director_facebook_likes"
]

def print_type(obj):
    print(type(obj)) # I did this to find out that movies.loc[:, cols] is a DataFrame
    return obj

sum_col = movies.loc[:, cols].pipe(print_type).sum(axis="columns") # I could have just used movies.[cols] to select.
sum_col.head()                                                     # the .sum() method ignores the missing number (NaN).
                                                                   # .sum() converts NaN to 0.


# In[ ]:


movies.assign(total_likes=sum_col)


# **it is possible to pass a function to the `.assign()` method also:**

# In[ ]:


def sum_likes(df):
    return df[
        [ ## this is a good way of doing filtering ...
            c
            for c in df.columns
            if "like" in c
            and ("actor" in c or "director" in c)
        ]
    ].sum(axis=1)

movies.assign(total_likes=sum_likes) # passes `movies` as a parameter and returns a Series with the sum quantity, then append that Series to the DataFrame


# Checking how many missing values there are in both `total` and `sum_col`: `total` was built using the + operator, and `sum_col` was built using the `.sum()` DataFrame method.

# In[ ]:


(
    movies.assign(total_likes=sum_col)["total_likes"]
    .isna()
    .sum()
)


# In[ ]:


(
    movies.assign(total_likes=total)["total_likes"]
    .isna()
    .sum()
)


# In[ ]:


# We could have filled in the missing values as well...
(
    movies.assign(total_likes=total.fillna(0))["total_likes"]
    .isna()
    .sum()
)


# "There is another column in the dataset named `cast_total_facebook_likes`.
# It would be interesting to see what percentage of this column comes from our newly
# created column, `total_likes`. Before we create our percentage column, let's do
# some basic data validation. We will ensure that `cast_total_facebook_likes`
# is greater than or equal to `total_likes`:"

# In[ ]:


def cast_like_gt_actor(df):
    return (
        df["cast_total_facebook_likes"]
        >= df["total_likes"]
    )

df2 = movies.assign(
    total_likes = total,
    is_cast_likes_more=cast_like_gt_actor,
)
df2


# In[ ]:


# Checking if all elements of the `is_cast_likes_more` column is True
df2["is_cast_likes_more"].all()


# Using `.drop()` method to delete the `total_likes` column:

# In[ ]:


df2 = df2.drop(columns="total_likes")
df2


# In[ ]:


# creating a Series of just total actor likes:
actor_sum = movies[
    [ ## filtering the columns
        c
        for c in movies.columns
        if "actor_" in c and "_likes" in c
    ]
].sum(axis="columns")
actor_sum.head()


# now chacking again...

# In[ ]:


print((movies["cast_total_facebook_likes"] >= actor_sum).all())
movies["cast_total_facebook_likes"] >= actor_sum


# ... or ...

# In[ ]:


print(movies["cast_total_facebook_likes"].ge(actor_sum).all())
movies["cast_total_facebook_likes"].ge(actor_sum)


# "Finally, let's calculate the percentage of the `cast_total_facebook_likes` that come from `actor_sum`:"

# In[ ]:


pct_like = (
    actor_sum
    .div(movies["cast_total_facebook_likes"])
    .mul(100)
)
pct_like.describe()


# In[ ]:


pd.Series(
    pct_like.to_numpy(), index=movies["movie_title"]
).head()


# Using `.insert()` and `.get_loc()`:

# In[ ]:


gross_index = movies.columns.get_loc("gross")
profit_index = gross_index + 1
profit_index


# In[ ]:


movies.insert( # the insert method modifies the original DataFrame, so there wont be a assignment statement.
    loc=profit_index,
    column="profit",
    value=movies["gross"] - movies["budget"]
)
movies


# An alternative to deleting columns with the `.drop` method is to use the `del` statement. This
# also does not return a new DataFrame, so favor `.drop` over this:

# In[ ]:


del movies["director_name"]


# In[ ]:


# and to see that `director_names` has been deleted
movies


# In[ ]:


print("Control: !#!413dsdasdf")


# In[ ]:




