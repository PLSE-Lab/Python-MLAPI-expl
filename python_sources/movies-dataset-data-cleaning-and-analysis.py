#!/usr/bin/env python
# coding: utf-8

# # Movies Dataset Analysis
# - by barisbatuhan
# - link: https://www.kaggle.com/rounakbanik/the-movies-dataset
# 
# ## Contents:
# - Imported Libraries
# - Reading the Data
# - Data Clearing and Formatting
# - Data Analysis
# - What Can Be Done Next?

# ## Imported Libraries
# - **pandas** for holding dataset and processing
# - **numpy** for list operations etc.
# - **matplotlib and seaborn** for graphics and data analysis
# - **ast** for its herler functions

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
import json
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Reading the Data

# In[ ]:


# reads the csv metadata and prints the head
df = pd.read_csv("../input/the-movies-dataset/movies_metadata.csv", low_memory=False)


# ## Data Clearing and Formatting
# 
# ### - Movies Metadata Dataset

# In[ ]:


df.head(5)


# ####  Columns to be Dropped
# - **original_title**: since title column is also included and original_title column has non-ASCII characters, it can be dropped.
# - **homepage**: there will be no analysis depending on the homepage of the movie, this column is uselesss for this specific analysis
# - **imdb_id**: both ratings.csv and keywords.csv has id column to match with metadata dataset, thus no need for this column.
# - **overview & tagline**: no text analysis will be made in this notebook. For retrieving the most important words, keywords.csv can be used
# - **video & poster_path**: no image, video related processing will be made
# - **spoken_languages**: original_language is included, no need

# In[ ]:


drop_df = ["homepage", "poster_path", "video", "imdb_id", "overview", "original_title", 
           "spoken_languages", "tagline"]
df = df.drop(drop_df, axis=1) # drops the selected columns
df = df.drop_duplicates(keep='first') # removes the duplicates from existing dataframe
df.dropna(how="all",inplace=True) # if each column is NaN or null in a row, drops this row


# In[ ]:


df.shape
df.info()


# Out of 45449 rows, there are 6 rows with no title. Let's drop that one, too. Moreover, the types of **id, popularity and budget** is object, although they had to be numeric. Errors will be handled with coerce option, thus invalid parsing will be set as NaN. Also converting release_date to datetime instead of object and extracting the year data may be helpful.

# In[ ]:


df.dropna(subset=["title"], inplace=True)
df["id"] =pd.to_numeric(df['id'], errors='coerce', downcast="integer")
df["popularity"] =pd.to_numeric(df['popularity'], errors='coerce', downcast="float") 
df["budget"] =pd.to_numeric(df['budget'], errors='coerce', downcast="float") 
df['release_date'] = pd.to_datetime(df['release_date'])
df['release_year'] = df['release_date'].dt.year


# As we can see from the dataset itself and *info()* function, **belongs_to_collection** column has too many null entries, therefore instead of giving the collection name, we can convert the data to 0 and 1, 0 for not belonging and 1 for belonging. 

# In[ ]:


df['belongs_to_collection'] = df['belongs_to_collection'].fillna("None")
df['belongs_to_collection'] = (df['belongs_to_collection'] != "None").astype(int)


# In adult column, only 9 True values are present, this information will not give us anything significant, thus, that column is also dropped.

# In[ ]:


df["adult"].value_counts()


# In[ ]:


df.drop(["adult"], inplace=True, axis=1)


# In[ ]:


df.info()


# For **status** column, less than 100 entries are null and it may be a good idea to fill these with most common data. For **runtime**, again a similar case occurs and it can be handled by filling NaN values with the mean.

# In[ ]:


df["status"].fillna(df["status"].value_counts().idxmax(), inplace=True)
df["runtime"] = df["runtime"].replace(0, np.nan)
df["runtime"].fillna(df["runtime"].mean(), inplace=True) 


# Since there are around 70 null **release_date** entries and filling that is not logical, they will be dropped, too. And also 1 row that has null as in column **original_language** may be dropped.

# In[ ]:


df.dropna(subset=["release_date"],inplace=True)
df.dropna(subset=["original_language"],inplace=True)


# There are some cells, which have stringified list of json inputs such as **genres, production_companies and production_countries**. For easier processing, these have to be converted into list of inputs. The function below achieves this:

# In[ ]:


# converts json list to list of inputs (from the label specified with 'wanted' parameter)
def json_to_arr(cell, wanted = "name"): 
    cell = literal_eval(cell)
    if cell == [] or (isinstance(cell, float) and cell.isna()):
        return np.nan
    result = []
    counter = 0
    for element in cell:
        if counter < 3:
            result.append(element[wanted])
            counter += 1
        else:
            break
    return result[:3]


# Let's apply this function to specified 3 parameters:

# In[ ]:


df[['genres']] = df[['genres']].applymap(json_to_arr)
df[['production_countries']] = df[['production_countries']].applymap(lambda row: 
                                                                     json_to_arr(row, "iso_3166_1"))
df[['production_companies']] = df[['production_companies']].applymap(json_to_arr)


# Many entries of **budget and revenue** are 0. However, instead of 0, having NaN is more logical for seeing how many entries are actually available.

# In[ ]:


df['budget'] = df['budget'].replace(0 , pd.np.nan)
df['revenue'] = df['revenue'].replace(0 , pd.np.nan)


# In[ ]:


print("Number of rows with budget < 100: ", len((df[(df["budget"].notna())&(df["budget"] < 100)])))
print("Number of rows with budget > 100 and < 1000: ", len(df[(df["budget"].notna())&(df["budget"] > 100)
                                                              &(df["budget"] < 1000)]))
print("Number of rows with budget > 1000 and < 10000: ", len(df[(df["budget"].notna())&(df["budget"] > 1000)
                                                              &(df["budget"] < 10000)]))


# There are some rows that have a budget and revenue value, that are not actually scaled. By checking some of the notebooks shared, I have decided to move on with the scaling function below. For example, if the value is 1, then it scales to 1 million. If an example will be given from the true data:
# - id: 17402
# - Title: Miami Rhapsody
# - Production Company: Hollywood Pictures	
# - Date: 1995-01-27	
# - Budget: 6
# - Revenue: 5 (by looking IMDB, actual revenue can be seen as around 5 million)

# In[ ]:


def scale_money(num):
    if num < 100:
        return num * 1000000
    elif num >= 100 and num < 1000:
        return num * 10000
    elif num >= 1000 and num < 10000:
        return num *100
    else:
        return num


# In[ ]:


df[['budget', 'revenue']] = df[['budget', 'revenue']].applymap(scale_money)


# After these steps, the columns can be osberved to see how many null or NaN entries there are. So, a heatmap and data is below:

# In[ ]:


sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')


# In[ ]:


print("NaN Genres Count: ", len(df[df["genres"].isna()]))
print("NaN Revenue Count: ", len(df[df['revenue'].isna()])) 
print("NaN Budget Count: ", len(df[df['budget'].isna()])) 
print("NaN Production Company Count: ", len(df[df["production_companies"].isna()]))
print("NaN Production Country Count: ", len(df[df["production_countries"].isna()]))


# For **revenue, budget and production company** filling the values with the most appearing entry or mean is not so logical, since the number of null or NaN entries are huge (More than %20 of whoel dataset). But for **genres and country** it may be done. The function below analyzes the most occuring values for columns in list formats.

# In[ ]:


# returns the values and occurance times or "limiter" amount of different parameters in a 2D list
def list_counter(col, limiter = 9999, log = True):
    result = dict()
    for cell in col:
        if isinstance(cell, float):
            continue
        for element in cell:
            if element in result:
                result[element] += 1
            else:
                result[element] = 1
    if log:
        print("Size of words:", len(result))
    result = {k: v for k, v in sorted(result.items(), key=lambda item: item[1], reverse=True)}
    if log:
        print("Sorted result is:")
    counter = 1
    sum_selected = 0
    total_selected = 0
    rest = 0
    returned = []
    for i in result: 
        if counter > limiter:
            total_selected += result[i]
        else:
            counter += 1
            sum_selected += result[i]
            total_selected += result[i]
            if log:
                print(result[i], " - ", i) 
            returned.append([i, result[i]])
    if log:
        print("Covered:", sum_selected, "out of", total_selected, "\n")
    return returned


# In[ ]:


genres_occur = list_counter(df["genres"].values, log=False)
genres = pd.DataFrame.from_records(genres_occur, columns=["genres", "count"])
genres.plot(kind = 'bar', x="genres")


# In[ ]:


countries_occur = list_counter(df["production_countries"].values, log=False)
countries = pd.DataFrame.from_records(countries_occur, columns=["countries", "count"])
countries.head(20).plot(kind = 'bar', x="countries")


# In[ ]:


companies_occur = list_counter(df["production_companies"].values, log=False)
companies = pd.DataFrame.from_records(companies_occur, columns=["companies", "count"])
companies.head(20).plot(kind = 'bar', x="companies")


# In **genres** *Drama* is the most occurring one with 20189 and in **production_countries** *US* is the most frequent entry. These can be placed into NA cells of these columns:

# In[ ]:


def fill_na_with_list(cell, data):
    if isinstance(cell, float):
        return data
    else:
        return cell


# In[ ]:


df[['genres']] = df[['genres']].applymap(lambda row:
                                        fill_na_with_list(row, [genres_occur[0][0]]))
df[['production_countries']] = df[['production_countries']].applymap(lambda row: 
                                        fill_na_with_list(row, [countries_occur[0][0]]))


# In[ ]:


df.shape
df.info()


# In[ ]:


df["profit"] = df["revenue"] - df["budget"]
df[["popularity", "revenue", "budget", "runtime", "vote_average","profit", "release_year"]].describe()


# Since difference between min and max values for **budget, revenue and profit** is not so small, I have normalized these. In order to preserve the signs of the parameters, the formula of normalization is applied as: 
# - value / (max - min)

# In[ ]:


min_val = df["budget"].min()
max_val = df["budget"].max()
df[["budget", "revenue", "profit"]] = df[["budget", "revenue", "profit"]].apply(lambda x: 
                                                            x / (max_val - min_val))


# From a notebook, I have found a way to arrange **vote_counts** and **vote_averages** with a weighted manner, since there are lots of 0s in the dataset in both columns. The process is implemented below and explained as:
# - Weighted Rating for a row (WR) = [(v + 1) / (v + m) * R] + [m / (m + v) * C]
# - v: number of votes for the movie
# - m: minimum votes required to be listed in the chart (quantile 0.75)
# - R: average rating of the movie 
# - C: mean vote across the whole report

# In[ ]:


vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()
m = vote_counts.quantile(0.75)
def weighted_rating(data):
    v = data['vote_count'] + 1 # added +1
    R = data['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

df['weighted_rating'] = df.apply(weighted_rating, axis=1)


# ### - Keywords Dataset

# First the csv file is read and head if the file is printed to see the format:

# In[ ]:


df_kwrd = pd.read_csv("../input/the-movies-dataset/keywords.csv")
df_kwrd.head()


# As one can see, **keywords** format is stringified list of json and it can be converted to simple list with using the function written above and problematic ones can be calculated:

# In[ ]:


df_kwrd["keywords"] = df_kwrd[['keywords']].applymap(json_to_arr)


# In[ ]:


df_kwrd.dropna(inplace=True)


# In[ ]:


keywords_occur = list_counter(df_kwrd["keywords"].values, log=False)
keywords = pd.DataFrame.from_records(keywords_occur, columns=["keywords", "count"])
keywords.head(20).plot(kind = 'bar', x="keywords")


# Since **id** parameters in both metadata and keywords directing to the same movie, the datasets can be merged.

# In[ ]:


df = pd.merge(df, df_kwrd, on=['id'], how='left')


# In[ ]:


df.info()


# ### - Credits Dataset

# First the dataset is read and first couple of columns are printed to see the appearance of the data:

# In[ ]:


df_cr = pd.read_csv("../input/the-movies-dataset/credits.csv")
df_cr.head()


# Since cast and crew type is stringified list of json, we can again extract the names from the cast and directors from the crew.

# In[ ]:


df_cr["cast"] = df_cr[['cast']].applymap(json_to_arr)


# In[ ]:


def get_director(x):
    x = literal_eval(x)
    for i in x:
        if i == "[]" or isinstance(i, float):
            return np.nan
        if i['job'] == 'Director':
            return i['name']
    return np.nan

df_cr['director'] = df_cr['crew'].apply(get_director)
df_cr.drop(["crew"], axis=1, inplace=True)


# If there are cells with both missing cast and director columns, they should be dropped:

# In[ ]:


print("Entries with no cast:", len(df_cr[df_cr["cast"].isna()]))
print("Entries with no directors:", len(df_cr[df_cr["director"].isna()]))
print("Entries missing both:", len(df_cr[(df_cr["cast"].isna())&(df_cr["director"].isna())]))
df_cr.drop(df_cr[(df_cr["cast"].isna())&(df_cr["director"].isna())].index, inplace=True)


# The **id** of metadata and **id** of credits columns point to the same movies, thus, both datasets can be converged.

# In[ ]:


df = pd.merge(df, df_cr, on=['id'], how='left')


# The final situation in the main dataframe is below:

# In[ ]:


df.shape
df.info()


# In[ ]:


df.head(3)


# ## Data Analysis

# First of all, let's list top 10 movies regarding **weighted_rating and popularity and profit**:

# In[ ]:


df.sort_values('weighted_rating', ascending=False)[["title", "director", "genres", "profit", 
                                                    "popularity", "weighted_rating"]].head(10)


# In[ ]:


df.sort_values('popularity', ascending=False)[["title", "director", "genres", "profit", 
                                                    "popularity", "weighted_rating"]].head(10)


# In[ ]:


df.sort_values('profit', ascending=False)[["title", "director", "genres", "profit", 
                                                    "popularity", "weighted_rating"]].head(10)


# ### - Numerical Data Analysis
# Then, let's look at the correlation values of each numeric column with each other:

# In[ ]:


sns.heatmap(df.corr(), cmap = 'YlGnBu')
df.drop(["id"], axis=1).corr()


# As we can see, there are strong correlation (value > 0.7) between these:
# - 0.73, budget and revenue
# - 0.78, vote_count and revenue
# - 0.75, profit and vote_count
# - 0.98, profit and revenue
# 
# The more **revenue** a movie has, the more **profit** the movie will have and this result is expected therefore. However, other conclusions can be reached, too:
# - If a movie has a higher **budget**, it is excpected to also have higher **revenue**.
# - The more the **number of votes** a movie has, the more **revenue** and therefore **profit** the movie has. 
# 
# About 2nd part, it seemed logical, because number of votes also indicates the **popularity** of a movie and popular ones probably tends to have more **revenue**. However, the relationship between **popularity** and **vote_count** or **profit / revenue** is not so strong. This result is surprizing. However, we can still say that there is a moderate correlation between:
# - 0.46, popularity and revenue
# - 0.56, popularity and vote_count
# - 0.44, popularity and profit
# - 0.61, budget and vote_count
# 
# The most surprizing result was having almost no correlation between **vote_average** and any other parameter except **weighted_rating**. Because it seems logical that higher voted movies tends to have move popularity and revenue. However, this is not the case. On the other hand, after some processing of **vote_average** in order to create **weighted_rating**, some moderate correlations between **weighted_rating** and other parameters are seen:
# - 0.41, popularity and weighted_rating
# - 0.42, vote_count and weighted_rating
# - 0.30, profit and weighted_rating

# In[ ]:


g = sns.scatterplot(x="vote_count", y="profit", data=df[["profit", "vote_count"]])


# In[ ]:


g = sns.scatterplot(x="budget", y="revenue", data=df[["budget", "revenue"]])


# In[ ]:


g = sns.scatterplot(x="vote_count", y="popularity", data=df[["popularity", "vote_count"]])


# In[ ]:


g = sns.scatterplot(x="popularity", y="weighted_rating", data=df[["popularity", "weighted_rating"]])


# After the analysis of the numerical data, we can also look at the categorical data entries:

# ### - Genre Analysis
# Let's construct a sub-dataset for specificly genres:

# In[ ]:


df_genres = df[["title", "genres", "popularity", "budget", "revenue", "vote_count", "weighted_rating"]]


# In[ ]:


df_genres.head()


# Now, let's create new columns for each genre type:

# In[ ]:


genres = list_counter(df_genres["genres"].values, log=False)


# In[ ]:


def list_to_col(data, col_name, col_list, limiter = 9999):
    counter = 0
    selected_items = set()
    for item in col_list:
        if counter >= limiter:
            break
        item = item[0]
        data[item] = 0
        selected_items.add(item)
        counter += 1
    
    for index, row in data.iterrows():
        for item in row[col_name]:  
            if item in selected_items:
                data.at[index, item] = 1
    data.drop([col_name], axis=1, inplace=True)
    return data


# In[ ]:


df_genres = list_to_col(df_genres, "genres", genres)
df_genres


# Now, we can calculate average of **weighted_rating, vote_count, popularity, budget and revenue** for each type and compare the results:

# In[ ]:


def binary_mean_dataset_generator(data, col_list, limiter = 9999):
    counter = 0
    items = []
    for item in col_list:
        if counter >= limiter:
            break
        items.append(item[0])
        counter += 1
    rows = []
    for item in items:
        value = data[data[item] == 1].mean()
        rows.append([item, value[0], value[1], value[2], value[3], value[4]])  
    
    df_genres_means = pd.DataFrame(rows, columns=["type", "popularity", "budget", "revenue", 
                                            "vote_count", "rating"])
    return df_genres_means


# In[ ]:


df_means_genres = binary_mean_dataset_generator(df_genres, genres)
df_means_genres


# In[ ]:


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(df_means_genres))
ax.barh(y_pos, df_means_genres['rating'], align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(df_means_genres['type'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Rating')
ax.set_title('Average Rating w.r.t. Genres')
plt.show()


# In[ ]:


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(df_means_genres))
ax.barh(y_pos, df_means_genres['popularity'], align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(df_means_genres['type'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Popularity')
ax.set_title('Popularity w.r.t. Genres')
plt.show()


# In[ ]:


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(df_means_genres))
ax.barh(y_pos, df_means_genres['vote_count'], align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(df_means_genres['type'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Vote Count')
ax.set_title('Vote Count w.r.t. Genres')
plt.show()


# In[ ]:


sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(10, 5))

sns.set_color_codes("muted")
sns.barplot(x="revenue", y="type", data=df_means_genres[['type', 'budget', 'revenue']],
            label="Revenue", color="b")

sns.set_color_codes("pastel")
sns.barplot(x="budget", y="type", data=df_means_genres[['type', 'budget', 'revenue']],
            label="Budget", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 0.5), ylabel="Movie Types",
       xlabel="Average Budget And Revenue w.r.t. Genres")
sns.despine(left=True, bottom=True)


# In[ ]:


sns.heatmap(df_genres[["popularity", "budget", "revenue", "vote_count", "weighted_rating"]].corr(), 
            cmap = 'YlGnBu')


# ##### Results:
# - About **average ratings**, genre has no significant effect.
# - In **popularity, vote count, revenue**, especially adventure, fantasy, animation, science fiction, family and action gives higher values. For these 3 different aspects, the genre distributions are similar and thus we can conculde that these have correlation between each other.

# ### - Country Analysis
# First of all, the same process made in **genres** has to be made. However, since there are many countries, first 10 will be analysed:

# In[ ]:


df_countries = df[["title", "production_countries", "popularity", "budget", "revenue", "vote_count", "weighted_rating"]]
countries = list_counter(df_countries["production_countries"].values, limiter=10, log=False)
df_countries = list_to_col(df_countries, "production_countries", countries, 10)
df_means_ct = binary_mean_dataset_generator(df_countries, countries)
df_means_ct 


# In[ ]:


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(df_means_ct))
ax.barh(y_pos, df_means_ct['rating'], height=0.5, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(df_means_ct['type'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Rating')
ax.set_title('Average Rating w.r.t. Countries')
plt.show()


# In[ ]:


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(df_means_ct))
ax.barh(y_pos, df_means_ct['popularity'], align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(df_means_ct['type'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Popularity')
ax.set_title('Popularity w.r.t. Countries')
plt.show()


# In[ ]:


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(df_means_ct))
ax.barh(y_pos, df_means_ct['vote_count'], align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(df_means_ct['type'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Vote Count')
ax.set_title('Average Vote Count w.r.t. Countries')
plt.show()


# In[ ]:


sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(8, 3))
sns.set_color_codes("pastel")
sns.barplot(x="revenue", y="type", data=df_means_ct[['type', 'budget', 'revenue']],
            label="Revenue", color="b")

sns.set_color_codes("muted")
sns.barplot(x="budget", y="type", data=df_means_ct[['type', 'budget', 'revenue']],
            label="Budget", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 0.3), ylabel="Movie Types",
       xlabel="Average Budget And Revenue w.r.t. Countries")
sns.despine(left=True, bottom=True)


# In[ ]:


sns.heatmap(df_countries[["popularity", "budget", "revenue", "vote_count", "weighted_rating"]].corr(), 
            cmap = 'YlGnBu')


# ##### Results:
# - About **average ratings**, genre has no significant effect.
# - In **popularity**, the values differ w.r.t. countries. While for US, GB, CA, DE, FR the popularity is greater, for RU and IT the average popularity becomes lower.
# - In **vote count, revenue**, especially US, GB, DE, CA, JP gives higher values. For these 2 different aspects, the genre distributions are similar and thus we can conculde that these have correlation between each other. For some countries like US, **popularity** feature gives also similar rankings.

# ### - Director Analysis

# In[ ]:


df_dir= df[["title", "director", "popularity", "budget", "revenue", "vote_count", "weighted_rating"]]
df_dir.dropna(subset=["director"], inplace=True)
directors = df_dir["director"].value_counts()
directors = directors.index.to_list()


# In[ ]:


def str_to_col(data, col_name, col_list, limiter = 9999):
    counter = 0
    selected = set()
    for item in col_list:
        if counter >= limiter:
            break
        data[item] = 0
        selected.add(item)
        counter += 1
    for index, row in data.iterrows():
        item = row[col_name]
        if(item in selected):
            data.at[index, item] = 1
    data.drop([col_name], axis=1, inplace=True)
    return data


# In[ ]:


def str_mean_dataset_generator(data, col_list, limiter = 9999):
    counter = 0
    items = []
    for item in col_list:
        if counter >= limiter:
            break
        items.append(item)
        counter += 1
    rows = []
    for item in items:
        value = data[data[item] == 1].mean()
        rows.append([item, value[0], value[1], value[2], value[3], value[4]])  
    
    df_genres_means = pd.DataFrame(rows, columns=["type", "popularity", "budget", "revenue", 
                                            "vote_count", "rating"])
    return df_genres_means


# In[ ]:


df_dir = str_to_col(df_dir, "director", directors[:15], 15)
df_means_dir = str_mean_dataset_generator(df_dir, directors[:15])
df_means_dir 


# In[ ]:


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(df_means_dir))
ax.barh(y_pos, df_means_dir['rating'], height=0.5, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(df_means_dir['type'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Rating')
ax.set_title('Average Rating w.r.t. Directors')
plt.show()


# In[ ]:


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(df_means_dir))
ax.barh(y_pos, df_means_dir['popularity'], align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(df_means_dir['type'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Popularity')
ax.set_title('Popularity w.r.t. Directors')
plt.show()


# In[ ]:


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(df_means_dir))
ax.barh(y_pos, df_means_dir['vote_count'], align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(df_means_dir['type'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Vote Count')
ax.set_title('Vote Count w.r.t. Directors')
plt.show()


# In[ ]:


sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(8, 3))
sns.set_color_codes("pastel")
sns.barplot(x="revenue", y="type", data=df_means_dir[['type', 'budget', 'revenue']],
            label="Revenue", color="b")

sns.set_color_codes("muted")
sns.barplot(x="budget", y="type", data=df_means_dir[['type', 'budget', 'revenue']],
            label="Budget", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="upper right", frameon=True)
ax.set(xlim=(0, 0.3), ylabel="Movie Types",
       xlabel="Average Budget And Revenue w.r.t. Directors")
sns.despine(left=True, bottom=True)


# In[ ]:


sns.heatmap(df_dir[["popularity", "budget", "revenue", "vote_count", "weighted_rating"]].corr(), 
            cmap = 'YlGnBu')


# ##### Results:
# - In **average ratings**, the values differ regarding the director. The highest rating belongs to Martin Scorsese and the lowest one is Julien Duvivier's.
# - In **popularity**, the value difference between directors significantly increases. However, the ones thathave higher **average ratings** usually haves higher **popularity** values, too (and vice versa). Therefore, **average_ratings** and **popularity** may have a significant correlation between each other.
# - In **vote count and revenue & budget**, Martin Scorsese has a great difference from all the other directors. But also, the ones that have higher values in previous categories are higher again compared to others.

# ### - Keyword Analysis

# In[ ]:


df_key = df[["title", "keywords", "popularity", "budget", "revenue", "vote_count", "weighted_rating"]]
df_key.dropna(subset=["keywords"], inplace=True)
keywords = list_counter(df_key["keywords"].values, 20, log=False)
df_key = list_to_col(df_key, "keywords", keywords)
df_means_key = binary_mean_dataset_generator(df_key, keywords, 20)
df_means_key


# In[ ]:


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(df_means_key))
ax.barh(y_pos, df_means_key['rating'], align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(df_means_key['type'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Rating')
ax.set_title('Average Rating w.r.t. Keywords')
plt.show()


# In[ ]:


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(df_means_key))
ax.barh(y_pos, df_means_key['popularity'], align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(df_means_key['type'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Popularity')
ax.set_title('Popularity w.r.t. Keywords')
plt.show()


# In[ ]:


plt.rcdefaults()
fig, ax = plt.subplots()
y_pos = np.arange(len(df_means_key))
ax.barh(y_pos, df_means_key['vote_count'], align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(df_means_key['type'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Vote Count')
ax.set_title('Vote Count w.r.t. Keywords')
plt.show()


# In[ ]:


sns.set(style="whitegrid")
f, ax = plt.subplots(figsize=(10, 5))

sns.set_color_codes("muted")
sns.barplot(x="revenue", y="type", data=df_means_key[['type', 'budget', 'revenue']],
            label="Revenue", color="b")

sns.set_color_codes("pastel")
sns.barplot(x="budget", y="type", data=df_means_key[['type', 'budget', 'revenue']],
            label="Budget", color="b")

# Add a legend and informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(0, 0.5), ylabel="Movie Types",
       xlabel="Average Budget And Revenue w.r.t. Keywords")
sns.despine(left=True, bottom=True)


# In[ ]:


sns.heatmap(df_key[["popularity", "budget", "revenue", "vote_count", "weighted_rating"]].corr(), 
            cmap = 'YlGnBu')


# ##### Results:
# - In **average ratings**, the values differ not so significantly.
# - In **popularity**, the value difference between keywords significantly increases. The most popular keywords are: *based on novel, prison, sex, paris*.
# - In **vote count** again the most popular ones are the same with the most popular ones. However, when a movie is **based on a novel** then the aoumt of votes significantly higher than other categories.
# - In revenue & budget**, the movies with **monsters** profits more than others in average.

# ### - Director and Cast Correlation
# First, top directors and casts have to be placed to columns. Since there are too many people in these categories, top 10 of each category will be selected:

# In[ ]:


df_cast_dir = df[["director", "cast"]].dropna()
df_cast_dir.head()


# Directors are assigned into columns:

# In[ ]:


director_list = df_cast_dir["director"].value_counts()
director_list = director_list.index.to_list()
df_cast_dir = str_to_col(df_cast_dir, "director", director_list[:10], 10)


# Cast is assigned into columns:

# In[ ]:


cast = list_counter(df_cast_dir["cast"].values, 10, log=False)
df_cast_dir = list_to_col(df_cast_dir, "cast", cast, 10)


# Rows with all zeros are removed:

# In[ ]:


df_cast_dir = df_cast_dir.loc[(df_cast_dir!=0).any(axis=1)]


# Correlation of the table is investigated:

# In[ ]:


df_cast_dir.shape


# In[ ]:


sns.heatmap(df_cast_dir.corr(), cmap = 'YlGnBu')


# ##### Results:
# - No significant correlation is found.

# ## What Can Be Done Next?

# - More analyses between different categorical values can be made,
# - Relationship between **cast** and numerical columns can be investigated,
# - A recommendation system can be implemented,
# - *ratings.csv* file can also be included to the analysis.
