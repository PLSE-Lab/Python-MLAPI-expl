#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# This is an example notebook for exploring the [Netherlands Rent Properties](https://www.kaggle.com/juangesino/netherlands-rent-properties) dataset. The goal is to show how this dataset can be furhter used for analysis.

# In[ ]:


# Import some necessary libraries
from IPython.display import display, Markdown, Latex
import json
import pandas as pd
import re


# ## Import & Read Data
# 
# The data is in a JSON file. To start with, we will import the file as a python dictionary.

# In[ ]:


raw_data_file = "/kaggle/input/netherlands-rent-properties/properties.json"

def load_raw_data(filepath):
    raw_data = []
    for line in open(filepath, 'r'):
        raw_data.append(json.loads(line))
    
    return raw_data
    
raw_data = load_raw_data(raw_data_file)

Markdown(f"""
Successfully imported {len(raw_data)} properties from the dataset.
""")


# Now we can create a Pandas' DataFrame from this list

# In[ ]:


df = pd.DataFrame(raw_data)

Markdown(f"""
Successfully created DataFrame with shape: {df.shape}.
""")


# In[ ]:


df.info()


# ## Flatten Columns
# 
# Because the source of these data was a JSON file (coming from MongoDB), there are a few columns that have some nested JSONs and/or lists.
# 
# In general, how to handle these will depend on the analysis that will be performed, but there are a few that are useful to handle from the start. For example, the `_id` and some date columns (`crawledAt`, `firstSeenAt`, `lastSeenAt`, `detailsCrawledAt`) are represented as JSON objects because they include the MongoDB type (for compatibility reasons). We can get rid of those and flatten the columns.

# In[ ]:


# Define all columns that need to be flatten and the property to extract
flatten_mapper = {
    "_id": "$oid",
    "crawledAt": "$date",
    "firstSeenAt": "$date",
    "lastSeenAt": "$date",
    "detailsCrawledAt": "$date",
}

# Function to do all the work of flattening the columns using the mapper
def flatten_columns(df, mapper):
    
    # Iterate all columns from the mapper
    for column in flatten_mapper:
        prop = flatten_mapper[column]
        raw_column_name = f"{column}_raw"
        
        # Check if the raw column is already there
        if raw_column_name in df.columns:
            # Drop the generated one
            df.drop(columns=[column], inplace=True)
            
            # Rename the raw back to the original
            df.rename(columns={ raw_column_name: column }, inplace=True)        
    
        # To avoid conflicts if re-run, we will rename the columns we will change
        df.rename(columns={
            column: raw_column_name,
        }, inplace=True)

        # Get the value inside the dictionary
        df[column] = df[raw_column_name].apply(lambda obj: obj[prop])
        
    return df
        


# In[ ]:


df = df.pipe(flatten_columns, mapper=flatten_mapper)


# Note that we haven't dealt with the `datesPublished` column. At the moment this column contains a list of all the days the property has been published on the website. We can later use this to generate a time series.

# ## Rename Columns
# 
# This step is entirely optional, but I prefer thta all columns have more "pythonic" names. The current column names come from JavaScript's camelCase conventions, we'll create a function to rename all columns into snake_case.

# In[ ]:


def rename_columns(df):
    # Store a dictionary to be able to rename later
    rename_mapper = {}
    
    # snake_case REGEX pattern
    pattern = re.compile(r'(?<!^)(?=[A-Z])')
    
    # Iterate the DF's columns
    for column in df.columns:
        rename_mapper[column] = pattern.sub('_', column).lower()
        
    # Rename the columns using the mapper
    df.rename(columns=rename_mapper, inplace=True)
    
    return df


# In[ ]:


df = df.pipe(rename_columns)


# ## Handle Types
# 
# Now we can start parsing the appropiate data types for our columns

# In[ ]:


def parse_types(df):
    
    df["crawled_at"] = pd.to_datetime(df["crawled_at"])
    df["first_seen_at"] = pd.to_datetime(df["first_seen_at"])
    df["last_seen_at"] = pd.to_datetime(df["last_seen_at"])
    df["details_crawled_at"] = pd.to_datetime(df["details_crawled_at"])
    df["latitude"] = pd.to_numeric(df["latitude"])
    df["longitude"] = pd.to_numeric(df["longitude"])
    
    return df


# In[ ]:


df = df.pipe(parse_types)


# Feel free to parse other columns or modify how we parse these.
# 
# Suggestions:
# * `user_last_logged_on` can also be parsed as date
# * `user_member_since` can also be parsed as date
# * `roommates` can also be parsed as numeric, but we need to handle somme text values

# ## Next Steps
# 
# Here are a few suggestions of next steps to clean this data furhter:
# 
# * Use the `latitude` and `longitude` columns to generate geomery points (for example using [GeoPandas](https://geopandas.readthedocs.io/en/latest/gallery/create_geopandas_from_pandas.html))
# * Parse the `posted_ago` column to determine how long ago the property was published (this might not be needed considering that the column `dates_published` has the entire daily history)
# * Perform NLP on the details and descriptions
# * Handle categorical variables better (`internet`, `pets`, `kitchen`, etc)
# * Parse match values (`matchAge`, `matchCapacity`) into numeric values
# * Combine with public datasets to get more features (distance to POI, distance to public transport, neighbourhood data, finacial data)
# 
# **Note**:
# 
# I tried as much as I could to use re-usable functions to wrangle the data to make it easier to reproduce. Simply import my functions and pipe them to the data:

# In[ ]:


raw_data = load_raw_data(raw_data_file)
df = pd.DataFrame(raw_data)
df = (df
      .pipe(flatten_columns, mapper=flatten_mapper)
      .pipe(rename_columns)
      .pipe(parse_types)
     )


# This makes it super easy to extend the functions or add more pipelines.

# ## Research Ideas
# 
# Finally, here are some ideas on how to use this dataset, some are more obvious than others:
# 
# * Can we predict the rental price of a property in The Netherlands?
# * Is there a real-estate bubble in Amsterdam (or any other city)?
# * What factors determine the price of a property?
# * Can we detect high profitability opportunities for rental businesses?
# * Can we find any advice/insights for people who are looking for accomodation?

# ## Bonus: Time Series
# 
# As a bonus, this is one approach to handle these data as a time series.
# 
# **Warning**: This blows up the dataset. It can take a while to finish, and RAM might be an issue. In this example I only performm it with a subset of the data (100 properties).

# In[ ]:


# Flatten column with list of objects
def flatten_col_list(lst):
    return list(map(lambda obj: obj["$date"], lst))

# Transform the DF into a time series
def to_timeseries(df, dates_column="dates_published"):
    # Get a list of columns without the target column
    columns = df.columns.values.tolist()
    columns.remove(dates_column)
    
    # Create a DF with all the dates
    dates_df = pd.DataFrame(df[dates_column].apply(flatten_col_list).tolist())
    
    # Create a wide representation of our DF
    wide = pd.concat([df, dates_df], axis=1).drop(dates_column, axis=1)
    
    # Melt the dataframe
    ts = pd.melt(wide, id_vars=columns, value_name='date')
    
    # [WARNING] Drop columns with missing date
    ts.dropna(inplace=True, subset=["date"])
    
    # Parse the date column
    ts["date"] = pd.to_datetime(ts["date"])
    
    # Offset the date column to account for timezone differences
    ts["date"] = ts["date"] + pd.DateOffset(hours=3)
    
    return ts


# In[ ]:


ts = df[:100].pipe(to_timeseries)


# In[ ]:


# Get a random property to show the time series
target_external_id = ts["external_id"].sample().iloc[0]
ts[ts["external_id"] == target_external_id][["date", "external_id", "city"]].head(10)

