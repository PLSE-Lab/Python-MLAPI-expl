#!/usr/bin/env python
# coding: utf-8

# # Accessing private datasets using Aircloak
# 
# [Aircloak](https://aircloak.com) is a privacy-preserving analytics solution using patented and proven data anonymization to enable GDPR-compliant and high-fidelity access to sensitive data.
# 
# In short: With Aircloak you can use the power of sql to explore private datasets without needing to worry about privacy. Aircloak restricts certain queries and adds pseudo-random noise to results to make sure potentially sensitive information stays hidden. 
# 
# This is a brief introduction to getting started with Aircloak for data exploration. We will be using a dataset containing responses to the [cov-clear](https://cov-clear.com/) COVID-19 survey.
# 
# Please refer to the [Aircloak documentation](https://covid-db.aircloak.com/docs/) for further reading. 
# 
# 
# Let's get started!

# ## Prelude
# 
# Aircloak presents a postgres-compatible query protocol so we can use the `psycopg2` postgres database client to submit queries to the anonymised data. Let's install it in our environment using pip.

# In[ ]:


get_ipython().system('pip install psycopg2')


# We will be using `pandas` for data exploration. Officially, pandas only supports SqlAlchemy and SQLite3. In practice, since we are only going to be retrieving query results and not writing to the database in any way, pandas *should work*&ast; with psycopg2 which implements the Python DB API 2.0 specifications.
# 
# We also import plotly express so we can draw pretty pictures. 
# 
# (&ast;*Insert disclaimer here*)

# In[ ]:


import pandas as pds
import plotly_express as px

pds.set_option('max_rows', 12)


# ## First things first
# 
# Let's declare some constants. The dataset we will be working with is hosted at _covid-db.aircloak.com_ and the postgres interface is accessible through port 9432. We will also need some credentials. 

# In[ ]:


AIRCLOAK_PG_HOST = "covid-db.aircloak.com"
AIRCLOAK_PG_PORT = 9432
AIRCLOAK_PG_USER = "covid-19-5BCFDEEB3CDD876492CD"
AIRCLOAK_PG_PASSWORD = "RjV+coInOrmahmEUDorvLL9XPNLEDgdsU4Zl1wr3cMpt04ojx5bH/1bnFLw4/WMf/yHpSXFIKkdMiMl2D4KrGQ=="
COVID_DATASET = "cov_clear"


# With this initial setup out of the way we can create a connection to the Aircloak instance using `psycopg2`. 

# In[ ]:


import psycopg2

conn = psycopg2.connect(
    user=AIRCLOAK_PG_USER, 
    host=AIRCLOAK_PG_HOST, 
    port=AIRCLOAK_PG_PORT, 
    dbname=COVID_DATASET,
    password=AIRCLOAK_PG_PASSWORD)


# ## Querying the database
# 
# We can use some pandas magic to extract query results straight into a dataframe for easy analysis.

# In[ ]:


def query(statement):
    return pds.read_sql_query(statement, conn)


# ### The data model
# 
# Let's warm up by looking at the data model. 
# 
# Aircloak provides a couple of helper queries for this: `SHOW TABLES` and `SHOW COLUMNS FROM {table}`.
# 
# We'll write a couple of helper functions:

# In[ ]:


def get_tables():
    return query("SHOW TABLES")

def get_table_columns(table):
    return query(f'SHOW COLUMNS FROM {table}')


# Now, let's check what tables are available:

# In[ ]:


get_tables()


# There are three tables in the dataset - questions, survey and symptoms. Note, two of the tables are labled as *personal*, the other as *non-personal*. More about that later. First, we'll take a look at each table, starting with questions:

# In[ ]:


get_table_columns("questions")


# The `questions` table is a simple look-up table linking column names to a textual representation of the question asked. It contains no personal information and can't be linked to personal information from other tables, so we can query it freely. All Air, no Cloak. Breathe while you can...

# In[ ]:


questions_df = query("SELECT * FROM questions")
questions_df[:10]


# Since that was so simple, let's try the same with a more interesting table, `survey`:

# In[ ]:


# survey_df = query("SELECT * FROM survey")
# survey_df[:10]


# Try uncommenting and running the above two lines of code. Oops.
# 
# This is our first taste of the Cloak. Fortunately, when Aircloak says *no*, it does so in a very informative manner:
# 
# > Directly selecting or grouping on the user id column in an anonymizing query is not allowed, as it would lead to a fully censored result.
# > Anonymizing queries can reference user id columns either by filtering on them, aggregating them, or processing them so that they have a chance to create anonymized buckets.
# 
# In short, the `survey` table contains personal data, so you can't read individual records. Any queries to this table must perform some kind of processing (usually aggregation) that groups records together. 
# 
# Anonymization is a complex subject. The [Aircloak documentation](https://covid-db.aircloak.com/docs/) is very thorough but the best way to quickly familiarise yourself with the basics is to fight against the system and let the error messages guide you.
# 
# Let's take a step back and see what columns are available in the `survey` TABLE.

# In[ ]:


get_table_columns("survey")


# That's a long list of columns... What we have here is a list of survey answers, where the name of each column corresponds to the question that was asked. In addition, there is personal data: email, age, postcode.
# 
# > Note also that Aircloak designates columns as `isolator?` columns and also defines a `key type` for certain columns. For the moment we can ignore this. 
# 
# As an example, let's see how people answered the question "How are you feeling right now?". We will follow Aircloak's advice and request a grouping rather than a list of items. 

# In[ ]:


# A simple query to extract counts of each distinct value in the tables
def count_distinct_values(table, column, order_by='count'):
    return query(f'''
        SELECT {column}, count(*) as count
        FROM {table} 
        GROUP BY {column}
        ORDER BY {order_by} DESC''')

count_distinct_values("survey", "feeling_now")


# Phew, the query worked *and* it looks like most people are feeling fine!
# 
# 
# Let's take a brief look at the last table, `symptoms`:

# In[ ]:


get_table_columns("symptoms")


# The `symptoms` table maps symptoms to a particular survey submission (uniquely identified by the `uid`). This means entries in this table can be linked back to personal data and thus Aircloak's anonymity restrictions apply. 
# 
# We can get an overview of the table by counting the distinct entries as above:

# In[ ]:


symptoms_count = count_distinct_values("symptoms", "symptom")

px.bar(symptoms_count, 
        x='count', 
        y='symptom', 
        orientation='h', 
        height=650)


# It looks like we have lot of fit, ok respondents!

# ## Recap
# 
# The distinction between personal and non-personal is crucial. Any data that can be linked back to an individual is considered personal. Personal data will be protected by Aircloak, meaning we will only be able to extract aggregate information, and even that will have some random noise added.
# 
# The only way we can explore sensitive data is in aggregated form. Here we simply counted the occurences of different values to gain an overview of the contents of the dataset, but of course this is just the beginning!
# 
# 

# ## Something more interesting
# 
# Most public datasets are anonymized before being published, meaning the data has been aggregated and/or censored in a variety of ways that are determined by the data provider. The advantage of Aircloak is dynamic anonymization of datasets, so the end-user determines where the restrictions are applied. 
# 
# As a slightly more advanced example, let's extract the average anxiety of survey respondents and see how this correlates with reported symptoms. Anxiety was reported on a scale of 0 to 10, 0 being 'Completely Calm' and 10 being 'Petrified'.

# In[ ]:


anxiety_symptoms = query('''
        SELECT symptom, avg(how_anxious) as avg_anxiety, count(*) as num_respondents
        FROM symptoms, survey
        WHERE symptoms.uid = survey.uid
        GROUP BY symptom
        ORDER BY avg(how_anxious) ASC''')

anxiety_symptoms


# In[ ]:


px.bar(anxiety_symptoms, 
       x='num_respondents', 
       y='symptom', 
       orientation = 'h', 
       color='avg_anxiety', 
       height=650)


# This initial analysis suggests that a small proportion of respondents with pre-existing conditions are more anxious about Covid-19 than the relatively larger cohort with symptoms of the illness.
