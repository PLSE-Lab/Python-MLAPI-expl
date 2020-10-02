#!/usr/bin/env python
# coding: utf-8

# **[SQL Home Page](https://www.kaggle.com/learn/intro-to-sql)**
# 
# ---
# 

# # Introduction
# 
# You've built up your SQL skills enough that the remaining hands-on exercises will use different datasets than you see in the explanations. If you need to get to know a new dataset, you can run a couple of **SELECT** queries to extract and review the data you need. 
# 
# The next exercises are also more challenging than what you've done so far. Don't worry, you are ready for it!
# 
# Run the code in the following cell to get everything set up:

# In[ ]:


# Set up feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.sql.ex4 import *
print("Setup Complete")


# The World Bank has made tons of interesting education data available through BigQuery. Run the following cell to see the first few rows of the `international_education` table from the `world_bank_intl_education` dataset.

# In[ ]:


from google.cloud import bigquery

# Create a "Client" object
client = bigquery.Client()

# Construct a reference to the "world_bank_intl_education" dataset
dataset_ref = client.dataset("world_bank_intl_education", project="bigquery-public-data")

# API request - fetch the dataset
dataset = client.get_dataset(dataset_ref)

# Construct a reference to the "international_education" table
table_ref = dataset_ref.table("international_education")

# API request - fetch the table
table = client.get_table(table_ref)

# Preview the first five lines of the "international_education" table
client.list_rows(table, max_results=5).to_dataframe()


# # Exercises
# 
# The value in the `indicator_code` column describes what type of data is shown in a given row.  
# 
# One interesting indicator code is `SE.XPD.TOTL.GD.ZS`, which corresponds to "Government expenditure on education as % of GDP (%)".
# 
# ### 1) Government expenditure on education
# 
# Which countries spend the largest fraction of GDP on education?  
# 
# To answer this question, consider only the rows in the dataset corresponding to indicator code `SE.XPD.TOTL.GD.ZS`, and write a query that returns the average value in the `value` column for each country in the dataset between the years 2010-2017 (including 2010 and 2017 in the average). 
# 
# Requirements:
# - Your results should have the country name rather than the country code. You will have one row for each country.
# - The aggregate function for average is **AVG()**.  Use the name `avg_ed_spending_pct` for the column created by this aggregation.
# - Order the results so the countries that spend the largest fraction of GDP on education show up first.
# 
# In case it's useful to see a sample query, here's a query you saw in the tutorial (using a different dataset):
# ```
# # Query to find out the number of accidents for each day of the week
# query = """
#         SELECT COUNT(consecutive_number) AS num_accidents, 
#                EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week
#         FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
#         GROUP BY day_of_week
#         ORDER BY num_accidents DESC
#         """
# ```

# In[ ]:


# Your code goes here
country_spend_pct_query = """
                          SELECT country_name, AVG(value) AS avg_ed_spending_pct
                          FROM `bigquery-public-data.world_bank_intl_education.international_education`
                          WHERE indicator_code = 'SE.XPD.TOTL.GD.ZS' and year >= 2010 and year <= 2017
                          GROUP BY country_name
                          ORDER BY avg_ed_spending_pct DESC
                          """

# Set up the query (cancel the query if it would use too much of 
# your quota, with the limit set to 1 GB)
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
country_spend_pct_query_job = client.query(country_spend_pct_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
country_spending_results = country_spend_pct_query_job.to_dataframe()

# View top few rows of results
print(country_spending_results.head())

# Check your answer
q_1.check()


# For a hint or the solution, uncomment the appropriate line below.

# In[ ]:


#q_1.hint()
#q_1.solution()


# ### 2) Identify interesting codes to explore
# 
# The last question started by telling you to focus on rows with the code `SE.XPD.TOTL.GD.ZS`. But how would you find more interesting indicator codes to explore?
# 
# There are 1000s of codes in the dataset, so it would be time consuming to review them all. But many codes are available for only a few countries. When browsing the options for different codes, you might restrict yourself to codes that are reported by many countries.
# 
# Write a query below that selects the indicator code and indicator name for all codes with at least 175 rows in the year 2016.
# 
# Requirements:
# - You should have one row for each indicator code.
# - The columns in your results should be called `indicator_code`, `indicator_name`, and `num_rows`.
# - Only select codes with 175 or more rows in the raw database (exactly 175 rows would be included).
# - To get both the `indicator_code` and `indicator_name` in your resulting DataFrame, you need to include both in your **SELECT** statement (in addition to a **COUNT()** aggregation). This requires you to include both in your **GROUP BY** clause.
# - Order from results most frequent to least frequent.

# In[ ]:


# Your code goes here
code_count_query = """
                    SELECT indicator_code, indicator_name, COUNT(1) AS num_rows
                    FROM `bigquery-public-data.world_bank_intl_education.international_education`
                    WHERE year = 2016
                    GROUP BY indicator_code, indicator_name
                    HAVING COUNT(1) >= 175
                    ORDER BY num_rows DESC
                    """

# Set up the query
safe_config = bigquery.QueryJobConfig(maximum_bytes_billed=10**10)
code_count_query_job = client.query(code_count_query, job_config=safe_config)

# API request - run the query, and return a pandas DataFrame
code_count_results = code_count_query_job.to_dataframe()

# View top few rows of results
print(code_count_results.head())

# Check your answer
q_2.check()


# For a hint or the solution, uncomment the appropriate line below.

# In[ ]:


q_2.hint()
q_2.solution()


# # Keep Going
# **[Click here](https://www.kaggle.com/dansbecker/as-with)** to learn how to use **AS** and **WITH** to clean up your code and help you construct more complex queries.

# ---
# **[SQL Home Page](https://www.kaggle.com/learn/intro-to-sql)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
