#!/usr/bin/env python
# coding: utf-8

# # Exploring NCAA Data with BigQuery

# ## What is BigQuery
# `BigQuery` is Google's fully managed, NoOps, low cost analytics database. With `BigQuery` you can query terabytes and terabytes of data without managing infrastructure or needing a database administrator. `BigQuery` uses `SQL` and takes advantage of the pay-as-you-go model. `BigQuery` allows you to focus on analyzing data to find meaningful insights.
# 
# `NCAA Basketball` dataset contains  games, teams, and players data. The game data covers play-by-play and box scores back to 2009, as well as final scores back to 1996.

# ## Setup

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from google.cloud import bigquery
client = bigquery.Client()


# ## Load the NCAA Basketball dataset

# In[ ]:


dataset_ref = client.dataset('ncaa_basketball', project='bigquery-public-data')


# The method [client.dataset](https://googleapis.github.io/google-cloud-python/latest/bigquery/reference.html#google.cloud.bigquery.dataset.DatasetReference) is named as if it returns a dataset, but it actually gives us a dataset reference.

# In[ ]:


type(dataset_ref)


# Once we have a reference, we can load the real dataset.

# In[ ]:


ncaa_dataset = client.get_dataset(dataset_ref)


# In[ ]:


type(ncaa_dataset)


# Use [client.list_tables](https://googleapis.github.io/google-cloud-python/latest/bigquery/generated/google.cloud.bigquery.table.Table.html#google.cloud.bigquery.table.Table) to get information about the tables within the dataset.

# In[ ]:


[x.table_id for x in client.list_tables(ncaa_dataset)]


# Let's take a closer look at the schema for the `team_colors` table. As with datasets, we'll need to pass a reference to the table to the `client.get_table` method.

# In[ ]:


ncaa_team_colors = client.get_table(ncaa_dataset.table('team_colors'))


# In[ ]:


type(ncaa_team_colors)


# ## Digging into the table commands.

# In[ ]:


# dir(ncaa_team_colors)


# In[ ]:


[command for command in dir(ncaa_team_colors) if not command.startswith('_')]


# ## The schema of the table

# In[ ]:


ncaa_team_colors.schema


# `List_rows` returns a slice of a dataset without scanning any other section of the table. If you've ever written a BQ query that included a limit clause, which limits the data returned rather than the data scanned, you probably actually wanted `list_rows` instead.
# 
# To see a subset of the columns, but the `selected_fields` parameter requires a schema object as an input. We'll need to build a subset of the schema first to pass for that parameter.

# In[ ]:


schema_subset = [col for col in ncaa_team_colors.schema if col.name in ('code_ncaa', 'color')]
results = [x for x in client.list_rows(ncaa_team_colors, start_index=100, selected_fields=schema_subset, max_results=10)]


# In[ ]:


results


# Convert the `google.cloud.bigquery.table.Row` results to dicts to get a version that prints a bit more nicely.

# In[ ]:


for i in results:
    print(dict(i))


# Let's check what resources we would have consumed by doing a full table scan instead of using `list_rows`.

# In[ ]:


BYTES_PER_GB = 2**30
ncaa_team_colors.num_bytes / BYTES_PER_GB


# In[ ]:


def estimate_gigabytes_scanned(query, bq_client):
    # see https://cloud.google.com/bigquery/docs/reference/rest/v2/jobs#configuration.dryRun
    my_job_config = bigquery.job.QueryJobConfig()
    my_job_config.dry_run = True
    my_job = bq_client.query(query, job_config=my_job_config)
    BYTES_PER_GB = 2**30
    return my_job.total_bytes_processed / BYTES_PER_GB


# Run a quick test checking the impact of selecting one column versus an entire table.

# In[ ]:


estimate_gigabytes_scanned("SELECT id FROM `bigquery-public-data.ncaa_basketball.team_colors`", client)


# In[ ]:


estimate_gigabytes_scanned("SELECT * FROM `bigquery-public-data.ncaa_basketball.team_colors`", client)


# ## Explore mbb_teams_games_sr and mbb_pbp_sr table

# ## mbb_teams_games_sr
# ### Description
# Team-level box scores from every men's basketball game from the 2013-14 season to the 2017-18 season. Each row shows a single team's stats in one game. This data is identical to mbb_games_sr, but is organized differently to make it easier to calculate a single team's statistics.
# [](http://)
# ## mbb_pbp_sr
# ### Description
# Play-by-play information from men's basketball games, starting with the 2013-14 season. Each row shows a single event in a game.

# In[ ]:


ncaa_mbb_teams_games_sr = client.get_table(ncaa_dataset.table('mbb_teams_games_sr'))
ncaa_mbb_pbp_sr = client.get_table(ncaa_dataset.table('mbb_pbp_sr'))


# ## Schema

# In[ ]:


ncaa_mbb_teams_games_sr.schema


# In[ ]:


ncaa_mbb_pbp_sr.schema


# ## What types of basketball play events are there?

# In[ ]:


#standardSQL
query="""SELECT
  event_type,
  COUNT(*) AS event_count
FROM `bigquery-public-data.ncaa_basketball.mbb_pbp_sr`
GROUP BY 1
ORDER BY event_count DESC"""

# Set up the query
query_job = client.query(query)
# API request - run the query, and return a pandas DataFrame
events_type = query_job.to_dataframe()
events_type


# ## Which 5 games featured the most three point shots made? How accurate were all the attempts?

# In[ ]:


#standardSQL
#most three points made
query="""SELECT
  scheduled_date,
  name,
  market,
  alias,
  three_points_att,
  three_points_made,
  three_points_pct,
  opp_name,
  opp_market,
  opp_alias,
  opp_three_points_att,
  opp_three_points_made,
  opp_three_points_pct,
  (three_points_made + opp_three_points_made) AS total_threes
FROM `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr`
WHERE season > 2010
ORDER BY total_threes DESC
LIMIT 5"""

# Set up the query
query_job = client.query(query)
# API request - run the query, and return a pandas DataFrame
most_three_points = query_job.to_dataframe()
most_three_points


# ## Which 5 basketball venues have the highest seating capacity?

# In[ ]:


#standardSQL
query="""SELECT
  venue_name, venue_capacity, venue_city, venue_state
FROM `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr`
GROUP BY 1,2,3,4
ORDER BY venue_capacity DESC
LIMIT 5"""


# Set up the query
query_job = client.query(query)
# API request - run the query, and return a pandas DataFrame
highest_seating_cap = query_job.to_dataframe()
highest_seating_cap


# ## Which teams played in the highest scoring game since 2010?

# In[ ]:


#standardSQL
#highest scoring game of all time
query="""SELECT
  scheduled_date,
  name,
  market,
  alias,
  points_game AS team_points,
  opp_name,
  opp_market,
  opp_alias,
  opp_points_game AS opposing_team_points,
  points_game + opp_points_game AS point_total
FROM `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr`
WHERE season > 2010
ORDER BY point_total DESC
LIMIT 5"""

# Set up the query
query_job = client.query(query)
# API request - run the query, and return a pandas DataFrame
highest_scoring_game = query_job.to_dataframe()
highest_scoring_game


# ## Since 2014, what was the biggest difference in final score for a National Championship?

# In[ ]:


#standardSQL
#biggest point difference in a championship game
query="""SELECT
  scheduled_date,
  name,
  market,
  alias,
  points_game AS team_points,
  opp_name,
  opp_market,
  opp_alias,
  opp_points_game AS opposing_team_points,
  ABS(points_game - opp_points_game) AS point_difference
FROM `bigquery-public-data.ncaa_basketball.mbb_teams_games_sr`
WHERE season > 2014 AND tournament_type = 'National Championship'
ORDER BY point_difference DESC
LIMIT 5"""



# Set up the query
query_job = client.query(query)
# API request - run the query, and return a pandas DataFrame
biggest_diff = query_job.to_dataframe()
biggest_diff


# ### If you liked my work, please upvote this kernel since it will keep me motivated to perform more in-depth reserach towards this subject.
