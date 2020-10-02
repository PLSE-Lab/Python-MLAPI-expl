#!/usr/bin/env python
# coding: utf-8

# # Super fast feature generation with Google Cloud Big Query
# 
# Generating new features on big datasets can be very challenging. If you don't have a huge amount of RAM, creating features with pandas in a traditional way can be really difficult.
# 
# If you want to work with pandas, there is an excellent notebook which explains how to optimize RAM usage: [How to Work with BIG Datasets on Kaggle Kernels (16G RAM)](https://www.kaggle.com/yuliagm/how-to-work-with-big-datasets-on-16g-ram-dask)
# 
# But even if you have access to a lot of RAM, pandas can be quite slow.
# 
# In this notebook, we will use Google Cloud Big Query service to **generate a csv with 19 new features in less than 4 minutes on the whole AdTracking competition training dataset!** With current google pricing policy, the free tier allows to do this without paying anything.

# ## Let's go! Upload our csv
# I will not write a full tutorial on how to import a csv in Big Query. Google documentation is clear. In brief, you have to:
# 
# * Upload the csv in a bucket on Google Cloud Storage
# * Go to Big Query and create a table by importing from Google Cloud Storage
# 
# **Important: On my computers, Big Query website does not work with Chrome of Firefox. It resulted in "Network Unreachable" error. I had to use Internet Explorer!** Google, please fix this bug...

# ## Create an index column
# 
# Unlike pandas, Big Query does not have an index column. For the test dataset, we could just use 'click_id', but for the training dataset, we have to create one, as we will need it further.
# 
# We could do it with pandas before uploading the csv, but it is really easy to do it directly in Big Query.
# 
# Click on COMPOSE QUERY, and copy/paste the following standardSQL request.
# 
# ```sql
# #standardSQL
# SELECT ROW_NUMBER() OVER(ORDER BY click_time) as index, *
# FROM `my_dataset.my_table`
# ORDER BY index
# ```
# Before clicking on RUN QUERY:
# 
# * replace my_dataset and my_table by the names of your dataset and your table (keep the ` chars around)
# * click on SHOW OPTIONS and specify a name for a destination table. This will save the result of the query in a new table.
# * in the options, validate the ALLOW LARGE RESULTS checkbox.

# ## Generate our features
# 
# Now we have a table with an index, we can generate our new features columns. We are going to generate different types of features:
# 
# * counts on groupby
# * cumulative counts on groupby
# * last click time and next click time
# 
# The groupbys should be applied to any field from the dataset, but also to HOUR extracted from click_time.
# 
# As it would be fastidious to write this in SQL, I prepared a python script that will generate the standardSQL query for us.

# In[ ]:


# Defines the new features

queries_config = [    
    #########################
    # COUNT
    #########################
    {
        'type': 'count',
        'groupby': ['ip', 'HOUR']   # the HOUR keyword is processed for truncating HOUR from click_time timestamp
    },
    {
        'type': 'count',
        'groupby': ['ip', 'app']
    },
    {
        'type': 'count',
        'groupby': ['ip', 'app', 'os']
    },    
    
    ##########################
    # COUNT UNIQUE
    ##########################
    {
        'type': 'countunique',
        'groupby': ['ip'],
        'unique': 'channel'
    },
    {
        'type': 'countunique',
        'groupby': ['ip', 'device', 'os'],
        'unique': 'app'
    }, 
    {
        'type': 'countunique',
        'groupby': ['ip'],
        'unique': 'app'
    }, 
    {
        'type': 'countunique',
        'groupby': ['ip', 'app'],
        'unique': 'os'
    }, 
    {
        'type': 'countunique',
        'groupby': ['ip'],
        'unique': 'device'
    }, 
    {
        'type': 'countunique',
        'groupby': ['app'],
        'unique': 'channel'
    },   
    
    #######################
    # CUMULATIVE COUNT
    #######################  
    {
        'type': 'cumcount',
        'groupby': ['ip'],
    },   
    {
        'type': 'cumcount',
        'groupby': ['ip', 'device', 'os'],
    },   
    
    #######################
    # NEXT CLICK
    #######################    
    {
        'type': 'last_click',
        'groupby': ['ip', 'app', 'device', 'os', 'channel'],
        'order': 'DESC'       # ASC for last_click, DESC for next_click
    },
    {
        'type': 'last_click',
        'groupby': ['ip', 'os', 'device'],
        'order': 'DESC'       # ASC for last_click, DESC for next_click
    },
    {
        'type': 'last_click',
        'groupby': ['ip', 'os', 'device', 'app'],
        'order': 'DESC'       # ASC for last_click, DESC for next_click
    },
    {
        'type': 'last_click',
        'groupby': ['ip', 'channel'],
        'order': 'DESC'       # ASC for last_click, DESC for next_click
    }, 
    #######################
    # LAST CLICK    
    #######################    
    {
        'type': 'last_click',
        'groupby': ['ip', 'app', 'device', 'os', 'channel'],
        'order': 'ASC'       # ASC for last_click, DESC for next_click
    },
    {
        'type': 'last_click',
        'groupby': ['ip', 'os', 'device'],
        'order': 'ASC'       # ASC for last_click, DESC for next_click
    },
    {
        'type': 'last_click',
        'groupby': ['ip', 'os', 'device', 'app'],
        'order': 'ASC'       # ASC for last_click, DESC for next_click
    },
    {
        'type': 'last_click',
        'groupby': ['ip', 'channel'],
        'order': 'ASC'       # ASC for last_click, DESC for next_click
    },    

    
]


# In[ ]:


def create_query(main_table_name, train=False):

    query = "#standardSQL \nWITH "
    #####################
    # Create WITH section
    #####################
    with_sections = []
    field_names = []
    temp_table_names = []
    where_clauses = []

    for c in queries_config:
        section = ""
        # Create field_name and table_name
        if c['type'] == 'countunique':
            field_name = c['type'] + "_" + c['unique'] + "_" + "by" + "_" + "_".join(c['groupby'])
        else:
            field_name = c['type'] + "_" + "by" + "_" + "_".join(c['groupby'])
        if 'order' in c:
            field_name += "_" + c['order']
        temp_table_name = field_name + "_table"
        field_names.append(field_name)
        temp_table_names.append(temp_table_name)
        section += temp_table_name + " AS (\n"
        # SELECT
        section += "  SELECT "
        # Insert function to select hours from timestamp when needed
        processed_groupby = [gb if gb != "HOUR" else "TIMESTAMP_TRUNC(click_time, HOUR, 'UTC') as HOUR" for gb in c['groupby']]    
        if   c['type'] == 'count':
            section += ", ".join(processed_groupby) + ", "
            section += "COUNT(*) "
        elif c['type'] == 'countunique':
            section += ", ".join(processed_groupby) + ", "
            section += "COUNT(DISTINCT " + c['unique'] + ") "
        elif c['type'] == 'cumcount':
            section += "index, ROW_NUMBER() OVER (PARTITION BY " + ", ".join(c['groupby']) + " ORDER BY click_time) "
        elif c['type'] == 'last_click':
            section += "index, TIMESTAMP_DIFF(click_time, LAG(click_time) OVER (PARTITION BY " + ", ".join(c['groupby']) + " ORDER BY click_time " + c['order'] + " ), SECOND)\n    "
        section += "as " + field_name + "\n"
        # FROM
        section += "  FROM " + main_table_name + "\n"
        # GROUP BY
        if c['type'] == 'count' or c['type'] == 'countunique' :
            section += "  GROUP BY " + ", ".join(c['groupby']) + "\n"
            where_clause = " AND ".join([main_table_name + "." + gb + " = " + temp_table_name + "." + gb for gb in c['groupby'] ])
            # Process HOUR
            where_clause = where_clause.replace(main_table_name + ".HOUR", "TIMESTAMP_TRUNC(" + main_table_name  +".click_time, HOUR, 'UTC')")
        else:
            where_clause = main_table_name + ".index = " + temp_table_name + ".index"
        section += ")"
        # Append to with_sections
        with_sections.append(section)
        where_clauses.append(where_clause)

    query += ", \n".join(with_sections) + "\n\n"

    #######################
    # Create SELECT section
    #######################
    query += "SELECT\n  "
    if not train:
        query += main_table_name + ".click_id, "
    query += main_table_name + ".ip, " + main_table_name + ".app, " + main_table_name + ".device, " + main_table_name + ".os, " + main_table_name + ".channel, " + main_table_name + ".click_time, "
    if train:
        query += "is_attributed, "
    query += ", ".join(field_names) + "\n"

    #######################
    # Create FROM section
    #######################
    query += "FROM " + main_table_name + ", "
    query += ", ".join(temp_table_names) + "\n"

    #######################
    # Create WHERE section
    #######################

    query += "WHERE\n  "
    query += "\n  AND ".join(where_clauses)
    query += "\n"

    #########################
    # Create ORDER BY section
    #########################

    # query += "ORDER BY ip, click_time" # Not needed for final computation, use it for debug if you wish

    return query


# ## Generate your query!
# 
# To generate your query, you just need to replace table_name below with your value, and run the code.
# 
# If you want to work on the test set, change the train variable to False. This is required because the columns of the training set and the test set differs slightly (is_attributed in the training set, click_id in the test set).

# In[ ]:


# Create query for train data

train = True
table_name = "`my_dataset.my_table_with_index`"

print(create_query(table_name, train=train))


# ## Run the query
# 
# Now, copy/paste the generated query in Big Query. Before clicking on run query, you will have to set the options to define a destination table, and to allow large results.
# 
# It will take less than 4 minutes! That's really quick compared to pandas!

# ## Export result
# 
# After running the query, you will get a new table with all your new features.
# 
# You cannot download this table directly from Big Query, as is it too large. You will have to export it to Google Cloud Storage, by giving it a name like my_export_data_\*.csv. The \* char will enable Big Query to split your file in several smaller files. If you wish, you can ask Big Query to gzip your files for smaller download sizes.
# 
# When the files are exported to Google Cloud Storage, you can download them from the web UI, or with gsutil [(see documentation)](https://cloud.google.com/storage/docs/gsutil/commands/cp).

# In[ ]:




