#!/usr/bin/env python
# coding: utf-8

# # TalkingData AdTracking Fraud Detection

# In this below notebook we will try to analyze the TalkingData AdTracking Fraud Detection Challenge data using SQLite. This method of using SQLite will help us to get an understanding of the data without requiring to load all the data in to the memory. Instead, with SQLite we will loa dchunks of data and create a database file and load all the data in to tables. Then, we can use SQL queries to get answers to our questions. With this method, we can use our good old Latops or Desktops with very less Memory to analyze the data.

# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/SQLite370.svg/640px-SQLite370.svg.png" />

# ### Load Libraries

# In[ ]:


import pandas as pd
import numpy as np
import sqlite3
import zipfile
import subprocess
import gc


# ### Gather Train and Test File Details

# You can uncomment the below blocks of markdown to use the ZIP files directly on your local machines.

# ```
# zf_train = zipfile.ZipFile('train.csv.zip', mode='r')
# zf_test = zipfile.ZipFile('test.csv.zip', mode='r')
# ```

# ```
# for info in zf_train.infolist():
#     print("File Name         -> {}".format(info.filename))
#     print("Compressed Size   -> {:.2f} {}".format(info.compress_size/(1024*1024), "MB"))
#     print("UnCompressed Size -> {:.2f} {}".format(info.file_size/(1024*1024), "MB"))
# ```

# **Output**
# ```
# File Name         -> mnt/ssd/kaggle-talkingdata2/competition_files/train.csv
# Compressed Size   -> 1238.43 MB
# UnCompressed Size -> 7188.46 MB
# ```

# ```
# for info in zf_test.infolist():
#     print("File Name         -> {}".format(info.filename))
#     print("Compressed Size   -> {:.2f} {}".format(info.compress_size/(1024*1024), "MB"))
#     print("UnCompressed Size -> {:.2f} {}".format(info.file_size/(1024*1024), "MB"))
# ```

# **Output**
# ```
# File Name         -> test.csv
# Compressed Size   -> 161.93 MB
# UnCompressed Size -> 823.28 MB
# ```

# As we see above, we have a very huge train file that has an UnCompressed Size of around 7.2GB and compressed file is around 1.2GB. Also the test file is around 823MB uncompressed and after compression comes to around 162MB.

# ### Load Sample Data From Train and Test Files

# In[ ]:


df_train_sample = pd.read_csv('../input/train.csv', nrows=10000) #10k
df_train_sample.info()


# In[ ]:


df_train_sample.head()


# In[ ]:


df_test_sample = pd.read_csv('../input/test.csv', nrows=10000) #10k
df_test_sample.info()


# In[ ]:


df_test_sample.head()


# ## Build a Database for Test Data

# As we have a very huge dataset that may not always fit in memory we will try to build a database table out of it for efficient processing. The database file will be stored on the disk and we can use SQL queries to get answers to our questions. BTW, the database file is created only the first time and you can use it for subsequent queries.

# Create a the database file or use it if it was already created.

# In[ ]:


con = sqlite3.connect("talkingdata_test.db")  # Opens file if exists, else creates file
cur = con.cursor()  # This object lets us actually send messages to our DB and receive results


# Run the below SQL query to check if there is already a table in the database.
# If there is no table then create it and read the data from the csv/zip file and load it in to the table.

# In[ ]:


sql = "SELECT sql FROM sqlite_master WHERE name='test_data'"
cur.execute(sql)

if not cur.fetchall():
    # In the below call you can use the actual test.csv.zip file when running it on a local machine
    for chunk in pd.read_csv("../input/test_supplement.csv", nrows=10000, chunksize=500):
        chunk.to_sql(name="test_data", con=con, if_exists="append", index=False)  #"name" is name of table
        gc.collect()


# Get the schema of the table to get a detail of the columns and their data type.

# In[ ]:


sql = "SELECT sql FROM sqlite_master WHERE name='test_data'"
cur.execute(sql)
cur.fetchall()


# Below we will get the total number of records in the test_data table.

# In[ ]:


sql = "select count(*) from test_data"
cur.execute(sql)
cur.fetchall()


# Below we will select 10 records from the table.

# In[ ]:


sql = "select * from test_data limit 10"
cur.execute(sql)
cur.fetchall()


# ## Build a Database for Test Data

# Below we will create a new database for the training data. If the database was already created, we will use the same without creating it again. So, with the actaul data we will just create the database once and load all the data. From then on we will use the same database to run our queries.

# In[ ]:


con = sqlite3.connect("talkingdata_train.db")  # Opens file if exists, else creates file
cur = con.cursor()  # This object lets us actually send messages to our DB and receive results


# Below we will check if we have the train_data table, otherwise we will create it and load the data from the train zip file.

# In[ ]:


sql = "SELECT sql FROM sqlite_master WHERE name='train_data'"
cur.execute(sql)

if not cur.fetchall():
    # In the below call you can use the actual train.csv.zip file when running it on a local machine
    for chunk in pd.read_csv("../input/train_sample.csv", nrows= 10000, chunksize=500):
        chunk.to_sql(name="train_data", con=con, if_exists="append", index=False)  #"name" is name of table
        gc.collect()


# We will do a select to get the count of records in our table. In the actual run this return **184903890** rows.

# In[ ]:


sql = "select count(*) from train_data"
cur.execute(sql)
cur.fetchall()


# Below we will print the schema of the train_data table.

# In[ ]:


sql = "SELECT sql FROM sqlite_master WHERE name='train_data'"
cur.execute(sql)
cur.fetchall()


# Below we will print 10 records from the train_data table.

# In[ ]:


sql = "select * from train_data limit 10"
cur.execute(sql)
cur.fetchall()


# Below we will print the max and min values for the **ip** column in the train_data table.
# The actul results returned for this query on the complete train data is **(1, 364778)**. This values will help us to choose a datatype for this column when loading the data in to a pandas DataFrame. So, for **ip** we can choose **uint32**

# In[ ]:


sql = "select min(ip), max(ip) from train_data"
cur.execute(sql)
cur.fetchall()


# Below we will print the max and min values for the **app** column in the train_data table.
# The actul results returned for this query on the complete train data is **(0, 768)**. This values will help us to choose a datatype for this column when loading the data in to a pandas DataFrame. So, for **app** we can choose **uint16**

# In[ ]:


sql = "select min(app), max(app) from train_data"
cur.execute(sql)
cur.fetchall()


# Below we will print the max and min values for the **device** column in the train_data table.
# The actul results returned for this query on the complete train data is **(0, 4227)**. This values will help us to choose a datatype for this column when loading the data in to a pandas DataFrame. So, for **device** we can choose **uint16**

# In[ ]:


sql = "select min(device), max(device) from train_data"
cur.execute(sql)
cur.fetchall()


# Below we will print the max and min values for the **os** column in the train_data table.
# The actul results returned for this query on the complete train data is **(0, 956)**. This values will help us to choose a datatype for this column when loading the data in to a pandas DataFrame. So, for **os** we can choose **uint16**

# In[ ]:


sql = "select min(os), max(os) from train_data"
cur.execute(sql)
cur.fetchall()


# Below we will print the max and min values for the **channel** column in the train_data table.
# The actul results returned for this query on the complete train data is **(0, 500)**. This values will help us to choose a datatype for this column when loading the data in to a pandas DataFrame. So, for **channel** we can choose **uint16**

# In[ ]:


sql = "select min(channel), max(channel) from train_data"
cur.execute(sql)
cur.fetchall()


# Below we will print the max and min values for the **is_attributed** column in the train_data table.
# The actul results returned for this query on the complete train data is **(0, 1)**. This values will help us to choose a datatype for this column when loading the data in to a pandas DataFrame. So, for **is_attributed** we can choose **uint8**

# In[ ]:


sql = "select min(is_attributed), max(is_attributed) from train_data"
cur.execute(sql)
cur.fetchall()


# Below we will get the records where the value of is_attributed is 1.
# For the actual data we can see that this value is **456846** which means that we have only **0.247 %** of records where is_attributed is 1.

# In[ ]:


sql = "select count(*) from train_data where is_attributed=1"
cur.execute(sql)
cur.fetchall()


# Below is the query which will allow us to show the top 25 IP Address by their activity.

# In[ ]:


sql = "select ip, count(ip) from train_data group by ip order by 2 desc limit 25"
cur.execute(sql)
cur.fetchall()


# Below is the query which shows the IP addresses with highest activity when is_attributed is 1.

# In[ ]:


sql = "select ip, count(ip) from train_data where is_attributed=1 group by ip order by 2 desc limit 20"
cur.execute(sql)
cur.fetchall()


# # Work in progress.... Please upvote if the notebook is helpful
