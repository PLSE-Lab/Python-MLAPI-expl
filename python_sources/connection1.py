# This R environment comes with all of CRAN preinstalled, as well as many other helpful packages
# The environment is defined by the kaggle/rstats docker image: https://github.com/kaggle/docker-rstats
# For example, here's several helpful packages to load in 

#library(ggplot2) # Data visualization
# library(readr) # CSV file I/O, e.g. the read_csv function
import sqlite3
import pandas.io.sql as sql

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

sql_conn = sqlite3.connect('../input/database.sqlite')
# Data = sql_conn.execute("SELECT * FROM May2015 LIMIT 10")



# for d in Data.description:
#     print(d[0])

# for a in Data:                                
#     print (a)

# 'Select * from May2015 LIMIT 10'+
# 'Select distinct subreddit from May2015'
# 'Select * from May2015 where subreddit ="politics"'
# 'Select  * from May2015 where subreddit ="politics"'
# 'Select distinct author from May2015 where subreddit ="politics"'
# 'Select * from May2015 where subreddit ="electronics"'
query = 'Select distinct subreddit from May2015'
    
table = sql.read_sql(query, sql_conn)
table.to_csv('output.csv')

# Any results you write to the current directory are saved as output.

