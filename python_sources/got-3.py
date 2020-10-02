# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import sqlite3
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

sql_conn = sqlite3.connect('../input/database.sqlite')
full_name = "jon snow"
i=full_name.find(" ",0,len(full_name))
name = full_name[0:i]
sql= 'SELECT body from May2015 WHERE ((subreddit="subreddit" AND subreddit_id="t5_2r2o9") or (subreddit="asoiaf" AND subreddit_id="t5_2rjz2"))'
command = sql_conn.execute(sql)
print("Executing sql code ::::::\n"+sql)
summary = []
flag = 0
for r in command:
    row = str(r[0])
    if(flag == 1):
        if((row.find('he') or row.find('him')) and not(row.find(full_name) or row.find(name))):
            #print(row)
            summary.append(row)
        else:
            flag = 0
    if(row.find(full_name) or row.find(name)):
        #print(row)
        summary.append(row)
        flag = 1
    


# Any results you write to the current directory are saved as output.
df = pd.DataFrame(data=summary,columns=['about'])
df.to_csv('got_3.csv')

