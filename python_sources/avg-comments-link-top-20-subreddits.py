# Which subreddits have the highest comment average per link?

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

sql_cmd = """
with t as (
select 
subreddit, 
count(*) as count_comments , 
count (distinct link_id) as count_links 
from May2015 
group by 1) 

select subreddit, count_comments / count_links as avg_comments_per_link  from t 
where count_comments > 400000 
--and subreddit <> 'AskReddit' 
order by avg_comments_per_link desc
""" #

data = pd.read_sql(sql_cmd, sql_conn)

print(data)
plt.style.use('ggplot')
data.plot(kind='bar', x='subreddit', title='Avg. Comments / Link')
plt.xlabel('subreddit')
plt.ylabel('Avg. Comments / Link')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.savefig('avgCommentsPerLink_Top20Subreddits.png')