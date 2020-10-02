# The data comes both as CSV files and a SQLite database

import pandas as pd
import sqlite3
import seaborn as sns
import numpy as np
import  matplotlib.pyplot as plt

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
df = pd.read_sql_query("""SELECT MetadataDateSent, SenderPersonId FROM emails""", con)
ID = pd.read_sql_query('SELECT Id, Name FROM persons', con, index_col='Id')

ID['Name'] = ID['Name'].str.lower()
df = df.dropna(how='all').copy()

# select HRC from the sender list
person_of_interest = 'hillary'
person_id = ID[ID.Name.str.contains(person_of_interest)].index.values
df = df[(df['SenderPersonId']==person_id[0])]

# create datetime objects
df['MetadataDateSent'] = pd.to_datetime(df['MetadataDateSent'])
df = df.set_index('MetadataDateSent')

# 0 for Monday, 6 for Sunday
df['dayofweek'] = df.index.dayofweek

sns.set_style('white')
t_labels = ['Mon', 'Tues', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']
ax = sns.barplot(x=np.arange(0,7), y=df.groupby('dayofweek').SenderPersonId.count(),\
    label=t_labels, palette="RdBu")
sns.despine(offset=10)
ax.set_xticklabels(t_labels)
ax.set_ylabel('Message Count')
ax.set_title('HRC\'s Sent Emails')
plt.savefig('seventhday.png', bbox='tight')