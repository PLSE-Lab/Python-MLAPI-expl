import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



fig = plt.figure()
ax = fig.add_subplot(111)
plt.gcf().subplots_adjust(bottom=0.25)
sql_conn = sqlite3.connect('../input/database.sqlite')


gaming_subreddits = ["leagueoflegends", "DestinyTheGame", "DotA2", "GlobalOffensive",
                    "gaming", "GlobalOffensiveTrade",  "witcher",
                    "hearthstone", "Games", "2007scape", "smashbros", "wow","Smite", "heroesofthestorm", "EliteDangerous",
                    "FIFA", "Guildwars2", "tf2", "summonerswar","runescape"]
c = sql_conn.cursor()
placeholder= '?' # For SQLite. See DBAPI paramstyle.
placeholders= ', '.join(placeholder for unused in gaming_subreddits)
query_authors="select subreddit, count(distinct author)  authors from may2015 where subreddit in (%s)  group by subreddit order by authors desc" % placeholders
df = c.execute(query_authors,gaming_subreddits)
all_rows = c.fetchall()
sql_conn.close()
data = []
xTickMarks = []

for row in all_rows:
   data.append(int(row[1]))
   xTickMarks.append(str(row[0]))


## necessary variables
ind = np.arange(len(data))                # the x locations for the groups
width = 0.35                          # the width of the bars
## the bars
rects1 = ax.bar(ind, data, width,
                color='black',
                error_kw=dict(elinewidth=2,ecolor='red'))


# axes and labels
ax.set_xlim(-width,len(ind)+width)
ax.set_ylim(0,130000)


ax.set_ylabel('Authors')
ax.set_xlabel('X LABEL')
ax.set_title('Unique Authors in Gaming Subreddits')
ax.set_xticks(ind+width)
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, rotation=90, fontsize=10)


plt.savefig("Commentdistrib.png")
print('1):', all_rows)
