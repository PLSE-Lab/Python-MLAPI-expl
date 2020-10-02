import sqlite3
#import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

df = sql_conn.execute("SELECT length(body), score FROM May2015 WHERE length(body) > 1 AND length(body) < 520 AND subreddit = 'mildlyinteresting' LIMIT 10000000")



lengthList = [0] * 25
scoreList = [0] * 25

for comment in df:
    commentLengthFilter = comment[0] // 20
    if commentLengthFilter >= 25:
        lengthList[24] += 1
        scoreList[24] += comment[1]
    else:
        lengthList[commentLengthFilter] += 1
        scoreList[commentLengthFilter] += comment[1]
        
for idx, score in enumerate(scoreList):
    if(score is not 0):
        scoreList[idx] = score/lengthList[idx]

print(str(sum(lengthList)))
# Setting the positions and width for the bars
pos = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5]
width = 1

# Plotting the bars
fig, ax = plt.subplots(figsize=(10,5))
#ax2 = ax.twinx()
#ax2.plot(ax.get_x_ticks(), scoreList, linestyle='-', marker='o', linewidth=2.0)


plt.bar(pos,
        lengthList,
        width,
        alpha=0.5,
        # with color
        #color=clrs,
        #label=keys
        )

# Set the y axis label
ax.set_ylabel('Number of comments')

# Set the chart's title
ax.set_title('Number of characters')


# Set the labels for the x ticks
ax.set_xticklabels(['0', '100', '200', '300', '400', '>500'])

# Setting the x-axis and y-axis limits
plt.xlim(min(pos)+width, max(pos)+width)
#plt.ylim([0, max(content_summary['count'])+20])


plt.savefig("LengthsCount.png")
plt.cla()

# Set the y axis label
ax.set_ylabel('Average comment score')

# Set the chart's title
ax.set_title('Number of characters')

# Set the labels for the x ticks
ax.set_xticklabels(['0', '100', '200', '300', '400', '>500'])

plt.plot(pos, scoreList, linestyle='-', marker='o', linewidth=2.0)

plt.savefig("LengthsScore.png")



