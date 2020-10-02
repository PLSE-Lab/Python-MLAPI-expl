#An investiagtion into why the activity by team flair in /r/NFL is what it is.
#By RLesser

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats
from collections import Counter

sql_conn = sqlite3.connect('../input/database.sqlite')

allNFL = sql_conn.execute("SELECT gilded, score, subreddit, author_flair_text, body FROM May2015 WHERE lower(subreddit) LIKE 'nfl'")

timeA = time.time()

bulkFlairs = []

for comment in allNFL:
    bulkFlairs.append(comment[3])
    
c = Counter(bulkFlairs)

flairList = c.most_common()

#for comment in allNFL:
#    if comment[3] not in [x[0] for x in flairList]:
#        flairList.append([comment[3],0])
#    for flair in flairList:
#        if comment[3] == flair[0]:
#            flair[1] += 1

timeB = time.time()
print("B-A",timeB-timeA)

#flairList.sort(key=lambda x: x[1], reverse=True)


#####
names = [x[0] for x in flairList[:36]]
y_pos = np.arange(0, len(names)*2, 2)
#y_pos = np.arange(len(names))
values = [x[1] for x in flairList[:36]]
#print(values)

fig1 = plt.figure()
plt.barh(y_pos, values, align='center', alpha=0.4)
plt.yticks(y_pos, names)
plt.gca().invert_yaxis()
plt.axis('tight')
plt.text(45000, 60, "What causes this distribution? \nLet's Investigate!", size=18, ha="center")
plt.tick_params(axis="y",labelsize="10")
plt.xlabel('Comment Count')
plt.title('Most active flairs on /r/NFL')
plt.savefig("1.FlairActivity.png",bbox_inches='tight')

#First Hypothesis: Team Subreddit Size
teamSubSize = [("Patriots",29224),("Eagles",19924),("Cowboys",14916),("Packers",26113),
               ("Seahawks",29097),("Vikings",12030),("Jets",7907),("49ers",20206),
               ("Giants",11786),("Colts",6853),("Broncos",12387),("Steelers",12451),
               ("Ravens",10488),("Dolphins",7029),("Raiders",6844),("Panthers",7351),
               ("Lions",10379),("Bears",19054),("Saints",7902),("Bills",7091),
               ("Browns",10364),("Texans",8873),("Chargers",8187),("Rams",3641),
               ("Buccaneers",3910),("Chiefs",6319),("Redskins",9878),("Falcons",7453),
               ("Jaguars",3269),("Titans",3320),("Bengals",5637),("Cardinals",3408)]
 
teamSubSize.sort(key=lambda x: x[1], reverse=True)

#####
names = [x[0] for x in teamSubSize]
y_pos = np.arange(0, len(names)*2, 2)
#y_pos = np.arange(len(names))
values = [x[1] for x in teamSubSize]
#print(values)

fig2 = plt.figure()
plt.barh(y_pos, values, align='center', alpha=0.6)
plt.yticks(y_pos, names)
plt.gca().invert_yaxis()
plt.axis('tight')
plt.text(17500, 53, "Data gathered from redditmetrics.com \nfor May 15th, 2015", size=14, ha="center")
plt.tick_params(axis="y",labelsize="10")
plt.xlabel('Subscribers')
plt.title('Team Subreddit Size in May 2015')
plt.savefig("2.SubSize.png",bbox_inches='tight')

for flair in flairList[:36]:
    if flair[0] == None or flair[0] == "NFL" or flair[0] == "AFC" or flair[0] == "NFC": 
        flairList.remove(flair)
flairList = flairList[:32]
flairList.sort(key=lambda x: x[0])
teamSubSize.sort(key=lambda x: x[0])

#####
y = [x[1] for x in flairList]
x = [x[1] for x in teamSubSize]

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
#bfl = np.polyfit(x, y, 1, full=True)
#slope=bfl[0][0]
#intercept=bfl[0][1]
xl = [min(x), max(x)]
yl = [slope*xx + intercept  for xx in xl]

fig3 = plt.figure()
plt.scatter(x, y)
plt.xlabel("Team Subreddit Size")
plt.ylabel("Team Flair Activity")
plt.plot(xl, yl, 'r--')
plt.text(27500,45000, "r^2 = " + str(round(r_value**2,4)),size=10,ha="center")
plt.text(12500,62000, "A strong correlation! Nice!\nBut what's up with the patriots?\nThe Answer - DeflateGate",size=14,ha="center")
plt.text(12500,42000, "For those who don't know, in May 2015, the Patriots\nQB Tom Brady was controversially suspended\nfor allegedly ordering footballs to be deflated.\n\nIt's no wonder Patriot fans have a lot to say!",size=10,ha="center")
plt.arrow(22500, 66000, 5500, 4000, head_width=800, head_length=1200, fc='k', ec='k')
plt.title("Flair Activity VS. Subreddit Size")
plt.savefig("3.FlairVSSub.png")

#Google trends link below for visual confirmation.
#http://www.google.com/trends/explore?hl=en-US&q=/m/05g3b,+/m/05tg3,+/m/03b3j,+/m/06rny,+/m/070xg&date=5/2015+1m&cmpt=q&tz=Etc/GMT%2B4&tz=Etc/GMT%2B4&content=1 
