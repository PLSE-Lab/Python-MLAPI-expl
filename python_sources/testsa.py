# People that comment "Thanks for gold!" after being gilded are chumps. 
#They should take their reward and shut up.
#Let's see what percent of gilded redditors are gilded chumps as well.

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

allGolds = sql_conn.execute("SELECT body, gilded, subreddit FROM May2015 WHERE gilded > 0")

goldCount = 0
chumpCount = 0
subList = []
subGold = []
subChump = []

for comment in allGolds:
    if comment[2] not in subList:
        subList.append(comment[2])
        subGold.append(0)
        subChump.append(0)
    currentSub = subList.index(comment[2])
    goldCount += 1
    subGold[currentSub] += 1
    if "gold" and "edit" in comment[0].lower():
        chumpCount += 1
        subChump[currentSub] += 1
        
goldCount = float(goldCount)

#print("goldCount",goldCount)
#print("chumpCount",chumpCount)

#####
labels = "Loudmouth Chumps","Silent Studs"
sizes = [chumpCount/goldCount*100,(goldCount-chumpCount)/goldCount*100]
colors = ["lightsalmon","khaki"]
fig1 = plt.figure()
plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.axis('equal')
plt.title('Chumps on reddit')
plt.savefig("1.GildedChumps.png")
#####

#print("subs with gold",len(subList))
chumpPercentList = []
for i in range(len(subList)):
    chumpPercentList.append(float(subChump[i]/float(subGold[i])*100))
    
allSubInfo = np.column_stack((subList,chumpPercentList,subGold))
#print(allSubInfo)
allSubInfo = allSubInfo.tolist()
#print(allSubInfo)
allSubInfo.sort(key=lambda x: float(x[1]), reverse=True)
bigSubInfo = []
for itm in allSubInfo:
    if int(itm[2]) >= 20:
        bigSubInfo.append(itm)
        
#####
names = [x[0] for x in bigSubInfo[:20]]
y_pos = np.arange(len(names))
values = [float(x[1]) for x in bigSubInfo[:20]]
#print(values)

fig2 = plt.figure()
plt.barh(y_pos, values, align='center', alpha=0.4)
plt.yticks(y_pos, names)
plt.gca().invert_yaxis()
plt.axis('tight')
plt.xlabel('Percent of gold recievers who are chumps (for subs with > 20 gildings)')
plt.title('Where are chumps most rampant?')
plt.savefig("2.ChumpSubs.png",bbox_inches='tight')