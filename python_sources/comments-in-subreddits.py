import sqlite3
import matplotlib.pyplot as plt
from collections import Counter

conn = sqlite3.connect('../input/database.sqlite')

c = conn.cursor()

subreddit_dict = {}

default_subs = ["announcements",
"Art",
"AskReddit",
"askscience",
"aww",
"blog",
"books",
"creepy",
"dataisbeautiful",
"DIY",
"Documentaries",
"EarthPorn",
"explainlikeimfive",
"Fitness",
"food",
"funny",
"Futurology",
"gadgets",
"gaming",
"GetMotivated",
"gifs",
"history",
"IAmA",
"InternetIsBeautiful",
"Jokes",
"LifeProTips",
"listentothis",
"mildlyinteresting",
"movies",
"Music",
"news",
"nosleep",
"nottheonion",
"OldSchoolCool",
"personalfinance",
"philosophy",
"photoshopbattles",
"pics",
"science",
"Showerthoughts",
"space",
"sports",
"television",
"tifu",
"todayilearned",
"TwoXChromosomes",
"UpliftingNews",
"videos",
"worldnews",
"WritingPrompts",
]

for subreddit in c.execute('SELECT subreddit FROM May2015 ORDER BY ups LIMIT 100'):
    sub_string = ""
    sub_string += subreddit[0]
    if sub_string in default_subs:
        if sub_string in subreddit_dict:
            subreddit_dict[sub_string] += 1
        else:
            subreddit_dict[sub_string] = 1

def make_percent(sub):
    total_comments = 0
    for entry in subreddit_dict:
        total_comments += subreddit_dict[entry]
    percent = subreddit_dict[sub] / total_comments * 100
    return percent

total = 0
percent_list = []
sub_list = []
for sub in subreddit_dict:
    sub_list.append(sub)
    percent = make_percent(sub)
    percent_list.append(percent)
    total += percent

plt.pie(percent_list, explode = None, labels = sub_list)
plt.savefig("pie.png")