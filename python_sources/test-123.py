import sqlite3
import pandas as pd
import re
import matplotlib.pyplot as plt

sql_conn = sqlite3.connect('../input/database.sqlite')

Users = pd.read_sql("SELECT author, author_flair_text FROM May2015 WHERE subreddit = 'ukpolitics' AND author_flair_text <> 'None'", sql_conn)
Users.drop_duplicates(inplace = True)

def find_politics(flair):
    pattern = r"[-]*\d+.\d+"
    politics = re.findall(pattern, str(flair))
    if len(politics) == 2:
        return politics
    else:
        return 'NA'

Users['Politics'] = Users['author_flair_text'].apply(find_politics)
Users = Users[Users.Politics != "NA"]
Users['Economics'] =  Users['Politics'].str[0]
Users['Social'] =  Users['Politics'].str[1]
Users[['Economics', 'Social']] = Users[['Economics', 'Social']].convert_objects(convert_numeric=True)

plt.scatter(x = Users['Economics'], y = Users['Social'])
plt.xlim([-10,10])
plt.ylim([-10,10])
plt.annotate('Authoritarian', xy=(0, 0), xytext=(-14, 6))
plt.annotate('Libertarian', xy=(0, 0), xytext=(-14, -6))
plt.annotate('Left Wing', xy=(0, 0), xytext=(-6, -12))
plt.annotate('Right Wing', xy=(0, 0), xytext=(3.5, -12))
plt.plot([0, 0], [-10, 10], 'k-', lw=0.5)
plt.plot([-10, 10], [0, 0], 'k-', lw=0.5)


plt.savefig("output.png", bbox_inches="tight")
