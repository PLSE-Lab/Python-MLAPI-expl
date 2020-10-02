import re
import sqlite3
import matplotlib.pyplot as plt
import numpy
from collections import Counter

conn = sqlite3.connect('../input/database.sqlite')

c = conn.cursor()

my_dict={}
for row in c.execute('SELECT subreddit FROM May2015 ORDER BY ups LIMIT 100'):
    if row in my_dict:
        my_dict[row] += 1
    else:
        my_dict[row] = 1

highest = Counter(my_dict).most_common()
labels, values = zip(*highest)
indexes = numpy.arange(len(labels))

plt.barh(indexes, values)
plt.yticks(indexes, labels)
plt.tight_layout()
plt.savefig('please.png')

"""
plt.barh(range(len(my_dict)), my_dict.values(), align = 'center')
plt.yticks(range(len(my_dict)), list(my_dict.keys()))

plt.savefig('please.png')
"""
"""
list_of_values = []
for entry in my_dict:
    list_of_values.append(my_dict[entry])


list_of_values.sort()

for entry in my_dict:
    if my_dict[entry] == list_of_values[-1]:
        print (entry, my_dict[entry])

created_utc
ups
subreddit_id
link_id
name
score_hidden
authorflaircss_class
authorflairtext
subreddit
id
removal_reason
gilded
downs
archived
author
score
retrieved_on
body
distinguished
edited
controversiality
parent_id
"""