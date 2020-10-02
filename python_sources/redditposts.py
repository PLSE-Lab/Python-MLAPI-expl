# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import sqlite3
import matplotlib.pyplot as plt
import numpy
from collections import Counter
"""
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
plt.savefig('please.png')

"""
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
"""
"""
empty_dict = {}
words_dict = {}
string = 'dog 1234'
empty_dict[string]= 0
while string.isalpha() == True:
    empty_dict[string] += 1
    break 
if string.isalpha() == False:
    print (empty_dict)
print (empty_dict)
"""
class String_to_Dict():
    def __init__(self, string):
        self.
    
string = "Youareadog"
#for i in range(comment_string):
    #if word.isalpha() == True:
        #comment_string[:
string.capwords()
"""