#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import sqlite3
import matplotlib.pyplot as plt
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#fetching the name of different fields
name = []
#create the connection with database
sqlite_database = '../input/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `name` FROM `seprate` DESC LIMIT 0, 5000;")
#store all name in as list
init_name = c.fetchall()
for each in init_name:
    text = (str(each)[2:len(each)-4]).replace("\\n","")
    name.append(text)
#close the connection with database
conn.close()

#fetching the number of publication field wise
sep = []
#connection create with database
sqlite_database = '/home/neel/scopus_data/scopus_data.sqlite'
conn = sqlite3.connect(sqlite_database)
c = conn.cursor()
c.execute("SELECT `number` FROM `seprate` DESC LIMIT 0, 5000;")
#store the data in sep as list
sep = c.fetchall()
#connection close with databae
conn.close()

#create a list of realtive percentage for publish paper field wise
per = []
for n in sep:
    text = str(n)[1:len(n)-3]
    n_to_per = int(text)
    val = (n_to_per*100)/1387
    val_2 = "%.2f"%val
    per.append(val_2)

#---------------------------Graph code------------------------------
label = []
x = 0
while x < len(per):
    label.append(str(name[x].upper())+" : "+str(per[x])+"%")
    x += 1

labels = label
sizes = per
patches, texts = plt.pie(sizes, startangle=90)
plt.legend(patches, labels,bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.title('Research done by Universities and Industries\n from 2001 to 2016\n Source: SCOPUS journal ')
plt.tight_layout()
plt.show()

