# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.
connection = sqlite3.connect("../input/database.sqlite")
df = pd.read_sql_query("SELECT * from data", connection)

# close sqlite connection
connection.close()
plt.xlabel("Language")
plt.ylabel("Count")
plt.title("Language used for tweets")

df.iloc[:,1].value_counts().plot(kind='bar')
plt.savefig('res1.png')

file ="../input/30days_geomapinterests.csv"
data = pd.read_csv(file,sep=",")
plt.plot(data['Emmanuel Macron'],data['Region'],color="red")
plt.savefig('res2.png')
