# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

sqlite_file = '../input/database.sqlite'
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

c.execute('Select * from May2015 LIMIT 100000')
print('fetch started')
all_rows = c.fetchall()
print('fetch finished')

randints = np.random.randint(100000, size=10)

for i in randints: 
    print(all_rows[i])

# Any results you write to the current directory are saved as output.