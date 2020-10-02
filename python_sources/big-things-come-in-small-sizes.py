# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sqlite3

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from subprocess import check_output
print('Files in "input" directory:')
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
print("This script serves as a Python sandbox for exploring the data.")
# Connect to database
conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM FoodFacts")
result = c.fetchall()
print("The database contains", result[0][0], "rows.")
c.execute("SELECT product_name FROM FoodFacts WHERE product_name LIKE '%cheerios%'")
result = c.fetchall()
for record in result:
    print(record[0])
conn.close()
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]
plt.figure(num=1, figsize=(10, 6))
plt.plot(x, y)
plt.show()