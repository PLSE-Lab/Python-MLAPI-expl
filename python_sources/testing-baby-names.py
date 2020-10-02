# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv as csv
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

test_file = open('../input/NationalNames.csv', 'r') # open file
test_file_object = csv.reader(test_file) #decalre object
# header = test_file_object.next() #get rit of header

data = []
for row in test_file_object:
    data.append(row)
data = np.array(data)

print(data[0])

Katherine_count = 0
Katherine_data = []
years = []
for row in data:
    if row[1] == 'Katherine':
        Katherine_count = Katherine_count + int(row[4])
        Katherine_data.append(row[4])
        years.append(row[2])
print("data received")

print(Katherine_count)

plt.plot(Katherine_data, years, 'ro')
plt.show()
print('graph done')
