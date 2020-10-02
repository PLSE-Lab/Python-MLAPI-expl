# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
import csv
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Importing the dataset

dataset = pd.read_csv('/kaggle/input/science-clips/science-clips.csv')

dataset = dataset[pd.isnull(dataset['Author']) == False]

dataset.index = pd.RangeIndex(len(dataset.index))

tempauthors = dataset.iloc[:, 2]
tempauthors.astype(str)

# Building Authors dataframe

authors = pd.DataFrame()
authors['Surname'] = []
authors['Name'] = []
authors['Articles Authored'] = []

# Creating temporary authors list

tempauthors = [(((tempauthors[i].replace(" ", "")).replace(".", "")).replace(";", ",")).split(",") \
               for i in range(tempauthors.size)]

# deleting last item of each row with more than 2 elements (it is void)

tempauthors = [tempauthors[i][:-1] for i in range(len(tempauthors)) if len(tempauthors[i]) > 2] + [tempauthors[i][:] for i in range(len(tempauthors)) if len(tempauthors[i]) <= 2]

# Deleting element in position 2 of every row (initials without surname attached)

tempauthors = [tempauthors[i][:1] + tempauthors[i][2:] for i in range(len(tempauthors)) if len(tempauthors[i]) > 2] + [tempauthors[i][:] for i in range(len(tempauthors)) if len(tempauthors[i]) <= 2]

#Generating list with Surname + Name elements

tempauthors2 = [tempauthors[i][2*j] + " " + tempauthors[i][2*j+1] for i in range(len(tempauthors)) for j in range(int(len((tempauthors[i]))/2))]

# Counting occurences

final_count = Counter(tempauthors2)

# Save data

w = csv.writer(open("output.csv", "w"))
for key, val in final_count.items():
    w.writerow([key, val])
