# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


f = open('../input/presidential_polls.csv')
csv_f = csv.reader(f)

topDelta = None
polls = []
pos = 0
for row in csv_f:
  if pos != 0:
    clintonVote = float(row[13])
    trumpVote = float(row[14])
    voteDelta = trumpVote - clintonVote
    
    if topDelta == None or trumpVote > clintonVote and voteDelta > topDelta:
      topDelta = voteDelta
      polls.append(row)

  pos = pos + 1

print(polls[0])
print(polls[1])
print(polls[2])
print(polls[3])

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.