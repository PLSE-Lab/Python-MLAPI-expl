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
path = r'../input/diskspace_data.csv'
frame = pd.read_csv(path, names=['MACHINE', 'DRIVE', 'TOTAL', 'USED', 'FREE', 'DATE'])
# print(frame[:10])

cols = ['MACHINE', 'DRIVE', 'DATE']
cols2 = ['MACHINE', 'DRIVE']
frame_sorted = frame.sort_values(cols, ascending=True)
frame_short = frame_sorted.drop(['TOTAL', 'USED'], axis=1)
grouped2 = frame_short.groupby(cols2, sort=True).mean()
# print(frame_short[:200])
# This is not what is wanted
print(grouped2[:200])
#                          FREE
#MACHINE       DRIVE           
#CORPMGTCAP1   C:     51.666667
#              D:     96.000000
#              E:     21.333333
#              F:     73.250000
#              G:      9.000000

#The goal is to get for each drive the moving average (for example, for 10 days)