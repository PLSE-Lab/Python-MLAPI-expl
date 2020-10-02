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

data = pd.DataFrame.from_csv("../input/HR_comma_sep.csv")


for i in range(len(data)):
    print(data.iloc[i])
# def AB_test(data, category):
#     """ Puts the satisfaction level into buckets, and then compares the desired category's
#         satisfaction levels.
#     """ 
#     level1 = 0
#     level2 = 0
#     level3 = 0
#     level4 = 0
#     level5 = 0
    
#     for item in data:
#         if item[0] < 0.21:
#             level1 += 1
#         elif item[0] > 0.20 and item[0] < 0.41:
#             level2 += 1
#         elif item[0] > 0.40 and item[0] < 0.61:
#             level3 += 1
#         elif item[0] > 0.60 and item[0] < 0.81: 
#             level4 += 1
#         else:
#             level5 += 1
            
#     return (level1, level2, level3, level4, level5)
    
# print(AB_test(data, "last_evaluation"))