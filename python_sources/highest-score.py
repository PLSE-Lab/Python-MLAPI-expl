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
deliveries=pd.read_csv("../input/deliveries.csv")
col_Names = deliveries.columns.tolist()
print(col_Names)

deliveries['batsman']
deliveries['batsman'].unique()

highest_individual_score = {}
for b in deliveries['batsman'].unique():
    All_balls_faced = deliveries[(deliveries['batsman'] == b)]
    list_of_scores = [];
    for i in All_balls_faced['match_id'].unique():
        innings_balls_faced = deliveries[(deliveries['match_id'] == i) & (deliveries['batsman'] == b)]
        innings_score = innings_balls_faced['batsman_runs'].sum()
        list_of_scores.append(innings_score)
    highest_score = max(list_of_scores)
    highest_individual_score[b] = highest_score
#print(highest_individual_score)
print(type(highest_individual_score))
Highest_score=sorted(highest_individual_score, key=highest_individual_score.get, reverse=True)[:5]
print(Highest_score)

maximum = max(highest_individual_score, key=highest_individual_score.get)  # Just use 'min' instead of 'max' for minimum.
print(maximum, highest_individual_score[maximum])
