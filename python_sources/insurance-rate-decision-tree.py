# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

fields = ['StateCode','IssuerId','PlanId','RatingAreaId','Age','IndividualRate','BusinessYear']
rate_chunks = pd.read_csv("../input/Rate.csv",iterator = True, chunksize = 1000, usecols = fields)
rates = pd.concat(chunk for chunk in rate_chunks)
rates = rates[rates.IndividualRate > 0]
rates = rates[rates.IndividualRate < 2000]  # filter away the unusual rate data which bigger than 9000
rates.drop_duplicates()

train_data = rates[rates.BusinessYear != 2016]
test_data = rates[rates.BusinessYear == 2016]
train_data.head(n = 10)
test_data.head(n = 10)
rates.head(n = 10)

# Any results you write to the current directory are saved as output.
# In this notebook, I want to build a decison tree to find out the relation ship between insurance rate and
# (StateCode,IssuerId, PlanId, RatingAreaId, Age).
import matplotlib.pyplot as plt
print(rates.describe())

plt.hist(rates.IndividualRate.values)
