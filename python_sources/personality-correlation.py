# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from pandas import DataFrame, Series


data = pd.read_csv('C:/Users/Muzammil/Desktop/responses_young_people.csv')
data.describe()

data.columns # pull out every variables

import seaborn as sns 
#using seaborn for visualising 

matplotlib inline  
# '%' is important


T = traits.iloc[:,:].corr() #'.corr' to create correlational matrix
plot.figure(figsize=(50, 50)) # size of heat map
Ts = sns.heatmap(T, annot=True,linewidths=1.0, fmt="0.00", square=True)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.