# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#%matplotlib inline

# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
dataFood= pd.read_csv("../input/FoodFacts.csv", low_memory=True)
# Creating a copy of the original dataset as sub. All experiments will be done on sub
sub=dataFood
#Determine the number of rows and columns in the dataset
print (sub.shape)
print(sub.head())
print(sub.info())
plt.figure()
'''
fig, (axis1) = plt.subplots(1,1,figsize=(15,4))
sns.countplot(x='product_name',hue='carbon_footprint_100g', data=sub,palette="husl", ax=axis1)
'''