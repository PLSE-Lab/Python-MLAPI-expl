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
train=pd.read_csv("../input/cereal.csv")
print(train.describe())

#Load in visualization libraries.
import matplotlib.pyplot as plt
#get just one column of a dataframe, you can use the syntax dataframe[“columnName”]

df_rating=train["rating"]

#Plot a histogram of that column.
(figure,axes) =  plt.subplots(figsize=(20,10))
axes.hist(df_rating,bins=100)
#Don’t forget to add a title! :) Use the:Python: plt.title() command
plt.title("rating")
#show the figure in Notbook
plt.show()
#save the current figure
plt.savefig("Day2.png")

