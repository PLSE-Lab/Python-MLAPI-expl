# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
training_df=pd.read_csv('../input/menu.csv')
print(training_df.info())
g = sns.FacetGrid(training_df, hue="Calories", col="Category")
g.map(plt.scatter, "Calories", "Sugars")
g.add_legend()
g = sns.factorplot(x="Calories", y="Category",data=training_df, saturation=.5,kind="bar", ci=None, aspect=.6)
plt.show()

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.