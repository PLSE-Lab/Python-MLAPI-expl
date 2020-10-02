# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

matplotlib.style.use('ggplot')


df = pd.read_csv("../input/data.csv")

print (df.head())


#feature engineering : category of products


df['category'] = df.Description.str.split().str[-1]

print  (df.category.value_counts().head(10))

#time series [ month wises ]

df.month = df.InvoiceDate.str.split('/').str[0]

print ("Sales by Month : \n", df.month.value_counts())

plt.scatter(df.month.value_counts().keys(),df.month.value_counts(), c='b', s=150, alpha=0.5)
plt.xlabel('Months 1-Jan, 2-Feb.....12-Dec')
plt.ylabel('Number of purchases')
plt.title('Time Series analysis by Month Jan-Dec')
plt.show()

##clearly visible that as christmas approches, people shop more

print (df.Country.value_counts())