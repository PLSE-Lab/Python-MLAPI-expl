# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/heart.csv')
print('Displaying the first five rows')
df.head()
print("Displaying the last five rows")
df.tail()
df.describe()
df.columns
df.shape
df.isnull().sum()
df.corr()
sns.barplot(x=df.age.value_counts()[:10].index,y=df.trestbps.value_counts()[:10].index)
plt.title("Age vs resting blood pressure")
plt.xlabel('Age')
plt.ylabel('trestbps')
plt.show()
df1=df.age[:10]
print(df1)
print("Regression plot for age vs chest pain")
sns.regplot(x='age',y='cp',data=df,color='blue',marker='+')
df.sort_values(['chol'], ascending='False', axis=0, inplace=True)
df2=df.head()
df2=df2['age'].transpose()
sns.set_style('darkgrid')
plt.plot(df2)
plt.title('Variation of cholestrol with age')
plt.xlabel('Cholestrol')
plt.ylabel('Age')
plt.show()
print("Analysis of heart patients in young, middle and old ages")
colors=['red','blue','green']
young_ages=df[(df.age>=29)&(df.age<40)]
middle_ages=df[(df.age>40)&(df.age<60)]
old_ages=df[(df.age>=60)]
plt.figure(figsize = (5,5))
plt.pie([len(young_ages),len(middle_ages),len(old_ages)],labels=['young ages','middle ages','old ages'],colors=colors,autopct='%1.1f%%')