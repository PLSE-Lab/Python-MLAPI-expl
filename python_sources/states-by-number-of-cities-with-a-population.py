# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

#Read Data
data=pd.read_csv('../input/cities_r2.csv')
print(type(data))
ax1=plt.subplot2grid((1,1),(0,0))

#To print unique vales from a column
#a=len(data['state_name'].unique())
# print(a)

#To set a color gradient for the bar plot
# my_colors = [(x/a, x/(a+5.0), 0.9) for x in range(a)]
# my_colors.reverse()

#Color gradient I prefer
my_colors=[(0.62,0,0.26),(0.84,0.24,0.31),(0.96,0.43,0.26),(0.99,0.68,0.38),(1,0.88,0.55),(0.9,0.96,0.6),
           (0.67,0.87,0.64),(0.4,0.76,0.65),(0.2,0.53,0.74),(0.27,0.46,0.71), (0.37,0.31,0.64)]


#Add title, xlabel,ylabel and adjust the subplot to fit the xlables
plt.title('States by number of cities with a population of over 1 lakh',position=(0.5,1.03),size=17)
plt.xlabel('States',size=12,color='K')
plt.ylabel('Number of Cities',size=17,color='K')
plt.subplots_adjust(left=0.12, right=0.9, top=0.9, bottom=0.27)


#Cross check for missing data
print(len(data),data.state_name.value_counts().sum())

#Plotting by counting the occurence of the name of states in the state name column\
d=data.state_name.value_counts().plot.bar(color=my_colors,ax=ax1)
#print(data.state_name.value_counts())




#To change the parameters of lables for each bar in the plot
plt.xticks(size=12,color='K',rotation=80)

#To save the output figure
plt.savefig ('abc.jpeg',dpi=1200)
plt.show()


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.