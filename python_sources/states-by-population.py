# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import style
style.use('ggplot')

#To display desired label format on y-axis
def millions(x, pos):
        return '%1dM' % (x*1e-6)

#Read Data
data=pd.read_csv('../input/cities_r2.csv')


#First Groupby States and then sum population_total of each city in order to convert groupbydata tp dataframe
data.rename(columns={'state_name':'States'},inplace=True)
a=data.groupby('States')
a=a['population_total'].sum()
a.sort_values(ascending=False,inplace=True)


#To set a color gradient for the bar plot
# my_colors = [(x/a, x/(a+5.0), 0.9) for x in range(a)]
# my_colors.reverse()

#Color gradient I prefer
my_colors=[(0.62,0,0.26),(0.84,0.24,0.31),(0.96,0.43,0.26),(0.99,0.68,0.38),(1,0.88,0.55),(0.9,0.96,0.6),
           (0.67,0.87,0.64),(0.4,0.76,0.65),(0.2,0.53,0.74),(0.27,0.46,0.71), (0.37,0.31,0.64)]


ax1=plt.subplot2grid((1,1),(0,0))

#Add title, xlabel,ylabel and adjust the subplot to fit the xlables
plt.title('States by total population',weight='light',position=(0.5,1.03),size=17)
plt.xlabel('States',weight='normal',size=17,color='K')#No effect here????
plt.ylabel('Population',size=17,color='K')
plt.subplots_adjust(left=0.07, right=0.97, top=0.9, bottom=0.3)

#Plot
a.plot.bar(color=my_colors)

#Add ytick,xtick

plt.xticks(size=12,color="K",rotation=80)
plt.yticks(size=12,color="K")
formatter=FuncFormatter(millions)
ax1.yaxis.set_major_formatter(formatter)
plt.savefig('States by Population.jpeg',dpi=600)
plt.show()

# Any results you write to the current directory are saved as output.