# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)import pandas as pd

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
#To alter the parameters of the graph
import matplotlib as mpl
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


df=pd.read_csv('../input/FoodFacts.csv',low_memory=False) #Reading the data


#Color gradient I prefer
my_colors=[(0.62,0,0.26),(0.84,0.24,0.31),(0.96,0.43,0.26),(0.99,0.68,0.38),(1,0.88,0.55),(0.9,0.96,0.6),
           (0.67,0.87,0.64),(0.4,0.76,0.65),(0.2,0.53,0.74),(0.27,0.46,0.71), (0.37,0.31,0.64)]


#Most common traces in foods and allergies
traces_list=['milk','nuts','eggs','gluten','peanuts','soybeans','mustard','celery','fish']
lst=[]
#Search the traces columns to check if they conatin any of the common traces
for trace in traces_list:
    lst.append(sum(df['traces_en'].dropna().apply(lambda x: +1 if str(trace) in x.lower() else 0)))
    
#Capitalize the xtickers    
traces_list=map(lambda x:x.title(),traces_list)    
    
#Series to plot 
top_traces=pd.Series(index=traces_list,data=lst)
#Sort
top_traces.sort_values(inplace=True)


#plotting
sns.set_style('white') #Set the canvas to white background
plt.figure(figsize=(10,8),dpi=600) #Changing the figure size and increasing the dpi of the plot
top_traces.plot.barh(color=my_colors) #Plotting a horizaontal plot and setting colors to my colors 
plt.title("Number of Products containing Traces",fontsize=18,x=0.5,y=1.1) # Setting a title and changing its x and y postiton co-ordinate
plt.ylabel("Traces",fontsize=16) #Setting the ylabel 
# Chaning the labelsize of both the x and y axis tickers
mpl.rcParams['xtick.labelsize'] = 14 
mpl.rcParams['ytick.labelsize'] = 16
# Removing the spine i.e. boundaries where no data is plotted
sns.despine()

plt.savefig('Number of Products containing common Traces.jpeg',dpi=600)
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.