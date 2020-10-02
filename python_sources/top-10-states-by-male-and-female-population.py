import pandas as pd
from matplotlib import style
style.use('ggplot')
from pylab import *


#To display original percentage on pie chart
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = float ( (pct * total / 100.0))
        return ' {v:.4f}%'.format(v=val)
    return my_autopct

a=0
#To display original data on pie chart
def make_autopct_male(values):
    def my_autopct(pct):
        global a
        total = sum(values)
        val = int ( (pct * total / 100.0))
        val=values.ix[a]
        a = a + 1
        return ' {v:d}'.format(v=val)

    return my_autopct


#Read File
data=pd.read_csv('../input/cities_r2.csv')
ax1=plt.subplot2grid((1,2),(0,0))
ax2=plt.subplot2grid((1,2),(0,1))


#Intialise 10 color lisr from colorbrewery
my_colors=[(0.62,0,0.26),(0.84,0.24,0.31),(0.96,0.43,0.26),(0.99,0.68,0.38),(1,0.88,0.55),(0.9,0.96,0.6),
           (0.67,0.87,0.64),(0.4,0.76,0.65),(0.2,0.53,0.74),(0.27,0.46,0.71), (0.37,0.31,0.64)]


#Preapre data for Male Pie Chart
data_male=data.groupby(by='state_name')
data_male=data_male['population_male'].sum()
data_male.sort_values(ascending=False,inplace=True)

#Add boundary around subplot 1
rec = Rectangle((-1.7,-1.2),3.1+0.2, 2.2+0.2,fill=False,lw=3,linestyle="solid",edgecolor="black")
rec=ax1.add_patch(rec)
rec.set_clip_on(False)

#Plot Pie Chart for male
data_male.head(10).plot.pie(autopct=make_autopct_male(data_male.head(10)),ax=ax1,colors=my_colors,fontsize=16)
ax1.yaxis.label.set_visible(False)
ax1.set_title(label='Top 10 states by total male population')


#                            '''Female'''
#Preapre data for Female Pie Chart
data_female=data.groupby(by='state_name')
data_female=(data_female['population_female'].sum()/data_female['population_total'].sum())
data_female.sort_values(ascending=False,inplace=True)
# print(data_female.head(10))

#Add boundary around subplot 2
rec = Rectangle((1.6,-1.2),3+0.2, 2.2+0.2,fill=False,lw=3,linestyle="solid",edgecolor="black")
rec=ax1.add_patch(rec)
rec.set_clip_on(False)

#Plot Pie Chart for male
data_female.head(10).plot.pie(ax=ax2,autopct=make_autopct(data_female.head(10)),colors=my_colors,fontsize=16)
ax2.yaxis.label.set_visible(False)
ax2.set_title(label="Top 10 states by female population proportion")



plt.show()
