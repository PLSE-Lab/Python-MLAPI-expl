#!/usr/bin/env python
# coding: utf-8

# ##Some of the plots available in seaborn 
# (This is my first kernel. Please give suggestions for improvement)
# 
# 
# 
# 
# 
# 

# In[ ]:


from google.colab import files
files.upload()


# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


# In[ ]:


data = pd.read_csv("../input/StudentsPerformance.csv")


# In[ ]:


print(data.columns)


# In[ ]:


# to get the first 6 rows
data.head(6)


# In[ ]:


#to get the last 3 rows
data.tail(3)


# In[ ]:


#to get 5 random values
data.sample(5)


# In[ ]:


data.describe()


# ###After plotting a graph, I removed or altered a few commands to see how they affect the graph.

# ## Bar Plot

# ##1.

# In[ ]:


sns.set(style='whitegrid')
ax=sns.barplot(x=data['gender'].value_counts().index,y=data['gender'].value_counts().values,palette="muted",hue=['female','male'])
plt.xlabel('gender')
plt.ylabel('Frequency')
plt.title('Gender Bar Plot')
plt.show()


# In[ ]:


plt.figure(figsize=(7,7))
sns.barplot(x=data['race/ethnicity'].value_counts().index,
              y=data['race/ethnicity'].value_counts().values,
              palette="muted")
#you can also write ax=sns.barplot(x=data['race/ethnicity'].value_counts().index etc
plt.xlabel('race/ethnicity')
plt.ylabel('frequency')
plt.title('Show of Race/Ethnicity Bar Plot')
plt.show()


# ##2.

# In[ ]:


sns.barplot(x = "parental level of education", y = "writing score", hue = "gender", data = data,palette='muted')
plt.xticks(rotation=30)
plt.show()


# ###Updated seaborn

# In[ ]:


get_ipython().system('pip3 install seaborn==0.9.0')
#catplot,boxenplot doesn't exist - this error was shown. Hence istalled the updated version using pip3


# ##3.

# In[ ]:


plt.figure(figsize=(10,10)) #I don't know the significance of this line since without this also the same plot appears.
sns.catplot(x="gender", y="math score",
                 hue="parental level of education",
                 data=data, kind="bar")
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.catplot(x="gender", y="math score",
                 hue="parental level of education",
                 data=data, kind="bar",height=1, aspect=10)                   #aspect - width of the rectangle,height - height of the rectangle/bar.
plt.show()                                                                    #try to change the  values of aspect and height to see the affect


# In[ ]:


plt.figure(figsize=(10,10))
sns.catplot(x="gender", y="math score",
                 hue="parental level of education",
                 data=data)                               #here I didn't mention (kind="bar").Have a look at the kind of plot
plt.show()                                                #This actually comes under stripplot and can be done using stripplot instead of catplot!!!


# ##4.

# In[ ]:


ax = sns.barplot("parental level of education", "writing score", data=data,
                  linewidth=2.5, facecolor=(1, 1, 1, 0),
                  errcolor=".2", edgecolor=".2")
plt.xticks(rotation=90)
plt.show()


# In[ ]:


sns.barplot("parental level of education", "writing score", data=data,linewidth=2, facecolor=(1,1,1,1),
                  errcolor=".2", edgecolor=".2")           #linewidth - the outline thickness of the bars
plt.xticks(rotation=90)                                    #facecolor(1,1,1,1)-the lines or grids in the background do not overlap the bars
                                                           #I still didn't understand the function of facecolor(0,0,0,0) or any other combinations!
plt.show()


# ##4.

# In[ ]:


plt.figure(figsize=(10,10))
# Draw a nested barplot to show survival for class and sex
g = sns.catplot(x="math score", y="writing score", hue="gender", data=data,
                height=6, kind="bar", palette="pastel")
g.despine(left=True)
g.set_ylabels("survival probability")
plt.tight_layout()
plt.xticks(rotation=90)
plt.show()


# In[ ]:


g = sns.catplot(x="math score", y="writing score", hue="gender", data=data, #How to create space between the various points on x axis?
                height=6, kind="bar", palette="pastel")                     #Without some commands (used above) also, same graph has been plotted!
g.set_ylabels("survival probability")                    


# ##5.

# In[ ]:


f,ax=plt.subplots(figsize=(9,10))
sns.barplot(x=data['gender'].value_counts().values,y=data['gender'].value_counts().index,alpha=0.5,color='red',label='gender')
sns.barplot(x=data['race/ethnicity'].value_counts().values,y=data['race/ethnicity'].value_counts().index,color='blue',alpha=0.7,label='race/ethnicity')
ax.legend(loc='upper right',frameon=True)
ax.set(xlabel='gender , race/ethnicity',ylabel='groups',title="gender vs race/ethnicity ")
plt.show()


# In[ ]:


#f,ax=plt.subplots(figsize=(9,10))        #this decides the size of the bar, also shows the labels/names of both the axis
sns.barplot(x=data['gender'].value_counts().values,y=data['gender'].value_counts().index,alpha=0.5,color='red',label='gender')
sns.barplot(x=data['race/ethnicity'].value_counts().values,y=data['race/ethnicity'].value_counts().index,color='blue',alpha=0.7,label='race/ethnicity')
ax.legend(loc='upper right',frameon=True) #I don't know the function of this 
ax.set(xlabel='gender , race/ethnicity',ylabel='groups',title="gender vs race/ethnicity ")
plt.show()


# In[ ]:


f,ax=plt.subplots(figsize=(9,10))
sns.barplot(x=data['gender'].value_counts().values,y=data['gender'].value_counts().index,alpha=0.9,color='blue',label='gender') 
# if you want to change the colour, write that in ''. Alpha decides the intensity or transperency of the color. Try alpha =0.2 and 0.9.
#You can makeout the difference.
sns.barplot(x=data['race/ethnicity'].value_counts().values,y=data['race/ethnicity'].value_counts().index,color='green',alpha=0.9,label='race/ethnicity')
ax.set(xlabel='gender , race/ethnicity',ylabel='groups',title="gender vs race/ethnicity ")
plt.show()


# ## Point Plot

# ##1.

# In[ ]:


#data.rename(columns=({'maths score':'ms','reading score':'rs'}),inplace=True)


# In[ ]:


data['race/ethnicity'].unique()
len(data[(data['race/ethnicity']=='group B')].ms)
f,ax1=plt.subplots(figsize=(25,10))
sns.pointplot(x=np.arange(1,191),y=data[(data['race/ethnicity']=='group B')].ms,color='lime',alpha=0.8)
sns.pointplot(x=np.arange(1,191),y=data[(data['race/ethnicity']=='group B')].rs,color='red'
plt.ylabel(('frequency'),alpha=0.5)
plt.xlabel('Group B index State')
plt.title('Group B Math Score & Reading_Score')
plt.xticks(rotation=90)
plt.grid()
plt.show()


# ##2.

# In[ ]:


ax = sns.pointplot(x="reading score", y="math score", hue="gender",data=data)
plt.xticks(rotation=90)
plt.show()


# In[ ]:


sns.pointplot(x="reading score", y="math score",data=data) # hue="gender" - it separates the plot into female and male in this case
plt.xticks(rotation=90)
plt.show()


# ##3.

# In[ ]:


ax = sns.pointplot(x="reading score", y="writing score", hue="gender",data=data,markers=["o", "x"],linestyles=["-", "--"])
plt.xticks(rotation=90)
plt.show()


# In[ ]:


ax = sns.pointplot(x="reading score", y="writing score", hue="gender",data=data) # even without this: linestyles=["-", "--"]; got the same plot. 
plt.xticks(rotation=90)                                                                             #I don't know the use of this
plt.show()                                                                                          #the shapes of the points which are marked on plot markers=["o", "x"]
                                                                                # default marker: "o"


# ## Joint Plot

# In[ ]:


data.rename(columns={'math score':'ms','reading score':'rs'},inplace=1)


# ##1.

# In[ ]:


plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['race/ethnicity']=='group B')].ms,color='blue',alpha=0.8)
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Frequency Race/Ethnicity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[ ]:


#plt.figure(figsize=(25,25))  #What's the use of this?
sns.jointplot(x=np.arange(1,191),y=data[(data['race/ethnicity']=='group B')].ms,color='blue',alpha=0.8)
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Frequency Race/Ethnicity')
plt.xticks(rotation=90)
#plt.tight_layout() used to avoid the overlap of axis label and coordinates on that axis.
plt.show()


# ##2.

# In[ ]:


plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['race/ethnicity']=='group B')].ms,color='lime',kind='hex',alpha=0.8)
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Frequency Race/Ethnicity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[ ]:


sns.jointplot(x=np.arange(1,191),y=data[(data['race/ethnicity']=='group B')].ms,color='lime',alpha=0.8) #If kind='hex' isn't mentioned, then dots are plotted instead of hexagon 
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Frequency Race/Ethnicity')
plt.xticks(rotation=90)
plt.show()


# ##3.

# In[ ]:


plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['race/ethnicity']=='group B')].ms,color='red',space=0,kind='kde')
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Frequency Race/Ethnicity')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[ ]:


sns.jointplot(x=np.arange(1,191),y=data[(data['race/ethnicity']=='group B')].ms,color='red')
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Frequency Race/Ethnicity')
plt.xticks(rotation=90)
plt.show()
#space=0 ,kind='kde' withouteither of them  it shows syntax error
# and if both are removed then a plot same as jointplot 1. shows up.


# ##4.

# In[ ]:


data['race/ethnicity'].unique()
len(data[(data['race/ethnicity']=='group B')].ms)
plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['race/ethnicity']=='group B')].rs,color='red').plot_joint(sns.kdeplot, zorder=0, n_levels=6)
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Math Score & Reading_Score')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[ ]:


data['race/ethnicity'].unique()
len(data[(data['race/ethnicity']=='group B')].ms)
#plt.figure(figsize=(10,10))
sns.jointplot(x=np.arange(1,191),y=data[(data['race/ethnicity']=='group B')].rs,color='red').plot_joint(sns.kdeplot, zorder=0, n_levels=20) #n_levels=k, then k no. of loops are plotted
plt.xlabel('Group B index State')
plt.ylabel('Frequency')
plt.title('Group B Math Score & Reading_Score')
plt.xticks(rotation=90)
plt.tight_layout() #here if this command is not used, then 'Group B Math Score & Reading_Score' overlaps some area of the graph
plt.show()


# ## Pie Chart

# ##1.

# In[ ]:


labels=data['race/ethnicity'].value_counts().index
colors=['blue','red','yellow','green','brown']
explode=[0,0,0.1,0,0]
values=data['race/ethnicity'].value_counts().values

#visualization
plt.figure(figsize=(7,7))
plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title('Race/Ethnicity According Analysis',color='black',fontsize=10)
plt.legend(['group A', 'group B','group C','group D','group E'] , loc=5) #gives a box having colours and it's category eg:red:group B
plt.tight_layout()                                                       #I don't know how to avoid the overlapping even after using tight layout
plt.axis('equal')
plt.show()


# In[ ]:


labels=data['race/ethnicity'].value_counts().index
colors=['blue','red','yellow','green','brown']
explode=[0,0,0.9,0,0.5] #removing the sector; 0-keeps intact, 0.9-goes relatively far,[group C,D,B,E,A]
values=data['race/ethnicity'].value_counts().values
plt.figure(figsize=(2,2)) #figsize: decides the size of the pie chart
plt.pie(values,explode=explode,labels=labels,colors=colors,autopct='%1.1f%%')
plt.title('Race/Ethnicity According Analysis',color='green',fontsize=10) #font size, colour of the title of the plot
plt.show()


# ##Lm Plot

# ##1.

# In[ ]:


sns.lmplot(x='ms',y='rs',data=data)
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.title('Math Score vs Reading Score')
plt.show()


# ##2.

# In[ ]:


sns.lmplot(x='ms',y='writing score',hue='gender',data=data,markers=['x','o'])
plt.xlabel('Math Score')
plt.ylabel('Writing Score')
plt.title('Math Score vs Writing Score')
plt.show()


# ##Kde Plot

# ##1.

# In[ ]:


sns.kdeplot(data['rs'],shade=True,color='r')              #shade= True then the line under is shaded and viceversa for false
sns.kdeplot(data['writing score'],shade=False,color='b')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Reading Score vs Writing Score Kde Plot System Analysis')
plt.show()


# In[ ]:


sns.kdeplot(data['rs'],shade=True,color='r')              #shade= True then the line under is shaded and viceversa for false
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Reading Score vs Writing Score Kde Plot System Analysis')
plt.show()


# ##2.

# In[ ]:


sns.kdeplot(data['rs'],data['writing score'],shade=False)


# In[ ]:


sns.kdeplot(data['rs'],data['writing score'],shade=True)


# In[ ]:


sns.kdeplot(data['rs'],data['writing score'],cmap='Reds',shade=True,shade_lowest=False)
sns.kdeplot(data['writing score'],data['rs'],cmap='Blues',shade=True,shade_lowest=False)
plt.show()


# In[ ]:


sns.kdeplot(data['rs'],data['writing score'],cmap='Reds',shade=False,shade_lowest=True)
sns.kdeplot(data['writing score'],data['rs'],cmap='Blues',shade=False,shade_lowest=True)
plt.show()


# ##Heatmap Plot

# In[ ]:


print(data.corr)


# ##1.

# In[ ]:


sns.heatmap(data.corr()) #I didn't understand why only 3 columns have been taken but when data.corr is printed it shows more than 3 columns??
plt.show()


# ##2.

# In[ ]:


sns.heatmap(data.corr(),annot=True)
plt.show()


# In[ ]:


sns.heatmap(data.corr(),cmap='YlGnBu',annot=True)
plt.show()


# ##3.

# In[ ]:


# Compute the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(6, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(10, 110, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# ##Box Plot

# ##1.

# In[ ]:


sns.set(style='whitegrid')
sns.boxplot(data['ms'])
plt.show()


# ##2.

# In[ ]:


sns.boxplot(x=data['gender'],y=data['ms'])
plt.show()


# ##3.

# In[ ]:


sns.boxplot(x=data['race/ethnicity'],y=data['writing score'],hue=data['gender'],dodge=False)
plt.show()


# ##4.

# In[ ]:


get_ipython().system('pip3 install seaborn==0.9.0')
#even after installing this I am not able to remove the error :(


# In[ ]:


sns.boxenplot(x=data['race/ethnicity'],y=data['writing score'],hue=data['gender'],palette="Set1")
plt.show()


# In[ ]:


sns.boxplot(x="race/ethnicity", y="writing score",
              color="b",
              scale="linear", data=data)
plt.show()


# ##Pair Plot

# ##1.

# In[ ]:


sns.pairplot(data)
plt.show()


# ##2.

# In[ ]:


sns.pairplot(data,diag_kind='kde')
plt.show()


# In[ ]:


sns.pairplot(data,diag_kind='reg') #diag: diagonal 
plt.show()


# In[ ]:


sns.pairplot(data, diag_kind="kde", markers="o",
                  plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                  diag_kws=dict(shade=True))
plt.show()


# In[ ]:


sns.pairplot(data, diag_kind="kde", markers="*",
                  plot_kws=dict(s=50, edgecolor="b", linewidth=1),
                  diag_kws=dict(shade=True))
plt.show()


# ##Count Plot

# 1.

# In[ ]:


sns.countplot(data['gender'])
plt.show()


# In[ ]:


sns.countplot(data['race/ethnicity'],hue=data['gender'])
plt.show()


# ##2.

# In[ ]:


sns.countplot(data['ms'],hue=data['race/ethnicity'])
plt.show()


# In[ ]:


sns.countplot(data['race/ethnicity'],hue=data['gender'])
plt.show()


# ##3.

# In[ ]:


sns.countplot(y=data['parental level of education'],palette="Set1",hue=data['gender'])
plt.legend(loc=4)
plt.show()


# ##4.

# In[ ]:


sns.countplot(x=data['lunch'],facecolor=(0,0,0,0),linewidth=5,edgecolor=sns.color_palette('dark',1))
plt.show()


# ##Strip Plot

# ##1.

# In[ ]:


sns.stripplot(x=data['ms'])
plt.show()


# ##2.

# In[ ]:


sns.stripplot(x="parental level of education",y='writing score',data=data)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


sns.stripplot(x="ms",y='gender',data=data)
plt.xticks(rotation=45)
plt.show()


# ##3.

# In[ ]:


sns.stripplot(x='test preparation course',y='rs',hue='gender',jitter=True,data=data)
plt.show()


# ##4.

# In[ ]:


sns.stripplot(x='lunch',y='ms',hue='lunch',jitter=True,dodge=True,size=20,marker='D',edgecolor='gray',alpha=.25,palette="Set2",data=data)
plt.legend(loc=10)
plt.show()


# ##DisPlot

# ##1.

# In[ ]:


ax = sns.distplot(data['ms'], rug=True, hist=False)
plt.show()


# ##2.

# In[ ]:


ax = sns.distplot(data['ms'], vertical=True) #vertical=true gave horizontal bars and viceversa for false
plt.show()


# In[ ]:


ax = sns.distplot(data['ms'], vertical=False, color="r") #use color=" " to get desired color
plt.show()


# ##3.

# In[ ]:


sns.set(style="white", palette="muted", color_codes=True)
rs = np.random.RandomState(10)

f, axes = plt.subplots(2, 2, figsize=(7, 7), sharex=True)
sns.despine(left=True)

# Generate a random univariate dataset
d = rs.normal(size=100)

# Plot a simple histogram with binsize determined automatically
sns.distplot(d, kde=False, color="y", ax=axes[0, 0])

# Plot a kernel density estimate and rug plot
sns.distplot(d, hist=False, rug=True, color="r", ax=axes[0, 1])

# Plot a filled kernel density estimate
sns.distplot(d, hist=False, color="g", kde_kws={"shade": True}, ax=axes[1, 0])

# Plot a historgram and kernel density estimate
sns.distplot(d, color="b", ax=axes[1, 1])

plt.setp(axes, yticks=[])
plt.tight_layout()
plt.show()


# ##Line Plot

# ##1.

# In[ ]:


data[data['gender']=='male']['ms'].value_counts().sort_index().plot.line(color='g')
data[data['gender']=='female']['ms'].value_counts().sort_index().plot.line(color='r')
plt.xlabel('Math_Score')
plt.ylabel('Frequency')
plt.title('Math_Score vs Frequency')
plt.show()


# ##2.

# In[ ]:


sns.lineplot(x='ms',y='rs',data=data) #How to remove the error?
plt.show()


# In[ ]:


female_filter=data[data['gender']=='female']
sns.lineplot(x='rs',y='writing score',data=female_filter,
            hue='lunch',style='test preparation course',dashes=False)
plt.show()


# ##References
# 
# 
# *   kaggle kernels pull kralmachine/seaborn-tutorial-for-beginners
# *   https://seaborn.pydata.org/
# 
# 

# ##Conclusion:
# This is my first kernel.There were some errors which I couldn't remove so please give suggestions. Upvote if it was useful :)
