#!/usr/bin/env python
# coding: utf-8

# <h1>Data Analysis of Heart Disease </h1>

# <h2>Import Required Modules</h2>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(color_codes=True)  # visualization tool


from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")


# <h2>Read Data</h2>

# In[ ]:


df=pd.read_csv("../input/heart.csv")


# In[ ]:


type(df)


# In[ ]:


df.head()


# In[ ]:


df.tail()


# In[ ]:


df.shape


# In[ ]:


df.describe()


# In[ ]:


df.info()


# <h3>Data Cleaning</h3>

# **Count the number ofmissing values in the DataFrame**

# In[ ]:


#count the number of missing values in each columns
df.isna().sum()


# In[ ]:


df.columns


# In[ ]:


df.dtypes


# In[ ]:


df.corr()


# <h2>Data Visualization</h2>

# In[ ]:


#subplots
df.plot(subplots=True,figsize=(18,18))
plt.show()


# In[ ]:


#visualize the correlation
plt.figure(figsize=(15,10))
sns.heatmap(df.iloc[:,0:15].corr(), annot=True,fmt=".0%")
plt.show()


# In[ ]:


#create a pair plot
sns.pairplot(df.iloc[:,0:8],hue="cp")
plt.show()


# In[ ]:


fig=plt.figure(figsize=(20,15))
ax=fig.gca()
df.hist(ax=ax)
plt.show()


# In[ ]:


fig=plt.figure(figsize=(20,10))
sns.boxplot(data = df,notch = True,linewidth = 2.5, width = 0.50)
plt.show()


# In[ ]:


# histogram subplot with non cumulative and cumulative 
fig,axes=plt.subplots(nrows=2,ncols=1)

df.plot(kind='hist',y='age',bins=50,range=(0,100),density=True,ax=axes[0])
df.plot(kind='hist',y='age',bins=50,range=(0,100),density=True,ax=axes[1],cumulative=True)
plt.show()


# In[ ]:


print(df['sex'].value_counts(dropna=False))


# In[ ]:


sns.barplot(x='sex',y='age',data=df)
plt.show()


# In[ ]:


#jointplot
sns.jointplot(x=df.age, y=df.sex, data=df, kind="kde");


# In[ ]:


sns.swarmplot(x = 'sex', y = 'age', data = df)
plt.show()


# In[ ]:


df['age']=df['age']
bins=[29,47,55,61,77]
labels=["Young Adult","Early Adult","Adult","Senior"]
df['age_group']=pd.cut(df['age'],bins,labels=labels)
fig=plt.figure(figsize=(20,5))
sns.barplot(x='age_group',y='sex',data=df)
plt.show()


# In[ ]:



fig=plt.figure(figsize=(20,5))
sns.violinplot(x ='age_group', y = 'sex', data = df)
plt.show()


# In[ ]:


#sns.set_style('whitegrid')

fig=plt.figure(figsize=(20,5))
sns.violinplot(x ='age_group', y = 'trestbps', data = df)
plt.show()


# In[ ]:


fig=plt.figure(figsize=(20,5))
sns.violinplot(x = 'age_group', y = 'chol', data = df)
plt.show()


# In[ ]:


fig=plt.figure(figsize=(20,15))
sns.violinplot(x = 'age_group', y = 'thalach', data = df)
plt.show()


# In[ ]:


grp =df.groupby("age")
x= grp["chol"].agg(np.mean)
y=grp["trestbps"].agg(np.mean)
z=grp["thalach"].agg(np.mean)


# In[ ]:


plt.figure(figsize=(16,5))
plt.plot(x,'ro',color='r')
plt.xticks(rotation=90)
plt.title("Age wise Chol")
plt.xlabel("Age")
plt.ylabel("Chol")
plt.show()


# In[ ]:


plt.figure(figsize=(16,5))
plt.plot(y,'r--',color='b')
plt.xticks(rotation=90)
plt.title("Age wise Trestbps")
plt.xlabel("Age")
plt.ylabel("Trestbps")
plt.show()


# In[ ]:


plt.figure(figsize=(16,5))
plt.plot(z,"g^",color='g')
plt.xticks(rotation=90)
plt.xlabel("Age")
plt.ylabel("Thalach")
plt.show()


# In[ ]:



fig=plt.figure(figsize=(20,5))
sns.violinplot(x ='age', y = 'trestbps', data = df)
plt.show()


# In[ ]:


ax = df.trestbps.plot.kde()
ax = df.chol.plot.kde()
ax = df.thalach.plot.kde()
ax.legend()
plt.show()


# In[ ]:


#xticks -> chol min,mean,max 
#yticks -> trestbps min,mean,max
filtered_class1=df[(df.chol>246) & (df.trestbps>131) & (df.sex==0)]
filtered_class4=df[(df.chol>246) & (df.trestbps>131) & (df.sex==1)]
fig=plt.figure(figsize=(20,15))

g = sns.lmplot(x="chol", y="trestbps", hue="cp", col="sex", height=4, aspect=.7, x_jitter=.1,palette="Set1",markers=["o","*","x",'s'],data=filtered_class1)
g = (g.set_axis_labels("chol", "trestbps").set(xlim=(120, 600), ylim=(90, 240),xticks=[120, 246, 600], yticks=[90, 131, 240]).fig.subplots_adjust(wspace=.02))

g = sns.lmplot(x="chol", y="trestbps", hue="cp", col="sex", height=4, aspect=.7, x_jitter=.1,palette="Set1",markers=["o","*","x",'s'],data=filtered_class4)
g = (g.set_axis_labels("chol", "trestbps").set(xlim=(120, 600), ylim=(90, 240),xticks=[120, 246, 600], yticks=[90, 131, 240]).fig.subplots_adjust(wspace=.02))


g = sns.lmplot(x="chol", y="trestbps", hue="cp",col="sex",height=5,aspect=.7, x_jitter=.1,palette="Set1",markers=["o","*","x",'s'],data=df)
g = (g.set_axis_labels("chol", "trestbps").set(xlim=(120, 600), ylim=(90, 240),xticks=[120, 246, 600], yticks=[90, 131, 240]).fig.subplots_adjust(wspace=.02))




##########################################################################################################################################
filtered_class2=df[(df.chol>246) & (df.thalach>149) & (df.sex==0)]
filtered_class5=df[(df.chol>246) & (df.thalach>149) & (df.sex==1)]

g = sns.lmplot(x="chol", y="thalach", hue="cp", col="sex", height=4, aspect=.7, x_jitter=.1,palette="Set1",markers=["o","*","x",'s'],data=filtered_class2)
g = (g.set_axis_labels("chol", "thalach").set(xlim=(120, 600), ylim=(65, 220),xticks=[120, 246, 600], yticks=[65, 149, 220]).fig.subplots_adjust(wspace=.02))

g = sns.lmplot(x="chol", y="thalach", hue="cp", col="sex",height=5,aspect=.7, x_jitter=.1,palette="Set1",markers=["o","*","x",'s'],data=df)
g = (g.set_axis_labels("chol", "thalach").set(xlim=(120, 600), ylim=(65, 220),xticks=[120, 246, 600], yticks=[65, 149, 220]).fig.subplots_adjust(wspace=.02))

g = sns.lmplot(x="chol", y="thalach", hue="cp", col="sex", height=4, aspect=.7, x_jitter=.1,palette="Set1",markers=["o","*","x",'s'],data=filtered_class5)
g = (g.set_axis_labels("chol", "thalach").set(xlim=(120, 600), ylim=(65, 220),xticks=[120, 246, 600], yticks=[65, 149, 220]).fig.subplots_adjust(wspace=.02))

############################################################################################################################################
filtered_class3=df[(df.trestbps>131) & (df.thalach>149) & (df.sex==0)]
filtered_class6=df[(df.trestbps>131) & (df.thalach>149) & (df.sex==1)]

g = sns.lmplot(x="trestbps", y="thalach", hue="cp", col="sex", height=4, aspect=.7, x_jitter=.1,palette="Set1",markers=["o","*","x",'s'],data=filtered_class3)
g = (g.set_axis_labels("trestbps", "thalach").set(xlim=(85, 220), ylim=(65, 220),xticks=[85, 131, 220], yticks=[65, 149, 220]).fig.subplots_adjust(wspace=.02))

g = sns.lmplot(x="trestbps", y="thalach", hue="cp", col="sex",height=5,aspect=.7, x_jitter=.1,palette="Set1",markers=["o","*","x",'s'],data=df)
g = (g.set_axis_labels("trestbps", "thalach").set(xlim=(85, 220), ylim=(65, 220),xticks=[85, 131, 220], yticks=[65, 149, 220]).fig.subplots_adjust(wspace=.02))

g = sns.lmplot(x="trestbps", y="thalach", hue="cp", col="sex", height=4, aspect=.7, x_jitter=.1,palette="Set1",markers=["o","*","x",'s'],data=filtered_class6)
g = (g.set_axis_labels("trestbps", "thalach").set(xlim=(85, 220), ylim=(65, 220),xticks=[85, 131, 220], yticks=[65, 149, 220]).fig.subplots_adjust(wspace=.02))


# In[ ]:


filtered_class1=df[(df.chol>246) & (df.trestbps>131) & (df.sex==0) & (df.age>60)]
filtered_class4=df[(df.chol>246) & (df.trestbps>131) & (df.sex==1) & (df.age>60)]

fig=plt.figure(figsize=(20,15))

g = sns.lmplot(x="chol", y="trestbps", hue="cp", col="sex", height=4, aspect=.7, x_jitter=.1,palette="Set1",data=filtered_class1)
g = (g.set_axis_labels("chol", "trestbps").set(xlim=(120, 600), ylim=(90, 240),xticks=[120, 246, 600], yticks=[90, 131, 240]).fig.subplots_adjust(wspace=.02))

g = sns.lmplot(x="chol", y="trestbps", hue="cp",col="sex",height=5,aspect=.7, x_jitter=.1,palette="Set1",data=df)
g = (g.set_axis_labels("chol", "trestbps").set(xlim=(120, 600), ylim=(90, 240),xticks=[120, 246, 600], yticks=[90, 131, 240]).fig.subplots_adjust(wspace=.02))

g = sns.lmplot(x="chol", y="trestbps", hue="cp", col="sex", height=4, aspect=.7, x_jitter=.1,palette="Set1",data=filtered_class4)
g = (g.set_axis_labels("chol", "trestbps").set(xlim=(120, 600), ylim=(90, 240),xticks=[120, 246, 600], yticks=[90, 131, 240]).fig.subplots_adjust(wspace=.02))

##########################################################################################################################################
filtered_class2=df[(df.chol>246) & (df.thalach>149) & (df.sex==0) & (df.age>60)]
filtered_class5=df[(df.chol>246) & (df.thalach>149) & (df.sex==1) & (df.age>60)]

g = sns.lmplot(x="chol", y="thalach", hue="cp", col="sex", height=4, aspect=.7, x_jitter=.1,palette="Set1",data=filtered_class2)
g = (g.set_axis_labels("chol", "thalach").set(xlim=(120, 600), ylim=(65, 220),xticks=[120, 246, 600], yticks=[65, 149, 220]).fig.subplots_adjust(wspace=.02))

g = sns.lmplot(x="chol", y="thalach", hue="cp", col="sex",height=5,aspect=.7, x_jitter=.1,palette="Set1",data=df)
g = (g.set_axis_labels("chol", "thalach").set(xlim=(120, 600), ylim=(65, 220),xticks=[120, 246, 600], yticks=[65, 149, 220]).fig.subplots_adjust(wspace=.02))

g = sns.lmplot(x="chol", y="thalach", hue="cp", col="sex", height=4, aspect=.7, x_jitter=.1,palette="Set1",data=filtered_class5)
g = (g.set_axis_labels("chol", "thalach").set(xlim=(120, 600), ylim=(65, 220),xticks=[120, 246, 600], yticks=[65, 149, 220]).fig.subplots_adjust(wspace=.02))

############################################################################################################################################
filtered_class3=df[(df.trestbps>131) & (df.thalach>149) & (df.sex==0) &(df.age>60)]
filtered_class6=df[(df.trestbps>131) & (df.thalach>149) & (df.sex==1) & (df.age>60)]

g = sns.lmplot(x="trestbps", y="thalach", hue="cp", col="sex", height=4, aspect=.7, x_jitter=.1,palette="Set1",data=filtered_class3)
g = (g.set_axis_labels("trestbps", "thalach").set(xlim=(85, 220), ylim=(65, 220),xticks=[85, 131, 220], yticks=[65, 149, 220]).fig.subplots_adjust(wspace=.02))

g = sns.lmplot(x="trestbps", y="thalach", hue="cp", col="sex",height=5,aspect=.7, x_jitter=.1,palette="Set1",data=df)
g = (g.set_axis_labels("trestbps", "thalach").set(xlim=(85, 220), ylim=(65, 220),xticks=[85, 131, 220], yticks=[65, 149, 220]).fig.subplots_adjust(wspace=.02))

g = sns.lmplot(x="trestbps", y="thalach", hue="cp", col="sex", height=4, aspect=.7, x_jitter=.1,palette="Set1",data=filtered_class6)
g = (g.set_axis_labels("trestbps", "thalach").set(xlim=(85, 220), ylim=(65, 220),xticks=[85, 131, 220], yticks=[65, 149, 220]).fig.subplots_adjust(wspace=.02))


# In[ ]:


print(df['cp'].value_counts(dropna=False))


# In[ ]:


fig=plt.figure(figsize=(20,5))
sns.swarmplot(x = 'age', y = 'chol', data = df)
plt.show()


# In[ ]:


fig=plt.figure(figsize=(20,5))
sns.swarmplot(x = 'cp', y = 'age', data = df)
plt.show()


# In[ ]:


fig=plt.figure(figsize=(20,5))
sns.swarmplot(x = 'age', y = 'thalach', data = df)
plt.show()


# In[ ]:


fig=plt.figure(figsize=(10,5))
sns.swarmplot(x="sex", y="age",hue="cp", data=df)
plt.show()


# In[ ]:


fig=plt.figure(figsize=(20,5))
sns.violinplot(x = 'sex', 
               y = 'age', 
               data = df, 
               inner = None, 
               )

sns.swarmplot(x = 'sex', 
              y = 'age', 
              data = df, 
              color = 'k', 
              alpha = 0.7)

plt.title('sex by age')
plt.show()


# In[ ]:


#barplot
sns.barplot(x='cp',y='age',data=df)
plt.show()


# In[ ]:


fig=plt.figure(figsize=(10,5))
sns.violinplot(x = 'age_group', y = 'cp', data = df)
plt.show()


# In[ ]:


#boxplot
df.boxplot(column='age', by='cp')
plt.show()


# In[ ]:


#jointplot
sns.jointplot(x=df.age, y=df.cp, data=df, kind="kde");


# In[ ]:


# histogram subplot with non cumulative and cumulative

fig,axes=plt.subplots(nrows=2,ncols=1)
df.plot(kind="hist",y="trestbps",bins=50, range=(0,250),normed=True,ax=axes[0])
df.plot(kind="hist",y="trestbps",bins=50, range=(0,250),normed=True,ax=axes[1],cumulative=True)
plt.savefig('graph.png')
plt.show()


# In[ ]:


#boxplot
df.boxplot(column='trestbps',by='age', figsize=(18,18))
plt.show()


# In[ ]:


# Scatter Plot
#age vs. trestbps
df.plot(kind='scatter',x='age',y='trestbps',alpha=0.5,color='r')
plt.xlabel('age') # label = name of label
plt.ylabel('trestbps')
plt.title('Age-trestbps Scatter Plot')  # title = title of plot
plt.show()


# In[ ]:


#jointplot
sns.jointplot(x="trestbps", y="age", data=df);


# In[ ]:


#jointplot hex

with sns.axes_style("white"):
    sns.jointplot(x=df.age, y=df.trestbps, kind="hex", color="k");


# In[ ]:


#scatter plot
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
age_group = ["Young Adult","Middle-Aged Adults","Old Adults"]
sns.scatterplot(x="age", y="trestbps",
                hue="age",
                palette="ch:r=-.2,d=.3_r",
                hue_order=age_group,
                sizes=(1, 8), linewidth=0,
                data=df, ax=ax)
plt.show()


# In[ ]:


#jointplot
sns.jointplot(x=df.age, y=df.trestbps, data=df, kind="kde");


# In[ ]:


#barplot
plt.figure(figsize=(15,10))
sns.barplot(x='age',y='trestbps',data=df)
plt.show()


# In[ ]:


# histogram subplot with non cumulative and cumulative

fig,axes=plt.subplots(nrows=2,ncols=1)
df.plot(kind="hist",y="chol",bins=50, range=(0,250),normed=True,ax=axes[0])
df.plot(kind="hist",y="chol",bins=50, range=(0,250),normed=True,ax=axes[1],cumulative=True)
plt.savefig('graph.png')
plt.show()


# In[ ]:


#barplot
plt.figure(figsize=(15,10))
sns.barplot(x='age',y='chol',data=df)
plt.show()


# In[ ]:


#boxplot
df.boxplot(column='chol',by='age',figsize=(18,18))
plt.show()


# In[ ]:


#jointplot
sns.jointplot(x="age", y="chol", data=df);


# In[ ]:


#scatter plot
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
age_group = ["Young Adult","Middle-Aged Adults","Old Adults"]
sns.scatterplot(x="age", y="chol",
                hue="age",
                palette="ch:r=-.2,d=.3_r",
                hue_order=age_group,
                sizes=(1, 8), linewidth=0,
                data=df, ax=ax)
plt.show()


# In[ ]:


#jointplot
with sns.axes_style("white"):
    sns.jointplot(x=df.age, y=df.chol, kind="hex", color="k");


# In[ ]:


#jointplot
sns.jointplot(x=df.age, y=df.chol, data=df, kind="kde");


# In[ ]:


print(df['fbs'].value_counts(dropna=False))


# In[ ]:


#barplot
sns.barplot(x='fbs',y='age',data=df)
plt.show()


# In[ ]:


#box plot
df.boxplot(column='age',by='fbs')
plt.show()


# In[ ]:


print(df['restecg'].value_counts(dropna=False))


# In[ ]:


#barplot
sns.barplot(x='restecg',y='age',data=df)
plt.show()


# In[ ]:


#jointplot
sns.jointplot(x=df.age, y=df.restecg, data=df, kind="kde");


# In[ ]:


#boxplot
df.boxplot(column='age',by='restecg')
plt.show()


# In[ ]:


# histogram subplot with non cumulative and cumulative

fig,axes=plt.subplots(nrows=2,ncols=1)
df.plot(kind="hist",y="thalach",bins=50, range=(0,250),normed=True,ax=axes[0])
df.plot(kind="hist",y="thalach",bins=50, range=(0,250),normed=True,ax=axes[1],cumulative=True)
plt.savefig('graph.png')
plt.show()


# In[ ]:


#barplot
plt.figure(figsize=(18,18))
sns.barplot(x='age',y='thalach',data=df)
plt.show()


# In[ ]:


#boxplot
df.boxplot(column='thalach', by='age',figsize=(18,18))
plt.show()


# In[ ]:


#scatter plot
f, ax = plt.subplots(figsize=(6.5, 6.5))
sns.despine(f, left=True, bottom=True)
age_group = ["Young Adult","Middle-Aged Adults","Old Adults"]
sns.scatterplot(x="age", y="thalach",
                hue="age",
                palette="ch:r=-.2,d=.3_r",
                hue_order=age_group,
                sizes=(1, 8), linewidth=0,
                data=df, ax=ax)
plt.show()


# In[ ]:


#jointplot
sns.jointplot(x="thalach", y="age", data=df);


# In[ ]:


#jointplot
with sns.axes_style("white"):
    sns.jointplot(x=df.age, y=df.thalach, kind="hex", color="k");


# In[ ]:


#jointplot
sns.jointplot(x=df.age, y=df.thalach, data=df, kind="kde");


# In[ ]:


print(df['exang'].value_counts(dropna=False))


# In[ ]:


#barplot
sns.barplot(x='exang',y='age', data=df)
plt.show()


# In[ ]:


#jointplot
sns.jointplot(x=df.age, y=df.exang, data=df, kind="kde");


# In[ ]:


#boxplot
df.boxplot(column='age',by='exang')
plt.show()


# In[ ]:


print(df['ca'].value_counts(dropna=False))


# In[ ]:


#barplot
sns.barplot(x='ca',y='age',data=df)
plt.show()


# In[ ]:


#boxplot
df.boxplot(column='age',by='ca', figsize=(10,10))
plt.show()


# In[ ]:


#jointplot
sns.jointplot(x=df.age, y=df.ca, data=df, kind="kde");


# In[ ]:


print(df['thal'].value_counts(dropna=False))


# In[ ]:


#barplot
sns.barplot(x='thal',y='age',data=df)
plt.show()


# In[ ]:


#boxplot
df.boxplot(column='age',by='thal',figsize=(10,10))
plt.show()


# In[ ]:


#jointplot
sns.jointplot(x=df.age, y=df.thal, data=df, kind="kde");


# In[ ]:


print(df['target'].value_counts(dropna=False))


# In[ ]:


#barplot
sns.barplot(x='target',y='age',data=df)
plt.show()


# In[ ]:


#boxplot
df.boxplot(column='age',by='target')
plt.show()


# In[ ]:


#jointplot
sns.jointplot(x=df.age, y=df.target, data=df, kind="kde");


# In[ ]:


#barplot
sns.barplot(x='cp',y='thalach',data=df)
plt.show()


# In[ ]:


#boxplot
df.boxplot(column='thalach',by='cp', figsize=(10,10))
plt.show()


# In[ ]:


#jointplot
sns.jointplot(x=df.cp, y=df.thalach, data=df, kind="kde");


# In[ ]:


#barplot
sns.barplot(x='target',y='thalach',data=df)
plt.show()


# In[ ]:


#jointplot
sns.jointplot(x=df.target, y=df.thalach, data=df, kind="kde");


# In[ ]:


#boxplot
df.boxplot(column='thalach',by='target', figsize=(7,7))
plt.show()


# In[ ]:


new_df=df.iloc[:,[0,1,3,4,7]]
new_df.head()


# **pd.plotting.scatter_matrix:
# **
# * green: female and red: male
# * c: color
# * figsize: figure size
# * diagonal: histohram of each features
# * alpha: opacity
# * s: size of marker
# * marker: marker type

# In[ ]:


color_list = ['red' if i==1 else 'green' for i in new_df.loc[:,'sex']]
pd.plotting.scatter_matrix(new_df.loc[:, new_df.columns != 'sex'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolor= "black")
plt.show()


# In[ ]:


f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='age',y='trestbps',data=df,color='lime',alpha=0.8)
sns.pointplot(x='age',y='chol',data=df,color='red',alpha=0.8)
sns.pointplot(x='age',y='thalach',data=df,color='blue',alpha=0.8)
plt.text(35,0.4,'age-trestbps',color='lime',fontsize = 15,style = 'italic')
plt.text(40,0.5,'age-chol',color='red',fontsize = 16,style = 'italic')
plt.text(45,0.6,'age-thalach',color='blue',fontsize = 17,style = 'italic')
plt.xlabel('age',fontsize = 15,color='blue')
plt.ylabel('values',fontsize = 15,color='blue')
plt.title('trestbps  -  chol - thalach',fontsize = 20,color='blue')
plt.grid()


# In[ ]:


df.cp.dropna(inplace = True)
labels = df.cp.value_counts().index
colors = ['green','yellow','orange','red']
explode = [0,0,0,0]
sizes = df.cp.value_counts().values

# visual cp
plt.figure(0,figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Target People According to Chestpain Type',color = 'blue',fontsize = 15)


# **<h1>Filtering</h1>**
# 

# In[ ]:


#female
df[(df['sex']==0) & (df['age']>50) & (df['ca']>0) & (df['chol']>=160) & (df['cp']>=1) & 
   (df['trestbps']>=140) & (df['fbs']==1) & (df['thalach']>=120)& (df['target']==1)] 


# In[ ]:


#female - ca
df[(df['sex']==0) & (df['age']>65) & (df['ca']>0)]


# In[ ]:


#female - ca - describe
df[(df['sex']==0) & (df['age']>65) & (df['ca']>0)].describe()


# In[ ]:


#female - chol
df[(df['sex']==0) & (df['age']>65) & (df['chol']>=246)] 


# In[ ]:


##female - chol - describe
df[(df['sex']==0) & (df['age']>65) & (df['chol']>=246)].describe() 


# In[ ]:


#female - cp
df[(df['sex']==0) & (df['age']>65) & (df['cp']>=1)] 


# In[ ]:


#female - cp -describe
df[(df['sex']==0) & (df['age']>65) & (df['cp']>=1)].describe()


# In[ ]:


#female - trestbps
df[(df['sex']==0) & (df['age']>65) & (df['trestbps']>=140)] 


# In[ ]:


#female - trestbps -describe
df[(df['sex']==0) & (df['age']>65) & (df['trestbps']>=140)].describe()


# In[ ]:


#female - fbs
df[(df['sex']==0) & (df['age']>65) & (df['fbs']==1)] 


# In[ ]:


#female - fbs - describe
df[(df['sex']==0) & (df['age']>65) & (df['fbs']==1)].describe()


# In[ ]:


#female - thalach
df[(df['sex']==0) & (df['age']>65) & (df['thalach']>=120)] 


# In[ ]:


#female - thalach -describe
df[(df['sex']==0) & (df['age']>65) & (df['thalach']>=120)].describe()


# In[ ]:


#female - target
df[(df['sex']==0) & (df['age']>65) & (df['target']==1)] 


# In[ ]:


#female - target -describe
df[(df['sex']==0) & (df['age']>65) & (df['target']==1)].describe() 


# In[ ]:


#female - exang
df[(df['sex']==0) & (df['age']>65) & (df['exang']==1)] 


# In[ ]:


#female - exang - describe
df[(df['sex']==0) & (df['age']>65) & (df['exang']==1)].describe()


# In[ ]:


#male
df[(df['sex']==1) & (df['age']>65) & (df['ca']>0) & (df['chol']>=160) & (df['cp']>=1) & 
   (df['trestbps']>=140)& (df['fbs']==1) & (df['thalach']>=120) & (df['target']==1)] 


# In[ ]:


#male - ca
df[(df['sex']==1) & (df['age']>65) & (df['ca']>0)] 


# In[ ]:


#male - ca - describe
df[(df['sex']==1) & (df['age']>65) & (df['ca']>0)].describe()


# In[ ]:


#male - chol
df[(df['sex']==1) & (df['age']>65) & (df['chol']>=160)] 


# In[ ]:


#male - chol - describe
df[(df['sex']==1) & (df['age']>65) & (df['chol']>=160)].describe()


# In[ ]:


#male - cp
df[(df['sex']==1) & (df['age']>65) & (df['cp']>=1)] 


# In[ ]:


#male - cp - describe
df[(df['sex']==1) & (df['age']>65) & (df['cp']>=1)].describe()


# In[ ]:


#male - trestbps
df[(df['sex']==1) & (df['age']>65) & (df['trestbps']>=140)]


# In[ ]:


#male - trestbps - describe
df[(df['sex']==1) & (df['age']>65) & (df['trestbps']>=140)].describe()


# In[ ]:


#male- fbs
df[(df['sex']==1) & (df['age']>65) & (df['fbs']==1)] 


# In[ ]:


#male- fbs - describe
df[(df['sex']==1) & (df['age']>65) & (df['fbs']==1)].describe()


# In[ ]:


#male - thalach
df[(df['sex']==1) & (df['age']>65) & (df['thalach']>=120)] 


# In[ ]:


#male - thalach - describe
df[(df['sex']==1) & (df['age']>65) & (df['thalach']>=120)].describe()


# In[ ]:


#male - target
df[(df['sex']==1) & (df['age']>65) & (df['target']==1)] 


# In[ ]:


#male - target - describe
df[(df['sex']==1) & (df['age']>65) & (df['target']==1)].describe() 


# In[ ]:


#male - exang
df[(df['sex']==1) & (df['age']>65) & (df['ca']>0) & (df['exang']==1)] 


# In[ ]:


#male - exang - describe
df[(df['sex']==1) & (df['age']>65) & (df['ca']>0) & (df['exang']==1)].describe() 


# In[ ]:


#general
df[(df['age']>60) & (df['chol']>=160) & (df['cp']>=1) & 
   (df['trestbps']>=140) & (df['fbs']==1) & (df['thalach']>=120) & (df['target']==1) & (df['exang']==1)] 


# In[ ]:


x=df['age']>65
df[x]


# In[ ]:


x=df['age']>65
df[x].describe()


# References:
# 
# https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners
# 
# https://www.kaggle.com/kanncaa1/feature-selection-and-data-visualization
# 
# https://www.kaggle.com/kanncaa1/seaborn-tutorial-for-beginners
# 
# https://www.kaggle.com/kanncaa1/plotly-tutorial-for-beginners

# # CONCLUSION
# Thank you for your votes and comments
# <br> **EDA and Data Visualization Titanic ** https://www.kaggle.com/nidaguler/eda-and-data-visualization-titanic/
# <br> **Titanic Survival Detection using Machine Learning** https://www.kaggle.com/nidaguler/titanic-survival-detection-using-machine-learning
# <br> **EDA and Data Visualization NY Airbnb** https://www.kaggle.com/nidaguler/eda-and-data-visualization-ny-airbnb
# <br> **Data Visualization World Happiness Report 2015** https://www.kaggle.com/nidaguler/data-visualization-world-happiness-report-2015
# <br> **Forest Fires in Brazil** https://www.kaggle.com/nidaguler/forest-fires-in-brazil
# <br>**If you have any question or suggest, I will be happy to hear it.**
# 
