#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#@Fallou Diagne: DataScientist@
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')  
get_ipython().run_line_magic('matplotlib', 'inline')

import os

# Any results you write to the current directory are saved as output.

import warnings
warnings.filterwarnings('ignore') 
plt.style.use('fivethirtyeight')

pd.options.mode.chained_assignment = None


# In[ ]:


df = pd.read_csv("../input/Iris.csv")


# In[ ]:


df.head()


# In[ ]:


#drop Id 
df.drop('Id',axis=1,inplace=True)


# In[ ]:


#verification 
df.head()


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


df['Species'].value_counts()


# In[ ]:


df.describe().plot(kind = "area",fontsize=27, figsize = (20,8), table = True,colormap="rainbow")
plt.xlabel('Statistics',)
plt.ylabel('Value')
plt.title("Statistiques datasets")


# In[ ]:


#plot count  
ax = sns.countplot(x="Species", data=df) 


# In[ ]:


f,ax=plt.subplots(1,2,figsize=(18,8))
df['Species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('df Species Count')
ax[0].set_ylabel('Count')
sns.countplot('Species',data=df,ax=ax[1])
ax[1].set_title('df Species Count')
plt.show()


# In[ ]:


fig=sns.jointplot(x='SepalLengthCm',y='SepalWidthCm',data=df)


# In[ ]:


sns.jointplot("SepalLengthCm", "SepalWidthCm", data=df, kind="reg")


# In[ ]:


fig=sns.jointplot(x='SepalLengthCm',y='SepalWidthCm',kind='hex',data=df)


# In[ ]:


sns.jointplot("SepalLengthCm", "SepalWidthCm", data=df, kind="kde",space=0,color='g')


# In[ ]:


g = (sns.jointplot("SepalLengthCm", "SepalWidthCm",data=df, color="k").plot_joint(sns.kdeplot, zorder=0, n_levels=6))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
sns.FacetGrid(df,hue='Species',size=5).map(plt.scatter,'SepalLengthCm','SepalWidthCm').add_legend()


# In[ ]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.boxplot(x='Species',y='PetalLengthCm',data=df,order=['Iris-virginica','Iris-versicolor','Iris-setosa'],
                linewidth=2.5,orient='v',dodge=False)


# In[ ]:


df.boxplot(by="Species", figsize=(12, 6))


# In[ ]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.stripplot(x='Species',y='SepalLengthCm',data=df,jitter=True,edgecolor='gray',size=8,palette='winter',orient='v')


# In[ ]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.boxplot(x='Species',y='SepalLengthCm',data=df)
fig=sns.stripplot(x='Species',y='SepalLengthCm',data=df,jitter=True,edgecolor='gray')


# In[ ]:


ax= sns.boxplot(x="Species", y="PetalLengthCm", data=df)
ax= sns.stripplot(x="Species", y="PetalLengthCm", data=df, jitter=True, edgecolor="gray")

boxtwo = ax.artists[2]
boxtwo.set_facecolor('yellow')
boxtwo.set_edgecolor('black')
boxthree=ax.artists[1]
boxthree.set_facecolor('red')
boxthree.set_edgecolor('black')
boxthree=ax.artists[0]
boxthree.set_facecolor('green')
boxthree.set_edgecolor('black')

plt.show()


# In[ ]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.violinplot(x='Species',y='SepalLengthCm',data=df)


# In[ ]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=df)


# In[ ]:


#pairs plot 
sns.pairplot(data=df,kind='scatter')


# In[ ]:


sns.pairplot(df,hue='Species')


# In[ ]:


fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.heatmap(df.corr(),annot=True,cmap='cubehelix',linewidths=1,linecolor='k',square=True,mask=False,
                vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)


# In[ ]:


df.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12,6)


# In[ ]:


sns.set(style="darkgrid")
fig=plt.gcf()
fig.set_size_inches(10,7)
fig = sns.swarmplot(x="Species", y="PetalLengthCm", data=df)


# In[ ]:


import numpy as np #importation de numpay 
sns.set(style="darkgrid")
fig=plt.gcf()
fig.set_size_inches(10,7)
fig= sns.boxplot(x="Species", y="PetalLengthCm", data=df, whis=np.inf)
fig= sns.swarmplot(x="Species", y="PetalLengthCm", data=df, color=".2")


# In[ ]:


sns.set(style="whitegrid")
fig=plt.gcf()
fig.set_size_inches(10,7)
ax = sns.violinplot(x="Species", y="PetalLengthCm", data=df, inner=None)
ax = sns.swarmplot(x="Species", y="PetalLengthCm", data=df,color="white", edgecolor="black")


# In[ ]:


sns.set(style="whitegrid")
fig=plt.gcf()
fig.set_size_inches(10,7)
ax = sns.violinplot(x="Species", y="PetalLengthCm", data=df, inner=None)
ax = sns.swarmplot(x="Species", y="PetalLengthCm", data=df,color="white", edgecolor="black")


# In[ ]:


fig=sns.lmplot(x="PetalLengthCm", y="PetalWidthCm",data=df)


# In[ ]:


sns.set(style="darkgrid")
sc=df[df.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='red',label='Setosa')
df[df.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green',label='Versicolor',ax=sc)
df[df.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='virginica', ax=sc)
sc.set_xlabel('Sepal Length in cm')
sc.set_ylabel('Sepal Width in cm')
sc.set_title('Sepal Length Vs Sepal Width')
sc=plt.gcf()
sc.set_size_inches(10,6)


# In[ ]:


sns.FacetGrid(df, hue="Species", size=6)    .map(sns.kdeplot, "PetalLengthCm")    .add_legend()
plt.ioff() 


# In[ ]:


from pandas.tools.plotting import andrews_curves
andrews_curves(df,"Species",colormap='rainbow')
plt.show()
plt.ioff()


# In[ ]:


from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(df, "Species")


# In[ ]:


from pandas.tools.plotting import radviz
radviz(df, "Species")


# In[ ]:


sns.factorplot('Species','SepalLengthCm',data=df)
plt.ioff()
plt.show()


# In[ ]:


#Thank for all ! 

