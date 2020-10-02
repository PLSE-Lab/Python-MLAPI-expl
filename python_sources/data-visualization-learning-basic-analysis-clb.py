#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf-8"))


# ## Read Data & Information

# In[ ]:


data = pd.read_csv("../input/Iris.csv")


# In[ ]:


data.info()


# In[ ]:


data.count()


# In[ ]:


data.describe()


# In[ ]:


data.groupby('Species').size()


# In[ ]:


data.corr()


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


species = iter(data['Species'])
print(*species)


# In[ ]:


print(data['Species'].value_counts(dropna=False))


# In[ ]:


data.Species.unique()


# In[ ]:


Counter(data.Species)


# ## ===SEABORN===

# **Bar Plot**

# In[ ]:


plt.figure(figsize=(5,5))
sns.barplot(x=data['Species'], y=data['SepalLengthCm'])
plt.xticks(rotation = 90)
plt.xlabel('Species')
plt.ylabel('Sepal Length Cm')
plt.title('Species for Sepal Length Cm')
plt.show()


# In[ ]:


PetalLengthCm = data.PetalLengthCm[data.PetalLengthCm > 1.0]
setosaAndVirgina = data.Species[(data.Species == 'Iris-setosa') | (data.Species == 'Iris-virginica')]
plt.figure(figsize=(5,5))
sns.barplot(x=setosaAndVirgina, y=PetalLengthCm,palette = sns.cubehelix_palette(2))
plt.xticks(rotation = 90)
plt.xlabel('Setosa And Virgina')
plt.ylabel('PetalLengthCm')
plt.show()


# **Point Plot**

# In[ ]:


# SepalLengthCm ,PetalLengthCm
newdata = pd.concat([data['SepalLengthCm'],data['PetalLengthCm'],data['Species']],axis=1)
newdata.sort_values('SepalLengthCm')

f,ax =  plt.subplots(figsize=(15,5))
sns.pointplot(x='Species',y='SepalLengthCm',data=newdata,color='lime',alpha=0.9)
sns.pointplot(x='Species',y='PetalLengthCm',data=newdata,color='red',alpha=0.7)
plt.text(1,5.5,' Species Sepal Length Cm', color='lime', fontsize=17,style='italic')
plt.text(1,3,' Species Petal Length Cm', color='red', fontsize=18, style='italic')
plt.xlabel('')
plt.ylabel('')
plt.title('Sepal Length Petal Length', fontsize=20,color='blue')
plt.grid()


# **Joint Plot**

# In[ ]:


g = sns.jointplot(newdata.SepalLengthCm,newdata.PetalLengthCm, kind="kde", size=5)
plt.show()


# In[ ]:


g2 = sns.jointplot('SepalLengthCm','PetalLengthCm',data=newdata,size=5,ratio=3,color='r')
plt.show()


# **Pie Chart**

# In[ ]:


labels = data.Species.value_counts().index
colors = ['red','blue','yellow']
explode = [0,0,0]
sizes = data.Species.value_counts().values

plt.figure(figsize=(7,7))
plt.pie(sizes,explode=explode,labels=labels,colors=colors,autopct='%1.2f%%')
plt.title('Species Type', color='green',fontsize=15)
plt.show()


# **LM Plot**

# In[ ]:


sns.lmplot(x = "SepalLengthCm" , y = "PetalLengthCm" , data=newdata)
plt.show()


# **KDE Plot**

# In[ ]:


sns.kdeplot(newdata.PetalLengthCm,newdata.SepalLengthCm, shade=True,cut=2)
plt.show()


# **Violin Plot**

# In[ ]:


pal = sns.cubehelix_palette(2,rot=-5,dark=.3)
sns.violinplot(data=newdata,palette=pal,inner="points")
plt.show()


# **Heat Map**

# In[ ]:


f,ax = plt.subplots(figsize=(3,3))
sns.heatmap(newdata.corr(),annot=True,linewidths=1.5, linecolor='red',fmt='.1f',ax=ax)
plt.show()


# **Box Plot**

# In[ ]:


newdatahead = newdata.head(10)
sns.boxplot(x="SepalLengthCm",y="PetalLengthCm", hue="Species", data=newdatahead,palette="PRGn")
plt.show()


# **Swarm Plot**

# In[ ]:


sns.swarmplot(x="SepalLengthCm",y="PetalLengthCm", hue="Species",data=newdata)
plt.show()


# **Pair Plot**

# In[ ]:


sns.pairplot(newdata)
plt.show()


# **Count Plot**

# In[ ]:


sns.countplot(newdata.SepalLengthCm)
plt.title("Species", color="blue",fontsize=15)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




