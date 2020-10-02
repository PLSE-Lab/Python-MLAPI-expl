#!/usr/bin/env python
# coding: utf-8

# #  INTRODUCTION

# In general, people like a kind of music, but they can listen to different music because of their environment or situation. I wanted to explore what they prefer as other genres of music are listening to.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Firstly, import dataset from csv format and  we quickly check on the dataset 

# In[ ]:


data = pd.read_csv("../input/responses.csv")
data.head()


# In[ ]:


print(data.info())


# # **Filtering and Cleaning**

# We separate data about music  because our dataset is too wide to process.

# In[ ]:


data_music = data.iloc[:,0:19]


# After separate music data, we add some demographic data

# In[ ]:


data_use = data_music
data_use["Gender"] = data["Gender"]
data_use["Age"] = data["Age"]
data_use["Alcohol"] = data["Alcohol"]
data_use["Education"] = data["Education"]


# When we look into the data, it is seen that there are some missing data in each columns.

# In[ ]:


print(data_use.info())


# In[ ]:


print(data_use.isnull().sum())


# Firstly, we drop the missing data from dataframe and reset our index

# In[ ]:


data_use.dropna(inplace = True)
data_use.reset_index(drop=True,inplace=True)
row = len(data_use.index)


# In[ ]:


data_use.info()


# All of numerical data has entered as float64 format and none of them has decimal part, thus we convert these data from float to integer format.

# In[ ]:


for each in range(0,23) :
    if type(data_use.iloc[1,each]) == np.float64 :
        data_use[data_use.columns[each]] = data_use[data_use.columns[each]].astype(int)
    else :
        data_use[data_use.columns[each]] = data_use[data_use.columns[each]]


# In addition of these,as shown in the chart below,   we remove data which a full score to music isn't given  because they are outlier for our data

# In[ ]:


data_use.boxplot(column='Music')
plt.show()


# In[ ]:


print(data_use['Music'].value_counts(dropna =False))


# In[ ]:


filtre = data_use.Music < 5
filt_list = list(data_use[filtre].index)
i=0
for each in filt_list:
    data_use.drop(data_use.index[each-i], inplace=True)
    i=i+1


# We reset index again

# In[ ]:


data_use.reset_index(drop=True,inplace=True)
row = len(data_use.index)


# In[ ]:


print(data_use['Music'].value_counts(dropna =False))


# The "Music" column can be removed after it used to filter data

# In[ ]:


data_use.drop(['Music'], axis=1,inplace = True)
plt.show()


# The demographic data in dataframe is object format. They are converted to integer format to process easily

# In[ ]:


data_use.Gender.unique()


# In "Gender" column, there are "female" and "male"  values, they convert as  "female"= 0 ,  "male = 1"

# In[ ]:


for sex in range(0,row) :
    if data_use.loc[sex,'Gender'] == 'female' :
        data_use.loc[sex,'Gender'] = 0
    else :
        data_use.loc[sex,'Gender'] = 1


# In[ ]:


data_use['Gender'].head()


# In[ ]:


data_use.Alcohol.unique()


# It is seen that there are three different values in "Alcohol" column. They re-encode as "never"=0,  "social drinker"=1 ve "drink a lot"=2

# In[ ]:


for drink in range(0,row) :
    if data_use.loc[drink,'Alcohol'] == 'never' :
        data_use.loc[drink,'Alcohol'] = 0
    elif data_use.loc[drink,'Alcohol'] == 'social drinker':
        data_use.loc[drink,'Alcohol'] = 1
    else :
        data_use.loc[drink,'Alcohol'] = 2


# In[ ]:


data_use['Alcohol'].head()


# In[ ]:


data_use.Education.unique()


# There are seven unique values in "Education" columns and they re-encode as 'currently a primary school pupil'=0 ,  'primary school' = 1 ,  'secondary school' = 2 , 'college/bachelor degree' = 3 ,  'masters degree' = 4  and 'doctorate degree' = 5

# In[ ]:


for edu in range(0,row) :
    if data_use.loc[edu,'Education'] == 'currently a primary school pupil' :
        data_use.loc[edu,'Education'] = 0
    elif data_use.loc[edu,'Education'] == 'primary school':
        data_use.loc[edu,'Education'] = 1
    elif data_use.loc[edu,'Education'] == 'secondary school':
        data_use.loc[edu,'Education'] = 2
    elif data_use.loc[edu,'Education'] == 'college/bachelor degree':
        data_use.loc[edu,'Education'] = 3
    elif data_use.loc[edu,'Education'] == 'masters degree':
        data_use.loc[edu,'Education'] = 4
    else :
        data_use.loc[edu,'Education'] = 5


# In[ ]:


data_use['Education'].tail(7)


# Once again, we check the our data and it ready for processing

# In[ ]:


data_use.info()


# # Processing Data

# In[ ]:


data_use.corr()


# We have total 750 entries in each columns and our critical correlation is 0.094 for %95 confidence interval for that value, thus we can assume that there is a reletion with two variable if corrolation value greater than 0.094

#     Coefficient,
#     Strength of Association         Positive	       Negative
#     Small	                       .1 to .3	      -0.1 to -0.3
#     Medium	                      .3 to .5	      -0.3 to -0.5
#     Large                           .5 to 1.0	     -0.5 to -1.0

# In[ ]:


f,ax = plt.subplots(figsize=(25, 20))
sns.heatmap(data_use.corr(), annot=True, linewidths=1, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


data_use.columns


# In[ ]:


# create dataframe
x2 = data_use.corr()
value = pd.DataFrame([np.zeros])
x = pd.DataFrame(data_use.columns.copy()).T
i=0
for each in data_use.columns :
    value.loc[0,i] = x2.loc[each,'Gender']
    i=i+1
    


# In[ ]:


# and drop itself from data
plot_values= value
clmn = int(np.where(plot_values>= 1)[1])
plot_values.drop([clmn], axis = 1, inplace = True)
x.drop([clmn], axis = 1, inplace = True)


# In[ ]:


# plot the values
plt.stem(x.loc[0,:], plot_values.loc[0,:])
plt.xticks(rotation=-90)
plt.plot(x.loc[0,0:21], np.linspace(0, 0, 21),'r-',linewidth=2)
plt.grid()
plt.show()


# The figure shown us that latino, musical and pop are more prefered by women. In addition, metal, hard rock, hiphop, rap, techno and trance type musics are prefered by men.

# In[ ]:


# create dataframe
x2 = data_use.corr()
value = pd.DataFrame([np.zeros])
x = pd.DataFrame(data_use.columns.copy()).T
i=0
for each in data_use.columns :
    value.loc[0,i] = x2.loc[each,'Classical music']
    i=i+1


# In[ ]:


# and drop itself from data
plot_values= value
clmn = int(np.where(plot_values>= 1)[1])
plot_values.drop([clmn], axis = 1, inplace = True)
x.drop([clmn], axis = 1, inplace = True)


# In[ ]:


# plot the values
plt.stem(x.loc[0,:], plot_values.loc[0,:])
plt.xticks(rotation=-90)
plt.plot(x.loc[0,0:21], np.linspace(0, 0, 21),'r-',linewidth=2)
plt.grid()
plt.show()


# When we analyze the graph, we can say that most of people who listen to classical music usually prefer to listen to opera. They  often listen to swing, jazz, folk, musical and alternative music. A small part of them listen to country, rock, metal, hard rock, rock n roll, latino  at the same time and they like to listen to slow music. Additionally, they dont like to listen to hiphop and rap.

# In[ ]:


# create dataframe
x2 = data_use.corr()
value = pd.DataFrame([np.zeros])
x = pd.DataFrame(data_use.columns.copy()).T
i=0
for each in data_use.columns :
    value.loc[0,i] = x2.loc[each,'Rock']
    i=i+1


# In[ ]:


# and drop itself from data
plot_values= value
clmn = int(np.where(plot_values>= 1)[1])
plot_values.drop([clmn], axis = 1, inplace = True)
x.drop([clmn], axis = 1, inplace = True)


# In[ ]:


# plot the values
plt.stem(x.loc[0,:], plot_values.loc[0,:])
plt.xticks(rotation=-90)
plt.plot(x.loc[0,0:21], np.linspace(0, 0, 21),'r-',linewidth=2)
plt.grid()
plt.show()


# As shown in the graph, people like to listen to rock music enjoy listening to metal, hardrock, punk and rock n roll. Furthermore, they can listen country, classical music, musical,raggae, swing, jazz and opera. However, they dislike dance, hiphop, rap, techno and trance music types.

# # Thanks

#  I want to thank to DatAI  for his guidance.
# 
# https://www.kaggle.com/kanncaa1/data-sciencetutorial-for-beginners
