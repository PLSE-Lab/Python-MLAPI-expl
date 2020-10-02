#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Handle table-like data and matrices
import numpy as np
import pandas as pd

# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.style.use( 'ggplot' )
sns.set_style( 'white' )
pylab.rcParams[ 'figure.figsize' ] = 8 , 6


# In[ ]:


df = pd.read_csv(r"../input/Pokemon.csv")


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.drop('#' , axis = 1 , inplace = True)   # removing '#' column because indexing already done


# In[ ]:


df.shape


# In[ ]:


df.describe(include = 'all')


# In[ ]:


df["Type 1"].unique()


# In[ ]:


df["Type 2"].unique()


# In[ ]:


df["Type 1"].value_counts()


# In[ ]:


df["Type 2"].value_counts()


# In[ ]:


df.isnull().sum()


# In[ ]:


df['Type 2'].fillna(df['Type 1'], inplace=True)    # replacing the NaN values of type 2 with type 1


# In[ ]:


df.isnull().sum()


# In[ ]:


# NaN values removed


# In[ ]:


labels = 'Water', 'Normal', 'Grass', 'Bug', 'Psychic', 'Fire', 'Electric', 'Rock', 'Other'
sizes = [112, 98, 70, 69, 57, 52, 44, 44, 175]
colors = ['B', 'silver', 'G', '#006400', '#E40E00', '#A00994', '#613205', '#FFED0D', '#16F5A7']
explode = (0.1, 0.0, 0.1, 0, 0.1, 0.0, 0.1, 0, 0.1) 
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=180)
plt.axis('equal')
plt.title("Percentage of Different types of Types I Pokemon")
plt.plot()
fig=plt.gcf()
fig.set_size_inches(10,10)
plt.show()


# In[ ]:


correlation_map = df[['Name', 'Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense',
       'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']].corr()
obj = np.array(correlation_map)
obj[np.tril_indices_from(obj)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,12)
sns.heatmap(correlation_map, mask=obj,vmax=.7, square=True,annot=True)


# In[ ]:


# this shows 
# 'HP', 'Attack', 'Defense','Sp. Atk', 'Sp. Def', 'Speed' are highly correlated to total.


# In[ ]:


#self relation would always be equal to 1


# In[ ]:


# generation is also having nearly no relation with every other variable
# so we can remove it


# In[ ]:


new=df.copy()
new.drop(["Total","Legendary"],axis = 1,inplace=True)
new.head()

# creating new dataframe


# In[ ]:




figure = plt.figure(figsize=(12,7))
sns.boxplot(data=new)


# In[ ]:


df.groupby("Type 1")["Defense"].mean()


# In[ ]:


df.groupby("Type 1")["Attack"].mean()


# In[ ]:


# similary for other attributes we can group by as above


# In[ ]:


# now visualization on the basis of type1:


# In[ ]:


# creating a function mean_attribute, which will take attribute such as Desfense,Attack,
# etc and group them by type
# and will take the mean of each and plot a bar plot 


# In[ ]:


def mean_attribute(type_no,attribute,dataframe):
    a=dataframe.groupby(type_no)[attribute].mean()
    temp=pd.DataFrame(a)
    temp=temp.reset_index()
    temp = temp.sort_values(by=[attribute])
    fig, axes = plt.subplots(3,1)
    fig.set_size_inches(15, 15)
    sns.stripplot(data=df,x="Type 1",y="Total",ax=axes[0],jitter=True)
    sns.boxplot(data=df,y="Total",x="Type 1",orient="v",ax=axes[1])
    sns.barplot(temp[type_no],temp[attribute],ax=axes[2])


# In[ ]:


mean_attribute("Type 1","Defense",df)


# In[ ]:


mean_attribute("Type 1","Attack",df)


# In[ ]:


mean_attribute("Type 1","Sp. Atk",df)


# In[ ]:


mean_attribute("Type 1","HP",df)


# In[ ]:


mean_attribute("Type 1","Speed",df)


# In[ ]:




mean_attribute("Type 1","Sp. Def",df)


# In[ ]:


# similarly you can go for type2 also.


# In[ ]:


# since generation column was deleted by mistake in the original dataframe df
# hence the need to read the csv file again arises


# In[ ]:


orig = pd.read_csv(r"../input/Pokemon.csv")      # new dataframe orig containg all the columns intact


# In[ ]:


orig.head()


# In[ ]:


orig.shape


# In[ ]:


orig.isnull().sum()


# In[ ]:


# cleaning   
orig['Type 2'].fillna(orig['Type 1'], inplace=True)    # replacing the NaN values of type 2 with type 1


# In[ ]:


orig.describe(include='all')


# In[ ]:


#checking if Generation has any effect on total


# In[ ]:


# using orig as our dataframe
figure = plt.figure(figsize=(10,9))
sns.boxplot(y="Total", x="Generation", data=orig)
ax = sns.swarmplot(x="Generation", y="Total", data=orig, color=".15")


# In[ ]:


figure = plt.figure(figsize=(12,8))

sns.boxplot(y="Defense", x="Generation", data=orig)
ax = sns.swarmplot(x="Generation", y="Defense", data=orig, color=".25")


# In[ ]:




figure = plt.figure(figsize=(12,8))

sns.boxplot(y="Attack", x="Generation", data=orig)
ax = sns.swarmplot(x="Generation", y="Attack", data=orig, color=".25")


# In[ ]:


#   seen above 
#   generation effect on total , defense , attack


# In[ ]:


# similarly can be made for others


# In[ ]:


# heat map previously drawn can also be rendered like this in full square 
plt.figure(figsize=(15,10)) #manage the size of the plot
sns.heatmap(orig.corr(),annot=True, square = True) 
plt.show()


# In[ ]:


#  speed is also having negligible relation with defense


# In[ ]:


# now making some violin plots for
# attack vs type 1
# defense vs type 1
# total vs generation


# In[ ]:


plt.subplots(figsize = (15,8))
plt.title('Attack by Type1')
sns.violinplot(x = "Type 1", y = "Attack",data = df)
plt.ylim(0,200)
plt.show()


# In[ ]:


# this clearly shows that dragon type is of highly attacking nature when considering higher attack points


# In[ ]:


plt.subplots(figsize = (15,8))
plt.title('Defense by Type1')
sns.violinplot(x = "Type 1", y = "Defense",data = df)
plt.ylim(0,200)
plt.show()


# In[ ]:


# clearly shows that steel type is more of defensive nature


# In[ ]:


plt.subplots(figsize = (15,8))
plt.title('Strongest Genaration')
sns.violinplot(x = "Generation", y = "Total",data = orig)
plt.show()


# In[ ]:


# generation 3 is the strongest of all 


# In[ ]:


plt.subplots(figsize = (20,10))
sns.swarmplot(x="Type 1", y="Total", data=orig, hue="Type 1");


# In[ ]:


# each dot above represents a pokemon 


# In[ ]:


# now considering that whether legendary is having any effect on our data


# In[ ]:




sns.factorplot(x="Type 1", y="Total", hue="Legendary",data=df, kind="bar",aspect=3,size=4)


# In[ ]:


sns.factorplot(x="Type 2", y="Total", hue="Legendary",data=df, kind="bar",aspect=3,size=4)


# In[ ]:


# from above
# we can see that generally Legendary pokemons have more total than any of non legendary pokemon
#therefore we can not discard Legendary from data.


# In[ ]:


# Now lets compare any two pokemons of our choice


# In[ ]:


# Taking example as of CharizardMega Charizard X and MewtwoMega Mewtwo X


# In[ ]:


def Comp(pok1 , pok2):
    a = orig[(orig.Name == pok1) | (orig.Name == pok2)]
    sns.factorplot(x="Name", y="Total",data=a , kind="bar",aspect=2,size=3)
    sns.factorplot(x="Name", y="Attack",data=a, kind="bar",aspect=2,size=3)
    sns.factorplot(x="Name", y="HP",data=a, kind="bar",aspect=2,size=3)
    sns.factorplot(x="Name", y="Defense",data=a, kind="bar",aspect=2,size=3)
    sns.factorplot(x="Name", y="Sp. Atk",data=a, kind="bar",aspect=2,size=3)
    sns.factorplot(x="Name", y="Sp. Def",data=a, kind="bar",aspect=2,size=3)
    sns.factorplot(x="Name", y="Speed",data=a, kind="bar",aspect=2,size=3)


# In[ ]:


Comp("CharizardMega Charizard X" , "MewtwoMega Mewtwo X")


# In[ ]:


# similarly we can take any two and just compare them in various attributes


# In[ ]:


#  THANK YOU


# In[ ]:




