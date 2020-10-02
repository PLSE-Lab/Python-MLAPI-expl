#!/usr/bin/env python
# coding: utf-8

# ## POKEMON STATS ANALYSIS

# In[ ]:


import pandas as pd   #importing all the important packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')


# In[ ]:


import os
print(os.listdir("../input/"))


# In[ ]:



df =  pd.read_csv("../input/Pokemon.csv")  #read the csv file and save it into a variable


# In[ ]:


df.head() # print the 5 first rows


# In[ ]:


df.columns # get all the columns


# In[ ]:


df[["Name","Type 1"]] # get spesific columns


# In[ ]:


df = df.set_index('Name') #change and set the index to the name attribute


# In[ ]:


df.head()


# In[ ]:


df=df.drop(['#'],axis=1) #remove the # column
df.head()


# ### CLEANING THE DATAFRAME

# lets get all the names that contain Mega in their name

# In[ ]:


df[df.index.str.contains("Mega")]


# Lets understand what happand:
# The following line give as array with boolean variables,it tells as if the string in the given place fulfill the condition.

# In[ ]:


df.index.str.contains("Mega")


# When putting it in the '[]' it will return us only the places where there is an True in the given array

# In[ ]:


df[df.index.str.contains("Mega")].head()


# In[ ]:


## The index of Mega Pokemons contained extra and unneeded text. Removed all the text before "Mega"  
df.index = df.index.str.replace(".*(?=Mega)", "")
df.head(10)


# Lets get all the values with mega in thier name again to see if it worked

# In[ ]:


df[df.index.str.contains("Mega")].head()


# In[ ]:


df.columns = df.columns.str.upper().str.replace('_', '') #change into upper case
df.head()


# In[ ]:


df[df['LEGENDARY']==True].head(20)  #Showing the legendary pokemons


# In[ ]:


print('The columns of the dataset are: ',df.columns) #show the dataframe columns
print('The shape of the dataframe is: ',df.shape)    #shape of the dataframe


# In[ ]:


#some values in TYPE2 are empty and thus they have to be filled or deleted
df['TYPE 2'].fillna(df['TYPE 1'], inplace=True) #fill NaN values in Type2 with corresponding values of Type
df.head(100)


# In[ ]:


df.loc['Bulbasaur'] #retrieves complete row data from index with value Bulbasaur


# In[ ]:


df.iloc[0] #retrieves complete row date from index 0 ; integer version of loc


# In[ ]:


df.ix[0] #similar to iloc


# In[ ]:


df.ix['Kakuna'] #similar to loc


# loc works on labels in the index.
# 
# iloc works on the positions in the index (so it only takes integers).
# 
# ix usually tries to behave like loc but falls back to behaving like iloc if the label is not in the index.
# 
# inoreder to find details about any pokemon, just specify its name

# In[ ]:


#filtering pokemons using logical operators
df[((df['TYPE 1']=='Fire') | (df['TYPE 1']=='Dragon')) & ((df['TYPE 2']=='Dragon') | (df['TYPE 2']=='Fire'))].head(3)


# In[ ]:


print("MAx HP:",df['HP'].idxmax())  #returns the pokemon with highest HP
print("Max DEFENCE:",(df['DEFENSE']).idxmax()) #similar to argmax()


# In[ ]:


df.sort_values('TOTAL',ascending=False).head(3)  #this arranges the pokemons in the descendng order of the Totals.
#sort_values() is used for sorting and ascending=False is making it in descending order


# In[ ]:


print('The unique  pokemon types are',df['TYPE 1'].unique()) #shows all the unique types in column
print('The number of unique types are',df['TYPE 1'].nunique()) #shows count of unique values 


# In[ ]:


print(df['TYPE 1'].value_counts(), '\n' ,df['TYPE 2'].value_counts())#count different types of pokemons


# In[ ]:


df.groupby(['TYPE 1']).size()  #same as above


# In[ ]:


(df['TYPE 1']=='Bug').sum() #counts for a single value


# In[ ]:


df_summary = df.describe() #summary of the pokemon dataframe
df_summary


# Lets see how we change values in a column, based on condition:
# 
# We will add 10 to speed to all Legedary Pokrmons.

# In[ ]:


df["LEGENDARY"] == True # the condition


# In[ ]:


df[df["LEGENDARY"] == True].head()


# In[ ]:


df.loc[df["LEGENDARY"] == True,"SPEED"].head()


# In[ ]:


df.loc[df["LEGENDARY"] == True,"SPEED"] += 10


# In[ ]:


df.describe()


# Lets revers the changes

# In[ ]:


df.loc[df["LEGENDARY"] == True,"SPEED"] -= 10


# ## VISUALISATIONS

# A bar chart of the number of pokemons from each generation

# In[ ]:


df["GENERATION"].value_counts().sort_index().plot.bar()


# A horizontal bar chart of the number of pokemons for each type 1

# In[ ]:


df["TYPE 1"].value_counts().plot.barh()


# ##### The attack distribution for the pokemons across all the genarations

# In[ ]:


df.hist(column='ATTACK')


# Above is a Histogram showing the distribution of attacks for the Pokemons. The average value is between 75-77

# ### Fire Vs Water

# In[ ]:


fire=df[(df['TYPE 1']=='Fire') | ((df['TYPE 2'])=="Fire")] #fire contains all fire pokemons
water=df[(df['TYPE 1']=='Water') | ((df['TYPE 2'])=="Water")]  #all water pokemins
ax = fire.plot.scatter(x='ATTACK', y='DEFENSE', color='Red', label='Fire')
water.plot.scatter(x='ATTACK', y='DEFENSE', color='Blue', label='Water', ax=ax);


# In[ ]:





# In[ ]:





# This shows that fire type pokemons have a better attack than water type pokemons but have a lower defence than water type.

# ### Strongest Pokemons By Types

# In[ ]:


strong=df.sort_values(by='TOTAL', ascending=False) #sorting the rows in descending order
strong.drop_duplicates(subset=['TYPE 1'],keep='first') #since the rows are now sorted in descending oredr
#thus we take the first row for every new type of pokemon i.e the table will check TYPE 1 of every pokemon
#The first pokemon of that type is the strongest for that type
#so we just keep the first row


# ## Distribution of various pokemon types

# In[ ]:


df['TYPE 1'].value_counts().plot.pie()


# # The following part is more advanced you can stop here

# ## All stats analysis of the pokemons

# In[ ]:


df2=df.drop(['GENERATION','TOTAL'],axis=1)
sns.boxplot(data=df2)
plt.ylim(0,300)  #change the scale of y axix
plt.show()


# In[ ]:


plt.subplots(figsize = (15,5))
plt.title('Attack by Type1')
sns.boxplot(x = "TYPE 1", y = "ATTACK",data = df)
plt.ylim(0,200)
plt.show()


# #### This shows that the Dragon type pokemons have an edge over the other types as they have a higher attacks compared to the other types. Also since the fire pokemons have lower range of values, but higher attacks, they can be preferred over the grass and water types for attacking.
# 

# In[ ]:


plt.subplots(figsize = (15,5))
plt.title('Attack by Type2')
sns.boxplot(x = "TYPE 2", y = "ATTACK",data=df)
plt.show()


# In[ ]:


plt.subplots(figsize = (15,5))
plt.title('Defence by Type')
sns.boxplot(x = "TYPE 1", y = "DEFENSE",data = df)
plt.show()


# This shows that steel type pokemons have the highest defence but normal type pokemons have the lowest defence

# ### Now lets see the same stats in violinplot

# In[ ]:


plt.subplots(figsize = (20,10))
plt.title('Attack by Type1')
sns.violinplot(x = "TYPE 1", y = "ATTACK",data = df)
plt.ylim(0,200)
plt.show()


# What the violinplot actually does is it plots according to the density of a region. This means that the parts of the plot where the width is thicker denotes a region with higher density points whereas regions with thinner area show less densely populated points.

# In[ ]:


plt.subplots(figsize = (20,10))
plt.title('Attack by Type1')
sns.violinplot(x = "TYPE 1", y = "DEFENSE",data = df)
plt.ylim(0,200)
plt.show()


# In[ ]:


plt.subplots(figsize = (15,5))
plt.title('Strongest Genaration')
sns.violinplot(x = "GENERATION", y = "TOTAL",data = df)
plt.show()


# This shows that generation 3  has the better pokemons

# ### Strong Pokemons By Type

# In[ ]:


plt.figure(figsize=(12,6))
top_types=df['TYPE 1'].value_counts()[:10] #take the top 10 Types
df1=df[df['TYPE 1'].isin(top_types.index)] #take the pokemons of the type with highest numbers, top 10
sns.swarmplot(x='TYPE 1',y='TOTAL',data=df1,hue='LEGENDARY') # this plot shows the points belonging to individual pokemons
# It is distributed by Type
plt.axhline(df1['TOTAL'].mean(),color='red',linestyle='dashed')
plt.show()


#  Legendary Pokemons are mostly taking the top spots in the Strongest Pokemons
# 

# ### Finding any Correlation between the attributes

# In[ ]:


plt.figure(figsize=(10,6)) #manage the size of the plot
sns.heatmap(df.corr(),annot=True) #df.corr() makes a correlation matrix and sns.heatmap is used to show the correlations heatmap
plt.show()


# From the heatmap it can be seen that there is not much correlation between the attributes of the pokemons. The highest we can see is the correlation between Sp.Atk and the Total 

# ### Number of Pokemons by Type And Generation

# ### Type 1

# In[ ]:


a=df.groupby(['GENERATION','TYPE 1']).count().reset_index()
a=a[['GENERATION','TYPE 1','TOTAL']]
a=a.pivot('GENERATION','TYPE 1','TOTAL')
a[['Water','Fire','Grass','Dragon','Normal','Rock','Flying','Electric']].plot(color=['b','r','g','#FFA500','brown','#6666ff','#001012','y'],marker='o')
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# We can see that water pokemons had the highest numbers in the 1st Generation. However the number has decreased with passing generations. Similarly Grass type pokemons showed an increase in their numbers till generation 5.

# In[ ]:


a=df.groupby(['GENERATION','TYPE 2']).count().reset_index()
a=a[['GENERATION','TYPE 2','TOTAL']]
a=a.pivot('GENERATION','TYPE 2','TOTAL')
a[['Water','Fire','Grass','Dragon','Normal','Rock','Flying','Electric']].plot(color=['b','r','g','#FFA500','brown','#6666ff','#001012','y'],marker='o')
fig=plt.gcf()
fig.set_size_inches(12,6)
plt.show()


# This graph shows that the number of Type2 Grass Pokemons has been steadily increasing. The same is the case for the Dragon Type Pokemons. For other Types the trends are somewhat uneven.

# In[ ]:





# In[ ]:




