#!/usr/bin/env python
# coding: utf-8

# ## Loading data into Pandas

# In[ ]:


import pandas as pd


# In[ ]:


df = pd.read_csv('../input/pokemon.csv')


# In[ ]:


#print the top rows
print(df.head())


# In[ ]:


#print the bottom 3 rows
df.tail(3)


# In[ ]:


#Read the headers
df.columns


# In[ ]:


#Reach each column
df[['Name','Type 1','HP']][0:5]


# In[ ]:


#Reach Each row
df.iloc[3] 


# In[ ]:


#Reach Each row with range
df.iloc[3:10] 


# In[ ]:


#Read each row with range 
for index , row in df.iterrows():
    print(index,row[['Name' , 'Total']])


# In[ ]:


#Read the specific location
df.iloc[2,1]


# In[ ]:


df.loc[df['Type 1'] == 'Fire']


# # Sorting / Describing Data

# In[ ]:


#describe data
df.describe()


# In[ ]:


df.sort_values(['Type 1' , 'HP'], ascending=True)


# In[ ]:


df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed']  


# In[ ]:


df.head(5)


# In[ ]:


#drop the column / delete the column
df = df.drop(columns= ['Total'])


# In[ ]:


df.head()


# In[ ]:


#create column using iloc
df['Total'] = df.iloc[:,4:10].sum(axis=1)


# In[ ]:


df.head()


# In[ ]:


#arrange the Dataframe
cols = list(df.columns)
df = df[cols[0:4] + [cols[-1]] + cols[4:12]]
df.head()


# ## Saving our data

# In[ ]:


#exporting the new dataframe to csv
df.to_csv('../input/modified.csv', index=False)


# In[ ]:


#exporting the new dataframe to exel
df.to_excel('modified_exel.xlsx' , index=False)


# In[ ]:


#exporting the new dataframe to text seperated by text
df.to_csv('modified_text.txt' , index=False , sep='\t')


# ## Filtering Data

# In[ ]:


df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison')]


# In[ ]:


df.loc[(df['Type 1'] == 'Grass') | (df['Type 2'] == 'Poison')]


# In[ ]:


new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison') & (df['HP'] > 70) ]
new_df


# In[ ]:


#resetting the index 
new_df = new_df.reset_index(drop=True)
new_df


# In[ ]:


df.loc[df['Name'].str.contains('Mega')]


# In[ ]:


#drop the name mega
df.loc[~df['Name'].str.contains('Mega')]


# In[ ]:


import re
df.loc[df['Type 1'].str.contains('fire|grass', flags= re.I , regex = True)]


# In[ ]:


#Start Names with 'pi'
df.loc[df['Name'].str.contains('^pi[a-z]*', flags= re.I , regex = True)]


# # Conditional Changes

# In[ ]:


df.loc[df['Type 1'] == 'Flamer' , 'Type 1'] = 'Fire' 


# In[ ]:


df


# In[ ]:


df.loc[df['Type 1'] == 'Flamer' , 'Legendary'] = True
df


# In[ ]:


df = pd.read_csv('modified.csv')


# In[ ]:


df.loc[df['Total'] > 500 , ['Generation','Legendary']] = ['Test 1 ','Test 2']


# In[ ]:


df


# ## Aggregate Statistics(groupby)

# In[ ]:


df = pd.read_csv('modified.csv')


# In[ ]:


df.groupby(['Type 1']).mean()


# In[ ]:


df.groupby(['Type 1']).mean().sort_values('Attack' , ascending=False)


# In[ ]:


df.groupby(['Type 1']).sum()


# In[ ]:


df.groupby(['Type 1']).count()


# In[ ]:


df['count'] = 1
df.groupby(['Type 1' , 'Type 2']).count()['count']

