#!/usr/bin/env python
# coding: utf-8

# # Pandas Python

# ## Pandas Series

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


# Passing ndarray to Pandas series
d1 = np.arange(100,110)
pd.Series(d1)


# In[ ]:


#Customized Index values
d2 = np.array(['a','b','c','d','e'])
pd.Series(d2,index=[1,2,3,4,5])


# In[ ]:


#with Dictionary
d3 = {"a":1,"b":2,"c":3,"d":4,"e":5}
pd.Series(d3)


# In[ ]:


#Retrive elements
data = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
s = pd.Series(data)


# 

# In[ ]:


s[:3]


# In[ ]:


s[-3:]


# In[ ]:


s[[0,2,4]]


# ## Pandas Dataframe

# In[ ]:


#creating a Dataframe
df = pd.DataFrame(
                  {"Name":["a","b","c","d"],
                  "Age" : [25,24,19,24]},
                   index =[1,2,3,4])
df


# In[ ]:


df = pd.DataFrame([['Ab', 25], ['Bb', 24], ['Cd', 19], ['Dd', 24]], columns=['Name','Age'])


# In[ ]:


df


# ## Data wrangling with pandas

# In[ ]:


ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings', 'kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
            'Rank': [1, 2, 2, 3, 3, 4, 1, 1, 2, 4, 1, 2],
            'Year': [2014, 2015, 2014, 2015, 2014, 2015, 2016, 2017, 2016, 2014, 2015, 2017],
            'Points':[876, 789, 863, 673, 741, 812, 756, 788, 694, 701, 804, 690]}
df = pd.DataFrame(ipl_data)


# In[ ]:


df


# In[ ]:


df.dtypes


# In[ ]:


df.ndim


# In[ ]:


df.shape


# In[ ]:


len(df)


# In[ ]:


df.size


# In[ ]:


df.values


# In[ ]:


df.head()


# In[ ]:


df.tail(2)


# In[ ]:


#summerize data
grades = [48, 99, 75, 80, 42, 80, 72, 68, 36, 78]

df = pd.DataFrame({'ID': ["x%d" % r for r in range(10)],
                   'Gender': ['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'M'],
                   'ExamYear': ['2007', '2007', '2007', '2008', '2008', '2008', '2008', '2009', '2009', '2009'],
                   'Class': ['algebra', 'stats', 'bio', 'algebra', 'algebra', 'stats', 'stats', 'algebra', 'bio', 'bio'],
                   'Participated': ['yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes'],
                   'Passed': ['yes' if x > 50 else 'no' for x in grades],
                   'Employed': [True, True, True, False, False, False, False, True, True, False],
                   'Grade': grades})


# In[ ]:


df


# In[ ]:


df['Grade'].value_counts()


# In[ ]:


df['ExamYear'].nunique()


# In[ ]:


df.describe()


# In[ ]:


#selectin the Data
df['Class']


# In[ ]:


df[['ID','Class','Grade']]


# In[ ]:


df.Grade


# In[ ]:


#subset observation
df[df.Gender == 'M']


# In[ ]:


df[(df.Grade> 20) & (df.ExamYear == '2008') & (df.Participated == 'yes')]


# In[ ]:


#index based slicing


# In[ ]:


grades = [48, 99, 75, 80, 42, 80, 72, 68]

df = pd.DataFrame({'ID': ["x%d" % r for r in range(8)],
                   'Gender': ['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M'],
                   'ExamYear': ['2007', '2007', '2007', '2008', '2008', '2008', '2008', '2009'],
                   'Class': ['algebra', 'stats', 'bio', 'algebra', 'algebra', 'stats', 'stats', 'algebra'],
                   'Participated': ['yes', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes'],
                   'Passed': ['yes' if x > 50 else 'no' for x in grades],
                   'Employed': [True, True, True, False, False, False, False, True],
                   'Grade': grades},
                 index = ["x%d" % r for r in range(8)])


# In[ ]:


df


# In[ ]:


df.loc[:,'Grade']


# In[ ]:


df.loc[:,['ID', 'Grade']]


# In[ ]:


df.loc['x2']


# In[ ]:


df.loc[['x1', 'x3', 'x5'],['ID', 'Grade','ExamYear']]


# In[ ]:


#integer based iloc
df.iloc[:4]


# In[ ]:


df


# In[ ]:


df.iloc[1:5, 2:7]


# ### Handling missing data

# In[ ]:


df = pd.DataFrame(
    {
        "Name": ['Ab', 'Bb', 'Cd', 'Dd', 'Ed', 'Fc'],
        "Age" : [25, 24, 19, 24, np.nan, 28],
        "Score" : [78, 84, 89, 74, 69, np.nan]},
    index = [1, 2, 3, 4, 5, 6])    


# In[ ]:


df


# In[ ]:


df.dropna() #drop row cloumn with row data


# In[ ]:


df['Age'].fillna(value = 15)


# In[ ]:


df


# In[ ]:


df.Age.mean()


# In[ ]:


df2 = df['Age'].fillna(value = df.Age.mean())


# In[ ]:


df2


# In[ ]:


df['Age'] = df['Age'].fillna(value = df.Age.mean())


# In[ ]:


df


# In[ ]:


df['Score'] = df['Age'].fillna(value=df.Age.mean())


# In[ ]:


df


# In[ ]:


pd.isnull(df)


# In[ ]:


#wide long form
data = {'Name': ['John', 'Smith', 'Liz'], 
        'Weight': [150, 170, 110], 
        'BP': [120, 130, 100]}

w_df = pd.DataFrame(data)


# In[ ]:


w_df


# In[ ]:


w_df.melt(id_vars='Name',var_name='key',value_name='value')


# In[ ]:


data = {'patient': [1, 1, 1, 2, 2],
        'obs': [1, 2, 3, 1, 2], 
        'treatment': [0, 1, 0, 1, 0],
        'score': [6252, 24243, 2345, 2342, 23525]}

Long_df = pd.DataFrame(data, columns = ['patient', 'obs', 'treatment', 'score'])


# In[ ]:


Long_df


# In[ ]:


Long_df.pivot(index='patient', columns='obs', values='score')


# In[ ]:


#making a new column


# In[ ]:


data = {'Name': ['John', 'Smith', 'Liz', 'Andy', 'Dri'], 
        'Weight': [150, 170, 110, 56, 75], 
        'BP': [120, 130, 100, 110, 125]}

df = pd.DataFrame(data)


# In[ ]:


df


# In[ ]:


df = df.assign(BPw = lambda df: df.Weight / df.BP)


# In[ ]:


df


# In[ ]:


df['BP2'] = df['Weight'] / df['BP']


# In[ ]:


df


# In[ ]:


df['BP_hl'] = np.where(df['BP'] > 100 , 'high' , 'low')


# In[ ]:


df


# In[ ]:


df.assign(st_weight = lambda df:df.Weight /df.Weight.sum())


# In[ ]:


ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Devils', 'Kings', 'Kings', 'Kings', 'Kings', 'Riders', 'Royals', 'Royals', 'Riders'],
            'Year': [2014, 2015, 2014, 2015, 2014, 2015, 2016, 2017, 2016, 2014, 2015, 2017],
            'Points':[876, 789, 863, 673, 741, 812, 756, 788, 694, 701, 804, 690]}
df = pd.DataFrame(ipl_data)


# In[ ]:


df


# In[ ]:


df.groupby(['Team']).sum()


# In[ ]:


df['ratio'] = df.groupby(['Team'], group_keys=False).apply(lambda g: g.Points/(g.Points).sum())


# In[ ]:


df


# In[ ]:


#concatenation
df_one = pd.DataFrame(
    {'Name': ['A1', 'A2', 'A3', 'A4', 'A5'],
     'subject_id':['S01','S02','S03','S04','S05'],
     'Marks_scored':[78, 50, 77, 69, 78]},
    index=[1, 2, 3, 4, 5])


# In[ ]:


df_two = pd.DataFrame(
    {'Name': ['B1', 'B2', 'B3', 'B4', 'B5'],
     'subject_id':['S01','S02','S03','S04','S05'],
     'Marks_scored':[89, 85, 78, 87, 88]},
    index=[1, 2, 3, 4, 5])


# In[ ]:


pd.concat([df_one,df_two])


# In[ ]:


pd.concat([df_one, df_two], keys=['x','y'])


# In[ ]:


pd.concat([df_one, df_two], keys=['x','y'], ignore_index=True)


# In[ ]:


#append columns of Dataframe
pd.concat([df_one, df_two], axis = 1)


# In[ ]:


#Merging Dataframe


# In[ ]:


df_left = pd.DataFrame(
    {'Student': ['St01', 'St02', 'St03', 'St01'],
     'Subject': ['Mat', 'Phy', 'Phy', 'Phy'],
     'Assign1': [54, 63, 56, 78],
     'Assign2': [66, 65, 75, 85]})

df_right = pd.DataFrame(
    {'Student': ['St01', 'St02', 'St01', 'St02'],
     'Subject': ['Mat', 'Mat', 'Phy', 'Phy'],
     'Assign3': [72, 56, 85, 96],
     'Assign4': [78, 89, 56, 88]})


# In[ ]:


df_left


# In[ ]:


df_right


# In[ ]:


pd.merge(df_left,df_right , on=['Student','Subject'])


# In[ ]:


df_right


# In[ ]:


#left join
pd.merge(df_left,df_right,on=['Student','Subject'] , how = 'left')


# In[ ]:


#right join
pd.merge(df_left, df_right, on=['Student', 'Subject'], how = 'right')


# In[ ]:


#inner join
pd.merge(df_left, df_right, on=['Student', 'Subject'], how = 'inner')


# In[ ]:


#outer join
pd.merge(df_left, df_right, on=['Student', 'Subject'], how = 'outer')

