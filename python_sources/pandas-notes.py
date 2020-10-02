#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os


# https://pandas.pydata.org/pandas-docs/stable/10min.html

# In[ ]:


df = pd.DataFrame([
        [2, 'foo', np.NaN],
        [2, 'bar', np.NaN],
        [3, 'x', np.NaN],
        [3, 'bar', np.NaN],
        [3, np.NaN, np.NaN],
    ], columns=['rtx', 'name','test'])

df


# In[ ]:


v = [0,0,0,1,0]
df['v'] = v
df


# In[ ]:


df_x = df[['rtx','name']]
df_x


# In[ ]:


df_x = df_x.replace(np.nan, 'unknown', regex=True)
df_x


# In[ ]:


(df['name'] == 'foo').sum(), (df['name'] == 'bar').sum()


# In[ ]:


df[df['rtx'] == 2]


# In[ ]:


df.groupby('name')['rtx'].mean()


# In[ ]:


df.name.value_counts()


# In[ ]:


df.describe()


# In[ ]:


round(df.rtx.std(),2)


# In[ ]:


round(df.rtx.median(),2)


# In[ ]:


pd.crosstab(df['rtx'], df['name'], margins=True)


# In[ ]:


df[df['name'] != 'x']


# In[ ]:


df = df[df['name'] != 'x'].groupby('name').count().sort_values('name',ascending=False)
df.head()


# In[ ]:


df = df.drop("v", axis=1)
df.head()


# In[ ]:





# # Concat

# In[ ]:


pd.concat([df,df], axis=0)


# In[ ]:


pd.concat([df,df], axis=1)


# In[ ]:


df.fillna(10,inplace=True)
df


# In[ ]:


#df = pd.read_csv(path/'train.csv',)#.sample(frac=0.05)


# In[ ]:


a = pd.DataFrame(columns = ['file','text'])
a.loc[0]=['text','s']
a.loc[1]=['text2','s2']
a


# In[ ]:


d = {}
for i in range(0,5):
     d.setdefault('result', [])
     d['result'].append(i)

df = pd.DataFrame(d)
df


# In[ ]:


df = pd.DataFrame({'a':range(10),'b':range(10 ,20),'c':sorted(list(range(5))+list(range(5)))})
df


# ## Index

# In[ ]:


df.set_index('c',inplace=True)
df


# In[ ]:


pd.DataFrame(index=df.index, data=df, columns = ['a'])


# # Group By

# In[ ]:


source_dict = {
    'A': ['foo', 'bar', 'baz', 'foo', 'bar', 'baz'],
    'B': ['cat_a', 'cat_a', 'cat_a', 'cat_b', 'cat_b', 'cat_b'],
    'C': [1, 2, 3, 2, 2, 0]
}

example = pd.DataFrame(source_dict)

example


# In[ ]:


example.groupby(['B']).max()


# In[ ]:


example.groupby(['B']).C.apply(lambda x: x.is_monotonic_decreasing)


# In[ ]:


example.groupby(['B']).C.apply(lambda x: max(x) )


# # Rolling Sum & Mean

# https://www.geeksforgeeks.org/python-pandas-dataframe-rolling/

# In[ ]:


df = pd.DataFrame({'a':range(10),'b':range(10 ,20),'c':sorted(list(range(5))+list(range(5)))})
df


# In[ ]:


chunk = 1
def mysum(a):
    global chunk
    #print("chunk",chunk)
    chunk += 1
    #print(a)
    return a.sum()

df.groupby('c').apply(mysum)


# In[ ]:


chunk = 1
df.rolling(window=2).apply(mysum,raw=True)


# In[ ]:


df = pd.DataFrame({'B': [1, 1, 1, 1, 1,1,1,1,2,2,2,2,2,2]})
df


# In[ ]:


df['B'].rolling(3, win_type ='triang').mean()


# In[ ]:


df.B =df.rolling(3, min_periods=1).sum()


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(df['B'],label= 'B')
plt.plot(df['B'].rolling(2).mean(),label= 'MA 2')
plt.plot(df['B'].rolling(6).mean(),label= 'MA 6')
plt.legend(loc='best')
plt.title('Wells Fargo\nClose and Moving Averages')
plt.show()


# # Sum by column

# In[ ]:


df = pd.DataFrame({'a': [1,2,3], 'b': [2,3,4], 'c':['dd','ee','ff'], 'd':[5,9,1]})
df


# In[ ]:


df['e'] = df.sum(axis=1)
df


# In[ ]:


col_list= list(df)
col_list.remove('e')
col_list


# In[ ]:


df['f'] = df[col_list].sum(axis=1)
df


# In[ ]:


df['g'] = df['a'] + df['b'] + df['d']
df


# In[ ]:


df['h']=df.iloc[:,[0,3]].sum(axis=1)
df


# In[ ]:


df.iloc[:,1:]


# In[ ]:


df['h']=df.iloc[:,1:].sum(axis=1)
df


# # Groupby

# In[ ]:


df = pd.DataFrame([
        [1, '10'],
        [1, '21'],
        [1, '30'],
        [2, '10'],
        [2, '20'],
        [2, '30'],
        [3, '10'],
        [3, '20'],
    ], columns=['userid', 'name'])

df


# In[ ]:


df_stack = df.sort_values('userid').groupby('userid')['name'].apply(lambda df: df.reset_index(drop=True)).unstack()
df_stack


# In[ ]:


df_stack = pd.DataFrame(df_stack.to_records())
df_stack


# In[ ]:


cat_cols = ['userid']


# In[ ]:


def f(x):
    if x != 'userid':
         return x+'S'
    else :
        return x


# In[ ]:


df_stack.rename(columns=lambda x: f(x), inplace=True)
df_stack


# # Regression

# In[ ]:


def regression(df):

    # first edit Date column from datetime to days
    x  = (df['Date'] - df['Date'].min()).astype('timedelta64[D]')
    y = df['Sales']
    
    # test if data contains more atleast 2 points
    if len(x) < 2:
        return np.nan
    
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html
    m, b = np.polyfit(x, y, 1) # degree 1 = linear regression

    return m


# In[ ]:


#example.groupby(['B']).apply(regression)


# # Text Encoding Categorical 

# In[ ]:


def catog() :
    df = pd.DataFrame({'A' : list('aabcda'), 'B' : list('bcdaae')})
    return df
df = catog()
df


# In[ ]:


#df.apply(lambda x: x.astype('category'))


# In[ ]:


#dict([(category, code) for code, category in enumerate(df.col.cat.categories)])


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['A'] = le.fit_transform(df['A'])
df


# In[ ]:


le.classes_


# In[ ]:


df.B.astype('category').cat.codes


# In[ ]:


df.B.astype('category').cat.categories


# In[ ]:


mycolumns = ['A', 'B']
df = pd.DataFrame(columns=mycolumns)
rows = [['a','b'],['d','g'],['y','z'],[np.nan,np.nan]]
for row in rows:
    df.loc[len(df)] = row
df


# In[ ]:


#pd.get_dummies(df, dummy_na=True)


# In[ ]:


char_cols = df.dtypes.pipe(lambda x: x[x == 'object']).index
label_mapping = {}

for c in char_cols:
    df[c], label_mapping[c] = pd.factorize(df[c])


# In[ ]:


df


# In[ ]:


label_mapping


# In[ ]:


for n,c in df.items(): 
    df = pd.get_dummies(df, dummy_na=True)
    df = pd.concat([df], axis=1)


# In[ ]:


df


# In[ ]:


df = pd.DataFrame({'A' : list('aabcda'), 'B' : list('bcdaae')})
df


# https://www.datacamp.com/community/tutorials/categorical-data

# In[ ]:


labels = df['A'].astype('category').cat.categories.tolist()
labels


# In[ ]:


replace_map_comp = {'A' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
replace_map_comp


# In[ ]:


df.replace(replace_map_comp, inplace=True)
df.head()


# In[ ]:


#df.A.replace(replace_map_comp, inplace=True)
#df.head()


# # Automated Text Encoding Categorical 

# In[ ]:


df = catog()
df


# In[ ]:


cat_dict = {col: {n: cat for n, cat in enumerate(df[col].astype('category').cat.categories)} 
     for col in df}

cat_dict


# In[ ]:


enc_dict = {}
for key, value in cat_dict.items():
    enc_dict[key] = dict(map(reversed, value.items()))


# In[ ]:


cat_dict 


# In[ ]:


#dict_a = {'A': {'a': 1, 'b': 2, 'c': 3, 'd': 4}}
#dict_b = {'B': {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e'}}


# In[ ]:


df.replace(enc_dict, inplace=True)
df.head()


# In[ ]:


df = df.replace(np.nan, 'np.nan', regex=True)
df


# **# .loc

# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.loc.html

# In[ ]:


df = catog()
df


# In[ ]:


df.loc[ (df['A'] == 'a') & ( df['B']=='b') ] 


# In[ ]:


df.loc[df['A'] == 'a', :]


# In[ ]:


df.loc[df['A'] == 'a', :].sort_values(by='B')


# # .iloc

# In[ ]:





# In[ ]:


df.iloc[ : , 1:2 ]


# # Merging

# https://pandas.pydata.org/pandas-docs/stable/merging.html

# In[ ]:


df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']})
df1.set_index('A', inplace=True)
df1


# In[ ]:


df2 = pd.DataFrame({'A': ['A3', 'A4', 'A5', 'A6'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']})
df2.set_index('A', inplace=True)
df2


# In[ ]:


cc_join = pd.concat([df1, df2], axis=1, join='inner')
cc_join


# In[ ]:


df3 = pd.DataFrame({'A': ['A3', 'A3', 'A5', 'A6'],
                    'E': ['B0', 'B1', 'B2', 'B3'],
                    'F': ['C0', 'C1', 'C2', 'C3'],
                    'G': ['D0', 'D1', 'D2', 'D3']})
df3.set_index('A', inplace=True)
df3


# In[ ]:


cc_merge = df1.merge(df3, on='A')

cc_merge.groupby('A').agg('count')


# In[ ]:


cc_merge = df1.join(df3, on='A')
cc_merge


# # KFold
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

# In[ ]:


import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
X = np.array([1, 2, 3, 4])
y = np.array([0, 0, 1, 1])
#skf = StratifiedKFold(n_splits=2)
skf = KFold(n_splits=4)
skf.get_n_splits(X, y)


# In[ ]:


print(skf)  
#StratifiedKFold(n_splits=3, random_state=None, shuffle=False)
#for train_index, test_index in skf.split(X, y):
for n_fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    print("************************* Training on fold " + str(n_fold+1) + " ***********************")
    
    print("train_index:", train_index, " test_index:", test_index)
    X_train = X[train_index]
    X_test = X[test_index]
    
    y_train = y[train_index]
    y_test = y[test_index]
    
    print("\nX_train:", X_train, " \nX_test:", X_test)
    print("\ny_train:", y_train, " \ny_test:", y_test)


# ## Time Series
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html

# In[ ]:


from sklearn.model_selection import TimeSeriesSplit
time_split = TimeSeriesSplit(n_splits=5)


# In[ ]:


from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])

print(time_split)  
TimeSeriesSplit(max_train_size=None, n_splits=5)
for train_index, test_index in time_split.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]


# # Fast AI

# In[ ]:


#from fastai import structured


# In[ ]:


df = pd.DataFrame({'col1' : [1, np.nan, 3], 'col2' : ['a', 'b', 'a']})
df


# In[ ]:


#structured.train_cats(df)    
#df


# In[ ]:


#x, y, nas = structured.proc_df(df, 'col1')
#x


# ## Graphs

# In[ ]:


df = pd.DataFrame([
        ['1', 16],
        ['1', 23],
        ['2', 25],
        ['2', 34],
     ['2', 30],
    ['2', 30],
    ['2', 30],
        ['3', 38],
        ['3', 44],
    ], columns=['class', 'age'])

df


# In[ ]:


df.hist()


# In[ ]:


df.hist('age', bins=5)


# In[ ]:


import seaborn as sns


# In[ ]:


sns.boxplot(df['class'], df['age']);


# In[ ]:


import matplotlib.pyplot as plt
# Set up the matplotlib figure
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

sns.barplot(x=df['class'], y=df['age'], palette="rocket", ax=ax1)
sns.barplot(x=df['class'], y=df['age'], palette="rocket", ax=ax2)


# ## Statistics

# In[ ]:


df.age.quantile(0.95)


# In[ ]:


df = pd.DataFrame({'A': [1, 2, 3], 
                   'B': [1, 1, 1],
                   'C': [1, 2, 1]})
df.head()


# In[ ]:


df.nunique(axis=0) #row wise


# In[ ]:


df[['A','B']]


# In[ ]:


df[['A','B']].nunique(axis=1) #column wise


# In[ ]:


df.nunique(axis=1) #column wise


# In[ ]:


np.log(df.A.values)


# ### Save to csv

# In[ ]:


df = pd.DataFrame([[1, 'a'], [2, 'b']],columns=['fn','label'])
df.head()


# In[ ]:


df.to_csv('train_labels_as_strings.csv', index=False)


# In[ ]:


get_ipython().system('cat train_labels_as_strings.csv')


# # Feather

# In[ ]:


df = pd.DataFrame([[1, 'a'], [2, 'b']],columns=['fn','label'])
df.head()


# In[ ]:


df.to_feather('feather.ftr');

readFrame = pd.read_feather('feather.ftr');

readFrame


# # References
# 
# 1. https://medium.com/@msalmon00/helpful-python-code-snippets-for-data-exploration-in-pandas-b7c5aed5ecb9
# 1. https://towardsdatascience.com/be-a-more-efficient-data-scientist-today-master-pandas-with-this-guide-ea362d27386

# In[ ]:





# In[ ]:




