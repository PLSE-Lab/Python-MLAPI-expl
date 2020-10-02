#!/usr/bin/env python
# coding: utf-8

# ## KKbox Music Recommender Preprocessing and EDA  

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
import missingno as msno
import gc
from sklearn.model_selection import cross_val_score,GridSearchCV,train_test_split
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.metrics import roc_curve,roc_auc_score,classification_report,mean_squared_error,accuracy_score
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,BaggingClassifier,VotingClassifier,AdaBoostClassifier
import lightgbm as lgb
from sklearn.metrics import precision_recall_curve,roc_auc_score,classification_report,roc_curve
from tqdm import tqdm


# In[ ]:


from subprocess import check_output

train = pd.read_csv('../input/wsdm-kkbox/train.csv')
test = pd.read_csv('../input/wsdm-kkbox/test.csv')
songs = pd.read_csv('../input/wsdm-kkbox/songs.csv')
members = pd.read_csv('../input/wsdm-kkbox/members.csv')
#sample = pd.read_csv('../input/wsdm-kkbox/sample_submission.csv')


# In[ ]:


train.head()


# In[ ]:


test.head()


# In[ ]:


songs.head()


# In[ ]:


members.head()


# In[ ]:


members.shape
train.info()
print("\n")
songs.info()
print("\n")
members.info()


# In[ ]:


plt.figure(figsize=(20,15))
sns.set(font_scale=2)
sns.countplot(x='source_type',hue='source_type',data=train)
sns.set(style="darkgrid")
plt.xlabel('source types',fontsize=30)
plt.ylabel('count',fontsize=30)
plt.xticks(rotation='45')
plt.title('Count plot source types for listening music',fontsize=30)
plt.tight_layout()


# In the above visualization we can see that the local library is more preferred than most of other source types and online playlist occupies the second position

# In[ ]:


plt.figure(figsize=(20,15))
sns.set(font_scale=2)
sns.countplot(y='source_screen_name',data=train,facecolor=(0,0,0,0),linewidth=5,edgecolor=sns.color_palette('dark',3))
sns.set(style="darkgrid")
plt.xlabel('source types',fontsize=30)
plt.ylabel('count',fontsize=30)
plt.xticks(rotation='45')
plt.title('Count plot for which  screen using ',fontsize=30)
plt.tight_layout()


# From the above plot we can higlight that most of the listeners listen to local playlist

# In[ ]:


plt.figure(figsize=(20,15))
sns.set(font_scale=2)
sns.countplot(x='source_system_tab',hue='source_system_tab',data=train)
sns.set(style="darkgrid")
plt.xlabel('source types',fontsize=30)
plt.ylabel('count',fontsize=30)
plt.xticks(rotation='45')
plt.title('Count plot for system tab there are using',fontsize=30)
plt.tight_layout()


# from the above plot we can see that most of the people who had installed kkbox app for music basically go back to their old songs rather than exploring to new songs.  from that we can assume that most of them are using it only as a player rather than music library. 

# In[ ]:


import matplotlib as mpl

mpl.rcParams['font.size']=40.0
labels=['Male','Female']
plt.figure(figsize = (10,10))
sizes = pd.value_counts(members.gender)

patches, texts, autotexts = plt.pie(sizes, labels=labels, autopct='%.0f%%',shadow=True, radius=0.6, startangle=90)

for t in texts:
    t.set_size('smaller')
plt.legend()
plt.show()


# from the plot we can see that the male and female ratio of the users using the app is almost the same. which can help us deduce that it maintains gender neutrality

# In[ ]:


import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 40.0
plt.figure(figsize = (20, 20)) 

group_names=['explore','my library','search','discover','radio','listen with','notification','settings']
group_size=pd.value_counts(train.source_system_tab)
print(group_size)
subgroup_names=['Male','Female']
subgroup_size=pd.value_counts(members.gender)

# Create colors
a, b, c,d,e,f,g,h=[plt.cm.autumn, plt.cm.GnBu, plt.cm.YlGn,plt.cm.Purples,plt.cm.cool,plt.cm.RdPu,plt.cm.BuPu,plt.cm.bone]

fig, ax = plt.subplots()
ax.axis('equal')
mypie, texts = ax.pie(group_size,radius=2, labels= group_names,colors = [a(0.6), b(0.6), c(0.6),d(0.6), e(0.6), f(0.6),g(0.6)])
plt.setp(mypie, width=0.3, edgecolor='white')

plt.legend() 
# show it
plt.show()


# from the above pie chart we can infer that most of the people are using explore feature in it and mylibrary the next 

# In[ ]:


mpl.rcParams['font.size'] = 40.0
plt.figure(figsize = (20, 20)) 
sns.distplot(members.registration_init_time)
sns.set(font_scale=2)
plt.ylabel('estimated-pdf',fontsize=40)
plt.xlabel('registration time ' ,fontsize=40)


# from the above distribution it can be infered that teh max people registered during an interval of 2015 to 2018 with most of them registering in the year 2016

# In[ ]:


members.describe()


# In[ ]:


songs.describe()


# In[ ]:


train.describe()


# it is visible that in members and songs csv files,there is a large differences between min and max values which can be give us an inference that there are outliers in the csv files. these has to be removed before proceeding further

# In[ ]:


train_members = pd.merge(train, members, on='msno', how='inner')
train_merged = pd.merge(train_members, songs, on='song_id', how='outer')
train_merged.head()


# In[ ]:


test_members = pd.merge(test, members, on='msno', how='inner')
test_merged = pd.merge(test_members, songs, on='song_id', how='outer')
test_merged.head()


# In[ ]:


print(len(test_merged.columns))


# In[ ]:


train_merged.columns.to_series().groupby(train_merged.dtypes).groups


# In[ ]:


test_merged.columns.to_series().groupby(test_merged.dtypes).groups


# **Analysing the Missing Values**

# In[ ]:


msno.heatmap(train_merged)


# we can observe that lot of missing values are coming up but we notice that most of the missing values are arrived from members and songs
# 
# missing values from the heatmap also shows information about which are missing and has positive correlation. gender with 4 variables of train.csv and rest of varibales with members.csv

# Now checking missing values and replacing them with some unique values

# In[ ]:


def check_missing_values(df):
    print(df.isnull().values.any())
    if(df.isnull().values.any() == True):
        columns_with_nan = df.columns[df.isnull().any()].tolist()
    print(columns_with_nan)
    
    for col in columns_with_nan:
        print("%s : %d "%(col,df[col].isnull().sum()))

check_missing_values(train_merged)
check_missing_values(test_merged)
    


# In[ ]:


def replace_nan_non_object(df):
    object_cols = list(df.select_dtypes(include=['float']).columns)
    for col in object_cols:
        df[col] = df[col].fillna(np.int(-5))

replace_nan_non_object(train_merged)
replace_nan_non_object(test_merged)


# Memory Consumption

# In[ ]:


#--- memory consumed by train dataframe ---
mem = train_merged.memory_usage(index=True).sum()
print("Memory consumed by training set  :   {} MB" .format(mem/ 1024**2))
 
#--- memory consumed by test dataframe ---
mem = test_merged.memory_usage(index=True).sum()
print("Memory consumed by test set      :   {} MB" .format(mem/ 1024**2))


# Data conversion of int , float and categorical has to be done to reduce the data size for computation as well as storage

# In[ ]:


def change_datatype(df):
    float_cols = list(df.select_dtypes(include=['float']).columns)
    for col in float_cols:
        if((np.max(df[col]) <= 127) and (np.min(df[col] >= -128))):
            df[col] = df[col].astype(np.int8)
        elif((np.max(df[col])<=32767) and (np.min(df[col] >= -32768))):
            df[col] = df[col].astype(np.int16)
        elif((np.max(df[col]) <= 2147483647) and (np.min(df[col] >= -2147483648))):
            df[col] = df[col].astype(np.int32)
        else:
            df[col]= df[col].astype(np.int64)

change_datatype(train_merged)
change_datatype(test_merged)


# In[ ]:


train_merged.info()


# In[ ]:


data = train_merged.groupby('target').aggregate({'msno':'count'}).reset_index()
a4_dims =(15,8)
fig, ax = plt.subplots(figsize = a4_dims)
ax = sns.barplot(x='target', y ='msno', data=data)


# In[ ]:


mpl.rcParams['font.size']= 40.0
plt.figure(figsize=(15,15))
data = train_merged.groupby('source_system_tab').aggregate({'msno': 'count'}).reset_index()
sns.barplot(x ='source_system_tab', y ='msno', data = data)
plt.xticks(rotation='90')


# In[ ]:


data = train_merged.groupby('source_screen_name').aggregate({'msno':'count'}).reset_index()
a4_dims = (15, 7)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(x='source_screen_name', y='msno', data=data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


# In[ ]:


data = train_merged.groupby('source_type').aggregate({'msno':'count'}).reset_index()
a4_dims = (15, 7)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(x='source_type', y='msno', data=data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


# In[ ]:


data = train_merged.groupby('language').aggregate({'msno':'count'}).reset_index()
a4_dims = (15, 7)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(x='language', y='msno', data=data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


# In[ ]:


data = train_merged.groupby('registered_via').aggregate({'msno':'count'}).reset_index()
a4_dims = (15, 7)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(x='registered_via', y='msno', data=data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


# In[ ]:


print(train_merged.columns)
data = train_merged.groupby('city').aggregate({'msno':'count'}).reset_index()
a4_dims = (15, 7)
fig, ax = plt.subplots(figsize=a4_dims)
ax = sns.barplot(x='city', y='msno', data=data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)


# In[ ]:


a4_dims = (15, 7)
fig, ax = plt.subplots(figsize=a4_dims)
ax=sns.countplot(x="source_system_tab",data=train_merged,palette=['lightblue','orange','green'],hue="target")
plt.xlabel("source_screen_tab")
plt.ylabel("count")
plt.title("source_system_tab vs target ")
plt.show()


# new user are coming form discover and my library and old ones are from my library

# In[ ]:


a4_dims = (15, 7)
fig, ax = plt.subplots(figsize=a4_dims)
ax=sns.countplot(x="source_screen_name",data=train_merged,palette=['#A8B820','yellow','#98D8D8'],hue="target")
plt.xlabel("source_screen_name")
plt.ylabel("count")
plt.title("source_screen_name vs target ")
plt.xticks(rotation='90')
plt.show()


# local playlist among new user and old one more most common way to get back their songs

# In[ ]:


a4_dims = (15, 7)
fig, ax = plt.subplots(figsize=a4_dims)
ax=sns.countplot(x="gender",data=train_merged,palette=['#705898','#7038F8','yellow'],hue="target")
plt.xlabel("male female participation")
plt.ylabel("count")
plt.title("male female participation vs target ")
plt.xticks(rotation='90')
plt.legend(loc='upper left')
plt.show()


# new female users are more than male users about 500 to 600

# In[ ]:


a4_dims = (15, 7)
fig, ax = plt.subplots(figsize=a4_dims)
ax=sns.heatmap(data=train_merged.corr(),annot=True,fmt=".2f")


# In[ ]:


a4_dims = (15, 7)
fig, ax = plt.subplots(figsize=a4_dims)
ax=sns.boxplot(x="gender",y="city",data=train_merged,palette=['blue','orange','green'],hue="target")
plt.xlabel("gender")
plt.ylabel("city")
plt.title("city vs registered_via  ")
plt.show()


# here we can see that most of our user are between 5 to 14 no of cities might be female ratio is same

# In[ ]:


ax=sns.lmplot(x="bd",y="registered_via",data=train_merged,palette=['blue','orange','green'],hue="target",fit_reg=False)
plt.xlabel("bd age group")
plt.ylabel("registred_via")
plt.title(" bd age group vs registration_via ")
plt.show()


# now we can see that music users vary in age form 0 to 100. we can see here are outliers to in bd but interesting information is that most users age group of younsters and 30+ age group form 5 to 10 registered_via index

# In[ ]:


ax=sns.lmplot(x="bd",y="city",data=train_merged,palette=['blue','orange','green'],hue="target",fit_reg=False)
plt.xlabel("bd age group")
plt.ylabel("city")
plt.title("bd (age group) vs city ")
plt.show()


# with outlier as we can we didn't remove till now we will remove bd outliers at final stages before applying Ml but that last results insights are telling we have age group 20 to 30+ ages and city index we most 5 to 14

# In[ ]:


a4_dims = (15, 7)
fig, ax = plt.subplots(figsize=a4_dims)
ax=sns.boxplot(x="bd",y="gender",data=train_merged,palette=['blue','orange','green'])
plt.xlabel("bd age group")
plt.ylabel("gender")
plt.title("bd age group vs gender ")
plt.show()


# we can see that mean age group we have 24 to 27 with max is 50 in female case and in male case 48 about age group is max and min in female it is about 16 and in male case 18
# 
# one more observation we can see that female outlier are more there reason behind this logic females always tend fill up the things in hurry way because in male we can't see male with 100 , as if this bit funny logic , apart from this it all due unclean data that's it which we have to remove outliers

# In[ ]:


train_merged.describe()
def remove_outlier(df_in, col_name):

    #q1 = df_in[col_name].quantile(0.25)
    #q3 = df_in[col_name].quantile(0.75)
    #iqr = q3-q1 #Interquartile range
    fence_low  = 12
    fence_high = 45
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out
df_final_train=remove_outlier(train_merged,'bd')


# Preprocessing
# 

# In[ ]:


data_path = '../input/wsdm-kkbox/'
train = pd.read_csv(data_path + 'train.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                  'source_screen_name' : 'category',
                                                  'source_type' : 'category',
                                                  'target' : np.uint8,
                                                  'song_id' : 'category'})
test = pd.read_csv(data_path + 'test.csv', dtype={'msno' : 'category',
                                                'source_system_tab' : 'category',
                                                'source_screen_name' : 'category',
                                                'source_type' : 'category',
                                                'song_id' : 'category'})
songs = pd.read_csv(data_path + 'songs.csv',dtype={'genre_ids': 'category',
                                                  'language' : 'category',
                                                  'artist_name' : 'category',
                                                  'composer' : 'category',
                                                  'lyricist' : 'category',
                                                  'song_id' : 'category'})
members = pd.read_csv(data_path + 'members.csv',dtype={'city' : 'category',
                                                      'bd' : np.uint8,
                                                      'gender' : 'category',
                                                      'registered_via' : 'category'},
                     parse_dates=['registration_init_time','expiration_date'])
songs_extra = pd.read_csv(data_path + 'song_extra_info.csv')
print('Done loading...')


# In[ ]:


song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
train = train.merge(songs[song_cols], on='song_id', how='left')
test = test.merge(songs[song_cols], on='song_id', how='left')

members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))

members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))

# Convert date to number of days
members['membership_days'] = (members['expiration_date'] - members['registration_init_time']).dt.days.astype(int)


# In[ ]:


# categorize membership_days 
members['membership_days'] = members['membership_days']//200
members['membership_days'] = members['membership_days'].astype('category')


# In[ ]:


member_cols = ['msno','city','registered_via', 'registration_year', 'expiration_year', 'membership_days']

train = train.merge(members[member_cols], on='msno', how='left')
test = test.merge(members[member_cols], on='msno', how='left')


# In[ ]:


train.info()


# In[ ]:


def isrc_to_year(isrc):
    if type(isrc) == str:
        if int(isrc[5:7]) > 17:
            return int(isrc[5:7])//5
        else:
            return int(isrc[5:7])//5
    else:
        return np.nan
#categorize song_year per 5years

songs_extra['song_year'] = songs_extra['isrc'].apply(isrc_to_year)
songs_extra.drop(['isrc', 'name'], axis = 1, inplace = True)


# In[ ]:


train = train.merge(songs_extra, on = 'song_id', how = 'left')
test = test.merge(songs_extra, on = 'song_id', how = 'left')


# In[ ]:


train['genre_ids'] = train['genre_ids'].str.split('|').str[0]
temp_song_length = train['song_length']
train.drop('song_length', axis = 1, inplace = True)
test.drop('song_length',axis = 1 , inplace =True)


# In[ ]:


train.head()


# In[ ]:


song_count = train.loc[:,["song_id","target"]]

# measure repeat count by played songs
song_count1 = song_count.groupby(["song_id"],as_index=False).sum().rename(columns={"target":"repeat_count"})

# count play count by songs
song_count2 = song_count.groupby(["song_id"],as_index=False).count().rename(columns = {"target":"play_count"})


# In[ ]:


song_repeat = song_count1.merge(song_count2,how="inner",on="song_id")
song_repeat["repeat_percentage"] = round((song_repeat['repeat_count']*100) / song_repeat['play_count'],1)
song_repeat['repeat_count'] = song_repeat['repeat_count'].astype('int')
song_repeat['repeat_percentage'] = song_repeat['repeat_percentage'].replace(100.0,np.nan)
#cuz most of 100.0 are played=1 repeated=1 values. I think it is not fair compare with other played a lot songs


# In[ ]:


train = train.merge(song_repeat,on="song_id",how="left")
test = test.merge(song_repeat,on="song_id",how="left")


# In[ ]:


# type cast
test['song_id'] = test['song_id'].astype('category')
test['repeat_count'] = test['repeat_count'].fillna(0)
test['repeat_count'] = test['repeat_count'].astype('int')
test['play_count'] = test['play_count'].fillna(0)
test['play_count'] = test['play_count'].astype('int')

