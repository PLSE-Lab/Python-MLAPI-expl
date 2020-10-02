#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This kernel will not present new ML techniques, at least not now, but it will focus on different methods to reduce dataframe memory.
# Since this is my first kernel and moreover, and since english is not my mother tongue, please be indulgent. But any remarks/comments are welcome and appreciated.
# 
# This notebook will described step-by-step the methods i have use to reduce dataframe memory and in particular focusing on this "famous to be so big" user_logs.csv. All methods presented here can be coupled with a "transformation" of the dataframe in an SQL DB or HDF-Storage.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from time import time # code performance benchmark
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.


# First step, we will read the file by chuncks and generate a dataframe from the different chuncks. In the aim to obtain the best performance on your station, it's better to increase chuncksize than increasing the number of chuncks used. Optimal values will depends on your hardware configuration.

# In[ ]:


# variables used in all parts
userfile = '../input/user_logs.csv'
chuncksize = 2*10**6 #a chuncksize of 2M rows as a starting point
chuncknumbers_max = 20 # we will not read all the file, only 20 chuncks, enough for the demonstration


# In[ ]:


chunck_number = 0
user_df = pd.DataFrame()
t = time()
for df in pd.read_csv(userfile, chunksize=chuncksize, iterator=True, header=0):
    user_df = user_df.append(df, ignore_index=True)
    chunck_number += 1
    if chunck_number == chuncknumbers_max :
        break
INITIAL_TIME = int(time()-t)
print('done in '+ str(INITIAL_TIME)+'s')

print('memory usage (MB) : ')
INITIAL_MEM = int(user_df.memory_usage(deep=True).sum()/1024**2)
print(INITIAL_MEM)
print('dataframe details')
print(user_df.info(memory_usage='deep'))


# Before the memory optimisations, there is a better way to create the dataframe, leading, as consequence, to increased performances. DataFrame.append is quite ineffective in fact, as it will copy old DataFrame to a new memory place each time and after the append will be made. 
# As example, if your df uses 9GB of memory and you want to add for 1MB values, pandas will copy your 9GB df to another memory place of size 9.001 GB and delete your old df after. If you are making a lot of append (lot = more thant 2/3) with an appended df smaller than the initial one, this is not the best way.
# Another way is to copy each partial DataFrame obtained at each iteration to a list and at the end, concat all the elements together. This way, for sure you will have to copy df to new memory places, but you will move df of smaller sizes ;) 

# In[ ]:


chunck_number = 0
user_df = None
list_of_df = []
t = time()
for df in pd.read_csv(userfile, chunksize=chuncksize, iterator=True, header=0):
    # this is a list().append function which is called here, not a Dataframe.append function
    list_of_df.append(df)
    chunck_number += 1
    if chunck_number == chuncknumbers_max :
        break
user_df = pd.concat(list_of_df, ignore_index=True)
# we don't need this list anymore so we suppress it (since it has almost the same size as the obtained dataframe )
del list_of_df
current_time = int(time()-t)
print('done in '+ str(current_time)+'s')
print('performance increase : '+ str(int(100*(1-current_time/INITIAL_TIME))) + '%')
print('memory usage (MB) : ')
current_mem = int(user_df.memory_usage(deep=True).sum()/1024**2)
print(current_mem)


# Nice, almost 35% faster this way :) Moreover, compared to user_df.append function, this 
# Finding the best datatypes for the different columns has been detailed in other kernels presented here, as consequence, i will use directly the optimal values to read the csv file.

# In[ ]:


# specify dtype associated with each columns of the csv, string dtype correspond also to object dtype
dtype_cols = {'msno': object, 'date':np.int64, 'num_25': np.int32, 'num_50': np.int32, 
             'num_75': np.int32, 'num_985': np.int32, 'num_100': np.int32, 
              'num_unq': np.int32, 'total_secs': np.float32}
user_df = None
chunck_number = 0
list_of_df = []
t = time()
for df in pd.read_csv(userfile, chunksize=chuncksize, iterator=True, header=0, dtype=dtype_cols):
    list_of_df.append(df)
    chunck_number += 1
    if chunck_number == chuncknumbers_max :
        break
user_df = pd.concat(list_of_df, ignore_index=True)
print('done in '+ str(int(time()-t))+'s')
print('memory usage (MB) : ')
current_mem = int(user_df.memory_usage(deep=True).sum()/1024**2)
print(current_mem)
gain = int(100*(1-current_mem/INITIAL_MEM))
print('gain :' + str(gain) + '%')


# Already a reduction of 16% on final size, let's continue...
# Looking more precisely to each columns :

# In[ ]:


print('memory usage (MB) : ')
user_df.memory_usage(deep=True)/1024**2


# We can remark that date used twice as other columns. But but but... msno uses almost 50% ot total dataframe size. We will as consequence, concentrate on this particular column.  
# 
# ## MSNO column
# 
# Msno's dtype is "object" (object can be seen as string dtype) which means it's not possible to reduce associated size by changing the dtype : string is a string. But there is an alternative. First of all,  we check how many different values are present in this column :

# In[ ]:


print('different msno numbers :')
print(len(user_df.msno.unique()))
print('ratio of unique msno :')
print(str(100*len(user_df.msno.unique())/user_df.shape[0])+'%')


# Only 5% of unique values present in the current dataframe (this ratio will decrease if we use more chuncks), as consequence, 'category' datatype can be efficient in that case. Category dtype is said to be efficient for series which have a lot of repetiting values. So the more data you load, the more this change will become interesting ;) 

# In[ ]:


user_df['msno'] = user_df['msno'].astype('category')
print(user_df.info(memory_usage='deep'))
current_mem = int(user_df.memory_usage(deep=True).sum()/1024**2)
print(current_mem)
gain = int(100*(1-current_mem/INITIAL_MEM))
print('gain :' + str(gain) + '%')


# A case for which category allows a huge memory reduction....**71%** size reduction in that case. You know what i'm happy...or at least, I start to be happy. 
# Let's continue with the date column... Do we really need an int64?
# ## date column
#  Transactions in user_logs started on 2015/01/01, so we can just memorize number of days after this date and change the associated datatype to int16 (down from int64). For that purpose we will create a function to computer number of days between a given date and 2015/01/01
# 

# In[ ]:


from datetime import datetime as dt
STARTDATE = dt(2015, 1, 1)
def intdate_as_days(intdate):
    return (dt.strptime(str(intdate), '%Y%m%d') - STARTDATE).days


# In[ ]:


# remark you need to use pandas > 0.19.1 to be able to use category dtype here 
dtype_cols = {'msno': 'category', 'date':np.int64, 'num_25': np.int32, 'num_50': np.int32, 
             'num_75': np.int32, 'num_985': np.int32, 'num_100': np.int32, 
              'num_unq': np.int32, 'total_secs': np.float32}
user_df = None
chunck_number = 0
list_of_df = []
t = time()
for df in pd.read_csv(userfile, chunksize=chuncksize, iterator=True, header=0, dtype=dtype_cols):
    df['date'] = df['date'].map(lambda x:intdate_as_days(x))
    df['date'] = df['date'].astype(np.int16)
    list_of_df.append(df)
    chunck_number += 1
    if chunck_number == chuncknumbers_max :
        break
user_df = pd.concat(list_of_df, ignore_index=True)
# if you use pandas<0.19, uncomment next line
# user_df['msno'] = user_df['msno'].astype('category')
print('done in '+ str(int(time()-t))+'s')
print('memory usage (MB) : ')
current_mem = int(user_df.memory_usage(deep=True).sum()/1024**2)
print(current_mem)
gain = int(100*(1-current_mem/INITIAL_MEM))
print('gain :' + str(gain) + '%')


# In[ ]:


print(user_df.info(memory_usage='deep'))


# And we can remark another 3% reduction for dataframe memory size, 
# Finally, compared to initial method, memory size has been reduced by **75%** and dataframe creation speed increased by **35%**. It's not so bad.
# It has to be noticed that on that docker image provided by kaggle, using map on dataframe is quite slow, which leads to this huge loss of performances for this step.

# # Application (/extension) of this loader
# Even in that case, you will need a lot or memory (quite more than 10 GB of free ram) to load the full file :( 
# An interesting subsampling can be to use only data corresponding to users in *train.csv* which can de done this way :

# In[ ]:


dtype_cols = {'msno': object, 'date':np.int64, 'num_25': np.int32, 'num_50': np.int32, 
             'num_75': np.int32, 'num_985': np.int32, 'num_100': np.int32, 
              'num_unq': np.int32, 'total_secs': np.float32}
user_df = None

# loading train.csv into another dataframe
train_df = pd.read_csv('../input/train.csv', dtype={'msno': object, 'is_churn': np.int8})

# we compute only unique values of msno, just in case....
cols_msno = train_df['msno'].unique()

chunck_number = 0
list_of_df = []
t = time()
for df in pd.read_csv(userfile, chunksize=chuncksize, iterator=True, header=0, dtype=dtype_cols):
    # addition to previous script, we will look only to dataframe's msno which are present in train_df
    # only save msno which are already in train_df, 
    append_cond = df['msno'].isin(cols_msno)
    df = df[append_cond]
    
    # as previously...
    df['date'] = df['date'].map(lambda x:intdate_as_days(x))
    df['date'] = df['date'].astype(np.int16)    
    list_of_df.append(df)
    chunck_number += 1
    if chunck_number == chuncknumbers_max :
        break
user_df = pd.concat(list_of_df, ignore_index=True)
user_df['msno'] = user_df['msno'].astype('category')
print('done in '+ str(int(time()-t))+'s')
current_mem = int(user_df.memory_usage(deep=True).sum()/1024**2)
print('memory usage (MB) : ' + str(current_mem))


# In the case presented here, new file is less than 1GB which is a good point. But we can remark that we have another dataframe linked to train.csv : **train_df**. Both df, train_df and user_df, share the same msno possible values. Let's try something on it...
# 
# *Remark *: For sure, it's possible to make a merge of both dataframes but for this presentation, i will not do it ;-)
# You will see why in next part
# 
# ## Focus on train_df and user_df

# In[ ]:


train_df = pd.read_csv('../input/train.csv', dtype={'msno': object, 'is_churn': np.int8})


# In[ ]:


print('Memory associated with train_df (MB): ')
TRAIN_INIT_MEM = int(train_df.memory_usage(deep=True).sum()/1024**2)
print(TRAIN_INIT_MEM)


# Previously I said that size of msno can be reduced through category datatype, let's do the trick again

# In[ ]:


train_df['msno'] = train_df['msno'].astype('category')
print('Memory associated with train_df (MB): ')
print(int(train_df.memory_usage(deep=True).sum()/1024**2))


# Hey wait...you said (and shown)  that category help to reduce memory but in that case it has increased, not as interesting. This can be explained if we compute the ratio of unique values in that dataframe.

# In[ ]:


print('different msno numbers in train :')
print(len(train_df.msno.unique()))
print('ratio of unique msno in train:')
print(str(100*len(train_df.msno.unique())/train_df.shape[0])+'%')


# All msno are unique in train_df that's why category are not efficient at all. 
# Even in that case we can apply the principle of category, making an alternative by ourselves using a dictionnary and an hexadecimal representation of the train_df.index, or just on the row line number
# 

# In[ ]:


# generate the hash dict
hashkey = {}
index = 0
msno_list = train_df['msno'].values
for msno_idx in range(0, len(msno_list)):
    msno = msno_list[msno_idx]
    hashkey.update({msno : '{:09x}'.format(msno_idx)})
# this dict can be saved to a csv file to use it after...
csv_key_file = 'hashkey.csv'
with open(csv_key_file, 'w') as f:
    f.write('msno,hexid\n')
    for k,v in hashkey.items():
        f.write('{0},{1}\n'.format(k,v))
        
# if you want to get  back msno from dict, generate the 'inverse' dict this way
hashkey_reverse = {}
for k,v in hashkey.items(): hashkey_reverse.update({v:k})

# apply this hash to train_df
train_df['msno'] = train_df['msno'].map(lambda x:hashkey.get(x,x))
train_df['msno'] = train_df['msno'].astype('str')
print('Memory associated with train_df (MB): ')
current_mem = int(train_df.memory_usage(deep=True).sum()/1024**2)
print(current_mem)
print('Reduction of (%)')
print(100*(1-current_mem/TRAIN_INIT_MEM))


# 35% gain, even for this small DataFrame. The reason is, event if associated type is still string/object, the length of each element is now only 9 characters. As a consequence, DataFrame use less memory (msno uses 40+ characters). The previous remark concerning category still apply here for train_df : using category is still inefficent.
# Since this hash seems promising, we can apply it to the main dataframe, user_df. 

# In[ ]:


user_df['msno'] = user_df['msno'].map(lambda x:hashkey.get(x,x))
user_df['msno'] = user_df['msno'].astype('category')
#user_df['msno'] = user_df['msno'].astype('category')
print('Memory associated with final version of user_df (MB): ')
current_mem = int(user_df.memory_usage(deep=True).sum()/1024**2)
print('Reduction of (%)')
print(100*(1-current_mem/INITIAL_MEM))


# ## Conclusion
# Mixing usage of adapted pandas function, with choice of good datatype, with or without transformations before/after,  permits to reduce memory size by a factor of **85%** here.
# In conclusion, where you have columns with (long) strings don't hesitate to use hash methods, coupled with category.
# 
# I hope this kernel will help you in current (and others) competitions since methods presented here are quite generic. And, as said in the introduction, feel free to use it, share it and ask any questions. I will (try to) answers the best I can.

# In[ ]:




