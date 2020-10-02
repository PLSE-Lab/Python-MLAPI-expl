#!/usr/bin/env python
# coding: utf-8

# I am curious about how to use the information in the column "Description". I got some idea when I loged in the RentHop. As we can see from the website, some location information is posted on it. Such as "tribeca" ,"bronx" and so on. These words are also the key words in the column "Description". So I did some analysis about the location information. This make me some little improvement. But I think there still some important information I didn't find in the column "Description". I hope we can talk about it to make more improvement.

# In[ ]:


#import some packages.
import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer


# **We should do some analysis about how these key words impact the distribution of 'interest_level'.**

# In[ ]:


# Load traing data.
data_path = "../input/"
train_file = data_path + "train.json"
train_df = pd.read_json(train_file).reset_index(drop=True)


# In[ ]:


# Compute the ratio of different interest_level.
num = train_df.shape[0]
low_num = np.where(train_df.interest_level == 'low')[0].shape[0]
medium_num = np.where(train_df.interest_level == 'medium')[0].shape[0]
high_num = np.where(train_df.interest_level == 'high')[0].shape[0]
print('low:',low_num/float(num))
print('medium:',medium_num/float(num))
print('high:',high_num/float(num))


# In[ ]:


# Let's do some pretratment for the column "Description".Code tin this cell was obtained in MosMK's kernal.


train_df['desc'] = train_df['description']
train_df['desc'] = train_df['desc'].apply(lambda x: x.replace('<p><a  website_redacted ', ''))
train_df['desc'] = train_df['desc'].apply(lambda x: x.replace('!<br /><br />', ''))

string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')

remove_punct_map = dict.fromkeys(map(ord, string.punctuation))

train_df['desc'] = train_df['desc'].apply(lambda x: x.translate(remove_punct_map))


# In[ ]:


# I change all the str into lower and drop all the blank.
train_df['desc_str'] = train_df.desc.apply(lambda x : x.lower())
train_df.desc_str = train_df.desc_str.apply(lambda x : x.replace(' ',''))


# In[ ]:


# Define a function to analys the ratio of interest_level.
def analysis_desc(df,str1):
    df['contain_'+str1] = train_df.desc_str.apply(lambda x : 1 if str1 in x else 0)
    df['contain_'+str1] = train_df.desc_str.apply(lambda x : 1 if str1 in x else 0)
    df['contain_'+str1] = train_df.desc_str.apply(lambda x : 1 if str1 in x else 0)
    df['contain_'+str1] = train_df.desc_str.apply(lambda x : 1 if str1 in x else 0)
    num = np.where(df['contain_'+str1] == 1)[0].shape[0]
    print(str1,':',num)
    low = np.where((df['contain_'+str1] == 1)&(df.interest_level == 'low'))[0].shape[0]
    medium = np.where((df['contain_'+str1] == 1)&(df.interest_level == 'medium'))[0].shape[0]
    high = np.where((df['contain_'+str1] == 1)&(df.interest_level == 'high'))[0].shape[0]
    print(low / float(num))
    print(medium / float(num))
    print(high / float(num))


# In[ ]:


# The features I found.

desc_feat = ['batteryparkcity','bowery','chinatown','eastvillage','greenwichvillage','littleitaly',
             'lowereastside','noho','nolita','soho','tribeca','twobridges','westvillage','chelsea',
             'flatirondistrict','garmentdistrict','gramercypark','hellskitchen','kipsbay','midtowneast',
             'murrayhill','nomad','stuyvesant','centralharlem','centralpark','eastharlem','inwood',
             'uppereastside','upperwestside','washingtonheights','westharlem','rooseveltisland',
             'randalls','bedfordstuyvesant','bushwick','greenpoint','williamsburg','boerumhill',
             'carrollgardens','cobblehill','gowanus','parkslope','prospectpark','redhook','sunsetpark',
             'windsorterrace','crownheights','flatbush','kensington','midwood','brooklynheights',
             'clintonhill','dumbo','downtownbrooklyn','fortgreene','prospectheights','bayridge',
             'bensonhurst','astoria','corona','elmhurst','foresthills','glendale','jacksonheights',
             'longislandcity','regopark','ridgewood','sunnyside','woodside','millbasin','flushing',
             'flushingmeadowscoronapark','kewgardens','kewgardenshills','whitestone','briarwood',
             'forestpark','howar','ozonepark','woodhaven','bedfordpark','bronx','concourse','fordham',
             'highbridge','kingsbridge','morris','mount','riverdale','university','morrispark',
             'westchester','thewaterfront','westside','shore']
for place in desc_feat:
    analysis_desc(train_df,place)


# **I listed more than ninety key words. Some of these features can affect the ratio of interest_level. You can choose the key words by yourself. I will give the function to save the selected features.**

# In[ ]:


# Load data.
data_path = "../input/"
train_file = data_path + "train.json"
test_file = data_path + "test.json"
train_df = pd.read_json(train_file)
test_df = pd.read_json(test_file)
train_test = pd.concat((train_df, test_df), axis=0).reset_index(drop=True)


# In[ ]:


# Some pretratment.
train_test['desc'] = train_test['description']
train_test['desc'] = train_test['desc'].apply(lambda x: x.replace('<p><a  website_redacted ', ''))
train_test['desc'] = train_test['desc'].apply(lambda x: x.replace('!<br /><br />', ''))
string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
remove_punct_map = dict.fromkeys(map(ord, string.punctuation))
train_test['desc'] = train_test['desc'].apply(lambda x: x.translate(remove_punct_map))
train_test['desc_str'] = train_test.desc.apply(lambda x:x.lower())
train_test.desc_str = train_test.desc_str.apply(lambda x : x.replace(' ',''))


# In[ ]:


# The function used to save features.
def analysis_desc(df,str1):
    df['contain_'+str1] = df.desc_str.apply(lambda x : 1 if str1 in x else 0)


# In[ ]:


# Add your features.
desc_feat_example = desc_feat


# In[ ]:


# Create a new dataframe to save featues.
desc_fea = pd.DataFrame()
desc_feat_names = []
for feat in desc_feat_example:
    analysis_desc(train_test,feat)
    temp = 'contain_'+feat
    desc_feat_names.append(temp)
    desc_fea = pd.concat((desc_fea,train_test[temp]),axis=1).reset_index(drop=True)


# In[ ]:


# Save features to your own path.
desc_fea.to_csv('desc_fea.csv',index=False)


# You can select the features  that are sute for you. I can't guarantee that these features are food for you. But I think these location information did reflact some rule of interest_level. I hope we can talk more about the column "Description".
