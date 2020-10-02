#!/usr/bin/env python
# coding: utf-8

# # 1 - Import data

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


raw_data = pd.read_csv("../input/online_sex_work.csv")

print('Initial shape:')
print(raw_data.shape)

raw_data=raw_data.dropna(axis=0, how='any').reset_index(drop=True)
print('\n Shape neglecting NaN:')
print(raw_data.shape)

raw_data.head()


# Asumming __User_ID__, __Verification__, __Looking_for__,  __Location__ and __Friends_ID_list*__ can be neglected:
# 
# _*actually this might be a very pertinent feature, let's just ignore it for now._

# In[3]:


raw_data = raw_data.drop(['User_ID', 'Verification', 'Looking_for', 'Location', 'Friends_ID_list'], axis=1)
raw_data.head()


# # 2 - Data Manipulation
# 
# Let us consider __Gender__, __Sexual_orientation__, __Looking_for__ and __Risk__ as cathegorical variables. The rest will be considered as numerical values, which will force us to manipulate some of the columns, in particular __Last_login__, __Member_since__ and __Time_spent_chating_H:M__.
# 
# For the case of __Last_login__ let's just take the numeric value enclosed in the string.
# 
# The __Member_since__ will be a bit more problematic. To facilitate, let's assume our day 0 is the day the older user registered and count days starting from that, i.e. we'll assume our reference is the older user.
# 
# In the __Time_spent_chating_H:M__ case, let's just convert everything to minutes (or hours, deppending on the data)

# ## 2.1 - Last_login

# In[4]:


import re

m=len(raw_data.Last_login)

rep=np.zeros(m)

for i in range(m):
    s = raw_data.Last_login[i]
    r = re.split('_+',s)
    rep[i] = int(r[1])
    
raw_data['Last_login']=pd.DataFrame(rep)


# In[ ]:


raw_data.head()


# ## 2.2 - Member_since

# In[5]:


m = len(raw_data.Member_since)

# Create 3 vectors: day, month, year
day = np.zeros(m)
month = day
year = day

for i in range (m):
    s = raw_data.Member_since[i]
    r = re.split('\.', s)
    day[i] = r[0]
    month[i] = r[1]
    year[i] = r[2]


# We got an error even though the code seems ok. Let's further inspect by printing the index, and its corresponding row, to see in which iteration are we getting an error and why.

# In[6]:


i


# In[7]:


raw_data[i:i+2]


# We can not but eliminate this row and see if the problem is solved.

# In[8]:


raw_data = raw_data.drop(raw_data.index[[i]]).reset_index(drop=True)
raw_data[773:775]


# Repeating the code:

# In[9]:


m = len(raw_data.Member_since)

# Create 3 vectors: day, month, year
day = np.zeros(m)
month = day
year = day

for i in range (m):
    s = raw_data.Member_since[i]
    r = re.split('\.', s)
    day[i] = r[0]
    month[i] = r[1]
    year[i] = r[2]


# In[10]:


date=np.stack((np.transpose(day),np.transpose(month),np.transpose(year)), axis=1)
date=pd.DataFrame(date)
date=date.astype(int)
date.columns = ['Day', 'Month', 'Year']
date.head()


# In[11]:


date.loc[date['Year'].idxmin()]


# In[12]:


date.loc[date['Year'].idxmax()]


# So the older and newest subjects have joined at 1/11/2009 and 5/1/2017. Knowing this we are able to normalize this feature, buy just subtracting 1/11/2009 to every sample and transform it into days. In other words, our zero year will be 2009.

# In[13]:


raw_data['Member_since'] = raw_data['Member_since'].str.replace('.', '-')

from datetime import datetime
raw_data['Member_since'] = pd.to_datetime(raw_data.Member_since, dayfirst=True)

# Subtracting 1/11/2009
raw_data['Member_since'] = raw_data['Member_since'] - pd.to_datetime('1/11/2009', dayfirst=True)
raw_data['Member_since'] = raw_data['Member_since'].astype('timedelta64[D]')
raw_data['Member_since'] = raw_data['Member_since'].astype(int) # Making it integer values

raw_data.head()


# ## 2.3 - Time_spent_chating_H:M

# Let us first rename the column __Time_spent_chating_H:M__ to __Time_spent_chating__ for the ':' could cause problems.

# In[14]:


raw_data.rename(columns={'Time_spent_chating_H:M': 'Time_spent_chating'}, inplace=True)
raw_data.head()


# In[15]:


m = len(raw_data.Time_spent_chating)

# Create 3 vectors: day, month, year
hours = np.zeros(m)
minutes = np.zeros(m)

for i in range (m):
    split = re.split('\:',raw_data.Time_spent_chating[i])
    hours[i] = int(split[0].replace(" ", ""))
    minutes[i] = int(split[1].replace(" ", ""))

t = hours+minutes/60


# In[16]:


time = pd.DataFrame(t)
raw_data['Time_spent_chating'] = time


# In[17]:


raw_data.head()


# ## 2.4 - Risk

# In[18]:


raw_data.Risk.astype(str).unique()


# In[19]:


mapping = {'No_risk': 0, 'High_risk': 1, 'unknown_risk': 2}
raw_data = raw_data.replace({'Risk': mapping})
raw_data.Risk.astype(int).unique()


# ## 2.5 - str to float

# In[20]:


raw_data.columns


# In[21]:


raw_data['Age'] = raw_data['Age'].str.replace(',', '.').astype(float)


# In[22]:


raw_data['Points_Rank'] = raw_data['Points_Rank'].astype(float)


# In[23]:


raw_data = raw_data[raw_data.Points_Rank.str.contains("a") == False].reset_index(drop=True)
raw_data['Points_Rank'] = raw_data['Points_Rank'].str.replace(" ", "").astype(float)


# In[24]:


raw_data['Number_of_Comments_in_public_forum'] = raw_data['Number_of_Comments_in_public_forum'].str.replace(' ', '').astype(float)


# In[25]:


raw_data['Number_of_advertisments_posted'] = raw_data['Number_of_advertisments_posted'].astype(float)


# In[27]:


raw_data['Number_of_offline_meetings_attended'] = raw_data['Number_of_offline_meetings_attended'].astype(float)


# In[28]:


raw_data['Profile_pictures'] = raw_data['Profile_pictures'].astype(float)


# In[29]:


raw_data.isnull().values.any()


# # 3 - Tensorflow Classifier

# ## 3.1 - Normalization

# In[30]:


cols2norm = ['Age', 'Points_Rank', 'Last_login', 'Member_since', 'Number_of_Comments_in_public_forum',
       'Time_spent_chating', 'Number_of_advertisments_posted', 'Number_of_offline_meetings_attended', 'Profile_pictures']


# In[31]:


raw_data[cols2norm] = raw_data[cols2norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


# In[32]:


raw_data.head()


# ## 3.2 - Tensorflow

# In[33]:


import tensorflow as tf


# To keep things simple, let us use the Estimator API, which force us to create feature columns -- we have to specify if a particular column is numeric or categorical.

# In[34]:


gender = tf.feature_column.categorical_column_with_hash_bucket('Gender', hash_bucket_size=10)
age = tf.feature_column.numeric_column('Age')
sex_or = tf.feature_column.categorical_column_with_hash_bucket('Sexual_orientation', hash_bucket_size=10)
rank = tf.feature_column.numeric_column('Points_Rank')
last_log = tf.feature_column.numeric_column('Last_login')
memb_since = tf.feature_column.numeric_column('Member_since')
comments = tf.feature_column.numeric_column('Number_of_Comments_in_public_forum')
time_spent = tf.feature_column.numeric_column('Time_spent_chating')
advertisements = tf.feature_column.numeric_column('Number_of_advertisments_posted')
meetings = tf.feature_column.numeric_column('Number_of_offline_meetings_attended')
pictures = tf.feature_column.numeric_column('Profile_pictures')
risk = tf.feature_column.categorical_column_with_hash_bucket('Risk', hash_bucket_size=5)


# In[35]:


feat_cols = [gender, age, sex_or, rank, last_log, memb_since, comments, time_spent, advertisements, meetings, pictures]


# ## 3.3 - Train-test split

# In[36]:


features = raw_data.drop('Risk', axis=1)

labels = raw_data['Risk']


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=101)


# In[39]:


input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train,
                                                 batch_size=10, num_epochs=1000, shuffle=True)


# In[40]:


model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=3)


# In[41]:


model.train(input_fn=input_func, steps=1000)


# In[42]:


eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)


# In[43]:


results = model.evaluate(eval_input_func)


# In[44]:


results


# I must say that by no means this code is robust and optimized. As the title suggests, this was just a quick draft. And even though it produced good results, one must be cautious to assume this to be a good model. Most probably this is the result of overfitting.
