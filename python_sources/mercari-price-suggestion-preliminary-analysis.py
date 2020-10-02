#!/usr/bin/env python
# coding: utf-8

#  Predict the sale price of a listing based on information a user provides for this listing.
#  
#  ### Data Dictionary
# *     **train_id or test_id** :  id of the listing
# *     **name** :  title of the listing
# *     **item_condition_id** :  condition of the items provided by the seller
# *     **category_name** : category of the listing
# *     **brand_name**:
# *     **price** :  This is the target variable that you will predict. The unit is USD. 
# *     **shipping** : 1 if shipping fee is paid by seller and 0 by buyer
# *    **item_description **:  full description of the item. 

# ### 1.  Import Necessary packages

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import re
import seaborn as sns
from subprocess import check_output
from sklearn.preprocessing import LabelEncoder
plt.style.use('ggplot')

pd.options.display.max_columns=999
pd.options.display.max_rows = 999


# ### 2. Check for the input files and load them

# In[ ]:


print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


train_data = pd.read_csv('../input/train.tsv', low_memory='False',sep='\t')
train_data = train_data.set_index('train_id')
train_data.head()


# In[ ]:


test_data = pd.read_csv('../input/test.tsv', low_memory='False',sep='\t')
test_data = test_data.set_index('test_id')
test_data.head()


# ### 3. Quick statistical description about train and test data

# In[ ]:


def data_overview(df):
    print('Data information:')
    df_info = df.info(verbose=False)
    df_describe = df.describe()
    df_missing = df.isnull().sum()[df.isnull().sum()>0]
    print('Data description : ')
    print(np.round(df_describe,2))
    print('Missing Data values:')
    print(df_missing)
#train_data.describe()


# In[ ]:


data_overview(train_data)


# In[ ]:


data_overview(test_data)


# ### Fill in the missing values

# In[ ]:


train_data['brand_name']= train_data['brand_name'].fillna('missing')
test_data['brand_name'] = test_data['brand_name'].fillna('missing')
train_data['item_description'] = train_data['item_description'].fillna('missing')


# ### 4. Statistical description and distribution of target variable.
# 
# **price** is the target variable. 

# In[ ]:


from pandas.plotting import table
fig,ax= plt.subplots(figsize=(12,8))
table(ax, train_data['price'].to_frame().describe(),loc='upper right', colWidths=[0.2, 0.2, 0.2]);
sns.distplot(train_data['price'],hist=False,kde=True);


# The distribution has a long tail because mean is very small compared to the maximum price.  Hence, we plot the distribution with respect to logarithmic value of price. We add 1 to avoid calculating log(0).

# In[ ]:


train_data['log_price']=np.log(train_data['price']+1)


# In[ ]:


from pandas.plotting import table
fig,ax= plt.subplots(figsize=(12,8))
#table(ax, train_data['log_price'].to_frame().describe(),loc='upper right', colWidths=[0.2, 0.2, 0.2]);
sns.distplot(train_data['log_price'],hist=False,kde=True);
ax.set_xlabel('log(price+1)');


# 
# ### 5. Do we observe any distinct outliers in the target variable ?

# In[ ]:


fig,ax = plt.subplots(figsize=(12,8))
train_data['price'].plot();
ax.set_ylabel('price');


# We do not observe any distint outliers in the target variable price. Next, we perform in depth exploration of the data.
# 
# ### 6. In-depth Data Exploration
# We explore each columns and create meaningful description from each of them. 
# 
#    ###  A) category_name

# In[ ]:


train_data.category_name.value_counts().iloc[0:10]


# We observe that the category name is divided into three parts. Also, there are missing values in the category name.  First let us separate these three categories. We name them as first_cat, second_cat and third_cat. We also convert NaN values to NaN/NaN/NaN so that NaN also acts as a category_name.  We do this for train dataset and test dataset.

# In[ ]:


train_data['category_name']=train_data['category_name'].fillna('NaN/NaN/NaN')
test_data['category_name'] = test_data['category_name'].fillna('NaN/NaN/NaN')


# In[ ]:


def create_cats(df):
    ## df - dataframe input: test_data or train_data
    first_cat = []
    second_cat = []
    third_cat = []
    
    for i in range(len(df)):
        cat_names = df.category_name[i].split('/')
        cat1 = cat_names[0]
        cat2 = cat_names[1]
        cat3 = cat_names[2]
        first_cat.append(cat1)
        second_cat.append(cat2)
        third_cat.append(cat3)
    return first_cat, second_cat, third_cat


# In[ ]:


first_cat_train, second_cat_train, third_cat_train = create_cats(train_data)


# In[ ]:


first_cat_test, second_cat_test, third_cat_test = create_cats(test_data)


# In[ ]:


train_data['first_cat']=first_cat_train
train_data['second_cat']=second_cat_train
train_data['third_cat']=third_cat_train


# In[ ]:


test_data['first_cat']=first_cat_test
test_data['second_cat']=second_cat_test
test_data['third_cat']=third_cat_test


# In[ ]:


train_data.head()


# In[ ]:


test_data.head()


# ###  I) How does the target variable vary with category_name ?

# In[ ]:


print("Numer of unique categories = %s" %len(train_data['category_name'].unique()))
fig,ax =plt.subplots()
train_data['category_name'].value_counts()[:30].plot(kind='barh',figsize=(12,8));
ax.set_xlabel('count');


# In[ ]:


fig,ax = plt.subplots()
train_data.groupby(by='category_name').mean()['price'].sort_values(ascending=False).iloc[0:30].plot(kind='barh', figsize=(12,8));
ax.set_xlabel('price');


# ### II) How does the target variable vary with first_cat ?

# In[ ]:


print("Number of first_cat=%s" %len(train_data['first_cat'].unique()))
fig,ax = plt.subplots()
train_data['first_cat'].value_counts().plot(kind='barh',figsize=(12,8));
ax.set_xlabel('count');


# In[ ]:


fig,ax = plt.subplots(figsize=(12,8))
sns.stripplot(x='price',y='first_cat', data=train_data, jitter=True);


# We can clearly observe some outliers in each categories. To be more clear we plot for all categories with price greater than 1000 USD. 

# In[ ]:


fig,ax = plt.subplots(figsize=(12,8))
sns.stripplot(x='price',y='first_cat', data=train_data[train_data.price>=1000], jitter=True);


# ### III) How does the target variable vary with second_cat ?

# In[ ]:


print("Number of second_cat = %s" %len(train_data['second_cat'].unique()))
train_sc_df = train_data.groupby(by='second_cat').filter(lambda x: len(x)>20000)
fig,ax = plt.subplots(figsize=(12,8))
sns.stripplot(x='price',y='second_cat', data=train_sc_df, jitter=True);
ax.set_title('Most common second categories');


# ### IV) How does the target variable vary with third_cat?

# In[ ]:


print("Number of third_cat = %s" %len(train_data['third_cat'].unique()))
train_tc_df = train_data.groupby(by='third_cat').filter(lambda x: len(x)>10000)
fig,ax = plt.subplots(figsize=(12,8))
sns.stripplot(x='price',y='third_cat', data=train_tc_df, jitter=True);
ax.set_title('Most common third categories');


#    ### B) name 
#    Name is one of the most important parameters to determine the price. 

# In[ ]:


train_data['name'].iloc[0:10]


# Name consists of list of words, with bunch of key words included in them. One of the obvious pattern we can explore from name is the length of words and another pattern is the sequence of keywords used. At first, we consider only length of words but later we include all the keywords in the total dataset to obtain some encoding pattern for words. 

# In[ ]:


train_data['name_len']= train_data['name'].apply(lambda x: len(x.split(' ')))
test_data['name_len']= test_data['name'].apply(lambda x: len(x.split(' ')))


# Let us observe how the target variable varies with the length of the name. 

# In[ ]:


print("Number of unique name_len = %s" %len(train_data['name_len'].unique()))
print(train_data['name_len'].value_counts())
sns.factorplot(x='name_len',y='price',data=train_data,size=8,aspect=1.5);


# Relatively shorter names are observed for lower priced items. Also, there are very few counts of items whose name has more than 10 words.  Let us plot only upto where the lenght of names<=10.

# In[ ]:


sns.factorplot(x='name_len',y='price',data=train_data[train_data.name_len<=10],size=8,aspect=1.5);


# ### C) shipping 
# It includes two values: 0 - when price paid by buyer and 1 when price is paid by seller. 

# In[ ]:


train_data.shipping.value_counts()


# In[ ]:


fig,ax = plt.subplots(figsize=(12,8))
sns.distplot(train_data['log_price'][train_data.shipping==0], hist=False,kde=True, label='shipping paid by buyer');
sns.distplot(train_data['log_price'][train_data.shipping==1], hist=False, kde=True,label='shipping paid by seller');
ax.set_xlabel('log(price+1)');


# 1. Shipping is paid by buyers most of the time
# 2. Average price of shipping paid by seller is slightly lower
# 
# But to be very clear let us observe the shipping paid by buyer and seller for different price ranges. 

# In[ ]:


plt.figure(figsize=(12,12))
ax1 = plt.subplot(321)
sns.distplot(train_data['price'][(train_data.shipping==0)& (train_data.price<=20)], hist=False,kde=True, label='shipping paid by buyer');
sns.distplot(train_data['price'][(train_data.shipping==1)& (train_data.price<=20)], hist=False, kde=True,label='shipping paid by seller');
ax2 = plt.subplot(322)
sns.distplot(train_data['price'][(train_data.shipping==0)& (train_data.price>20)& (train_data.price<=100)], hist=False,kde=True, label='shipping paid by buyer');
sns.distplot(train_data['price'][(train_data.shipping==1)& (train_data.price>20)& (train_data.price<=100)], hist=False, kde=True,label='shipping paid by seller');
ax3 = plt.subplot(323)
sns.distplot(train_data['price'][(train_data.shipping==0)& (train_data.price>100)& (train_data.price<=500)], hist=False,kde=True, label='shipping paid by buyer');
sns.distplot(train_data['price'][(train_data.shipping==1)& (train_data.price>100)& (train_data.price<=500)], hist=False, kde=True,label='shipping paid by seller');
ax4 = plt.subplot(324)
sns.distplot(train_data['price'][(train_data.shipping==0)& (train_data.price>500)& (train_data.price<=1500)], hist=False,kde=True, label='shipping paid by buyer');
sns.distplot(train_data['price'][(train_data.shipping==1)& (train_data.price>500)& (train_data.price<=1500)], hist=False, kde=True,label='shipping paid by seller');
ax5 = plt.subplot(325)
sns.distplot(train_data['price'][(train_data.shipping==0)& (train_data.price>1500)& (train_data.price<=2500)], hist=False,kde=True, label='shipping paid by buyer');
sns.distplot(train_data['price'][(train_data.shipping==1)& (train_data.price>1500)& (train_data.price<=2500)], hist=False, kde=True,label='shipping paid by seller');


# This shows that we can't clearly tell  much about the price, only based on shipping information.  Let us look at the shipping and category

# In[ ]:


g =sns.factorplot(x='first_cat',y='price',hue='shipping', data=train_data, size=8, aspect=1.5, kind='bar');
g.set_xticklabels(rotation=75);


# Very interestng obervation is that the average price of items paid by seller under all categories is lower than that paid by buyer.

# In[ ]:


sns.factorplot('shipping',col='first_cat',col_wrap=4, data=train_data, size=8, aspect=.5,sharey=False, sharex=False, kind='count');


# The items in categories like Men, Women, Home, kids, sports etc are paid by buyers most of the time, while  items in categories like Electronics, vintage and collectibles, handmade etc are paid by sellers most of the time. 
# 
# ### D)  item_condition_id
# specifies the condition of each item. 

# In[ ]:


train_data.item_condition_id.value_counts()


# Let us observe the distribution of the target variable for each item_condition_id.

# In[ ]:


fig,ax = plt.subplots(figsize=(12,8))
sns.distplot(train_data['log_price'][(train_data.item_condition_id==1)], hist=False,kde=True, label='1');
sns.distplot(train_data['log_price'][(train_data.item_condition_id==2)], hist=False,kde=True, label='2');
sns.distplot(train_data['log_price'][(train_data.item_condition_id==3)], hist=False,kde=True, label='3');
sns.distplot(train_data['log_price'][(train_data.item_condition_id==4)], hist=False,kde=True, label='4');
sns.distplot(train_data['log_price'][(train_data.item_condition_id==5)], hist=False,kde=True, label='5');
ax.set_xlabel('log(price+1)');


# Could not observe so clear difference on item_condition_id, let us split the price into different categories and observe the distribution.

# In[ ]:


plt.subplots(figsize=(12,12))
ax1=plt.subplot(321)
sns.distplot(train_data['price'][(train_data.item_condition_id==1)& (train_data.price<=20)], hist=False,kde=True, label='1');
sns.distplot(train_data['price'][(train_data.item_condition_id==2)& (train_data.price<=20)], hist=False,kde=True, label='2');
sns.distplot(train_data['price'][(train_data.item_condition_id==3)& (train_data.price<=20)], hist=False,kde=True, label='3');
sns.distplot(train_data['price'][(train_data.item_condition_id==4)& (train_data.price<=20)], hist=False,kde=True, label='4');
sns.distplot(train_data['price'][(train_data.item_condition_id==5)& (train_data.price<=20)], hist=False,kde=True, label='5');

ax2 = plt.subplot(322)
sns.distplot(train_data['price'][(train_data.item_condition_id==1)& (train_data.price>20)& (train_data.price<=100)], hist=False,kde=True, label='1');
sns.distplot(train_data['price'][(train_data.item_condition_id==2)& (train_data.price>20)& (train_data.price<=100)], hist=False,kde=True, label='2');
sns.distplot(train_data['price'][(train_data.item_condition_id==3)& (train_data.price>20)& (train_data.price<=100)], hist=False,kde=True, label='3');
sns.distplot(train_data['price'][(train_data.item_condition_id==4)& (train_data.price>20)& (train_data.price<=100)], hist=False,kde=True, label='4');
sns.distplot(train_data['price'][(train_data.item_condition_id==5)& (train_data.price>20)& (train_data.price<=100)], hist=False,kde=True, label='5');

ax3 = plt.subplot(323)
sns.distplot(train_data['price'][(train_data.item_condition_id==1)& (train_data.price>100)& (train_data.price<=500)], hist=False,kde=True, label='1');
sns.distplot(train_data['price'][(train_data.item_condition_id==2)& (train_data.price>100)& (train_data.price<=500)], hist=False, kde=True,label='2');
sns.distplot(train_data['price'][(train_data.item_condition_id==3)& (train_data.price>100)& (train_data.price<=500)], hist=False,kde=True, label='3');
sns.distplot(train_data['price'][(train_data.item_condition_id==4)& (train_data.price>100)& (train_data.price<=500)], hist=False, kde=True,label='4');
sns.distplot(train_data['price'][(train_data.item_condition_id==5)& (train_data.price>100)& (train_data.price<=500)], hist=False, kde=True,label='5');

ax4 = plt.subplot(324)
sns.distplot(train_data['price'][(train_data.item_condition_id==1)& (train_data.price>500)& (train_data.price<=1500)], hist=False,kde=True, label='1');
sns.distplot(train_data['price'][(train_data.item_condition_id==2)& (train_data.price>500)& (train_data.price<=1500)], hist=False, kde=True,label='2');
sns.distplot(train_data['price'][(train_data.item_condition_id==3)& (train_data.price>500)& (train_data.price<=1500)], hist=False,kde=True, label='3');
sns.distplot(train_data['price'][(train_data.item_condition_id==4)& (train_data.price>500)& (train_data.price<=1500)], hist=False, kde=True,label='4');
sns.distplot(train_data['price'][(train_data.item_condition_id==5)& (train_data.price>500)& (train_data.price<=1500)], hist=False,kde=True, label='5');

ax5 = plt.subplot(325)
sns.distplot(train_data['price'][(train_data.item_condition_id==1)& (train_data.price>1500)& (train_data.price<=2500)], hist=False,kde=True, label='1');
sns.distplot(train_data['price'][(train_data.item_condition_id==2)& (train_data.price>1500)& (train_data.price<=2500)], hist=False, kde=True,label='2');
sns.distplot(train_data['price'][(train_data.item_condition_id==3)& (train_data.price>1500)& (train_data.price<=2500)], hist=False,kde=True, label='3');
sns.distplot(train_data['price'][(train_data.item_condition_id==4)& (train_data.price>1500)& (train_data.price<=2500)], hist=False, kde=True,label='4');
sns.distplot(train_data['price'][(train_data.item_condition_id==5)& (train_data.price>1500)& (train_data.price<=2500)], hist=False,kde=True, label='5');


# Obviuously poor item condition low prices but different categories have different price range. So it is better to obesrve prices with each item condition for first categories as: 
# 
# ### Any relation between item_condition_id and first_category?

# In[ ]:


g =sns.factorplot(x='first_cat',y='price',hue='item_condition_id', data=train_data, size=8, aspect=1.5, kind='bar');
g.set_xticklabels(rotation=75);


# **Interesting observations:**
# * All categories have all item_condition_id
# * Men, women , Home and kids item_condition_id = 1 has higher average price. 
# * In all other categories other item_condition_ids have higher average price.
# * People like to buy expensive items with higher item_condition_id, look at electronics category. 
# 
# 
# ### E)  brand_name
# Whats in the brand name ? Most of the brand names are missing. So, it is logical to divide all items as branded (1) and not branded (0).

# In[ ]:


train_data.groupby(by='brand_name').mean()['price'].sort_values(ascending=False).iloc[0:10]


# In[ ]:


print('Total number of brands used in train_data set= %s' %len(train_data['brand_name'].value_counts()))
print('Total number of brands used in test_data set= %s' %len(test_data['brand_name'].value_counts()))
train_data['branded'] = train_data['brand_name'].apply(lambda x: 0 if x =='missing' else 1)
test_data['branded'] = test_data['brand_name'].apply(lambda x: 0 if x =='missing' else 1)


# Look at the distribution of branded and not branded price values.

# In[ ]:


fig,ax= plt.subplots(figsize=(12,8))
sns.distplot(train_data['log_price'][train_data.branded==0], hist=False, kde=True,label='Not branded');
sns.distplot(train_data['log_price'][train_data.branded==1], hist=False, kde=True, label='Branded');
ax.set_xlabel('log(price+1)');


# Branded have higher average price than the not branded ones as expected.  Let us observe the same for different categories:

# In[ ]:


g =sns.factorplot(x='first_cat',y='price',hue='branded', data=train_data, size=8, aspect=1.5, kind='bar');
g.set_xticklabels(rotation=75);


# In all categories, branded price is higher than not branded, which is expected. 

# ### F) item_description 
# This is the most important part of the analysis with a lot of text data in them. Like in the name, we can observe the length of words used to describe an item and observe the relation between the length of words and price of an item. 

# In[ ]:


train_data['desc_len']=train_data['item_description'].apply(lambda x: len(x.split(' ')))
test_data['desc_len']=test_data['item_description'].apply(lambda x: len(x.split(' ')))


# #### How does the price vary with the length of item_description ? 

# In[ ]:


fig,ax= plt.subplots(figsize=(12,8))
train_data.groupby('desc_len').count()['price'][0:50].plot(kind='bar');
ax.set_ylabel('count');


# Most of the items are described by three words and the number of items and length of words description follow specific pattern.  Also, important is to observe the variation of price with the length of item_description. We can ignore those items with desc_len > 200. 

# In[ ]:


fig,ax= plt.subplots(figsize=(12,8))
train_data.groupby('desc_len').mean()['price'].iloc[0:150].plot();
ax.set_ylabel('average price');


# The average price is rising up and falling down.  This is also not very clear to me why this is happening. So we want to explore more with the categories. 

# In[ ]:


fig,ax= plt.subplots(figsize=(12,12))
train_data_gb_fc_dl = train_data.groupby(['first_cat','desc_len']).mean()['price']
plt.plot(train_data_gb_fc_dl.xs('Beauty')[0:50],'-bo')
plt.plot(train_data_gb_fc_dl.xs('Women')[0:50],'-ro')
plt.plot(train_data_gb_fc_dl.xs('Men')[0:50],'-go')
plt.plot(train_data_gb_fc_dl.xs('Electronics')[0:50],'-ko')
plt.plot(train_data_gb_fc_dl.xs('Kids')[0:50],'-yo')
plt.plot(train_data_gb_fc_dl.xs('Home')[0:50],'-mo')
plt.plot(train_data_gb_fc_dl.xs('Sports & Outdoors')[0:50],'-co')
plt.plot(train_data_gb_fc_dl.xs('Vintage & Collectibles')[0:50],'-k*')
plt.plot(train_data_gb_fc_dl.xs('Handmade')[0:50],'-g*')
#plt.plot(train_data_gb_fc_dl.xs('Other')[0:50],'--y*');
#plt.plot(train_data_gb_fc_dl.xs('NaN')[0:50],'-mo');
plt.legend(['Beauty','Women','Men','Electronics','Kids','Home','Sports & Outdoors','Vintage & Collectibles',
           'Handmade']);
ax.set_xlabel('desc_len')
ax.set_ylabel('average price');


# In[ ]:


desc_len_range = [(0,20),(21,50),(51,100),(101,150),(151,250)]
item_description_len = []
def cont_to_range(range_list,series):
    #range_list: list of range we want to generate
    #series: pandas series whose values are converted as range
    Series_col_range=[]
    for j in range(len(series)):
        for i in range(len(range_list)):
            if series[j] in range(range_list[i][0], range_list[i][1]):
                series_range = range_list[i]
            else:
                pass
        Series_col_range.append(series_range)
    return Series_col_range


# In[ ]:


train_data['item_description_len']=cont_to_range(desc_len_range,train_data.desc_len)
test_data['item_description_len']=cont_to_range(desc_len_range,test_data.desc_len)


# In[ ]:


g =sns.factorplot(x='first_cat',y='price',col='item_description_len',col_wrap=3, data=train_data, size=12, aspect=.5,kind='bar');
g.set_xticklabels(rotation=75);


# Let us look at item description in a more detailed way and try to understand the behavior from each terms and not only the length of the description. At first, let us count the number of significant words in the descriotion. 

# In[ ]:


import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
def count_significant_words(desc):
    try:
        desc =  desc.lower()
        desc_reg = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        desc_txt = desc_reg.sub(" ", desc)

        words = [w for w in desc_txt.split(" ") if not w in stop_words.ENGLISH_STOP_WORDS and len(w)>3]
        return len(words)
    except: 
        return 0


# In[ ]:


train_data['item_desc_word_count'] = train_data['item_description'].apply(lambda x: count_significant_words(x))
test_data['item_desc_word_count'] = test_data['item_description'].apply(lambda x: count_significant_words(x))


# In[ ]:


train_data.head()


# In[ ]:


fig,ax= plt.subplots(figsize=(12,8))
train_data.groupby('item_desc_word_count').mean()['price'].iloc[0:150].plot();
ax.set_ylabel('average price');


# In[ ]:


from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')
def tokenized(desc):
    desc = desc.lower()
    desc_reg = re.compile('[' +re.escape(string.punctuation) + '0-9\\r\\t\\n]')
    desc_txt = desc_reg.sub(" ", desc)
    tokenized_words = word_tokenize(desc_txt) 
    tokens = list(filter(lambda t: t.lower() not in stop, tokenized_words))
    filtered_tokens = [w for w in tokens if re.search('[a-zA-Z]', w)]
    filtered_tokens = set([w.lower() for w in filtered_tokens if len(w)>=3])
    filtered_tokens = list(filtered_tokens)
    return filtered_tokens
    


# In[ ]:


train_data['item_desc_tokenized'] =train_data['item_description'].map(tokenized)
test_data['item_desc_tokenized'] = test_data['item_description'].map(tokenized)


# In[ ]:


for description, tokens in zip(train_data['item_description'].head(),
                              train_data['item_desc_tokenized'].head()):
    print('description:', description)
    print('tokens:', tokens)
    print()


# In[ ]:




