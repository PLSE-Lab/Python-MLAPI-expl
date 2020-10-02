#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# ![Mercari](http://schimiggy.com/wp-content/uploads/2016/05/mercari_logo_horizontal-20160302.jpg)
# 
# **Mercari ** is one of the biggest C2B2C e-commerse platform in Japan that is similar to ebay.
# I first know them because their funny commercial. But now, they are a global company that branches in US and UK.
# 
# This competition will help them to automatically generate a recommendation price to their users, which will be a strong competitive advantage for them!
# 
# Moreover, Mercari profit from charging service fee for each of the transection, which means this automatic price recommendation will be a boost of their revenue!!!

# ## Outline
# 
# * Loading Data and libraries
# * Imputin Data
# * EDA
# * Feature Engineering + Modeling 
# * Cross Validation

# ## Loading Data and Libraries

# **Import tools**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from string import ascii_letters
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.cluster.hierarchy import linkage,dendrogram


# **Import data**

# In[2]:


train_df = pd.read_csv("../input/mercaritest/train.tsv", delimiter='\t')
test_df = pd.read_csv("../input/mercaritest/test.tsv", delimiter='\t')


# In[3]:


train_df.head()


# As I can see, there are not too many features in this dataset. In my opinion, the key to have a good model is having good feature engineering, especially how you deal with text!

# In[4]:


# Checking missing values
print(pd.isnull(train_df).sum())
print("------------")
print(pd.isnull(test_df).sum())


# ## Imputing Data

# In[5]:


# Fill those products with No Brand with NoBrand
train_df["brand_name"] = train_df["brand_name"].fillna("NoBrand")
test_df["brand_name"] = test_df["brand_name"].fillna("NoBrand")


# In[6]:


# Fill those products with no category with No/No/No
train_df["category_name"] = train_df["category_name"].fillna("No/No/No")
test_df["category_name"] = test_df["category_name"].fillna("No/No/No")


# Since there are three category in the category_name column, I'll extract each of them and store in new columns. In my opinion, I think the first one is the main category.

# In[7]:


def split(txt):
    try :
        return txt.split("/")
    except :
        return ("No Label", "No Label", "No Label")


# In[8]:


train_df['general_category']='' 
train_df['subcategory_1'] = '' 
train_df['subcategory_2'] = ''


# In[9]:


# zip to make it work faster and so does lambda
train_df['general_category'],train_df['subcategory_1'],train_df['subcategory_2'] = zip(*train_df['category_name'].apply(lambda x: split(x)))


# In[10]:


train_df.head()


# ## EDA

# ### Statistical overlook

# In[11]:


# force Python to display entire number
pd.set_option('float_format', '{:f}'.format)

train_df.describe()


# Here I can see some basic information, like more than half of the transactions are without shipping, or the mean price is 26 but the median price is 17. The highest price is 2009, which is pretty high and I assume the distribution of price may be pretty skewed, which we'll see later.

# In[12]:


train_df.price.plot.hist(bins=50, figsize=(8,4), edgecolor='white',range=[0,300])
plt.title('Price Distribution')


# As I mentioned, the distribution of price is skewed

# In[13]:


np.log(train_df['price']+1).plot.hist(bins=50, figsize=(8,4), edgecolor='white')
plt.title('Price Distribution (log price +1 )')


# Therefore, I need to transform it while feature engineering

# In[14]:


sns.set(rc={'figure.figsize':(11.7,8.27)})

ax = sns.countplot('general_category',data=train_df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Count of each general category')


# It is obvious that most of the products are for Women. Beauty and Kids occupies second and third place. There isn't too many products with no general category.

# In[15]:


ax = sns.countplot('item_condition_id',data=train_df)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Count of each item condition')


# Rarely products have condition 4 or 5. Most of them are between 1 and 3 condition.

# In[16]:


ax = sns.countplot(x="item_condition_id", hue="general_category", data=train_df, palette="Set3")
ax.set_title('Count of each item condition by general category')


# We can see that most of the products with condition 4 are for women. And most of the Beauty products are in condition 1. I'll try to demonstrate the distribution with a different way.

# In[17]:


def highlight_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: yellow' if v else '' for v in is_max]


# In[18]:


pd.crosstab(train_df.general_category,train_df.item_condition_id).apply(lambda r: r/r.sum(), axis=1).style.apply(highlight_max,axis=1)


# In this way, I can clearly see which condition occupy greatest proportion in each general category! ( yellow one )

# Now, move on to the brand_name part.
# I belive there are tons of unique values in brand_name

# In[19]:


print("There are",train_df['brand_name'].nunique(),"brands in this dataset")


# In[20]:


train_df.brand_name.value_counts()[:10]


# Here we can see some brands that show up frequently in this dataset

# In[21]:


top10_brands = ['NoBrand','PINK', 'Nike',"Victoria's Secret", 'LuLaRoe','Apple','FOREVER 21','Nintendo','Lululemon','Michael Kors']
# Subset those top 10 brands
df = train_df[train_df.brand_name.isin(top10_brands)]


# In[22]:


df.pivot_table(index='brand_name',columns='item_condition_id',aggfunc={'price':'mean'}).style.apply(highlight_max,axis=1)


# Here I see that there is one thing interesting. It's obvious that people love new stuff ( in condition 1 ). But in terms of Apple products, people prefer products in condition 2 and 3, which catch my eyes. Let me check these out.

# ### Deep explore

# In[23]:


Apple = df[df['brand_name'] == 'Apple']
Apple[Apple['item_condition_id'] == 1].head(5)


# In[24]:


Apple[Apple['price'] > 100].head(5)


# Obviously, those Apple products in condition 1 are just cases. And the rest are secondhand iPhone. I believe there are few other brands that have similar phenomenon.

# In[25]:


ax = sns.boxplot(x="general_category", y="item_condition_id", data=train_df)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Condition distribution by general category')

plt.tight_layout()
plt.show()


# I can see that the conditions of most of the products are lower ( including ) than 3. Even in some general category, like Home, Beauty, Other, and Handmade, most of the products are either in condition 1 or 2.

# In[26]:


ax = sns.boxplot(x="general_category", y="price", data=train_df)

ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_yscale('log')
ax.set_title('Price distribution by general category')

plt.tight_layout()
plt.show()


# I see that most of the products are under $1,000. And there is some slight difference between categories

# ### Multivariable exploratory

# In[27]:


ax = sns.countplot(x="general_category", hue="shipping", data=train_df, palette="Set3")
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Count of shipping by general category')


# I see that the products of most of the categories are without shipping. But Electronics, Vantage & Collectibles, Beauty, Handmade have greater counts for products with shipping.
# But it's not hard to understand that customers would like to have these tiny things being well packaged.

# In[28]:


ax = sns.boxplot(x="shipping", y="price", data=train_df)
ax.set_yscale('log')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Price distribution by shipping')


# In[29]:


ax = sns.violinplot(x='general_category', y='price', hue='shipping', data=train_df, palette="Pastel1",legend_out=False)
plt.legend(loc='lower left')
ax.set_yscale('log')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set_title('Price distribution by general category and shipping')


# In my opinio, I think, in each category, there are specific products that shipping significantly effect the price.
# 
# * For men product, shipping or not actually don't effect the distribution of price.
# * But for electronics and Handmade, the distritbutions of price are significantly different!
# 
# As you can imagine, you probably will prefer electronics products that priced within shipping fee. It's the same that people always hope a price within tax instead of being charged for extra money when you checkout! For me, I hate buying a 9.99 dollars  product but paying something like 10.69 dollars.

# ### Mean price of factor variables

# In[30]:


train_df.groupby(['subcategory_1'])["price"].mean().sort_values(ascending=False)[:10]


# Here we can see the mean price of second category in the category column isn't too high. It still includes a variety of different prouducts under it.

# In[31]:


print("There are",train_df['subcategory_1'].nunique(),"subcategory 1 in this dataset")


# In[32]:


train_df.groupby(['subcategory_2'])["price"].mean().sort_values(ascending=False)[:10]


# Here, after splitting, the third category is pretty specific. The mean price is much higher than the second category.

# In[33]:


print("There are",train_df['subcategory_2'].nunique(),"subcategory 2 in this dataset")


# In[34]:


train_df.groupby(['brand_name'])["price"].mean().sort_values(ascending=False)[:10]


# The most specific one must be the brand. As we known, the mean price of a brand definitely varied more than category.

# In[35]:


print("There are",train_df['brand_name'].nunique(),"brands in this dataset")


# However, the amount of unique brands is much more higher than category.

# In[36]:


train_df.head()


# In[37]:


train_df.item_description = train_df.item_description.fillna('Empty')
train_df['log_price'] = np.log(train_df['price']+1)

train_df['des_len'] = train_df.item_description.apply(lambda x : len(x))


# In[38]:


df = train_df.groupby(['des_len'])['log_price'].mean().reset_index()

plt.plot('des_len','log_price', data=df, marker='o', color='mediumvioletred')
plt.show()


# For those products with description shorter than 100 words, the mean price is less than 1,000.
# But what I found is about the standard deviation. We can see that for the products with description shorter than 400, there is a exponential distribution. However, for those with description longer than 400, the price varied.

# In[39]:


train_df['name_len'] = train_df.name.apply(lambda x : len(x))


# In[40]:


df = train_df.groupby(['name_len'])['log_price'].mean().reset_index()

plt.plot('name_len','log_price', data=df, marker='o', color='mediumvioletred')
plt.show()


# Basically, the products with name length between 10 and 40 are linear distributed. The mean price of other length varied. I guess the proudcts with too long name probably is some old stuff.

# **NLP ( tokenizer)**

# In[41]:


from keras.preprocessing.text import Tokenizer


# In[42]:


text = np.hstack([train_df.item_description.str.lower(), 
                      train_df.name.str.lower()])


# In[43]:


tok_raw = Tokenizer()
tok_raw.fit_on_texts(text)
train_df["seq_item_description"] = tok_raw.texts_to_sequences(train_df.item_description.str.lower())
train_df["seq_name"] = tok_raw.texts_to_sequences(train_df.name.str.lower())


# In[44]:


train_df['desc_point'] = train_df.seq_item_description.apply(lambda x : np.linalg.norm(x))
train_df['name_point'] = train_df.seq_name.apply(lambda x : np.linalg.norm(x))


# In[45]:


fig = plt.figure()
ax = plt.gca()
ax.scatter(train_df['desc_point'] ,train_df['price'] , c='blue', alpha=0.05)
ax.set_yscale('log')


# In[46]:


fig = plt.figure()
ax = plt.gca()
ax.scatter(train_df['name_point'] ,train_df['price'] , c='blue', alpha=0.05)
ax.set_yscale('log')


# To be honest, I think there is only slight correlation between the point I calculate and the price. I'll check the correlation later.

# **Cluster the products**

# In[47]:


train_df.head()


# In[48]:


tr = train_df.drop(['train_id','brand_name','category_name','item_description','name','price','shipping'
                    ,'general_category','subcategory_1','subcategory_2','seq_item_description','seq_name'],axis=1)


# In[49]:


model = KMeans(n_clusters = 12)
scaler = StandardScaler()
model.fit(tr)
labels = model.predict(tr)
cluster = make_pipeline(scaler,model)


# In[50]:


train_df['cluster']=labels


# **Get dummies**

# In[51]:


clusters = pd.get_dummies(train_df['cluster'],prefix='Cluster',drop_first=False)

train_test = pd.concat([train_df,clusters],axis=1).drop('cluster',axis=1)


# In[52]:


conditions  = pd.get_dummies(train_df['item_condition_id'],prefix='Condition',drop_first=False)

train_df = pd.concat([train_df,conditions],axis=1)


# In[53]:


train_df.head()


# In[54]:


general_category  = pd.get_dummies(train_df['general_category'],drop_first=True)

train_df = pd.concat([train_test,general_category],axis=1).drop('general_category',axis=1)


# **Correlation cluster heatmap**

# In[55]:


train_df = train_df.drop(['train_id','log_price'],axis=1)


# In[57]:


corr = train_df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True)


# ### Summary
# 
# * There are valued products and cheap little things in each category. ( For example, iPhone and iPhone case show up in same brand and same category )
# * With shipping or not indicates higher price in some sections while it doesn't in the other sections.
# * In some brands or category, valued products usually have condition other than 1
# * There are a lot of subcategory and brands that are hard to deal with
# * The data lack of features that are strong correlated since most of them are slightly negative correlated
# * Name length and name length may be useful for modeling,but not description
# * Basically, the price distribution varied in each category. And once you split them with condition and with shipping fee or not, you can see more pattern in each category!
# 

# In[ ]:




