#!/usr/bin/env python
# coding: utf-8

# # Data Preparation and Analysis
# * Well, This is forked from the previous notebook because Im lazy doing and typing the importing formalities
# * I dont know what to do but i found something interesting while working on Excel
# * Might also try to see something in Phyton

# In[ ]:


#Copied
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #Cuz why not
import seaborn as sns #Not using it now

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


#Lets just see the November data
df = pd.read_csv("/kaggle/input/ecommerce-events-history-in-cosmetics-shop/2019-Nov.csv")

#Formalities time conversion
df['event_time'] = pd.to_datetime(df['event_time'],infer_datetime_format=True)

##Encode? Lets not do that now. categorical/label encoding of the session IDs (instead of string - save memory/file size):
#Actually its new to me to know that it reduces the size so thanks
#df['user_session'] = df['user_session'].astype('category').cat.codes

#Check how many row and column
print(df.shape)
#Formalities see the topmost data
df.head()


# In[ ]:


df.info()


# ### About the data types  
# We see there is 4635837 row of data and 9 columns
# <br> event_time is datetime (of course)
# <br> event_type is object, its string so it must be categoric
# <br> product_id is integer, but i know it should be categoric
# <br> actually most of the data is categoric, the only numeric is price 
# <br> and even it doesnt mean a thing, like, i see that (in Excel) its same for the same product meant its the retail price whereas the quantity bought doesnt exist in the data
# 

# In[ ]:


#Oh so this is how you count cat value occurence. That's neat
ev_count = df["event_type"].value_counts()
#Might as well draw a pie chart
#func to get percent string to show to  
def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d} entry)".format(pct, absolute)
ax = ev_count.plot.pie(figsize=(6,6),radius=2,autopct=lambda pct: func(pct, ev_count),textprops=dict(color="w"),legend=True)
ax.legend(loc=2)


# ### Want to know event_type
# There are only 4 types of event
# * View, yeah just stroll through the product page and buy nothing, most of us do that, especially during work time
# * Cart, add to cart, cuz why not, not gonna buy it though
# * Remove_from_cart, see what i say, some of us just want to put it in and put them out all again
# * Purchase, good job, you are model netizens now just pour in more money
# 
# Intuitively when seeing these 4 types of event it leads us to think that a session will consists of:
# * Start >> VIEW >> CART >> REMOVE/PURCHASE
# 
# However, thats not the case here
# <br> Anyway we can see that only 7% of the entry is actual purchase
# <br> Removing and Purchasing combined into 27% of the data
# <br> And adding to cart consist of 28.3% data. Huh
# <br> Huh, whats going on? the number doesnt add up.

# In[ ]:


#Might do the same to product_id and others
prid_s = df["product_id"].value_counts()
#Of course not, the product_id is long you'd waste space printing it 
print('no. of products: ',len(prid_s))
#Use describe instead
print(prid_s.describe())
#So what we count here is how active ea product is, it doesnt matter what the activity though might be only viewing
#carting purchasing and even removing. But we get the big picture here that how active each product was on Nov 2019
#So we gonna step up to see which is most popular
prid_s = prid_s.sort_values(ascending=False) 
print('Top 10 Active Products')
print(prid_s.head(10))
#Sadly we dont know what the real product which lies on top of the list
#Lets step up to see the product access count distribution (with histogram of course)
prid_s.plot.hist(bins=100)
#That doesnt look good right,try this
#prid_s.plot.hist(bins=[0,10,20,50,100,200,500,1000,2000,5000,10000,20000],logx=True,logy=True)


# ### Want to know the product_id
# What we can see in general is how many entry per product, in this case we saw the popularity of ea product_id in term of access, so it doesnt linearly mean its more successful. On average (median), we might see a product be accessed 34 times a month, meanwhile the numerical mean is about 100 so the distribution is heavily unbalanced toward the lower end. That means a ton of unclicked things in the net (and here i thought online business is the way to get easy money, well thats not the case here). <br>
# We can see clear imbalances in the top 10 as the second place is about the third of the first place. Wow, million people didnt realized this. Thats the edge of capitalism for us, where the best just getting better and the worst gets even worse. So to warp it up, i presented the graph on log scale. 

# In[ ]:


category_dict=df[['category_id','product_id']].drop_duplicates()
print('no. of category',len(df['category_id'].drop_duplicates()))
cat_counts=category_dict['category_id'].value_counts().sort_values(ascending=False)
print('Top 10 Category')
print(cat_counts.head(10))
print('Last 10 Category')
print(cat_counts.tail(10))
cat_counts.plot.bar(logy=True)


# ### Want to know - categories
# There are 491 category. Wow thats a lot (even this is only in cosmetics). As we can see, the category isnt evenly distributed, I plotted the number of product in each category. The top encloses thousands of product however the last may seem like exclusive product categories <br>
# 
# What else we might uncover from category?
# <br>Oh lets try to scatter plot the activeness of each category to the number of product

# In[ ]:


print(type(cat_counts)) #Check the type of cat_count oh no its a series
cat_act_c=df['category_id'].value_counts()
print(cat_act_c.head(10))
print(cat_counts.head(10))
print(len(cat_act_c.index))
print(len(cat_counts.index))
print(cat_act_c.index[:10])
print(cat_counts.index[:10])
#we can see the number of cat is same however it isnt aligned
#if we join
df_act_cat=pd.DataFrame(cat_act_c)
df_act_cat.rename(columns={'category_id':'activeness'}, inplace=True)
df_cnt_cat=pd.DataFrame(cat_counts)
df_cnt_cat.rename(columns={'category_id':'n_product'}, inplace=True)
print(df_act_cat.head(10))
print(df_cnt_cat.head(10))
joined = df_act_cat.join(df_cnt_cat)
joined.head(10)
# Oh it actually joined well because the index is already the category number
joined.plot.scatter(x='n_product',y='activeness',figsize=(16,8))


# The figure represents activity in opposition to product number. Hypothetically, it would follow a straight line as more popular a product category tend to attract more business doer. And so, it increases the chance of getting exposure on the category. We observed a bound on activity per product and you may draw a straight line that approximate a linear model of the relation

# In[ ]:


print(df['category_code'].value_counts())
print('allRows =',len(df.index))
print('nulls =',len(df.index)-sum(df['category_code'].value_counts()))
##df.drop(["event_time"],axis=1).nunique()


# As we see, most of the code is null, i dont know whats the importance of this column however the category_id should just be sufficient to hold the information. 

# In[ ]:


print(df['brand'].value_counts())
print('allRows =',len(df.index))
print('nulls =',len(df.index)-sum(df['brand'].value_counts()))
print(df['brand'].value_counts().head(20))
print('brand accessed ', sum(df['brand'].value_counts())/len(df.index)*100)


# Although null value is still abundant the brand is a different story altogether. Brand might encourage activity through itself. Look at the number on the top brand, it rivals the activity of the top product. And (after several googling) i found out that 4 of the top 5 is nail polish product (I couldnt find irisk).  <br>
# 
# Oh yeah does the price affect the popularity of a product?
# First lets see the popularity of each price level

# In[ ]:


c_p_act = df['price'].value_counts()
c_p_act.sort_index(inplace=True)
print(c_p_act.head(10))
c_p_act.plot.line()# Huh there are negative price??? And zero prices oh come on wth


# In[ ]:


#Looks like a detective job here
print(df.loc[df['price']<0][['product_id','price']].drop_duplicates())


# In[ ]:


#So we're down to five lets see the transaction of each product
print(df.loc[df['product_id']==5716855][['event_time','event_type','product_id','price','user_id']])
#Straight purchase huh
print(df.loc[df['product_id']==5716859][['event_time','event_type','product_id','price','user_id']])
print(df.loc[df['product_id']==5716857][['event_time','event_type','product_id','price','user_id']])
print(df.loc[df['product_id']==5716861][['event_time','event_type','product_id','price','user_id']])
print(df.loc[df['product_id']==5670257][['event_time','event_type','product_id','price','user_id']])


# I dont know, it doesnt seem like pricing error because no price changed in a period of a month. All the event type is direct purchase (so you can buy without adding to cart huh). Although the price should affect consumption behavior, it can be only seen a low number of activity on this errorneous pricing

# In[ ]:


print(len(df[['product_id','price']].drop_duplicates()))
print(df[['product_id','price']].drop_duplicates().sort_values(by='product_id').head(20))
print(df.drop(["event_time"],axis=1).nunique())
#oh there may be a change of price, so, does the aforementioned thing is pure error?
#Lets see the most changing product
price_list=df[['product_id','price']].drop_duplicates()
print(price_list['product_id'].value_counts().head(10))


# In[ ]:


#lets see how product 5900886 
print(df.loc[df['product_id']==5900886][['event_time','event_type','product_id','price','user_id']].iloc[100:150])


# In[ ]:


def day_far(series):
    time_span=series.max()-series.min()
    if time_span.days==0: 
        return 1
    else:
        return time_span.days
pvt_pp=pd.pivot_table(df,values=['event_time','event_type'],index=['product_id','price'],aggfunc={'event_time':day_far,'event_type':len})


# In[ ]:


pvt_pp['act_p_day']=pvt_pp['event_type']/pvt_pp['event_time']
print(pvt_pp.head(10))
pvt_pp.reset_index().plot.scatter(x='price',y='act_p_day',figsize=(16,8))


# it seemed like an exponential pattern. However, it look more arbitrary than having any pattern. It should be more meaningful if we see it on each product

# In[ ]:


print(pvt_pp.head(10))
pvt_pp.loc[5900886].reset_index().plot.line(x='price',y='act_p_day',figsize=(16,8))
pvt_pp.loc[5906079].reset_index().plot.line(x='price',y='act_p_day',figsize=(16,8))
pvt_pp.loc[5816649].reset_index().plot.line(x='price',y='act_p_day',figsize=(16,8))
pvt_pp.loc[5788139].reset_index().plot.line(x='price',y='act_p_day',figsize=(16,8))
pvt_pp.loc[5900579].reset_index().plot.line(x='price',y='act_p_day',figsize=(16,8))
pvt_pp.loc[5901864].reset_index().plot.line(x='price',y='act_p_day',figsize=(16,8))
pvt_pp.loc[5900883].reset_index().plot.line(x='price',y='act_p_day',figsize=(16,8))


# ### Want to know sessions
# Session is a bit confusing lets see sessions done by a user
# 1. Look for a user that has most activity

# In[ ]:


df['user_id'].value_counts()


# In[ ]:


#oh 527021202 is the most active user lets see whats done
#df.loc[(df['user_id']==527021202)&(df['event_time']>pd.to_datetime('2019-11-06'))&(df['event_time']<pd.to_datetime('2019-11-08'))].tail(60)
df.loc[(df['user_id']==527021202)&(df['event_type']=='purchase')].tail(60)
#Oh no the most active user doesnt purchase anything 


# Crosstab is a powerful tool/function, with this only we can do most basic analytics. I found the thing from AnalyticsVidhya you might check it out as well [Link](https://www.analyticsvidhya.com/blog/2016/01/12-pandas-techniques-python-data-manipulation/)

# In[ ]:


cross_usev=pd.crosstab(df['user_id'],df['event_type'])
print(cross_usev.sort_values(by='purchase',ascending=False).head(20))


# In[ ]:


#so 557790271 is seemingly the most information rich user, it also has a good proportion of each event type
#lets see when they did purchase 
df.loc[(df['user_id']==557790271)&(df['event_type']=='purchase')]
#big purchase list lets see whats going on 13 Nov
data_snippet = df.loc[(df['user_id']==557790271)&(df['event_time']>pd.to_datetime('2019-11-12'))&(df['event_time']<pd.to_datetime('2019-11-14'))]
#that what we typically want to see a session(multiple sessions actually) started with viewing and ended with purchase 
#im curious of how it went for each product
data_snippet.sort_values(by=['product_id','event_time']).head(31)


# Here we can see a typical view>>cart>>purchase on product 5304 
# <br>however mostly view process is skipped
# <br>we also see typical add remove sequences
# <br>lets see the rest of the data

# In[ ]:


data_snippet.sort_values(by=['product_id','event_time']).tail(31)


# Now that we have used crosstab on event_type we might as well see the effect of action to purchasing number on each consumer 

# In[ ]:


fig, ax = plt.subplots()
cross_usev.plot(kind='scatter',x='view',y='cart',c='purchase',s=8,colormap='PiYG', figsize=(16,8), xlim=(0,4000), ylim=(0,1000), ax=ax)


# In[ ]:


#purchasers
purchaser=cross_usev.loc[cross_usev['purchase']>0]
print(purchaser.describe())
purchaser.plot(kind='scatter',x='view',y='cart',c='purchase',s=8,colormap='PiYG', figsize=(16,8), xlim=(0,4000), ylim=(0,1000))


# In[ ]:


big_shot=cross_usev.loc[cross_usev['purchase']>100]
big_shot.plot(kind='scatter',x='view',y='cart',c='purchase',s=20,colormap='PiYG', figsize=(16,8), xlim=(0,4000), ylim=(0,1000))


# In[ ]:


typical_purchaser=cross_usev.loc[(cross_usev['purchase']<20)&(cross_usev['purchase']>0)]
typical_purchaser.plot(kind='scatter',x='view',y='cart',c='purchase',s=20,colormap='PiYG', figsize=(16,8), xlim=(0,4000), ylim=(0,1000))


# In[ ]:


non_purchaser=cross_usev.loc[cross_usev['purchase']==0]
non_purchaser.plot(kind='scatter',x='view',y='cart',c='black',s=20, figsize=(16,8), xlim=(0,4000), ylim=(0,1000))


# No pattern can be seen in plotting cart to view that correlates with purchase.
# <br>
# Moreover, there is no distinguishable pattern between no-purchaser to your typical purchaser. They just putting into cart and viewing the same amount
# <br>typically more view lead to more purchase alternatively more carting also lead to more purchase. Lets incorporate the remove from cart information

# In[ ]:


cross_usev['delta_cart']=cross_usev['cart']-cross_usev['remove_from_cart']
cross_usev.plot(kind='scatter',x='view',y='delta_cart',c='purchase',s=8,colormap='PiYG', figsize=(16,8), xlim=(0,4000), ylim=(-600,600))
typical_purchaser=cross_usev.loc[(cross_usev['purchase']<20)&(cross_usev['purchase']>0)]
typical_purchaser.plot(kind='scatter',x='view',y='delta_cart',c='purchase',s=20,colormap='PiYG', figsize=(16,8), xlim=(0,4000), ylim=(-600,600))
no_purchaser=cross_usev.loc[(cross_usev['purchase']==0)]
no_purchaser.plot(kind='scatter',x='view',y='delta_cart',c='black',s=20, figsize=(16,8), xlim=(0,4000), ylim=(-600,600))


# In[ ]:


print(df.drop(["event_time"],axis=1).nunique())
print(df[['product_id','price','brand']].loc[df['product_id']==5809910].drop_duplicates())


# In[ ]:


len(cross_usev.loc[cross_usev['purchase']>0].index)


# ### Whats next
# We have plotted several graphs for each column data. We got the basic information on each column. What I can remark from the data is as follows:
# * There are 4 type of event with purchase is only 7% of all the entry.
# * 43419 kind of products in 491 category, on average accessed 34 times a month. 
# * The most popular product is branded nail polish product.
# * Brand have tendency to encourage activity, 57% data is branded. Is it really the case?
# * Price can change, price affect activity, price can be negative?? 
# * 368232 Users only 31524 do purchase, even the most active doesnt purchase.
# * Session may change on access it should have no meaning.
# * Customer event type doesnt correlate to each other. Activity intent may be purely externally driven *
# <br><br>
# So what to do next?
# <br> The big goal here is to produce something digestible by some machine learning scheme
# <br> From the forked notebook, we can do the same target as:
#  Predicting purchase of a product
# <br> However, I might do a different approach on it since the session is unreliable 
# <br> I would make two tables first
# * Product scores
# * Customer scores

# #### Product Scores
# On each product, basically we want to know:
# 1. Activity (number of each event type)
# 2. Price (Average)
# 3. Category (maybe how many competitor)
# 4. User (User purchasing, user viewing)
# 5. Brand (is branded?)
# 
# #### Customer Scores
# As for customer, we might be wanting:
# 1. Active time
# 2. Activity (for each event type)
# 3. Num Product (for each event type)
# 4. Num Product cat (for each event type)
# 5. Total Transaction (as only retail price shown it should be unavailable)

# ### Get all the value wanted in separate tables 

# In[ ]:


# 1. Get Activity of each product
cross_prev = pd.crosstab(df['product_id'],df['event_type'])
print(cross_prev.head(5))
# 2. Get Price Average
pivot_prodprice = pd.pivot_table(df,index=['product_id'],values=['price'],aggfunc=np.mean)
print(pivot_prodprice.head(5))
# 3. Get Category
unique_cat_pr = df[['category_id','product_id']].drop_duplicates()
cat_size = unique_cat_pr['category_id'].value_counts()
print(cat_size.head(5))
# 4. Get number of unique user
unique_user_event=df[['product_id','user_id','event_type']].drop_duplicates()
cross_prevuser=pd.crosstab(unique_user_event['product_id'],unique_user_event['event_type'])
print(cross_prevuser.head(5))


# In[ ]:


# 5. Isbranded
df_prbr=df[['product_id','brand']].drop_duplicates()
print(len(df_prbr.index)) #oh there are some product that brand added later
df_prbranded=df.loc[df['brand'].notnull(),['product_id']].drop_duplicates()
np_prbrand=df_prbranded.to_numpy()
print(len(df_prbranded.index)/43419)
df_prcat = df[['product_id','category_id']].drop_duplicates()
#trytrylah = np.where((df_prcat['product_id'] in [232423,121212]), True, False)
def isBrandPoduct(series):
    if series['product_id'] in np_prbrand:
        return True
    else:
        return False
df_prcat['isBranded'] = df_prcat.apply(isBrandPoduct,axis='columns')    
print(df_prcat.head(25))
# Wow so long for just labeling this, help me shorten this plz
# now lets jjjjoin


# ### Join them all

# In[ ]:


#Use df_prcat as the base
join_1 = df_prcat.join(cross_prev,on='product_id')
#print(join_1.head(5))
#print(cross_prev.loc[[5802432,5844397,5837166,5876812,5826182]])
#for k in [5802432,5844397,5837166,5876812,5826182]:
#    print(df_prcat.loc[df_prcat['product_id']==k])
join_2 = join_1.join(pivot_prodprice,on='product_id')
join_2.rename(columns={"price": "avg_price"},inplace=True)
#print(join_2.head(5))
join_3 = join_2.join(cat_size,on='category_id',rsuffix='_new')
join_3.rename(columns={"category_id_new": "competitor"},inplace=True)
#print(join_3.head(5))
join_4 = join_3.join(cross_prevuser,on='product_id',rsuffix='_user')
#print(join_4.head(5))
#print(cross_prev.loc[5802432])
#print(cross_prevuser.loc[5802432])
join_4.to_csv("product_score.csv.gz",index=False,compression="gzip")


# ### Now user scores

# In[ ]:


#Time is a bit confusing
#lets experiment a little bit
#remember this
#def day_far(series):
#    time_span=series.max()-series.min()
#    if time_span.days==0: 
#        return 1
#    else:
#        return time_span.days
span = df['event_time'].max()-df['event_time'].min()
print(type(span))
# so type is pandas Timedelta
print(span.value,'nanoseconds')
print(span.days,'days')
print(span.seconds,'seconds')
actual_seconds=span.value/1000000000
print('actual_seconds',actual_seconds)
#how many seconds a day?
day_sec=60*60*24
#span second should be same as
second_left = actual_seconds%day_sec
print('seconds_left',second_left)
#on the mark
sekon=span.seconds%60
minut=((span.seconds-sekon)/60)%60
hourz=(span.seconds-sekon-60*minut)/3600
print('so there should be',hourz,'hours',minut,'minute',sekon,'seconds')


# Lets get back to customer scoring

# In[ ]:


#Now Getting easy customer scores

# 1. Get Activity of each customer
cross_usev = pd.crosstab(df['user_id'],df['event_type'])
print(cross_usev.head(5))
# 2. Get n unique product
unique_us_pr = df[['user_id','product_id']].drop_duplicates()
user_reach = unique_us_pr['user_id'].value_counts()
print(user_reach.head(5))
# 3. Get unique product on each event
unique_us_pr_ev=df[['user_id','product_id','event_type']].drop_duplicates()
cross_usevprod=pd.crosstab(unique_us_pr_ev['user_id'],unique_user_event['event_type'])
print(cross_usevprod.head(5))
# 4. Get n unique category
unique_us_cat = df[['user_id','category_id']].drop_duplicates()
user_diver = unique_us_cat['user_id'].value_counts()
print(user_diver.head(5))
# 5. Get unique category on each event
unique_us_cat_ev=df[['user_id','category_id','event_type']].drop_duplicates()
cross_usevcat=pd.crosstab(unique_us_cat_ev['user_id'],unique_user_event['event_type'])
print(cross_usevcat.head(5))

