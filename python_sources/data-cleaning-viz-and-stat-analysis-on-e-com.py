#!/usr/bin/env python
# coding: utf-8

# # Olist - Brazilian E-Commerce Dataset
# ## The main goal of this analysis is to understand what variables yeilds possitive reviews (by using OLS method -  linear regression). This tutorial is divided into three equally important parts: 1 - Data Cleansing, 2 - Visualization, and finally 3 - Statistical analysis. I will try to explain every single step of my analysis as best as I can. Hope you will find it useful or at least, worthy of your time and attention! 

# In[ ]:


# Import all of the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import matplotlib as mpl
mpl.rcParams['patch.force_edgecolor'] = True
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#read the csv file and change all columns with date information into datetime objects
olist = pd.read_csv("../input/olist_public_dataset_v2.csv", index_col=0, parse_dates=['order_purchase_timestamp', 
                    'order_aproved_at', 'order_estimated_delivery_date', 'order_delivered_customer_date', 
                    'review_creation_date', 'review_answer_timestamp'])


# In[ ]:


olist.info()


# #### Luckily, we don't have to deal with the missing data

# ### Part I - Cleaning the Data!
# ### Let's explore each feature/columns and clean/sort them out for applying some cool visualization and statistical analysis!

# #### Change the datatype of "Order_status" and "product_category_name" columns from regular string to categorical variable  

# In[ ]:


stat_cat = olist['order_status'].unique().tolist()
stat_cat


# In[ ]:


status_cat = pd.Categorical(olist['order_status'], categories=stat_cat, ordered=False)
status_cat


# In[ ]:


#reassign status_cat to original "order_status" column
olist['order_status'] = status_cat


# In[ ]:


#Do the same operation on the "product_category_name" column
cat_name = olist['product_category_name'].unique().tolist()
olist['product_category_name'] = pd.Categorical(olist['product_category_name'], categories=cat_name, ordered=False)
olist['product_category_name'].describe()


# In[ ]:


#Make ordered list of qty of photos for future viz
ordered_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
olist['product_photos_qty'] = pd.Categorical(olist['product_photos_qty'], categories = ordered_list, ordered=True)
olist['product_photos_qty'].dtype


# #### Let's normalize the text

# In[ ]:


#capitalize first character of each word
olist['customer_city'] = olist['customer_city'].str.title()
olist['customer_state'] = olist['customer_state'].str.upper()
olist['customer_state'].value_counts().head(7)


# In[ ]:


olist['review_comment_title'] = olist['review_comment_title'].str.strip().str.lower()


# In[ ]:


olist['product_category_name'][1]


# In[ ]:


#replace underlines with spaces
olist['product_category_name'] = olist['product_category_name'].str.replace('_', ' ').str.lower()
olist['product_category_name'].value_counts().head(13)


# In[ ]:


to_eng_cat_name = olist['product_category_name'].unique().tolist()
to_eng_cat_name[0]


# In[ ]:


#translator = Translator(service_urls=[
#      'translate.google.com',
#      'translate.google.com.br',
#    ])
#translations = translator.translate(to_eng_cat_name, dest='en')


# #### Unfortunately, googletrans library is not functioning properly and raising an AttributeError. I will try to solve the problem by the next kernel upload. Ideally, I would have created a python's dictionary comprehention (where result would've looked like {original Por text : En text, ect}); then, assign the values of english text with the help of .map() function
# #### But for now, let's all pretend that we know Portuguese!

# #### Let's create a column which tells us how quickly or slowly each item was delivered. Also, let's create one more column which will be the total value/cost of each transaction.  

# In[ ]:


(olist['order_estimated_delivery_date'] - olist['order_delivered_customer_date']).describe()


# In[ ]:


((olist['order_estimated_delivery_date'] - olist['order_delivered_customer_date']) / (np.timedelta64(1, 'D'))).plot(kind='hist', bins=50)


# In[ ]:


#On average more items came earlier than estimated delivery date 
olist['delivery_accuracy'] = ((olist['order_estimated_delivery_date'] - olist['order_delivered_customer_date']) 
                               / (np.timedelta64(1, 'D')))


# In[ ]:


olist['total_value'] = olist['order_products_value'].add(olist['order_freight_value'])


# In[ ]:


#Let's check whether the changes were made successfully (with correct datatype) or not 
olist.info()


# In[ ]:


olist.describe(include='all')


# ### Part II - Visualization
# ### Diving deeper to understand more about our dataset

# #### How many items are ordered at one transaction and how the total cost of each transaction increases as more quantity is added?

# In[ ]:


olist['order_items_qty'].value_counts().sort_index()


# In[ ]:


olist.groupby('order_items_qty')['total_value'].mean().plot(kind='bar',figsize=(12,5))


# #### 93% of transactions have only one ordered item. Moreover, more than 6 units of items were ordered in transaction only in 0.024% cases. 
# #### As the quantity of item increased at a given transaction, the quality (meaning the average price of each unit) didn't diminish much - maybe they are returning customers who had already dealt with the seller !? Unfortunately, we can't test our hypothesis with the given dataset as it lacks some features  

# In[ ]:


sns.set_style("whitegrid")
plt.figure(figsize=(12,6))
sns.distplot(olist['total_value'], bins=800 ,kde=False, color='b')
plt.xlim([0, 600])


# #### As it was expected, we got right skewed histogram for total_value column - most of the times, people buy cheaply priced goods on olist

# In[ ]:


state_grouped = (olist.groupby('customer_state')[['order_products_value', 'review_score']]
                             .agg({'review_score': ['mean', 'count'], 'order_products_value':['mean']})
                ).sort_values(by=('review_score','mean'), ascending=False)
                 
state_grouped.head()


# In[ ]:


state_grouped.plot(kind='barh', figsize=(12,11), logx=True)


# #### Customer reviews doesn't change much from state to state (roughly, 0.6 point difference on average), but total volume of orders along side with number count of reviews fluctuate quite dramatically. In short, some states inclined to shop more frequently than others but their experience doesn't differ much 

# In[ ]:


fig, (ax1, ax2) = plt.subplots(2, 1)
fig.set_figheight(8)
fig.set_figwidth(15)


(olist.groupby(olist['order_purchase_timestamp'].dt.month)['order_products_value'].mean()
      .plot(kind='bar', ax=ax1, ylim=(115,140), 
            title='Average Prices for Orders in Brazilian Real Per Month')
)
(olist.groupby(olist['order_purchase_timestamp'].dt.month)['order_products_value'].sum()
      .plot(kind='bar', ax=ax2, ylim=(600000,1350000), sharex=True,
           title='Total Volume of Orders in Brazilian Reals Per Month')
)


# #### Volume of goods ordered on Olist were lowest on December for some reason - Maybe people shop locally on holiday season!? While average prices of orders were reletively higher on Sept and Oct  

# In[ ]:


olist['review_comment_title'].value_counts().head()


# #### As soon as I will find out how to deal with googletrans' AttributeError, I will do some interesting text analysis!

# In[ ]:


pweekday = olist['order_purchase_timestamp'].dt.weekday
phour = olist['order_purchase_timestamp'].dt.hour
pprice = olist['total_value']
purchase = pd.DataFrame({'day of week': pweekday, 'hour': phour, 'price': pprice})
purchase['day of week'] = purchase['day of week'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
purchase.head()


# In[ ]:


purchase_count = purchase.groupby(['day of week', 'hour']).count()['price'].unstack()
plt.figure(figsize=(16,6))
sns.heatmap(purchase_count.reindex(index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']), 
            cmap="YlGnBu", annot=True, fmt="d", linewidths=0.5)


# #### The most frequently, customers tend to shop online on weekdays from 10am to 4pm. There are sudden peaks around 8-9pm too (Mon-Thu) and intuitively, on sunday nights (5-9pm), online buyers restart their shopping habbits from relatively low Saturday. 

# In[ ]:


dweekday = olist['order_delivered_customer_date'].dt.weekday
dhour = olist['order_delivered_customer_date'].dt.hour
dprice = olist['total_value']
delivery = pd.DataFrame({'day of week': dweekday, 'hour': dhour, 'price': dprice})
delivery['day of week'] = delivery['day of week'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
delivery_count = delivery.groupby(['day of week', 'hour']).count()['price'].unstack()
plt.figure(figsize=(16,6))
sns.heatmap(delivery_count.reindex(index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']), 
            cmap="BuPu", annot=True, fmt="d", linewidths=0.5)


# #### The story of delivery reveals that weekdays from 3-9pm are heaviest postal delivery truck operators!

# In[ ]:


top6 = olist['customer_city'].value_counts().head(6)
top6 = top6.index.tolist()
top6_cities = olist[olist['customer_city'].isin(top6)]
top6_cities['customer_city'].describe()


# In[ ]:


import warnings
warnings.filterwarnings('ignore')
(
    ggplot(data=top6_cities)
      + aes(y='review_score', x='delivery_accuracy')
      + aes(color='order_status', size='product_description_lenght')
      + geom_point(alpha=0.05)
      + geom_jitter()
      + facet_wrap('~customer_city', nrow=3, ncol=2)
      + theme_classic()
      + theme(figure_size=(18,15))
)


# #### We coundn't see clear positive relationship between item being delivered earlier than promised and review score through our scatter plot. Let's go to the next stage of our analysis  

# ### Part III - statistical analysis
# 

# ### What are the expected reletionships of independent variables with review score?
# * order_items_qty ("+") - if consumer gets more than one item from the same seller, it should mean that he/she knows the quality of the good. Therefore, increase in item quantity should increase review score  
# * product_description_lenght ("+") - buyer having more information about buying product should have positive relationship with review score
# * product_photos_qty ("+") 
# * delivery_accuracy ("+") - item coming on time or earlier that it was described should have positive relationship with review score
# * order_products_value ("unknown") - more expensive perhaps means better quality or higher expectation; thus, it is hard to predict
# 

# In[ ]:


new_df = olist[['order_items_qty', 'product_description_lenght', 'product_photos_qty', 'delivery_accuracy', 'order_products_value', 'review_score']]
new_df.info()


# In[ ]:


cor = new_df.corr()
sns.heatmap(cor, annot=True, fmt=".2g", linewidths=0.5)


# #### Most variables do have quite weak relationship with review score

# In[ ]:


import statsmodels.api as sm


# In[ ]:


model = sm.OLS.from_formula('review_score ~ order_items_qty + product_description_lenght + product_photos_qty + delivery_accuracy + order_products_value', data=new_df)


# In[ ]:


result = model.fit()


# In[ ]:


print(result.summary())


# ### SummaryAll of our independent variables are statistically significant which is a great news. R^2 is not as strong as we want but considering the huge amount of observations, it is somewhat tolerable
# ### Results:
# * As it was expected in our initial hypotheses stage, product description lenght, photos quantities and delivery time have positive relationship with review score
# * While more expensive items sold online do have negative relationship with review score
# * Suprisingly, order item quantity have negative relationship with review score. Do you guys have any guesses (why)?
# 
# 

# ## I hope I was able to demonstrate you all some valuable insight about this Olist E-Commerce data. If you liked my analysis, please upvote my kernel and please feel free to share your disagreements, thoughts, or suggestions on the comment section below!
