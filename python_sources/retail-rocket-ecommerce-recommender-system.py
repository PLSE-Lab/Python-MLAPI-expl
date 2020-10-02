#!/usr/bin/env python
# coding: utf-8

# # Objective:
#     1. Show how the dataset is structured
#     2. Explore a bit about customer behaviour based on the data and do basic customer segmentation
#     3. Recommendations for future analysis

# This dataset was taken from the Retail Rocket Recommender System dataset: https://www.kaggle.com/retailrocket/ecommerce-dataset/home
# 
# And data was between June 2, 2015 and August 1, 2015

# In[ ]:


import pandas as pd
import numpy as np

import datetime 
import time

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


# Let us load the Retail Rocket CSV files into DataFrames

# In[ ]:


events_df = pd.read_csv('../input/events.csv')
category_tree_df = pd.read_csv('../input/category_tree.csv')
item_properties_1_df = pd.read_csv('../input/item_properties_part1.csv')
item_properties_2_df = pd.read_csv('../input/item_properties_part2.csv')


# # Let's take a peek at the Events dataframe

# In[ ]:


events_df.head()


# The timestamp portion is in Unix Epoch format e.g. 1433221332117 will be converted to Tuesday, 2 June 2015 5:02:12.117 AM GMT
# 
# Visitor Id is the unique user currently browsing the website
# 
# Event is what the user is currently doing in that current timestamp
# 
# Transaction ID will only have value if the user made a purchase as shown below

# In[ ]:


#Which event has a value in its transaction id
events_df[events_df.transactionid.notnull()].event.unique()


# The rest of the events with NaN transaction ids are either view or add to cart

# In[ ]:


#Which event/s has a null value
events_df[events_df.transactionid.isnull()].event.unique()


# # Now let's take a look at the Item Properties

# In[ ]:


item_properties_1_df.head()


# Timestamp is still the same Unix / Epoch format
# 
# Item id will be the unique item identifier
# 
# Property is the Item's attributes such as category id and availability while the rest are hashed for confidentiality purposes
# 
# Value is the item's property value e.g. availability is 1 if there is stock and 0 otherwise
# 
# Note: Values that start with "n" indicate that the value preceeding it is a number e.g. n277.200 is equal to 277.2

# # Category IDs

# In[ ]:


category_tree_df.head()


# Category IDs explain the relationship of different products with each other e.g. Category ID 1016 is a child of Parent ID 213.
# 
# Below shows the number of items under category id 1016

# In[ ]:


item_properties_1_df.loc[(item_properties_1_df.property == 'categoryid') & (item_properties_1_df.value == '1016')].sort_values('timestamp').head()


# # Customer behaviour exploration

# I think it's prudent to start separating customers into two categories, those who purchased something and those who didn't

# In[ ]:


#Let's get all the customers who bought something
customer_purchased = events_df[events_df.transactionid.notnull()].visitorid.unique()
customer_purchased.size


# Assumptions:
#     1. Since we have no information whether there were any repeat users who bought something from the site, I'll just have to assume for now that the 11,719 visitors are unique and made at least a single purchase

# In[ ]:


#Let's get all unique visitor ids as well
all_customers = events_df.visitorid.unique()
all_customers.size


# Out of 1,407,580 unique visitor ids, let's take out the ones that bought something

# In[ ]:


customer_browsed = [x for x in all_customers if x not in customer_purchased]


# In[ ]:


len(customer_browsed)


# So there were actually 1,395,861 unique site visitors who didn't buy anything, again assuming that there were no repeat users with different visitor IDs

# In[ ]:


#Another way to do it using Numpy
temp_array = np.isin(customer_browsed, customer_purchased)
temp_array[temp_array == False].size


# In[ ]:


#A sample list of the customers who bought something
customer_purchased[:10]


# # Below is a snapshot of visitor id 102019 and their buying journey from viewing to transaction (purchase)

# In[ ]:


events_df[events_df.visitorid == 102019].sort_values('timestamp')


# If we want to convert the UNIX / Epoch time format to readable format then just do the code below

# In[ ]:


tz = int('1433221332')
new_time = datetime.datetime.fromtimestamp(tz)
new_time.strftime('%Y-%m-%d %H:%M:%S')


# In[ ]:


tz = int('1438400163')
new_time = datetime.datetime.fromtimestamp(tz)
new_time.strftime('%Y-%m-%d %H:%M:%S')


# # What insights can we offer the visitor to guide them in their buying journey?
# 
# -perhaps we can offer them a list of what previous visitors bought together with the item they are currently viewing

# In[ ]:


# Firstly let's create an array that lists visitors who made a purchase
customer_purchased = events_df[events_df.transactionid.notnull()].visitorid.unique()
    
purchased_items = []
    
# Create another list that contains all their purchases 
for customer in customer_purchased:

    #Generate a Pandas series type object containing all the visitor's purchases and put them in the list
    purchased_items.append(list(events_df.loc[(events_df.visitorid == customer) & (events_df.transactionid.notnull())].itemid.values))                                  


# So now all items purchased together are presented as a list of lists, shown below are the first 5 samples

# In[ ]:


purchased_items[:5]


# In[ ]:


# Write a function that would show items that were bought together (same of different dates) by the same customer
def recommender_bought_bought(item_id, purchased_items):
    
    # Perhaps implement a binary search for that item id in the list of arrays
    # Then put the arrays containing that item id in a new list
    # Then merge all items in that list and get rid of duplicates
    recommender_list = []
    for x in purchased_items:
        if item_id in x:
            recommender_list += x
    
    #Then merge recommender list and remove the item id
    recommender_list = list(set(recommender_list) - set([item_id]))
    
    return recommender_list


# # So now we can present to the visitor a list of the other items a customer previously bought along with what item the current visitor is viewing e.g. item number 302422

# In[ ]:


recommender_bought_bought(302422, purchased_items)


# That was a very crude way of recommending other items to the visitor

# # What other insights can we gather from the items that were viewed, added to cart and sold?

# # Can we perhaps cluster the visitors and see if classes appear?
# 
# For that I will need to create a new dataframe and engineer a few features for it

# How many unique visitors did we have for the site from June 2, 2015 to August 1, 2015?
# 
# Shown below are the total number of visitors for that time duration (was also shown at the close to the start of this paper)

# In[ ]:


#Put all the visitor id in an array and sort it ascendingly
all_visitors = events_df.visitorid.sort_values().unique()
all_visitors.size


# In[ ]:


buying_visitors = events_df[events_df.event == 'transaction'].visitorid.sort_values().unique()
buying_visitors.size


# Out of 1,407,580 visitors, ony 11,719 bought something so around 1,395,861 visitors just viewed items

# In[ ]:


viewing_visitors_list = list(set(all_visitors) - set(buying_visitors))


# Now lets create a function that creates a dataframe with new features: visitorid, number of items viewed, total viewcount, bought something or not

# In[ ]:


def create_dataframe(visitor_list):
    
    array_for_df = []
    for index in visitor_list:

        #Create that visitor's dataframe once
        v_df = events_df[events_df.visitorid == index]

        temp = []
        #Add the visitor id
        temp.append(index)

        #Add the total number of unique products viewed
        temp.append(v_df[v_df.event == 'view'].itemid.unique().size)

        #Add the total number of views regardless of product type
        temp.append(v_df[v_df.event == 'view'].event.count())

        #Add the total number of purchases
        number_of_items_bought = v_df[v_df.event == 'transaction'].event.count()
        temp.append(number_of_items_bought)

        #Then put either a zero or one if they made a purchase
        if(number_of_items_bought == 0):
            temp.append(0)
        else:
            temp.append(1)

        array_for_df.append(temp)
    
    return pd.DataFrame(array_for_df, columns=['visitorid', 'num_items_viewed', 'view_count', 'bought_count', 'purchased'])


# Let's apply this to buying visitors first

# In[ ]:


buying_visitors_df = create_dataframe(buying_visitors)


# In[ ]:


buying_visitors_df.shape


# I think I'll only get around 27,821 samples from the viewing visitors list so that there is a 70-30 split for training and test data. 

# In[ ]:


#Let's shuffle the viewing visitors list for randomness
import random
random.shuffle(viewing_visitors_list)


# In[ ]:


viewing_visitors_df = create_dataframe(viewing_visitors_list[0:27820])


# In[ ]:


viewing_visitors_df.shape


# Now let's combine the two dataframes

# In[ ]:


main_df = pd.concat([buying_visitors_df, viewing_visitors_df], ignore_index=True)


# Let's plot main_df and see if anything comes up

# In[ ]:


#Let's shuffle main_df first
main_df = main_df.sample(frac=1)


# In[ ]:


#Plot the data
sns.pairplot(main_df, x_vars = ['num_items_viewed', 'view_count', 'bought_count'],
             y_vars = ['num_items_viewed', 'view_count', 'bought_count'],  hue = 'purchased')


# The plot above clearly indicates that the higher the view count, the higher the chances of that visitor buying something. Duh!

# # Since the relationship is Linear, let's try a simple Logistic Regression model to predict future visitor purchase behaviour

# We separate the features (drop visitorid since it's categorical data and bought count) and the target (which is whether the visitor bought something or not)

# In[ ]:


X = main_df.drop(['purchased', 'visitorid', 'bought_count'], axis = 'columns')
y = main_df.purchased


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = 0.7)


# In[ ]:


logreg = LogisticRegression()


# In[ ]:


logreg.fit(X_train, y_train)


# In[ ]:


# Let's now use the model to predict the test features
y_pred_class = logreg.predict(X_test)


# In[ ]:


print('accuracy = {:7.4f}'.format(metrics.accuracy_score(y_test, y_pred_class)))


# # So our model's accuracy in predicting buying visitors is around 79.46%

# In[ ]:


# Generate the prediction values for each of the test observations using predict_proba() function rather than just predict
preds = logreg.predict_proba(X_test)[:,1]

# Store the false positive rate(fpr), true positive rate (tpr) in vectors for use in the graph
fpr, tpr, _ = metrics.roc_curve(y_test, preds)

# Store the Area Under the Curve (AUC) so we can annotate our graph with theis metric
roc_auc = metrics.auc(fpr, tpr)

# Plot the ROC Curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw = lw, label = 'ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color = 'navy', lw = lw, linestyle = '--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc = "lower right")
plt.show()


# The graph above shows the accuracy of our binary classifier (Logistic Regression). Just means that the closer the orange curve leans to the top left hand part of the graph, the better the accuracy.
