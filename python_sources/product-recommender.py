#!/usr/bin/env python
# coding: utf-8

# # Product Recommendation Using Jaccard Similarity

# Problem statement:
# Given list of user with their purchased brands, find the recommended brands using Jaccard Similarity

# In[ ]:


import pandas as pd
user = pd.read_csv("../input/peoplerecommender.csv")
user.head()


# In[ ]:


user.Store.value_counts()


# The data that is stored will not be having 1N form , i.e it will have repeated entires for same user.
# So let us create a python dictionary such that userid as key, and the value will be the array of brand names that the user purchased.

# In[ ]:


userid = user.ID.copy()
user_brands = dict(user.groupby('ID').Store.apply(lambda x:list(x)))
#Print top 2 dictionary items
dict(list(user_brands.items())[0:2])


# We can use Jaccard similarity formula to calculate the recommended brands.
#  
# The Jaccard index, also known as Intersection over Union and the Jaccard similarity coefficient is a statistic used for gauging the similarity and diversity of sample sets. The Jaccard coefficient measures similarity between finite sample sets, and is defined as the size of the intersection divided by the size of the union of the sample sets:(Wikipedia)
# 
# 
# $$ J(A,B) = {{|A \cap B|}\over{|A \cup B|}} = {{|A \cap B|}\over{|A| + |B| - |A \cap B|}}$$
#   

# In[ ]:


def jac_sim (x, y):
    x = set(x)
    y = set(y)
    return len(x & y) / float(len( x | y))


# In[ ]:


#Example
print(jac_sim(user_brands[80139],user_brands[80135])) # This is for user 80135
# This is for user 80139, and it should give 1 as we are calcualting similarity with itself.
print(jac_sim(user_brands[80139],user_brands[80139])) 


# For each user, we will create a list of all Jaccard similarity value with respect to other user.
# 
# To just test the functionality, let us do it for one user test_user who purchased from brands - 'Target','Old Navy', 'Banana Republic', 'H&M'.
# 
# Create a new dictioanry where Key  = userid, Value = Jaccard value of the other user with respect to our test_user
#  

# In[ ]:



test_user = ['Target','Old Navy', 'Banana Republic', 'H&M']
jac_list = {}
for userid, brand in user_brands.items():
    jac_list[userid] = jac_sim(test_user,brand)
#Print top 2 dictionary items
dict(list(jac_list.items())[0:2])
    


# Take only the top 5 users who have greater Jaccord value (high similarity)

# In[ ]:


top_users = sorted(jac_list.items(), key = lambda x: x[1], reverse = True)[:5]
top_users


# 
# The next step is to get all the brands of those 5 users where the test_user does not purchased yet and recommend it.

# In[ ]:


#K Most similar users
recommendedbrands = set()
for user in top_users:
    for brand in user_brands[user[0]]:
        if brand not in test_user:
            recommendedbrands.add(brand)
    


# In[ ]:


recommendedbrands


# The above is for one user; we can loop through the entire user list and create data structure which has
# user id and their respective recommended brands.
# We can also combine all the above functionalities as a python function and invoke it in the loop.

# In[ ]:


def getRecommendedBrands(userid):
    userbrand = user_brands[userid]
    jac_list = {}
    for userid, brand in user_brands.items():
        jac_list[userid] = jac_sim(userbrand,brand)
    top_users = sorted(jac_list.items(), key = lambda x: x[1], reverse = True)[:5]
    recommendedbrands = set()
    for user in top_users:
        for brand in user_brands[user[0]]:
            if brand not in test_user:
                recommendedbrands.add(brand)
    return recommendedbrands
    


# In[ ]:


#Testing the method for the user id 80010
getRecommendedBrands(80010)
 


# Final output : Dictionary of Users and their Recommended brands for all the users in the csv file

# In[ ]:


user_vs_recommenders = {}
for userid, brand in user_brands.items():
    user_vs_recommenders[userid] = getRecommendedBrands(userid)

#Print top 10 dictionary items
dict(list(user_vs_recommenders.items())[0:10])

