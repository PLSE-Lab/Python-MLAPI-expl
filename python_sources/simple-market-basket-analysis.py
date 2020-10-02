#!/usr/bin/env python
# coding: utf-8

# For all fellow marketers like myself, the intersection of data science with digital marketing is particularly interesting and is a testament to why in the future, digital marketing data scientists will be more in demand. **Market basket analysis or association rules** is one such example of using data science to support marketing decisions.

# The question you might ask is what exactly is market basket analysis. Well if I was to explain it with an example, the recommendations of amazon. I am sure you have all come across the heading of "People who bought this also bought". That right there is market basket analysis in work. MBA (alias for Market Basket from now on) is a series of methodologies for discovering interesting relationships between variables in a database. The outcome can be interpreted as if this then that. 
# 
# If a customer buys bread, then there is a high chance he will buy peanut butter too. We build rules such as these using the transaction data we have. Now peanut butter and jelly is pretty easy and you might say I don't need a bunch of code to derive that. True, but what if you had a store with 10,000 products. Guessing product combinations can get cumbersome.
# 
# So the next question is how we do come up with such rules and how do we know which rules are more likely to increase sales than the rest. I can't just redesign my whole store or start promoting products together on a hunch!. The rules are backed by maths. There are three ratios that are important to decide which rules or itemsets to be more precise are worth looking into: 
# 
# - support 
# - confidence 
# - lift 
# 
# Now in case if I haven't put you to sleep with all this text, let us take a break and read in the dataset. A bit of code to help break the pattern of text abundancy. The dataset that we are working with is from the 
# [UCI Machine Learning Repo](file:///C:/Users/hamza/OneDrive/Documents/Courses%20and%20Certificates/Dataquest/Datasets/Market%20Basket%20Analysis/UCI%20Machine%20Learning%20Repository_%20Online%20Retail%20Data%20Set.html).

# In[ ]:


import pandas as pd


# In[ ]:


data = pd.read_excel("../input/online-retail-store-data-from-uci-ml-repo/Online Retail.xlsx")


# In[ ]:


data.head()


# Let us explore the dataset and get an idea of what the dataset represents

# ### Initial Exploration: 

# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data['Country'].value_counts()


# In[ ]:


data.describe()


# At its simplest, MBS requires just a transaction data column which is unqiue for each row and product names that were bought in each transaction. While data cleaning on the above dataset can be performed, most of these columns will not be neccesary when performing MBA so let's just leave it at that. So let's view our objective.
# 
# You are a digital marketing data scientist for $$.com, a massive e-commerce company and have been tasked to come up with growth strategies for the german site. You decide to conduct MBA use its results as an input for
# 
# - targeted email marketing campaigns
# - implement a site wide recommendation system
# 
# Let us get back to the 3 ratios that we discussed earlier. So what is support, confidence and lift. 
# 
# **Support:** In simple words support is the number of times the items we are studying occur in our dataset. Example: you want to calculate the support for {Bread} -> {Peanut Butter}. You have 15 transactions out of which peanut butter and bread occur 5 times together in a transaction, then support is 5. Usually support is represented as a fraction hence support would be 
# 
# support{Bread} -> {Peanut Butter} = 5/15 = 33.33%
# 
# For a rule to be considered valid or strong, it must have high support otherwise we can dismiss it as occuring due to sheer chance. 
# 
# **Confidence:** Can be explained as the probability or likelihood that the right hand side occuring in a transaction if the item on the lhs is also present. The technical formula is 
# 
# confidence {Bread} -> {Peanut Butter} = support(bread U peanut butter) / support(bread)
# 
# The higher the confidence the more valid this particular rule is. 
# 
# **Lift:** Lift is a measure of how much more likely this association is rather than occuring due to chance. Life is calculated as 
# 
# lift {Bread} -> {Peanut Butter} = confidence(bread,peanut butter) / support(bread) * support(peanut butter)
# 
# If the lift value is equal to one, we say the items are independant and any transaction together was most probably a chance.
# If the lift value is greater than 1, we say there is higher chance of the itemsets being bought together 
# If the lift value is less than 1, we say there is actually a reduced chance of the itemsets being together. 

# ### Performing Market Basket Analysis:

# MBA is made easier by using the apriori algorithm. I won't go into the details of the algorithm, but it has been beautifully explained [here](https://www.geeksforgeeks.org/apriori-algorithm/). Let us manipulate our data so that we can perform MBA on it. 

# In[ ]:


data.head()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.dropna(subset = ['Description'], inplace = True)


# In[ ]:


data.shape


# In[ ]:


germany_data = data.query("Country == 'Germany'")


# In[ ]:


germany_data.shape


# In[ ]:


germany_data.isnull().sum()


# If you go through the data dictionary regarding the invoice no. you will find "InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation."

# In[ ]:


germany_data['Description'].head()


# In[ ]:


germany_data['Description'] = germany_data['Description'].str.strip()


# In[ ]:


germany_data['InvoiceNo'].str.contains('C').sum()


# In[ ]:


germany_data['InvoiceNo'] = germany_data['InvoiceNo'].astype('str')


# In[ ]:


germany_data = germany_data[~germany_data['InvoiceNo'].str.contains('C')]


# In[ ]:


germany_data.shape


# In[ ]:


basket = germany_data.groupby(['InvoiceNo','Description'])['Quantity'].sum().unstack()


# In[ ]:


basket.head()


# Do not be alarmed by abundance of NaN values. 

# In[ ]:


basket.notnull().sum()


# In[ ]:


basket = basket.fillna(0)


# In[ ]:


basket.head()


# The quanity or price columns are not important to us. If you have performed NLP or sentiment analysis then the above table might look similar. The last step before we apply the algorithm is to convert our table values as either 1 or 0.

# In[ ]:


def convert_values(value):
    if value >= 1:
        return 1
    else:
        return 0 


# In[ ]:


basket = basket.applymap(convert_values)


# In[ ]:


basket = basket.drop('POSTAGE', axis = 1)


# ### The Algorithm:

# In[ ]:


from mlxtend.frequent_patterns import apriori 
from mlxtend.frequent_patterns import association_rules


# In[ ]:


basket_items = apriori(basket, min_support = 0.05, use_colnames = True)


# In[ ]:


rules = association_rules(basket_items, metric = 'lift')


# In[ ]:


rules


# So this is what our rules set looks like. Antecedents are basically items on the LHS and consequents are items on the RHS. You can now start shorlisting the rules and items and start building your strategies. 
# 
# It is important to mention here that choosing rules is more than just picking the highest lift value. There are different perspectives to take into account. Let's say for example you had a brick and mortar store. You had a offer running for buy a single loaf of bread and get a jar of peanut better free. The lift value would be high but we have take into account the fact that items might be co occuring because we presented them together. 
# 
# Your business model also affects the thresholds. Ink manufacturers will have to set a lower threshold values for confidence and lift as compared to a retail store. Now back to our strategy. For our email marketing campaign we would select a few but highly scored rules. For our recommendation system, we will rather have an expansive set of rules hence opt towards lower scores. The score that are borderline threhold can also be used. We can showcase them at the cart page and maybe offer a slight discount to encourage upselling. 
