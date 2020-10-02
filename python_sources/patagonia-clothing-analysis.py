#!/usr/bin/env python
# coding: utf-8

# # Patagonia Clothing Price and Color Analysis
# 
# **Goal: Compare the average cost and color availability for men's and women's clothing sold at Patagonia**
# 
# Why this goal? [Several studies](https://www.insider.com/women-more-expensive-products-2018-8#girls-backpacks-cost-slightly-more-than-boys-11) have found that products for women tend to be pricier than products for men, and this [includes clothing](https://www1.nyc.gov/assets/dca/downloads/pdf/partners/Study-of-Gender-Pricing-in-NYC.pdf)! We want to see if Patagonia falls into this same gender disparity, or if they are doing a good job at pricing their clothes equally, regardless of the "gender" 
# 
# (I'm using "quotes" above because these gendered categorizations for clothing assume the traditional binary gender identities, which aren't necessarily valid -- you can buy clothing from whichever category you want!)  
# 
# In addition, as far as I know, this hasn't been the specific topic of any particular published study, but any casual consumer can observe that womens clothing tends to have a wider variety of options, including colors. Is Patagonia providing mean with an equal number of color options to women?
# 
# These data were collected using a web scraping tool I built, which you can access on my [github repository](https://github.com/bethmorrison25/Web-Scraping-Tutorial)
# 
# Below we will visualize differences in the price and color options of mens and womens clothing, as well as perform some basic statistical analyses to test for differences between the mens and womens categories.  

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt ## pyplot is a submodule within matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
## this allows us to print the plots in the Jupyter Notebook


# In[ ]:


item_data = pd.read_csv('../input/patagoniaclothing/Patagonia_WebScrape_ClothingItems_v1.csv')


# In[ ]:


item_data.head()


# ## Visualizing the data
# 
# We can use the `seaborn` and `matplotlib` modules to visualize our data. Remember we're interested in looking at whether there are differences in womens and mens clothing prices and color options. 
# 
# We can visualize this a couple different ways, and for demonstration purposes we'll plot multiple graphs. 
# 
# First, we can plot the distribution of item prices in the womens and mens categories using a density plot

# In[ ]:


genders = ['womens', 'mens']
for gender in genders:
    subset = item_data[item_data['item_gender'] == gender]
    with sns.color_palette("muted"): ## changes the color palette
        sns.distplot(subset['item_price'], 
                 hist = False, 
                 kde = True, 
                 kde_kws = {'linewidth':3, 'shade': True}, 
                 label = gender)


# They actually don't look too different! For both the men and womens clothing, it looks like most products are priced around 100 dollars, with a few items that are more expensive, with a greater proportion of the womens items being priced at around 100 dollars. We can also see that the data are a little right skewed - they don't follow a completely normal distribution and the tail goes out to the right to capture those higher priced items.
# 
# Another way to visualize this is with a boxplot. 

# In[ ]:


with sns.color_palette("muted"):
    sns.boxplot(
        x = 'item_gender', 
        y = 'item_price',
        data = item_data)


# In the boxplot we can see that the median item price for womens and mens clothes is indeed around 100 dollars, with some items that are more expensive, and a greater number of more expensive items in the mens section (but not by much). Based on this boxplot too, we can again see that the data do not form a normal distribution, but are instead right skewed. 
# 
# What about the number of color options? 

# In[ ]:


genders = ['womens', 'mens'] ## to plot the men and womens colors separately
for gender in genders:
    subset = item_data[item_data['item_gender'] == gender]
    with sns.color_palette("muted"): ## changes the color
        sns.distplot(subset['item_colors'], 
                     hist = False, 
                     kde = True, 
                     kde_kws = {'linewidth':3, 'shade': True}, 
                     label = gender)


# The number of color options available for mens and womens clothing looks about the same too! Items on average have around 3 color options, with a higher proportion of womens clothes having this many options, but several items in both the mens and womens sections have a high number of color options. As with the item prices, this means that the data are right skewed. 
# 
# Let's look at this using a boxplot as well

# In[ ]:


with sns.color_palette("muted"):
    sns.boxplot(
        x = 'item_gender', 
        y = 'item_colors',
        data = item_data)


# We see here the median number of color options for womens and mens clothing is indeed around 3 colors, with a very similar number of color options between the gender categories. There are also a few more men's items that have a high number of color options. 

# ## A quick hypothesis test
# 
# Thanks to our visualizations above, we can see that the womens and mens clothing prices and color options are about equal. It doesn't hurt to do a quick statistical test though, just to make sure our eyes aren't deceiving us!
# 
# First let's get some numbers on the difference in the average price between mens and womens clothing (this can be thought of as a very basic measure of effect size) and get an idea of the parameters of the distribution for the mens and womens clothing prices (i.e. the mean and standard deviation)
# 

# In[ ]:


## we'll want to do this for prices and colors so we can write a little function

def item_ef(category):
    """Prints mean, std, and effect size of the mens and womens item information"""
    
    w_info = item_data[category].loc[item_data['item_gender'] == 'womens']
    m_info = item_data[category].loc[item_data['item_gender'] == 'mens']
    
    ## the .3f will round the answer to 3 decimal places
    print('Womens Items: Mean=%.3f, Standard Deviation=%.3f' % (np.mean(w_info), np.std(w_info)))
    print('Mens Items: Mean=%.3f, Standard Deviation=%.3f' % (np.mean(m_info), np.std(m_info)))
    print('Difference in means=%.3f' % (np.mean(m_info) - np.mean(w_info)))


# In[ ]:


item_ef('item_price')


# Our very basic measure of effect size (difference in means) tells us that mens clothes are $14.43 more expensive on average than womens clothing (interesting!)

# To test for differences in average value between the two categories, we could simply perform a t-test. However, one of the assumptions of a t-test is that the data are equally distributed, and we saw in our visualizations that our data are right skewed. If you have a high enough n (number of samples), non-normality isn't a huge problem (read more about this here https://blog.minitab.com/blog/adventures-in-statistics-2/choosing-between-a-nonparametric-test-and-a-parametric-test), and we do have quite a large n, but we have other options that do not assume normality and we can use those instead. 
# 
# Rather than a t-test, we can use the Mann-Whitney U-test, a non-parametric test that still tests for differences in the average between two categories except it doesn't assume a normal distribution. It does have some other assumptions, which you can read about here (https://statistics.laerd.com/statistical-guides/mann-whitney-u-test-assumptions.php), and here https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/, the main one being that the data in each category have similar distributions (which ours do!) 
# 
# Another test we could do is a permutation test (https://towardsdatascience.com/how-to-assess-statistical-significance-in-your-data-with-permutation-tests-8bb925b2113d) but Mann-Whitney is appropriate and quite easy.
# 
# Onward with the Mann Whitney U test!
# 
# First we need to import some functions from a library that we haven't imported yet: the `mannwhitneyu` function from the `stats` component of the `scikitlearn` library

# In[ ]:


from scipy.stats import mannwhitneyu


# In[ ]:


def item_mannwhitney(category):
    """Prints statistic and P value for Mann Whitney U test for diff in mens and womens items"""
    
    w_info = item_data[category].loc[item_data['item_gender'] == 'womens']
    m_info = item_data[category].loc[item_data['item_gender'] == 'mens']
    
    ## the .3f will round the answer to 3 decimal places
    stat, p = mannwhitneyu(w_info, m_info)
    print('Statistic=%.3f, p-value=%.3f' % (stat, p))


# In[ ]:


item_mannwhitney('item_price')


# And we can see based on this test that our conclusions based on our visualization were correct - there is not a significant difference between the prices for mens and womens clothing. Our null hypothesis is that mens and womens clothing are the same price on average. The statistical test provided us with a p-value of 0.17, which greater than 0.05. Thus, at an alpha level 0.05 (the common standard in statistical tests), our the difference in prices between the two categories is not significant and thus we fail to reject the null hypothesis. 

# Now onto the color options data
# 
# First, lets look at effect size (diff in means)

# In[ ]:


item_ef('item_colors')


# It looks like the average number of color options available for mens and womens items are about the same! 
# 
# Let's check with a Mann-Whitney test

# In[ ]:


item_mannwhitney('item_colors')


# Indeed there is no statistically significant difference between the average number of color options between the mens and womens items!

# ### Conclusions
# 
# **Goal: Compare the average cost and color availability for men's and women's clothing sold at Patagonia**
# 
# 
# 
# We found that on average womens and mens clothing items are **about the same**, and mens and womens clothing provide the same number of color options on average! 
# 
# Ultimately, this is good for Patagonia! It means they are not unfairly pricing their items and making womens clothing items more expensive (as many companies do! whether that's consciously or unconsciously) and they are providing equal numbers of color choices between the mens and womens clothing items as well.
# 

# In[ ]:




