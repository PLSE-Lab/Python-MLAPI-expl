#!/usr/bin/env python
# coding: utf-8

# # [Cartier Jewelry](https://www.cartier.com/) - Product Analisys
# 
# On my spare time I am a silversmith. Since eventually I plan on selling some pieces, any info on how the pricing and catalogue of a jewelry store is appreciated.
# 
# Since Cartier is one of the biggest Jewelry sellers and one of the most famous, let's do some web scraping on their site and see if we can get some info.
# 
# Fortunately for us, their site also has a simple structure, which will not be too hard for us to scrap.
# 
# 
# We primarily want info on wich metal/gem is more used (which we will use as a proxy for popularity), which kind of jewelry they have more on catalogue (rings, earings, colars, etc) and which price they practice.

# ## Table of Contents
# 
# - 1. Web Scraping
#     - 1.1. Categories Layer
#     - 1.2. Colection Layer
#     - 1.3. Product Layer
# - 2. Data Cleaning
#     - 2.1. Data Cleaning - ref
#     - 2.2. Data Cleaning - price
#     - 2.3. Data Cleaning - tags
#     - 2.4. Data Cleaning - description
# - 3. Data Analysis
#     - 3.1. Product Distribution
#     - 3.2. Metal Distribution
#     - 3.3. Material Distribution
#     - 3.4. Price Distribution
#         - 3.4.1. Price Distribution
#         - 3.4.2. Price/Categorie Distribution
#         - 3.4.3. Price/Metal Distribution
# - 4. Conclusion
#     - 4.1. Export
#     - 4.2. TL;DR
#     - 4.3. Objective & Final Considerations

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# ## 1. Web Scraping
# 
# Cartier has a good, simple structured site. 
# The site has some simple layers and is realy intuitive from a web scraping side of things.
# 
# The site has the following simplified map: Collections -> Categories -> Collections -> Product
# 
# The first collection divide the type of products (Jewelry, watches, leather goods, etc), categories divide the types of that kindo of product (Jewelry for instance has rings, bracelets, earings, etc), then the second collections wich is the design 'collection' and finaly the product.
# 
# Since we are only interested in Jewelry, we will start there and branch out our webscraping.
# 
# ### 1.1. Categories Layer

# In[ ]:


from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

main_url = 'https://www.cartier.com'

html = urlopen(main_url + '/en-us/collections/jewelry.html')
bsObj = BeautifulSoup(html)

categories_url_list = []

for link in bsObj.findAll('a', href=re.compile('(.)*(\/categories\/)(.)*(viewall\.html)')):
    if 'href' in link.attrs:
        categories_url_list.append(link.attrs['href'])


# ### 1.2. Colection Layer

# In[ ]:


products_url_list = []

for categorie in categories_url_list:
    match = re.match(r'([\w\-\/]+).*', categorie)

    html = urlopen(main_url + categorie)
    bsObj = BeautifulSoup(html)

    for link in bsObj.findAll('a', href=re.compile(match.group(1)+'/')):
        if 'href' in link.attrs:
            products_url_list.append(link.attrs['href'])


# ### 1.3. Product Layer
# 
# Cartier has a more **a picture is worth a thousand words** approach on their product page.
# 
# Each product has a title, price, some tags (normaly the metal and gems used), ref. number and a general description with better info on the metal and gems used.

# In[ ]:


# define panda dataframe
product_df = pd.DataFrame(columns=['ref', 'categorie', 'title', 'price', 'tags', 'description', 'image'])

for product_url in products_url_list:

    html = urlopen(main_url + product_url)
    bsObj = BeautifulSoup(html)

    try:
        html = urlopen(main_url + product_url)
    except HTTPError as e:
        print(e)
        #return null, break, or do some other "Plan B"
    else:        
        #REF
        ref = bsObj.find('span', {'class':'local-ref'}).get_text().strip()

        #Title 
        title = bsObj.find('h1', {'class':'c-pdp__cta-section--product-title js-pdp__cta-section--product-title'}).get_text().strip()

        #Price
        price = bsObj.find('div', {'class':'price js-product-price-formatted hidden'}).get_text().strip()

        #Tags
        tags = bsObj.find('div', {'class':'c-pdp__cta-section--product-tags js-pdp__cta-section--product-tags'}).get_text().strip()
        
        #Description
        description = bsObj.find('div', {'class':'tabbed-content__content-column'}).p.get_text().strip()

        #Image Link
        image = bsObj.find('div', {'class':'c-pdp__zoom-wrapper js-pdp-zoom-wrapper'}).img.attrs['src']
        
        #Categorie
        categorie = re.findall(r'\/categories\/(\w*)\/',product_url)[0]

        product = pd.Series([ref, categorie, title, price, tags, description, image], index=product_df.columns)
        product_df = product_df.append(product, ignore_index=True)


# Nice, we now have the porly formated catalogue of Cartier Jewelery.
# Let's clean it.
# 
# ## 2. Data Cleaning
# 
# A lot of the data that we just scraped need some cleaning.
# - The **ref** number always has a 'REF.:' preamble
# - The **price** is not int formated
# - The **tags** are not propely formated
# - The **description** is just a lot of text as this point
# 
# ### 2.1. Data Cleaning - ref
# 
# This one is easy, we just need to clean the 'REF.:' preamble

# In[ ]:


product_df['ref'] = product_df['ref'].str.replace('REF.:','')


# ### 2.2. Data Cleaning - price
# 
# Another easy one. Just remove the '$' and ',' characters and change the type to float.
# 
# We also have some products that have diferent sizes, like bracelets that have a 40 and 42cm model, and the price is 'from $XXXX' wich is normaly the price of the smaller one. For this cases, let's just remove the 'from' and consider only the cheaper option.
# 
# We should probably use integers here since Cartier doesn't count cents, but then we couldn't use a non-value (NaN) on the prices that we don't have access to, since Cartier doesn't reveal the prices of their most expensive itens upfront on the site.

# In[ ]:


product_df['price'] = product_df['price'].str.replace('$','')
product_df['price'] = product_df['price'].str.replace(',','')
product_df['price'] = product_df['price'].str.replace('from','')

product_df[product_df['price']=='(1)'] = np.nan
product_df[product_df['price']=='0'] = np.nan

product_df['price'] = product_df['price'].astype(float)


# ### 2.3. Data Cleaning - tags
# 
# Ok, this one is a problem.
# We have a lot of tags, normaly about the metal and gems used.
# Let's see if we can identify what are some normal tags. 

# In[ ]:


product_df['tags'] = product_df['tags'].str.lower()
product_df['tags'].unique()[:20]


# Ok, I don't think we should break and change this one.
# 
# Cartier has a filter for metals (Yellow Gold, Rose Gold, Platinum, White Gold, Three-Gold). We could do those 4 columns (Three-gold is just Yellow, Rose and White Gold), but even this doesn't map for everything, we have for exemple a *Non-rhodiumized white gold* on one of the rings.
# 
# And even if we could break the tags, what for?
# 
# Sure, I want to check if they have more Yellow Gold then Rose Gold products, but then I can just do that on the analysis insted of trying to order the dataset here.
# 
# ### 2.4. Data Cleaning - description
# 
# Same on this one, is a lot of unformated info, but we should probably just leave it like that.
# 
# ## 3. Data Analysis
# 
# As mentioned on the intro, I want to have a better understanding of price metrics and what kind of distribution a jewelry store has.
# Do they have more earrings or rings on the catalog?
# What kind of gold they use more?
# Are any gems more commun?
# 
# ### 3.1. Product Distribution
# 
# Let's star with the product distribution.
# What categorie has more models?

# In[ ]:


# Pie chart
x_axis = product_df['categorie'].dropna().unique()
y_axis = product_df['categorie'].value_counts()

print(pd.concat([product_df['categorie'].value_counts(), product_df['categorie'].value_counts(normalize=True)], keys=['counts', 'normalized_counts'], axis=1))

import seaborn as sns

sns.set_style('darkgrid')
f, ax = plt.subplots(figsize=(20, 8))
sns.set(font_scale=1.5)

ax = sns.barplot(y=product_df.categorie.value_counts().index, x=product_df.categorie.value_counts(), orient='h')
ax.set_title('Product Distribution',fontsize=20);


# So we have 37% of the products being rings, 23% bracelets, 22% necklaces and 17% earrings.
# 
# This distribution diverges from what some [small sellers recomend](http://www.bluebuddhaboutique.com/blog/2013/07/what-kind-of-jewelry-sells-best-behind-the-scenes-at-a-craft-fair/) as a rule of thumb to take on craft shows.
# 
# We could do some analysis with other jewelery suppliers and check if  the difference is because Cartier is a high end supplier or if the difference is more on the point of sale.

# ### 3.2. Metal Distribution
# 
# What kind of metal has more models?
# 
# Since the metals and material are together on the **'tags'** column, we are going to do a dictionary of the words on the tags, do a count of how many times weach tag appear and then create move the metal info to a **'metal_list'** created by hand.
# 
# Cartier sells primarily 4 alloys:
# - **Yellow gold:** Yellow gold is made of pure gold mixed with alloy metals such as copper and zinc
# - **White gold:** White gold is an alloy of gold and at least one white metal (usually nickel, silver, or palladium)
# - **Pink gold:** Also know as rose gold, pink gold is made of pure gold mixed with copper and silver alloys
# - **Platinum:** 95-98% Platinum
# 
# At the date of this writing, Platinum is USD 24.98/gram and Gold is USD 55.76/gram.  
# That being said, platinum is more dense, and the Platinum alloy has 95-98% purity, while the most common gold alloy on the US is the 18k (which means 75% purity).  
# So normaly, a Platinum ring is more expensive then a similar Gold one.
# 
# Source: https://www.diamonds.pro/education/platinum-vs-gold/

# In[ ]:


metal_list = {'yellow gold':0,'white gold':0,'pink gold':0, 'platinum':0}

tag_list = {}

for tags in product_df['tags'].dropna():
    for tag in tags.split(', '):
        if tag[-1] == 's':
            tag = tag[:-1]
        if tag not in tag_list:
            tag_list[tag] = 1
        else:
            tag_list[tag] += 1
            
for metal in metal_list:
    metal_list[metal] = tag_list[metal]

print('metal\t\t: counts\t: normalized_counts')
for metal, value in metal_list.items():
    print('{}\t: {}\t\t: {:0.2f}'.format(metal, value, value/len(product_df['tags'].dropna())))
    
    
pd_metal = pd.DataFrame(list(metal_list.items()))
pd_metal.columns =['Metal','Count']
pd_metal = pd_metal.sort_values('Count',ascending=False).reset_index()

f, ax = plt.subplots(figsize=(20, 8))
ax = sns.barplot(y='Metal', x='Count',data=pd_metal, order=pd_metal['Metal'],orient='h')
ax.set_title('Metal Distribution',fontsize=20);


# We can see that Cartier sells primarily gold alloys, with 7% of the products being Platinum.
# 
# The popularity fallows **white gold > pink gold > yellow gold**. With a diference of 5% between the alloys.
# 
# ### 3.3. Material Distribution
# 
# What kind of material is used on more models?
# 
# We are going to remove the metals from the **'tag_list'**, this give us the **'material_list'**

# In[ ]:


material_list = tag_list
for metal in metal_list:
    if metal in material_list:
        del material_list[metal]

pd_material = pd.DataFrame(list(material_list.items()))
pd_material.columns =['Material','Count']
pd_material = pd_material.sort_values('Count',ascending=False).reset_index()
      
f, ax = plt.subplots(figsize=(20, 8))
ax = sns.barplot(y='Material', x='Count',data=pd_material, order=pd_material['Material'],orient='h')
ax.set_title('Material Distribution',fontsize=20);


# Cartier is a high-end jewellery seller, so it stand to reason that their prefered gem/material would be diamonds.  
# Emeralds and shapphires are on the big three gemstones that everyone knows, but it is strange that rubies are so under utilized on Cartier colections.  
# Onyx, ceramic and lacquer are a big surprise since onyx is not considered a high-end gem. 
# 
# ### 3.4. Price Distribution
# 
# Let's check the price distribution, price/categorie distribution, price/metal distribution and price/material distribution.
# This should give us a good ideia of how the pricing is done on Cartier.
# 
# #### 3.4.1. Price Distribution

# In[ ]:


print(product_df['price'].describe())

f, ax = plt.subplots(figsize=(20, 8))
ax = sns.distplot(product_df['price'], kde=False, norm_hist=False);
ax.set_title('Price Distribution',fontsize=20)

plt.xlabel('Price', fontsize=20)
plt.ylabel('Frequency', fontsize=20)

plt.xlim(0, product_df['price'].max())
plt.show()


# Cartier is definatly a high-end seller, with the mean price of 27.057 USD for a product.
# 
# That being said, most of their products are not that expensive.
# 25% of their products is less then 2.620 USD and 50% are less then 6.800 USD.
# 
# #### 3.4.2. Price/Categorie Distribution

# In[ ]:


plt.figure(figsize=(20, 8))
ax = sns.boxplot(y='categorie', x='price', orient='h',data=product_df, showfliers=False)
ax.set_xticks(np.arange(0, 110000, 10000))
ax.set(ylabel='Categories', xlabel='Price')
plt.show()


# Rings and necklaces have a small mean value, so they probably have a good number of products that serve as entry points for clients.
# Strangely the same can't be said for earrings (which normally are also popular entry points for jewelry).
# 
# The bracelets seen to be the items with the bigger price range, wich stand to reason since they have the most area to go crazy and fill every surface with diamonds.
# 
# #### 3.4.3. Price/Metal Distribution

# In[ ]:


metal_df = pd.DataFrame(columns=['title', 'metal', 'price'])

for index,row in product_df.dropna().iterrows():
    for tag in row['tags'].split(', '):
        if tag in metal_list:
            metal_series = pd.Series([row['title'], tag, row['price']], index=metal_df.columns)
            metal_df = metal_df.append(metal_series, ignore_index=True)

plt.figure(figsize=(20, 8))
ax = sns.boxplot(y='metal', x='price', orient='h',data=metal_df, showfliers=False)
ax.set_xticks(np.arange(0, 90000, 10000))
ax.set(ylabel='Categories', xlabel='Price')
plt.show()


# If you remember the **'3.2. Metal Distribution'** topic, platinum items are normaly more expensive then similar gold ones.
# 
# So why do we have more white gold expensive itens? 
# Let's check the most expensive items with white gold and yellow gold. 

# In[ ]:


metal_df[metal_df['metal']=='white gold'].sort_values(by='price', ascending=False).head(20)


# In[ ]:


metal_df[metal_df['metal']=='yellow gold'].sort_values(by='price', ascending=False).head(20)


# As we can see, the items with white gold are significant more expensive then the white gold ones.  
# Since we have more white gold items, and those items are more expensive, that explain why we have a bigger mean and quartiles for the white gold.
# 
# This also flag to us that Cartier does the more expensive items in white gold. This is probably a market trend that we could verify with more data.
# 
# ## 4. Conclusion
# ### 4.1. Export
# Let's do some final export of the data and update the dataset on Kaggle.

# In[ ]:


product_df.dropna().to_csv('/kaggle/working/cartier_catalog.csv',index=False)


# ### 4.2. TL;DR
# 
# I'm a silversmith on my spare time, and would like to analyse the products on Cartier.
# 
# They have on the catalogue:
# - 37% rings, 23% bracelets, 22% necklaces and 17% earrings items.
# - 40% white gold, 35% pink gold, 30% yellow gold and 7% platinum items.
# 
# They use:
# - Diamond on most of their catalogue (At least 55%).
# - Onyx, emeralds, tsavorites and sapphires as their secondary stones.
# 
# The price:
# - 50% of the items are less than 6.800 USD, but the price spread is big, the mean is 27k USD and the most expensive item that as a price on the site costs 370k USD. 
# - The cheapest item is 500 USD
# - The price of categories are in order of high to lowest is: bracelets > necklaces > earrings > rings
# - The price of the products by alloys are in order of high to lowest is: white gold>yellow gold>pink gold>platinum
# 
# Some considerations:  
# Rings and necklaces have a small mean value, so they probably have a good number of products that serve as entry points for clients.
# Strangely the same can't be said for earrings (which normally are also popular entry points for jewelry).
# The bracelets seen to be the items with the bigger price range, wich stand to reason since they have the most area to go crazy and fill every surface with diamonds.
# 
# Cartier does the more expensive items in white gold. This is probably a market trend that we could verify with more data.
# 
# ### 4.3. Objective & Final Considerations
# The initial objective was accomplished, since we got some info on the price spread and some insights on how Cartier catalogue is distributed.
# 
# But they definitely are a high end store. We should probably find some low and mid-end stores and do the same analysis on then and see what differences appear.
