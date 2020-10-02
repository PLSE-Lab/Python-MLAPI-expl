#!/usr/bin/env python
# coding: utf-8

# ![](https://thumbs.gfycat.com/AlarmedOpulentAardwolf-small.gif)
# # Introduction
# 
# Ramen has been the staple of Japanese cuisine for years. It made its way to the United States after the creation of instant noodles in the 1970s. Momofuku Ando, the inventor of instant noodle, realised that he could partially fry and dehydrate the noodle to improve its shelf-life. He then moved on to found Nissin Foods and introduced instant Chikin Ramen to Japanese consumers. 5 years later, the company created the cup noodle, a product that revolutionise the entire processed-food industry. Needless to say, the instant ramen was a great success worldwide, including the United States.
# 
# ![Even Michael Scott endorses it](https://productplacementblog.com/wp-content/uploads/2019/06/Nissin-Cup-Noodles-Held-by-Steve-Carell-Michael-Scott-in-The-Office-1.jpg)
# In The Office, Cup Noodles is a popular lunch among the characters of the show (specifically Michael Scott and Kevin Malone).
# 
# Here, we will identify the following:
# - Which brand has the most review?
# - Which country produce the most ramen brand?
# - Which style of ramen (cup, tray, pack) is the best?
# - What are the common words used in these products?

# In[ ]:


# Importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import random 


# In[ ]:


def overview():
        
    data = pd.read_csv('../input/ramen-ratings/ramen-ratings.csv')
    # Print the first 5 lines of data
    print("First 5 lines of data \n\n")
    print(data.head())

    # Print data type
    print("\n\n\nDatatype\n")
    print(data.dtypes)

    # Print number of null values 
    print("\n\n\nNumber of null values\n")
    print(data.isnull().sum())

    # Print data summary
    print("\n\n\nData summary\n")
    print(data.describe())

    # Print data shape
    print("\n\n\nData shape\n")
    print("Data has {} rows and {} columns".format(data.shape[0], data.shape[1]))

    return data

data = overview()


# ### What do we see?
# - We do not need 'Top Ten' column so we will remove it.
# - We can convert stars to numerical values.
# - We will identify the 2 null values in style and replace them with the appropriate style.

# In[ ]:


# Dropping 'Top Ten' column 
data.drop(columns = ['Top Ten'], inplace = True)

# Convert 'Stars' column to numeric
data['Stars'] = pd.to_numeric(data['Stars'],errors='coerce')

# Identify the NaN values in 'Style' column and replace them with the right style
data[data['Style'].isna()]


# After doing a simple Google search we managed to identify the 2 items as 'Pack'.
# 
# **Kamfen E-men chicken**
# ![](https://www.theramenrater.com/wp-content/uploads/2011/06/2011_6_17_428_001.jpg)
# 
# **Unif 100 Furong Shrimp**
# ![](https://photos1.blogger.com/blogger/4974/1575/1600/06-04-16%20001.jpg)
# 
# We will replace the NaN values with 'Pack'.

# In[ ]:


# Replace NaN with 'Pack'
data['Style'].fillna('Pack', inplace = True)

# Check if NaN is replaced
target = ['E Menm Chicken', '100 Furong Shrimp']
data.loc[data['Variety'].isin(target)]


# Great! Now let's start answering the questions!
# 
# ## Which ramen has the most review?
# - The more reviews a brand has, the more people have tried it.
# - We won't clasify a brand as good based on the number of stars since any brand can easily make the list with just one 5* product.

# In[ ]:


top_reviews = data['Brand'].value_counts().head(10)
print(top_reviews)

# Using visualisation
sns.countplot(y="Brand", data=data, palette="Oranges_r",
              order=data.Brand.value_counts().iloc[:10].index)


# ### What do we see?
# ![](https://d1yjjnpx0p53s8.cloudfront.net/styles/logo-thumbnail/s3/0021/7500/brand.gif?itok=liupGSRX)
# ![](https://pngimage.net/wp-content/uploads/2018/06/nongshim-logo-png-4.png)
# ![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR3d_A7RlYmP2hGRtkcLbbIKs9kvyaPXkqYlw&usqp=CAU)
# 
# - The top reviewed brand is Nissin, followed by Nongshim and Maruchan.

# ## Country with the most brand
# - This could mean that the country has more brand which resulted in more reviews

# In[ ]:


ramen_coun = data.groupby('Country').agg({'Brand':'count'}).reset_index()
ramen_coun = ramen_coun.rename(columns = {'Brand':'Amount of brand'})
ramen_coun = ramen_coun.sort_values(['Amount of brand', 'Country'], ascending = [False, True])
print(ramen_coun)
# Visualising
sns.barplot(y="Country", x = 'Amount of brand', data=ramen_coun, palette="Blues_r",
              order=data.Country.value_counts().iloc[:10].index)


# ### What do we see here?
# - It seems like Japan is ranked top for this study, followed by USA and South Korea.

# ## The most popular style of ramen

# In[ ]:


top_style = data['Style'].value_counts()
top_style


# I was expecting both pack and cup to be pretty close in numbers but it seems that there are more packs in the market.

# ## Create word cloud of product name for the top 100 ramen

# In[ ]:


# Rank ramen by Stars column
ramen_sort = data.sort_values('Stars', ascending = False).dropna(subset = ['Stars'])

# Showing top 100 
ramen_top = ramen_sort.head(100)
ramen_top


# In[ ]:


# Join the top 100 ramen product name into a string
ramen_top_str = ramen_top['Variety'].str.cat(sep=',')

# For generate color
def color_func(word, font_size, position, orientation, random_state=None,                    **kwargs):
    return "hsl(%d, 100%%, 60%%)" % random.randint(20, 55)

# Plot word cloud of the top 100
stopword_list = ['Noodle', 'Noodles', 'Instant Noodle', 'Instant', 'Flavor', 'Flavour', 'Ramen', 'With']
plt.figure(figsize=(10,6))
top_wordcloud = WordCloud(max_font_size= 50, background_color='black',                       prefer_horizontal = 0.7, stopwords = stopword_list).generate(ramen_top_str)
plt.imshow(top_wordcloud.recolor(color_func = color_func, random_state = 3), interpolation='bilinear')
plt.axis('off')
plt.show()

