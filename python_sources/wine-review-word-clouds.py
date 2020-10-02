#!/usr/bin/env python
# coding: utf-8

# # Wine Review Word Clouds Separated by Country of Origin
# The inspiration for experimenting with python based word clouds on kaggle comes from looking at [this kernel](https://www.kaggle.com/tommichaels/kaggle-survey-word-cloud-for-beginners) by user [T Michaels](https://www.kaggle.com/tommichaels). I have used word cloud generators in the past but wanted to experiment with word clouds generated in python. 
# 
# I have uploaded shilouette images of the countries with the most reviewed wines that I use as a mask for generating the word clouds for a country's wine reviews in the shape of that country. 
# 
# Possible next steps would be to refine the stopwords to include common but not descriptive wine review words like "wine" or "palate" and possibly color each word cloud based on the country's flag colors, why not! (This has been added at the bottom, using  each country's flag as a mask and then coloring it.) Word clouds are ment to be informative and **fun**!
# 
# Hopefully you like this visualization kernel.
# 
# Images from <a href="https://www.freepik.com/free-vector/country-maps-collection_991910.htm">Freepik</a>, and <a href="http://d-maps.com/index.php?lang=en">D-Maps</a>.
# 
# ***
# *There's also a bonus tree map of the top countries sized by number of reviews*
# 
# 

# In[ ]:


# Typical imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Not so typical
import matplotlib.image as image
import matplotlib.colors
from collections import defaultdict
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
from IPython.display import Image as im
import squarify as sq


# ## The data
# Lets pull in the data and make a DataFrame showing the countries with the most wine reviews.

# In[ ]:


data = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv', index_col=0)
countries = data.country.value_counts()

# Limit top countries to those with more than 500 reviews
temp_dict = countries[countries>500].to_dict()
temp_dict['Other'] = countries[countries<501].sum()
less_countries = pd.Series(temp_dict)
less_countries.sort_values(ascending=False, inplace=True)

# Turn Series into DataFrame for display purposes
df = less_countries.to_frame()
df.columns=['Number of Reviews']
df.index.name = 'Country'
df


# ## Tree Map of the Top Countries

# In[ ]:


# New colors for tree map since base ones are bland
cmap = plt.cm.gist_rainbow_r
norm = matplotlib.colors.Normalize(vmin=0, vmax=15)
colors = [cmap(norm(value)) for value in range(15)]
np.random.shuffle(colors)

# Use squarify to plot the tree map with the custom colors
fig,ax = plt.subplots(1,1,figsize=(11,11))
sq.plot(sizes=less_countries.values, label=less_countries.index.values, alpha=0.5, ax=ax, color=colors)
plt.axis('off')
plt.title('Countries by Number of Wine Reviews')
plt.show()


# ## Cleaning the Reviews for Word Cloud Generation
# This section uses a defaultdict to store each wine review description as an item in a list indexed by the country as the dictionary key. AKA a dictionary of lists. The default dict just makes it easier since we don't have to check if a key exists for a given country before we append to it's list. This allows us to complete the data processing within a lambda function since no assignment is necessary to generate the dictionary

# In[ ]:


descriptions = defaultdict(list)
data.apply(lambda x: descriptions[x.country].append(x.description), axis=1)
descriptions['Italy'][0:5]


# Next up is tokenization. For each countries reviews we need to do the following:
# 1. Combine all the reviews into one string
# 2. Remove all the unwanted punctuation
# 3. Split the string into a list of words
# 4. Take each word greater than 3 characters, make it lower case, and store it in a list
# 5. Join this final list together
# 
# Very possible there is a more straightforward solution but this works relatively quickly for the number of reviews/words being processed.

# In[ ]:


unwanted_characters = re.compile('[^A-Za-z ]+')
for country in list(data.country.unique()):
    desc_string = ' '.join(descriptions[country])
    descriptions[country] = ' '.join([w.lower() for w in re.sub(unwanted_characters, ' ', desc_string).split() if len(w) > 3])


# In[ ]:


descriptions['Italy'][0:500]


# Great! Now we have a dictionary with each country's wine reviews tokenized and ready for word cloud generation.
# 
# ## Cloud Generation
# Let's define a function that takes a string of words, an image for masking the cloud, and an optional filename for saving the wordcloud to a file. It will return a wordcloud instance for us to work with/display in the notebook.

# In[ ]:


wine_stopwords = ['drink','wine','wines','flavor','flavors','note','notes','palate','finish','hint','hints','show','shows']
for w in wine_stopwords:
    STOPWORDS.add(w)


# In[ ]:


def generate_country_wordcloud(words, mask_image, filename=None, colormap='jet'):
    mask = np.array(Image.open(mask_image))
    wc = WordCloud(background_color="white", max_words=3000, mask=mask, stopwords=STOPWORDS, colormap=colormap)
    wc.generate(words)
    if filename:
        wc.to_file(filename)
    return wc


# In[ ]:


masks = dict()
masks['Argentina'] = '../input/outline-images-of-countries/argentina_bw_map.jpg'
masks['Australia'] = '../input/outline-images-of-countries/australia_bw_map.jpg'
masks['Austria'] = '../input/outline-images-of-countries/austria_bw_map.jpg'
masks['Chile'] = '../input/outline-images-of-countries/chile_bw_map.jpg'
masks['France'] = '../input/outline-images-of-countries/france_bw_map.jpg'
masks['Italy'] = '../input/outline-images-of-countries/italy_bw_map.jpg'
masks['Portugal'] = '../input/outline-images-of-countries/portugal_bw_map.jpg'
masks['Spain'] = '../input/outline-images-of-countries/spain_bw_map.jpg'
masks['US'] = '../input/outline-images-of-countries/usa_bw_map.jpg'
masks['Germany'] = '../input/outline-images-of-countries/germany_bw_map.jpg'
masks['Israel'] = '../input/outline-images-of-countries/israel_bw_map.jpg'
masks['New Zealand'] = '../input/outline-images-of-countries/newzealand_bw_map.jpg'
masks['South Africa'] = '../input/outline-images-of-countries/southafrica_bw_map.jpg'


# Time to generate the word clouds!

# # **United States**

# In[ ]:


us_wc = generate_country_wordcloud(descriptions['US'], masks['US'], 'US.jpg')
us_wc.to_image()


# # **France**

# In[ ]:


france_wc = generate_country_wordcloud(descriptions['France'], masks['France'], 'France.jpg')
france_wc.to_image()


# # **Italy**

# In[ ]:


italy_wc = generate_country_wordcloud(descriptions['Italy'], masks['Italy'], 'Italy.jpg')
italy_wc.to_image()


# # **Spain**

# In[ ]:


spain_wc = generate_country_wordcloud(descriptions['Spain'], masks['Spain'], 'Spain.jpg')
spain_wc.to_image()


# # **Portugal**

# In[ ]:


portugal_wc = generate_country_wordcloud(descriptions['Portugal'], masks['Portugal'], 'Portugal.jpg')
portugal_wc.to_image()


# # **Chile**

# In[ ]:


chile_wc = generate_country_wordcloud(descriptions['Chile'], masks['Chile'], 'Chile.jpg')
chile_wc.to_image()


# # **Argentina**

# In[ ]:


argentina_wc = generate_country_wordcloud(descriptions['Argentina'], masks['Argentina'], 'Argentina.jpg')
argentina_wc.to_image()


# # **Austria**

# In[ ]:


austria_wc = generate_country_wordcloud(descriptions['Austria'], masks['Austria'], 'Austria.jpg')
austria_wc.to_image()


# # **Australia**

# In[ ]:


australia_wc = generate_country_wordcloud(descriptions['Australia'], masks['Australia'], 'Australia.jpg')
australia_wc.to_image()


# # **Germany**

# In[ ]:


germany_wc = generate_country_wordcloud(descriptions['Germany'], masks['Germany'], 'Germany.jpg')
germany_wc.to_image()


# # **New Zealand**

# In[ ]:


nz_wc = generate_country_wordcloud(descriptions['New Zealand'], masks['New Zealand'], 'NewZealand.jpg')
nz_wc.to_image()


# # **South Africa**

# In[ ]:


south_africa_wc = generate_country_wordcloud(descriptions['South Africa'], masks['South Africa'], 'SouthAfrica.jpg')
south_africa_wc.to_image()


# # **Israel**

# In[ ]:


israel_wc = generate_country_wordcloud(descriptions['Israel'], masks['Israel'], 'Israel.jpg')
israel_wc.to_image()


# # Flag Word Clouds

# In[ ]:


flags = dict()
flags['Argentina'] = '../input/world-flags/Argentina.png'
flags['Australia'] = '../input/world-flags/Australia.png'
flags['Austria'] = '../input/world-flags/Austria.png'
flags['Chile'] = '../input/world-flags/Chile.png'
flags['France'] = '../input/world-flags/France.png'
flags['Italy'] = '../input/world-flags/Italy.png'
flags['Portugal'] = '../input/world-flags/Portugal.png'
flags['Spain'] = '../input/world-flags/Spain.png'
flags['US'] = '../input/world-flags/United_States.png'
flags['Germany'] = '../input/world-flags/Germany.png'
flags['Israel'] = '../input/world-flags/Israel.png'
flags['New Zealand'] = '../input/world-flags/New_Zealand.png'
flags['South Africa'] = '../input/world-flags/South_Africa.png'


# In[ ]:


# Developed to see what the flag masks would look like without blocking out the white areas of the flags.
# Turns out its not as nice IMO but I will leave it here for others to mess with.
def replace_color(img_data, old_color, new_color):
    
    r1, g1, b1 = old_color # Original value
    r2, g2, b2 = new_color # Value that we want to replace it with

    red, green, blue = img_data[:,:,0], img_data[:,:,1], img_data[:,:,2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    img_data[:,:,:3][mask] = [r2, g2, b2]
    
    return img_data


# In[ ]:


def generate_flag_wordcloud(words, flag_image, filename=None):
    mask = np.array(Image.open(flag_image))
    #mask = replace_color(mask, (255, 255, 255), (200, 200, 200))
    wc = WordCloud(background_color="white", max_words=3000, mask=mask, stopwords=STOPWORDS, 
                   max_font_size=50, random_state=42)
    wc.generate(words)
    image_colors = ImageColorGenerator(mask)
    wc.recolor(color_func=image_colors)
    if filename:
        wc.to_file(filename)
    return wc


# # **United States**

# In[ ]:


us_wc_flag = generate_flag_wordcloud(descriptions['US'], flags['US'], 'US_flag.jpg')
us_wc_flag.to_image()


# # **France**

# In[ ]:


france_wc_flag = generate_flag_wordcloud(descriptions['France'], flags['France'], 'France_flag.jpg')
france_wc_flag.to_image()


# # **Italy**

# In[ ]:


italy_wc_flag = generate_flag_wordcloud(descriptions['Italy'], flags['Italy'], 'Italy_flag.jpg')
italy_wc_flag.to_image()


# # **Spain**

# In[ ]:


spain_wc_flag = generate_flag_wordcloud(descriptions['Spain'], flags['Spain'], 'Spain_flag.jpg')
spain_wc_flag.to_image()


# # **Portugal**

# In[ ]:


portugal_wc_flag = generate_flag_wordcloud(descriptions['Portugal'], flags['Portugal'], 'Portugal_flag.jpg')
portugal_wc_flag.to_image()


# # **Chile**

# In[ ]:


chile_wc_flag = generate_flag_wordcloud(descriptions['Chile'], flags['Chile'], 'Chile_flag.jpg')
chile_wc_flag.to_image()


# # **Argentina**

# In[ ]:


argentina_wc_flag = generate_flag_wordcloud(descriptions['Argentina'], flags['Argentina'], 'Argentina_flag.jpg')
argentina_wc_flag.to_image()


# # **Austria**

# In[ ]:


austria_wc_flag = generate_flag_wordcloud(descriptions['Austria'], flags['Austria'], 'Austria_flag.jpg')
austria_wc_flag.to_image()


# # **Australia**

# In[ ]:


australia_wc_flag = generate_flag_wordcloud(descriptions['Australia'], flags['Australia'], 'Australia_flag.jpg')
australia_wc_flag.to_image()


# # **Germany**

# In[ ]:


germany_wc_flag = generate_flag_wordcloud(descriptions['Germany'], flags['Germany'], 'Germany_flag.jpg')
germany_wc_flag.to_image()


# # **New Zealand**

# In[ ]:


nz_wc_flag = generate_flag_wordcloud(descriptions['New Zealand'], flags['New Zealand'], 'NewZealand_flag.jpg')
nz_wc_flag.to_image()


# # **South Africa**

# In[ ]:


south_africa_wc_flag = generate_flag_wordcloud(descriptions['South Africa'], flags['South Africa'], 'SouthAfrica_flag.jpg')
south_africa_wc_flag.to_image()


# # **Israel**

# In[ ]:


israel_wc_flag = generate_flag_wordcloud(descriptions['Israel'], flags['Israel'], 'Israel_flag.jpg')
israel_wc_flag.to_image()


# In[ ]:




