#!/usr/bin/env python
# coding: utf-8

# This is my first kernel on Kaggle.
# 
# It is still a WIP and I'm busy adding more visualizations. Suggestions are more than welcome :)
# 
# Completed:
# * Finding all artists' favorite words
# * Wordcloud based on the entire lyrics dataset 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


from wordcloud import WordCloud
from pprint import pprint
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import numpy as np

from nltk.corpus import stopwords

stopwords = stopwords.words('english')
stopwords.append("verse")
stopwords.append("chorus")
stopwords.append("choru")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dat = pd.read_csv("../input/songdata.csv")

# defining a couple of helper functions
def clean(sting):
    char_to_rem = ["\n", "'", ",", "]", "[", ")", "("]

    for c in char_to_rem:
        sting = sting.replace(c, "")

    final_sting = []

    for word in sting.split(' '):
        word = word.lower()
        if word == "fag" or word == "ho" or word == "hoe" or word == "ass":
            final_sting.append(word)
            continue
            
        if len(word) > 3 and word not in stopwords:
            final_sting.append(word)

    return final_sting


def update(dic1, dic2):
    for key, value in dic2.items():
        if key in dic1:
            dic1[key] = dic1[key] + dic2[key]
        else:
            dic1[key] = dic2[key]
            


# Starting with evaluating the top 5 words for every artist along with their frequencies

# In[ ]:


# Starting with evaluating the top 5 words for every artist along with their frequencies

grouped_by_artist = dat.groupby('artist')

# saving the total words in this dict
# total number of songs
ar_di = {}
tot_words = {}
tot_words_list = []

artist_strings = {}

for artist_name, songs in grouped_by_artist:
    num_total_words = 0
    num_songs = 0
    artist_string = []
    
    words = {}

    for index, rows in songs.iterrows():
        num_songs += 1
        clean_text_list = clean(rows["text"])
        num_total_words += len(clean_text_list)

        tot_words_list += clean_text_list
        artist_string += clean_text_list
        
        for word in clean_text_list:
            if word in words:
                words[word] = words[word] + 1
            else:
                words[word] = 1

        update(tot_words, words)
        artist_strings[artist_name] = list(artist_string)
        
    print ("Talkin 'bout ", artist_name)
    print ("Total words in all songs", num_total_words)
    
    for key, val in sorted(words.items(), key=lambda tup: (tup[1], tup[0]), reverse=True)[:5]:
        print ("\t", key, "used", val, "times")
    
    print ("\n\n")


# In[ ]:


print("Now we'll try to make a word cloud out of this")
text = " ".join(tot_words_list)

print("Got the string")

import random

def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)

wc = WordCloud(max_words=1000, background_color="white").generate(text)

plt.figure(figsize=(9, 6))
plt.axis("off")
plt.imshow(wc)
# plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))

plt.savefig("wc_completelyrics.png")


# Don't know, cause well, Life!
# 
# It's really amazing to see what wordclouds bring out with text datasets :)

# Now, working on the top cuss words in the music industry

# In[ ]:


cuss_words = ["fuck", "fag", "dick", "tits", "pussy", "ho", "ass", "n-word", "shit", "cock", "bitch", "cunt"]

# preprocessing to update some counts
tot_words["n-word"] = tot_words["nigger"] + tot_words["nigga"]
del tot_words["nigger"]
del tot_words["nigga"]

tot_words["ho"] = tot_words["ho"] + tot_words["hoe"]
del tot_words["hoe"]

counts_cuss_words = [tot_words[x] for x in cuss_words]


# In[ ]:


fig = plt.figure(figsize=(9, 6))

cuss_series = pd.Series.from_array(counts_cuss_words)

# plt.bar(np.arange(len(cuss_words)), counts_cuss_words, color="grey")
ax = cuss_series.plot(kind='bar')
# ax.set_title("Amount Frequency")
# ax.set_xlabel("Amount ($)")
# ax.set_ylabel("Frequency")
ax.set_xticklabels(cuss_words)
# ax.xaxis.set_visible(False)

# plt.xticks(cuss_words)
plt.show()


plt.savefig("bar_cuss_words.png")


# Now, I'll try to find out the artist's who swear the most!

# In[ ]:


artist_cuss = []

for artist in artist_strings.keys():
    counter = 0
    
    for sting in artist_strings[artist]:
        if sting in cuss_words:
            counter += 1
     
    artist_cuss.append((counter, artist))


# In[ ]:


sorted(artist_cuss, reverse=True)[:50]


# If you look carefully, you'd note an artist called [**Lata Mangeshkar**](https://en.wikipedia.org/wiki/Lata_Mangeshkar) who is a famous Bollywood singer from the 70's and I've absolutely no idea about how she landed up in the top cussers list.
# 
# Let's find out!

# In[ ]:


# I notice Lata Mangeshkar amongst the top cuss'ers (if you may)
# that's strange

artist = "Lata Mangeshkar"

lata = {}
for word in cuss_words:
    lata[word] = 0

for sting in artist_strings[artist]:
    if sting in cuss_words:
        lata[sting] += 1


pprint(lata)


# And there you go, being a Bollywood singer, almost all her songs have been sung in Hindi and the word "ho" in Hindi roughly translates to "to exist" in English is apparently very frequent. 
# 
# Hence the discrepancy!

# In[ ]:




