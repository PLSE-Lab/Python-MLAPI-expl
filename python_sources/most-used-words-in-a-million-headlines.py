#!/usr/bin/env python
# coding: utf-8

# I used the abc news dataset to analyze the most used words. I was unable to use some of the headlines but I don't think the amount would have changed the result. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from collections import Counter
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
data = pd.read_csv("../input/abcnews-date-text.csv",error_bad_lines=False,usecols =["headline_text"])
cnt = Counter()
data_new = data[data['headline_text'].notnull()]
for x in data_new["headline_text"]:
   items = x.split(" ")
   for y in items:
       cnt[y]+=1
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
ind = np.arange(15)  # the x locations for the groups
width = 0.35       # the width of the bars
first10keys = [k for k in sorted(cnt,key = cnt.get,reverse = True)[:15]]
first10vals = [k for k in sorted(cnt.values(),reverse = True)[:15]]
rects1 = ax.bar(ind, first10vals, width,
               color='black')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(first10keys)
ax.set_title("Untouched Results")
plt.show()
#Filtering stop words
unwanted = ("a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the");
for k in unwanted:
   del cnt[k]

fig2 = plt.figure(figsize=(11,8))
ax2 = fig2.add_subplot(111)
first15keys2 = [k for k in sorted(cnt,key = cnt.get,reverse = True)[:15]]
first15vals2 = [k for k in sorted(cnt.values(),reverse = True)[:15]]
rects2 = ax2.bar(ind, first15vals2, width,
               color='black')
ax2.set_xticks(ind + width / 2)
ax2.set_xticklabels(first15keys2)
ax2.set_title("After removing stop words")
plt.show()

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

