#!/usr/bin/env python
# coding: utf-8

# This Kernel is an analysis of JJR Tolkien's 'Lord of the rings' (all 6 books) with use of Natural Language Processing library TextBlob.
# It will answer how much every realm and race has been mentioned and what is the writing style of the author by analysis of most often used words and sentiment analysis of all sentences in a book.
# As a source data I use .txt of the book.

# In[ ]:


from textblob import TextBlob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
sns.set()

with open('../input/lotr.txt', 'r', errors='ignore') as file:
    data = file.read().replace('\n', '')

text= TextBlob(data)
my_list=text.tags


# At the beggining let's compare how much each realm was mentioned:

# In[ ]:


Eriador = data.count("Eriador")
Arnor = data.count("Arnor")
Rohan = data.count("Rohan")
Gondor = data.count("Gondor")
Mordor = data.count("Mordor")
Rhun = data.count("Rhun")
realms_list = [['Gondor', Gondor], ['Mordor', Mordor], ['Rohan', Rohan], ['Arnor', Arnor], ['Eriador', Eriador]]
df_realms=pd.DataFrame(realms_list, columns=['Realm', 'Times mentioned'])
colors = ["crimson", "forrest green", "true blue", "amber", "black"]
sns.set_style("darkgrid")
plt.figure(figsize=(10, 5))
with sns.xkcd_palette(colors):
    sns.barplot(x="Realm", y="Times mentioned", saturation=0.9, data=df_realms).set_title("Lord of the Rings - number of times realms being mentioned")


# I came up with an idea to compare how often each race has been mentioned:

# In[ ]:


Orc = data.count("Orc")+data.count("Orcs")+data.count("orc")+data.count("orcs")+data.count("orcish")
Human = data.count("Man")+data.count("Mankind")+data.count("Men")+data.count("men")+data.count("human")
Elf = data.count("Elf")+data.count("Elves")+data.count("elf")+data.count("elves")+data.count("elven")
Dwarf = data.count("Dwarf")+data.count("Dwarves")+data.count("dwarf")+data.count("dwarves")+data.count("dwarven")
Halfling = data.count("Halfling")+data.count("Hobbit")+data.count("Hobbits")+data.count("Halflings")+data.count("halfling")+data.count("hobbit")+data.count("hobbits")+data.count("halflings")
Ent = data.count("Ents")+data.count("Ent")
Troll = data.count("Troll")+data.count("troll")+data.count("Trolls")+data.count("trolls")
Dragon = data.count("dragon")+data.count("dragons")+data.count("Dragon")+data.count("Dragons")
Balrog = data.count("Balrog")+data.count("Balrogs")+data.count("balrog")+data.count("balrogs")
Goblin = data.count("Goblin")+data.count("Goblins")+data.count("goblin")+data.count("goblins")
Warg = data.count("Warg")+data.count("Wargs")+data.count("warg")+data.count("wargs")
Huorn = data.count("Huorn")+data.count("Huorns")+data.count("huorn")+data.count("huorns")
Beorning = data.count("Beorning")+data.count("Beornings")+data.count("beorning")+data.count("beornings")+data.count("Skin-changers")+data.count("Skin-changer")+data.count("skin-changer")+data.count("skin-changers")
races_list = [['Men', Human], ['Hobbits/Halflings', Halfling], ['Elves', Elf], ['Orcs', Orc], ['Dwarves', Dwarf],  ['Goblins', Goblin], ['Ents', Ent], ["Dragons", Dragon], ['Trolls', Troll], ["Wargs", Warg], ['Huorns', Huorn], ["Balrogs", Balrog], ["Beornings", Beorning]]
df_races=pd.DataFrame(races_list, columns=['Race', 'Times mentioned'])
colors = ["amber", "brown", "dark sea green", "forrest green", "crimson", "black",  "brown", "true blue", "black", "forrest green", "brown", "crimson", "blue"]
sns.set_style("darkgrid")
plt.figure(figsize=(15, 7))
with sns.xkcd_palette(colors):
    sns.barplot(x="Race", y="Times mentioned", saturation=0.9, data=df_races).set_title("Lord of the Rings - number of times races being mentioned")


# As visible above Elves has been mentioned a lot more often than Dwarves. Hobbit's are on the high second position (let me know if I have missed any race)

# This brief part will take you through analysis of most often used words.Let's start with checking the top20 most word types and present them on a barplot:

# In[ ]:


full_t = pd.DataFrame(my_list)
full_t.columns = ['Words', "Word type"]
xft=full_t.groupby('Word type').count().reset_index()
top20ft=xft.nlargest(20, 'Words')

sns.set_style("darkgrid")
plt.figure(figsize=(10, 5))
sns.barplot(x="Words", y="Word type", palette="rocket", saturation=0.9, data=top20ft).set_title("Lord of the Rings - top 20 word types used")


# Below I declare a function that will be used to create a top10 most used words of selected type:

# In[ ]:


def word_analysis(word_type):
    filtered = [row for row in my_list if str(word_type) in row[1]]
    print("filtered for " + word_type)
    df = pd.DataFrame(filtered)
    df.columns = ["Word", "Occurences"]
    x=df.groupby('Word').count().reset_index()
    y=x.sort_values(by=['Occurences'], ascending=False)
    top10=y.nlargest(10, 'Occurences')
    plt.figure(figsize=(10, 5))
    sns.barplot(x="Word", y="Occurences", palette="rocket", saturation=0.9, data=top10).set_title("Lord of the rings - most frequently used "+ word_type +" type word")


# Lets find most commonly used nouns:

# In[ ]:


word_type = 'NN'
word_analysis(word_type)


# Okay, lets check it for Proper nouns only (nouns starting with a capital letter):

# In[ ]:


word_type = 'NNP'
word_analysis(word_type)


# It's worth to see adjective analysis as well:

# In[ ]:


word_type = 'JJ'
word_analysis(word_type)


# Below you may find top10 verbs:

# In[ ]:


word_type = 'VB'
word_analysis(word_type)


# Okay, lets start with heavylifting: sentiment analysis.

# In[ ]:


sentiment=[]
x=0

for sentence in text:
    text.sentiment

for sentence in text.sentences:
    sentiment.append(sentence.sentiment)
    
sentence_df = pd.DataFrame(sentiment)
sentence_df.describe()


# Above we may see a brief stats of book done with use of Pandas pandas.DataFrame.describe() function. I will swiftly point out most important elements of it below.
# 
# Book contains 41'167 sentences in total(in comparison, Harry Potter and Philosophers' stone contains 6396 sentences). All senteces has been marked with polarity score ranging from -1.0 to +1.0 and subjectivity score ranging from 0.0 to 1.0. Polarity score mean of +0.036 (to compare Harry Potter part 1. has a score +0.027) is considered a rather neutral. Subjectivity score mean of +0.285 is slightly subjective (close to Harry Potter p.1).

# In[ ]:


sentence_df['order'] = sentence_df.index
sentence_df = pd.DataFrame(sentiment)
sentence_df['order'] = sentence_df.index
polarity = pd.Series(sentence_df["polarity"])
plt.figure(figsize=(10, 10))
sns.jointplot("order", "polarity", data=sentence_df, kind="kde")


# As we can see on the jointplot - the most of sentences has been classied as neutral. Jointplot shows small clusters at ranges of -0.25 to +0.5 troughtout the book. Lets remove the neutral values to perform deeper analysis of non-neutral sentences.

# In[ ]:


plt.figure(figsize=(15, 5))
sns.jointplot("order", "polarity", data=sentence_df[sentence_df.polarity != 0], kind="kde")


# Firstly I need to mention that above jointplot shows a really small part of total sentences we should assume Tolkien's style is heavily based on neutral polarity sentences. I can see that sentences polarity very rarely goes down below -0.5. Worth mentioning is a significant amount of sligtly positive  sentences around sentences 3000-4000. 

# In[ ]:


subjectivity = pd.Series(sentence_df["subjectivity"])
plt.figure(figsize=(10, 10))
sns.jointplot("order", "subjectivity", data=sentence_df, kind="kde")


# As present on jointplot the subjectivity score of 0.0 is clearly dominating. Lets have a closer look to a jointplot without a 0.0 values:

# In[ ]:


sns.jointplot("order", "subjectivity", data=sentence_df[sentence_df.subjectivity != 0], kind="kde")


# As We may observe: throughout the book there is a sgnificant amount of 0.4 and 0.5 subjectivity sentences.

# Lets take a look at the polarity/subjectivity jointplot to analyse the corelation between polarity and subjectivity of the sentences:

# In[ ]:


plt.figure(figsize=(10, 10))
sns.jointplot("polarity", "subjectivity", data=sentence_df[(sentence_df.subjectivity != 0)], kind="kde")


# As in Harry Potter's part.1 analysis we may observe that downwards facing triangle shape. It means that the more subjective the sentence is the more polarized it may be: we can deduct it from the shape of kde.

# Hey, if you are reading this please let me know what are your ideas to go further into the analysis and what is your feedback :)
