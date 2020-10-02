#!/usr/bin/env python
# coding: utf-8

# # Main Code Used to Analyze Word Embeddings
# * Point of contact: madeleinecheyette@gmail.com
# * Code largely adopted from academic paper,'Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings,' github: https://github.com/tolga-b/debiaswe, input here '../input/DebiasweStanford'
# * Input further detailed below
# * Output: for a given input embedding file and bias direction, outputs word lists and pie charts of the most commonly associated professions, personality words, and family words related to that bias. Computes the overall bias of the word lists and Sentinet scores of the personality words
# 
# To run the code with various embeddings, change the input as enumerated below. 
# 

# * Input: 
#     * Embeddings generated with news data from https://www.kaggle.com/snapcrack/all-the-news 
#         * Data for aggregated news files: '../input/combined-nondivided'
#         * Data for single news files: '../input/make-embeddings'
#         * Data for aggregated news files split before and after the election: '../input/combined-embeddings'
#     * Embeddings generated from Cornell Movie Dialogue Corpus (https://www.kaggle.com/Cornell-University/movie-dialog-corpus) in '../input/star-wars-analysis'
#     * Neurtral word lists:
#         * personalities.txt
#         * family words in categories '../input/family-div/family.txt'
#         * professions categorized by salary and industry '../input/professions_sal_ind'
#     * Definitional words for PCA analysis: '../input/definitions-gender' and '../input/lgbtq'

# In[ ]:


import numpy as np 
import pandas as pd 
from subprocess import check_output
from gensim.models import KeyedVectors
from nltk.corpus import sentiwordnet 
from nltk.corpus import sentiwordnet 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import os
import re 
import multiprocessing
import spacy
import random
from collections import Counter 
from nltk.corpus import sentiwordnet as swn
import seaborn as sns
get_ipython().run_line_magic('env', 'PYTHONHASHSEED=0')


import os
print(os.listdir("../input"))


# Navigate to directory with https://github.com/tolga-b/debiaswe

# In[ ]:


os.chdir("../input/debiaswestanford/repository/tolga-b-debiaswe-10277b2")
print("CORES: %d" % multiprocessing.cpu_count())
print(check_output(["ls"]).decode("utf8"))
os.chdir("debiaswe")
print(check_output(["ls"]).decode("utf8"))


# In[ ]:


from shutil import copyfile
from data import *
from we import *


# **Feed in input embedding file- change as needed**
# 
# In this example, we use an aggregated embedding from democratic articles

# In[ ]:


GE = WordEmbedding('../../../../combined-nondivided/d5_model.bin')


# **Specify Bias Direction-- Choose one as needed **

# Option 1: Specify Bias Direction Between Two Specific Words

# In[ ]:


bias_direction = GE.diff('homosexual', 'heterosexual')


# Option 2: Use PCA to specify Bias Direction

# In[ ]:


# lgbtq direction 
print(check_output(["ls"]).decode("utf8"))
with open('../../../../definitionslgbtq/definitionsLGBTQ.json', "r") as f:
    defs = json.load(f)
    print("definitional", defs)

bias_direction = doPCA(defs, GE, 5).components_[0]   


# In[ ]:


# gender direction 
print(check_output(["ls"]).decode("utf8"))
defs = [['woman', 'man'], ['girl', 'boy'], ['she', 'he'], ['mother', 'father'], ['daughter', 'son'] ,['her', 'his']]
bias_direction = doPCA(defs, GE, 9).components_[0]   


# # Profession Analysis

# Method for dividing professions based on industry and salary level

# In[ ]:


def load_prof():
    salary_file = os.path.join(PKG_DIR, '../../../../professions-sal-ind', 'professions_salary.txt')
    industry_file = os.path.join(PKG_DIR, '../../../../professions-sal-ind', 'profession_industry.txt')
    with open(industry_file, 'r') as f:
        industry_words = f.read().splitlines()
    industry_val = []
    # separate profession + industry association
    for i in range(len(industry_words)): 
        w_val = industry_words[i].split(" ")
        tup = w_val[0], w_val[1]
        industry_words[i] = tup
    # separate profession + salary association
    with open(salary_file, 'r') as f:
        salary_words = f.read().splitlines()
    for i in range(len(salary_words)): 
        s_val = salary_words[i].split(" ")
        s_val[1] = s_val[1].replace(',', '')
        for each in industry_words:
            if s_val[0] in each:
                tup_i = each
                tup = s_val[0], int(s_val[1]), tup_i[1]
                salary_words[i] = tup
            else: 
                continue
        if tup == 'null':
            tup = s_val[0], int(s_val[1])
            salary_words[i] = tup
    return salary_words


# In[ ]:



professions = load_prof()

# sort by profession cosine similarity to bias direction
profession_rank = [];
for i in range(len(professions)): 
    w = professions[i]
    try: 
         profession_rank += ([(GE.v(w[0]).dot(bias_direction), w[0],w[1], w[2])])
    except: 
        continue
sp = sorted(profession_rank, key=lambda x: x[0])

# find professions closely related to each end of the bias direction
print("Words close to each side of bias direction, ordered by cosine similarity: ")
diff2 = []
for i in range(0, 20): 
    diff2.append(sp[i])
    print(sp[i])
print("")

diff1 = []
for i in range(-20,0): 
    diff1.append(sp[i])
    print(sp[i])

#calculate bias metric
sp_m = []
for i in range(0, len(sp)):
    tup = sp[i]
    sp_m.append(abs(tup[0]))
metric = sum(sp_m)
print("Bias metric: ", (1/len(sp_m)) * metric)


# Prepare data for pie chart visualizations- count which associations are associated with each industry and salary level

# In[ ]:


art =[] 
service=[]
activism=[]
sports=[]
business=[]
education=[]
science=[]
medicine=[]
media =[]
law_enformcent =[]
government=[]
technology=[]
other = []
religion = []

for i in diff2: 
    if(i[3] =='arts'):
        art += i
    elif(i[3] =='service'):
        service += i
    elif(i[3] =='activism'):
        activism += i
    elif(i[3] =='sports'):
        sports += i
    elif(i[3] =='business'):
        business += i
    elif(i[3] =='education'):
        education += i
    elif(i[3] =='science'):
        science += i
    elif(i[3] =='medicine'):
        medicine += i
    elif(i[3] =='media'):
        media += i
    elif(i[3] =='law_enforce'):
        law_enformcent += i
    elif(i[3] =='government'):
        government+= i
    elif(i[3] =='technology'):
        technology+= i
    elif(i[3] =='other'):
        other+= i
    elif(i[3] =='religion'):
        religion += i


# In[ ]:


industryCount = {'Industry':['art', 'service', 'activism', 'sports', 'business', 'education', 'science', 'medicine', 'media', 'law_enformcent', 'government', 'technology', 'religion', 'other'],
             'Count':[len(art), len(service), len(activism), len(sports), len(business), len(education), len(science), len(medicine), len(media), len(law_enformcent), len(government), len(technology), len(religion), len(other)]}
df = pd.DataFrame(industryCount, columns=['Industry','Count'])
explode = (0.18,0.18, 0.18,0.18,0.18, 0.18,0.18,0.18, 0.18,0.18,0.18, 0.18,0.18,0.18)
colors = ['fuchsia', 'firebrick', 'lawngreen', 'darkblue', 'forestgreen', 'gold', 'darkmagenta', 'teal', 'lightcoral', 'lightpink', 'c', 'darkorange', 'dodgerblue', 'crimson', 'orchid']
patches, texts, autotexts = plt.pie(df['Count'], labels = df['Industry'], colors = colors,
       autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)

#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

for autotext in autotexts:
    autotext.set_color('white')

plt.title("Identity 1-Associated Industry Levels", y=1.08)
plt.tight_layout()


# In[ ]:


art =[] 
service=[]
activism=[]
sports=[]
business=[]
education=[]
science=[]
medicine=[]
media =[]
law_enformcent =[]
government=[]
technology=[]
other = []
religion = []
for i in diff1: 
    if(i[3] =='arts'):
        art += i
    elif(i[3] =='service'):
        service += i
    elif(i[3] =='activism'):
        activism += i
    elif(i[3] =='sports'):
        sports += i
    elif(i[3] =='business'):
        business += i
    elif(i[3] =='education'):
        education += i
    elif(i[3] =='science'):
        science += i
    elif(i[3] =='medicine'):
        medicine += i
    elif(i[3] =='media'):
        media += i
    elif(i[3] =='law_enforce'):
        law_enformcent += i
    elif(i[3] =='government'):
        government+= i
    elif(i[3] =='technology'):
        technology+= i
    elif(i[3] =='other'):
        other+= i
    elif(i[3] =='religion'):
        religion += i


# In[ ]:


industryCount = {'Industry':['art', 'service', 'activism', 'sports', 'business', 'education', 'science', 'medicine', 'media', 'law_enformcent', 'government', 'technology', 'religion', 'other'],
             'Count':[len(art), len(service), len(activism), len(sports), len(business), len(education), len(science), len(medicine), len(media), len(law_enformcent), len(government), len(technology), len(religion), len(other)]}
df = pd.DataFrame(industryCount, columns=['Industry','Count'])
explode = (0.18,0.18, 0.18,0.18,0.18, 0.18,0.18,0.18, 0.18,0.18,0.18, 0.18,0.18,0.18)
colors = ['fuchsia', 'firebrick', 'lawngreen', 'darkblue', 'forestgreen', 'gold', 'darkmagenta', 'teal', 'lightcoral', 'lightpink', 'c', 'darkorange', 'dodgerblue', 'crimson', 'orchid']
patches, texts, autotexts =plt.pie(df['Count'], labels = df['Industry'], colors = colors,
       autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)

#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

for autotext in autotexts:
    autotext.set_color('white')

plt.title("Identity 2-Associated Industry Levels")
plt.tight_layout()


# In[ ]:


six_fig =[]
high_income = []
upper_mid = []
mid = []
entry = []
for i in diff2: 
    if(i[2] > 100000):
        six_fig += i
    elif(i[2] > 80000):
        high_income += i
    elif(i[2] > 50000):
        upper_mid += i
    elif(i[2] > 30000):
        mid += i
    elif(i[2] > 10000):
        entry += i


# In[ ]:


industryCount = {'Income Level':['Six', 'High Income', 'Upper Middle', 'Middle', 'Entry'],
             'Count':[len(six_fig), len(high_income), len(upper_mid), len(mid), len(entry)]}
df = pd.DataFrame(industryCount, columns=['Income Level','Count'])
explode = (0.05,0.05,0.05,0.05, 0.05)
patches, texts, autotexts = plt.pie(df['Count'], labels = df['Income Level'], autopct='%1.1f%%', 
        startangle=90, pctdistance=0.85, explode = explode)

#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

for autotext in autotexts:
    autotext.set_color('white')
    
plt.title("Identity 1-Associated Income Levels")
plt.tight_layout()


# In[ ]:


six_fig =[]
high_income = []
upper_mid = []
mid = []
entry = []
for i in diff1: 
    if(i[2] > 100000):
        six_fig += i
    elif(i[2] > 80000):
        high_income += i
    elif(i[2] > 50000):
        upper_mid += i
    elif(i[2] > 30000):
        mid += i
    elif(i[2] > 10000):
        entry += i


# In[ ]:


industryCount = {'Income Level':['Six', 'High Income', 'Upper Middle', 'Middle', 'Entry'],
             'Count':[len(six_fig), len(high_income), len(upper_mid), len(mid), len(entry)]}
df = pd.DataFrame(industryCount, columns=['Income Level','Count'])
explode = (0.05,0.05,0.05,0.05, 0.05)
patches, texts, autotexts = plt.pie(df['Count'], labels = df['Income Level'], autopct='%1.1f%%', 
        startangle=90, pctdistance=0.85, explode = explode)

#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

for autotext in autotexts:
    autotext.set_color('white')
    
plt.title("Identity 2-Associated Income Levels")
plt.tight_layout()


# # Personality Analysis

# In[ ]:


def load_personalities():
    personalities_file = os.path.join(PKG_DIR, '../../../../personalitiestxt', 'personalities.txt')
    with open(personalities_file, 'r') as f:
        personalities = f.read().splitlines()
    print('Loaded personalities\n')
    return personalities


# In[ ]:


# load and sort personality words based on cosine similarity
personalities_words = load_personalities()
print(len(personalities_words))
personalities_rank = [];
for w in personalities_words: 
    try: 
        personalities_rank += [(GE.v(w).dot(bias_direction), w)]
    except: continue
sp = sorted(personalities_rank)

# print word associations close to either end of the direction
print("Words close to each side of bias direction, ordered by cosine similarity: ")
for i in range(0,20): 
    print(sp[i])
print("")
for i in range(-20,0): 
    print(sp[i])

# Calculate bias metric
sp_m = []
for i in range(0, len(sp)):
    tup = sp[i]
    sp_m.append(abs(tup[0]))
metric = sum(sp_m)
print("Bias metric: ", 1/len(sp_m) * metric)


# **Sentinet Analysis**

# In[ ]:


from nltk.corpus import sentiwordnet as swn

# Caclculate positive and negative sentiment 
posscore=0
negscore=0
for i in range(0,20): 
    tup = sp[i]
    good = swn.senti_synsets(tup[1], 'a')
    for synst in good:
        posscore+=synst.pos_score()
        negscore+=synst.neg_score()
print("diff2 positive score", posscore)
print("diff2 negative score", negscore)
print("diff2 overall score", posscore-negscore)
posscore=0
negscore=0
print()
for i in range(-20,0): 
    tup = sp[i]
    good = swn.senti_synsets(tup[1], 'a')
    for synst in good:
        posscore+=synst.pos_score()
        negscore+=synst.neg_score()
print("diff1 positive score", posscore)
print("diff1 negative score", negscore)
print("diff1 overall score", posscore-negscore)


# # Family Words 

# In[ ]:


def load_family_words():
    family_file = os.path.join(PKG_DIR, '../../../../familydiv/', 'family.txt')
    with open(family_file, 'r') as f:
        family_words = f.read().splitlines()
    for i in range(len(family_words)): 
        w_val = family_words[i].split(" ")
        tup = w_val[0], w_val[1]
        family_words[i] = tup
    return family_words


# In[ ]:


# load and sort family words by cosine similarity
family_words = load_family_words()
family_rank = [];
for i in range(len(family_words)): 
    w = family_words[i]
    try: 
        family_rank += [(GE.v(w[0]).dot(bias_direction), w[0], w[1])]
    except: continue
sp = sorted(family_rank, key=lambda x: x[0])

# print word associations close to either end of the direction
print("Words close to each side of bias direction, ordered by cosine similarity: ")
diff2 = []
for i in range(0,20): 
    diff2.append(sp[i])
    print(sp[i])
print("")
diff1 = []
for i in range(-20,0): 
    diff1.append(sp[i])
    print(sp[i])

# calculate bais metric
sp_m = []
for i in range(0, len(sp)):
    tup = sp[i]
    sp_m.append(abs(tup[0]))
metric = sum(sp_m)
print("Bias metric: ", 1/len(sp_m) * metric)


# Prepare data for pie chart visualizations- count number of associations in each category for each end of the bias direction

# In[ ]:


men =[]
dow = []
div = []
cb = []
parenting = []
lin = []
rela = []
other = []
for i in diff2: 
    if(i[2] == 'men'):
        men += i
    elif(i[2] == 'dow'):
        dow += i
    elif(i[2] == 'div'):
        upper_mid += i
    elif(i[2] == 'cb'):
        cb += i
    elif(i[2] == 'parenting'):
        parenting += i
    elif(i[2] == 'lin'):
        lin += i
    elif(i[2] == 'rela'):
        rela += i
    elif(i[2] == 'other'):
        other += i


# In[ ]:


familyCount = {'Family Category':['Marriage', 'Dating', 'Divorce', 'Childbirth', 'Parenting', 'Linneage', 'Relatives', 'Other'],
             'Count':[len(men), len(dow), len(div), len(cb), len(parenting), len(lin), len(rela), len(other)]}
df = pd.DataFrame(familyCount, columns=['Family Category','Count'])
explode = (0.05,0.05,0.05,0.05, 0.05, 0.05, 0.05, 0.05)
patches, texts, autotexts = plt.pie(df['Count'], labels = df['Family Category'], autopct='%1.1f%%', 
        startangle=90, pctdistance=0.85, explode = explode)

#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

for autotext in autotexts:
    autotext.set_color('white')
    
plt.title("Identity 1-Associated family words")
plt.tight_layout()


# In[ ]:


men =[]
dow = []
div = []
cb = []
parenting = []
lin = []
rela = []
other = []
for i in diff1: 
    if(i[2] == 'men'):
        men += i
    elif(i[2] == 'dow'):
        dow += i
    elif(i[2] == 'div'):
        upper_mid += i
    elif(i[2] == 'cb'):
        cb += i
    elif(i[2] == 'parenting'):
        parenting += i
    elif(i[2] == 'lin'):
        lin += i
    elif(i[2] == 'rela'):
        rela += i
    elif(i[2] == 'other'):
        other += i


# In[ ]:


familyCount = {'Family Category':['Marriage', 'Dating', 'Divorce', 'Childbirth', 'Parenting', 'Linneage', 'Relatives', 'Other'],
             'Count':[len(men), len(dow), len(div), len(cb), len(parenting), len(lin), len(rela), len(other)]}
df = pd.DataFrame(familyCount, columns=['Family Category','Count'])
explode = (0.05,0.05,0.05,0.05, 0.05, 0.05, 0.05, 0.05)
patches, texts, autotexts = plt.pie(df['Count'], labels = df['Family Category'], autopct='%1.1f%%', 
        startangle=90, pctdistance=0.85, explode = explode)

#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

for autotext in autotexts:
    autotext.set_color('white')
    
plt.title("Identity 2-Associated family words")
plt.tight_layout()

