#!/usr/bin/env python
# coding: utf-8

# ## NIPS Analysis ##
# 
#  - Number of Papers published every year
#  - Authors published more papers
#  - Number of authors per paper
#  - Top Authors in n authors per per categories
#  - Word Cloud for Paper Text
#  - Word Cloud for Title of Papers
#  - Which author has published papers for many years ?
#  - Which author has worked with many different co-authors?
#  - How the ratio of number of different co-authors per paper will look?
# 
#  If this EDA helps you, make sure to leave an upvote to motivate me to make more! :)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.gridspec as gridspec
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from wordcloud import WordCloud,STOPWORDS


# In[ ]:


authors = pd.read_csv("../input/authors.csv")
paper_authors = pd.read_csv("../input/paper_authors.csv")
papers = pd.read_csv("../input/papers.csv")


# ## Number of Papers published every year ##

# In[ ]:


g = sns.countplot(papers.year)
plt.xticks(rotation=90)


# ## Authors Published more papers ##

# In[ ]:


authors_new = authors.rename(columns = {'id':'author_id'})
paper_authors_new = pd.merge(paper_authors, authors_new, on='author_id', how='left')


# In[ ]:


author_paper_count = paper_authors[["author_id","paper_id"]].groupby("author_id").count().sort_values(by="paper_id",ascending=False).reset_index()
author_paper_count = author_paper_count.rename(columns = {'paper_id':'paper_count'})
authors_new = authors.rename(columns = {'id':'author_id'})
author_paper_count = pd.merge(author_paper_count, authors_new, on="author_id",how="left")
top15_authorsbycount = author_paper_count.iloc[:,1:].head(15)
plt.figure( figsize =(18,6))
top15_authorsbycount.plot(x="name",y ="paper_count", kind='bar', alpha=0.55)
plt.title("Highest number of papers published authors")


# ## Number of Authors Per Paper ##

# In[ ]:


grpby_papers = paper_authors.iloc[:,1:].groupby("paper_id").size().reset_index()
grpby_papers = grpby_papers.rename(columns = {0:'num_of_authors'})
num_authors_perpaper = grpby_papers.groupby("num_of_authors").size()
plt.figure( figsize =(15,6))
num_authors_perpaper.to_frame().rename(columns = {0:'num_of_authors'}).plot( kind='bar')
plt.title("Number of Authors per Paper")


# In[ ]:


## Top Authors in n authors per per categories ##


# In[ ]:


numofpapers_grp = grpby_papers.groupby("num_of_authors")
numofpapers = [1,2,3,4]
for gr in numofpapers:
    top10 = paper_authors[paper_authors["paper_id"].isin(list(numofpapers_grp.get_group(gr)["paper_id"]))].groupby("author_id").size().sort_values(ascending= False).head(10).reset_index()
    top10 = top10.rename(columns={0 : "num_of_papers_published"})
    top10_n = pd.merge(top10,authors_new,on="author_id",how="left").iloc[:,1:]
    plt.figure( figsize =(8,4))
    top10_n.plot(x="name", kind='bar', alpha=0.55)
    plt.title("Top Authors in "+str(gr)+" author per paper category")


# ## WordCloud for Paper Text ##

# In[ ]:


def realdata(papertexts):
    s = ""
    for i in range(len(papertexts)):
        w_list = papertexts[i].split()
        indexvalue= w_list.index("abstract")+1 if "abstract" in w_list else 0
        s = s+ " ".join( w_list[indexvalue: ] )
    return s


# In[ ]:


papertext = papers['paper_text'].str.lower()
completestring = realdata(papertext)


# In[ ]:


stopwords = STOPWORDS
stopwords.update(["this","that","thus","from","does","example","however","since","given","et","al"])
wordcloud = WordCloud(
                      stopwords=stopwords,
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(completestring)
plt.figure( figsize =(15,6))
plt.imshow(wordcloud)
plt.axis('off')


# ## WordCloud for Title ##

# In[ ]:


complete_titletext = " ".join(papers["title"].str.lower())


# In[ ]:


stopwords = STOPWORDS
#stopwords.update(["this","that","thus","from","does","example","however","since","given","et","al"])
wordcloud2 = WordCloud(
                      stopwords=stopwords,
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(complete_titletext)
plt.figure( figsize =(15,6))
plt.imshow(wordcloud2)
plt.axis('off')


# ## Authors who published papers for more years ##
# In 29 years (1987 - 2016), which author has published papers for many years ?

# In[ ]:


pap_auth = paper_authors.iloc[:,1:]
pap_year = papers[["id","year"]]
pap_year = pap_year.rename(columns ={"id":"paper_id"})
pap_auth_year = pd.merge(pap_auth, pap_year, on="paper_id",how="left")
pap_auth_year_grp = pap_auth_year.groupby(["author_id"]).year.nunique().sort_values(ascending=False)
sns.set_style("whitegrid")
fig, (ax1, ax2) = plt.subplots(ncols=2)
#plt.figure(figsize=(10,6))
ax1.figure.set_size_inches(8,4)
sns.boxplot(x=pap_auth_year_grp, ax = ax1)
sns.distplot(pap_auth_year_grp, ax = ax2)


# The above distribution graph shows that there are few percentage of authors are publishing papers for many year. Let's see who it is !

# In[ ]:


authors_grt_15 = pap_auth_year_grp[pap_auth_year_grp >15].reset_index().rename(columns={"year":"no_of_years"})
authors_grt_15= pd.merge(authors_grt_15,authors_new,on="author_id",how="left")[["name","no_of_years"]]
plt.figure(figsize=(9,4))
authors_grt_15.plot(x="name",kind="barh")
plt.xlabel("years")
plt.title("Authors who published papers for more years")


# The above bar chart show that **Geoffrey E. Hinton** published papers for **25 years out of 29 years**

# ## Which author has worked with many different co-authors? ##
# Let's see !

# In[ ]:


x = paper_authors.groupby("paper_id")["author_id"]
x2 = paper_authors.groupby("author_id")["paper_id"]
authors_dict = {}
for author_id in x2.groups:
    co_authors = []
    #print(author_id)
    #print(x2.get_group(author_id).values)
    for paper_id in x2.get_group(author_id).values:
        z = list(x.get_group(paper_id).values)
        z.remove(author_id)
        co_authors+=z
    authors_dict[author_id]= len(set(co_authors))
co_author_count = pd.Series(authors_dict).sort_values(ascending=False)
co_author_count = co_author_count.reset_index().rename(columns={0:"no_of_coauthors","index":"author_id"})


# In[ ]:


author_hist = pd.merge(author_paper_count,co_author_count, on= "author_id",how="left")
author_hist.sort_values(by="no_of_coauthors", ascending=False).head(10)


# Interesting ! Even though **Prof.Bernhard Sch?lkopf and Yoshua Bengio** has published papers less than **Michael l. Jordan**, **Prof.Bernhard Sch?lkopf and Yoshua Bengio** has worked with more co-authors than **Michael l. Jordan**.

# ## How the ratio of number of different co-authors per paper will look? ##

# Calculating the ratio and also not considering the authors who have published than less than 6 papers.

# In[ ]:


author_hist ["no_of_coauthors_perpaper"] = author_hist.apply(lambda row: row["no_of_coauthors"]/row["paper_count"], axis=1)
dd = author_hist[author_hist.paper_count > 5]
sns.distplot(dd.no_of_coauthors_perpaper)


# This show at least 1 new co-author will be used per paper.
