#!/usr/bin/env python
# coding: utf-8

# # Lets analyze github data and find out which language is more prevalent and what does the data say about the average source code size for a language

# <b> In this notebook we are going to analayze which programming language is most found on github, Moreover we will do a comparision of commonly used languages like c/c++/python/java/c# and see which language takes the throne</b>
# <br>
# <br>
# <b> In second section we will do some basic analysis on the average file size for each language. We will use interactive bar charts along with wordcloud to represent our data in the best explainable form. <b>
# <br>

# <b> Section 1
# 
# <a href='#section1'>1.1    WordCloud for Most found programming language files on Github</a>
# <br>
# <a href='#section11'>1.2   BarChart Representing Most found Programming Language on Github</a>
# <br>
# <a href='#section2'>1.3  Analysis on Common Programming Languages</a>
# <br>
# <br>
# <b> Section 2</b>
# 
# <a href='#biggest_file_size'>2.1  Wordcloud Representing Average File sizes for Languages used on Github</a>
# <br>
# <a href='#biggest_file_size_bar'>2.2  Bar Chart for File Size of Languages</a>
# <br>
# <a href='#least_file_size'>2.3  Bar Chart for File Size of Languages Represeting smallest space occupying languages</a>
# <br>
# <a href='#common_language_size'>2.4  Bar Chart For Common Languages Sizes</a>
# <br>
# <br>
# 
# <b>Let us First check the Query Size:</b>
# 
# 

# In[ ]:


import numpy as np
import operator
import pandas as pd 
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)
plt.rcParams['figure.figsize']=(12,5)
from google.cloud import bigquery
from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "github_repos")
QUERY = """
        SELECT language
        FROM `bigquery-public-data.github_repos.languages`
        """
print ("QUERY SIZE:   ")
print (str(round((bq_assistant.estimate_query_size(QUERY)),2))+str(" GB"))


# ## The query cost approximately 0.11 GB of data. Hence we can safely execute this query

# In[ ]:


QUERY = """
        SELECT count(language) as COUNT
        FROM `bigquery-public-data.github_repos.languages`
        """
df = bq_assistant.query_to_pandas_safe(QUERY)
print (df)
QUERY = """
        SELECT language
        FROM `bigquery-public-data.github_repos.languages`
        limit 50000
        """
df = bq_assistant.query_to_pandas_safe(QUERY)


# In[ ]:


## lets find out which file is the most language file is the most found in github repositories?
Names=[]
for x in df.language:
    Names.extend(x)


# In[ ]:


Count_Names={}
Average_Size={}
for x in Names:
    if x["name"] not in Count_Names:
        Count_Names[x["name"]]=0
        Average_Size[x["name"]]=0
    Count_Names[x["name"]]+=1
    Average_Size[x["name"]]+=x["bytes"]

for x in Count_Names.keys():
    Average_Size[x]=Average_Size[x]/Count_Names[x]


# In[ ]:


def Create_WordCloud(Frequency):
    wordcloud = WordCloud(background_color='black',
                              random_state=42).generate_from_frequencies(Frequency)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
def Create_Bar_plotly(list_of_tuples, items_to_show=40, title=""):
    list_of_tuples=list_of_tuples[:items_to_show]
    data = [go.Bar(
            x=[val[0] for val in list_of_tuples],
            y=[val[1] for val in list_of_tuples]
    )]
    layout = go.Layout(
    title=title,xaxis=dict(
        autotick=False,
        tickangle=290 ),)
    fig = go.Figure(data=data, layout=layout)
    #py.offline.iplot(data,layout=layout)
    
    py.offline.iplot(fig)


# <a id='section1'></a>
# ## Lets find which language source file is found most in the Github Repository using Wordcloud
# 
# 

# In[ ]:


Create_WordCloud(Count_Names)


# <a id='section11'></a>
# ## Now lets find out which programming language have the most number of files in all of the github
# 

# In[ ]:


sorted_names = sorted(Count_Names.items(), key=operator.itemgetter(1), reverse=True)
Create_Bar_plotly(sorted_names,30,"Most found programming languages source files in github")


# <a id='section2'></a>
# #  Lets compare commonly used languages C/C++/java/python/javascript/C#
# 

# In[ ]:


Common_languages=["C","Python", "Java", "C++", "JavaScript","C#"]
Common=[]
for x in Common_languages:
    Common.append((x, Count_Names[x]))
Create_Bar_plotly(Common,40,"Comparision of Commonly known languages")


# ## It seems that amongst the commonly used languages Javascript have taken the throne with python as the succesor

# <a id='biggest_file_size'></a>
# # Now lets look at which programming language source file have the biggest average size
# 

# In[ ]:


Create_WordCloud(Average_Size)


# <a id='biggest_file_size_bar'></a>
# # Now lets create a bar chart representing Programming languages with their Average source code sizes.
# 

# In[ ]:


Common_languages=["C","Python", "Java", "C++", "JavaScript","C#"]
Common=[]
for x in Common_languages:
    Common.append((x, Average_Size[x]))

sorted_average = sorted(Average_Size.items(), key=operator.itemgetter(1), reverse=True)
Create_Bar_plotly(sorted_average,35,"Programming languages Average source file size")


# ## Note that python does'nt even comes in the list of top35 Space occupying source codes !!.

# 
# <a id='least_file_size'></a>
# # Lets see which language occupies least space in terms of source code

# In[ ]:


sorted_average = sorted(Average_Size.items(), key=operator.itemgetter(1))
Create_Bar_plotly(sorted_average,40)


# 
# <a id='common_language_size'></a>
# # Lets do some analysis on average source file size of commonly used languages

# In[ ]:


Create_Bar_plotly(Common,40)


# ## Python Takes the throne over here for least file size hence it implies that python code will take less lines of code (as obvious) than codes of C/java/c++ etc.
# 
# <br>
# <b>Upvote it you find it useful</b>
# <br>
# <br>
# <b>Please do give your suggessions on this. </b>
# <br>
# <b> If you think there is anything more I can add then do suggest </b>
# <br><br>

# In[ ]:


#I can only fetch 50000 entries of the total of 3359866 entries, more than that it
# gives error Please suggest on how to fix it.

