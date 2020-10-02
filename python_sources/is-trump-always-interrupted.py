#!/usr/bin/env python
# coding: utf-8

# # Exploring the first presidential debate by data mining

# ## Fire up everything

# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas
import numpy
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


debate = pandas.read_csv('../input/debate.csv',encoding = 'iso-8859-1')
debate.head(10)


# **The data set is not in the form we want, so the first step of the analysis will be data frame transformation. In this project, I just focus on the presidential debate between Trump and Clinton.**

# ## Dataframe Transformation

# **A. See when does the host interrupt, or try to interrupt the debate of a certain candidate**

# In[ ]:


Debate=debate[debate['Date']=='2016-09-26']


# **We can see that the host always tends to warm up the whole debate and greet audience/candidate. We first delete those data without useful information.**

# In[ ]:


print(Debate['Speaker'].unique())
print(Debate[Debate['Speaker']=='Audience']['Text'].unique())
print(len(Debate[Debate['Speaker']=='Clinton']))
print(len(Debate[Debate['Speaker']=='Trump']))
print(len(Debate[Debate['Speaker']=='Holt']))
print(len(Debate[Debate['Speaker']=='Audience']))


# **Checking the script quickly, we can delete the heading and ending of the data set directly since it mainly focuses on the greeting by the moderator.**

# In[ ]:


Debate=Debate.iloc[7:350,:].reset_index(drop=True)


# **As the debate is something mainly between two candidates, we can assume that when the moderator get involved in, the specific candidate is getting interrupted.**

# In[ ]:


Interrupt_clinton=[]
Interrupt_trump=[]
for i in range(len(Debate)-1):
    if Debate['Speaker'][i]=='Clinton' and Debate['Speaker'][i+1]=='Holt':
        Interrupt_clinton.append(i)
    elif Debate['Speaker'][i]=='Trump' and Debate['Speaker'][i+1]=='Holt':
        Interrupt_trump.append(i)


# **B. See when the audience laugh/applaud**

# In[ ]:


Laugh_clinton=[]
Applaud_clinton=[]
Laugh_Trump=[]
Applaud_Trump=[]
for i in range(len(Debate)-1):
    if Debate['Speaker'][i]=='Clinton' and Debate['Text'][i+1]=='(APPLAUSE)':
        Applaud_clinton.append(i)
    elif Debate['Speaker'][i]=='Trump' and Debate['Text'][i+1]=='(APPLAUSE)':
        Applaud_Trump.append(i)
    elif Debate['Speaker'][i]=='Clinton' and Debate['Text'][i+1]=='(LAUGHTER)':
        Laugh_clinton.append(i)
    elif Debate['Speaker'][i]=='Trump' and Debate['Text'][i+1]=='(LAUGHTER)':
        Laugh_Trump.append(i)


# **C: Let's create a new data frame**

# In[ ]:


Laugh=[]
Interupted=[]
Applause=[]
Interuptted_text=[]
for i in range(len(Debate)):
    if i in Laugh_clinton or i in Laugh_Trump:
        Laugh.append(1)
    else:
        Laugh.append(0)
    if i in Applaud_clinton or i in Applaud_Trump:
        Applause.append(1)
    else:
        Applause.append(0)
    if i in Interrupt_clinton or i in Interrupt_trump:
        Interupted.append(1)
        Interuptted_text.append(Debate['Text'][i+1])
    else:
        Interupted.append(0)
        Interuptted_text.append('No Interruption')
    


# In[ ]:


Debate.insert(4,'Laugh',Laugh)
Debate.insert(5,'Interupted',Interupted)
Debate.insert(6,'Interupted Text',Interuptted_text)
Debate.insert(7,'Applause',Applause)
del Debate['Line']
del Debate['Date']


# In[ ]:


Debate=Debate[Debate['Speaker']!='Holt']
Debate=Debate[Debate['Speaker']!='Audience']
Debate=Debate[Debate['Speaker']!='CANDIDATES']


# **Finally, we got a data frame in **

# In[ ]:


Debate.head()


# ## Analyzing the presidential debate 

# **Overall, what do the candidates said during the presidential debate?**

# In[ ]:


from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords


# In[ ]:


def to_words(content):
    letters_only = re.sub("[^a-zA-Z]", " ", content) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words )) 


# In[ ]:


def wordcloud(candidate):
    df=Debate[Debate['Speaker']==candidate]
    clean_text=[]
    for each in df['Text']:
        clean_text.append(to_words(each))
    if candidate=='Trump':
        color='black'
    else:
        color='white'
    wordcloud = WordCloud(background_color=color,
                      width=3000,
                      height=2500
                     ).generate(clean_text[0])
    print('==='*30)
    print('word cloud of '+candidate+' is plotted below')
    plt.figure(1,figsize=(8,8))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


# In[ ]:


wordcloud('Trump')
wordcloud('Clinton')


# **Hope the word cloud can provide you with an outline of the political standpoint of the candidates.**
# 
# **With different viewpoint, the audience and mediator are gonna make different reactions. Next, we will check how many times the candidate is interrupted/applauded/causing laugh during the debate.**

# In[ ]:


ind = numpy.arange(3)
trump=(len(Laugh_Trump),len(Applaud_Trump),len(Interrupt_trump))
clinton=(len(Laugh_clinton),len(Applaud_clinton),len(Interrupt_clinton))
fig, ax = plt.subplots()
width=0.35
rects1 = ax.bar(ind, trump,width, color='r')
rects2 = ax.bar(ind+width , clinton, width,color='y')
ax.set_ylabel('Counts')
ax.set_title('Counts of behavior of mediator and audience')
ax.set_xticks(ind)
ax.set_xticklabels(('Making laugh','Making applaud','Be interrupted'),rotation=45)
ax.legend((rects1[0], rects2[0]), ('Trump', 'Clinton'))
plt.show()


# **We can see that Trump was more frequently interrupted/questioned by the host than Clinton. The reason behind this may be the fact that Trump was a little bit more radical than Clinton**

# **So, people are maybe curious: how many words did the candidate said before being interrupted/questioned by the mediator?  And what did the candidate say when he was questioning/interrupting the candidate?** 

# In[ ]:


def interruption_analytic(candidate):
    if candidate=='Trump':
        color1='black'
        color2='r'
    else:
        color1='white'
        color2='y'
    df=Debate[Debate['Speaker']==candidate]
    df=df[df['Interupted']==1]
    length=[]
    text=[]
    for each in df['Text']:
        text.append(to_words(each))
        length.append(len(to_words(each).split()))
    print("="*40+'Analytic of '+candidate+'='*40)
    plt.hist(length,facecolor=color2)
    plt.title("Histogram of the count of words when being interrupted/questioned.")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.figure(1,figsize=(8,8))
    wordcloud = WordCloud(background_color=color1,
                      width=3000,
                      height=2500
                     ).generate(text[0])
    plt.figure(2,figsize=(8,8))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    


# In[ ]:


interruption_analytic('Trump')


# In[ ]:


interruption_analytic('Clinton')


# **Let us take a look at few conversations between mediators and candidates.**

# In[ ]:


trump_interupt=Debate[Debate['Speaker']=='Trump']
trump_interupt=trump_interupt[trump_interupt['Interupted']==1].reset_index(drop=True)
clinton_interupt=Debate[Debate['Speaker']=='Clinton']
clinton_interupt=clinton_interupt[clinton_interupt['Interupted']==1].reset_index(drop=True)


# In[ ]:


print('='*30+'Trump part'+'='*30)
print('Trump '+'\n'+trump_interupt['Text'][11])
print('Holt'+'\n'+trump_interupt['Interupted Text'][11])
print('Trump '+'\n'+trump_interupt['Text'][23])
print('Holt'+'\n'+trump_interupt['Interupted Text'][23])
print('Trump '+'\n'+trump_interupt['Text'][38])
print('Holt'+'\n'+trump_interupt['Interupted Text'][38])
print('Trump '+'\n'+trump_interupt['Text'][44])
print('Holt'+'\n'+trump_interupt['Interupted Text'][44])
print('='*30+'Clinton part'+'='*30)
print('Clinton '+'\n'+clinton_interupt['Text'][12])
print('Holt'+'\n'+clinton_interupt['Interupted Text'][12])
print('Clinton '+'\n'+clinton_interupt['Text'][17])
print('Holt'+'\n'+clinton_interupt['Interupted Text'][17])


# **We can find that Trump was consistently interrupted/questioned, even when he just said a few words. In contrary, Clinton has relatively uninterrupted speech, if you just take a look into the word count histogram of the speech before being interrupted.  Given few examples of the debates, we can find that Trump is a little bit 'in-experienced' in comparison with Clinton.**

# ##Conclusion

# **The volume of data set is small so I do not implement any machine learning technique in my analysis. In contrary, I choose to look at the data itself and find several funny stuff.**
# 
# **1. Trump is always interrupted, or the mediator is not patient enough to listen to him. Will this be a trend of his popularity?** 
# 
# **2. Clinton and Trump, their political standpoints are truly different and featured. WHO WILL WIN? NOBODY KNOWS.**
