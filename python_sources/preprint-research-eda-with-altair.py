#!/usr/bin/env python
# coding: utf-8

# <h1> Please do Upvote this kernel!!</h1>

# In[ ]:


import pandas as pd
df=pd.read_csv("/kaggle/input/covid19-research-preprint-data/COVID-19-Preprint-Data_ver2.csv")


# <h3> Welcome to this notebook, Here lets get a glimpse of the data and list the columns</h3>
# 

# In[ ]:



print(df.columns)
df.head()


# <h4> Ah of course the first thing I always would do is to split the Datetime into day,month,year and day in year(like 1st febuary is the 32nd day of the year , this helps in plotting :D)</h4>

# In[ ]:


from datetime import datetime
df["day"]=df["Date of Upload"].apply(lambda x: int(datetime.strptime(x,'%Y-%m-%d').day))
df["month"]=df["Date of Upload"].apply(lambda x:int( datetime.strptime(x,'%Y-%m-%d').month))
df["year"]=df["Date of Upload"].apply(lambda x: int( datetime.strptime(x,'%Y-%m-%d').year))
df["day_in_year"]=df["Date of Upload"].apply(lambda x:int( datetime.strptime(x,'%Y-%m-%d').timetuple().tm_yday))


# <h4> I am going to drop the columns below as I do not aim to use it in my EDA</h4>

# In[ ]:


df.drop(["Preprint Link","DOI","Date of Upload"],axis=1,inplace=True)


# <h4> Our first plot in the notebook is a simple one and I have done it using Altair, here I plot the number of abstracts(or generally papers ) submitted per day 
# <ul><li> -X= day in the year</li>
# <li> -Y= the count of the number of abstracts(research papers) published in that particular day</li></ul>
# <h4> the below plot is a bubble plot which increases its size as the number of abstracts per day, you can hover for more info</h4>

# In[ ]:


import altair_render_script
import altair as alt
alt.Chart(df.groupby(["day_in_year"]).count().reset_index()).mark_point().encode(
    x='day_in_year',
    y='Abstract',
    tooltip=["Abstract","day_in_year"],
    size="Abstract"
).interactive()


# In[ ]:


df[df["day_in_year"]==137]


# <h4> we can see above that after a certain day (about 137th day of the year) which is 16th May 2020</h4>
# <p><h4>Below we are going to plot the number of abstracts(research papers) published in the given months of 2020, we first filter out only months of 2020</h4></p>

# In[ ]:



alt.Chart(df[df["year"]==2020].groupby(["month"]).count().reset_index()).mark_area(
    line={'color':'darkblue'},
    color=alt.Gradient(
        gradient='linear',
        stops=[alt.GradientStop(color='white', offset=0),
               alt.GradientStop(color='blue', offset=1)],
        x1=1,
        x2=1,
        y1=1,
        y2=0

    )
).encode(
    alt.X('month'),
    alt.Y('Abstract',title="Abstract Count published"),
    tooltip=["month","Abstract"]
).interactive()


# <h4>Over here we will import nltk stopwords and remove the stopwords from the abstract and I have made a function for it, next I have done is as follows:
# <ul><li> I creat a function(get_top_n_words) for doing all in one </li>
# <li>it first removes stopwords using Countvectorerizer</li>
# <li>our function which takes n as input is the top n word frequencies we want</li>
# <li>using vecotor transform and bag of words we get the frequencies and the words of the whole <b>abstract column</b> which we will input as the agrument <b>corpus</b></li>
# <li> we then sort it according to the most number of frequencies and return the top n words and their frequencies in a <b>list of tuples</b></li>
# </ul>
# </h4>

# In[ ]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
stop_words=set(stopwords.words('english'))
def removeSW(x):
    x=x.lower()
    word_tokens = word_tokenize(x) 
    
    filtered_sentence = [w for w in word_tokens if not w in stop_words] 
    return " ".join(filtered_sentence)


df["Abstract"]=df["Abstract"].apply(removeSW)
from sklearn.feature_extraction.text import CountVectorizer
def get_top_n_words(corpus, n=None):
  
    vec = CountVectorizer(stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# <h4>Lets get our unigrams of the abstract column and the title column using this function </h4>

# In[ ]:


unigrams=get_top_n_words(df["Abstract"],20)
unigrams_title=get_top_n_words(df["Title of preprint"],20)


# In[ ]:


unigrams


# In[ ]:


unigrams_title


# <h4> Yes! I know both the list of tuples looks very much same since many of the title and abstract words are column, now lets get down in making a seperate dataframe and join this 2 list of tuples as rows and their values as column ,also a column for <b>type</b> indicates that wether it is abstract or title</h4>

# In[ ]:


d={"word":[],"count":[],"type":[]}
for i in unigrams:
    d["word"].append(i[0])
    d["count"].append(i[1])
    d["type"].append("Abstract")
for i in unigrams_title:
    d["word"].append(i[0])
    d["count"].append(i[1])
    d["type"].append("Title")
count_df=pd.DataFrame(d)


# <h4> We plot!!</h4>

# In[ ]:



source = count_df

alt.Chart(source).mark_bar().encode(
    tooltip=["word","count"],
    column='type',
    x='word',
    y='count',
    color='type'
)


# <h4> Now we modify the function before and create a new function called<b> get_top_gram</b> It takes one more argument called grams, which will be the number of grams we want, bigram trigram etc and after that I will plot the bigram as similarly as the unigrams, and then I plot the pentagrams!!<h4>

# In[ ]:


def get_top_gram(corpus,grams, n=None):
  
    vec = CountVectorizer(ngram_range=grams,stop_words = 'english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]


# 

# In[ ]:


bigrams=get_top_gram(df["Abstract"],(2,2),20)
bigrams_title=get_top_gram(df["Title of preprint"],(2,2),20)
d={"word":[],"count":[],"type":[]}
for i in bigrams:
    d["word"].append(i[0])
    d["count"].append(i[1])
    d["type"].append("Abstract")
for i in bigrams_title:
    d["word"].append(i[0])
    d["count"].append(i[1])
    d["type"].append("Title")
count_df=pd.DataFrame(d)

source = count_df

alt.Chart(source).mark_bar().encode(
    tooltip=["word","count"],
    column='type',
    x='word',
    y='count',
    color='type'
)


# In[ ]:


fivegrams=get_top_gram(df["Abstract"],(5,5),20)
fivegrams_title=get_top_gram(df["Title of preprint"],(5,5),20)
d={"word":[],"count":[],"type":[]}
for i in fivegrams:
    d["word"].append(i[0])
    d["count"].append(i[1])
    d["type"].append("Abstract")
for i in fivegrams_title:
    d["word"].append(i[0])
    d["count"].append(i[1])
    d["type"].append("Title")
count_df=pd.DataFrame(d)

source = count_df

alt.Chart(source).mark_bar().encode(
    tooltip=["word","count"],
    column='type',
    x='word',
    y='count',
    color='type'
)


# <h4> Now lets get onto the authors and the Universities!!</h4>
# <p> Here I use json to extract the dictionary present in the column <b>Author(s) Institutions</b></p>
# <p> after that I show the length of the set and list created, basically the set contains the number of unique Institutions and the list contains everytime the Institution is mentioned in the dataframe, we will use it ahead</p>
# 

# In[ ]:


import json
myset=set()
mylist=[]
for i in df["Author(s) Institutions"].index:
    l=set(json.loads(df.loc[i,"Author(s) Institutions"]).keys())
    for j in l:
        myset.add(j)
        mylist.append(j)


# In[ ]:


len(myset)


# In[ ]:


len(mylist)


# <h4>Another function phew!!!, well this is quite simple we use the list and count how many times each intistute has appeared and then a dictionary example ({"university":10})</h4>
# <p> basically what we want to do is that the frequency represents how many times the institution has published a paper since its occurence in the dataframe in each row is a count for its publications!!!</p>

# In[ ]:


def CountFrequency(my_list): 
      
  
   count = {} 
   for i in my_list: 
       count[i] = count.get(i, 0) + 1
       
   
    
   return count 
frequency=CountFrequency(mylist)
  
d={"UNI":[],"Publishes":[]}
for i in frequency.keys():
    d["UNI"].append(i)
    d["Publishes"].append(frequency[i])

uni_df=pd.DataFrame(d)

uni_df=uni_df.sort_values(by=["Publishes"],ascending=False)
for i in uni_df.index:
    if len(uni_df.loc[i].UNI)<4:
        uni_df.drop(i,axis=0,inplace=True)


# In[ ]:


source = uni_df.iloc[:50,:]

alt.Chart(source).mark_bar().encode(
    x='Publishes',
    y="UNI",
    tooltip=["Publishes"]
).properties(height=700)


# <h4> Voila! the graph above shows how many publications done by the institutions(here I plotted only the top 50 )<p>HOVER!!! and we find out <b> Oxford University</b> has published 61 times!!! 
# </p>
# <p>Now, lets check for the authors , below I used the previous dictionary of frequencies we outputed and change the values to list of values example
# <p>{"uni":1}===>{"uni":[1,0]}</p>
# here we add the 0 to all as we are going to store the authors count for this value!!!
# </p>
# </h4>

# In[ ]:


for i in frequency.keys():
   frequency[i]=[frequency[i],0]


# <h4>See the name of the institue is given and also the value with it are the authors number! which we did not consider above , well now we are!!</h4>

# In[ ]:


df["Author(s) Institutions"].head()


# <h4> We now again use json to get the values of author institution column but we will extract the names of the Uni as a key for our dictionary and will upadte the auhors value to it</h4>

# In[ ]:



mylist=[]
for i in df["Author(s) Institutions"].index:
    l=set(json.loads(df.loc[i,"Author(s) Institutions"]).keys())
    for j in l:
        frequency[j][1]=frequency[j][1]+json.loads(df.loc[i,"Author(s) Institutions"])[j]


# In[ ]:



d={"UNI":[],"AuthorCount":[]}
for i in frequency.keys():
   d["UNI"].append(i)
   d["AuthorCount"].append(frequency[i][1])

uni_df=pd.DataFrame(d)

uni_df=uni_df.sort_values(by=["AuthorCount"],ascending=False)
for i in uni_df.index:
   if len(uni_df.loc[i].UNI)<4:
       uni_df.drop(i,axis=0,inplace=True)


# <h4>We did the same procdeure and created a new dataframe for university and their authors count!</h4>

# In[ ]:


uni_df


# <h4> Finally we plot it!!, and hovering over the data we can see<b> Icahn School of Medicine</b> has the highest amount of authors who have published for covid-19 around 470</h4>

# In[ ]:


source = uni_df.iloc[:50,:]

alt.Chart(source).mark_bar().encode(
    x='AuthorCount',
    y="UNI",
    tooltip=["AuthorCount","UNI"]
).properties(height=700)


# <h1>Thank You!</h1>

# In[ ]:




