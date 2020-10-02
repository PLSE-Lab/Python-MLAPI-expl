#!/usr/bin/env python
# coding: utf-8

# #Analyzing a projects success.

# Indie projects are a mixed bag. On one hand, someone, somewhere has a great idea, and they only need some resources to make that project into a reality. On the other hand, there's no guarantee that the end result will be something of any good.  People like these post every day in Indiegogo for a cause. Some are looking for help or assistance in saving something they deem valuable. Others want their ultimate creation to be an object brought upon reality. Some, just want to make a sandwich. But what is the factor that will determine whether these valiant creatures visions will be a reality or not? Will we find out? 

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dir = '../input/'

from subprocess import check_output
print(check_output(["ls", dir]).decode("utf8"))
import glob

data = pd.DataFrame()
for f in glob.glob((dir+'*.csv')): # all files in the directory that matchs the criteria.
    data = pd.concat([data,pd.read_csv(f)])


# *A first look*

# In[ ]:


data.head()


# ## Cleaning the dataset

# Let's get rid of columns that will have no use for the analysis, like urls. Altough friends may seem like an useful feature to keep, i'll not use it for the analysis because...reasons. 

# In[ ]:


useless_columns = ["id","url","category_url","igg_image_url","compressed_image_url","card_type",
                   "category_slug","source_url","friend_team_members","friend_contributors"]
data = data.drop(useless_columns, axis = 1)


# Now, let's start checking the columns to see what we need to clean and tidy up. Can't have a mess in our room now can we?

# In[ ]:


data.head(20)


# Right off the bat, we see some numerical columns that include percentages, dollar signs, etc. We need to keep this as  only numbers, so let's clean that up.

# In[ ]:


import re
def Remove_Non_Numeric(column):
    return re.sub(r"\D", "", column)

data.balance = data.balance.apply(lambda row : Remove_Non_Numeric(row) )
data.collected_percentage = data.collected_percentage.apply(lambda row : Remove_Non_Numeric(row) )
data.head()


# Now, let's divide the percentage columns by 100.

# In[ ]:


data.nearest_five_percent = data.nearest_five_percent.apply(lambda row: float(row)/100)
data.collected_percentage = data.collected_percentage.apply(lambda row: float(row)/100)
data.head()


# Now, let's  clean up the amount of time left column, using days as the measurement.  

# In[ ]:


def Get_Days_Left(time):
    if  "hour" in time:
        return float(Remove_Non_Numeric(time))/24
    elif "day" in time:
        return float(Remove_Non_Numeric(time))
    else:
        return 0.0
   


# In[ ]:


data.amt_time_left = data.amt_time_left.apply(lambda row: Get_Days_Left(row))
data.head()


# Let's also clean the in forever funding column, as ,even though it's a boolean column, it has multiple value types.

# In[ ]:


def Clean_Funding(column):
    if  "true" in column.lower():
        return 1
    elif "false" in column.lower() :
        return -1
    else:
        return 0
    
data.in_forever_funding = data.in_forever_funding.apply(lambda row: Clean_Funding(str(row)))
data.in_forever_funding.unique()


# In[ ]:


data.head()


# Now we can start some data exploration.

# ## Data Exploration and Visualization

# First, let's view the amount of projects by category name.

# In[ ]:


data.balance = data.balance.apply(lambda row: float(row))


# In[ ]:


def sb_BarPlot(data,label,measure):
    a4_dims = (11.0, 8.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    plot = sns.barplot(y=label, x=measure,ax=ax, data=data,orient="horizontal")
    
sb_BarPlot(data,"category_name","balance")


# So the categories with the highest mean balance are Audio and Energy & Green Tech.  Let's see what exactly these categories talk about.

# In[ ]:


data.loc[data.category_name == "Audio"].head()


# So, audio is usually about concerts, tours and the likes, which certainly are expensive. What about Energy and Green Tech?

# In[ ]:


data.loc[data.category_name == "Energy & Green Tech"].head()


# So these are revolutionary projects that will change the face of history! As if, but we can see we have some duplicated rows, which could affect our analysis. Let's remove them.

# In[ ]:


data = data.drop_duplicates()
data.shape


# Let's retry the plot, and see if anything changes.

# In[ ]:


sb_BarPlot(data,"category_name","balance")


# Not much has changed, Design has a higher mean balance, some other categories have lower values, nothing too relevant. How about  the relationship between numerical features?

# In[ ]:


a4_dims = (11.0, 8.27)
corr = data.corr()
fig, ax = plt.subplots(figsize=a4_dims)
hm = sns.heatmap(corr,annot=True)


# As suspected, balance and pledge count  are strongly correlated. Other notable correlations are pledge count/balance and the collected percentage and funding with nearest five percent. 
# 
# Now, let's modify the collected percentage feature into a label,  Complete or Incompletely funded.

# In[ ]:


data.collected_percentage = data.collected_percentage.apply(lambda row : np.where(row >= 1.0,1,0))


# Now let's visualize categories by percentage of completely funded projects. If anyone has an idea on how to optimize this, I'm welcome to hear it, since it takes a long time to run.

# In[ ]:


fig, ax = plt.subplots(figsize=a4_dims)
sns.barplot(x='collected_percentage', y='category_name', 
            data=data, orient = "horizontal", ax=ax,
            estimator=lambda x: sum(x==1)/len(x)*100)


# Now this is a difference!  Unfortunately for lovers of the culinary arts, it doesn't seem that Food and Beverage projects get funded that much. On the flip side, Audio, Theater and Comic book lovers can rejoice, since these categories have the highest funded ratios. Still, the maximum percentage of funding doesn't even reach 40%.
# 
# The next step, is to analyze the text of the title feature and we'll use the scikit learn Count Vectorizer for this.

# In[ ]:


import re
from nltk.corpus import stopwords
#Taken from bag of words competition.
def clean_text(text):    
    letters_only = re.sub("[^a-zA-Z]", " ",text) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops]   
    return( " ".join( meaningful_words ))   


# In[ ]:


new_titles = data.title.apply(lambda title: clean_text(str(title)))


# In[ ]:


data.title = new_titles


# Now that we have cleaned the titles, let's out a Count Vectorizer to figure out the vocabulary and get the counts.

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word",   
                             tokenizer = None,    
                             preprocessor = None, 
                             stop_words = None,   
                             max_features = 5000) 


# In[ ]:


titles = vectorizer.fit_transform(data.title).toarray()
words = vectorizer.get_feature_names()
counts = np.sum(titles, axis=0)
Word_Count = pd.DataFrame({"Word":words,"Count":counts})
Word_Count = Word_Count.sort_values(by = "Count", ascending = False)
Word_Count.head()


# Now, let's plot the top 20 words. 

# In[ ]:


a4_dims = (11.0, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
plot = sns.barplot(y="Word", x="Count",ax=ax, data=Word_Count.head(20),
                   orient="horizontal",estimator = sum)


# **HELP!** What most project makers  of indie gogo are looking for is **HELP**.  Other than that, we can see a lot of interest in films, music,books and art, in the artistic media side of things.  Other than that, it's words similar to help, like save, support, fund, etc (But at least we have love right?).

# ##Training a Random Forest Classifier

# Finally, let's train a Random Forest Classifier. Although I'll not aim for excellent performance or anything of the sort, I just want to see what the model considers as the most important factors in determining whether a project will reach 100% funding or not.

# In[ ]:


data.head()


# In[ ]:


#Dropping these columns, although I would like to use title.
succesful_pj = data.collected_percentage.values
columns = ["title","tagline","collected_percentage"]
data = data.drop(columns, axis = 1)


# In[ ]:


from sklearn import preprocessing
def transform_label(series):
    le = preprocessing.LabelEncoder()
    le.fit(data.category_name)
    return le.transform(data.category_name)
 
data.category_name = transform_label(data.category_name)
data.currency_code = transform_label(data.currency_code)


# In[ ]:



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 25)
clf.fit(data,succesful_pj)


# In[ ]:


importances = clf.feature_importances_
std = np.std([clf.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
plt.figure(figsize = (14,8))
plt.title("Feature importances")
plt.bar(range(data.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(data.shape[1]), data.columns,rotation = "vertical")
plt.xlim([-1, data.shape[1]])
plt.show()


# Well isn't that a pain. According to the classifier, the most important feature is the one I don't even understand!  If someone can comment what the nearest five percent is, it would be much appreciated. Anyway,  pledge count and balance are the other most important features, and as we saw before, they are strongly correlated. A surprising contender is the currency code. Maybe the region ( which determines the type of currency) has some saying in if a project is funded or not. 

# ##Conclusion

# Initially, I was expecting more weight to be put on the category of the project, that is, a project's category had a strong impact in the final result of the funding. And although historical data certainly does show some projects are more funded than others, other factors are more important that category. If anyone is up for it, feel free to try to determine  if the words in the title have any impact on whether a project gets complete funding or not!
