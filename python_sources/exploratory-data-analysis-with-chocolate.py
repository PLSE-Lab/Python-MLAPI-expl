#!/usr/bin/env python
# coding: utf-8

# <h1><center>Exploratory Data Analysis with Chocolate</center></h1>

# <img src="https://i.ndtvimg.com/i/2015-06/chocolate_625x350_81434346507.jpg" width="px"/>

# <h3>Context</h3>
# <p>Chocolate is one of the most popular candies in the world. Each year, residents of the United States collectively eat more than 2.8 billions pounds. However, not all chocolate bars are created equal! This dataset contains expert ratings of over 1,700 individual chocolate bars, along with information on their regional origin, percentage of cocoa, the variety of chocolate bean used and where the beans were grown.</p>
# <h3>Rating System</h3>
# <ul>
#     <li>5= Elite (Transcending beyond the ordinary limits)</li>
#     <li>4= Premium (Superior flavor development, character and style)</li>
#     <li>3= Satisfactory(3.0) to praiseworthy(3.75) (well made with special qualities)</li>
#     <li>2= Disappointing (Passable but contains at least one significant flaw)</li>
#     <li>1= Unpleasant (mostly unpalatable)</li>
# </ul>
# <h3>Acknowledgements</h3>
# <p>These ratings were compiled by Brady Brelinski, Founding Member of the Manhattan Chocolate Society. For up-to-date information, as well as additional content (including interviews with craft chocolate makers), please see his website: <a href="http://flavorsofcacao.com/index.html">Flavors of Cacao</a></p>

# ## Loading Data
# 

# In[ ]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# load the dataset from local storage
df=pd.read_csv('../input/flavors_of_cacao.csv')

# Understanding the basic ground information of our data
def all_about_my_data(df):
    print("Here is some Basic Ground Info about your Data:\n")
    
    # Shape of the dataframe
    print("Number of Instances:",df.shape[0])
    print("Number of Features:",df.shape[1])
    
    # Summary Stats
    print("\nSummary Stats:")
    print(df.describe())
    
    # Missing Value Inspection
    print("\nMissing Values:")
    print(df.isna().sum())

all_about_my_data(df)


# ## The Tantrums of the Feature Names
# Imagine an unsuspecting analyst runs the** df.head()** command for this dataset and then tries to view the first 5 entries of the **Review Date** feature based on the **head()** command's output. What does he get?

# In[ ]:


df["Review Date"].head()


# **This is what he gets!!**
# 
# What went wrong? Why is the feature not recognized?

# ### The above cell's output makes a revelation about our data and it is not a very pleasant one!
# The feature names are a bit messy as the names have the <strong>"\n" or "newline"</strong> character amidst them (as describe by our **df.dtypes** command)and this will lead to <strong>unidentifiable</strong> errors and if identified, they will cause <strong>excruciating</strong> methods of rectification(Nobody prefers going to each feature name and renaming it explicitly!).<br>

# In[ ]:


# Cleaning our feature names

cols = list(df.columns)

### Function to replace newline characters and spaces in the feature names
def rec_features(feature_names):
    rec_feat = []
    for f in feature_names:
        rec_feat.append(((f.casefold()).replace("\n","_")).replace(" ","_"))
    return rec_feat

print("Feature Names before Cleaning:")
print(cols)
print("\nFeature Names after Cleaning:")
print(rec_features(cols))


# Now, our features look much safer than they were before. However, the **"company\x..." feature still looks very convoluted**. Let's take that down with some manual removal. Finally, we shall re-assign the new feature names to our dataframe.

# In[ ]:


# Manual Removal

new_feature_names = rec_features(cols)
new_feature_names[0] = "company"

df=df.rename(columns=dict(zip(df.columns,new_feature_names)))
df.dtypes


# > **The features names look a lot more friendly now**!

# In[ ]:


df.head()


# ## Are we Missing Something?
# ### Identifying missing values within our dataset and solving the problem

# In[ ]:


# Checking out if we have missing values
df.info()


# There are just two missing values in our dataset.

# In[ ]:


df[['bean_type', 'broad_bean_origin']].head()


# **BUT WAIT! **
# 
# The **"bean_type"** feature clearly has loads of empty values according to the above cell's output even though the **df.info()** command only takes about 1 missing value! So, why this conundrum?<br>
# Let's check it out with a bit of **"Intuitively Written Test Code"**.

# In[ ]:


# What are these missing values in "bean_type" encoded as?

print(df['bean_type'].value_counts().head())
print("Missing Spaces encoded as:")
list(df['bean_type'][0:10])


# Oops...so we have **887 instances** in which "bean_type" is encoded as **space** or **\xa0**. 

# In[ ]:


# Replace the weird spaces with None (Symbolizes no data) 

def repl_space(x):
    if(x is "\xa0"):
        return "None"

# apply()        
df['bean_type'] = df['bean_type'].apply(repl_space)
df.head()


# Thus, we have filled those weird ambiguous missing values with a much better alternative.

# ## Convert Cocoa_percent to numerical values
# ### The % notation in 'cocoa_percent' is going to be a perpetual pain later on as it masks a numerical feature to be of an object dtype. So, let's make that conversion next.

# In[ ]:


# Making that much needed conversion

df['cocoa_percent']=df['cocoa_percent'].str.replace('%','').astype(float)/100
df.head()


# ## The Effect of Time - How did the Chocolate world change along the years?

# In[ ]:


### Cocoa Percentage patterns over the years

d5 = df.groupby('review_date').aggregate({'cocoa_percent':'mean'})
d5 = d5.reset_index()

# Plotting
sns.set()
plt.figure(figsize=(15, 4))
ax = sns.lineplot(x='review_date', y='cocoa_percent', data=d5)
ax.set(xticks=d5.review_date.values)
plt.xlabel("\nDate of Review")
plt.ylabel("Average Cocoa Percentage")
plt.title("Cocoa Percentage patterns over the years \n")
plt.show()


# #### Percentage of Cocoa over the years (Taking the average amounts per year)
# * The highest percentage of cocoa in a chocolate bar came in 2008 and was about 73%.
# * The lowest percentage of cocoa followed in the very next year, 2009 and hit 69%.
# * There was a rocky rise in the amount of cocoa in chocolate from 2009 to 2013 where it rose to about 72.2% from 69%.
# * From 2014, a steady decline in cocoa percentage in chocolate bars have been noticed and in 2017, it stands at just above 71.5%.

# In[ ]:


### Rating patterns over the years

d6 = df.groupby('review_date').aggregate({'rating':'mean'})
d6 = d6.reset_index()

# Plotting
sns.set()
plt.figure(figsize=(15, 4))
ax = sns.lineplot(x='review_date', y='rating', data=d6)
ax.set(xticks=d6.review_date.values)
plt.xlabel("\nDate of Review")
plt.ylabel("Average Rating")
plt.title("Average Rating over the years \n")
plt.show()


# #### Rating over the years (Taking the average amounts per year)
# * The lowest ever average rating was around 3 and it came in 2008.
# * Since then to 2011, there was a steady increase in average ratings and in 2011 it was at 3.26.
# * From 2011 to 2017, there have been several fluctuations in the ratings and in 2017, the rating lies at its apex at around 3.31.

# ### The Year 2008 - Year of Coincidence or something more than that?
# * The highest average coca percent was in 2008.
# * The lowest average ratings came in 2008.
# 
# The next year 2009 saw two major changes from the previous year :
# * There was a drastic reduce in cocoa content on an average
# * The average rating had a very steep increase to 3.08 from 3.00 in 2008
# 
# Is this an indication of how chocolate producers tried reducing their cocoa content to make better chocolate? OR was this just co-incidence?
# <br>
# **Let's leave that to your speculation.**

# ## The Chocolate Companies - The Best, The Patterns

# In[ ]:


### Top 5 companies in terms of chocolate bars in this dataset
d = df['company'].value_counts().sort_values(ascending=False).head(5)
d = pd.DataFrame(d)
d = d.reset_index() # dataframe with top 5 companies

# Plotting
sns.set()
plt.figure(figsize=(10,4))
sns.barplot(x='index', y='company', data=d)
plt.xlabel("\nChocolate Company")
plt.ylabel("Number of Bars")
plt.title("Top 5 Companies in terms of Chocolate Bars\n")
plt.show()


# * Soma has the highest number of chocolate bars in this dataset with 47.
# * The next closest competitor, Bonnat falls short of the leader by 20 bars.

# In[ ]:


# Distribution of Chocolate Bars

sns.set()
plt.figure(figsize=(8,6))
sns.countplot(df['company'].value_counts().sort_values(ascending=False))
plt.xlabel("\nCount of chocolate bars")
plt.ylabel("Number of Companies")
plt.title("Distribution of Chocolate Bars")
plt.show()


# * **120+ companies** have just one of their chocolate bars in the dataset.

# In[ ]:


### Top 5 companies in terms of average ratings
d2 = df.groupby('company').aggregate({'rating':'mean'})
d2 = d2.sort_values('rating', ascending=False).head(5)
d2 = d2.reset_index()

# Plotting
sns.set()
plt.figure(figsize=(20, 6))
sns.barplot(x='company', y='rating', data=d2)
plt.xlabel("\nChocolate Company")
plt.ylabel("Average Rating")
plt.title("Top 5 Companies in terms of Average Ratings \n")
plt.show()


# * Tobago Estate (Pralus) has a rating of 4.0 (the highest), however it has only one chocolate bar entry in this dataset.
# * These top 5 companies have very high ratings, however they have very low chocolate bars in the dataset.
# * Amedei has 13. Rest all have under 5.

# In[ ]:


### Top 5 companies in terms of average Cocoa Percentage
d2 = df.groupby('company').aggregate({'cocoa_percent':'mean'})
d2 = d2.sort_values('cocoa_percent', ascending=False).head(5)
d2 = d2.reset_index()

# Plotting
sns.set()
plt.figure(figsize=(15, 4))
sns.barplot(x='company', y='cocoa_percent', data=d2)
plt.xlabel("\nChocolate Company")
plt.ylabel("Average Cocoa Percentage")
plt.title("Top 5 Companies in terms of Average Cocoa Percentage \n")
plt.show()


# * All these have very high cocoa percentages (more than 80%)

# In[ ]:


### Average rating over the years (Top 5)

top5_dict = {}
for element in list(d['index']):
    temp = df[df['company']==element]
    top5_dict[element]=temp

top5_list = list(top5_dict.keys())

### Rating patterns over the years
d7 = df.groupby(['review_date', 'company']).aggregate({'rating':'mean'})
d7 = d7.reset_index()
d7 = d7[d7['company'].isin(top5_list)]

# Plotting
sns.set()
plt.figure(figsize=(15, 4))
ax = sns.lineplot(x='review_date', y='rating', hue="company", data=d7, palette="husl")
ax.set(xticks=d6.review_date.values)
plt.xlabel("\nDate of Review")
plt.ylabel("Average Rating")
plt.title("Average Rating over the years (Top 5 Producer Companies)\n")
plt.show()


# #### Time and the Chocolate Companies
# * Pralus and Bonnat were the earliest companies among these top 5 to be reviewed in 2006, while A. Morin was among the latest at 2012.
# * Both Bonnat and Pralus started around with the same average rating in 2006 of around 3.40, but in the very next year of 2007, whle Pralus hit it's highest ever rating of 4.00, Bonnat slumped to it's lowest of 2.50. As of 2016, Bonnat stands 0.25 rating points clear of Pralus on the yearly average.
# * The worst rating among these top 5 came in 2009 when Pralus got only a 2.00 average. This was a result of Pralus's steady decline from 4.00 in 2007 to 2.00 in 2009. (There is significant future scope of study here!)
# * Co-incidentally, the highest rating was just a year back, 2008 when Bonnat hit 4.00 (a feat Pralus had achieved the previous year).
# * From 2011 to 2015, Pralus has shown consistency in the average ratings.
# * A. Morin was reviewed only for the years 2012, 2013, 2014, 2015 and 2016. As of 2016, it's got the highest average rating at 3.75.
# * Fresco has not been reviewed after 2014, and its last review gave it around 3.30 on average rating.
# * Soma, the largest producer of chocolate bars, showcases constant fluctuations.
# * Soma was first reviewed in 2009 where it got around 3.42. In it's latest review in 2016, it has a 3.61.
# * Soma's lowest rating came in 2009 (3.42) and this is still higher than the lowest ratings other companies have got over all years.

# ## Following the Largest Chocolate Bar Producer (In terms of quantity) - Soma

# In[ ]:


### Preparing Soma for analysis

soma = df[df['company']=='Soma']


# In[ ]:


### Where does Soma get it's beans from ?

d3 = soma['broad_bean_origin'].value_counts().sort_values(ascending=False).head(5)
d3 = pd.DataFrame(d3)
d3 = d3.reset_index()
# Plotting
sns.set()
plt.figure(figsize=(10, 6))
sns.barplot(x='index', y='broad_bean_origin', data=d3)
plt.xlabel("\nBroad Bean Origin")
plt.ylabel("Number of Chocolate Bars")
plt.title("Where does Soma get it's beans from? \n")
plt.show()


# * Venezuela is the largest provider of Soma's beans.

# In[ ]:


### How are ratings of Chocolate bars by Soma ?

sns.kdeplot(soma['rating'], legend=False, color="brown", shade=True)
plt.xlabel("\nRating of the Chocolate Bar")
plt.ylabel("Proportion of Chocolate Bars")
plt.title("Ratings of Chocolate produced by Soma\n")
plt.show()


# * Soma has a major proportion of its bars rated from satisfactory levels to really high. So, they do produce some **good** chocolate.

# In[ ]:


### Soma's performance over the years
d4 = soma.groupby('review_date').aggregate({'rating':'mean'})
d4 = d4.reset_index()
# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(x='review_date', y='rating', data=d4)
plt.xlabel("\nDate of Review")
plt.ylabel("Average Rating")
plt.title("Soma's Average Rating over the years\n")
plt.show()


# #### Re-analyzing Soma Ratings through Time
# 
# * The worst average rating Soma ever got came in the year 2009 at 3.42, when it was first reviewed.
# * The highest average rating achieved came in 2010 at 3.75 (a significant rise from it's nadir the previous year).
# * Between 2012 and 2014, Soma's average rating saw a slump which revived after.
# * 3.75 was achieved in 2015 again; it slumped to 3.61 in 2016.

# In[ ]:


### Soma's performance over the years
d4 = soma.groupby('review_date').aggregate({'cocoa_percent':'mean'})
d4 = d4.reset_index()
# Plotting
plt.figure(figsize=(10, 6))
sns.lineplot(x='review_date', y='cocoa_percent', data=d4)
plt.xlabel("\nDate of Review")
plt.ylabel("Percentage of Cocoa")
plt.title("Soma's Percentage of Cocoa over the years\n")
plt.show()


# #### Cocoa percent in Soma chocolates over Time
# * First review in 2009 showed 70% cocoa.
# * The lowest percentage of cocoa in a Soma bar was in 2011 at 69%.
# * In 2015, Soma had the highest ever cocoa percent in their chocolate bar at 72.5%.
# * Latest review in 2016 discloses 69.6% cocoa.

# ## Categorizing Chocolate based on Ratings

# ### How many Chocolate bars are above or below 'Satisfactory levels' ?

# In[ ]:


# Chocolate Bar levels

unsatisfactory = df[df['rating'] < 3.0]
satisfactory = df[(df['rating'] >= 3.0) & (df.rating < 4)]
pre_elite = df[df['rating'] >= 4.0]
label_names=['Unsatisfactory','Above Satisfactory (Excludes Premium and Elite)','Premium and Elite']
sizes = [unsatisfactory.shape[0],satisfactory.shape[0],pre_elite.shape[0]]
# Now let's make the donut plot
explode = (0.05,0.05,0.05)
my_circle=plt.Circle((0,0),0.7,color='white')
plt.figure(figsize=(7,7))
plt.pie(sizes,labels=label_names,explode=explode,autopct='%1.1f%%',pctdistance=0.85,startangle=90,shadow=True)
fig=plt.gcf()
fig.gca().add_artist(my_circle)
plt.axis('equal')
plt.tight_layout()
plt.show()


# * This donut plot affirms that premium and elite chocolate is very rare, at only 5.6%.
# * 75% of the chocoalte bars in the study belong to 'Above Satisfactory'('premium and elite' are also a part of this category).
# * And, 25% of the chocolate bars that have been rated have ratings under 3.0.

# ### Rating Distributions

# In[ ]:


# The counts of each rating

r=list(df['rating'].value_counts())
rating=df['rating'].value_counts().index.tolist()
rat=dict(zip(rating,r))
for key,val in rat.items():
    print ('Rating:',key,'Reviews:',val)
plt.figure(figsize=(10,5))
sns.countplot(x='rating',data=df)
plt.xlabel('Rating of chocolate bar',size=12,color='blue')
plt.ylabel('Number of Chocolate bars',size=12,color='blue')
plt.show()


# * Most bars have been rated at 3.5.
# * Only 2 bars are rated at 5.0 (elite). Both belong to **Amedei**.

# ### Number of Chocolate bars per percentage of Cocoa

# In[ ]:


# Cocoa percent and choco bars

plt.figure(figsize=(10,5))
df['cocoa_percent'].value_counts().head(10).sort_index().plot.bar(color=['#d9d9d9','#b3b3b3','#808080','#000000','#404040','#d9d9d9','#b3b3b3','#404040','#b3b3b3'])
plt.xlabel('Percentage of Cocoa',size=12,color='black')
plt.ylabel('Number of Chocolate bars',size=12,color='black')
plt.show()


# * The above plot has the top 10 cocoa percentages in terms of number of chocolate bars.
# * The vast majority of bars have 70% cocoa, followed by 75% and 72%.

# ## What is the relation between 'Cocoa Percent' and 'Rating'?

# <p> Is there any correlation between Cocoa Percent and Rating of the bar? 
#     <br>
#     If it is, is that a positive correlation or a negative one?
#     <br>
#     Can we predict rating of a bar given it's cocoa percentage?</p>

# In[ ]:


# Cocoa Percent and Rating

sns.lmplot(x='cocoa_percent',y='rating',fit_reg=False,scatter_kws={"color":"darkred","alpha":0.3,"s":100},data=df)
plt.xlabel('Percentage of Cocoa',size=12,color='darkred')
plt.ylabel('Expert Rating of the Bar',size=12,color='darkred')
plt.show()


# #### Cocoa Percent versus Rating - Reading the Scatterplot above
# * No evident correlation. A numerical correlation gives a weak positive correlation coefficient of 0.09.
# * The density of the graph is highest between 65% and 80% of cocoa.
# * Chocolate bars with low cocoa percentage(less than 50%) and high cocoa percentage(above 90%) are less in number, but the most important fact is that most of these chocolate bars have a rating of less than 3,i.e they have been deemed 'Unsatisfactory'.
# * **Seems like people do not prefer very low or very high cocoa percentages in their chocolate!**

# <p>From the scatter plot above, we can infer that it would not be a good idea to guess a chocolate's rating based on its Cocoa Percentage.</p>

# ## Where are the Best Cocoa Beans grown?

# In[ ]:


#to get the indices
countries=df['broad_bean_origin'].value_counts().index.tolist()[:5]
# countries has the top 5 countries in terms of reviews
satisfactory={} # empty dictionary
for j in countries:
    c=0
    b=df[df['broad_bean_origin']==j]
    br=b[b['rating']>=3] # rating more than 4
    for i in br['rating']:
        c+=1
        satisfactory[j]=c    
# Code to visualize the countries that give best cocoa beans
print(satisfactory)
li=satisfactory.keys()
plt.figure(figsize=(10,5))
plt.bar(range(len(satisfactory)), satisfactory.values(), align='center',color=['#a22a2a','#511515','#e59a9a','#d04949','#a22a2a'])
plt.xticks(range(len(satisfactory)), list(li))
plt.xlabel('\nCountry')
plt.ylabel('Number of chocolate bars')
plt.title("Top 5 Broad origins of the Chocolate Beans with a Rating above 3.0\n")
plt.show()


# * Venezuela has the largest number of chocolate bars that have a rating above 3.0

# In[ ]:


#to get the indices
countries=df['broad_bean_origin'].value_counts().index.tolist()[:5]
# countries has the top 5 countries in terms of reviews
best_choc={} # empty dictionary
for j in countries:
    c=0
    b=df[df['broad_bean_origin']==j]
    br=b[b['rating']>=4] # rating more than 4
    for i in br['rating']:
        c+=1
        best_choc[j]=c    
# Code to visualize the countries that give best cocoa beans
print(best_choc)
li=best_choc.keys()
plt.figure(figsize=(10,5))
plt.bar(range(len(best_choc)), best_choc.values(), align='center',color=['#a22a2a','#511515','#a22a2a','#d04949','#e59a9a'])
plt.xticks(range(len(best_choc)), list(li))
plt.xlabel('Country')
plt.ylabel('Number of chocolate bars')
plt.title("Top 5 Broad origins of the Chocolate Beans with a Rating above 4.0\n")
plt.show()


# * So, here we see that the best cocoa beans are also grown in Venezuela.
# * There are 21 bars from Venezuela that have a rating of 4 and above.

# ## Analysis of the Producing Countries!!

# In[ ]:


df.columns


# In[ ]:


# Countries

print ('Top Chocolate Producing Countries in the World\n')
country=list(df['company_location'].value_counts().head(10).index)
choco_bars=list(df['company_location'].value_counts().head(10))
prod_ctry=dict(zip(country,choco_bars))
print(df['company_location'].value_counts().head())

plt.figure(figsize=(10,5))
plt.hlines(y=country,xmin=0,xmax=choco_bars,color='skyblue')
plt.plot(choco_bars,country,"o")
plt.xlabel('Country')
plt.ylabel('Number of chocolate bars')
plt.title("Top Chocolate Producing Countries in the World")
plt.show()


# * U.S.A has way more chocolate companies than any other country has according to this data.
# * Would it seem like a decent guess if we said that U.S.A consumes most chocolate as 'More the demand, more the production!'.
# * Let's leave that to speculation.

# In[ ]:


#reusing code written before
countries=country
best_choc={} # empty dictionary
for j in countries:
    c=0
    b=df[df['company_location']==j]
    br=b[b['rating']>=4] # rating more than 4
    for i in br['rating']:
        c+=1
        best_choc[j]=c    
# Code to visualize the countries that produce the best choclates
print(best_choc)
li=best_choc.keys()
# The lollipop plot
plt.hlines(y=li,xmin=0,xmax=best_choc.values(),color='darkgreen')
plt.plot(best_choc.values(),li,"o")
plt.xlabel('Country')
plt.ylabel('Number of chocolate bars')
plt.title("Top Chocolate Producing Countries in the World (Ratings above 4.0)")
plt.show()


# * USA produces the highest number of **4 and above rated choco bars**.

# ### To Be Continued...

# In[ ]:




