#!/usr/bin/env python
# coding: utf-8

# # Welcome ! 
# 
# In this notebook, I'll go through my process of preparing the raw data for analysis, and take you through several analytics that will eventually provide some insight on the Google Play Store and can help make decisions when planning your mobile app.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy import stats
# Importing the data from file:
import os
print(os.listdir("../input"))
df = pd.read_csv("../input/googleplaystore.csv")


# We'll start off by taking a look at the data we just loaded into our environemnt:

# In[ ]:


# Exploring the data set to better understand what're we facing
df.head()
df.shape
df.info()
df.describe()


# We can observe that something is odd about the description of Rating, with a Max rating of 19.
# Looking for that specific cell we see that the entire row was shifted. So we will simply remove this app for easier handling:
# 

# In[ ]:


df = df.drop(10472)


# Let's look at some of the unique values we have, to better understand the dataset:

# In[ ]:


df.Category.unique()
df.Size.unique() # should be translated to ints
df.Installs.unique() # categorical, perhaps should be left as is
df.Price.unique() # should be translated to float 
df['Content Rating'].unique()
df['Last Updated'].unique() # should be converted to datetime
df['Android Ver'].unique()  # should be considered to convert 
df['Current Ver'] # should be considered to convert
df.Genres.unique() # seems like there's some repetition. 


# Based on the findings above, we can now restructure our data:

# In[ ]:


# dropping duplicates from the datasets based on key fields 
df.drop_duplicates(subset=['App','Category','Current Ver','Last Updated'],inplace=True)

# changing last-update column to datetime
df['Last Updated'] =  pd.to_datetime(df['Last Updated'], format='%B %d, %Y')

# converting size to integers - 
#starting off by converting k and M to exponential based, removing "Varies with device", and converting it all to numeric:
df.Size=df.Size.str.replace('k','e+3')
df.Size=df.Size.str.replace('M','e+6')
df.Size=df.Size.replace('Varies with device',np.nan)
df.Size=pd.to_numeric(df.Size)

# converting reviews to integer - 
df.Reviews=pd.to_numeric(df.Reviews)

# converting price to float - 
df.Price=df.Price.str.replace('$','')
df.Price=pd.to_numeric(df.Price)

# converting installs to numeric (will be "at least X installed") - 
df.Installs = df.Installs.str.replace('+','')
df.Installs = df.Installs.str.replace(',','')
df.Installs = pd.to_numeric(df.Installs)

# to handle repetition in genres, we'll remove anything after ; and classify 
# based on what's coming before the ';'
df.Genres = df.Genres.str.split(';').str[0]


# Now the data is ready for exploration (and possibly training, if we choose to train any model on it).

# In[ ]:


df.head()


# # Visual exploration and insight generation
# We start off by asking a few questions we may ask to understand the ecosystem, and possibly, how to better our app's status (increasing it's download count and improving the rating, things that can potentially increase income).
# 
# We begin with taking a grand view over our data using seaborn's pairplot:

# In[ ]:


sns.set(font_scale=1.5)
# Initial relationship view - pairplot on all variables
sns.pairplot(df, hue = 'Type')


# No clear separations are visisble. Some interesting observations:
# * Paid apps are not as prominent in the ecosystem as free apps
# * Most of the paid apps are priced on the lower side of the spectrum
# * Very little apps made it to 1B downloads and the vast majority is on the lower side of the spectrum
# * Most ratings are higher than expected, and are distributed between 4 and 5 stars
# 
# We continue our exploration. We deep dive into the effects of putting a price tag on your app - is it a good strategy, or a growth-inhibitor?
# Let's look at amounts, first of all:

# In[ ]:


# Free vs Paid apps:
sns.countplot(x="Type", data=df)


# Clearly, there's very little portion of the ecosystem that's paid in advance. But how did these apps perform in terms of installations? Let's look at that breakdown:

# In[ ]:


# Distribution downloads per type (free/paid)
sns.countplot(x="Installs", data=df, hue="Type")


# Not surprisingly, free apps have a higher ceiling, and are not just more prominent in the ecosystem, but also downloaded far more than paid apps. Seems like free apps peak at around 1,000,000 downloads, while paid apps peak at around 1,000 downloads. That's very interesting and good to remember when planning the pricing model of an app, and concidering in-app purchases over prepaid apps.
# We can take a last look at free vs. paid by looking at how engaged users were - by looking at amount of reviews left for the apps:

# In[ ]:


fig, ax = plt.subplots()
# limiting the number of reviews to 50k, to make the histogram more readable. The vast majority of review counts is far less than 50k
sns.distplot(df[(df["Type"]=='Free') & (df["Reviews"]<50000)].Reviews,ax=ax,
                label="Free Apps",color='b',kde=False)
sns.distplot(df[(df["Type"]=='Paid') & (df["Reviews"]<50000)].Reviews,ax=ax,
                label="Paid Apps",color='g',kde=False)
ax.set(ylabel='Review Count')
plt.legend()
plt.show()


# As expected, free apps generate more user traffic and feedback, which reinforces our previous conclusion - generally, you better keep your app free, and try to generate revenue via in-app purchases.
# 
# Let's take a look at apps' ratings, and check what can we learn from that:

# In[ ]:


# Rating distribution:
sns.distplot(df['Rating'].dropna())


# As mentioned earlier, we see a surpsingly positive rating distribution: most of the apps get a high rating. But what happens when we look at specific genres?

# In[ ]:


# Look at rating per genre:
temp0 = df.groupby(["Genres"]).mean().reset_index().sort_values('Rating', ascending=False)
sns.barplot(x="Genres", y="Rating", data=temp0.iloc[np.r_[0:5, -5:0]], palette="GnBu_d")


# We can see which genres are often seen in positive manner (events, puzzles, art), but more interestingly the negatively viewed ones. It seems as users are not as happy with their dating apps (is it the app to blame, in this case, or the user?), navigation apps, trivia apps and tools. Perhaps these categories are worth looking into if you consider building an app to answer a specific need.
# 
# We can inspect the two aspects we looked at so far - priced apps and ratings. Do they have anything to do with one another?

# In[ ]:


fig, ax = plt.subplots()
sns.distplot(df[df["Type"]=='Free'].Rating.dropna(),ax=ax,
                label="Free Apps",color='b',kde=False)
sns.distplot(df[df["Type"]=='Paid'].Rating.dropna(),ax=ax,
                label="Paid Apps",color='g',kde=False)
ax.set(ylabel='Rating Count')
plt.legend()
plt.show()


# In fact, it seems as the general distribution is similar, with both free apps and paid apps peaking between 4-5 stars. However, we cannot be sure of it, so we can perform a t-test (Welch's) to test it out.
# The null hypothesis in this case is that both free and paid apps have the same average rating per app.

# In[ ]:


t, p = stats.ttest_ind(df[df["Type"]=='Free'].Rating.dropna(),
                       df[df["Type"]=='Paid'].Rating.dropna(), 
                       equal_var=False)


# As we see from the test, where p is much smaller than 0.05 (5%), we can reject the null hypothesis, and get to the conclusion that in fact, paid apps have a slightly higher average rating.
# 
# So we have seen so far that paid apps might get a higher rating, but this does not translate to higher download rates. In addition, we saw several genres where user satisfaction is not as high as in others, and could be a white space for someone planning an app from scratch. 
# Now we may want to ask the inevitable - does size matter?

# In[ ]:


# start off by looking at size distribution
df.Size.isnull().values.any()
ax = sns.distplot((df['Size']/1e6).dropna(),kde=False, bins = 100)
ax.set(xlabel='Size in Mb',ylabel='Count')


# The distribution definitely gives a good hint of the story to come. Most of the downloaded apps are well within the range of 2-10MB.
# If we compare the size to the number of installations, or even the amount of reviews, it's easy to conclude that smaller applications gain more attention and interaction. 
# Looking at size vs. installs:

# In[ ]:


ax1 = sns.jointplot(x='Size',y='Installs',data=df[df["Installs"]<10000001])


# And at size vs. reviews:

# In[ ]:


ax2 = sns.jointplot(x=(df[df["Reviews"]<50000]['Size']/1000000),
                   y=df[df["Reviews"]<50000].Reviews)


# We can take a deeper look at different categories and genres as well. We saw already that some categories may have better ratings, but what about the greater picture? Let's look at install  counts for both, to understand the users priorities:

# In[ ]:


fig, (ax3, ax4) = plt.subplots(nrows=2, sharex=False)
temp1 = df.groupby(["Category"]).mean().reset_index().sort_values('Installs', ascending=False)
sns.barplot(x="Category", y="Installs", data=temp1[temp1.Installs>5000000], palette="BuGn_r", ax=ax3).set_title('Installs per Category')
temp2 = df.groupby(["Genres"]).mean().reset_index().sort_values('Installs', ascending=False)
sns.barplot(x="Genres", y="Installs", data=temp2[temp2.Installs>5000000], palette = "GnBu_d",ax=ax4).set_title('Installs per Genre')
plt.show()


# Communication apps, such as Messenger, WhatsApp and Line, lead the way with nearly 4 billion installations (!). That's plenty. So do video players (YouTube and the Android default Google Movies) and social networks (Facebook, Twitter etc.). 
# Going back to user satisfaction with apps, we can now cross examine downloads with rating to understand whether those highly common apps are actually pleasing the users or not:

# In[ ]:


sns.scatterplot(x="Installs",y="Rating", data =temp2[temp2.Installs>5000000], hue="Genres", palette="GnBu_d", s=500)


# As you might have expected, the most common apps, are also the ones with the generally lower rating score - meaning, the fact that they're highly common and downloaded often, doesn't mean they necessarily work as users hope. Another aspect to consider is factory default apps, like Google Play Movies & TV which boasts a billion download, but if we looked at reviews - we can gain an insight on whether apps are actually used or not.

# In[ ]:


sns.boxplot(x="Installs", y="Reviews", data=df[(df["Installs"]>5000000)])


# Here, we saw that in fact those 1B downloads app, are in practice reviewed in about the same rate as those downloaded 500m times (50%) and not much more than those downloaded "only" 100m times (10%). This just gives an indication that these 1B downloads apps aren't necessarily often used, and might be factory default apps that come with the device and users did not even have the option to opt out.
# 
# I'll finish off this exploration with a final look, just for fun (and because it can be handy if you intend on applying any ML method later on), at  correlation within our dataset. Highly correlated variables can provide a lot of insight (and often this would be an early step rather than late), and when it comes to modeling ML models, you would usually prefer to avoid using features that are highly correlated, as it will not add a lot of valuable information and will take longer to process.

# In[ ]:


corr = df.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True)


# We see that most variables have very little to no correlation, however reviews and installs do show some correlation, which is logical - we expect more reviews on more downloads. However as we just showed before, it's not a clear correlation, since in some cases the installs are counted, but no one reviews the apps since they aren't put to use.
# 
# # Summary
# 
# To summarize this EDA session, we've learned a bit about the Android ecosystem - preference of free apps over paid ones, strong and weak categories, ghost defaulty apps with no users, and user ratings and preferences.
# Please leave feedback or ask any question if you have!
