#!/usr/bin/env python
# coding: utf-8

# # Exploring Olympic History 

# In this notebook we explore interesting statistics about the Olympic games, using a dataset consisting of the features listed below, where each row corresponds to an athlete competing in a particular event. Note that this means that a single athlete may occur in multiple data entries, even within a single year or event. A single entry can be thought of the "event" corresponding to one athlete competing in one Olympic event.  
# 
# Name, Sex, Age, Height, Weight, Team, National Olympic Committee (NOC), Games (year and season), Year, Season, City, Sport, Event, Medal
# 
# This document is structured as follows. 
# 
# **1. Understanding the data **
# 
# **2. Basic Questions ** 
# 
# **3. Trends over time ** 
# 
# **4. Can we predict medalists?** 
# 
# **5. Fun with Kernel Density Estimation (KDE) **
# 

#  ## Section 1: Understanding the Data

# ### 1.1 Visualize data in a table

# Before we start analyzing the data, let's first get a feel for all the features and roughly what they look like. First we import the relevant modules we'll need, import the data, and display the head. 

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (16,8)


# In[ ]:


df_ath = pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv")
print(df_ath.shape)
df_ath.head()


# There are 15 features, and around 270000 rows in the data set. Each row corresponds to a particular athlete competing in an event, and contains features such as the Name, Height, Weight, Sex, Age, and Team of the athlete.  It also contains information about the year they competed, in which event, sport and whether they obtained a medal. 

# ### 1.2 Check for NaN values 
# 
# We can already see above that some values are NaN, so first off let's see how many NaN values we have in each feature (if most values are NaN, the the feature is probably not that useful).  

# In[ ]:


df_ath.isnull().sum()


# There are quite a few missing Age, Height, and Weight values, but not quite on the scale of hundreds of thousands, which is the size of the data set. It's worth keeping in mind that these features have quite a lot of NaN values, but it doesn't make those features completely useless. NaN values in the Medal feature indicate that no medal was obtained, so a high value there is not concerning. 

# ### 1.3 Visualize the distributions of numerical data 
# 
# Let's look at a few basic properties of the data set to get a feel for it. Let's look at the Age, Height, Weight and Year features since they're all numerical values and check whether their distributions over the whole data set makes intuitive sense. First, we can use the `describe()` which tells us some properties of the distributions. 

# In[ ]:


features = ["Age","Height","Weight","Year"]
df_ath.loc[:,features].describe()


# 
# We can visualize the distributions of each numerical feature as a box and whisker plot. 

# In[ ]:


plt.figure(figsize=(16,4))
for i,f in enumerate(features):
    plt.subplot(1,len(features),i+1)
    plt.boxplot(df_ath[f].dropna(), patch_artist=True, showfliers=True)
    plt.xticks([], [])
    plt.title(f)
plt.show()


# All these features seem to take resonable values. It's worth checking this to verify that things are within reasonable orders of magnitude. For interest's sake, the oldest recorded individual (97 years old) took part in an arts event, and the tallest recorded player (226 cm) was a basketball player.  

# In[ ]:


df_ath.loc[df_ath.Age == max(df_ath.Age)]


# In[ ]:


df_ath.loc[df_ath.Height == max(df_ath.Height)]


# It's important to remember that each row in this dataframe represents an athlete competing in an event. That means that a given athlete can feature in multiple rows, even within a single year. 

# ## Section 2: Basic Questions
# 
# ### 2.1 Winter and Summer Sports
# Which sports dominate in terms of participants in winter compared to summer? We show bar charts of the number of participants under each sport, aggregated over all of history. 

# In[ ]:


seasons = ["Summer","Winter"]
n_sports = 10
plt.figure(figsize=(16,10))
for i,s in enumerate(seasons):
    df = df_ath.loc[df_ath.Games.str.contains(s)]
    counts = df.groupby("Sport").size().sort_values(ascending=False)[:n_sports]
    plt.subplot(2,1,i+1)
    counts.plot.bar(rot=0)
    plt.ylabel("Number of Entries")
    plt.title(s)


# ### 2.2. Old and Young Sports 
# 
# Which sports have a lot of young athletes, say younger than 15 years old? And which sports have elderly participants, say those over 55? Let's find out. 

# In[ ]:


n_sports = 8
plt.figure(figsize=(16,10))
young = df_ath.loc[df_ath.Age < 15].groupby("Sport").size().sort_values(ascending=False)[:n_sports]
old = df_ath.loc[df_ath.Age > 55].groupby("Sport").size().sort_values(ascending=False)[:n_sports]
data = [young,old]
labels = ["Younger than 15","Older than 55"]
for i,series in enumerate(data):
    plt.subplot(2,1,i+1)
    series.plot.bar(rot = 0)
    plt.title(labels[i])
    plt.ylabel("Number of Entries")
plt.show()


# Swimming and gymnastics have many under-15 athletes, with other sports trailing behind. Art competitions is by far the sport with the most participants over 55, with shooting and other sports coming after.
# 
# ### 2.3 Dominant Countries

# In[ ]:


# import country names 
df_countries = pd.read_csv("../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv")
countries = {series.NOC : series.region for (_,series) in df_countries.iterrows()}
# fix bug in countries data
countries["SGP"] = countries["SIN"]
n_countries = 15
s = df_ath.groupby("NOC").size().sort_values(ascending=False)[:n_countries]
s.index = [countries[i] for i in s.index]
plt.figure(figsize=(16,4))
s.plot.bar(rot=0)
plt.ylabel("Number of Entries")
plt.title("Total Entries Over History of Dominant Countries")
plt.show()


# ## Section 3 : Trends over Time

# ### 3.1 Female participation 
# 
# Olympic sports have historically been male dominated. In the oldest recorded games in this dataset, the 1896 games, no women participated. 

# In[ ]:


len(df_ath.loc[df_ath.Year == 1896].loc[df_ath.Sex=="F"])


# After that female participation steadily increased. To visualize how this changed over time, we will look at the ratio of female athletes in the games for various sports over time. This ratio is zero when no women compete, and is 1 when only women compete. To see that men have been dominant, we list the sports for which women dominated (i.e. the ratio is above 0.5) in only three or more years throughout history.

# In[ ]:


# define a function to calculate the sex ratio
sexratio = lambda df : np.sum(df.Sex=="F")/df.shape[0]
# group the data by sport and year 
grouped = df_ath.groupby(["Sport","Year"])
for sport in df_ath.Sport.unique():
    years = sorted(df_ath.loc[df_ath.Sport==sport].Year.unique())
    ratios = [sexratio(grouped.get_group((sport,y))) for y in years]
    # print sport if women dominated for at least 3 years in history
    if np.sum(np.array(ratios) > 0.5) >= 3:
        print(sport)


# In these 6 sports there were more instances of females competing than males competing for at least three years. This is a small minority of all the sports in the Olympics, which we now list.

# In[ ]:


all_sports = df_ath.Sport.unique()
print("Number of sports: %i" % len(all_sports))
print()
print(all_sports)


# Most of these sports have historically been male domiated. We now show how this ratio changed over time in each sport. To see the data as a smooth progression in time, we shall compute the exponentially weighted average (moving average) of the time series for each sport. This makes the data less noisy and easier to see the trend. Exponential weighting for a series of data $x_1, x_2, ..., x_n$ is computed as a series $y_1, ..., y_n$ as follows: 
# 
# $$ y_1 = x_1$$
# $$ y_i = \beta y_{i-1} + (1 - \beta)x_i \quad i > 1$$

# In[ ]:


# group the dataset by sport, year and then sex 
grouped = df_ath.sort_values(by="Year").groupby(["Sport","Year"])
def expWeighted(data, beta = 0.8):
    """ Computes the exponentially weighted average of time series data. beta controls the weighting on history, where beta = 0 reduces to the time series itself."""
    v = data[0]
    result = np.zeros(len(data))
    for i in range(len(data)):
        v = beta*v + (1-beta)*data[i] 
        result[i] = v
    return result


# In[ ]:


beta = 0.5 # exponential weighting parameter, 0 = no weighting on history
sports = ["Athletics","Swimming","Cycling","Boxing","Basketball","Trampolining","Figure Skating"] # sports under consideration 
plt.figure(figsize=(15,8))
# compute the ratio over time for each sport 
for s in sports:
    years = sorted(df_ath.Year[df_ath.Sport == s].unique())
    sex_ratios = np.array([sexratio(grouped.get_group((s,y))) for y in years])
    plt.plot(years, expWeighted(sex_ratios, beta=beta), linewidth=2)
# compute the ratio over time over all sports
years = sorted(df_ath.Year.unique())
sex_ratios = [sexratio(df_ath.loc[df_ath.Year==y]) for y in years]
plt.plot(years, expWeighted(sex_ratios, beta=beta), c = "Grey", linestyle = "--",linewidth=3)
plt.plot([min(years),max(years)],[0.5,0.5],c="k",linestyle="--")
plt.title("Fraction of female Olympic participants by year")
plt.xlabel("Year")
plt.ylabel("Fraction of female participants")
plt.legend(sports + ["All Sports"])
plt.show()     


# We can see that for most of the sports shown above, female participation increases throughout history. Aggregated over all sports, we see a gradual increase over time. Since we are using a moving average with $\beta = 0.8$, we can think of each point as representing an average over the last $\frac{1}{1 - \beta} = 5$ years.
# 

# ### 3.2 African countries: has African participation increased? 
# 
# Being South African I'd like to know a bit more about the participation of African countries over the years. To do this I will import the other dataset which contains the country / region names by NOC (national olympic committee), as well as a dataset containing information about the continent of each country. Due to differences in NOC values and country names, a few NOC values could not be linked up successfully with a continent. However, thankfully no African countries suffered from this -- below we print the NOC and region for which we failed to find the continent.   

# In[ ]:


df_continents = pd.read_csv("../input/world-countries-and-continents/countries and continents.csv") 
# continent by official olympic committee
continents = {series.IOC : series.Continent for (_,series) in df_continents.iterrows()}
# continent by country name 
cont_by_country = {series.official_name_en : series.Continent for (_,series) in df_continents.iterrows()}
african = []
for noc in countries.keys():
    if noc in continents:
        if continents[noc]=="AF":
            african.append(noc) 
    elif countries[noc] in cont_by_country:
        if cont_by_country[countries[noc]]=="AF":
            african.append(noc) 
    else:
        print(noc, countries[noc])


# As you can see, none of these are african countries. We now aggregate all african countries / regions, and compare African participants to other parts of the world by looking at the fraction of entries filled by African participants in each year. I will focus on the Summer games, since African nations tend to participate less in Winter events.  

# In[ ]:


nations = ["FRA","GBR","USA"]
years = df_ath.loc[df_ath.Year > 1920].loc[df_ath.Games.str.contains("Summer")].Year.sort_values().unique()
grouped = df_ath.groupby(df_ath.Year)
getRatio = lambda df, nocs : np.sum(df.NOC.isin(nocs))/df.shape[0]

plt.figure(figsize=(15,5))
ratios = np.array([getRatio(grouped.get_group(y), african) for y in years])
plt.plot(years,expWeighted(ratios*100))
for noc in nations:
    ratios = np.array([getRatio(grouped.get_group(y), [noc]) for y in years])
    plt.plot(years,expWeighted(ratios*100))
    
plt.legend(["African"] + [countries[noc] for noc in nations])
plt.ylabel("Percentage of Entries")
plt.xlabel("Summer games year")
plt.show()


# Indeed, African nations have had steadily more participation over history. However, the scale of participation for the whole of Africa  is similar of that for the dominant countries like Great Britain, France, and USA (rather than, for example, all three put together, which is more along the lines of what I expected before producing this chart).  

# # Section 4: Can we Predict Medalists? 
#  
# In this section we explore to what extent medalists can be predicted by features such as weight, height, age and so on. Can we quantify how much more likely an athlete is to win a medal given that they satisfy certain criteria? For concreteness, we'll focus on female swimmers. Note that in the following analysis, individual swimmers are counted multiple times, but as we are considering physique and age, this makes sense as these attributes can change over time for a single person, potentially affecting their ability to win medals. 
# 
# I'm going to hypothesize that height, weight, and age all have some kind of predictive value on whether an athlete will win a medal. To get a sense, let's plot the distributions of heights, weights and ages both for gold medalists and non gold medalists. 

# In[ ]:


sport = "Swimming"
sex = "F"
df_sport = df_ath.loc[df_ath.Sport == sport].loc[df_ath.Sex == sex]
# group by medal / no medal
grouped = df_sport.groupby(df_sport.Medal.notnull())
# function to normalize histograms 
getWeights = lambda x : (1/len(x))*np.ones_like(x)
plt.figure(figsize=(14,10))
for i,attr in enumerate(["Height","Weight","Age"]):
    medal    = grouped.get_group(True)[attr].dropna()
    no_medal = grouped.get_group(False)[attr].dropna()
    plt.subplot(2,2,i+1)
    plt.hist([medal, no_medal], weights = [getWeights(medal), getWeights(no_medal)])
    plt.legend(["Medal", "No Medal"])
    plt.title("%s dists of medal vs no-medal" % attr)
plt.show()


# The medal-scoring swimmers definitely have different distributions of age, weight and height than non-medalists, although the difference isn't striking. Does that mean, though, that we have any chance in predicting whether a swimmer will score a medal given their physique and age? If $x$ is an feature like age or weight, we can write down Bayes' rule: 
# 
# $$ P(medal \; | \;x ) = \frac{P(x \; | \; medal) P(medal)}{P(x)} $$
# 
# In principle then, given the value of the feature $x$ we should be able to work out the probability of a medal, since we know $P(x \; | \; medal)$ i.e. the distribution of the feature given medal value (plotted above) as well as the marginal probabilities of obtaining a medal and having the feature $x$ (given by the data). 
# 
# 
# To see if we can put this to use, let's train a classifier based on the Naive Bayes algorithm. 

# In[ ]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
X = df_sport.loc[:,["Height","Weight","Age"]].dropna()
y = df_sport.loc[X.index,:].Medal.notnull()
clf = GaussianNB()
scores = cross_val_score(clf, X, y, cv =5)
print(scores)


# We trained a model with over 85% accuracy to predict medalists from their age, weight and height. Is this really such a good model, though? Often accuracy is not a very good measure of how good a model is. For example, we may ask: What portion of the actual medalists did we correctly predict? 

# ### 4.1 Recall: predicting correctly for true medalists
# 
# Most people in the Olympics don't win a medal. In this sense, getting a medel is a rare property. How good is our model for those rare individuals who do win medals? Recall measures exactly this: the ratio of correctly predicted medalists (true positives) to the actual number of medalists (true positives plus false negatives). To test how well our model does on recall, we do cross-validation, computing the recall score on each test set "left out" during model training. 

# In[ ]:


from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer

cross_val_score(clf, X, y, scoring=make_scorer(recall_score), cv=5)


# We get really poor recall scores! I suspect that's because our model predicts *'no medal'* for most athletes, which is true, but not very helpful for recalling those athletes that *did* get a medal! To see whether this is true, we train the model on the full data set, and have a look at the fraction of data points that are predicted to be medalists by the model, compared to the true medalist distribution. 

# In[ ]:


clf.fit(X,y)
ypred = clf.predict(X)
L = len(ypred)
plt.figure(figsize=(16,3))
plt.subplot(1,2,1)
plt.title("Model")
plt.bar(["Medal","No Medal"],[np.sum(ypred)/L,np.sum(ypred == False)/L])
plt.subplot(1,2,2)
plt.title("True values")
plt.bar(["Medal","No Medal"],[np.sum(y)/L,np.sum(y == False)/L])
plt.show()
print("Model predicts %i out of %i instances are medalists" % (np.sum(ypred), len(ypred)))


# As we suspected, our model is "cheating" by simply classifying almost all instances as non-medalists. To be precise it predicts 12 medalists out of 8445 instances, which amounts to about 0.1%. That's compared to a 13.8% of instances that are truly medalists in the data set! 

# ### 4.2 Precision: being sure about your predictions 
# 
# We can also look at another metric, which is precision. This essentially measures the accuracy of positive predictions. In other words, for those entries predicted to be medalists, were they correct? This is computed as the number of true positives divided by the total number of positives predicted (true positives plus false positives)
# 
# With some wishful thinking and anthropomorphism, our model might be being very conservative about its positive predictions, only saying "Medalist!" when it's really sure. We can see whether that's true by measuring the precision score on a cross-validated set.  

# In[ ]:


from sklearn.metrics import precision_score
cross_val_score(clf, X, y, scoring=make_scorer(precision_score), cv=3)


# Sadly, our precision scores are poor too. This means that, even for those very few entries that the model predicted as medalists, not even they were medalists very often! (Note: the warning above tells us that the model made no positive predictions for some of the test sets, making the precision value undefined)
# 
# We've learned a valuable lesson. Although we trained a model to predict medalists with about 85% accuracy, that didn't make it a useful model, because a) it mostly predicted negative, making it useless for *identifying* true medalists, and b) even those positive predictions that it made were on mostly not correct. Perhaps age, weight, and height are just not sufficient for identifying medalists, although we have not shown that here, and will not endeavour to try to prove or disprove that statement.  In the end, we didn't quantify the likelihood of being a medalist given phyisque values (this probably requires a bit more exploration) but we did learn a lesson about accuracy, precision and recall. 

# ## Section 5: Fun with Kernel Density Estimation (KDE) 

# ### 5.1 Physique distributions with KDE
# 
# Earlier we looked at the Weight, Height and Age distributions of medalist and non-medalists swimmers. To visualize the distributions, we plotted histograms, which essentially bin the data into discrete bins, which one can think of as a discrete way of estimating the distribution that generated the data. Another way to estimate the distribution generating data is kernel density estimation (KDE). This estimates a continuous distribution from the data, which makes more sense for a continuous quantity. We now use KDE on the weight and height features of female swimmers. 

# In[ ]:


from sklearn.neighbors import KernelDensity

df = df_sport.loc[df_sport.Height.notnull() & df_sport.Weight.notnull()]
features = ["Height","Weight"]
units = ["cm","kg"]
kde = KernelDensity(bandwidth=5)

plt.figure(figsize=(16,3))
for i, feat in enumerate(features):
    # weight / height data 
    x = df[feat]
    x_range = np.linspace(min(x),max(x),1000)
    # fit kernel density to data
    kde.fit(x[:,None])
    # compute the log probability over a range 
    logprob = kde.score_samples(x_range[:, None])
    plt.subplot(1,2,i+1)
    plt.fill_between(x_range, np.exp(logprob))
    plt.xlabel(feat + " in " + units[i])
    plt.ylabel("Probability Density")
plt.show()


# The KDE function has computed continuous probability density functions for the height and weight data of female swimmers. they look roughly Gaussian, which is quite nice and expected. The area under a given section of the curve corresponds to the probability of a sample having a height / weight in the given section, and as such the area under the whole curve is 1. We can see that these are qualitatively similar to the histograms we computed before. Now we use the same technique to compute distributions over weight and height at the same time, so that we can visualize the *joint distribution* of weight and height, seeing how the two features co-occur. We will do this for medalist and non-medalist swimmers, and see if we can spot a difference.    

# In[ ]:


plt.figure(figsize=(15,4))
labels = ["No Medal","Medal"]
h_range = np.linspace(155,190,50)
w_range = np.linspace(40,80,50)
X,Y = np.meshgrid(h_range,w_range)
# plot 2D probability density functions for medalists and non-medalists  
for i, medal in enumerate([False,True]):
    # 2D weight / height data
    D = df.loc[df.Medal.notnull() == medal].loc[:,["Height","Weight"]] 
    kde = KernelDensity(bandwidth=5)
    kde.fit(D)
    def F(x,y):
        return np.exp(kde.score_samples(np.array([[x,y]])))
    Z = np.vectorize(F)(X,Y)
    plt.subplot(1,2,i+1)
    cntr = plt.contour(h_range,w_range,Z)#,levels=[0.0,0.001,0.002])
    plt.clabel(cntr, inline=1, fontsize=10)
    plt.pcolor(h_range,w_range,Z)
    plt.colorbar()
    plt.xlabel("Height in cm")
    plt.ylabel("Weight in kg")
    plt.title(labels[i])
plt.show()


#  This time, the volume under each surface shown above should sum to one. We can see that the weight / height distribution of medalists is shifted up and right compared to the non-medalists distribution. This implies that medalist swimmers are heavier and taller on average. The distributions here again look roughly Gaussian, and from the way they are skewed, we can tell that weight and height are positively correlated, i.e. tall people tend to weight more and vice versa, and this makes intuitive sense.  

# 
