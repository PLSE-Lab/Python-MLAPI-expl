#!/usr/bin/env python
# coding: utf-8

# # Extensive Exploratory Data Analysis of Yelp Business Dataset

# This notebook attempts to perform an extensive exploratory data analysis of the Yelp business dataset with various summary statistics and plots. I hope some of you might find it useful. Please upvote and fork the notebook if you like it, and please comment on your take on the approach I have taken in this EDA.
# 
# My original goal of doing time-series analysis on few top major categories by extracting the reviews data on some top business categories identified in this EDA.
# However, I have halted my attempt, for now, to use reviews data as I want to read all of the dataset and then filter on to extract all the reivews associated with few specific business categories. I found that it is not feasible to do that in Kaggle due to RAM limit at 16GB which caused the memory to max out whenver I read the entire review dataset. Given the data format the review file is in (.json that is not in proper .json format), I could not find an efficient way of reading by streaming the data only saving what I wanted to a dataframe. 

# In[1]:


#First we will load the necessary libraries and classes
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.io.json import json_normalize
import json
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('pylab', 'inline')
import string

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))


# Next, lets load the business data into pandas dataframe and look into the size, and structure of the dataframe, columns and their data types
# 

# In[3]:


biz_f = open('../input/yelp_academic_dataset_business.json',encoding="utf8")
biz_df = pd.DataFrame([json.loads(x) for x in biz_f.readlines()])
biz_f.close()
biz_df.info()


# In[ ]:


# We should have a feel for how the data frame itself looks like before doing anything else on it
biz_df.head(5)


# In[ ]:


#Check the missing values for each column
biz_df.isnull().sum()


# There are a significant number of missing values for business attributes and business hours. For our purpose, we will not be using these two columns and thus do not need to worry about them. 
# 
# However, we will use business categories in the analysis and need to decide what to do about the missing values. We cannot impute the missing business categories, thus we will most certainly need to drop these observations. Let's take a detailed look at those rows.

# In[ ]:


biz_df[biz_df['categories'].isnull()].head()


# In[ ]:


#Total number of reviews associated with nan values in categories
biz_df[biz_df['categories'].isnull()]['review_count'].sum()


# Let's go ahead and drop the rows that have 'None' values for business categories.

# In[ ]:


biz_df.dropna(subset=['categories'],inplace=True)


# Next, we will parse and conver the categories to lists so that other operations can be performed on it.

# In[ ]:


#Total number of unique business categories in the dataset
biz_df['categories'].nunique()


# 93,385 business categories are too many categories for the purpose of our analysis.
# 
# Next, we will look at top 25 most reviewed business and see what business categories they belong to.

# In[ ]:


#Plotting top 25 most reviewed businesses among all categories
ax = sns.catplot(x="review_count", y="name",data= biz_df.nlargest(20,'review_count'), 
                 kind="bar",hue= "categories", dodge= False, height= 10 )

plt.subplots_adjust(top=0.9)
ax.fig.suptitle('Top 25 Most Reviewed Businesses And Categories Lables Used') # can also get the figure from plt.gcf()


# We see that there is some variation among businesses in the way and number of keywords they use in the categories.
# Some specific businesses, for example, Secret Pizza is using just two labels, 'Resturants' and 'Pizza' in their categories.
# On the other hand, The Buffet is using 10 different labels in their categories. One other thing to notice is that the categories keywords are not always alphabetically ordered. If we order them all, we might end up with lower than 90,000+ unique categories.

# In[ ]:


#Trim any leading white-space. This takes care of cases where some items were not being sorted properly due to whitespaces.
biz_df['categories'].str.strip()

#Create an sorted string, does not create list, and can use .nunique() function if it's not list
#biz_df['categories'] = biz_df['categories'].apply(lambda x: ', '.join(sorted(x.split(', '))))

#Following turns into lists, .nunique() does not work on lists, so need to count list length of the items (!!USE THIS!!)
biz_df['categories'] = biz_df.categories.map(lambda x: [i.strip() for i in sorted(x.split(", "))])

#biz_df['categories'].nunique()
biz_df['categories']

#Count of unique combinations of business categories after alphabetically ordering the labels
print (biz_df['categories'].apply(tuple).nunique())


# Ordering the lists of business labels in the categories brought the unique counts down to 48,856 from 93,385. Still, 48,000 plus is a lots of categories for category based analyses.
# 
# One challenge in business category based analyses is deciding how to treat non-specialized businesses that label themselves using multiple diverse lables. Should we extract all unique categories keywords used in the dataset and let businesses using diverse labels to be counted/used across all such categories lables? That might not actually be a good idea.
# 
# According to Yelp, businesses can be in up to three categories. It lists only 22 top-level businesses categories that have over 1200 sub-categories overall, which can lead to millions of category combinations. [Yelp website](https://blog.yelp.com/2018/01/yelp_category_list#section13)
# 
# Given the Yelp policy that businesses can be in up to three top-level categories, we can see that lots of businesses have used multiple top-level categories to describe themselves, and many are still using more than three, probably keywords from the sub-categories.
# 
# For our purpose, we will use the 22 top-level categories of Yelp to find out which businesses are using how many of these labels and create new classification column to re-assign the categories using top-level words only. This should help put many businesses into their most representative categories high-level categories while also brining down the total number of unique categories for the analysis. 

# In[ ]:


#Add a new column to count the number of category keywords used
biz_df['Num_Keywords'] = biz_df['categories'].str.len()

#Top 20 categories with most keyword
biz_df[['categories','Num_Keywords']].sort_values('Num_Keywords',ascending = False).head(10)


# We can see that some businesses are using as much as 37 keywords in their categories. Let's look at some of these businesses.
# 
# How about the overall distribution of the number of keywords used by businesses? Let's discern that information.

# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)

x = biz_df['Num_Keywords']
numBins = 100
ax.hist(x,numBins,color='green',alpha=0.7)
plt.show()


# Looking at the distribution of numbers of keywords used by businesses in their categories, vast majority of them are using between one to five keywords. Very few businesses are using over ten keywords, and it is likely to drop such businesses due to possible uncertainity in determining their actual representative top-level category.

# In[ ]:


#Populating the number of times each combination of unique category combinations found used in the dataset.
df_biz_CountBizPerCat = pd.DataFrame(biz_df.groupby(biz_df['categories'].map(tuple))['Num_Keywords'].count())

#Looking at 'n' category combinations
n = 10
df_biz_CountBizPerCat.sort_values(['Num_Keywords'], ascending = 1).head(n)


# In[ ]:


#New lets look at the distribution of review counts across major categories
df_BusinessesPerCategories_pre = pd.DataFrame(biz_df.groupby(biz_df['categories'].map(tuple))['review_count'].sum())
df_BusinessesPerCategories_pre.reset_index(level=0, inplace=True) #reset index to column
df_BusinessesPerCategories_pre['Cum_review_count'] = df_BusinessesPerCategories_pre['review_count'].cumsum(axis = 0)
df_BusinessesPerCategories_pre['Percent'] = (df_BusinessesPerCategories_pre['review_count']/df_BusinessesPerCategories_pre['review_count'].sum())*100.00
df_BusinessesPerCategories_pre = df_BusinessesPerCategories_pre.sort_values(['Percent'], ascending = 0)

df_BusinessesPerCategories_pre['Cum_Percent'] = df_BusinessesPerCategories_pre['Percent'].cumsum(axis = 0)
df_BusinessesPerCategories_pre = df_BusinessesPerCategories_pre.sort_values(['Percent'], ascending = 0)

df_BusinessesPerCategories_pre.head(10)


# In[ ]:


#What are the top-10 business categories before we remove the businesses using more than 3 top-level labels
ax = sns.catplot(x="Percent", y="categories",kind="bar",data=df_BusinessesPerCategories_pre.head(10), aspect= 1.5)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle('Top 10 Business Categories by Total Review Counts (Before)') # can also get the figure from plt.gcf()


# We see that the most reviewed resturant category businesses are including at least one other label from the sub-categories, and there are plenty of variants/combinations of sub-categories used with the top-level resturant. For our analysis, we are only interested in the top-level category, for example, 'Resturant' or 'Shopping', not 'Mexican' or 'Chinese' resturant. We will discard the sub-category labels such as 'Mexican', 'Pizza', 'Sushi Bars' that are not among the 22 top-level catetogy labels as found in Yelp website, and focus on determining which of the 22 top-level categories businesses belong to before performing our categories based analyses.

# In[ ]:


#List of top-level business categories as foudn in Yelp website
major_categories = ['Active Life', 'Arts & Entertainment', 'Automotive', 'Beauty & Spas', 'Education', 'Event Planning & Services', 'Financial Services', 'Food','Health & Medical',
                    'Home Services', 'Hotels & Travel', 'Local Flavor', 'Local Services', 'Mass Media', 'Nightlife', 'Pets', 'Professional Services', 'Public Services & Government', 
                    'Real estate','Religious Organizations','Restaurants', 'Shopping']


# In[ ]:


#Creating two empty columns for major category keywords and the counts of such use for each business and categories they have used
biz_df['Count_MajorCategories'] = NaN
biz_df['MajorCategories'] = NaN

#Populating the values in the new columns for each business
biz_df['Count_MajorCategories'] = biz_df['categories'].apply(lambda x: len([value for value in major_categories if value in x]))
biz_df['MajorCategories'] = biz_df['categories'].apply(lambda x: [str(value) for value in major_categories if str(value) in x])

##Printing count of businesses with exactly one top-level category
print(len(biz_df[biz_df.Count_MajorCategories <= 3].sort_values('Count_MajorCategories'))/biz_df['business_id'].count()*100)

##Printing count of businesses with more than one to-level category
print(len(biz_df[biz_df.Count_MajorCategories > 3].sort_values('Count_MajorCategories'))/biz_df['business_id'].count()*100)


# In[ ]:


print(biz_df['Count_MajorCategories'].value_counts()/biz_df['business_id'].count()*100)

#Count of unique combinations of top-level business categories
print (biz_df['MajorCategories'].apply(tuple).nunique())

#Count of unique combinations of top-level business categories for businesses with no more than 3 such labels used
print (biz_df[biz_df.Count_MajorCategories > 3]['MajorCategories'].apply(tuple).nunique())


# We see that over 67.60% of businesses are using a single top-level business category, 25.69% are using two categories, and only 5.37% are using three cateories. 
# 
# Also, the unique categories are now down to 1602 from over 48,000 if we went by the original labels used by the businesses that included many different sub-category keywords. If we discard the businesses using more than 3 top-level categories, the unique counts lowers almost by half to 844 unique combinations.

# In[ ]:


#New lets look at the distribution of review counts across major categories
# df_BusinessesPerCategories_major = biz_df.groupby(['MajorCategories'])['review_count'].sum().reset_index().sort_values('review_count',ascending = False)
# df_BusinessesPerCategories_major['Percent'] = (df_BusinessesPerCategories_major['review_count']/df_BusinessesPerCategories_major['review_count'].sum())*100.00
# df_BusinessesPerCategories_major.head()

#New lets look at the distribution of review counts across major categories
df_BusinessesPerCategories_major = pd.DataFrame(biz_df.groupby(biz_df['MajorCategories'].map(tuple))['review_count'].sum())
df_BusinessesPerCategories_major.reset_index(level=0, inplace=True) #reset index to column
df_BusinessesPerCategories_major['Cum_review_count'] = df_BusinessesPerCategories_major['review_count'].cumsum(axis = 0)
df_BusinessesPerCategories_major['Percent'] = (df_BusinessesPerCategories_major['review_count']/df_BusinessesPerCategories_major['review_count'].sum())*100.00
df_BusinessesPerCategories_major = df_BusinessesPerCategories_major.sort_values(['Percent'], ascending = 0)

df_BusinessesPerCategories_major['Cum_Percent'] = df_BusinessesPerCategories_major['Percent'].cumsum(axis = 0)
df_BusinessesPerCategories_major = df_BusinessesPerCategories_major.sort_values(['Percent'], ascending = 0)

df_BusinessesPerCategories_major.head(10)


# In[ ]:


#Converting the list back to string because some operations and plots are not easy to deal with list types.
biz_df['MajorCategories'] = biz_df.MajorCategories.apply(', '.join)
pd.DataFrame(biz_df['MajorCategories'].unique()).head(10)


# In[ ]:


# #Plotting top 25 most reviewed businesses among all categories
ax = sns.catplot(x="review_count", y="name",data= biz_df.nlargest(25,'review_count'), 
                  kind="bar", hue = 'MajorCategories', dodge= False, height= 10 )

plt.subplots_adjust(top=0.9)
ax.fig.suptitle('Top 25 Most Reviewed Businesses And Categories Lables Used') # can also get the figure from plt.gcf()


# In[ ]:


#Drop the businesses that use more than three top-level business categories
to_drop = list(biz_df.query("Count_MajorCategories > 3")['MajorCategories'])
to_drop
biz_df = biz_df[~biz_df.MajorCategories.isin(to_drop)]
biz_df.head(3)


# In[ ]:


print (biz_df['MajorCategories'].nunique())


# There are 758 unique combination of 3 or less top-level categories used by businesses.

# In[ ]:


#Following turns into lists, .nunique() does not work on lists, so need to count list length of the items (!!USE THIS!!)
#biz_df['MajorCategories'] = biz_df.MajorCategories.map(lambda x: [i.strip() for i in sorted(x.split(", "))])


# In[ ]:


biz_df.head(5)


# Next, for the remaining 758 unique combinations of categories we will look at the which ones by most used and important by ranking them by the counts of businesses using them and the total reviews for all businesses in each categories.

# In[ ]:


#Finding the distrubution of the counts of businesses per category
df_BusinessesPerCategories = pd.DataFrame(biz_df['MajorCategories'].value_counts())
df_BusinessesPerCategories.reset_index(level=0, inplace=True) #reset index to column
df_BusinessesPerCategories.rename(columns={'MajorCategories':'Count_Businesses','index':'MajorCategories'}, inplace=True) #Renaming columns
df_BusinessesPerCategories['Cum_Count_Businesses'] = df_BusinessesPerCategories['Count_Businesses'].cumsum(axis = 0)

df_BusinessesPerCategories['Percent_Busnisses'] = (df_BusinessesPerCategories['Count_Businesses']/df_BusinessesPerCategories['Count_Businesses'].sum())*100.00
df_BusinessesPerCategories = df_BusinessesPerCategories.sort_values(['Percent_Busnisses'], ascending = 0)
df_BusinessesPerCategories = df_BusinessesPerCategories.reset_index(drop=True)

df_BusinessesPerCategories['Cum_Percent_Busnisses'] = df_BusinessesPerCategories['Percent_Busnisses'].cumsum(axis = 0)
df_BusinessesPerCategories = df_BusinessesPerCategories.sort_values(['Percent_Busnisses'], ascending = 0)

#df_categoriesDist['Cum_'] = df_categoriesDist['MajorCategories'].cumsum(axis = 0)
df_BusinessesPerCategories.head(10)


# In[ ]:


#Finding the distribution of review counts per category
df_ReviewsPerCategories = pd.DataFrame(biz_df.groupby(['MajorCategories'])['review_count'].sum())
df_ReviewsPerCategories.reset_index(level=0, inplace=True) #reset index to column
df_ReviewsPerCategories['Cum_review_count'] = df_ReviewsPerCategories['review_count'].cumsum(axis = 0)

df_ReviewsPerCategories['Percent_reviews'] = (df_ReviewsPerCategories['review_count']/df_ReviewsPerCategories['review_count'].sum())*100.00
df_ReviewsPerCategories = df_ReviewsPerCategories.sort_values(['Percent_reviews'], ascending = 0)
df_ReviewsPerCategories = df_ReviewsPerCategories.reset_index(drop=True)

df_ReviewsPerCategories['Cum_Percent_reviews'] = df_ReviewsPerCategories['Percent_reviews'].cumsum(axis = 0)
#df_ReviewsPerCategories = df_ReviewsPerCategories.sort_values(['Percent'], ascending = 0)

#df_categoriesDist['Cum_'] = df_categoriesDist['MajorCategories'].cumsum(axis = 0)
df_ReviewsPerCategories.head(10)


# Top 50 records contain over all the categories with over 92% of businesses as well as over 94% of all user reviews. Therefore, we will cut down the categories we will look at from over 758 to 50 by discarding records other than top 50 in the above datasets.

# In[ ]:


df_BusinessesPerCategories= df_BusinessesPerCategories.head(50)
df_ReviewsPerCategories = df_ReviewsPerCategories.head(50)


# In[ ]:


#Joining the reviews counts and business counts dataframe per business category
df_CtReviewsAndBiz_PerCat = df_ReviewsPerCategories.merge(df_BusinessesPerCategories, on='MajorCategories', how='inner')

#Adding weighted column Ct_ReviewPerBiz
df_CtReviewsAndBiz_PerCat['Weight'] = df_CtReviewsAndBiz_PerCat['Percent_reviews']*df_CtReviewsAndBiz_PerCat['Percent_Busnisses']

#Lets also add the differences in percent of business vs. percent of review and sort the df with this new column to help with vizualization later on
df_CtReviewsAndBiz_PerCat['Percent_Diff'] = df_CtReviewsAndBiz_PerCat['Percent_reviews']-df_CtReviewsAndBiz_PerCat['Percent_Busnisses']
df_CtReviewsAndBiz_PerCat = df_CtReviewsAndBiz_PerCat.sort_values(['Percent_Diff'], ascending = 0)

df_CtReviewsAndBiz_PerCat = df_CtReviewsAndBiz_PerCat.reset_index(drop=True)
df_CtReviewsAndBiz_PerCat.head(5)


# In[ ]:


import seaborn as sns
sns.set(style="white")

# Load the example mpg dataset
plt_data = df_CtReviewsAndBiz_PerCat.head(50)

# Plot miles per gallon against horsepower with other semantics
sns.relplot(x="Percent_reviews", y="Percent_Busnisses", hue="MajorCategories", size="Weight",
            sizes=(100, 1000), alpha=0.7, palette="muted", height=10, data=plt_data)


# With the scatterplot above, we can see that businesses using singular 'Resturant' keyword in category represent proportionally very high number of businesses as well as reivews. Another group of businesses using 'Nightlife, Resturants' and 'Food, Resurants' could be seen as a group with around 10-15% of reviews and 2.5-7.5% of businesses. 
# 
# Other six categories can be seen as a third cluster representing 5-9% of businesses with only 2-5% of reviews. Yet another less significant cluster appears to be businesses using categories that represent less than 2.5% of businesses and less than 4% of all reviews.
# 
# We can clearly see that 'Resturants' represent the highest categories of businesses in Yelp and also receive proportionally highest number of user reviews.
# 
# We will vizualize this using a differnt approach to try better understand the data by using bar plot.

# In[ ]:


plt_data = df_CtReviewsAndBiz_PerCat.head(50)

plt_data = plt_data[['MajorCategories','Percent_reviews','Percent_Busnisses']]
plt_data = plt_data.melt('MajorCategories', var_name='Group', value_name='Percent')
plt_data.head(5)


# In[ ]:


sns.set(style="whitegrid")

# Load the example Titanic dataset
titanic = sns.load_dataset("titanic")

# Draw a nested barplot to show survival for class and sex
g = sns.catplot(x="Percent", y="MajorCategories", hue="Group", data=plt_data,
                height=12, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Business Categories")


# In addition to information seen in the scatterplot above, the bar plot above shows a few additional information. To vizualize the differences in the percentage of total businesses in the given category vs. total percentage of reviews of all businesses in that category, the categories are sorted with the highest positive difference between percent reviews minus percent businesses on the top.
# 
# We can see that shopping has the highest negative difference- meaning, although this category represents 7.8% of businesses, they received only 2.8% of user reviews in Yelp, which is the highest difference among all the categories.

# In[ ]:


#Next lets also look at the reviews distribution by cities
df_reviewsPerCity = pd.DataFrame(biz_df.groupby(['city'])['review_count'].sum().sort_values(ascending=False))
df_reviewsPerCity.reset_index(level=0, inplace=True) #reset index to column
df_reviewsPerCity['Cum_review_count'] = df_reviewsPerCity['review_count'].cumsum(axis = 0)

df_reviewsPerCity['Percent_reviews'] = (df_reviewsPerCity['review_count']/df_reviewsPerCity['review_count'].sum())*100.00
df_reviewsPerCity = df_reviewsPerCity.sort_values(['Percent_reviews'], ascending = 0)
df_reviewsPerCity = df_reviewsPerCity.reset_index(drop=True)

df_reviewsPerCity['Cum_Percent_reviews'] = df_reviewsPerCity['Percent_reviews'].cumsum(axis = 0)

ax = sns.catplot(x="Percent_reviews", y="city",kind="bar",data=df_reviewsPerCity.head(25), height = 7)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle('Top 25 Cities by Total Review Counts') # can also get the figure from plt.gcf()

df_reviewsPerCity.head(5)


# In[ ]:


#Next lets also look at the business distribution by cities
df_bizPerCity = pd.DataFrame(biz_df.groupby(['city'])['business_id'].count().sort_values(ascending=False))
df_bizPerCity.reset_index(level=0, inplace=True) #reset index to column
df_bizPerCity['Cum_Biz_Count'] = df_bizPerCity['business_id'].cumsum(axis = 0)

df_bizPerCity['Percent_Biz'] = (df_bizPerCity['business_id']/df_bizPerCity['business_id'].sum())*100.00
df_bizPerCity = df_bizPerCity.sort_values(['Percent_Biz'], ascending = 0)
df_bizPerCity = df_bizPerCity.reset_index(drop=True)

df_bizPerCity['Cum_Percent_Biz'] = df_bizPerCity['Percent_Biz'].cumsum(axis = 0)

ax = sns.catplot(x="Percent_Biz", y="city",kind="bar",data=df_bizPerCity.head(25), height = 7)
plt.subplots_adjust(top=0.9)
ax.fig.suptitle('Top 25 Cities by Total Business Counts') # can also get the figure from plt.gcf()

df_reviewsPerCity.head(5)


# In[ ]:


#Joining the reviews counts and business counts dataframe per business category
df_CtReviewsAndBiz_PerCity = df_bizPerCity.merge(df_reviewsPerCity, on='city', how='inner')

# #Adding weighted column Ct_ReviewPerBiz
df_CtReviewsAndBiz_PerCity['Weight'] = df_CtReviewsAndBiz_PerCity['Percent_reviews']*df_CtReviewsAndBiz_PerCity['Percent_Biz']

#Lets also add the differences in percent of business vs. percent of review and sort the df with this new column to help with vizualization later on
df_CtReviewsAndBiz_PerCity['Percent_Diff'] = df_CtReviewsAndBiz_PerCity['Percent_reviews']-df_CtReviewsAndBiz_PerCity['Percent_Biz']
df_CtReviewsAndBiz_PerCity = df_CtReviewsAndBiz_PerCity.sort_values(['Percent_Diff'], ascending = 0)

# df_CtReviewsAndBiz_PerCity = df_CtReviewsAndBiz_PerCity.reset_index(drop=True)
df_CtReviewsAndBiz_PerCity.head(10)


# In[ ]:


import seaborn as sns
sns.set(style="white")

# Load the example mpg dataset
plt_data = df_CtReviewsAndBiz_PerCity.head(50)

# Plot miles per gallon against horsepower with other semantics
sns.relplot(x="Percent_reviews", y="Percent_Biz", hue="city", size="Weight",
            sizes=(100, 1000), alpha=0.7, palette="muted", height=10, data=plt_data)


# In[ ]:


plt_data = df_CtReviewsAndBiz_PerCity.head(20)

plt_data = plt_data[['city','Percent_reviews','Percent_Biz']]
plt_data = plt_data.melt('city', var_name='Group', value_name='Percent')

sns.set(style="whitegrid")

# Draw a nested barplot to show survival for class and sex
g = sns.catplot(x="Percent", y="city", hue="Group", data=plt_data,
                aspect=2, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("City")


# Next, we will repeat the analysis per city per category

# In[ ]:


df_BusinessesPerCategories_perCity = (biz_df.groupby(['city','MajorCategories']).agg({'business_id':'count', 'review_count': 'sum'}).reset_index().rename(columns={'business_id':'biz_count','MajorCategories':'MajorCat'}))


# In[ ]:


#Order by cities and biz counts to populate ordered cum  counts of biz per city
df_BusinessesPerCategories_perCity = df_BusinessesPerCategories_perCity.sort_values(['city','biz_count'], ascending = [0,0])
df_BusinessesPerCategories_perCity['Cum_biz_ct'] = df_BusinessesPerCategories_perCity.groupby(['city'])['biz_count'].transform(pd.Series.cumsum)

#Order by cities and review counts to populate ordered cum  counts of reviews per city
df_BusinessesPerCategories_perCity = df_BusinessesPerCategories_perCity.sort_values(['city','review_count'], ascending = [0,0])
df_BusinessesPerCategories_perCity['Cum_review_ct'] = df_BusinessesPerCategories_perCity.groupby(['city'])['review_count'].transform(pd.Series.cumsum)

#Populating percent business and cumulitive percent business columns
total = df_BusinessesPerCategories_perCity.groupby('city')['biz_count'].transform('sum')
df_BusinessesPerCategories_perCity['biz_pc'] = (df_BusinessesPerCategories_perCity['biz_count']/total)*100
#Sort the df by percent biz before populating cumulitive biz percent
df_BusinessesPerCategories_perCity = df_BusinessesPerCategories_perCity.sort_values(['city','biz_pc'], ascending = [0,0])
df_BusinessesPerCategories_perCity['Cum_biz_pc'] = df_BusinessesPerCategories_perCity.groupby(['city'])['biz_pc'].transform(pd.Series.cumsum)

#Populating percent reviews and cumulitive reviews  columns
total = df_BusinessesPerCategories_perCity.groupby('city')['review_count'].transform('sum')
df_BusinessesPerCategories_perCity['review_pc'] = (df_BusinessesPerCategories_perCity['review_count']/total)*100
#Sort the df by percent reviews before populating cumulitive reviews percent
df_BusinessesPerCategories_perCity = df_BusinessesPerCategories_perCity.sort_values(['city','review_pc'], ascending = [0,0])
df_BusinessesPerCategories_perCity['Cum_review_pc'] = df_BusinessesPerCategories_perCity.groupby(['city'])['review_pc'].transform(pd.Series.cumsum)


# In[ ]:


n_top_categories = 10
df_topX_reviewCats_perCity = df_BusinessesPerCategories_perCity.loc[df_BusinessesPerCategories_perCity.
                                                                    groupby('city')['review_pc'].nlargest(n_top_categories)
                                                                    .reset_index()['level_1']]

n_top_cities = 10
cities_to_analyze = df_CtReviewsAndBiz_PerCity.nlargest(n_top_cities, 'review_count')['city']
plt_data = df_topX_reviewCats_perCity[df_topX_reviewCats_perCity.city.isin(cities_to_analyze)]
plt_data = plt_data[['city','MajorCat','biz_pc','review_pc']]

#plt_data.melt('name').sort_values('name')
#plt_data = plt_data[['city','MajorCat','biz_count']]
#plt_data = plt_data.melt('city', var_name='Group', value_name='Percent')
plt_data[plt_data.city == 'Las Vegas'].head(5)


# In[ ]:


sns.set(style="whitegrid")

# Draw a nested barplot to show survival for class and sex
g = sns.catplot(x="review_pc", y="MajorCat", col = "city",col_wrap = 3 ,data=plt_data,
                aspect=0.9,height=5, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("City")


# In[ ]:


analysis_categories = ['Restaurants','Food,Restaurants','Nightlife,Restaurants','Food,Nightlife,Restaurants']


# In[ ]:


df_biz_Resturants = biz_df[biz_df.MajorCategories.isin(analysis_categories)]
print(df_biz_Resturants['MajorCategories'].value_counts())
df_biz_Resturants.head(5)


# In[ ]:


df_Resturants_LV = df_biz_Resturants[df_biz_Resturants['city'] == 'Las Vegas']


# In[ ]:


sns.regplot(x="latitude", y="longitude",data=df_Resturants_LV,scatter_kws={"color":"darkred","alpha":0.2,"s":3},fit_reg=False)


# In[ ]:


# Lets also load the reviews data into pandas dataframe and look into the size, and structure of the dataframe, columns and their data types
# review_file = open('../input/yelp_academic_dataset_review.json',encoding="utf8")
# review_df = pd.DataFrame([json.loads(next(review_file)) for x in range(5200000)])
# review_file.close()
# review_df.info()


# In[ ]:


# review_df['stars'].value_counts()


# In[ ]:


# # Lets observe the reviews dataframe as well
# review_df.head()


# In[ ]:


# #Drop the reviews of businesses that do not belong to the categories we are interested in.
# to_keep = list(biz_df['business_id'].unique())
# to_keep
# review_df = review_df[review_df.business_id.isin(to_keep)]
# review_df.info()

# del review_df
# del restaurant_reviews

# del [[review_df,restaurant_reviews]]
# gc.collect()


# In[ ]:


#join dataframe
restaurant_reviews = biz_df.merge(review_df, on='business_id', how='inner')
restaurant_reviews.info()

