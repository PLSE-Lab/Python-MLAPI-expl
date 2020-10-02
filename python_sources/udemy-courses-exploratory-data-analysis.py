#!/usr/bin/env python
# coding: utf-8

# # Udemy Courses - Exploratory Data Analysis
# 
# 
# 
# ![](https://about.udemy.com/wp-content/themes/wp-about/assets/images/udemy-logo-red.svg)
# 
# This dataset includes an abbreviated collection of online courses scraped from Udemy.com. Udemy currently offers over 150,000 courses that cover a wide range of topics. This dataset includes approximately 3,600 courses uploaded between 2011-07-09 and 2017-07-06. 
# 
# Courses included in this dataset come from four subject areas:   
# * Web Development
# * Business Finance
# * Musical Instruments
# * Graphic Design
# 
# This notebook will provide a simple exploratory data analysis of the dataset.

# ## Import Libaries & Load Data

# In[ ]:


get_ipython().system('pip install joypy -q')

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
from matplotlib import cm
import joypy
import seaborn as sns
plt.rcParams["figure.figsize"] = (16,8)

from scipy.stats import pearsonr
from scipy.stats import spearmanr


# In[ ]:


df = pd.read_csv('../input/udemy-courses/udemy_courses.csv')


# # General Analysis

# In[ ]:


df.head()


# ### Check Dataset for NaN / Null Values

# In[ ]:


print('The raw dataset has {} rows and {} columns.'.format(df.shape[0],df.shape[1]))
print('-----------------')
print('Columns with NaN values: ')
nan_cols = df.isna().any()[df.isna().any() == True]
if len(nan_cols)>0:
    print(nan_cols) 
else:
    print('none')
print('-----------------')
print('Columns with null values: ')
null_cols = df.isnull().any()[df.isnull().any() == True]
if len(null_cols)>0:
    print(null_cols)
else:
    print('none')


# ### Remove Duplicate Values

# In[ ]:


orig_rows = df.shape[0]
df.drop_duplicates(inplace=True)
print('After removing duplicate rows, the dataset has {} rows remaining. {} duplicate rows were removed.'.format(df.shape[0], orig_rows - df.shape[0]))


# # Course Subjects

# In[ ]:


ax = sns.barplot(df.subject.value_counts().index, df.subject.value_counts().values)
ax.set(title='Number of Courses by Subject', xlabel='Subject', ylabel='Number of Courses')
plt.show()


# # Course Titles
# 
# ### Check for Duplicate Titles

# In[ ]:


print('The following courses have duplicate titles:')
for item in df.course_title.value_counts()[df.course_title.value_counts() > 1].index: print('\t' + item) 


# Let's look at a couple examples of courses with duplicate titles to see if they represent truly different courses with the same name or if they are different versions of the same course that have been updated separately.

# In[ ]:


df.loc[df.course_title == 'Acoustic Blues Guitar Lessons']


# In[ ]:


df.loc[df.course_title == 'Creating an animated greeting card via Google Slides']


# It looks like some of the courses with duplicate titles are truly unique, while others may be updated versions of the same course. 

# ### Coures Prices
# 
# Courses on Udemy are either free or paid. First we will look at the breakdown of paid vs. free cources both across the entire dataset as well as grouped by course subject.

# In[ ]:


print('There are {} paid courses and {} free courses.'.format(df[df.is_paid == True].shape[0], df[df.is_paid == False].shape[0]))


# In[ ]:


subject_names = [x for x in df.subject.unique()]
total_courses = [x for x in df.subject.value_counts().values]
paid_courses = [x for x in df[df.is_paid == True].subject.value_counts().values]
free_courses = [x for x in df[df.is_paid == False].subject.value_counts().values]
count_values = np.array([total_courses, paid_courses, free_courses])


# In[ ]:


pay_prop = pd.DataFrame(count_values, columns = df.subject.value_counts().index.to_list())
pay_prop['course_breakdown'] = ['All Courses', 'Paid Courses', 'Free Courses']
pay_prop.set_index('course_breakdown', inplace=True)
ax = pay_prop.plot(kind='barh', 
                   stacked=True,
                   title='Course Proportion Breakdown (Free, Paid, and Total)')
ax.set_xlabel('Number of Courses')
ax.set_ylabel('')
plt.show()


# Let's take a look at the distribution of prices across all of the courses. First, we'll create a boxplot to get an idea of our interquarterile range. Since there are a hard limits on the maximum (\$200) and minimum (\$0) prices for Udemy courses, we have clusters of values at each extreme. As a result, our boxplot will not show any outliers.

# In[ ]:


ax = sns.boxplot(y=df.price, orient='h', width=0.2)
ax.set(xlabel='Course Price', title='Distribution of Course Prices (All Courses)')
plt.show()


# Next we'll look at some general statistics for the price data.

# In[ ]:


price_stats = df.price.describe()

print('Course prices range from ${:.2f} to ${:.2f}.'.format(round(price_stats['min'], 2), round(price_stats['max'], 2)))
print('The mean course price is ${:.2f}, and the standard deviation is ${:.2f}.'.format(round(price_stats['mean'], 2), round(price_stats['std'], 2)))
print('The median coure price is ${:.2f}.'.format(round(price_stats['50%'], 2)))
print('The middle 50% of course prices are between ${:.2f} and ${:.2f}.'.format(round(price_stats['25%'], 2), round(price_stats['75%'], 2)))


# In[ ]:


ax = sns.distplot(df.price)
ax.set(title='Distribution of Course Prices (All Courses)', xlabel='Course Price')
plt.show()


# ### Price Data by Subject

# In[ ]:


ax = sns.boxplot(x=df.subject, y=df.price)
ax.set(title='Course Price Distribution by Subject', xlabel='Subject', ylabel='Course Price')
plt.show()


# In[ ]:


price_summary = df.groupby('subject').describe().price.reset_index(drop=False)
price_dict = df.price.describe().to_dict()
price_dict['subject'] = 'ALL COURSES'
price_summary.append(price_dict, ignore_index=True)

price_summary


# Course prices for Web Development and Business Finance are higher on average than prices for courses related to Musical Instruments or Graphic Design. Web Development and Business Finance also show the most variation in price.
# 
# Next we'll visualize the price distributions for each of the four subjects using a ridgeline plot.

# In[ ]:


fig, ax = joypy.joyplot(df, by='subject', column='price', figsize=(12, 6), title='Distribution of Course Prices by Subject')
plt.xlabel('Course Price')
plt.show()


# We can see that the price distribution each subject includes two peaks: one near the median price for the subject and one at \$200 USD. Since this is the maximum course price that Udemy allows, there are clusters of courses with this price in each subject group.

# # Top 10 Courses 
# 
# ## By Number of Subscribers

# In[ ]:


top_10_by_sub = df.num_subscribers.groupby(df.subject).nlargest(10).reset_index(drop=False)
top_10_by_sub['course_title'] = top_10_by_sub.level_1.apply(lambda x: df.iloc[x].course_title)
top_10_by_sub.drop('level_1', axis=1, inplace=True)

fig, axs = plt.subplots(4, 1, figsize=(14,16))
plt.subplots_adjust(hspace=0.6)

sns.barplot(data=top_10_by_sub.loc[top_10_by_sub.subject == subject_names[0]], x='num_subscribers', y='course_title', ax=axs[0], color='b')
sns.barplot(data=top_10_by_sub.loc[top_10_by_sub.subject == subject_names[1]], x='num_subscribers', y='course_title', ax=axs[1], color='g')
sns.barplot(data=top_10_by_sub.loc[top_10_by_sub.subject == subject_names[2]], x='num_subscribers', y='course_title', ax=axs[2], color='r')
sns.barplot(data=top_10_by_sub.loc[top_10_by_sub.subject == subject_names[3]], x='num_subscribers', y='course_title', ax=axs[3], color='orange')

for i in range(len(axs)):
    axs[i].set(ylabel='', xlabel='Number of Subscribers', title='Top 10 {} Courses Based on Number of Subscribers'.format(subject_names[i]), xlim=(0,300000))


# ## By Number of Reviews

# In[ ]:


top_10_by_rev = df.num_reviews.groupby(df.subject).nlargest(10).reset_index(drop=False)
top_10_by_rev['course_title'] = top_10_by_rev.level_1.apply(lambda x: df.iloc[x].course_title)
top_10_by_rev.drop('level_1', axis=1, inplace=True)

fig, axs = plt.subplots(4, 1, figsize=(14,16))
plt.subplots_adjust(hspace=0.6)

sns.barplot(data=top_10_by_rev.loc[top_10_by_rev.subject == subject_names[0]], x='num_reviews', y='course_title', ax=axs[0], color='b')
sns.barplot(data=top_10_by_rev.loc[top_10_by_rev.subject == subject_names[1]], x='num_reviews', y='course_title', ax=axs[1], color='g')
sns.barplot(data=top_10_by_rev.loc[top_10_by_rev.subject == subject_names[2]], x='num_reviews', y='course_title', ax=axs[2], color='r')
sns.barplot(data=top_10_by_rev.loc[top_10_by_rev.subject == subject_names[3]], x='num_reviews', y='course_title', ax=axs[3], color='orange')

for i in range(len(axs)):
    axs[i].set(ylabel='', xlabel='Number of Reviews', title='Top 10 {} Courses Based on Number of Reviews'.format(subject_names[i]), xlim=(0,30000))


# ## By Engagement (Number of Reviews / Number of Subscribers)
# 
# For this analysis, we will define engagement as the ratio of number of reviews to number of subscribers. A course with high engagement has a higher proportion of subscribers who write reviews than a course with low engagement. 
# 
# When reviewing engagement, we considered only courses with a minimum of 50 subscribers

# In[ ]:


df['engagement'] = df['num_reviews'] / df['num_subscribers']
df.fillna(0, inplace=True)
df.reset_index(drop=True, inplace=True)

top_10_by_eng = df.loc[df.num_subscribers > 50].engagement.groupby(df.subject).nlargest(10).reset_index(drop=False)
top_10_by_eng['course_title'] = top_10_by_eng.level_1.apply(lambda x: df.iloc[x].course_title)
top_10_by_eng.drop('level_1', axis=1, inplace=True)

fig, axs = plt.subplots(4, 1, figsize=(14,16))
plt.subplots_adjust(hspace=0.6)

sns.barplot(data=top_10_by_eng.loc[top_10_by_eng.subject == subject_names[0]], x='engagement', y='course_title', ax=axs[0], color='b')
sns.barplot(data=top_10_by_eng.loc[top_10_by_eng.subject == subject_names[1]], x='engagement', y='course_title', ax=axs[1], color='g')
sns.barplot(data=top_10_by_eng.loc[top_10_by_eng.subject == subject_names[2]], x='engagement', y='course_title', ax=axs[2], color='r')
sns.barplot(data=top_10_by_eng.loc[top_10_by_eng.subject == subject_names[3]], x='engagement', y='course_title', ax=axs[3], color='orange')

for i in range(len(axs)):
    axs[i].set(ylabel='', xlabel='Number of Subscribers', title='Top 10 {} Courses Based on Engagement'.format(subject_names[i]), xlim=(0,0.4))


# Note that the distributions of engagement are much closer across the course subjects than the number of reviews or subscribers.

# # Paid vs. Free Courses

# ## Number of Subscribers

# In[ ]:


paid_sub = df.groupby(['subject', 'is_paid']).agg('mean')['num_subscribers']
paid_sub = pd.DataFrame(paid_sub)
paid_sub.reset_index(drop=False, inplace=True)

g = sns.FacetGrid(paid_sub, col='subject', height=6, aspect=0.75)
g = g.map(sns.barplot, 'is_paid', 'num_subscribers')


# ## Number of Reviews

# In[ ]:


paid_rev = df.groupby(['subject', 'is_paid']).agg('mean')['num_reviews']
paid_rev = pd.DataFrame(paid_rev)
paid_rev.reset_index(drop=False, inplace=True)

g = sns.FacetGrid(paid_rev, col='subject', height=6, aspect=0.75)
g = g.map(sns.barplot, 'is_paid', 'num_reviews')


# ## Engagement

# In[ ]:


paid_eng = df.groupby(['subject', 'is_paid']).agg('mean')['engagement']
paid_eng = pd.DataFrame(paid_eng)
paid_eng.reset_index(drop=False, inplace=True)

g = sns.FacetGrid(paid_eng, col='subject', height=6, aspect=0.75)
g = g.map(sns.barplot, 'is_paid', 'engagement')


# Note that, although the average numbers of reviews and subscribers are higher for free courses, engagement is higher for paid courses.

# # Correlations
# 
# ## Number of Subscribers and Number of Reviews

# In[ ]:


# NOTE: We are removing two outliers that contain > 150,000 subscribers
ax = sns.regplot(data=df.loc[df.num_subscribers < 150000], x='num_subscribers', y='num_reviews')
ax.set(title='Number of Reviews vs. Number of Subscribers', xlabel='Number of Subscribers', ylabel='Number of Reviews')
plt.show()


# In[ ]:


from scipy.stats import pearsonr
from scipy.stats import spearmanr

sub_rev_pearson = pearsonr(df.loc[df.num_subscribers < 150000].num_subscribers, df.loc[df.num_subscribers < 150000].num_reviews)[0]
print("Pearson's correlation between number of subscribers and number of reviews (outlers removed): {}".format(sub_rev_pearson))
print('------------------')
sub_rev_spearman = spearmanr(df.loc[df.num_subscribers < 150000].num_subscribers, df.loc[df.num_subscribers < 150000].num_reviews)[0]
print("Spearman's correlation between number of subscribers and number of reviews (outlers removed): {}".format(sub_rev_spearman))


# ## Number of Lectures and Content Duration

# In[ ]:


ax = sns.regplot(data=df, x='num_lectures', y='content_duration')
ax.set(title='Number of Lectures vs. Content Duration', xlabel='Number of Lectures', ylabel='Content Duration (hrs)')
plt.show()


# In[ ]:


sub_rev_pearson = pearsonr(df.num_lectures, df.content_duration)[0]
print("Pearson's correlation between number of lectures and content duration: {}".format(sub_rev_pearson))
print('------------------')
sub_rev_spearman = spearmanr(df.num_lectures, df.content_duration)[0]
print("Spearman's correlation between number of lectures and content duration: {}".format(sub_rev_spearman))


# # Course Level

# In[ ]:


ax = sns.countplot(df.level)
ax.set(title='Number of Courses by Level (All Courses)', xlabel='Level', ylabel='Number of Courses')
plt.show()


# In[ ]:


fig, axs = plt.subplots(2, 2, figsize=(16,10))
plt.subplots_adjust(hspace=0.4)

sns.countplot(df.loc[df.subject == subject_names[0]].level, ax=axs[0][0], color='b')
sns.countplot(df.loc[df.subject == subject_names[1]].level, ax=axs[0][1], color='g')
sns.countplot(df.loc[df.subject == subject_names[2]].level, ax=axs[1][0], color='r')
sns.countplot(df.loc[df.subject == subject_names[3]].level, ax=axs[1][1], color='orange')

axs[0][0].set(title='Number of {} Courses by Level'.format(subject_names[0]), ylabel='Number of Courses', xlabel='Course Level')
axs[0][1].set(title='Number of {} Courses by Level'.format(subject_names[1]), ylabel='Number of Courses', xlabel='Course Level')
axs[1][0].set(title='Number of {} Courses by Level'.format(subject_names[2]), ylabel='Number of Courses', xlabel='Course Level')
axs[1][1].set(title='Number of {} Courses by Level'.format(subject_names[3]), ylabel='Number of Courses', xlabel='Course Level')
plt.show()


# # Content Duration

# In[ ]:


df.content_duration.describe()


# In[ ]:


fig, ax = plt.subplots(figsize=(20,8))
ax = sns.boxplot(y=df.content_duration, orient='h', width=0.2)
ax.set(title='Content Duration Distribution', xlabel='Content Duration (hrs)')
plt.show()


# # Published Timestamp

# In[ ]:


df.published_timestamp = pd.to_datetime(df.published_timestamp)


# In[ ]:


df['year_published'] = df.published_timestamp.apply(lambda x: x.year)
df['month_published'] = df.published_timestamp.apply(lambda x: x.month)
df['day_of_week_published'] = df.published_timestamp.apply(lambda x: x.dayofweek)


# In[ ]:


year_group = df.groupby('year_published').agg('count').course_id
month_group = df.groupby('month_published').agg('count').course_id
day_of_week_group = df.groupby('day_of_week_published').agg('count').course_id


# In[ ]:


ax = sns.barplot(x=year_group.index, y=year_group.values)
ax.set(title='Number of Courses Added by Year', xlabel='Year', ylabel='Number of Courses')
plt.show()


# It should be noted that 2011 only includes July - December, while 2017 only includes January - July. As a result the values for these years may appear lower than normal.

# In[ ]:


ax = sns.barplot(x=month_group.index, y=month_group.values)
ax.set(title='Number of Courses Added by Month', xlabel='Month', ylabel='Number of Courses')
plt.show()


# We can see an intresting seasonal upload pattern, with a peak in May each year and a valley in September. 

# In[ ]:


day_of_week_group.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ax = sns.barplot(x=day_of_week_group.index, y=day_of_week_group.values)
ax.set(title='Number of Courses Added by Day of Week', xlabel='Day of Week', ylabel='Number of Courses')
plt.show()


# We also see that more courses are uploaded on Monday than any other day of the week, while Saturday and Sunday see the fewest course uploads. 

# # Language
# 
# Course language classification was not included as part of the dataset, but we will try and determine it using the course title. This will not be 100% accurate since we do not have a lot of text to work with. There are also some courses with an English title that advertise the course itself is taught in a different language. Here are a few examples:

# In[ ]:


df[df.course_title.str.contains('Urdu')]


# In[ ]:


get_ipython().system('pip install langdetect')


# We will use the langdetect module for language identification. This module returns the classified ISO 639-1 code for each language. 
# 
# I initially used langid, but I encountered too many misclassifications for it to be useful. I also looked at TextBlob. It works well but it limits the number of requests that can be made since it uses Google. 
# 
# I'm not sure which language detection module works best ([this StackOverflow post](https://stackoverflow.com/questions/39142778/python-how-to-determine-the-language) lists a bunch of them), but I also didn't have to install anything new to use langid, which is a plus.

# In[ ]:


from langdetect import detect

df['language'] = df.course_title.apply(lambda x: detect(x))


# In[ ]:


df.language.value_counts()


# 
# We'll look at the low-frequency languages (< 100 observations) to try and correct some misclassifications. There are tons of mistakes in the higher-frequency languages as well (take a look at 'de', 'it' and 'fr'), but to identify them all would be time consuming. The language classification algorithm seems to have an especially hard time with the Web Development subject. The names of different types of software and programming languages are unfamiliar, so the classification algorithm gets confused. Ex. 'Google Blogger Course.'
# 
# I did not include the check of each low-frequency language, but feel free to repeat it. My knowledge of languages outside of English is pretty limited since I'm an American rube, so I may have made some mistakes. In some cases, like 'ro' and 'ca', there were multiple languages represented. I classified all of them as English to simplify things.

# In[ ]:


misclass_lang_en = ['sw', 'vi', 'hr', 'et', 'id', 'sv', 'da', 'ro', 'af', 'nl', 'tl', 'no', 'ca']

df['language'] = df.language.apply(lambda x: 'en' if x in misclass_lang_en else x)
df.loc[df.course_title.str.contains('Urdu'), 'language'] = 'ur'


# Now let's take a look at the top 5 languages other than English that were represented in the course titles. I removed English since it is the most common language represented.

# In[ ]:


ax = sns.barplot(df.language.value_counts().values[1:6], df.language.value_counts().index[1:6])
ax.set(title='Top 5 Non-English Languages Represented in Course Titles', xlabel='Approx. Number of Courses', ylabel='ISO 639-1 Language Code')
plt.show()


# # Course Title Unigrams

# In[ ]:


from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

eng_stop_words = stopwords.words('english')
spa_stop_words = stopwords.words('spanish')

from nltk.tokenize import word_tokenize


# We'll remove Spanish and English stopwords since these two languages make up 92% (3362/3672) of the courses.

# In[ ]:


words = [w.lower() for w in word_tokenize(" ".join(df.course_title.values.tolist()))]
words_nostop = [w for w in words if (w.isalpha()) and ((w not in eng_stop_words) and (w not in spa_stop_words))]


# In[ ]:


common_counter = Counter(words_nostop)
unigram_counts = common_counter.most_common()
top_unigrams = [x[0] for x in unigram_counts]
top_unigram_counts = [x[1] for x in unigram_counts]

ax = sns.barplot(top_unigrams[:10], top_unigram_counts[:10])
ax.set(title='Top Unigrams in Course Titles (All Courses)', xlabel='Unigram', ylabel='Count')
plt.show()


# # More to Come!
# 
# This notebook is a work in progress, and I will add some additional analyses in near future.
# 
# ### Possible Future Work:
# 
# * Parse course titles and perform some basic NLP - note that there are multiple languages represented, with a large variety of character tokens. 
# * Build a course name generator.
# * Use LDA or another approach to break down the subjects even further. For example, identify the most popular languages that are being taught in Web Development courses or the most popular musical instruments.
# * Create a ranking metric using price, subscriber, review, and duration data to provide an overall assessment of course quality.
