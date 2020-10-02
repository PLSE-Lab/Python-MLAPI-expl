#!/usr/bin/env python
# coding: utf-8

# Hi everyone!
# 
# This is my first public kernel and I figured I would start with a deep dive into the data and try to make it as newbie friendly as possible. Hopefully this helps y'all in your model building!
# 
# Please leave a comment if you found this helpful/interesting and if you have any suggestions!
# 
# Updates:
# I added a few more examples of using pandas directly instead of writing code to do things manually. And I've added a heatmap of some correlations at the end.

# In[1]:


# Let's import everything we'll be using. Keep it all at the top to make your life easy.
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os


# In[2]:


# Where are our files? Check down below! Don't forget the ../input when you try to load them in.
print(os.listdir("../input"))


# In[3]:


train = pd.read_csv('../input/train.csv')
resource = pd.read_csv('../input/resources.csv')


# In[4]:


print(train.shape)
print(resource.shape)


# I always like to do a quick shape on any data I read in since that can tell me a little bit of what to expect.  Right off the bat I can see that our training data has 182,080 examples and a total of 16 categories. However, our resource dataset has 1,541,272 rows and 4 categories. Since the number of rows in each aren't the same then we immediately know that these aren't 1-to-1 and I'll have to be careful using the resource data in my final algorithm.
# 
# Now let's look at the head of both data sets to quickly orient ourselves.

# In[5]:


train.head()


# In[6]:


resource.head()


# From looking at the above we probably have a decent sense of the data types in our data set, but we should know for sure and not make assumptions.

# In[7]:


train.dtypes


# In[8]:


resource.dtypes


# "object" is pandas speak for a str data type. So we see that we have a lot of strings, but not exclusively strings. The things we expect to be numbers (like quantity and price) are loaded in as numbers and of the types that make most sense (int and float, respectively for quantity and price).

# # Categories in the DonorChoose Data Set
# 
# Let's look at the descriptions of each field taken directly from the competition page:
# 
# ## Data fields
# 
# train.csv (and test.csv):
# 
# *     id - unique id of the project application
# *     teacher_id - id of the teacher submitting the application
# *     teacher_prefix - title of the teacher's name (Ms., Mr., etc.)
# *     school_state - US state of the teacher's school
# *     project_submitted_datetime - application submission timestamp
# *     project_grade_category - school grade levels (PreK-2, 3-5, 6-8, and 9-12)
# *     project_subject_categories - category of the project (e.g., "Music & The Arts")
# *     project_subject_subcategories - sub-category of the project (e.g., "Visual Arts")
# *     project_title - title of the project
# *     project_essay_1 - first essay*
# *     project_essay_2 - second essay*
# *     project_essay_3 - third essay*
# *     project_essay_4 - fourth essay*
# *     project_resource_summary - summary of the resources needed for the project
# *     teacher_number_of_previously_posted_projects - number of previously posted applications by the submitting teacher
# *     project_is_approved - whether DonorsChoose proposal was accepted (0="rejected", 1="accepted") (train.csv only)
# 
# Note: Prior to May 17, 2016, the prompts for the essays were as follows:
# 
# *      project_essay_1: "Introduce us to your classroom"
# *      project_essay_2: "Tell us more about your students"
# *      project_essay_3: "Describe how your students will use the materials you're requesting"
# *      project_essay_4: "Close by sharing why your project will make a difference"
# 
# Starting on May 17, 2016, the number of essays was reduced from 4 to 2, and the prompts for the first 2 essays were changed to the following:
# 
# *     project_essay_1: "Describe your students: What makes your students special? Specific details about their background, your neighborhood, and your school are all helpful."
# *     project_essay_2: "About your project: How will these materials make a difference in your students' learning and improve their school lives?"
# 
# For all projects with project_submitted_datetime of 2016-05-17 and later, the values of project_essay_3 and project_essay_4 will be NaN.
# 
# resources.csv:
# 
# Proposals also include resources requested. Each project may include multiple requested resources. Each row in resources.csv corresponds to a resource, so multiple rows may tie to the same project by id.
# 
# *     id - unique id of the project application; joins with test.csv. and train.csv on id
# *     description - description of the resource requested
# *     quantity - quantity of resource requested
# *     price - price of resource requested
# 

# The description immediately tells us that we're going to have NaN's (Not a Number) in our data. When we looked at the train.head() we saw a bunch of NaN's, and now we know why. It isn't missing data from a teacher not including it, it is simply the guidelines. That means we'll have to implement a way of dealing with it. If we read the essay descriptions it sounds like essay 1 and 2 in the old format correspond well to essay 1 in the new format and likewise for old essays 3 and 4 with the new essay 2. 
# 
# When it comes to making an algorithm I will tentatively choose to merge the essays as appropriate.
# 
# Do we have other NaNs?

# In[9]:


for label in list(train.columns.values):
    print(f'{label} has {sum(train[label].isna())}')


# In[18]:


# Using this pandas code you can infer how many nan's there are. 
# One line of code instead of my 2 above, plus this gives more info!

#train.info()


# We're good on the NaN front! Except there are 4 in teacher_prefix.

# In[ ]:


# If we wanted to look at them in depth then we'd use the following code. 
# For the sake of length I'll omit the output since it isn't particularly interesting.
#idx = np.where(train['teacher_prefix'].isna() == True)[0]
#print(train.iloc[idx])


# Let's look at the categorical columns. How many different categories do we have? Do we see anything weird?

# In[10]:


for label in ['teacher_prefix', 'school_state', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories']:
    total = 0
    print(f'=== {label} ===')
    for i, item in enumerate(train[label].unique()):
        count = len(np.where(train[label] == item)[0])
        print('{}: {}'.format(item, count))
        total += count
    print(f'== Total categories: {i+1} and {total} values out of {len(train)}===\n')


# In[ ]:


# Here is the slick pandas way to do the above code. 2 lines versus 8! 
# Note that doing it this way doesn't show the nan's in the teacher_prefix, 
# so you still want to make sure you use something to count nan's!

#for label in ['teacher_prefix', 'school_state', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories']:
#    print(train[label].value_counts())


# In[22]:


train.describe(include=['O'])


# It's interesting to note that there are repeated essays. Clearly there are teachers sending in the same, or similar, applications multiple times. I can imagine an argument for filtering these out, but currently I don't plan to do so.

# Let's go through feature by feature and see what we learn!
# 
# ## Teacher Prefix
# There are 6 different prefixes, but at the end of the list we see a "nan" with 0 instances. Those are the same nan we saw above but it is odd that they aren't being counted when done this way, but this is why I implemented the total counts just to be sure.  We also see that Mr is dwarfed by Ms + Mrs. Pending investigation into the nans, I'm contemplating throwing out this column anyways since I like the idea of my algorithm being gender-blind.
# ## School States
# All 50 states are represnted plus 1 for Washington D.C.! From a quck skim it is clear that the applications are not uniform in amount across states. To go one step further in our analysis we could plot these numbers vs the population of each state and see if there is any correlation.
# 
# ## Grades
# There are 4 grade categories and each one has a healthy amount of data though note that as the grade level goes up the number of entries go down.
# 
# ## Subject Categories
# There are a lot of subject categories at 51! Some of them have tens of thousands of entries, but there is also one with only a single entry. That might be problematic and one should consider combining categories since there is strong overlap.
# 
# ## Subject Subcategories
# And finally we have 407 subcategories. No category has a whopping amount from first glance, but many categories have 10 or less entries. My gut reaction is that subcategory won't be a very good feature. Also, in the subcategories I notice two blank lines. This seems odd since it is a blank line and doesn't have a corresponding number so it doesn't seem to just be an instance of an empty string as a subcategory. This also warrants further investigation, especially if we want to use subcategories as a feature!

# Now let's take a peek at some numerical data such as from describe() and some histograms of how many teachers have previously submitted projects.

# In[20]:


train.describe()


# On average a teacher submits about a dozen projects, but there's a pretty big spread.  Also a huge percentage of applications are accepted.

# In[21]:


plt.figure()
plt.hist(train['teacher_number_of_previously_posted_projects'], bins=30)
plt.title('Histogram Counting # of Teachers that Previously Posted Projects')
plt.xlabel('Projects')
plt.ylabel('Count')
plt.show()

nonzeroidx = np.where(train['teacher_number_of_previously_posted_projects'] != 0 )[0]
plt.figure()
plt.hist(train['teacher_number_of_previously_posted_projects'].iloc[nonzeroidx], bins=30)
plt.title('Histogram Counting # of Teachers that Previously Posted Projects (teachers with 0 removed)')
plt.xlabel('Projects')
plt.ylabel('Count')
plt.show()

testidx = np.where(train['teacher_number_of_previously_posted_projects'] > 5 )[0]
plt.figure()
plt.hist(train['teacher_number_of_previously_posted_projects'].iloc[testidx], bins=30)
plt.title('Histogram Counting # of Teachers that Previously Posted Projects (teachers with 0 removed)')
plt.xlabel('Projects')
plt.ylabel('Count')
plt.show()


# From the first histogram we see that a huge number of teachers have submitted just a few projects and then it falls off heavily from there. For the second, I filtered out all teachers who have never submitted before (roughly 50,000) and we still see the same behavior. If we were to dig into the numbers more we'd know that ~26,000 have submitted once before, ~17,000 twice, and ~12,000 thrice.

# This gets me curious, we have a number corresponding to each teacher regarding how many projects they've submitted before. Are all of these previous projects in the dataset? Let's figure that out! I wouldn't anticipate it telling us something actionable, but I'm certainly intrigued! 
# 
# From the initial head() command we know that the first teacher in the set has submitted 26 times before as of that submission. How many times does their teacher_id appear in the data set?

# In[23]:


len(np.where(train['teacher_id'] == train['teacher_id'][0])[0])


# It appears only ten times! So there are at least 16 of their applications not in this data set. Let's look to see at the number of previously posted projects they have for each entry.

# In[24]:


train['teacher_number_of_previously_posted_projects'].iloc[np.where(train['teacher_id'] == train['teacher_id'][0])]


# We seem to have a random assortment of their submissions.

# Is there a pattern in acceptance vs number of previous submissions?

# In[26]:


plt.figure()
plt.scatter(train['teacher_number_of_previously_posted_projects'], train['project_is_approved'])
plt.show()


# We see that there are people with past submissions in the hundreds who do sometimes still fail. Experience alone is not a sufficient metric. Let's dig into a single one a little bit more: 

# In[27]:


reject = np.where(train['project_is_approved'] == 0)[0] # Indexes where an application failed.
big_submit = np.where(train['teacher_number_of_previously_posted_projects'] > 375)[0] # Indexes where a teacher has sent more than 350 applications before.
idx = np.intersect1d(reject, big_submit)
id_bigsubmit_but_fail = train['teacher_id'].iloc[idx] # All the ids for teachers that have submitted a bunch but didn't suceed on one.

print(id_bigsubmit_but_fail.head(1)) # Let's look at just a single id.
print(train['teacher_number_of_previously_posted_projects'].iloc[np.where(train['teacher_id'] == id_bigsubmit_but_fail.iloc[0])])
print(train['project_is_approved'].iloc[np.where(train['teacher_id'] == id_bigsubmit_but_fail.iloc[0])])


# From this we can see they failed on their 380th submission (they had 379 before that) but they succeeded on many before and after that submission.

# In[28]:


# Now that I've changed my analysis above, this is redundant. But I'll keep it in for posterity.

rejected_rate = len(reject)/len(train)
print(f'The acceptance rate is {(1 - rejected_rate)*100}%')


# # Summary of Train
# 
# We've learned a few things that could be useful in our model building.
# 
# * There are 4 missing values in the prefix column.
# * Not all submissions from a teacher are in our dataset.
# * We need to appropriately deal with the fact that the number of essays change part way through our data set.
# * We have a ton of subcategories and many of them have very few submissions. There are a few categories that also lack enough submissions to really make any good claims.
# * Simply having a large number of previous submissions does not guarantee future success. Likewise few submissions does not guarantee failure.
# * The acceptrance rate in our dataset is 84.768%.
# 
# ## Some Choices I'll Make After This Analysis
# 
# * I prefer a gender blind algorithm so I'm going to remove the prefix column. This conveniently means we don't need to deal with the NaN.
# * I'm going to try and combine categories and subcategories in a reasonable way and see how many entries of each type that leaves us. If I still have subcategories with only a handful of entries then I may throw that out depending on my future analysis.

# # Quick Exploration of Resource
# Let's do a quick look at resources now.

# In[29]:


for label in list(resource.columns.values):
    print(f'{label} has {sum(resource[label].isna())}')


# We have a decent number of NaN in our description. If we want to use this data then we'll have to deal with that.  What I'm most interested in is plotting the total cost of the resources requested vs. acceptance and plotting total cost vs. how many previous submissions.
# 
# Let's make a new data frame that aggregates all the ids and their associated total costs (quantity * price).

# In[30]:


id_cost = pd.DataFrame({'id': resource['id'], 'total_cost': resource['quantity'] * resource['price']})


# In[31]:


id_cost.head()


# In[33]:


id_total_cost = id_cost.groupby(id_cost['id'], sort=False).sum().reset_index()

# Small note, I originally wrote the above code as a for loop that looped 
# through all unique ids and and summed up every instance.
# However, it ran slooowwww. I projected it'd take aboout 3.5 hours to run. 
# The above code runs in under a second. Use pandas built-in methods!


# In[34]:


id_total_cost.head()


# In[35]:


id_total_cost.describe()


# In[36]:


id_total_cost.sort_values(by=['total_cost'])


# In[37]:


plt.figure()
plt.hist(id_total_cost['total_cost'], bins=50)
plt.show()


# It seems safe to say that the minimum cost that can be requested is $100. Most total costs are clustered around the lower range -- sub 1000 bucks. However there is a long and skinny tail extending all the way to 17,901.94! We'll want to normalize these values if we use them in a machine learning algorithm.

# In[38]:


print(len(id_total_cost['id']))
print(len(train['id']))


# After these manipulations we see that there are still more id's in our collected resources then there are in train. Let's figure out if every id in train has an id in in_total_cost

# In[39]:


train['id'].isin(id_total_cost['id']).head()


# In[40]:


print(sum(train['id'].isin(id_total_cost['id'])))
print(len(train['id'].isin(id_total_cost['id'])))


# It looks like all the ids in train have a corresponding id in id_total_cost! Lets use pandas to merge these appropriately! To reiterate from above, use pandas' built in methods whenever possible. I struggled doing this a cumbersome way and it either wouldn't work or would run forever. However, once again, using pandas resulted in a super fast and clean solution.

# In[41]:


train_aug = pd.merge(train, id_total_cost, on='id', sort=False)


# In[42]:


sum(pd.merge(train, id_total_cost, on='id', sort=False)['total_cost'].isna())


# In[43]:


train_aug.head()


# In[44]:


plt.figure()
plt.scatter(train_aug['teacher_number_of_previously_posted_projects'], train_aug['total_cost'])
plt.ylabel('Total Cost')
plt.xlabel('Number of Previous Submissions')
plt.show()

plt.figure()
plt.scatter(train_aug['project_is_approved'], train_aug['total_cost'])
plt.ylabel('Total Cost')
plt.xlabel('Approved')
plt.show()


# We see that there appears to be a rough correlation with number of submissions and funds requested, specifically that the more submissions you send in the less you're going to ask for. However, in the second plot we see that there doesn't seem to be much impact on approval based off the total cost of resources. Some very expensive proposals get approved and some do not.
# 
# Let's look at these correlations a bit more rigorously.

# In[66]:


ax = sns.heatmap(train_aug.corr(), annot=True, cmap='coolwarm')


# There is a weak positive correlation that if you've submitted more projects than your chances of approval are higher and there's a weak correlation with total cost and the approval of projects AND how many projects have been submitted in the past. This seems reasonable since if you ask for way too much money than you won't get approved and also the more you've won (or failed to win) the less you'll ask for.

# # Wrap Up
# 
# I hope this analysis helped! I definitely think this sort of careful analysis is an important starting point before diving into building a machine learning algorithm. I'm excited to start building now that I have a strong sense of what the underlying data looks like.
# 
# Once again, please comment with any suggestions or questions! 
# 
# Happy building
