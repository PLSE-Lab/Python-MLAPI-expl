#!/usr/bin/env python
# coding: utf-8

# **I.  Getting Started**
# 1. Introduction
# 2. Kaggle vs Colab (Google's Colabratory)
# 3. Setup - Importing Packages and Data Files
# 4. Some Quirks in the Data:  Training Data Split over Multiple Files, Multiple Rows for One Entry, Missing Data
# 5. Pandas Pain
#   - NaNs in the Data
#   - Condensing Multiple Rows into a Single Row
#   - Combining Two DataSets
#   - Changing Only Certain "Cells" (that is, Update Column Values for Only Certain Rows)
#   - Editing Lots of String Data (Removing Punctuation, Setting to Lowercase)
#   - Handling Date Information
# 
# **II.  Data Visualization**
# 1.  Bar Chart
# 2. Scatter Plot
# 3. Line Plot
# 4. Stacked-Bars Barchart
# 5. Pie Chart
# 6. More Line Plots to Look at Dates (Month, Week, Weekday)
# 
# **III.  Applying the Crash Course to this Project to Create a Linear Classifier**
# 1. Randomizing the Data
# 2. Creating Training and Validation Sets
# 3. Choosing Good Features
# 4. Setting Up For Training
# 5. (Regularization)
# 6. Training
# 
# **IV. Applying the Crash Course to this Project to Create a DNN Classifier**
# 
# **V. Dealing with Text-Heavy Features:  Using the Crash Course to Bypass TensorFlow Pain**
# 

# **I. Getting Started**

# **1. Introduction**
# 
# If you're like me, perhaps your only machine learning experience is from Google's [ Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/next-steps).  This will be a brief guide to help you transition from the nice and tidy Programming Exercises of that course to this Kaggle competition that has some issues not encountered in those exercises.  In short, hopefully this will help save you some time and spare you from some hassle.  I'll also be documenting my own attempt to tackle this machine learning problem.
# 
# I originally built a Linear Classifier and DNN Classifier using a lot of the code from the Programming Exercises.  For the features, I only incorporated a few features (basically, any whose data did NOT consist of a bunch of sentences like the essays and the descriptions).  But these models didn't do much better than a random guesser.  So the next step was to incorporate those text-heavy features into the model.
# 
# However, when I attempted to do so, I couldn't get the code to work in the way that I wanted.  I ended up having to completely change how I load in the data (because I couldn't find another solution) so that TensorFlow would look at those strings as individual words to then compare to a vocabulary_list as opposed to it looking at those strings as full sentences (making the comparison to a vocabulary_list made up of individual words useless).  I've kept the old code for the few-feature models if you want to see how to get those models up and running.  But mixed within some of that code may be code that only really applies to the newer model that looks at text data as individual words.

# **2.  Kaggle vs Colab (Google's Colabratory)**
# 
# The Programming Exercises in the Machine Learning Crash Course were done in Google's Colabratory environment.  So my first question was "How do I get the data from this Kaggle competition into Colab?"  While you can likely do that using...
# 
# * [External data: Drive, Sheets, and Cloud Storage](https://colab.research.google.com/notebooks/io.ipynb#scrollTo=c2W5A2px3doP)
# * and/or [Kaggle API](https://github.com/Kaggle/kaggle-api)
# 
# ...it's unnecessary.  The Kaggle environment is effectively the same, making it easier just to get started right here.
# 
# So, to use Kaggle and get your programming environment set up, you'll need:
# 
# i. A [Kaggle account](https://www.kaggle.com/account/login) (which requires an email address and a phone number for verification)
# 
# ii. To start [a new kernel](https://www.kaggle.com/kernels)
# 
# iii. To then choose Notebook (if you want your environment to pretty much have the same feel as Colab)
# 
# iv. To click on the Data tab at the top of your Notebook and then click "Add Data Source" (searching for the DonorsChoose competition and then agreeing to its rules)
# 
# That's pretty much it!  You should now have access to the DonorsChoose files in your Notebook and can begin working with them in your code.

# **3. Setup - Importing Packages and Data Files**
# 
# This should be similar to what you saw in the Programming Exercises.  Your Notebook should load up with a handy note that the data files have this file path:  '../input/filename.csv', which we'll use to load the csv files into a Pandas DataFrame in the code below: 

# In[229]:


# Packages
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset
import sklearn.metrics as metrics
import os # to access data files (found in the "../input/" directory)

# More Packages from https://colab.research.google.com/notebooks/mlcc/sparsity_and_l1_regularization.ipynb?hl=en#scrollTo=pb7rSrLKIjnS
import math
from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt

# Data Files
training_dataset = pd.read_csv('../input/train.csv', sep=',')
resources_dataset = pd.read_csv('../input/resources.csv', sep=',')
test_dataset = pd.read_csv('../input/test.csv', sep=',')


# **4. Some Quirks in the Data**
# 
# i.  In the Programming Exercises, all your feature training data would come from a single csv file and get loaded into a single Pandas DataFrame.  However, in this project, part of the training data is in train.csv and part is in resources.csv, the common link being the same 'id' for a given entry in both files.  Thus, we're going to need some way to combine the two data sets.
# 
# ii.  Additionally, resources.csv has the added twist that *not all of an id's information is in the same row *.  Instead, some project ids are requesting multiple resources, which thus span multiple lines.  For example, submission p069063:

# In[230]:


resources_dataset[0:9]


#   So, you might be thinking about how to group some (or all) of that information together as a single-line entry.
#   
#   iii.  If you play around with resources.csv, you may eventually discover that some of the description data is actually missing.  Some of the cells are filled with NaN, which you may want to handle.  For example, p194324:

# In[231]:


resources_dataset[37602:37612]


# **5. Pandas Pain**
# 
# In the Programming Exercises from the course, there really weren't that many different functions used from Pandas.  Pandas is cool because it can do a million different complex things.  However, Pandas is also frustrating for the same reason, especially when you just want to do something that seems trivial and yet have to comb through documentation or StackOverflow to figure out how to do that.  To spare you from (some of) that searching, let's work through the quirks listed above in reverse order.
# 
# *iii.  NaNs in the data*
# 
# A handy Pandas function to know for a DataFrame is:  **.isnull().any()**, which alerts you to missing values in any of your columns:
# 

# In[232]:


resources_dataset.isnull().any()


# (Note:  If you wanted to locate one of the rows and see the missing data for yourself, you could use the **.loc()** function.)

# In[233]:


resources_dataset.loc[resources_dataset["description"].isnull()]


# Now, let's go ahead and replace those NaNs so that they don't cause any trouble if we were to, say, try to join together all the description strings for a project id into a single string.  To do so, we'll use the function **.fillna({column_name : replacement_value})**:

# In[234]:


resources_dataset = resources_dataset.fillna({'description' : 'no_detail'})
resources_dataset.isnull().any()


# *ii. Condensing multiple rows into one single row*
# 
# In order to do this, we'll need the **.groupby()** function for grouping and the **.agg()** function in order to perform an operation on the cells that are being condensed.  But first, let's create a new feature that we'll likely want to consider:  the cost of a given request.  The resources.csv file contains a request's quantity and its price.  If we multiply those two values together, we'll end up with the cost for that request.  This is something done in the Programming Exercises, so it should likely look familiar:

# In[235]:


# Make a cost column (quantity * price)
resources_dataset['cost'] = resources_dataset['quantity'] * resources_dataset['price']
resources_dataset[0:9]


# Now, let's use the **.groupby()** function and the **.agg()** function to condense, for example, all those p069063 rows into a single line such that we just have one p069063 entry with a total cost that has summed up all the costs of the individual requests.  In this example, I'm only interested in the 'id' and the 'cost' columns, so the final DataFrame will only have those two columns:

# In[236]:


#create a total_cost column

grouped_ids = resources_dataset.groupby(['id'], as_index=False)
resources_condensed = grouped_ids.agg({'cost' : 'sum'}).rename(columns={'cost' : 'total_cost'})
#resources_condensed.loc[resources_condensed['id'] == 'p069063']
resources_condensed[69060:69065]


# Cool!  Let's step through the code.  We pass to **.groupby()** the column that contains the values that repeat in multiple rows.  We also pass in as_index as False so that 'id' (such as p069061, p069062, etc.) doesn't get used as the index.  Instead, each row will have a number index like normal (such as 69060, 69061, etc.)
# 
# The **.agg()** function is then called so that we can perform an operation on the cost values that all belong to the same id.  (In this case, we want to add them all up together.)  We pass in a dictionary of the form *{ column_name : operation_we_want_to_perform }*.  Lastly, I just renamed the column to show that it's the final, total cost for all of the requests made by a given id.

# (Note:  If you wanted to combine the other columns as well, you can pass multiple dictionaries into **.agg()**, as shown in the hidden code snippet.)

# In[237]:


#Join together all of the columns
group_the_ids = resources_dataset.groupby(['id'], as_index=False)
all_condensed = grouped_ids.agg({'description' : lambda x: ' '.join(x), #there's no simple single keyword option like 'sum'
                                 'quantity' : 'sum',
                                 'cost' : 'sum'}).rename(columns={'description' : 'full_description',
                                                                  'quantity' : 'total_quantity',
                                                                  'cost' : 'total_cost'})
all_condensed[69060:69065]


# (Note: Trying to look at the full string of a single cell is a bit of a pain.  See the next code snippet if curious.)

# In[238]:


#Show that p069063's full_description actually contains the joined text:
entry = all_condensed.loc[all_condensed['id'] == 'p069063'].reset_index()
entry.loc[0, 'full_description']


# *i. Merging together two DataFrames*
# 
# Okay, we finally have the desired data from resources.csv organized!  We now want to combine this DataFrame with our other training DataFrame.  To do so, we'll use the **.merge()** function.  This function takes as arguments the DataFrame we want as our "left columns" followed by the DataFrame we want to tack on as the "right columns".  We also specify the common link between the two DataFrames (in this case, 'id') via *on='id'*:

# In[239]:


combined_training_dataset = pd.merge(training_dataset, resources_condensed, on='id')
combined_training_dataset[0:9]


# Woo hoo!  All that's left to do is the same kind of merging but with resources.csv and test.csv:

# In[240]:


combined_test_dataset = pd.merge(test_dataset, resources_condensed, on='id')
combined_test_dataset[0:9]


# I believe the data is now organized in a manner familiar to what we encountered in the Programming Exercises.  Don't forget to also check for and handle other NaNs in the datasets:

# In[241]:


combined_training_dataset['teacher_prefix'] = combined_training_dataset['teacher_prefix'].fillna('none')
combined_test_dataset['teacher_prefix'] = combined_test_dataset['teacher_prefix'].fillna('none')

combined_training_dataset = combined_training_dataset.fillna('')
combined_test_dataset = combined_test_dataset.fillna('')


# *iv. Changing Only Certain "Cells" (that is, Update Column Values for Only Certain Rows)*
# 
# I'm using "cells" because I think it's easier to think in terms of spreadsheets.  In the data for this competition is the note that applications before a certain date required 4 essays.  It turns out that old essays 1 and 2 cover the same topics as current essay 1, and old essays 3 and 4 cover the same topics as current essay 2.  So what I'd like to do is combine essays 1 and 2 together into the project_essay_1 column and combine essays 3 and 4 together into the project_essay_2 column **ONLY for those "old application" rows**.  The "new application" rows should be left alone.
# 
# *(Note:  I don't know if it matters that the essays are treated separately or not.  If not, then you could just smash all the essays together into a single column and not even worry about this step.)*

# The "cell selection" method in Pandas that I'll be using is .loc[row_indexer, col_indexer].  To select the "old application" rows, I'm going to ask Pandas to find all the rows that currently DO have text in project_essay_4.  That will be the row_indexer.  The col_indexer will be the column where I want to leave the two combined essays.

# In[242]:


combined_training_dataset.loc[combined_training_dataset['project_essay_4'] != '', 'project_essay_1'] = combined_training_dataset['project_essay_1'] + ' ' + combined_training_dataset['project_essay_2']
combined_training_dataset.loc[combined_training_dataset['project_essay_4'] != '', 'project_essay_2'] = combined_training_dataset['project_essay_3'] + ' ' + combined_training_dataset['project_essay_4']

#and do the same for the test data:
combined_test_dataset.loc[combined_test_dataset['project_essay_4'] != '', 'project_essay_1'] = combined_test_dataset['project_essay_1'] + ' ' + combined_test_dataset['project_essay_2']
combined_test_dataset.loc[combined_test_dataset['project_essay_4'] != '', 'project_essay_2'] = combined_test_dataset['project_essay_3'] + ' ' + combined_test_dataset['project_essay_4']

combined_training_dataset[16:20]


# In[243]:


#To see if the above worked, look at the whole string in an "old application"'s cell:

combined_training_dataset.loc[18, 'project_essay_2']


# *v. Editing Lots of String Data (Removing Punctuation, Setting to Lowercase)*
# 
# This pertains only to the newer models where I'm incorporating the text-heavy features.  To make it so that the vocabulary_lists can match the words in the text sentences, it's best to strip out punctuation marks and lowercase all the letters:

# In[244]:


#Processing the text in the heavy text columns:
heavy_text_columns = [
    'project_title',
    'project_essay_1',
    'project_essay_2',
    'project_resource_summary',
    'project_subject_categories',  # probably better two keep these two as consisting of separate phrases
    'project_subject_subcategories'  # but since some entries belong to multiple categories, it might just be easier to treat them the same way (at least for now)
]

for col_name in heavy_text_columns:
    combined_training_dataset[col_name] = combined_training_dataset[col_name].str.lower()
    combined_test_dataset[col_name] = combined_test_dataset[col_name].str.lower()

#remove punctuation:
#surely this is a common problem???  Shouldn't there be one simple solution that actually fully works?  whatever... here's my hacky way
pattern_n = r'\\n'
pattern_r = r'\\r'
punc_pattern = '[!"#$%&\'()*+,-./:;<=>?@\\\\^_`{|}~]'
for col_name in heavy_text_columns:
    #forgot to replace \n and \r with spaces the first time.  Wonder how much of an effect that has? (bc some words get smashed together)
    combined_training_dataset[col_name] = combined_training_dataset[col_name].replace(pattern_n, ' ', regex=True)
    combined_training_dataset[col_name] = combined_training_dataset[col_name].replace(pattern_r, ' ', regex=True)
    combined_training_dataset[col_name] = combined_training_dataset[col_name].replace(punc_pattern, '', regex=True)
    #don't know if having spaces exist in the vocabulary_list matters or not.  Just to be safe,
    #replace the instances of \r\n and individual \r or \n's that were replaced with spaces
    #[which was done to prevent last of one sentence being smashed with start of next sentence] :
    combined_training_dataset[col_name] = combined_training_dataset[col_name].str.replace('   ', ' ')
    combined_training_dataset[col_name] = combined_training_dataset[col_name].str.replace('  ', ' ')
    
    combined_test_dataset[col_name] = combined_test_dataset[col_name].replace(pattern_n, ' ', regex=True)
    combined_test_dataset[col_name] = combined_test_dataset[col_name].replace(pattern_r, ' ', regex=True)
    combined_test_dataset[col_name] = combined_test_dataset[col_name].replace(punc_pattern, '', regex=True)
    combined_test_dataset[col_name] = combined_test_dataset[col_name].str.replace('   ', ' ')
    combined_test_dataset[col_name] = combined_test_dataset[col_name].str.replace('  ', ' ')

combined_training_dataset[16:22]    


# *vi. Handling Date Information*
# 
# After working more on this project, I've decided to incorporate the project_submitted_datetime as a feature.  Based on some data visualization later on, I've decided to bucketize this information by week of the years this dataset spans.  I don't know if there's some preferred method for doing this, but I like the idea of converting these values to an integer like 'last two digits of year' * 1000 + week (for example, 16 * 1000 + 25 = 1625). 

# In[245]:


#https://stackoverflow.com/questions/25146121/extracting-just-month-and-year-from-pandas-datetime-column-python
#First, get the project_submitted_datetime column into a format that Pandas can work with
combined_training_dataset['project_submitted_datetime'] = pd.to_datetime(combined_training_dataset['project_submitted_datetime'])
combined_training_dataset['year_week'] = combined_training_dataset['project_submitted_datetime'].map(lambda x: 100*(x.year - 2000) + x.week)

combined_test_dataset['project_submitted_datetime'] = pd.to_datetime(combined_test_dataset['project_submitted_datetime'])
combined_test_dataset['year_week'] = combined_test_dataset['project_submitted_datetime'].map(lambda x: 100*(x.year - 2000) + x.week)
combined_training_dataset[0:10]


# I'm wondering if the two datasets span the same amount of time and/or if the test dataset has any unique time entries:

# In[246]:


training_year_weeks = combined_training_dataset['year_week'].unique().tolist()
test_year_weeks = combined_test_dataset['year_week'].unique().tolist()
print('training_year_weeks min/max:')
print('%d , %d' % (min(training_year_weeks), max(training_year_weeks)))
print('test_year_weeks min/max:')
print('%d , %d' % (min(test_year_weeks), max(test_year_weeks)))

#https://stackoverflow.com/questions/45098206/unique-values-between-2-lists
set(test_year_weeks) - set(training_year_weeks)


# Doesn't appear to be the case, so hopefully all the buckets for the training data suffice.  (Not actually sure what happens when the test data has data outside of the given buckets...)

# **II.   Data Visualization**

# Colaboratory has a [Welcome notebook](https://colab.research.google.com/notebooks/welcome.ipynb#scrollTo=yv2XIwi5hQ_g) with a small section on Visualization, but a more detailed [Charts notebook](https://colab.research.google.com/notebooks/charts.ipynb) with several Matplotlib examples can be found in the "For more information" section of that Welcome site.
# 
# Much of the visualization in the Programming Exercises was already coded in, so.... I want to try making some of these charts for myself!
# 
# You can find much nicer and more interesting charts in other kernels, so definitely check those out.  This is now just me messing around with code.

# **1.  Bar Chart** - *Number of Approved/Not-Approved Applications Based on teacher_prefix*

# In[247]:


import matplotlib.pyplot as plt


# In[248]:


#Prefixes and Approved vs Unapproved Application
grouped_prefixes = combined_training_dataset.groupby(["teacher_prefix", "project_is_approved"])
grouped_prefixes = grouped_prefixes.agg({'teacher_prefix' : 'count'}).rename(columns={'teacher_prefix' : 'count'})
grouped_prefixes


# In[249]:


arr_counts = grouped_prefixes["count"].tolist()
#split arr_counts into two separate lists:
y_no = []
y_yes = []
for i in range(0, len(arr_counts) - 1):
    if i % 2 == 0:
        y_no.append(arr_counts[i])
    else:
        y_yes.append(arr_counts[i])
y_yes.append(arr_counts[i+1])
y_no.append(0)
x_labels = ['Dr', 'Mr.', 'Mrs', 'Ms', 'Teacher', 'None']

#Make a multiple bar graph
#via https://stackoverflow.com/questions/14270391/python-matplotlib-multiple-bars
N = len(x_labels)
ind = np.arange(N)  # the x locations for the groups
width = 0.4      # the width of the bars
fig = plt.figure()
ax = fig.add_subplot(111)

rects1 = ax.bar(ind, y_yes, width, color='#7CFC00')
rects2 = ax.bar(ind+width, y_no, width, color='#DC143C')

ax.set_ylabel('Submissions Count')
ax.set_xticks(ind+width)
ax.set_xticklabels(x_labels)
ax.legend( (rects1[0], rects2[0]), ('Approved', 'NOT Approved') )

def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.title('Submissions Approved or NOT Approved Based on Name Prefix')
plt.show()


# And what are those percentages?

# In[250]:


#Chance to get approved by prefix
print('Chance to get approved by prefix')
arr_total_applications = []
for i in range(0, len(y_yes)):
    arr_total_applications.append(y_yes[i] + y_no[i])
    print(x_labels[i] + ': ' + "%f" % (y_yes[i] / arr_total_applications[i]))


# So, for immediate acceptance, just don't submit a prefix on your application, obviously!  #percentagesneverlie

# **2. Scatter Plot** - *Acceptance Rate vs. Request Cost within a Binned Range*
# 
# Curious to see if the acceptance rate differs as the cost of the request gets higher.  There are too many different values for the cost (100, 100.01, 100.02, etc.), so I'm also going to try binning with Pandas to group those costs into ranges.

# In[251]:


bins = [0, 150, 200, 250, 300, 350, 400, 500, 600, 750, 1000, 1000000]
combined_training_dataset['binned_costs'] = pd.cut(combined_training_dataset['total_cost'], bins)

grouped_costs = combined_training_dataset.groupby(["binned_costs", "project_is_approved"])
grouped_costs = grouped_costs.agg({'binned_costs' : 'count'}).rename(columns={'binned_costs' : 'count'})

arr_percent_yes = []
for i in range(0, (len(bins) - 1) * 2):
    if i % 2 == 0:
        k = i // 2
        label = 'no'
        no_count = grouped_costs["count"].values[i]
        print("%d+ - %s: %d" % (bins[k], label, no_count))
    else:
        label = 'yes'
        yes_count = grouped_costs["count"].values[i]
        percent_yes = yes_count / (no_count + yes_count)
        arr_percent_yes.append(percent_yes)
        print("%d+ - %s: %d, percent: %f" % (bins[k], label, yes_count, percent_yes))


# In[252]:


arr_x_labels = []
for i in range(0, len(bins)-2):
    arr_x_labels.append("%d+" % (bins[i]))
arr_x_labels.append("_1000+") #add the _ so that the plot doesn't alphabetize the numbers
 
plt.scatter(arr_x_labels, arr_percent_yes)
plt.xlabel("Cost of Request")
plt.ylabel("Yes Acceptance Rate")
plt.show()


# I just sorta eyeballed the bin ranges.  Still, that downward trend seems sensible as I would expect a higher percentage of cheaper requests to get approved over much pricier ones.
# 
# But I'm curious to try out a more official way (using quantiles to split the data into equal group sizes):

# In[253]:


#function code via https://colab.research.google.com/notebooks/mlcc/sparsity_and_l1_regularization.ipynb?hl=en#scrollTo=bLzK72jkNJPf
def quantile_based_buckets(feature_values, num_buckets):
    quantiles = feature_values.quantile(
        [(i+1.)/(num_buckets + 1.) for i in range(num_buckets)])
    return [quantiles[q] for q in quantiles.keys()]

quantile_bins = quantile_based_buckets(combined_training_dataset["total_cost"], 12)
#^But why does the DataFrame later show NaN for the lowest range (from 0 to 133)?
#...Looks like Pandas requires you to specify the lower/upper limits if you pass in a list to pd.cut?
quantile_bins.insert(0, 0)
quantile_bins.append(1000000)

combined_training_dataset['quantile_costs'] = pd.cut(combined_training_dataset['total_cost'], quantile_bins)

grouped_costs = combined_training_dataset.groupby(["quantile_costs", "project_is_approved"])
grouped_costs = grouped_costs.agg({'quantile_costs' : 'count'}).rename(columns={'quantile_costs' : 'qcount'})

arr_percent_yes = []
for i in range(0, (len(quantile_bins) - 1) * 2):
    if i % 2 == 0:
        k = i // 2
        label = 'no'
        no_count = grouped_costs["qcount"].values[i]
        print("%d+ - %s: %d" % (quantile_bins[k], label, no_count))
    else:
        label = 'yes'
        yes_count = grouped_costs["qcount"].values[i]
        percent_yes = yes_count / (no_count + yes_count)
        arr_percent_yes.append(percent_yes)
        print("%d+ - %s: %d, percent: %f" % (quantile_bins[k], label, yes_count, percent_yes))


# In[254]:


arr_x_labels = []
for i in range(0, len(quantile_bins)-2):
    arr_x_labels.append("%d+" % (quantile_bins[i]))
arr_x_labels.append("_1000+") #add the _ so that the plot doesn't alphabetize the numbers
 
plt.scatter(arr_x_labels, arr_percent_yes)
plt.xlabel("Cost of Request")
plt.ylabel("Yes Acceptance Rate")
plt.show()


# **3. Line Plot** - *Chance of Acceptance Based on Number of Previous Submissions from 0 to 100*
# 
# Out of curiosity, I'm also interested in odds based on previous submissions.

# In[255]:


grouped_previous_submissions = combined_training_dataset.groupby(["teacher_number_of_previously_posted_projects", "project_is_approved"])
grouped_previous_submissions = grouped_previous_submissions.agg({'teacher_number_of_previously_posted_projects' : 'count'}).rename(columns={'teacher_number_of_previously_posted_projects' : 'count'})
grouped_previous_submissions[0:202]


# In[256]:


arr_percents_yes = []
for i in range(0,202):
    if i % 2 == 0:
        k = i / 2
        label = 'no'
        no_count = grouped_previous_submissions["count"].values[i]
        print("%d - %s: %d" % (k, label, no_count))
    else:
        label = 'yes'
        yes_count = grouped_previous_submissions["count"].values[i]
        percent_yes = yes_count / (no_count + yes_count)
        arr_percents_yes.append(percent_yes)
        print("%d - %s: %d, percent: %f" % (k, label, yes_count, percent_yes))
    


# In[257]:


x_num_previous = np.arange(0, 101)
#y is arr_percents_yes

plt.plot(x_num_previous, arr_percents_yes)

plt.xlabel("Number of Previous Submissions")
plt.ylabel("Chance of Accepted Submission")
plt.title("Chance of Accepted Submission Based on Previous Number of Submissions")
plt.show()


# An expected general upward trend that tapers off (Note: higher x values [say, 30+] have far less entries, especially the much higher x values).  Also interesting that the chance of acceptance for a new applicant is quite high (~82%).

# **4. Stacked-Bars Bar Chart** - *Comparing the Top 5 vs Bottom 5 States in Terms of Acceptance Rate*
# 
# And why not look at the states, too?  (Also going to try getting the *groupby* to act like a normal DataFrame this time.)

# In[258]:


grouped_states = combined_training_dataset.groupby(["school_state", "project_is_approved"])
#a way to eliminate the multi-index?:
#https://stackoverflow.com/questions/39778686/pandas-reset-index-after-groupby-value-counts
grouped_states = grouped_states.size().rename('count').reset_index()

arr_percents = []
for i in range(0,102):
    if i % 2 == 0:
        no_count = grouped_states["count"].values[i]
    else:
        yes_count = grouped_states["count"].values[i]
        total_count = no_count + yes_count
        percent_no = no_count / total_count
        percent_yes = yes_count / total_count
        arr_percents.append(percent_no)
        arr_percents.append(percent_yes)
        
grouped_states['chances'] = pd.Series(arr_percents, index=grouped_states.index)
grouped_states


# In[259]:


#Find the lowest approval rates:
grouped_states.loc[(grouped_states['project_is_approved'] == 1) & (grouped_states['chances'] < .83)]


# In[260]:


#And the highest ones:
grouped_states.loc[(grouped_states['project_is_approved'] == 1) & (grouped_states['chances'] > .868)]


# In[261]:


#Make a stacked bar chart of Top 5 vs Bottom 5 acceptance rates because...  I just wanna see what it looks like...

idxes = ['1 DE/DC', '2 WY/TX', '3 OH/NM', '4 CT/FL', '5 WA/MT']
lowest = [.812639, .815670, .822052, .824500, .828125]
highest = [.891341, .875706, .871467, .871294, .868050]

#apparently the 2nd bar just paints over the first, so the 2nd must be smaller or it gets hidden
plt.bar(idxes, highest, label="DE, WY, OH, CT, WA", color='#87CEFA')
plt.bar(idxes, lowest, label="DC, TX, NM, FL, MT", color='#B22222')

plt.plot()

#make the scale more useful
#https://stackoverflow.com/questions/3777861/setting-y-axis-limit-in-matplotlib
axes = plt.gca()
axes.set_ylim([.75, .92])

plt.title('Submission Acceptance Rates for Top 5 States vs Bottom 5 States')
plt.legend()
plt.xlabel('State')
plt.ylabel("Chance of Accepted Submission")
plt.show()


# **5. Pie Chart** - *Percentage of Applications Accepted and NOT Accepted According to Grade*

# In[262]:


grouped_grades = combined_training_dataset.groupby(["project_grade_category", "project_is_approved"])
#a way to eliminate the multi-index?:
#https://stackoverflow.com/questions/39778686/pandas-reset-index-after-groupby-value-counts
grouped_grades = grouped_grades.size().rename('count').reset_index()
grouped_grades


# In[263]:


arr_no = []
arr_yes = []
arr_labels = []
arr_all = []
for i in range(0, len(grouped_grades["project_grade_category"].values)):
    if i % 2 == 0:
        arr_labels.append("%s - No" % (grouped_grades["project_grade_category"].values[i]))
        arr_no.append(grouped_grades["count"][i])
    else:
        arr_labels.append("%s - Yes" % (grouped_grades["project_grade_category"].values[i]))
        arr_yes.append(grouped_grades["count"][i])
    arr_all.append(grouped_grades["count"][i])

colors = ['#DC143C', '#7CFC00']
plt.pie(arr_all, labels=arr_labels, colors=colors,
        startangle=50,
        explode = (.3, .3, .3, .3, .3, .3, .3, .3),
        autopct = '%1.2f%%',
        shadow=True)

plt.axis('equal')
plt.title('Pie Chart Example')
plt.show()


# Would be cool to split the pie into grouped pieces (for example, Grades 3-5 Yes and No exploding off as a combined piece), but [that looks nontrivial](http://https://stackoverflow.com/questions/20549016/explode-multiple-slices-of-pie-together-in-matplotlib/20556088).

# **6. More Line Plots to Look at Dates (Month, Week, Weekday)**
# 
# I'm curious if it looks like the date the application was submitted may have any impact on its approval.  If so, this could be a good feature to incorporate.
# 
# This requires a couple new Pandas functions.  The first is pd.to_datetime to get the "project_submitted-datetime" column into a format that Pandas can work with.  To group by month, the groupby function is going to take a pd.Grouper() in place of a typical column name's string.

# In[264]:


#combined_training_dataset['project_submitted_datetime'] = pd.to_datetime(combined_training_dataset['project_submitted_datetime'])
  #^done in an earlier step when preprocessing the data
#try grouping by month:
#https://stackoverflow.com/questions/44908383/how-can-i-group-by-month-from-a-date-field-using-python-pandas

grouped_months = combined_training_dataset.groupby([pd.Grouper(key='project_submitted_datetime', freq='1M'), "project_is_approved"])
grouped_months = grouped_months.size().rename('count').reset_index()

arr_percents_month = []
for i in range(0,26):
    if i % 2 == 0:
        no_count = grouped_months["count"].values[i]
    else:
        yes_count = grouped_months["count"].values[i]
        total_count = no_count + yes_count
        percent_no = no_count / total_count
        percent_yes = yes_count / total_count
        arr_percents_month.append(percent_no)
        arr_percents_month.append(percent_yes)
        
grouped_months['chances'] = pd.Series(arr_percents_month, index=grouped_months.index)
grouped_months


# And there's definitely some variation, so let's graph that:

# In[265]:


x_num_previous_m = np.arange(0, 13)

arr_percents_m_yes = []
for i in range(0,26):
    if not (i % 2 == 0):
        arr_percents_m_yes.append(arr_percents_month[i])


plt.plot(x_num_previous_m, arr_percents_m_yes)

plt.xlabel("Month")
plt.ylabel("Chance of Accepted Submission")
plt.title("Chance of Accepted Submission Based on Month")
plt.show()


# So it might be useful to bucketize the data by month.  But would weekly look any different?

# In[266]:


#only difference here is freq='1W' instead of '1M'
grouped_weeks = combined_training_dataset.groupby([pd.Grouper(key='project_submitted_datetime', freq='1W'), "project_is_approved"])
grouped_weeks = grouped_weeks.size().rename('count').reset_index()

arr_percents_w = []
for i in range(0,106):
    if i % 2 == 0:
        no_count = grouped_weeks["count"].values[i]
    else:
        yes_count = grouped_weeks["count"].values[i]
        total_count = no_count + yes_count
        percent_no = no_count / total_count
        percent_yes = yes_count / total_count
        arr_percents_w.append(percent_no)
        arr_percents_w.append(percent_yes)
        
grouped_weeks['chances'] = pd.Series(arr_percents_w, index=grouped_weeks.index)
grouped_weeks


# And graphing that:

# In[267]:


x_num_previous_w = np.arange(0, 53)

arr_percents_yes_w = []
for i in range(0,106):
    if not (i % 2 == 0):
        arr_percents_yes_w.append(arr_percents_w[i])


plt.plot(x_num_previous_w, arr_percents_yes_w)

plt.xlabel("Week")
plt.ylabel("Chance of Accepted Submission")
plt.title("Chance of Accepted Submission Based on Week")
plt.show()


# Perhaps monthly would do, but I'm going to go with bucketizing weekly since that one stretch has some pretty wild swings.
# 
# I also checked to see if the day of the week an application was submitted mattered, but the odds seemed pretty much the same for each weekday.  The code for that is in the hidden snippet if you're curious to see how it's implemented.

# In[268]:


#going to also try by day of the week:
#https://stackoverflow.com/questions/13740672/in-pandas-how-can-i-groupby-weekday-for-a-datetime-column
combined_training_dataset['weekday'] = combined_training_dataset['project_submitted_datetime'].dt.weekday

grouped_weekday = testing.groupby(["weekday", "project_is_approved"])
#grouped_dates = grouped_dates.agg({'project_submitted_datetime' : 'count'})
grouped_weekday = grouped_weekday.size().rename('count').reset_index()

arr_percents_weekday = []
for i in range(0,14):
    if i % 2 == 0:
        no_count = grouped_weekday["count"].values[i]
    else:
        yes_count = grouped_weekday["count"].values[i]
        total_count = no_count + yes_count
        percent_no = no_count / total_count
        percent_yes = yes_count / total_count
        arr_percents_weekday.append(percent_no)
        arr_percents_weekday.append(percent_yes)
        
grouped_weekday['chances'] = pd.Series(arr_percents_weekday, index=grouped_weekday.index)
grouped_weekday


# **III.  Applying the Crash Course to this Project to Create a Linear Classifier**
# 
# I was looking over my course notes, and the code for just about everything I want to incorporate into a model that uses a linear classifier can be found in the [Programming Exercises from Chapter 13 - Regularization: Sparsity](https://colab.research.google.com/notebooks/mlcc/sparsity_and_l1_regularization.ipynb?hl=en):
# 
# - Randomizing the data; stressed in Chapter 6
# - Splitting the data into training and validation sets; stressed in Chapter 5
# - Choosing good features (binning/bucketizing, scaling, handling outliers); stressed in Chapter 8
# - Setting up everything related to training (feature columns, input functions, training, predictions); from throughout the course
# - Regularization; from Chapters 10, 11, and 13 (L1 Regularization)
# 
# The only thing missing is the code for determining the ROC and AUC, which can be found in the [Programming Exercises from Chapter 12 - Classification](https://colab.research.google.com/notebooks/mlcc/logistic_regression.ipynb?hl=en).
# 
# Thus, I'll be working off the code from Chapter 13, and I'll try to keep it organized in a similar flow so that it looks familiar.

# **1. Randomizing the Data**
# 
# The data here looks like it's already been randomized, but just to be safe, I'll go ahead and make sure:

# In[269]:


# Data sets are:
# combined_training_dataset
# combined_test_dataset

combined_training_dataset = combined_training_dataset.reindex(
    np.random.permutation(combined_training_dataset.index))


# **2. Creating Training and Validation Sets**
# 
# (Note:  I'll be skipping the "preprocessing features functions" from those exercises since that stuff was basically handled in the Getting Started section.)

# In[270]:


USING_OLD_MODELS = False

#combined_training_dataset.shape  -->  182,080 examples
#split 80% training, 20% validation --> 145,664 training, 36,416 validation
HEAD = 145664
TAIL = 36416
TARGET_STR = "project_is_approved"

training_examples = combined_training_dataset.head(HEAD)
validation_examples = combined_training_dataset.tail(TAIL)

if USING_OLD_MODELS:  
    #new models = don't separate examples from targets here
    #old models require targets to be separate:
    training_targets = combined_training_dataset[[TARGET_STR]].head(HEAD)
    validation_targets = combined_training_dataset[[TARGET_STR]].tail(TAIL)
    


# **3. Choosing Good Features**
# 
# This is where my code will differ some from the Exercises.  I plan to experiment with the different features to see which I really want to include.  However, doing that in the Exercises can involve editing or commenting out a lot of code.  So instead I just want to make a list of "desired features"  and then make that the only thing I'll have to update.

# In[271]:


# GLOBAL CONSTANTS
#USING_OLD_MODELS = False  #(located in cell above)

if USING_OLD_MODELS:
    DESIRED_FEATURES = [
    #'id',
    #'teacher_id',
    'teacher_prefix',   # =  odds here are mostly the same for everyone
    'school_state',
    #'project_submitted_datetime',
    #'project_grade_category',  # = odds also seem minimally different
    #'project_subject_categories',
    #'project_subject_subcategories',
    #'project_title',
    #'project_essay_1',
    #'project_essay_2',
    #'project_essay_3',
    #'project_essay_4',
    #'project_resource_summary',
    'teacher_number_of_previously_posted_projects',
    #'total_description',
    #'total_quantity',
    'total_cost'
    ]
else:
    DESIRED_FEATURES = [
        #'id',
        #'teacher_id',
        #'teacher_prefix',   #=  odds here are mostly the same for everyone. Experiments = no positive effect
        'school_state',
        #'project_submitted_datetime',
        #'project_grade_category',  #= odds also seem minimally different.  Experiments = no positive effect
        'project_subject_categories',
        'project_subject_subcategories',
        'project_title',
        'project_essay_1',
        'project_essay_2',
        #'project_essay_3',
        #'project_essay_4',
        'project_resource_summary',
        'teacher_number_of_previously_posted_projects',
        #'total_description',
        #'total_quantity',
        'total_cost',
        'year_week',  #effectively takes the place of project_submitted_datetime
        'project_is_approved'  #new model includes the target and handles it separately later.
    ]

# Buckets to determine quantiles
NUM_COST_BUCKETS = 12
NUM_PREVIOUS_SUBMISSIONS_BUCKETS = 6

# Alternatively, I may also want to try setting my own buckets for previous_submissions:
PREVIOUS_SUBMISSIONS_BUCKETS = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0]
YEAR_WEEK_BUCKETS = sorted(training_examples['year_week'].unique().tolist())
#I feel like you had to manually set lower/upperbounds for TensorFlow...
YEAR_WEEK_BUCKETS.insert(0, 0)
YEAR_WEEK_BUCKETS.append(2000)

# construct_feature_columns (shown later) is designed to work with either the Linear Classifer
# or the DNN Classifier dependening on the following constant
# That is, if training the DNN Classifier, turn categorical_column_with_vocabulary_list into embeddings:
IS_DNN_CLASSIFIER = False
EMBEDDING_DIMENSIONS = 2

print('This code ran.')


# *Bucketizing*
# 
# I'm not currently planning on using any continuous numeric data.  Instead, I'll be bucketizing them, and here's the function to create quantile-based buckets:

# In[272]:


def get_quantile_based_buckets(feature_values, num_buckets):
    """
    Args:
        feature_values:  Pandas DataFrame (one column)
        num_buckets:  how many buckets
        
    Returns:
        An array of the quantiles
    """
    quantiles = feature_values.quantile(
        [(i+1.)/(num_buckets + 1.) for i in range(num_buckets)])
    return [quantiles[q] for q in quantiles.keys()]


# **4. Setting Up for Training**
# 
# First up is the general input function to create more specific input functions for training vs validation vs prediction:

# In[273]:


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """
    Args:
        features: pandas DataFrame of features
        targets: pandas DataFrame of targets
        batch_size: Size of batches to be passed to the model
        shuffle: True or False. Whether to shuffle the data.
        num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
        Tuple of (features, labels) for next data batch
    """
    
    # Grab only the features specified in DESIRED_FEATURES:
    selected_features_data = features[DESIRED_FEATURES]
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(selected_features_data).items()}                                            

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# Next up is the function to make Feature Columns.  This contains code for making bucketized numeric columns as well as categorical columns with vocabulary list.  Depending on the value of global IS_DNN_CLASSIFIER, if that value is true, it then converts the categorical columns into embedding columns.

# In[274]:


#vocabulary lists 
if USING_OLD_MODELS:
    prefix_list = training_examples['teacher_prefix'].unique().tolist()
    state_list = training_examples['school_state'].unique().tolist()
    grade_list = training_examples['project_grade_category'].unique().tolist()
else:
    #list of unique single string identifies in full string features:
    if 'teacher_prefix' in DESIRED_FEATURES:
        prefix_list = combined_training_dataset['teacher_prefix'].unique().tolist()
    if 'school_state' in DESIRED_FEATURES:
        state_list = combined_training_dataset['school_state'].unique().tolist()
    if 'project_grade_category' in DESIRED_FEATURES:
        grade_list = combined_training_dataset['project_grade_category'].unique().tolist()
    #vocabulary_lists of individual unique words within heavy-text columns:
    if 'project_resource_summary' in DESIRED_FEATURES:
        project_summary_vocab_list = set()
        combined_training_dataset["project_resource_summary"].str.split().apply(project_summary_vocab_list.update)
    if 'project_title' in DESIRED_FEATURES:
        project_title_vocab_list = set()
        combined_training_dataset["project_title"].str.split().apply(project_title_vocab_list.update)
    #I wonder if the 2 essays really need to be treated separately?  If not, this code would change:
    if 'project_essay_1' in DESIRED_FEATURES:
        essay_1_vocab_list = set()
        combined_training_dataset["project_essay_1"].str.split().apply(essay_1_vocab_list.update)
    if 'project_essay_2' in DESIRED_FEATURES:
        essay_2_vocab_list = set()
        combined_training_dataset["project_essay_2"].str.split().apply(essay_2_vocab_list.update)

        #Currently treating 'project_subject_categories' and 'project_subject_subcategories' as separate features
        #AND, since I assume it's just easier, comparing the possible multiple categories it belongs to as individual words
    if 'project_subject_categories' in DESIRED_FEATURES:
        categories_list = set()
        combined_training_dataset["project_subject_categories"].str.split().apply(categories_list.update)
    if 'project_subject_subcategories' in DESIRED_FEATURES:
        subcategories_list = set()
        combined_training_dataset["project_subject_subcategories"].str.split().apply(subcategories_list.update)

def construct_feature_columns():
    """Construct the TensorFlow Feature Columns.

    Returns:
        A set of feature columns
    """
    
    arr_bucket_columns = []
    arr_vocabulary_columns = []
    
    if 'total_cost' in DESIRED_FEATURES:
        bucketized_total_cost = tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column("total_cost"),
            boundaries=get_quantile_based_buckets(training_examples["total_cost"], NUM_COST_BUCKETS))
        arr_bucket_columns.append(bucketized_total_cost)
    
    if 'teacher_number_of_previously_posted_projects' in DESIRED_FEATURES:
        bucketized_previous_submissions = tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column("teacher_number_of_previously_posted_projects"),
            #boundaries=get_quantile_based_buckets(training_examples["teacher_number_of_previously_posted_projects"], NUM_PREVIOUS_SUBMISSIONS_BUCKETS))
            boundaries=PREVIOUS_SUBMISSIONS_BUCKETS) #buckets of my own choosing for now
        arr_bucket_columns.append(bucketized_previous_submissions)
        
    if 'year_week' in DESIRED_FEATURES:
        bucketized_year_week = tf.feature_column.bucketized_column(
            tf.feature_column.numeric_column("year_week"),
            boundaries=YEAR_WEEK_BUCKETS) #each unique year_week is a bucket
        arr_bucket_columns.append(bucketized_year_week)
        
    if 'teacher_prefix' in DESIRED_FEATURES:
        prefix_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key="teacher_prefix", vocabulary_list=prefix_list)
        arr_vocabulary_columns.append(prefix_column)
        
    if 'school_state' in DESIRED_FEATURES:
        state_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key='school_state', vocabulary_list=state_list)
        arr_vocabulary_columns.append(state_column)
        
    if 'project_grade_category' in DESIRED_FEATURES:
        grade_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key='project_grade_category', vocabulary_list=grade_list)
        arr_vocabulary_columns.append(grade_column)
        
    if 'project_resource_summary' in DESIRED_FEATURES:
        summary_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key='project_resource_summary', vocabulary_list=project_summary_vocab_list)
        arr_vocabulary_columns.append(summary_column)
        
    if 'project_title' in DESIRED_FEATURES:
        title_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key='project_title', vocabulary_list=project_title_vocab_list)
        arr_vocabulary_columns.append(title_column)
        
    if 'project_essay_1' in DESIRED_FEATURES:
        essay_1_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key='project_essay_1', vocabulary_list=essay_1_vocab_list)
        arr_vocabulary_columns.append(essay_1_column)
        
    if 'project_essay_2' in DESIRED_FEATURES:
        essay_2_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key='project_essay_2', vocabulary_list=essay_2_vocab_list)
        arr_vocabulary_columns.append(essay_2_column)
        
    #again, currently treating categories and subcategories as separate and as consisting of individual words
    if 'project_subject_categories' in DESIRED_FEATURES:
        categories_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key='project_subject_categories', vocabulary_list=categories_list)
        arr_vocabulary_columns.append(categories_column)
        
    if 'project_subject_subcategories' in DESIRED_FEATURES:
        subcategories_column = tf.feature_column.categorical_column_with_vocabulary_list(
            key='project_subject_subcategories', vocabulary_list=subcategories_list)
        arr_vocabulary_columns.append(subcategories_column)
        
    if IS_DNN_CLASSIFIER:
        arr_vocabulary_columns = list(map(lambda x : tf.feature_column.embedding_column(x, dimension=EMBEDDING_DIMENSIONS),
                                    arr_vocabulary_columns))

    feature_columns = set(arr_bucket_columns + arr_vocabulary_columns)

    return feature_columns


# Then we have the function to train the model.  (Note:  This is where **5. Regularization** gets implemented.)
# 
# **An important note about the train_model function:**
# In the Colab Programming Exercises, they split the training into "periods" so that you can observe changes in loss as the training progresses.  However, as far as I can tell, that makes it *really* slow here in Kaggle.  So I've skipped applying periods in this function.  (All the periods do is split up the number of steps.  For example, if you have 10 periods and 1000 steps, then using periods will train the model for 100 steps, stop, start up again, train for 100 more, stop, start up again, etc.  Using no periods means that all 1000 steps happen after a single start, which appears to be much, much faster with the same end result.  But since there are no periods, you won't get a visual graph of the training/validation loss changes over time.)

# In[275]:


def train_linear_classifier_model(
    learning_rate,
    regularization_strength,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets,
    periods=1):
    """Trains a linear classifier model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
        learning_rate: A `float`, the learning rate.
        regularization_strength: A `float` that indicates the strength of the L1
            regularization. A value of `0.0` means no regularization.
        steps: A non-zero `int`, the total number of training steps. A training step
            consists of a forward and backward pass using a single batch.
        feature_columns: A `set` specifying the input feature columns to use.
        training_examples: A `DataFrame` containing one or more columns to use as input features for training.
        training_targets: A `DataFrame` containing exactly one column from to use as target for training.
        validation_examples: A `DataFrame` containing one or more columns to use as input features for validation.
        validation_targets: A `DataFrame` containing exactly one column to use as target for validation.
        periods: A integer, the number of times to train the model.  #Programming Exercises had periods = 7

    Returns:
    A `LinearClassifier` object trained on the training data.
    """

    #steps_per_period = steps / periods  SKIPPING PERIODS
    
    # Create a linear classifier object.
    my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate, l1_regularization_strength=regularization_strength)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(
        feature_columns=feature_columns,
        optimizer=my_optimizer
    )
    
    # Create input function for training (dependent on batch_size passed in)
    training_input_fn = lambda: my_input_fn(training_examples, 
                      training_targets[TARGET_STR], 
                      batch_size=batch_size)
    
    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss (on validation data):")
    #training_log_losses = []  SKIPPING PERIODS
    #validation_log_losses = []
    for period in range (0, periods):
        # Train the model, starting from the prior state.
        linear_classifier.train(
        input_fn=training_input_fn,
        steps=steps
        )
        # Take a break and compute predictions.
        training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
        training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

        validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
        validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

        # Compute training and validation loss.
        training_log_loss = metrics.log_loss(training_targets, training_probabilities)
        validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
        # Occasionally print the current loss.
        print("  Training loss for period %02d: %0.2f" % (period, training_log_loss))
        print("  Validation loss for period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.  SKIPPING PERIODS
        #training_log_losses.append(training_log_loss)
        #validation_log_losses.append(validation_log_loss)
        
    print("Model training finished.")

    # Periods slow down Kaggle, so only do one; makes this graph pointless
    # Output a graph of loss metrics over periods.
    #plt.ylabel("LogLoss")
    #plt.xlabel("Periods")
    #plt.title("LogLoss vs. Periods")
    #plt.tight_layout()
    #plt.plot(training_log_losses, label="training")
    #plt.plot(validation_log_losses, label="validation")
    #plt.legend()

    return linear_classifier


# **6. Training**
# 
# First, create the input functions used to predict.  (Unlike the Programming Exercises, I'm doing this outside of the train_model function since they need to be used later outside of that function:

# In[276]:


# Create input functions.
predict_training_input_fn = lambda: my_input_fn(training_examples, 
                              training_targets[TARGET_STR], 
                              num_epochs=1, 
                              shuffle=False)
predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                validation_targets[TARGET_STR], 
                                num_epochs=1, 
                                shuffle=False)


# And now it's time to train!

# In[277]:


if USING_OLD_MODELS & (not IS_DNN_CLASSIFIER):
    linear_classifier = train_linear_classifier_model(
        learning_rate=.1,
        regularization_strength=0.1,
        steps=1000,  
        batch_size=140,
        periods=1,
        feature_columns=construct_feature_columns(),
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets) 


# **7. ROC and AUC**
# 
# The code for this is found in [Chapter 12's Programming Exercises](https://colab.research.google.com/notebooks/mlcc/logistic_regression.ipynb?hl=en).
# 

# In[278]:


if USING_OLD_MODELS & (not IS_DNN_CLASSIFIER):
    #training_metrics = linear_classifier.evaluate(input_fn=predict_training_input_fn)
    validation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

    #print("AUC on the training set: %0.2f" % training_metrics['auc'])
    #print("Accuracy on the training set: %0.2f" % training_metrics['accuracy'])

    print("AUC on the validation set: %0.2f" % validation_metrics['auc'])
    print("Accuracy on the validation set: %0.2f" % validation_metrics['accuracy'])


# In[279]:


if USING_OLD_MODELS & (not IS_DNN_CLASSIFIER):
    validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
    # Get just the probabilities for the positive class.
    validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])

    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
        validation_targets, validation_probabilities)
    plt.plot(false_positive_rate, true_positive_rate, label="our model")
    plt.plot([0, 1], [0, 1], label="random classifier")
    _ = plt.legend(loc=2)


# *Experiment Results*
# 
# Using various features (previous_submissions, total_cost, teacher_prefix, school_state, grade_category) and various hyperparameters have only yielded an AUC of .57.
# 
# Next steps in this project will be to incorporate more of the word entries (like the essays, titles, descriptions, etc.) to see if they do any better
#     

# **IV. Creating a DNN Classifier**
# 
# Towards the end of the Crash Course was a lesson on DNN Classifiers and embeddings.  The code to do this can be found in this lesson's [Programming Exercises](https://colab.research.google.com/notebooks/mlcc/intro_to_sparse_data_and_embeddings.ipynb?hl=en#scrollTo=eQS5KQzBybTY).
# 
# Though handled in code from earlier, you'll also want to use "embedding columns" instead of "categorical columns with vocabulary list".

# Create the function to train the DNN Classifier:

# In[280]:


def train_dnn_classifier_model(
    learning_rate,
    hidden_units,
    #regularization_strength,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
    """Trains a DNN Classifier model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
        learning_rate: A `float`, the learning rate.
        #regularization_strength: A `float` that indicates the strength of the L1
        #    regularization. A value of `0.0` means no regularization.
        steps: A non-zero `int`, the total number of training steps. A training step
            consists of a forward and backward pass using a single batch.
        feature_columns: A `set` specifying the input feature columns to use.
        training_examples: A `DataFrame` containing one or more columns to use as input features for training.
        training_targets: A `DataFrame` containing exactly one column from to use as target for training.
        validation_examples: A `DataFrame` containing one or more columns to use as input features for validation.
        validation_targets: A `DataFrame` containing exactly one column to use as target for validation.

    Returns:
    A `DNNClassifier` object trained on the training data.
    """
    
    # Create a DNN classifier object.
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=hidden_units,
        optimizer=my_optimizer
    )
    
    # Create input function for training (dependent on batch_size passed in)
    training_input_fn = lambda: my_input_fn(training_examples, 
                      training_targets[TARGET_STR], 
                      batch_size=batch_size)
    
    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")

    # Train the model
    dnn_classifier.train(
    input_fn=training_input_fn,
    steps=steps
    )

    # Compute predictions.
    training_probabilities = dnn_classifier.predict(input_fn=predict_training_input_fn)
    training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

    validation_probabilities = dnn_classifier.predict(input_fn=predict_validation_input_fn)
    validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

    # Compute training and validation loss.
    training_log_loss = metrics.log_loss(training_targets, training_probabilities)
    validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
    print("  Training loss: %0.2f" % (training_log_loss))
    print("  Validation loss: %0.2f" % (validation_log_loss))
        
    print("Model training finished.")

    return dnn_classifier


# Train:

# In[281]:


if USING_OLD_MODELS & IS_DNN_CLASSIFIER:
    dnn_classifier = train_dnn_classifier_model(
        learning_rate=.1,
        hidden_units=[10,10],
        steps=1000,  
        batch_size=140,
        feature_columns=construct_feature_columns(),
        training_examples=training_examples,
        training_targets=training_targets,
        validation_examples=validation_examples,
        validation_targets=validation_targets) 


# AUC

# In[282]:


if USING_OLD_MODELS & IS_DNN_CLASSIFIER:
    validation_metrics = dnn_classifier.evaluate(input_fn=predict_validation_input_fn)

    print("AUC on the validation set: %0.2f" % validation_metrics['auc'])
    print("Accuracy on the validation set: %0.2f" % validation_metrics['accuracy'])


# **V. Dealing with Text-Heavy Features:  Using the Crash Course to Bypass TensorFlow Pain**

# By using either the Linear Classifier or the DNN Classifier, various experiments with different features (that didn't require analyzing a lot of text) and with different hyperparameters never resulted in a good AUC (only .57 or so).   The goal now is to start working through some of the features that contain a large amount of varied text (like the essays) to see if they'll yield any improvements.

# *A Note on TensorFlow Pain and Why the Previous Models Couldn't Handle the Heavy-Text Features*
# 
# In the style of this set of [Programming Exercises](https://colab.research.google.com/notebooks/mlcc/intro_to_sparse_data_and_embeddings.ipynb?hl=en), I implemented (or at least thought I had) a giant vocabulary list for use against the feature "project_resource_summary" text.  However, it had absolutely no affect on AUC at all, which surprised me.
# 
# Taking a look at the training process, it looked like TensorFlow was treating a given example's data for that feature as a *full entire string* (for example:  "My students need 7 ipads and 8 ipod nanos and cookies.").  It was then taking that *full* string and comparing it with the individual words in the vocabulary list ("my", "students", "need", "cookies", etc.).  But since none of those full sentences exist in the vocabulary_list, it was basically doing *nothing*.  What I wanted it to do was look at each individual word in the full string and compare those to the words in the vocabulary list.
# 
# However, I was never able to get TensorFlow to do that with the previous models.  I'm sure there's a way, but nothing I tried worked, and I either didn't ask the right questions in web searches or just couldn't find anything.  Plus, I got tired and frustrated of looking.
# 
# So... what's my solution?  Well, the code in the Programming Exercises linked above has TensorFlow evaluating a full sentence as individual words.  (YES!!!)  Thus, whatever it was doing, I knew that code worked.  That Exercise, however, is based on *loading the data into TensorFlow in a completely different way,* by using something called a TFRecord.  So my solution is to take the data that's currently in my Pandas Dataframe, [save it as a TFRecord](https://stackoverflow.com/questions/41402332/tensorflow-create-a-tfrecords-file-from-csv), and then use the code in the Programming Exercises to have TensorFlow actually do what I want.  It's ridiculous... but it works!!!  

# Note:  From earlier in this notebook, the data has already been shuffled  and split into training/validation sets.
# 
# The below functions are to make it much easier to add and remove features.  All you have to do is edit the DESIRED_FEATURES list.  The following functions then prepare the conversion of that feature data to a format that TFRecord can handle.  (One special note about strings is that TFRecord can't handle those.  Instead, they need to be converted to byte strings first with .encode().)

# In[283]:


#functions to write correct values to TFRecord
#depending on the column value, TFRecord either needs to store a bytes_list, int64_list, or float_list value
def make_bytes_list_append(col_index, value, example):
    example.features.feature[DESIRED_FEATURES[col_index]].bytes_list.value.append(value.encode())

#for the data with actual sentences, split them into individual words
def make_bytes_list_extend(col_index, value, example):
    arr_strings = value.split(' ')
    arr_bstrings = list(map(lambda x: x.encode(), arr_strings))
    example.features.feature[DESIRED_FEATURES[col_index]].bytes_list.value.extend(arr_bstrings)
    
def make_int64_list(col_index, value, example):
    example.features.feature[DESIRED_FEATURES[col_index]].int64_list.value.append(value)

def make_float_list(col_index, value, example):
    example.features.feature[DESIRED_FEATURES[col_index]].float_list.value.append(value)


#function to create an array of the right function to call depending on the feature
def match_feature_with_tfrecord_function():
    arr_functions = []
    for string in DESIRED_FEATURES:
        #features consisting of a full string = bytes, append
        if (string == 'teacher_prefix') or (string == 'school_state') or (string == 'project_grade_category'):
            arr_functions.append(make_bytes_list_append)
        #features consisting of text that should be broken down into words = bytes, extend
        elif (string == 'project_title') or (string == 'project_essay_1') or (string == 'project_essay_2') or (string == 'project_resource_summary'):
            arr_functions.append(make_bytes_list_extend)
        #features consisting of non-integer numbers = float
        elif (string == 'total_cost'):
            arr_functions.append(make_float_list)
        #features consisting of integers = int64
        elif (string == 'teacher_number_of_previously_posted_projects') or (string == 'project_is_approved') or (string == 'year_week'):
            arr_functions.append(make_int64_list)
        
        #not-entirely-decide features.  For these two, treating as separate for now, and as should be split into words = bytes, extend
        elif (string == 'project_subject_categories') or (string == 'project_subject_subcategories'):
            arr_functions.append(make_bytes_list_extend)
            
        #not sure what doing with these yet:
        #'project_submitted_datetime',
        #(there was also description data in the resources file)
        
    return arr_functions

#loop through all the rows in a dataset, convert each individual cell to the right data type that TFRecord requires,
#take those tf.examples and save them into a TFRecord file
#NOTE:  REQUIRES arr_tfrecord_funcs = match_feature_with_tfrecord_function() to be called first
def save_rows_as_TFRecord(array_of_rows, tfrecord_file_name):
    with tf.python_io.TFRecordWriter(tfrecord_file_name) as writer:
        last_column = len(DESIRED_FEATURES)
        for row in array_of_rows:
            example = tf.train.Example()
            for col_index in range(0, last_column):
                arr_tfrecord_funcs[col_index](col_index, row[col_index], example)  #for each feature, call the corresponding function

            writer.write(example.SerializeToString())
    return

#just a test function to make sure all the above works
def TEST_save_rows_as_TFRecord(array_of_rows, tfrecord_file_name):
    last_column = len(DESIRED_FEATURES)
    for row in array_of_rows:
        example = tf.train.Example()
        for col_index in range(0, last_column):
            arr_tfrecord_funcs[col_index](col_index, row[col_index], example)  #for each feature, call the corresponding function
        print(example)


# Now test to make sure that all works as intended by looking at a couple of examples before writing to a TFRecord::

# In[284]:


#testing new functions:
arr_tfrecord_funcs = match_feature_with_tfrecord_function()
testdata = training_examples[DESIRED_FEATURES][0:2]

TEST_save_rows_as_TFRecord(testdata.values, 'test')


# Using all the previous functions, now actually save the data sets as TFRecords.  This step takes a LONG time.  However, using this TFRecord method makes the actual training later on go blazing fast!

# In[285]:


#Files to save:
TRAINING_TFRECORD = "training.tfrecords" #for training_examples
VALIDATION_TFRECORD = "validation.tfrecords" #for validation_examples
TEST_TFRECORD = "test.tfrecords" #for combined_test_dataset

arr_tfrecord_funcs = match_feature_with_tfrecord_function() #grab the conversion functions to call for each cell
#make the TFRecord for training data:
save_rows_as_TFRecord(training_examples[DESIRED_FEATURES].values, TRAINING_TFRECORD)
#make the TFRecord for validation data:
save_rows_as_TFRecord(validation_examples[DESIRED_FEATURES].values, VALIDATION_TFRECORD)
#make the TFRecord for test data:
#save_rows_as_TFRecord(combined_test_dataset[DESIRED_FEATURES].values, TEST_TFRECORD)


# The following code is taken straight from the Exercise but adapted to fit the data for this project and, once again, to make it easier to add/remove features by only editing the DESIRED_FEATURES constant from earleir:

# In[286]:


def parse_function(record):
    """Extracts features and labels from a TFRecord file.

    Args:
        record: file name of the TFRecord file    
    Returns:
        A `tuple` `(features, labels)`:
        features: A dict of tensors representing the features
        labels: A tensor with the corresponding labels.
    """
    features_to_parse = {}
    for string in DESIRED_FEATURES:
        #features consisting of a full string that wasn't split apart
        if (string == 'teacher_prefix') or (string == 'school_state') or (string == 'project_grade_category'):
            features_to_parse[string] = tf.FixedLenFeature(shape=[1], dtype=tf.string)
        #features consisting of a string split apart into pieces of varying lengths.  Thus, need the VarLenFeature:
        elif (string == 'project_title') or (string == 'project_essay_1') or (string == 'project_essay_2') or (string == 'project_resource_summary'):
            features_to_parse[string] = tf.VarLenFeature(dtype=tf.string)
        #features consisting of non-integer numbers = float
        elif (string == 'total_cost'):
            features_to_parse[string] = tf.FixedLenFeature(shape=[1], dtype=tf.float32)
        #features consisting of integers = int64
        elif (string == 'teacher_number_of_previously_posted_projects') or (string == 'project_is_approved') or (string == 'year_week'):
            features_to_parse[string] = tf.FixedLenFeature(shape=[1], dtype=tf.int64)
            
        #not-entirely-decide features.  For these two, treating as separate for now
        #and as should be split into words = need the VarLenFeature
        elif (string == 'project_subject_categories') or (string == 'project_subject_subcategories'):
            features_to_parse[string] = tf.VarLenFeature(dtype=tf.string)

    parsed_features = tf.parse_single_example(record, features_to_parse)
    
    final_features = {}
    for string in DESIRED_FEATURES:
        #features consisting of a full string that wasn't split apart
        if (string == 'teacher_prefix') or (string == 'school_state') or (string == 'project_grade_category'):
            final_features[string] = parsed_features[string]
        #features consisting of a string split apart into pieces of varying lengths.  Thus, need the VarLenFeature:
        elif (string == 'project_title') or (string == 'project_essay_1') or (string == 'project_essay_2') or (string == 'project_resource_summary'):
            final_features[string] = parsed_features[string].values ##DO NOT FORGET .values for the VarLenFeature strings (doing so creates a cryptic error that was hard to figure out)
        #features consisting of non-integer numbers = float
        elif (string == 'total_cost'):
            final_features[string] = parsed_features[string]
        #features consisting of integers = int64  (except project_is_approved is handled separately)
        elif (string == 'teacher_number_of_previously_posted_projects') or (string == 'year_week'):
            final_features[string] = parsed_features[string]
            
        #currently treating as separate and as pieces of varying length.  VarLenFeature  ** .values  !!
        elif (string == 'project_subject_categories') or (string == 'project_subject_subcategories'):
            final_features[string] = parsed_features[string].values
            
    labels = parsed_features['project_is_approved']
    
    return final_features, labels


# Check to see that this is working and that the data is in the correct format (ie, separated strings).  By the way, if you happen to look at the output in the Programming Exercises, it won't have the byte strings displayed like b'teaching'.  Instead, it will have odd numerical representations like '\x0323teaching'.  That difference initially concerned me that my TFRecord attempt here had failed.  However, the Colab Programming Exercises use Python 2, which apparently displays byte strings in that way.  Python 3 here on Kaggle just uses the b'value' approach.  So the data is formatted correctly.

# In[287]:


ds = tf.data.TFRecordDataset(TRAINING_TFRECORD)
ds = ds.map(parse_function)

n = ds.make_one_shot_iterator().get_next()
sess = tf.Session()
sess.run(n)


# And also check the validation set just in case:

# In[288]:


ds = tf.data.TFRecordDataset(VALIDATION_TFRECORD)
ds = ds.map(parse_function)

n = ds.make_one_shot_iterator().get_next()
sess = tf.Session()
sess.run(n)


# Now need to edit code to make changes easier (like just setting a single steps value)based on the Programming Exercises here:
# https://colab.research.google.com/notebooks/mlcc/intro_to_sparse_data_and_embeddings.ipynb?hl=en#scrollTo=5_C5-ueNYIn_

# In[292]:


# Create an input_fn that parses the tf.Examples from the given files,
# and split them into features and targets.
def record_input_fn(input_filenames, batch_size=100, num_epochs=None, shuffle=True):
    # Same code as above; create a dataset and map features and labels.
    ds = tf.data.TFRecordDataset(input_filenames)
    ds = ds.map(parse_function)

    if shuffle:
        ds = ds.shuffle(10000)

    # Our feature data is variable-length, so we pad and batch
    # each field of the dataset structure to whatever size is necessary.
    ds = ds.padded_batch(batch_size, ds.output_shapes)

    ds = ds.repeat(num_epochs)


    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# In[297]:


if USING_OLD_MODELS == False:
#let's just use their exact code for now:
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=.05)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

    feature_columns = construct_feature_columns()

    classifier = tf.estimator.LinearClassifier(
      feature_columns=feature_columns,
      optimizer=my_optimizer,
    )

    classifier.train(
      input_fn=lambda: record_input_fn(TRAINING_TFRECORD),
      steps=2000)

    predict_training_input_fn = lambda: record_input_fn(TRAINING_TFRECORD, 
                                  num_epochs=1, 
                                  shuffle=False)
    predict_validation_input_fn = lambda: record_input_fn(VALIDATION_TFRECORD, 
                                    num_epochs=1, 
                                    shuffle=False)

# Compute predictions.
#training_probabilities = classifier.predict(input_fn=predict_training_input_fn)
#training_probabilities = np.array([item['probabilities'] for item in training_probabilities])

#validation_probabilities = classifier.predict(input_fn=predict_validation_input_fn)
#validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])

# Compute training and validation loss.
#training_log_loss = metrics.log_loss(training_targets, training_probabilities)
#validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
#print("  Training loss: %0.2f" % (training_log_loss))
#print("  Validation loss: %0.2f" % (validation_log_loss))

#^Yeah... don't know how you would get the target labels via this method.  Hopefully this shows them:


# In[298]:


if USING_OLD_MODELS == False:
    #evaluation_metrics = classifier.evaluate(
    #  input_fn=predict_training_input_fn,
    #  steps=1800)

    #print("Training set metrics:")
    #for m in evaluation_metrics:
    #    print(m, evaluation_metrics[m])
    #print("---")

    evaluation_metrics = classifier.evaluate(
      input_fn=predict_validation_input_fn,
      steps=2000)

    print("Test set metrics:")
    for m in evaluation_metrics:
        print(m, evaluation_metrics[m])
    print("---")


# YESSSSSSSS, IT WORKED!!!  Finally something that is NOT .56 AUC.  (Surprised that this is also using less disk space somehow?  Not sure what that means.)  [...or because set wasn't randomized?]
# 1st experiment = 500 steps for 0.64 AUC on the validation set.  (AUC on Training was .76, but... I'm not sure what that even means in the context.)
# 
# 2nd experiment = 1000 steps for .65 AUC on the validation set.
# 
# How come this is SO much faster, too?  Feels like the training finishes in a matter of seconds.  
# 

# 3rd Experiment = BLAZING FAST.  How is this so fast now?
# **Added project_title as a feature**
# 500 steps, learning rate = .01
# AUC = .56, so much worse.  Interesting.  And that awful .56 again....  Huh.
# 
# 4th
# 1000 steps, learning rate = .1
# AUC = **.67**  (Well, .665)
# 
# 5th
# 1500, learning rate = .5
# AUC =.63
# Yeah, doesn't seem like project_title has really added anything
# 
# 6th
# 1500, learning rate = .1
# AUC = .66 or so again
# 

# 7th:  added the two essays as features (separate vocab lists for each)
# 1000 steps, learning rate = .1
# **AUC = .716, highest yet**
# Accuracy actually seemed low, though (.81); wasn't the dataset like 85% got accepted?  Some bias in the model?
# 
# 8th:  = same but altering batch_size this time from 25 to 100
# wow, that made it way worse.  (final loss = 47 instead of 10ish)
# AUC = .735. Wait, what?  What does that loss value mean then?  ????
# Also, is it strange that the evaluation only did 300/1000 steps?  (Maybe that's just because it ran out of samples to run due to the higher batch size?)
# 
# 9th= same (still batch size of 100) but steps = 1800
# **AUC = .742**  Very.
# 
# 10th = same (but I'm curious what shuffle=False does):
# AUC = .741.  Apparently, not much!

# 11th = don't think this will matter (and may make things worse), but let's also include teacher prefix and project grade category.  batch_size = 100, learning_rate = .1, shuffle=true, steps = 1800
# AUC = .738.  Yeah, those two features didn't do anything.
# 
# 12th = same hyperparameters BUT ditched teacher_prefix and project_grade_category.  Added in project_category and project_subcategory
# AUC = .743, so not much different.
# 
# 13th = different hyperparameters out of curiosity.  batch_size = 100, steps = 2500, learning_rate = .3
# AUC = .71
# 
# 14th = different hyperparameters:  batch_size = 100, steps = 2000, learning_rate = .05
# **AUC = .754**, but not really that different.

# 15th = batch_size = 100, steps = 2000, learning_rate = .05  **included the new year_week feature**
# AUC = .755.  No difference, huh.
# 
# 16th = same but steps = 2500, learning_rate = .001
# Yikes, AUC of .69.  Definitely doesn't like lower learning rates
# 
# 17th = learning_rate of 1  steps = 2000  (high learning rates = just spike the loss like crazy)
# AUC = .64.  Haha, yeah, no high learning_rates
# 
# 18th = learning_rate = .1, 1800 steps AUC = .74
