#!/usr/bin/env python
# coding: utf-8

# # Understanding Approval:- Donor Choose EDA
# ![](https://cdn.donorschoose.net/images/media-logo-tagline-reversed@2x.jpg)
# 
# # Contents:
# * Introduction
#     * About Donors Choose
#     * Competition Objective
#     * Kernel objective
# * Imports and overview
# * Custom Helper Functions
#     * Plotting Functions
#     * Text functions
#         * Extract text stats
#         * Make Wordclouds
# * Individual Feature impact on Approval rates
#     * Categorical features - Teacher-prefix, Gender, Grade/class
#     * Cleaning up - Subject category and Subject sub-category
# * Text columns exploration
#     * Title
#     * Student description
#     * Project description
#     * Resource summary
# * Resources dataset
# * Price points
#     * Exploring some costly items
# * Pre-processing and cleaning text
# * Feature Engineering
#     * Label encoding
#     * Create date features
#     * Custom Vectorizer for ELI5 compatability
# * Baseline Models -- XGBoost and LightGBM
#     * ROC curve and 
# * Understanding how the model predicts - ELI5
#     * Explore correct classifications
#     * Explore mis-classifications
# 
# # 1. Introduction:
# ## 1.1 About Donors Choose:
# [Donorschoose.org](https://www.donorschoose.org/about) is a crowdfunding platform which connects Public school teachers and Donors. 
# ![](http://stuffonix.com/wp-content/uploads/2017/09/donorschoose-how-it-work.jpg)
# 
# 
# As per their [website](https://www.donorschoose.org/about/impact.html), they have raised $645,575,280 till date and claim that 77 percent of all the public schools in America have at least one teacher who has posted a project on DonorsChoose.org. Amazing!
# 
# With such high numbers, the number of applications they receive is increasing every year and the current screening process is manually vetting the applications by a team of volunteers. As a result, there are three main problems they need to solve:
# 
# * How to scale current manual processes and resources to screen 500,000 projects so that they can be posted as quickly and as efficiently as possible
# * How to increase the consistency of project vetting across different volunteers to improve the experience for teachers
# * How to focus volunteer time on the applications that need the most assistance
# 
# ## 1.2 Competition Objective:
# 
# The goal of the competition is to predict whether or not a DonorsChoose.org project proposal submitted by a teacher will be approved, using the text of project descriptions as well as additional metadata about the project, teacher, and school. DonorsChoose.org can then use this information to identify projects most likely to need further review before approval.
# 
# ## 1.3 Kernel objective:
# 
# To explore and understand factors that make a successful project and hopefully create an approval process pipeline/algorithm to help Donorschoose.org with the vetting process of approving a project.
# 
# # 2. Imports and overview:
# 
# Lets get started by importing all the required packages and performing basic sanity checks like the test-train split ratio, Missing value checks,etc.

# In[ ]:


#peak
get_ipython().system('ls -l ../input/*')


# In[ ]:


#import required packages
#basics
import pandas as pd 
import numpy as np

#viz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
import seaborn as sns
from wordcloud import WordCloud ,STOPWORDS

#nlp
import re    #for regex
import nltk
from nltk.corpus import stopwords
import gensim
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()
eng_stopwords = set(stopwords.words("english"))


#misc
import gc
import time
import warnings

#settings
start_time=time.time()
color = sns.color_palette()
sns.set_style("dark")
warnings.filterwarnings("ignore")


get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#import all the files!
train=pd.read_csv("../input/train.csv")
resources=pd.read_csv("../input/resources.csv")
test=pd.read_csv("../input/test.csv")
sample_sub=pd.read_csv("../input/sample_submission.csv")


# In[ ]:


# peak at the data
train.head()


# In[ ]:


#take a peak
resources.head()


# ### Check the test-train split ratio:

# In[ ]:


#check test train split
nrow_train=train.shape[0]
nrow_test=test.shape[0]
sum=nrow_train+nrow_test
print("Checking proportion of Test-train split")
print("       : train  : test")
print("rows   :",nrow_train,":",nrow_test)
print("perc   :",round(nrow_train*100/sum),"    :",round(nrow_test*100/sum))


# In[ ]:


# check for missing values
print("Check for Percent of missing values in Train dataset")
null_check=train.isnull().sum()
(null_check/len(train))*100


# In[ ]:


# check for missing values
print("Check for Percent of missing values in RESOURCES file")
null_check=resources.isnull().sum()
(null_check/len(resources))*100


# ### Target Variable:
# The target variable for this competition is a Binary variable which indicates if the project was **approved to be hosted on the site** or not.
# 
# Note that this does not indicate if the project was **funded** or not! 

# In[ ]:


x=train.project_is_approved.value_counts()
#plot
plt.figure(figsize=(8,4))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Target Variable")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Project is approved?', fontsize=12)
#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
print("Approval rate:",x[1]/(x[0]+x[1])*100)


# 
# During the training period, there is an impressive **84%** approval rate! 
# 
# There are some null values in some fields, Project essays 3,4 and Resource description.
# 
# As per the data description, the Project essays 3,4 are just optional descriptive fields that the teachers can enter. Hence, it is ok that we observe around 96.5% empty values.
# 
# But the null entries in the resources dataset is fishy. Let's explore more on that.

# In[ ]:


# take IDs of the projects which have null description
print("There are",resources.description.isnull().sum(),"NULL entries in description column of Resources dataset")
null_ids=resources[resources.description.isnull()].id
print("Those Null entries are from",len(null_ids.unique()),"projects")
null_entries_train=train[train.id.isin(null_ids)]
print("There are",len(null_entries_train),"of these projects are in train")
null_entries_test=test[test.id.isin(null_ids)]
print("There are",len(null_entries_test),"of these projects are in test")


# In[ ]:


x=null_entries_train.project_is_approved.value_counts()
print("Approval rate of projects with NULL as the description under Resources:",x[1]/(x[0]+x[1])*100)
x=null_entries_train[null_entries_train.teacher_number_of_previously_posted_projects==0].project_is_approved.value_counts()
print("Approval rate of projects with NULL as the description under Resources and 0 previous project submissions:",x[1]/(x[0]+x[1])*100)


# There is a significant dip in approval rates (from 85% to 62%) if there is no description of the resources and if the project is the first submission by a teacher.

# In[ ]:


null_entries_train[null_entries_train.project_is_approved==0].head(5)
# No obvious predictable pattern :(


# In[ ]:


null_entries_train[null_entries_train.project_is_approved==0].project_resource_summary.iloc[0]


# In[ ]:


null_entries_train[null_entries_train.project_is_approved==0].project_resource_summary.iloc[1]


# In[ ]:


null_entries_train[null_entries_train.project_is_approved==0].project_resource_summary.iloc[6]


# In[ ]:


null_entries_train[null_entries_train.project_is_approved==0].project_resource_summary.iloc[9]


# A lot of them seem to be **Art Supplies!? **. Is that supposed to mean something? Anyways, let's continue on...
# 
# 
# # Custom helper functions:
# Creating some functions here that would be used across multiple analysis.
# 
# ## Plotting functions:
# Creating a simple function to create two plots.
# * Frequency plot
# * Approval rate across the target column entries
# 

# In[ ]:


#making this for easy subsetting later
approvals=train[train.project_is_approved==1]
rejects=train[train.project_is_approved==0]

# lets make a simple re-usable function to make the plots!
# This lets us add more functionality if needed later and it would be replicated across all plots!
def make_custom_plot(target_column_name='',title='',total_counts=None,approvals_counts=None,rejects_counts=None,x_rotation_angle=0):
    """        
    Description:
        Creates a 1x2 plot of 1. the # of projects across the variable and the 2. A stacked Percentage bar chart of the Approval rates across the variable
    Useage: 1) make_custom_plot('gender','Analyzing Gender')
            2) make_custom_plot(total_counts=x,approvals_counts=x1,rejects_counts=x2)
    """
    if(target_column_name!=''):
        x=train[target_column_name].value_counts()
        x1=approvals[target_column_name].value_counts()
        x2=rejects[target_column_name].value_counts()
    else:
        x=total_counts
        x1=approvals_counts
        x2=rejects_counts
        target_column_name=title
    #plot initiate
    plt.figure(figsize=(16,6))
    
    #super title
    plt.suptitle(title,fontsize=18)
    plt.subplot(121)
    #title and labels for plot1
    plt.title('Total Projects Submitted',fontsize=12)
    plt.ylabel('# of Projects', fontsize=12)
    plt.xlabel(target_column_name, fontsize=12)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=x_rotation_angle)
    # Barplot
    ax= sns.barplot(x.index, x.values, alpha=0.8)

    #adding the text labels
    rects = ax.patches
    labels = x.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
    
    
    plt.subplot(122)
    #title and labels for plot2
    plt.title('Approval Rate',fontsize=12)
    plt.ylabel('Percent Approved Projects', fontsize=12)
    plt.xlabel(target_column_name, fontsize=12)
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=x_rotation_angle)
    # https://python-graph-gallery.com/13-percent-stacked-barplot/
    r=np.arange(len(x))
    totals=x
    greenBars = [i / j * 100 for i,j in zip(x1, totals)]
    redBars = [i / j * 100 for i,j in zip(x2, totals)]

    barWidth = 0.85
    names = x.index
    # Create green Bars
    plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth, label="Approved")
    # Create red Bars
    plt.bar(r, redBars, bottom=greenBars, color='red', edgecolor='white', width=barWidth, label="Rejected")
    # Custom x axis
    plt.xticks(r, names)
    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1)
    plt.show()


# ## Text Functions:
# Creating custom functions to be used in the various text fields in the dataset.
# ### Extract text stats:
# This function does the following
# * Gets basic text statistics (Word count, Unique word count) from the text column
# * Plot Violin plot(Extension of box plot) across Project approval for the computed variables
# * Create a KDE plot for unique word percent
# 

# In[ ]:


# Making a function instead of writing code for single text columns so that its more scalable and can be applied to all text cols
def get_text_stats(text_col):
    """
    Get Wordcount,Unique Wordcount and WordCount Percent and make appropriate visuals
    Todo: Add more text stats
    """
    title="Text Stats of " + text_col
    target_col='project_is_approved'
    # Borrowed from previous work at https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda
    text_stats=pd.DataFrame()
    text_stats[target_col]=train[target_col]
    #Word count 
    text_stats['word_count']=train[text_col].apply(lambda x: len(str(x).split()))
    #Unique word 
    text_stats['count_unique_word']=train[text_col].apply(lambda x: len(set(str(x).split())))
    #Word count percent in each comment:
    text_stats['word_unique_percent']=(text_stats['count_unique_word']*100)/text_stats['word_count']
    
    temp_df = pd.melt(text_stats, value_vars=['word_count', 'count_unique_word'], id_vars=target_col)
    
    print("------ Sample from an Approved project ------\n")
    print(approvals[text_col].iloc[0])
    
    print("\n------ Sample from a Rejected project ------\n")
    print(rejects[text_col].iloc[0])
    
    
    #plotting
    plt.figure(figsize=(16,5))
    plt.subplot(121)
    plt.suptitle(title,fontsize=16)
    #re-shaping as required
    plt.title("Word Count")
    sns.violinplot(x='variable', y='value', hue=str(target_col), data=temp_df,inner='quartile')
    plt.ylabel('# of projects', fontsize=12)
    
    plt.subplot(122)
    plt.title("Percentage of Unique words - effect on Approval")
    ax=sns.kdeplot(text_stats[text_stats.project_is_approved == 0].word_unique_percent, label="Not Approved",shade=True,color='r')
    ax=sns.kdeplot(text_stats[text_stats.project_is_approved == 1].word_unique_percent, label="Approved")
    plt.legend()
    plt.xlabel('Percent unique words', fontsize=12)
    plt.ylabel('# of projects', fontsize=12)
    plt.show()


# ### Make Wordclouds:
# A simple function that takes in a text column/field and makes separate word-clouds for approved and rejected projects.

# In[ ]:


# for the wordcloud
stopword=set(STOPWORDS)

# Custom Adding some stop words to make better and more meaningful wordclouds
stopword.add('will')
stopword.add('student')
stopword.add('students')
stopword.add('class')
stopword.add('classroom')
stopword.add('child')
stopword.add('children')
stopword.add('teacher')
stopword.add('school')
stopword.add('need')
stopword.add('needs')

def make_word_clouds(text_col):
    """
    Makes two wordclouds : one for Approvals and one for rejects
    
    Todo: Think of faceted clouds across categorical var (Eg:Grade category)
    """
    plt.figure(figsize=(25,8))
    plt.subplot(211)
    # Get text col from approvals subset
    text=approvals[text_col].values
    # make wordcloud
    wc= WordCloud(background_color="black",max_words=100,stopwords=stopword)
    wc.generate(" ".join(text))

    plt.axis("off")
    plt.title("Words frequented in Approved Projects", fontsize=12)
    #https://matplotlib.org/examples/color/colormaps_reference.html for colormaps
    plt.imshow(wc.recolor(colormap='Pastel1',random_state=17), alpha=0.98)
    
    plt.subplot(212)
    # Get text col from Rejects subset
    text=rejects[text_col].values

    # make wordcloud
    wc= WordCloud(background_color="white",max_words=100,stopwords=stopword)
    wc.generate(" ".join(text))
    
    plt.axis("off")
    plt.title("Words frequented in Rejected Projects", fontsize=12)
    plt.imshow(wc.recolor(colormap='inferno',random_state=17), alpha=0.98)
    plt.show()


# # Individual Feature's impact on Approval:
# 
# Let's explore the impact of our descriptive features on Project approval rates.
# 
# ## 1) Teacher-prefix:
# This variable gives information of the title of the teacher submitting the request. 
# 
# Also, indirectly we can infer the gender from this variable.

# In[ ]:


make_custom_plot('teacher_prefix','Does a Title affect approval?')


# ### 2) Gender:
# This is a created field from the teacher prefix field. The mapping is as follows,
# * Mrs, Ms --> Female
# * Mr. --> Male
# * Teacher,Dr --> Unknown

# In[ ]:


# Creating the gender column
gender_mapping = {"Ms.": "Female", "Mrs.":"Female", "Mr.":"Male", "Teacher":"Unknown", "Dr.":"Unknown", np.nan:"Unknown"  }
train["gender"] = train.teacher_prefix.map(gender_mapping)
approvals["gender"] = approvals.teacher_prefix.map(gender_mapping)
rejects["gender"] = rejects.teacher_prefix.map(gender_mapping)
test['gender'] = test.teacher_prefix.map(gender_mapping)
make_custom_plot('gender','Analyzing Gender')


# Unsurprisingly, there are more Female teachers. But that does not affect/bias the Approval rate at all.
# 
# ### 3) Project Grade Category:
# This variable shows us the class/grade of the students that would benefit from the donation!
# 

# In[ ]:


make_custom_plot('project_grade_category','Do smaller kids get more approval rates?')


# While Smaller kids seem to get more projects, there is no significant effect of Grade/class on Approval rates.
# 
# ## 4) Subject Category:
# This field shows us the Subject category/categories that this project aims to help at. Note that sometimes, there are multiple subject categories tagged to one project.

# In[ ]:


x= train.project_subject_categories.value_counts()
#prep for chart
x=x.sort_values(ascending=False)
x=x.iloc[0:20]

#chart
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8,color=color[0])
plt.title("What are the frequent subject categories?",fontsize=16)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)
plt.ylabel('# Projects', fontsize=12)
plt.xlabel('Subject Category', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
print("There are ",len(train.project_subject_categories.unique()),"unique Subject Categories")


#  The subject categories can be cleaned a bit further.
#  
#  For example, the third most popular category can be broken down into 1)"Literacy & language", 2)"Math & science" separately

# In[ ]:


# Grouping similar categories for overall
subject_cats=','.join(train['project_subject_categories'])
cats=pd.DataFrame(subject_cats.split(','),columns=['project_subject_categories']) # to split on ","
cats.project_subject_categories=cats.project_subject_categories.str.strip() # to remove unwanted spaces
x=cats.project_subject_categories.value_counts()
print("There are",len(x),"different subject categories after cleaning")

# repeat for approved group and rejected group
# Grouping similar categories for approved 
subject_cats=','.join(approvals['project_subject_categories'])
cats=pd.DataFrame(subject_cats.split(','),columns=['project_subject_categories']) # to split on ","
cats.project_subject_categories=cats.project_subject_categories.str.strip() # to remove unwanted spaces
x1=cats.project_subject_categories.value_counts()

# Grouping similar categories for rejected
subject_cats=','.join(rejects['project_subject_categories'])
cats=pd.DataFrame(subject_cats.split(','),columns=['project_subject_categories']) # to split on ","
cats.project_subject_categories=cats.project_subject_categories.str.strip() # to remove unwanted spaces
x2=cats.project_subject_categories.value_counts()


# In[ ]:


make_custom_plot(title='Subject Category',total_counts=x,approvals_counts=x1,rejects_counts=x2,x_rotation_angle=80)


# Literature/Language supplies seem to be in high demand. I guess books would fall into this category. Let's explore that more in detail in another section.
# 
# ## 5) Project Sub-Category:
# Performing a similar analysis for the project sub-category.

# In[ ]:


x= train.project_subject_subcategories.value_counts()
#prep for chart
x=x.sort_values(ascending=False)
x=x.iloc[0:20]

#chart
plt.figure(figsize=(16,4))
ax = sns.barplot(x.index, x.values, alpha=0.8,color=color[0])
plt.title("What are the frequent subject sub categories?",fontsize=16)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)
plt.ylabel('# Projects', fontsize=12)
plt.xlabel('Subject sub-Category', fontsize=12)

#adding the text labels
rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
print("There are",len(train.project_subject_subcategories.unique()),"unique Subject Categories")


# In[ ]:


# Grouping similar categories for overall
subject_cats=','.join(train['project_subject_subcategories'])
cats=pd.DataFrame(subject_cats.split(','),columns=['project_subject_subcategories']) # to split on ","
cats.project_subject_subcategories=cats.project_subject_subcategories.str.strip() # to remove unwanted spaces
x=cats.project_subject_subcategories.value_counts()
print("There are ",len(x)," different subject sub-categories after Cleaning")
print("They are:- ",x.index.values)
# repeat for approved group and rejected group
# Grouping similar categories for approved 
subject_cats=','.join(approvals['project_subject_subcategories'])
cats=pd.DataFrame(subject_cats.split(','),columns=['project_subject_subcategories']) # to split on ","
cats.project_subject_subcategories=cats.project_subject_subcategories.str.strip() # to remove unwanted spaces
x1=cats.project_subject_subcategories.value_counts()

# Grouping similar categories for rejected
subject_cats=','.join(rejects['project_subject_subcategories'])
cats=pd.DataFrame(subject_cats.split(','),columns=['project_subject_subcategories']) # to split on ","
cats.project_subject_subcategories=cats.project_subject_subcategories.str.strip() # to remove unwanted spaces
x2=cats.project_subject_subcategories.value_counts()


# In[ ]:


# Plotting top 8 to avoid clutter
make_custom_plot(title='Subject Sub-Category',total_counts=x.iloc[0:8],approvals_counts=x1.iloc[0:8],rejects_counts=x2.iloc[0:8],x_rotation_angle=80)


# In[ ]:


end_preprocess=time.time()
print("Time till sub-category:",end_preprocess-start_time,"s")


# ## Title:
# 
# Project title. 
# 
# 

# In[ ]:


text_col='project_title'
target_col='project_is_approved'
# Borrowed from previous work at https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda
text_stats=pd.DataFrame()
text_stats[target_col]=train[target_col]
#Word count 
text_stats['word_count']=train[text_col].apply(lambda x: len(str(x).split()))
#Unique word 
text_stats['count_unique_word']=train[text_col].apply(lambda x: len(set(str(x).split())))
text_stats.groupby('project_is_approved').mean()


# In[ ]:


temp=train.groupby('project_title')['project_is_approved'].agg(['sum','count'])
temp['approval_rate']=(temp['sum']*100)/temp['count']
temp.columns=['# of projects approved','# of total projects','Approval rate']
temp=temp.sort_values(by='# of total projects',ascending=False)
temp=temp.iloc[0:25]
temp


# ### Looks like **Wiggle while you work!** is a really famous phrase on the platform. It's having several entries with minor changes. 
# ### Also, the Approval rates are impressive **~91%** for projects with that title.

# In[ ]:


make_word_clouds(text_col='project_title')


# # Project Essays:
# 
# On May 17th, 2016, the DonorsChoose.org application switched from having 4 essay prompts to just 2 prompts, so from that point forward, only project_essay_1 and project_essay_2 contain text, and project_essay_3 and project_essay_4 have NaNs.
# 
# Here's a summary of the essay prompts before and after that date.
# 
# **Before May 17th, 2016:**
# 
# * project_essay_1: "Introduce us to your classroom"
# * project_essay_2: "Tell us more about your students"
# * project_essay_3: "Describe how your students will use the materials you're requesting"
# * project_essay_4: "Close by sharing why your project will make a difference"
# 
# **May 17th, 2016 and beyond: **
# 
# * project_essay_1: "Describe your students: What makes your students special? Specific details about their background, your neighborhood, and your school are all helpful."
# * project_essay_2: "About your project: How will these materials make a difference in your students' learning and improve their school lives?"
# 
# 
# As @HeadsorTails explains in this [discussion post](https://www.kaggle.com/c/donorschoose-application-screening/discussion/51352#292941), performing the following changes to clean up the NaNs.
# * Combine essay_1 and essay_2 before May 17th to make "student_description" and use essay_1 after May 17th directly
# * Combine essay_3 and essay_4 before May 17th to make "project_description" and use essay_2 after May 17th directly
# 

# In[ ]:


# Before performing changes , simple check
x=train[train.project_essay_3.notnull()]
print("The last time an entry occured in Project essay 3 -- ",x['project_submitted_datetime'].max())


# In[ ]:


# Making the First essay column :student_description
train['student_description']=train['project_essay_1']
#performing the adjustment
# df.loc[selection criteria, columns I want] = value
train.loc[train.project_essay_3.notnull(),'student_description']=train.loc[train.project_essay_3.notnull(),'project_essay_1']+train.loc[train.project_essay_3.notnull(),'project_essay_2']


# In[ ]:


#repeat for test dataset
test['student_description']=test['project_essay_1']
test.loc[test.project_essay_3.notnull(),'student_description']=test.loc[test.project_essay_3.notnull(),'project_essay_1']+test.loc[test.project_essay_3.notnull(),'project_essay_2']


# In[ ]:


# Making the second essay column : project_description
train['project_description']=train['project_essay_2']
#performing the adjustment
# df.loc[selection criteria, columns I want] = value
train.loc[train.project_essay_3.notnull(),'project_description']=train.loc[train.project_essay_3.notnull(),'project_essay_3']+train.loc[train.project_essay_3.notnull(),'project_essay_4']


# In[ ]:


test['project_description']=test['project_essay_2']
test.loc[test.project_essay_3.notnull(),'project_description']=test.loc[test.project_essay_3.notnull(),'project_essay_3']+test.loc[test.project_essay_3.notnull(),'project_essay_4']


# In[ ]:


#check
train[train.project_essay_3.notnull()].head(2)


# In[ ]:


# check
test[test.project_essay_3.notnull()].head(1).project_description.values


# In[ ]:


#remove unwanted colunms
del(train['project_essay_1'])
del(train['project_essay_2'])
del(train['project_essay_3'])
del(train['project_essay_4'])
del(test['project_essay_1'])
del(test['project_essay_2'])
del(test['project_essay_3'])
del(test['project_essay_4'])

#update the subsets
approvals=train[train.project_is_approved==1]
rejects=train[train.project_is_approved==0]


# ## Student Description (Project essay 1):
# 
# "Describe your students: What makes your students special? Specific details about their background, your neighborhood, and your school are all helpful."
# 
# Note: The data has been altered to account for the change at May 17th, 2016.

# In[ ]:


get_text_stats(text_col='student_description')


# Almost Identical plots across project approval in both word count and unique word count! Looks like they won't be much use here. 

# In[ ]:


make_word_clouds(text_col='student_description')


# ## Project Description (Project essay 2):
# "About your project: How will these materials make a difference in your students' learning and improve their school lives?"
# 
# Note: The data has been altered to account for the change at May 17th, 2016.

# In[ ]:


get_text_stats(text_col='project_description')


# In[ ]:


make_word_clouds(text_col='project_description')


# ## Project resource summary:
# This variable contains a short summary of the resources needed for the project.

# In[ ]:


get_text_stats('project_resource_summary')


# In[ ]:


make_word_clouds('project_resource_summary')


# In[ ]:


end_time=time.time()
print("Time till plotting section",end_time-start_time,"s")


# # Adding features from the Resources Dataset:
# 

# In[ ]:


resources['total_cost']=resources['quantity']*resources['price']


# In[ ]:


# Group by and get concat of the description
resources['description']=resources['description'].astype(str)
x=resources.groupby('id')['description'].apply(lambda x: "%s" % ', '.join(x))   #https://stackoverflow.com/questions/17841149/pandas-groupby-how-to-get-a-union-of-strings
x.head(2)


# In[ ]:


# project level resource stats
resources_agg=resources.groupby('id')['quantity','price','total_cost'].agg({'quantity':['sum','count'],'price':['mean'],'total_cost':['sum']})
resources_agg.columns=['item_quantity_sum','variety_of_items','avg_price_per_item','total_cost']
resources_agg['collated_description']=x
resources_agg=resources_agg.reset_index()
#resources_agg=resources_agg.sort_values("total_cost",ascending=False)
resources_agg.head()


# In[ ]:


train_merge=pd.merge(left=train,right=resources_agg,on='id',how='left')
train_merge.sort_values("total_cost",ascending=False).head()


# In[ ]:


x=train_merge.total_cost.value_counts()
#sort by price
x=x.sort_index(ascending=False)
#subset for alteast 5 entries
x=x[x>5]
# get the top 20
x=x.iloc[0:20]
#plot
plt.figure(figsize=(16,5))
ax= sns.barplot(x.index, x.values, alpha=0.8)
plt.title("Frequent(more than 5 projects) Price points")
plt.ylabel('# of Occurrences', fontsize=12)
plt.xlabel('Price point', fontsize=12)
plt.show()


# There are some pricey items on the list. 
# 
# Let's explore some of the most costly items requested to hopefully find a useful pattern.
# 
# # Google Expeditions Kit : 9999, 6999, 3999 :
# This seems to be the costliest item under DonorsChoose. Its a set of 30 Virtual Reality glasses for the entire classroom. 
# 
# I would've loved to have this during my schooling days! [Demo](https://support.google.com/edu/expeditions/answer/7375176?hl=en&ref_topic=6334250)
# 
# ![](https://images.bbycastatic.ca/sf/projects/bestbuyforbusiness/education/contents/google-expeditions/assets/featured-kit-size-30.jpg)
# 
# The item comes at three price points (9999,6999,3999).
# 
# There were 34 requests for this item and only two rejections at 9999
# 
# There were 13 requests for the item and 0 rejects at 6999
# 
# 57 requests for this item at 3999 and 5 rejects

# In[ ]:


# subset expedition kit requests
subset=train_merge[train_merge.total_cost==9999]
print("Number of times requested:",len(subset))
subset[subset.project_is_approved==0]


# In[ ]:


# subset expedition kit requests
subset=train_merge[train_merge.total_cost==6999]
print("Number of times requested:",len(subset))
subset[subset.project_is_approved==0]


# In[ ]:


# subset expedition kit requests
subset=train_merge[train_merge.total_cost==3999]
print("Number of times requested:",len(subset))
subset[subset.project_is_approved==1]


# # Engage 2 - Interactive table - 4995.95 :
# 
# ![](https://images.kaplanco.com/images/products/engage2-interactive-table2015.jpg)
# 
# Interactive table for kindergardeners!
# 
# Requested 7 times and no rejects!

# In[ ]:


# subset expedition kit requests
subset=train_merge[train_merge.total_cost==4995.95]
print("Number of times requested:",len(subset))
subset[subset.project_is_approved==1]


# # Apple products:
# 
# Ipad , Ipad mini, Macbook seem to be frequent in the wishlist of teachers.

# In[ ]:



price_points=[1999.99, 1999.96, 1999.95,1999.9]
# subset expedition kit requests
subset=train_merge[train_merge.total_cost.isin(price_points)]
print("Number of times requested:",len(subset))
print("Number of times approved:",len(subset[subset.project_is_approved==1]))
subset[subset.project_is_approved==0].head()


# # Feature Engineering (ELI5 version):
# 
# The competition is more focussed on explainability (ie) understanding why a project gets approved, so that they can pre-approve some applications and pass on some of the difficult ones to Human volunteers.
# 
# Hence, Ive built the model to be compatable with [ELI5](https://www.kaggle.com/lopuhin/eli5-for-mercari), so that we can hopefully understand why the model thinks that certain projects are rejected!
# 
# ## Pre-processing/Cleaning text fields:
# 
# The following steps have been done for pre-processing.
# 
# * Tokenization (Splitting into seperate words )
# * Basic pre-processing ( convert to lower,etc) by Gensim
# * Remove stop words
# * Lemmatization (Converting word to its root form : babies --> baby ; children --> child)
# 
# 

# In[ ]:


# Using cleaning functions from previous work --> https://www.kaggle.com/jagangupta/understanding-the-topic-of-toxicity/notebook
def preprocess_and_clean(text_col):
    """
    Function to build tokenized texts from input comment and the clean them
    Following transformations will be done
    1) Stop words removal from the nltk stopword list
   #commenting out for speed issues 2) Bigram collation (Finding common bigrams and grouping them together using gensim.models.phrases) (Eg: new + york --> new_york )
    3) Lemmatization (Converting word to its root form : babies --> baby ; children --> child)
    """
    
    word_list = gensim.utils.simple_preprocess(text_col, deacc=True)
    #Phrases help us group together bigrams :  new + york --> new_york
    #bigram = gensim.models.Phrases(text_col)
    
    #remove stop words
    clean_words = [w for w in word_list if not w in eng_stopwords]
    #collect bigrams
    #clean_words = bigram[clean_words]
    #Lemmatize
    clean_words=[lem.lemmatize(word, "v") for word in clean_words]
    return(' '.join(clean_words))  

#check clean function
print("Before clean:",train.project_description.iloc[16])
print("After clean:",preprocess_and_clean(train.project_description.iloc[16]))


# In[ ]:


# Null treatment
train.teacher_prefix=train.teacher_prefix.fillna('Unknown')
test.teacher_prefix=test.teacher_prefix.fillna('Unknown')
train.gender=train.gender.fillna('Unknown')
test.gender=test.gender.fillna('Unknown')

print(train.shape)
print(test.shape)
y=train['project_is_approved']
train_id=train['id']
test_id=test['id']
del(train['project_is_approved'])
all_data=pd.concat([train,test],axis=0)
print(all_data.shape)
all_data_merge=pd.merge(left=all_data,right=resources_agg,on='id',how='left')


# In[ ]:


# taking some FE ideas from public kernals
# thanks owl, --> https://www.kaggle.com/the1owl/the-choice-is-yours
# and jmbull --> https://www.kaggle.com/jmbull/xtra-credit-xgb-w-tfidf-feature-stacking


# In[ ]:


from sklearn import *
from tqdm import tqdm
# Label encode some columns
cols = [
    'teacher_id', 
    'teacher_prefix', 
    'school_state', 
    'project_grade_category',
    'project_subject_categories', 
    'project_subject_subcategories',
    'gender']
for c in tqdm(cols):
    le = preprocessing.LabelEncoder()
    le.fit(all_data_merge[c].astype(str))
    all_data_merge[c] = le.transform(all_data_merge[c].astype(str))


# In[ ]:


# Log1p transform price columns
all_data_merge['avg_price_per_item']=np.log1p(all_data_merge['avg_price_per_item'])
all_data_merge['total_cost']=np.log1p(all_data_merge['total_cost'])

# date features
all_data_merge['project_submitted_datetime'] = pd.to_datetime(train['project_submitted_datetime'])
all_data_merge['datetime_dow'] = all_data_merge['project_submitted_datetime'].dt.dayofweek
all_data_merge['datetime_year'] = all_data_merge['project_submitted_datetime'].dt.year
all_data_merge['datetime_month'] = all_data_merge['project_submitted_datetime'].dt.month
all_data_merge['datetime_hour'] = all_data_merge['project_submitted_datetime'].dt.hour
all_data_merge['datetime_day'] = all_data_merge['project_submitted_datetime'].dt.day


# In[ ]:


#process text cols
text_cols=['project_title', 
           'collated_description',
           'project_resource_summary',
           'student_description', 
           'project_description']
for c in tqdm(text_cols):
    all_data_merge[c+'_len']=all_data_merge[c].apply(len)               # get length (ie) letter count
    all_data_merge[c+'_word_count']=all_data_merge[c].apply(lambda x: len(str(x).split())) # get word count
    all_data_merge[c]=all_data_merge[c].apply(preprocess_and_clean)
end_time=time.time()
print("Time till end",end_time-start_time,"s")


# In[ ]:


train_shape=train.shape
test_shape=test.shape
# del(train)
# del(test)
del(train_merge)
del(resources)
del(resources_agg)
del(subset)
del(approvals)
del(rejects)
del(all_data)
gc.collect()


# In[ ]:


############################
# needs debugging
########################
# from sklearn.pipeline import FeatureUnion,TransformerMixin,Pipeline
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.feature_extraction import DictVectorizer
# from sklearn.metrics import auc

# # https://www.kaggle.com/lopuhin/eli5-for-mercari
# # https://github.com/scikit-learn/scikit-learn/issues/2034

# class GetItemTransformer(TransformerMixin):
#     """
#     Custom class to fetch just the column needed from the numpy nd array from pandas.values that is passed to the vectorizer
#     """
#     def __init__(self, field):
#         self.field = field
#     def fit(self, X, y=None):
#         return self
#     def transform(self,X):
#         field_idx = list(all_data_merge.columns).index(self.field)
#         return X[:,field_idx]
    

# vectorizer = FeatureUnion([
#     ('project_title',
#          Pipeline([
#             ('get', GetItemTransformer('project_title')),
#             ('vectorize',CountVectorizer(
#             ngram_range=(1, 2),
#             max_features=2000))
#          ])),
#     ('project_resource_summary',
#          Pipeline([
#             ('get', GetItemTransformer('project_resource_summary')) ,
#             ('vectorize',TfidfVectorizer(
#             ngram_range=(1, 2),
#             max_features=5000))
#          ])),
#     ('student_description',
#          Pipeline([
#             ('get', GetItemTransformer('student_description')) ,
#             ('vectorize',TfidfVectorizer(
#             ngram_range=(1, 2),
#             max_features=20000))
#          ])),
#     ('project_description',
#          Pipeline([
#             ('get', GetItemTransformer('project_description')),
#             ('vectorize',TfidfVectorizer(
#             ngram_range=(1, 2),
#             max_features=20000))
#          ])),
#     ('collated_description',                                                 # using count vect here as this coulmn contains mostly product descr. 
#          Pipeline([
#             ('get', GetItemTransformer('collated_description')),
#             ('vectorize',CountVectorizer(
#             ngram_range=(1, 2),
#             max_features=10000))
#          ]))
# ])
# error due to pipeline not having get feature names!! --> https://github.com/scikit-learn/scikit-learn/issues/6424
# # todo : to add other fields(non-text) into the pipeline itself


# In[ ]:


del(all_data_merge['project_submitted_datetime'])
#https://stackoverflow.com/questions/29815129/pandas-dataframe-to-list-of-dictionaries
all_data_merge_1=all_data_merge.to_dict('records')


# In[ ]:


from sklearn.pipeline import FeatureUnion,TransformerMixin,Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import auc

# https://www.kaggle.com/lopuhin/eli5-for-mercari
# https://github.com/scikit-learn/scikit-learn/issues/2034

vectorizer = FeatureUnion([
        ('project_title',CountVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            preprocessor=lambda x: x['project_title'])),
        ('project_resource_summary',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            preprocessor=lambda x: x['project_resource_summary'])),
        ('student_description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=40000,
            preprocessor=lambda x: x['student_description'])),
        ('project_description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=40000,
            preprocessor=lambda x: x['project_description'])),
        ('collated_description',TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=30000,
            preprocessor=lambda x: x['collated_description'])),
        ('Non_text',DictVectorizer())
    ])

# todo : to add other fields(non-text) into the pipeline itself


# In[ ]:


all_data_vectorized = vectorizer.fit_transform(all_data_merge_1)
# split train and test 
# train_text_data_vectorizer=vectorizer.fit_transform(all_data_merge.iloc[:train_shape[0]])
# test_text_data_vectorizer=vectorizer.fit_transform(all_data_merge.iloc[train_shape[0]:])

from scipy.sparse import csr_matrix, hstack
final_dataset=all_data_vectorized.tocsr()
end_time=time.time()
print("total time till Sparse mat creation",end_time-start_time,"s")


# In[ ]:


# older version
# all_text_data_vectorized = vectorizer.fit_transform(all_data_merge.values)
# train_x_only_text=all_text_data_vectorized[0:train_shape[0]]
# test_x_only_text=all_text_data_vectorized[train_shape[0]:]
# all_cols=all_data_merge.columns.values
# non_text_cols = [col for col in all_cols if col not in text_cols]
# non_text_cols.remove('id')
# non_text_cols.remove('project_submitted_datetime')
# from scipy.sparse import csr_matrix, hstack
# all_non_text_data = csr_matrix(all_data_merge[non_text_cols].values)
# final_dataset=hstack((all_text_data_vectorized,all_non_text_data)).tocsr()
# end_time=time.time()
# print("total time till Sparse mat creation",end_time-start_time,"s")


# In[ ]:


train_x=final_dataset[0:train_shape[0]]
test_x=final_dataset[train_shape[0]:]


# In[ ]:


# del final_dataset,all_text_data_vectorized,all_non_text_data
# gc.collect()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(train_x, y, test_size=0.33, random_state=2018)
# Using LGBM params from https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter/code
params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 16,
        'num_leaves': 31,
        'learning_rate': 0.25,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 1,
        'num_threads': 4,
        'lambda_l2': 1,
        'min_gain_to_split': 0,
        'seed':1234
}  


# In[ ]:


import lightgbm as lgb

model = lgb.train(
        params,
        lgb.Dataset(X_train, y_train),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose_eval=25)


# In[ ]:


from sklearn.metrics import roc_auc_score
valid_preds = model.predict(X_valid, num_iteration=model.best_iteration)
test_preds = model.predict(test_x, num_iteration=model.best_iteration)
auc = roc_auc_score(y_valid, valid_preds)
print('AUC:',auc)
plt.show()


# In[ ]:


end_time=time.time()
print("total time till LGB model",end_time-start_time,"s")


# In[ ]:


import xgboost as xgb
xgb_params = {'eta': 0.2, 
                  'max_depth': 5, 
                  'subsample': 0.8, 
                  'colsample_bytree': 0.8, 
                  'objective': 'binary:logistic', 
                  'eval_metric': 'auc', 
                  'seed': 1234
                 }
# d_train = xgb.DMatrix(X_train, y_train)
# d_valid = xgb.DMatrix(X_valid, y_valid)
# d_test = xgb.DMatrix(test_x)
X_train, X_valid, y_train, y_valid = train_test_split(train_x, y, test_size=0.33, random_state=2018)
#for eli5
d_train = xgb.DMatrix(X_train, y_train)
d_valid = xgb.DMatrix(X_valid, y_valid)
d_test = xgb.DMatrix(test_x)


# In[ ]:


watchlist = [(d_train, 'train'), (d_valid, 'valid')]
model_xgb = xgb.train(xgb_params, d_train, 500, watchlist, verbose_eval=50, early_stopping_rounds=20)


# In[ ]:


xgb_pred_test = model_xgb.predict(d_test)
xgb_pred_valid = model_xgb.predict(d_valid)
auc = roc_auc_score(y_valid, xgb_pred_valid)
print('AUC:',auc)


# In[ ]:


from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_valid, xgb_pred_valid)
roc_auc = metrics.auc(fpr, tpr)
fpr_1,tpr_1,thresholds_1=roc_curve(y_valid,valid_preds)
roc_auc_1 = metrics.auc(fpr_1, tpr_1)
plt.figure(figsize=(8,6))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'XGBoost-AUC = %0.2f' % roc_auc)
plt.plot(fpr_1, tpr_1, 'g', label = 'LGBM-AUC = %0.2f' % roc_auc_1)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# end_time=time.time()
# print("total time till XBG model",end_time-start_time,"s")


# In[ ]:


xgb_pred_train = model_xgb.predict(d_train)
import eli5


# In[ ]:


# eli5.explain_weights_lgb(model_xgb, vec=vectorizer)     # out of bounds error
# text_features=vectorizer.get_feature_names()
# other_features=non_text_cols
# all_features=text_features+non_text_cols
# eli5.explain_weights_xgboost(model_xgb, feature_names=all_features)          


# In[ ]:


eli5.show_weights(model_xgb,vec=vectorizer)    


# Lets use ELI5 to understand why a particular entry was selected or rejected

# In[ ]:


# random entry
print("Project is Approved?:Actual",y[100])
print("Project is Approved?:Predicted prob:",xgb_pred_train[100])
display(eli5.show_prediction(model_xgb, doc=all_data_merge_1[100], vec=vectorizer,show_feature_values=True,top=20))    


# In[ ]:


# random entry
print("Project is Approved?:Actual",y[500])
print("Project is Approved?:Predicted prob:",xgb_pred_train[500])
display(eli5.show_prediction(model_xgb, doc=all_data_merge_1[500], vec=vectorizer,show_feature_values=True,top=20))    


# ## Some interesting observations here.
# The fact that some words(set,ipad,materials) are missing contributes to the model thinking that the project is approved!!  I am not sure, if I am interpreting this correctly here though.
# 
# 
# Also,The sentences are cleaned(ie) Stop words removed, Lemmatized.  Need to think of a way to perform the cleaning in the vectorizer itself, to display the original sentence. Do let me know in the comments section if you have any ideas :)
# 
# 
# 

# In[ ]:


from IPython.display import display
no_missing = lambda feature_name, feature_value: not np.isnan(feature_value)
for i in range(5):
    print("Project is Approved?:Actual",y[i])
    print("Project is Approved?:Predicted prob:",xgb_pred_train[i])
    display(eli5.show_prediction(model_xgb, doc=all_data_merge_1[i], vec=vectorizer,show_feature_values=True,top=30,feature_filter=no_missing))  


# In[ ]:


final_preds=0.4*xgb_pred_test+0.6*test_preds


# In[ ]:


# Making submission
x_preds = pd.DataFrame(final_preds)
x_preds.columns = ['project_is_approved']
sub_id=sample_sub['id']
submission = pd.concat([sub_id, x_preds], axis=1)
submission.to_csv('lgbm_xgb_blend.csv', index=False)


# # Footnotes:
# To be done:
# * Tune XGB,LGBM better
# * Explore other models
# * Interpret ELi5 output and make changes to the model
# * Explore more on State and Time variables
# * Explore interactions
# * Explore Topic Modeling, text clustering
# * Explore vector based features
# * Add more functionality to text stats, wordcloud functions
# * Add confidence interval to the plotting function
# * Think of a way to perform cleaning in the vectorizer itself , so that ELI5 displays the original sentence instead of the cleaned one
# 
# ### To be continued....
# 
# 
# ## Do leave an upvote if you liked the content :) 
# 

# In[ ]:


# To be continued....

