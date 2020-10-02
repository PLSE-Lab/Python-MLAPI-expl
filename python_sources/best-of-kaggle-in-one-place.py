#!/usr/bin/env python
# coding: utf-8

# ## Best of Kaggle in one place 

# <font color='purple'> <B> Kaggle is a sea of Data Science and is growing exponentially. For Kagglers, specially newbies, finding best of Kaggle items is like finding gems from ocean. I am trying to put best of Kaggle materials at one place through this kernel. <B/></font>

# ![](https://media.giphy.com/media/26ufdipQqU2lhNA4g/giphy.gif)
# <div align="center"><font size="1">Source:giphy.com</font></div> 

# In[ ]:


# imports
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
# fetch datasets
competitions = pd.read_csv('../input/meta-kaggle/Competitions.csv')
datasets = pd.read_csv('../input/meta-kaggle/Datasets.csv')
dataset_version = pd.read_csv('../input/meta-kaggle/DatasetVersions.csv')
users = pd.read_csv('../input/meta-kaggle/Users.csv')
kernels = pd.read_csv("../input/meta-kaggle/Kernels.csv")
kernel_versions = pd.read_csv('../input/meta-kaggle/KernelVersions.csv')
forum_topics = pd.read_csv("../input/meta-kaggle/ForumTopics.csv")
import datetime
 

# for clickable url
def make_clickable_both(val): 
    title, competition_url = val.split('#')
    return f'<a href="{competition_url}">{title}</a>'


# In[ ]:


print(f'Note: This Kernal is executed on meatadata avaialble at : {datetime.datetime.now()}')


# ## Best of Kaggle: Kernels

# In[ ]:


print (f'Total Number of Kernels Created in Kaggle  :{ kernels.shape[0]}')


# ### Top 25 Most Voted Kaggle Kernels

# In[ ]:


df = kernels.sort_values(by ='TotalVotes' , 
                                     axis=0, ascending=False, inplace=False, 
                                     kind='quicksort', na_position='last')[:25]

df = pd.merge(df, kernel_versions, how='left', left_on=['CurrentKernelVersionId'],
               right_on=['Id'])
 
df= df[['AuthorUserId_x','CurrentUrlSlug','Title', 'TotalVotes_x']]

df = pd.merge(df, users, how='left', left_on=['AuthorUserId_x'], right_on=['Id'])
df['keranel_url'] ='https://www.kaggle.com/'+df['UserName'] +'/'+df['CurrentUrlSlug']
df['Kernel'] = df['Title'] + '#' + df['keranel_url']
df= df.filter(['Kernel','TotalVotes_x'], axis=1)
df.rename(columns={'TotalVotes_x':'TotalVotes'}, inplace=True)
df.reset_index(drop=True).style.format({'Kernel': make_clickable_both}).hide_index()


# ### Top 25 Most Viewed Kaggle Kernels  

# In[ ]:


df = kernels.sort_values(by ='TotalViews' , 
                                     axis=0, ascending=False, inplace=False, 
                                     kind='quicksort', na_position='last')[:25]

df = pd.merge(df, kernel_versions, how='left', left_on=['CurrentKernelVersionId'],
               right_on=['Id'])
 
df= df[['AuthorUserId_x','CurrentUrlSlug','Title', 'TotalViews']]

df = pd.merge(df, users, how='left', left_on=['AuthorUserId_x'], right_on=['Id'])
df['keranel_url'] ='https://www.kaggle.com/'+df['UserName'] +'/'+df['CurrentUrlSlug']
df['Kernel'] = df['Title'] + '#' + df['keranel_url']
df= df.filter(['Kernel','TotalViews'], axis=1)
#df.rename(columns={'TotalVotes_x':'TotalVotes'}, inplace=True)
df.reset_index(drop=True).style.format({'Kernel': make_clickable_both}).hide_index()


# ### Top 25 Most Commented Kaggle Kernels

# In[ ]:


df = kernels.sort_values(by ='TotalComments' , 
                                     axis=0, ascending=False, inplace=False, 
                                     kind='quicksort', na_position='last')[:25]

df = pd.merge(df, kernel_versions, how='left', left_on=['CurrentKernelVersionId'],
               right_on=['Id'])
 
df= df[['AuthorUserId_x','CurrentUrlSlug','Title', 'TotalComments']]

df = pd.merge(df, users, how='left', left_on=['AuthorUserId_x'], right_on=['Id'])
df['keranel_url'] ='https://www.kaggle.com/'+df['UserName'] +'/'+df['CurrentUrlSlug']
df['Kernel'] = df['Title'] + '#' + df['keranel_url']
df= df.filter(['Kernel','TotalComments'], axis=1)
 
df.reset_index(drop=True).style.format({'Kernel': make_clickable_both}).hide_index()


# ## Best of Kaggle: Competitions

# In[ ]:


print(f'Total Number of Hosted Competitions in Kaggle :{competitions.shape[0]}')


# In[ ]:


ax = competitions['HostSegmentTitle'].value_counts().plot(kind='barh', figsize=(10,7),
                                                 color="slateblue", fontsize=13);
ax.set_alpha(0.8)
ax.set_title(" Competitions in Kaggle", fontsize=18)
ax.set_xlabel("Number of Competitions", fontsize=18);
ax.set_xticks([0, 500, 1000, 1500, 2000 ])
for i in ax.patches:
    ax.text(i.get_width()+.1, i.get_y()+.31,             str(round((i.get_width()), 2)), fontsize=15, color='dimgrey')

# invert for largest on top 
ax.invert_yaxis()


# ### Top 25 Kaggle Competitions with most number of submissions

# In[ ]:


df = competitions.sort_values(by ='TotalSubmissions' , 
                                     axis=0, ascending=False, inplace=False, 
                                     kind='quicksort', na_position='last')[:25]
df['competition_url'] ='https://www.kaggle.com/c/'+df['Slug']
df['Competition'] = df['Title'] + '#' + df['competition_url']
df= df.filter(['Competition','TotalSubmissions', 'HostSegmentTitle'], axis=1)
df.rename(columns={'TotalSubmissions':'Total Submissions','HostSegmentTitle':'Competition Type'}, inplace=True)
df.reset_index(drop=True).style.format({'Competition': make_clickable_both}).hide_index()


# ### Top 25 Kaggle Competitions with most number of participated teams

# In[ ]:


df = competitions.sort_values(by ='TotalTeams' , 
                                     axis=0, ascending=False, inplace=False, 
                                     kind='quicksort', na_position='last')[:25]
df['competition_url'] ='https://www.kaggle.com/c/'+df['Slug']
df['Competition'] = df['Title'] + '#' + df['competition_url']
df= df.filter(['Competition','TotalTeams', 'HostSegmentTitle'], axis=1)
df.rename(columns={'TotalTeams':'Total Teams','HostSegmentTitle':'Competition Type'}, inplace=True)
df.reset_index(drop=True).style.format({'Competition': make_clickable_both}).hide_index()


# ### Top 25 Kaggle competitions  with most number of prizes

# In[ ]:


df = competitions.sort_values(by ='NumPrizes' , 
                                     axis=0, ascending=False, inplace=False, 
                                     kind='quicksort', na_position='last')[:25]
df['competition_url'] ='https://www.kaggle.com/c/'+df['Slug']
df['Competition'] = df['Title'] + '#' + df['competition_url']
df= df.filter(['Competition','NumPrizes', 'HostSegmentTitle'], axis=1)
df.rename(columns={'HostSegmentTitle':'CompetitionType'}, inplace=True)
df.reset_index(drop=True).style.format({'Competition': make_clickable_both}).hide_index()


# ## Best of Kaggle: DataSets

# In[ ]:


print (f'Total Number of Datasets Uploaded on Kaggle   :{ datasets.shape[0]}')


# ### Top 25 Most Downloaded Kaggle Datasets  

# In[ ]:


df = datasets.sort_values(by ='TotalDownloads' , 
                                     axis=0, ascending=False, inplace=False, 
                                     kind='quicksort', na_position='last')[:25]
df = df[['Id','CreatorUserId','TotalDownloads']]
df2 = dataset_version[['DatasetId','Title','Slug']].drop_duplicates(subset=['DatasetId','Title','Slug'])

df = pd.merge(df, df2, how='left', left_on=['Id'], right_on=['DatasetId']).drop_duplicates(subset=['Id','CreatorUserId','TotalDownloads'])
df = pd.merge(df, users, how='left', left_on=['CreatorUserId'], right_on=['Id'])

# special case
df['UserName'].replace(
    to_replace=['promptcloud'],
    value='PromptCloudHQ',
    inplace=True
)

df['dataset_url'] ='https://www.kaggle.com/'+df['UserName'] +'/'+df['Slug']
df['Dataset'] = df['Title'] + '#' + df['dataset_url']
df= df.filter(['Dataset','TotalDownloads'], axis=1)
df.reset_index(drop=True).style.format({'Dataset': make_clickable_both}).hide_index()


# ### Top 25 Most Viewed Kaggle Datasets  

# In[ ]:


df = datasets.sort_values(by ='TotalViews' , 
                                     axis=0, ascending=False, inplace=False, 
                                     kind='quicksort', na_position='last')[:25]
df = df[['Id','CreatorUserId','TotalViews']]
df2 = dataset_version[['DatasetId','Title','Slug']].drop_duplicates(subset=['DatasetId','Title','Slug'])

df = pd.merge(df, df2, how='left', left_on=['Id'], right_on=['DatasetId']).drop_duplicates(subset=['Id','CreatorUserId','TotalViews'])
df = pd.merge(df, users, how='left', left_on=['CreatorUserId'], right_on=['Id'])

# special case
df['UserName'].replace(
    to_replace=['promptcloud'],
    value='PromptCloudHQ',
    inplace=True
)

df['dataset_url'] ='https://www.kaggle.com/'+df['UserName'] +'/'+df['Slug']
df['Dataset'] = df['Title'] + '#' + df['dataset_url']
df= df.filter(['Dataset','TotalViews'], axis=1)
df.reset_index(drop=True).style.format({'Dataset': make_clickable_both}).hide_index()


# ### Top 25 Most Voted Kaggle Datasets  

# In[ ]:


df = datasets.sort_values(by ='TotalVotes' , 
                                     axis=0, ascending=False, inplace=False, 
                                     kind='quicksort', na_position='last')[:25]
df = df[['Id','CreatorUserId','TotalVotes']]
df2 = dataset_version[['DatasetId','Title','Slug']]

df = pd.merge(df, df2, how='left', left_on=['Id'], right_on=['DatasetId'])
df= df.drop_duplicates(subset=['Title','Slug'])
df = pd.merge(df, users, how='left', left_on=['CreatorUserId'], right_on=['Id'])
df= df.drop_duplicates(subset=['CreatorUserId','DatasetId'])


df['UserName'].replace(
    to_replace=['promptcloud'],
    value='PromptCloudHQ',
    inplace=True
)

df['dataset_url'] ='https://www.kaggle.com/'+df['UserName'] +'/'+df['Slug']
df['Dataset'] = df['Title'] + '#' + df['dataset_url']
df= df.filter(['Dataset','TotalVotes'], axis=1)
df.reset_index(drop=True).style.format({'Dataset': make_clickable_both}).hide_index()


# ## Best of Kaggle: Users

# In[ ]:


performancetier = {0 : "Novice",1: "Contributor", 2: "Expert", 3: "Master", 4: "Grandmaster", 5 :"Kaggle Team Member" }
users['PerformanceTier'] = users['PerformanceTier'].map(performancetier)

print (f'Total Number of Users Registered in Kaggle   :{ users.shape[0]}')


# In[ ]:


ax = users['PerformanceTier'].value_counts().plot(kind='barh', figsize=(10,7),
                                                 color="slateblue", fontsize=13);
ax.set_alpha(0.8)
ax.set_title(" Status of Kaggle Users  ", fontsize=18)
ax.set_xlabel("Number of Users ", fontsize=18);
#ax.set_xticks([0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000 ])


# set individual bar lables using above list
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width()+.1, i.get_y()+.31,             str(round((i.get_width()), 2)), fontsize=15, color='dimgrey')


# #####  Acknowledgment : 
# 
# I thanks Kaggle team for sharing this [Meta Kaggle](https://www.kaggle.com/kaggle/meta-kaggle) dataset and Kaggle stalwarts specially  [shivamb](https://www.kaggle.com/shivamb), [srk](https://www.kaggle.com/sudalairajkumar), [kabure](https://www.kaggle.com/kabure) for writing cool kernels on this dataset and showing how to wonderfully use this metadata. 
# 
# 

# ### Thanks for reading. If you find the kernel usefull please do <font color='purple'> Upvote</font>.
# 
