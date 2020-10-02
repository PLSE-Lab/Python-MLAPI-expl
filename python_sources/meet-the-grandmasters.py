#!/usr/bin/env python
# coding: utf-8

# # Meet The Grandmasters

# [Meta Kaggle](https://kaggle.com/kaggle/meta-kaggle) is "Kaggle's public data on competitions, users, submission scores, and kernels". After I wrote a notebook titled [Meet The Kaggle Team](https://kaggle.com/sahidvelji/meet-the-kaggle-team), I realized I could reuse much of my code to create one for grandmasters as well.

# <a id="table-of-contents"></a>

# ## Table of contents
# 1. [The Grandmasters](#grandmasters)
# 1. [Where are Grandmasters located?](#location)
# 1. [Followers](#followers)
# 1. [Discussions](#discussions)
# 1. [Competitions](#comps)
# 1. [Kernels](#kernels)
# 1. [Datasets](#datasets)

# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[ ]:


import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import json

import seaborn as sns
import matplotlib.pyplot as plt

from geopy.geocoders import Photon
from geopy.extra.rate_limiter import RateLimiter

from geopandas.tools import geocode
import folium
from folium import Marker
from folium.plugins import MarkerCluster

from IPython.display import display, HTML
        
KAGGLE_BASE_URL = "https://kaggle.com/"
TOP_N = 10


# In[ ]:


# # Datasets
datasets = pd.read_csv('/kaggle/input/meta-kaggle/Datasets.csv', 
                       parse_dates=['CreationDate', 'ReviewDate', 'FeatureDate',
                                    'LastActivityDate'], infer_datetime_format=True
                      )
dataset_versions = pd.read_csv('/kaggle/input/meta-kaggle/DatasetVersions.csv', 
                               parse_dates=['CreationDate'], 
                               infer_datetime_format=True
                              )
# dataset_votes = pd.read_csv('/kaggle/input/meta-kaggle/DatasetVotes.csv', 
#                             parse_dates=['VoteDate'], infer_datetime_format=True
#                            )
# dataset_tags = pd.read_csv('/kaggle/input/meta-kaggle/DatasetTags.csv')
# data_sources = pd.read_csv('/kaggle/input/meta-kaggle/Datasources.csv', 
#                            parse_dates=['CreationDate'], infer_datetime_format=True
#                           )

# # Kernels
kernels = pd.read_csv('/kaggle/input/meta-kaggle/Kernels.csv', 
                      parse_dates=['CreationDate', 'EvaluationDate',
                                   'MadePublicDate', 'MedalAwardDate'],
                      infer_datetime_format=True
                     )
kernel_versions = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersions.csv', 
                              parse_dates=['CreationDate', 'EvaluationDate'],
                              infer_datetime_format=True
                             )
# kernel_votes = pd.read_csv('/kaggle/input/meta-kaggle/KernelVotes.csv', 
#                            parse_dates=['VoteDate'], infer_datetime_format=True
#                           )
# kernel_version_comp = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionCompetitionSources.csv')
# kernel_version_out = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionOutputFiles.csv')
# kernel_version_datasets = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionDatasetSources.csv')
# kernel_langs = pd.read_csv('/kaggle/input/meta-kaggle/KernelLanguages.csv')
# kernel_version_kernel_sources = pd.read_csv('/kaggle/input/meta-kaggle/KernelVersionKernelSources.csv')
# kernel_tags = pd.read_csv('/kaggle/input/meta-kaggle/KernelTags.csv')

# # Forums
messages = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessages.csv', 
                       parse_dates=['PostDate', 'MedalAwardDate'], 
                       infer_datetime_format=True
                      )
topics = pd.read_csv('/kaggle/input/meta-kaggle/ForumTopics.csv', 
                     parse_dates=['CreationDate', 'LastCommentDate'], 
                     infer_datetime_format=True
                    )
# forums = pd.read_csv('/kaggle/input/meta-kaggle/Forums.csv')
# forum_votes = pd.read_csv('/kaggle/input/meta-kaggle/ForumMessageVotes.csv', 
#                           parse_dates=['VoteDate'], infer_datetime_format=True
#                          )

# # Competitions
# teams = pd.read_csv('/kaggle/input/meta-kaggle/Teams.csv', 
#                     parse_dates=['ScoreFirstSubmittedDate', 'LastSubmissionDate',
#                                  'MedalAwardDate'], 
#                     infer_datetime_format=True
#                    )
# comps = pd.read_csv('/kaggle/input/meta-kaggle/Competitions.csv', 
#                     parse_dates=['EnabledDate', 'DeadlineDate', 
#                                  'ProhibitNewEntrantsDeadlineDate', 'TeamMergerDeadlineDate', 
#                                  'TeamModelDeadlineDate', 'ModelSubmissionDeadlineDate'], 
#                     infer_datetime_format=True
#                    )
# comp_tags = pd.read_csv('/kaggle/input/meta-kaggle/CompetitionTags.csv')
# team_membership = pd.read_csv('/kaggle/input/meta-kaggle/TeamMemberships.csv', 
#                               parse_dates=['RequestDate'], 
#                               infer_datetime_format=True
#                              )
# submissions = pd.read_csv('/kaggle/input/meta-kaggle/Submissions.csv', 
#                           parse_dates=['SubmissionDate', 'ScoreDate'], 
#                           infer_datetime_format=True
#                          )

# # Users
users = pd.read_csv('/kaggle/input/meta-kaggle/Users.csv',
                    parse_dates=['RegisterDate'], 
                    infer_datetime_format=True
                   )
user_followers = pd.read_csv('/kaggle/input/meta-kaggle/UserFollowers.csv', 
                             parse_dates=['CreationDate'], 
                             infer_datetime_format=True
                            )
# user_orgs = pd.read_csv('/kaggle/input/meta-kaggle/UserOrganizations.csv', 
#                         parse_dates=['JoinDate'], 
#                         infer_datetime_format=True
#                        )
# user_ach = pd.read_csv('/kaggle/input/meta-kaggle/UserAchievements.csv', 
#                        parse_dates=['TierAchievementDate'], 
#                        infer_datetime_format=True
#                       )

# # Misc
# tags = pd.read_csv('/kaggle/input/meta-kaggle/Tags.csv')
# orgs = pd.read_csv('/kaggle/input/meta-kaggle/Organizations.csv', 
#                    parse_dates=['CreationDate'], infer_datetime_format=True
#                   )


# In[ ]:


gms = (users[users['PerformanceTier'] == 4]
       .copy()
       .reset_index(drop=True)
       .drop(columns=['PerformanceTier'])
      )


# In[ ]:


string = f"There are currently {gms.shape[0]} Grandmasters."
display(HTML(string))


# <a id="grandmasters"></a>
# [Return to table of contents](#table-of-contents)

# # The Grandmasters
# The images in the table below link to the grandmaster's profile. A missing location or occupation in the table indicates that the user has not filled in this information in their profile.

# In[ ]:


def display_html(df, cols=None, num_rows=0):
    """Display columns cols and num_rows rows of the data 
    frame df in HTML.
    """
    
    if num_rows != 0:
        df_to_display = df.head(num_rows)
    else:
        df_to_display = df
    
    df_html = df_to_display.to_html(columns=cols, index=False, na_rep='',
                              escape=False, render_links=True)
    display(HTML(df_html))


# In[ ]:


for index in gms.index:
    time.sleep(2)
    row = gms.iloc[index]
    
    username = row.UserName
    profile_url = f'{KAGGLE_BASE_URL}{username}'
    displayname = row.DisplayName
    
    result = requests.get(profile_url)
    src = result.content
    soup = BeautifulSoup(src, 'html.parser').find_all("div", id="site-body")[0].find("script")
    
    user_info = re.search('Kaggle.State.push\(({.*})', str(soup)).group(1)
    user_dict = json.loads(user_info)
    
    city = user_dict['city']
    region = user_dict['region']
    country = user_dict['country']
    avatar_url = user_dict['userAvatarUrl']
    occupation = user_dict['occupation']
    organization = user_dict['organization']
    num_followers = user_dict['followers']['count']
    num_following = user_dict['following']['count']
    num_posts = user_dict['discussionsSummary']['totalResults']
    num_datasets = user_dict['datasetsSummary']['totalResults']
    num_kernels = user_dict['scriptsSummary']['totalResults']
    num_comps = user_dict['competitionsSummary']['totalResults']
    
    
    gms.loc[index, 'Image'] = f'<a href="{profile_url}" target="_blank" title="{displayname}"><img src="{avatar_url}" width="100" height="100"></a>'
    
    if city and region and country:
        gms.loc[index, 'Location'] = f'{city}, {region}, {country}'
        
    if occupation and organization:
        gms.loc[index, 'Occupation'] = f'{occupation} at {organization}'
    elif organization:
        gms.loc[index, 'Occupation'] = organization
    elif occupation:
        gms.loc[index, 'Occupation'] = occupation
    
    gms.loc[index, 'NumFollowers'] = num_followers
    gms.loc[index, 'NumFollowing'] = num_following
    gms.loc[index, 'NumPosts'] = num_posts
    gms.loc[index, 'NumDatasets'] = num_datasets
    gms.loc[index, 'NumKernels'] = num_kernels
    gms.loc[index, 'NumCompetitions'] = num_comps

gms = gms.convert_dtypes()
display_html(gms, cols=['UserName', 'DisplayName', 'RegisterDate', 'Image', 'Location', 'Occupation'])


# <a id="location"></a>
# [Return to table of contents](#table-of-contents)

# # Where are grandmasters located?

# In[ ]:


geolocator = Photon(timeout=15)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=3, max_retries=3, error_wait_seconds=30)
m = folium.Map(location=[0, 30], tiles='OpenStreetMap', zoom_start=1)
mc = MarkerCluster()
locations = {}
for index in gms[gms.Location.notna()].index:
    row = gms.iloc[index]
    displayname = row.DisplayName
    location = row.Location
    popup = '{}; {}'.format(displayname, location)
    tooltip = '{}; {}'.format(displayname, location)
    
    if location in locations:
        lat, long = locations[location]
    else:
        point = geocode(location)
        lat, long = point.latitude, point.longitude
        locations[location] = [lat, long]
   
    mc.add_child(Marker([lat, long], popup=popup, tooltip=tooltip))
    
m.add_child(mc)


# <a id="followers"></a>
# [Return to table of contents](#table-of-contents)

# # Followers
# 
# ### Who has the most followers?

# In[ ]:


followers = gms.nlargest(TOP_N, 'NumFollowers').copy()
display_html(followers, cols=['UserName', 'DisplayName', 'Image', 
                              'Occupation', 'NumFollowers', 'NumFollowing'])


# ### Who follows other users the most?

# In[ ]:


following = gms.nlargest(TOP_N, 'NumFollowing').copy()
display_html(following, cols=['UserName', 'DisplayName', 'Image', 
                              'Occupation', 'NumFollowing', 'NumFollowers'])


# Next, we will break down followers and follows by tiers.
# ### Who do the grandmasters follow?

# In[ ]:


gms_following_tier = (user_followers[['UserId', 'FollowingUserId']]
                      .merge(gms['Id'], left_on='UserId', right_on='Id')
                      .drop(columns='Id')
                      .merge(users, left_on='FollowingUserId', right_on='Id')
                      .drop_duplicates('Id')
                      .groupby('PerformanceTier')
                      .size()
                     )
plt.figure(figsize=(8, 5))
plt.title('Number of follows by grandmasters, by performance tier')
ax = sns.barplot(gms_following_tier.index, gms_following_tier.values, color='steelblue')


# Of all users followed by at least one grandmaster, most are experts. We'll divide the counts by the number of users in that tier and visualize the proportion of users in a tier that are followed by at least one grandmaster.

# In[ ]:


tier_counts = users['PerformanceTier'].value_counts()
tier_counts


# In[ ]:


gms_following_tier_prop = gms_following_tier / tier_counts
plt.figure(figsize=(8, 5))
plt.title('Proportion of users followed by at least one grandmaster, by performance tier')
ax = sns.barplot(gms_following_tier_prop.index, gms_following_tier_prop.values, color='steelblue')


# Most grandmasters and Kaggle team members are followed by at least one grandmaster.

# ### Who follows the grandmasters?

# In[ ]:


gms_followers_tier = (pd.merge(user_followers[['UserId', 'FollowingUserId']], gms['Id'], left_on='FollowingUserId', right_on='Id')
                      .drop(columns='Id')
                      .merge(users, left_on='UserId', right_on='Id')
                      .drop_duplicates('Id')
                      .groupby('PerformanceTier')
                      .size()
                     )
plt.figure(figsize=(8, 5))
plt.title('Number of users that follow at least one grandmaster, by performance tier')
ax = sns.barplot(gms_followers_tier.index, gms_followers_tier.values, color='SeaGreen')


# As expected, most users that follow the grandmasters are novices. That's not surprising since most users are novices. Similar to the above, we'll visualize the proportion of users that follow at least one grandmaster, by performance tier.

# In[ ]:


gms_followers_tier_prop = gms_followers_tier / tier_counts
plt.figure(figsize=(8, 5))
plt.title('Proportion of users that follow at least one grandmaster, by performance tier')
ax = sns.barplot(gms_followers_tier_prop.index, gms_followers_tier_prop.values, color='SeaGreen')


# About half of all grandmasters follow at least one grandmaster.

# <a id="discussions"></a>
# [Return to table of contents](#table-of-contents)

# # Discussions
# ### Number of posts by grandmasters
# CPMP dominates in terms of post count.

# In[ ]:


gms_posts = gms[['UserName', 'DisplayName', 'Image', 'Occupation', 'NumPosts']].nlargest(TOP_N, 'NumPosts').copy()
display_html(gms_posts)


# ### Popular topics by grandmasters
# First place solutions to competitions can become very popular.

# In[ ]:


forum_topics = (topics[topics.KernelId.isna()]
                .merge(messages[['Id', 'PostUserId']], left_on='FirstForumMessageId', right_on='Id')
                .merge(gms, left_on='PostUserId', right_on='Id')
                .nlargest(TOP_N, 'Score')
               )

display_html(forum_topics,
             cols=['UserName', 'DisplayName', 'Image', 'Occupation', 'Score', 'Title']
            )


# <a id="comps"></a>
# [Return to table of contents](#table-of-contents)

# # Competitions
# ### Number of competitions

# In[ ]:


gms_comps = gms[['UserName', 'DisplayName', 'Image', 'Occupation', 'NumCompetitions']].nlargest(TOP_N, 'NumCompetitions').copy()
display_html(gms_comps)


# <a id="kernels"></a>
# [Return to table of contents](#table-of-contents)

# # Kernels
# ### Number of kernels

# In[ ]:


gms_kernels = gms[['UserName', 'DisplayName', 'Image', 'Occupation', 'NumKernels']].nlargest(TOP_N, 'NumKernels').copy()
display_html(gms_kernels)


# ### Popular kernels by grandmasters

# In[ ]:


gms_kernels = (pd.merge(gms[['Id', 'UserName', 'DisplayName', 'Image']],
                        kernels[['Id', 'AuthorUserId', 'CurrentKernelVersionId', 'TotalVotes', 'CurrentUrlSlug']],
                        left_on='Id', right_on='AuthorUserId')
               .nlargest(TOP_N, 'TotalVotes')
               .merge(kernel_versions[['Title', 'Id']], left_on='CurrentKernelVersionId', right_on='Id')
              )
gms_kernels['url'] = KAGGLE_BASE_URL + gms_kernels.UserName + '/' + gms_kernels.CurrentUrlSlug
display_html(gms_kernels, cols=['UserName', 'DisplayName', 'Image', 'TotalVotes', 'Title', 'url'])


# <a id="datasets"></a>
# [Return to table of contents](#table-of-contents)

# # Datasets
# ### Number of datasets

# In[ ]:


gms_data = gms[['UserName', 'DisplayName', 'Image', 'Occupation', 'NumDatasets']].nlargest(TOP_N, 'NumDatasets').copy()
display_html(gms_data)


# ### Popular datasets by grandmasters

# In[ ]:


gms_datasets = (datasets[datasets.OwnerUserId.notna() & datasets.OwnerUserId.eq(datasets.CreatorUserId)]
                .merge(gms[['Id', 'UserName', 'DisplayName', 'Image']], left_on='CreatorUserId', right_on='Id')
                .nlargest(TOP_N, 'TotalVotes')
                .merge(dataset_versions[['Id', 'Title', 'Slug']], left_on='CurrentDatasetVersionId', right_on='Id')
               )
gms_datasets['url'] = KAGGLE_BASE_URL + gms_datasets.UserName + '/' + gms_datasets.Slug
display_html(gms_datasets, cols=['UserName', 'DisplayName', 'Image', 'TotalVotes', 'Title', 'url'])

