#!/usr/bin/env python
# coding: utf-8

# # Meet The Kaggle Team

# [Meta Kaggle](https://kaggle.com/kaggle/meta-kaggle) is "Kaggle's public data on competitions, users, submission scores, and kernels". There is actually a "Meet Our Team" page on Kaggle already: https://kaggle.com/about/team. However, not every Kaggle team member is listed. Also, the Meta Kaggle dataset allows us to go into a bit more detail.

# <a id="table-of-contents"></a>

# ## Table of contents
# 1. [The Kaggle team](#the-kaggle-team)
# 1. [Where are Kaggle team members located?](#location)
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

from geopandas.tools import geocode
import folium
from folium import Marker
from folium.plugins import MarkerCluster

from IPython.display import display, HTML
        
KAGGLE_BASE_URL = "https://kaggle.com/"
TOP_N = 10
REQUEST_DELAY = 2


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


kaggle_team = (users[users['PerformanceTier'].eq(5)]
               .copy()
               .reset_index(drop=True)
               .drop(columns='PerformanceTier')
              )


# In[ ]:


string = f"There are currently {kaggle_team.shape[0]} employees on the Kaggle team."
display(HTML(string))


# <a id="the-kaggle-team"></a>
# [Return to table of contents](#table-of-contents)

# # The Kaggle Team
# The images in the table below link to the Kaggle team member's profile. A missing location or occupation in the table indicates that the Kaggle team member has not filled in this information in their profile.

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


for index in kaggle_team.index:
    time.sleep(REQUEST_DELAY)
    row = kaggle_team.iloc[index]
    
    username = row.UserName
    profile_url = f'{KAGGLE_BASE_URL}{username}'
    displayname = row.DisplayName
    
    result = requests.get(profile_url)
    src = result.text
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
    
    
    kaggle_team.loc[index, 'Image'] = f'<a href="{profile_url}" target="_blank" title="{displayname}"><img src="{avatar_url}" width="100" height="100"></a>'
    
    if city and region and country:
        kaggle_team.loc[index, 'Location'] = f'{city}, {region}, {country}'
        
    if occupation and organization:
        kaggle_team.loc[index, 'Occupation'] = f'{occupation} at {organization}'
    elif organization:
        kaggle_team.loc[index, 'Occupation'] = organization
    elif occupation:
        kaggle_team.loc[index, 'Occupation'] = occupation
    
    kaggle_team.loc[index, 'NumFollowers'] = num_followers
    kaggle_team.loc[index, 'NumFollowing'] = num_following
    kaggle_team.loc[index, 'NumPosts'] = num_posts
    kaggle_team.loc[index, 'NumDatasets'] = num_datasets
    kaggle_team.loc[index, 'NumKernels'] = num_kernels
    kaggle_team.loc[index, 'NumCompetitions'] = num_comps

kaggle_team = kaggle_team.convert_dtypes()
display_html(kaggle_team, cols=['UserName', 'DisplayName', 'RegisterDate',
                                'Image', 'Location', 'Occupation'])


# <a id="location"></a>
# [Return to table of contents](#table-of-contents)

# # Where are Kaggle team members located?
# The [Geospatial Analysis](https://kaggle.com/learn/geospatial-analysis) course on Kaggle was one of my favourites. We'll put that knowledge to use here to geocode the locations from the table above.

# In[ ]:


m = folium.Map(location=[0, 30], tiles='OpenStreetMap', zoom_start=1)
mc = MarkerCluster()
locations = {}
for index in kaggle_team[kaggle_team.Location.notna()].index:
    time.sleep(REQUEST_DELAY)
    row = kaggle_team.iloc[index]
    displayname = row.DisplayName
    location = row.Location
    popup = f'{displayname}; {location}'
    tooltip = f'{displayname}; {location}'
    
    if location in locations:
        lat, long = locations[location]
    else:
        point = geocode(location, provider='nominatim').geometry.iloc[0]
        lat, long = point.y, point.x
        locations[location] = [lat, long]
    
    mc.add_child(Marker([lat, long], popup=popup, tooltip=tooltip))
    
m.add_child(mc)


# As expected, most Kaggle team members are in the United States.

# <a id="followers"></a>
# [Return to table of contents](#table-of-contents)

# # Followers
# 
# ### Who has the most followers?
# Meg has more than twice as many followers as anyone else on the Kaggle team!

# In[ ]:


followers = kaggle_team.nlargest(TOP_N, 'NumFollowers').copy()
display_html(followers, cols=['UserName', 'DisplayName', 'Image',
                              'Occupation', 'NumFollowers', 'NumFollowing'])


# ### Who follows other users the most?
# Paul follows more than twice as many users as anyone else on the Kaggle team!

# In[ ]:


following = kaggle_team.nlargest(TOP_N, 'NumFollowing').copy()
display_html(following, cols=['UserName', 'DisplayName', 'Image',
                              'Occupation', 'NumFollowing', 'NumFollowers'])


# Next, we will break down followers and follows by tiers.
# ### Who does the Kaggle team follow?

# In[ ]:


kaggle_following_tier = (pd.merge(user_followers[['UserId', 'FollowingUserId']], kaggle_team['Id'], left_on='UserId', right_on='Id')
                         .drop(columns='Id')
                         .merge(users, left_on='FollowingUserId', right_on='Id')[['Id', 'UserName', 'DisplayName', 'PerformanceTier']]
                         .drop_duplicates('Id')
                         .groupby('PerformanceTier')
                         .size()
                        )
plt.figure(figsize=(8, 5))
plt.title('Number of users followed by at least one Kaggle team member, by performance tier')
ax = sns.barplot(kaggle_following_tier.index, kaggle_following_tier.values, color='steelblue')


# Of all users that are followed by Kaggle team members, most are novices. If we look at the number of users on the platform by performance tier, we see that most users are novices. We'll divide the counts by the number of users in that tier and visualize the proportion of users in a tier that are followed by at least one Kaggle team member.

# In[ ]:


tier_counts = users['PerformanceTier'].value_counts()
tier_counts


# In[ ]:


kaggle_following_tier_prop = kaggle_following_tier / tier_counts
plt.figure(figsize=(8, 5))
plt.title('Proportion of users followed by at least one Kaggle team member, by performance tier')
ax = sns.barplot(kaggle_following_tier_prop.index, kaggle_following_tier_prop.values, color='steelblue')


# Almost every Kaggle team member is followed by another Kaggle team member. Also, about 40% of grandmasters are followed by at least one Kaggle team member.

# ### Who follows the Kaggle team?

# In[ ]:


kaggle_followers_tier = (pd.merge(user_followers[['UserId', 'FollowingUserId']], kaggle_team['Id'], left_on='FollowingUserId', right_on='Id')
                         .drop(columns='Id')
                         .merge(users, left_on='UserId', right_on='Id')[['Id', 'UserName', 'DisplayName', 'PerformanceTier']]
                         .drop_duplicates('Id')
                         .groupby('PerformanceTier')
                         .size()
                        )
plt.figure(figsize=(8, 5))
plt.title('Number of users that follow at least one Kaggle team member, by performance tier')
ax = sns.barplot(kaggle_followers_tier.index, kaggle_followers_tier.values, color='seagreen')


# As expected, most users that follow the Kaggle team are novices. Just as we did above, we'll visualize the proportion of users that follow at least one Kaggle team member, by performance tier.

# In[ ]:


kaggle_followers_tier_prop = kaggle_followers_tier / tier_counts
plt.figure(figsize=(8, 5))
plt.title('Proportion of users that follow at least one Kaggle team member, by performance tier')
ax = sns.barplot(kaggle_followers_tier_prop.index, kaggle_followers_tier_prop.values, color='seagreen')


# For the most part, the higher a user's tier, the more likely they are to follow at least one Kaggle team member.

# <a id="discussions"></a>
# [Return to table of contents](#table-of-contents)

# # Discussions
# ### Number of posts by the Kaggle team
# 

# In[ ]:


kaggle_team_posts = kaggle_team[['UserName', 'DisplayName', 'Image', 'Occupation', 'NumPosts']].nlargest(TOP_N, 'NumPosts').copy()
display_html(kaggle_team_posts)


# ### Popular topics by the Kaggle team

# In[ ]:


forum_topics = (topics[topics.KernelId.isna()]
                .merge(messages[['Id', 'PostUserId']], left_on='FirstForumMessageId', right_on='Id')
                .loc[:, ['Title', 'Score', 'TotalMessages', 'TotalReplies', 'PostUserId']]
                .merge(kaggle_team, left_on='PostUserId', right_on='Id')
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


kaggle_team_comps = kaggle_team[['UserName', 'DisplayName', 'Image', 'Occupation', 'NumCompetitions']].nlargest(TOP_N, 'NumCompetitions').copy()
display_html(kaggle_team_comps)


# <a id="kernels"></a>
# [Return to table of contents](#table-of-contents)

# # Kernels
# ### Number of kernels
# Of course, Kaggle Kerneler is at the top.

# In[ ]:


kaggle_team_kernels = kaggle_team[['UserName', 'DisplayName', 'Image', 'Occupation', 'NumKernels']].nlargest(TOP_N, 'NumKernels').copy()
display_html(kaggle_team_kernels)


# ### Popular kernels by the Kaggle team

# In[ ]:


kaggle_team_kernels = (pd.merge(kaggle_team[['Id', 'UserName', 'DisplayName', 'Image']], 
                               kernels[['Id', 'AuthorUserId', 'CurrentKernelVersionId', 'TotalVotes', 'CurrentUrlSlug']], 
                               left_on='Id', right_on='AuthorUserId')
                       .nlargest(TOP_N, 'TotalVotes')
                       .merge(kernel_versions[['Title', 'Id']], left_on='CurrentKernelVersionId', right_on='Id')
                      )
kaggle_team_kernels['url'] = KAGGLE_BASE_URL + kaggle_team_kernels.UserName + '/' + kaggle_team_kernels.CurrentUrlSlug
display_html(kaggle_team_kernels, cols=['UserName', 'DisplayName', 'Image', 'TotalVotes', 'Title', 'url'])


# <a id="datasets"></a>
# [Return to table of contents](#table-of-contents)

# # Datasets
# ### Number of datasets

# In[ ]:


kaggle_team_data = kaggle_team[['UserName', 'DisplayName', 'Image', 'Occupation', 'NumDatasets']].nlargest(TOP_N, 'NumDatasets').copy()
display_html(kaggle_team_data)


# ### Popular datasets by the Kaggle team

# In[ ]:


kaggle_team_datasets = (datasets[datasets.OwnerUserId.notna() & datasets.OwnerUserId.eq(datasets.CreatorUserId)]
                        .loc[:, ['Id', 'CreatorUserId', 'TotalVotes', 'CurrentDatasetVersionId']]
                        .merge(kaggle_team[['Id', 'UserName', 'DisplayName', 'Image']], left_on='CreatorUserId', right_on='Id')
                        .nlargest(TOP_N, 'TotalVotes')
                        .merge(dataset_versions[['Id', 'Title', 'Slug']], left_on='CurrentDatasetVersionId', right_on='Id')                        
                       )
kaggle_team_datasets['url'] = KAGGLE_BASE_URL + kaggle_team_datasets.UserName + '/' + kaggle_team_datasets.Slug
display_html(kaggle_team_datasets, cols=['UserName', 'DisplayName', 'Image', 'TotalVotes', 'Title', 'url'])

