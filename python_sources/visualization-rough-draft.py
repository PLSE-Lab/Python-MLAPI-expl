#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/dota2_2019_survey.csv')


# In[ ]:


df.head(3)


# In[ ]:


df.columns


# In[ ]:


age_coder= {'<13':0, '13-17':1, '18-20':2, '21-23':3, '24-26':4, '27-29':5, '30-35':6, 
            '36-40':7,'>40':8, '19-22':3, '23-29':4, '30-40':6}
gender_coder= {'Female':0, 'Male':1, 'Prefer not to say':1, 'Other':1, 'Trans female':1, 'Trans-male':1}
years_dota2_coder= {"I don't play":0, '< 6 months':1, '6 months - 1 year':2, '1 - 3 years':3,
                    '3 - 5 years':4, '5 - 10 years':5, '10+ years':6}
games_week_coder= {'0':0, '0 - 1 (play less than a game a week but still play)':1, '1 - 5':2, '6 - 10':3, 
                   '11 - 20':4, '20 - 30':5,  '30 - 40':6, '> 40':7}
badge_coder= {"I'm not ranked.":0, 'Herald':1, 'Guardian':2, 'Crusader':3, 'Archon':4, 'Legend':5,
              'Ancient':6, 'Divine':7, 'Immortal':8}
game_mode_coder= {'All Pick/Single Draft/Random Draft/':0,'All Pick/Single Draft/Random Draft':0, 
                  'Custom Games':1, 'Turbo':1, 'Captains Mode/Captains Draft':1, 'Ability Draft':1}
role_coder= {'Mid':2, 'Carry':3, 'Support (4/5)':0, 'Offlane':1}
subreddit_years_coder= {'<3 Months':0, '3 - 6 Months':1, '6 Months - 1 Year':2, '1 - 2 Years':3, '2 - 3 Years':4
                        , '3+ Years':5,  '3 - 5 Years':5,  '+5 Years':6}
subreddit_check_coder= {'Only when something is happening (New Patch Drama, etc.)':0, 'Once a month or less':1, 
                        'Once a week':2, 'Once a day':3, 'Multiple times a day':4}
post_freq_coder= {'Never':0, 'Seldom':1, 'Sometimes':2, 'Often':3, None:0}
comment_freq_coder= {'Never':0, 'Seldom':1, 'Sometimes':2, 'Often':3, None:0}
patch_news_importance_coder= {'Very important':3, 'Sort of important':3, '4 = Very Important':3, '3':2,'2':1, '1 = Not Important':0}
bug_reports_importance_coder= {'Very important':3, 'Sort of important':3, '4 = Very Important':3, '3':2,'2':1, '1 = Not Important':0}
user_created_content_coder= {'Very important':3, 'Sort of important':3, '4 = Very Important':3, '3':2,'2':1, '1 = Not Important':0}
eports_news_coder= {'Very important':3, 'Sort of important':3, '4 = Very Important':3, '3':2,'2':1, '1 = Not Important':0}
gamplay_discuss_coder= {'Very important':3, 'Sort of important':3, '4 = Very Important':3, '3':2,'2':1, '1 = Not Important':0}
gamplay_highlights_coder= {'Very important':3, 'Sort of important':3, '4 = Very Important':3, '3':2,'2':1, '1 = Not Important':0}
guides_tips_coder= {'Very important':3, 'Sort of important':3, '4 = Very Important':3, '3':2,'2':1, '1 = Not Important':0}
comedy_posts_coder= {'Very important':3, 'Sort of important':3, '4 = Very Important':3, '3':2,'2':1, '1 = Not Important':0}
requests_suggestions_complaints_coder= {'Very important':3, 'Sort of important':3, '4 = Very Important':3, '3':2,'2':1, '1 = Not Important':0}
sidebar_usage_coder= {'Never':0, 'Seldom':1, 'Sometimes':2, 'Often':3, None:0}
rate_d2tournamentthreads_coder= {'Meh':0, None:1,'Okay':2, 'Like em':3, 'Love em':4}
rate_competitive_hero_trends_coder= {'Meh':0, None:1,'Okay':2, 'Like em':3, 'Love em':4}
rate_findbattlecupparty_coder= {'Meh':0, None:1,'Okay':2, 'Like em':3, 'Love em':4}
rate_competitive_player_spotlights_coder= {'Meh':0, None:1,'Okay':2, 'Like em':3, 'Love em':4}
rate_competitive_team_discussion_coder= {'Meh':0, None:1,'Okay':2, 'Like em':3, 'Love em':4}
rate_hero_discussions_coder= {'Meh':0, None:1,'Okay':2, 'Like em':3, 'Love em':4}
rate_item_discussions_coder= {'Meh':0, None:1,'Okay':2, 'Like em':3, 'Love em':4}
rate_free_talk_threads_coder= {'Meh':0, None:1,'Okay':2, 'Like em':3, 'Love em':4}
rate_news_megathreads_coder= {'Meh':0, None:1,'Okay':2, 'Like em':3, 'Love em':4}
rate_this_week_learned_coder= {'Meh':0, None:1,'Okay':2, 'Like em':3, 'Love em':4}
rate_stupid_questions_coder= {'Meh':0, None:1,'Okay':2, 'Like em':3, 'Love em':4}
show_case_badge_coder= {"I don't care":0, "It shouldn't be done at all":1, 
                        "It should only be allowed for very high ranked players.":2, 
                        "It should be allowed for everyone":3}
follow_dota_esports_coder= {"I don't watch at all":0, 'I only watch The International': 1, 
                            'I watch some events and the highlights for the ones I miss': 2, 'I follow only the major tournaments (more than just The International)': 3, 
                            'I watch nearly every event':4}


# In[ ]:


df['age_encoder']= df['age'].map(age_coder)
df['gender_encoder']= df['gender'].map(gender_coder)
df['years_dota2_encoder']= df['years_dota2'].map(years_dota2_coder)
df['games_week_encoder']= df['games_week'].map(games_week_coder)
df['badge_encoder']= df['badge'].map(badge_coder)
df['game_mode_encoder']= df['game_mode'].map(game_mode_coder)
df['role_encoder']= df['ingame_role'].map(role_coder)
df['subreddit_years_encoder']= df['subreddit_years'].map(subreddit_years_coder)
df['subreddit_check_encoder']= df['subreddit_check'].map(subreddit_check_coder)
df['post_freq_encoder']= df['post_freq'].map(post_freq_coder)
df['comment_freq_encoder']= df['comment_freq'].map(comment_freq_coder)
df['patch_news_importance_encoder']= df['patch_news_importance'].map(patch_news_importance_coder)
df['bug_reports_importance_encoder']= df['bug_reports_importance'].map(bug_reports_importance_coder)
df['user_created_content_encoder']= df['user_created_content_imp'].map(user_created_content_coder)
df['eports_news_encoder']= df['eports_news_imp'].map(eports_news_coder)
df['gamplay_discuss_encoder']= df['gamplay_discuss_imp'].map(gamplay_discuss_coder)
df['gamplay_highlights_encoder']= df['gamplay_highlights_imp'].map(gamplay_highlights_coder)
df['guides_tips_encoder']= df['guides_tips_imp'].map(guides_tips_coder)
df['comedy_posts_encoder']= df['comedy_posts_imp'].map(comedy_posts_coder)
df['requests_suggestions_complaints_encoder']= df['requests_suggestions_complaints_imp'].map(requests_suggestions_complaints_coder)
df['sidebar_usage_encoder']= df['sidebar_usage'].map(sidebar_usage_coder)
df['rate_d2tournamentthreads_encoder']= df['rate_d2tournamentthreads'].map(rate_d2tournamentthreads_coder)
df['rate_competitive_hero_trends_encoder']= df['rate_competitive_hero_trends'].map(rate_competitive_hero_trends_coder)
df['rate_findbattlecupparty_encoder']= df['rate_findbattlecupparty'].map(rate_findbattlecupparty_coder)
df['rate_competitive_player_spotlights_encoder']= df['rate_competitive_player_spotlights'].map(rate_competitive_player_spotlights_coder)
df['rate_competitive_team_discussion_encoder']= df['rate_competitive_team_discussion'].map(rate_competitive_team_discussion_coder)
df['rate_hero_discussions_encoder']= df['rate_hero_discussions'].map(rate_hero_discussions_coder)
df['rate_item_discussions_encoder']= df['rate_item_discussions'].map(rate_item_discussions_coder)
df['rate_free_talk_threads_encoder']= df['rate_free_talk_threads'].map(rate_free_talk_threads_coder)
df['rate_news_megathreads_encoder']= df['rate_news_megathreads'].map(rate_news_megathreads_coder)
df['rate_this_week_learned_encoder']= df['rate_this_week_learned'].map(rate_this_week_learned_coder)
df['rate_stupid_questions_encoder']= df['rate_stupid_questions'].map(rate_stupid_questions_coder)
df['show_case_badge_encoder']= df['show_case_badge'].map(show_case_badge_coder)
df['follow_dota_esports_encoder']= df['follow_dota_esports'].map(follow_dota_esports_coder)


# In[ ]:


plt.figure(figsize=(12,8))
df_age= df['age_encoder'].value_counts()
ax= sns.barplot(y=df_age.values, x=df_age.index)
ax.set(xticklabels=['<13', '13-17', '18-20', '21-23', '24-26', '27-29', '30-35', '36-40','>40'])
ax.set(xlabel='Age Brackets', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_gender= df['gender_encoder'].value_counts()
ax= sns.barplot(y=df_gender.values, x=df_gender.index)
ax.set(xticklabels=['Female', 'Male'])
ax.set(xlabel='Gender', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_years_dota2= df['years_dota2_encoder'].value_counts()
ax= sns.barplot(y=df_years_dota2.values, x=df_years_dota2.index)
ax.set(xticklabels=["I don't play", '< 6 months', '6 months - 1 year', '1 - 3 years',
                    '3 - 5 years', '5 - 10 years', '10+ years'])
ax.set(xlabel='Years Playing Dota', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_games_week= df['games_week_encoder'].value_counts()
ax= sns.barplot(y=df_games_week.values, x=df_games_week.index)
ax.set(xticklabels=['None', '0 - 1', '1 - 5', '6 - 10', 
                   '11 - 20', '20 - 30',  '30 - 40', '> 40'])
ax.set(xlabel='Games played per day', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_badge= df['badge_encoder'].value_counts()
ax= sns.barplot(y=df_badge.values, x=df_badge.index)
ax.set(xticklabels=["None", 'Herald', 'Guardian', 'Crusader', 'Archon', 'Legend',
              'Ancient', 'Divine', 'Immortal'])
ax.set(xlabel='Ranked Badge', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_game_mode= df['game_mode_encoder'].value_counts()
ax= sns.barplot(y=df_game_mode.values, x=df_game_mode.index)
ax.set(xticklabels=['All Pick/Single Draft/Random Draft', 'Others'])
ax.set(xlabel='Played Game Mode', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_role= df['role_encoder'].value_counts()
ax= sns.barplot(y=df_role.values, x=df_role.index)
ax.set(xticklabels=['Support (4/5)','Offlane', 'Mid', 'Carry'])
ax.set(xlabel='Preferred Role', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_subreddit_years= df['subreddit_years_encoder'].value_counts()
ax= sns.barplot(y=df_subreddit_years.values, x=df_subreddit_years.index)
ax.set(xticklabels=['<3 Months', '3 - 6 Months', '6 Months - 1 Year', '1 - 2 Years', '2 - 3 Years'
                       ,  '3 - 5 Years',  '+5 Years'])
ax.set(xlabel='How long have you subscribed to the subreddit', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_subreddit_check= df['subreddit_check_encoder'].value_counts()
ax= sns.barplot(y=df_subreddit_check.values, x=df_subreddit_check.index)
ax.set(xticklabels=['Only when Trending', 'Once a month', 
                        'Once a week', 'Once a day', 'Multiple times a day'])
ax.set(xlabel='How often do you check the subreddit', ylabel='Count')
plt.show()


# In[ ]:


'Never':0, 'Seldom':1, 'Sometimes':2, 'Often':3, None:0


# In[ ]:


plt.figure(figsize=(12,8))
df_post_freq= df['post_freq_encoder'].value_counts()
ax= sns.barplot(y=df_post_freq.values, x=df_post_freq.index)
ax.set(xticklabels=['Never', 'Seldom', 'Sometimes', 'Often'])
ax.set(xlabel='How often do you post on the subreddit', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_comment_freq= df['comment_freq_encoder'].value_counts()
ax= sns.barplot(y=df_comment_freq.values, x=df_comment_freq.index)
ax.set(xticklabels=['Never', 'Seldom', 'Sometimes', 'Often'])
ax.set(xlabel='How often do you comment on the subreddit', ylabel='Count')
plt.show()


# In[ ]:


'Very important':3, 'Sort of important':3, '4 = Very Important':3, '3':2,'2':1, '1 = Not Important'


# In[ ]:


plt.figure(figsize=(12,8))
df_patch_news_importance= df['patch_news_importance_encoder'].value_counts()
ax= sns.barplot(y=df_patch_news_importance.values, x=df_patch_news_importance.index)
ax.set(xticklabels=['Not Important', 'Sort of Important', 'Important', 'Very Important'])
ax.set(xlabel='Patch News Important?', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_bug_reports_importance= df['bug_reports_importance_encoder'].value_counts()
ax= sns.barplot(y=df_bug_reports_importance.values, x=df_bug_reports_importance.index)
ax.set(xticklabels=['Not Important', 'Sort of Important', 'Important', 'Very Important'])
ax.set(xlabel='Bug Reports Important?', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_user_created_content= df['user_created_content_encoder'].value_counts()
ax= sns.barplot(y=df_user_created_content.values, x=df_user_created_content.index)
ax.set(xticklabels=['Not Important', 'Sort of Important', 'Important', 'Very Important'])
ax.set(xlabel='User Created Content Important?', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_eports_news= df['eports_news_encoder'].value_counts()
ax= sns.barplot(y=df_eports_news.values, x=df_eports_news.index)
ax.set(xticklabels=['Not Important', 'Sort of Important', 'Important', 'Very Important'])
ax.set(xlabel='Esports News Important?', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_gamplay_discuss= df['gamplay_discuss_encoder'].value_counts()
ax= sns.barplot(y=df_gamplay_discuss.values, x=df_gamplay_discuss.index)
ax.set(xticklabels=['Not Important', 'Sort of Important', 'Important', 'Very Important'])
ax.set(xlabel='Gameplay Discussions Important?', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_gamplay_highlights= df['gamplay_highlights_encoder'].value_counts()
ax= sns.barplot(y=df_gamplay_highlights.values, x=df_gamplay_highlights.index)
ax.set(xticklabels=['Not Important', 'Sort of Important', 'Important', 'Very Important'])
ax.set(xlabel='Gameplay Highlights Important?', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_guides_tips= df['guides_tips_encoder'].value_counts()
ax= sns.barplot(y=df_guides_tips.values, x=df_guides_tips.index)
ax.set(xticklabels=['Not Important', 'Sort of Important', 'Important', 'Very Important'])
ax.set(xlabel='Guides and Tips Important?', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_comedy_posts= df['comedy_posts_encoder'].value_counts()
ax= sns.barplot(y=df_comedy_posts.values, x=df_comedy_posts.index)
ax.set(xticklabels=['Not Important', 'Sort of Important', 'Important', 'Very Important'])
ax.set(xlabel='Comedy Posts Important?', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_requests_suggestions_complaints= df['requests_suggestions_complaints_encoder'].value_counts()
ax= sns.barplot(y=df_requests_suggestions_complaints.values, x=df_requests_suggestions_complaints.index)
ax.set(xticklabels=['Not Important', 'Sort of Important', 'Important', 'Very Important'])
ax.set(xlabel='Requests and Suggestions Important?', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_sidebar_usage= df['sidebar_usage_encoder'].value_counts()
ax= sns.barplot(y=df_sidebar_usage.values, x=df_sidebar_usage.index)
plt.show()


# In[ ]:


'Meh':0, None:1,'Okay':2, 'Like em':3, 'Love em':4


# In[ ]:


plt.figure(figsize=(12,8))
df_rate_d2tournamentthreads= df['rate_d2tournamentthreads_encoder'].value_counts()
ax= sns.barplot(y=df_rate_d2tournamentthreads.values, x=df_rate_d2tournamentthreads.index)
ax.set(xticklabels=['Meh','Blank' ,'Okay', 'Like em', 'Love em'])
ax.set(xlabel='Rate: Dota2 Tournament Threads', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_rate_competitive_hero_trends= df['rate_competitive_hero_trends_encoder'].value_counts()
ax= sns.barplot(y=df_rate_competitive_hero_trends.values, x=df_rate_competitive_hero_trends.index)
ax.set(xticklabels=['Meh','Blank' ,'Okay', 'Like em', 'Love em'])
ax.set(xlabel='Rate: Competitive Hero Threads', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_rate_findbattlecupparty= df['rate_findbattlecupparty_encoder'].value_counts()
ax= sns.barplot(y=df_rate_findbattlecupparty.values, x=df_rate_findbattlecupparty.index)
ax.set(xticklabels=['Meh','Blank' ,'Okay', 'Like em', 'Love em'])
ax.set(xlabel='Rate: Competitive Hero Threads', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_rate_competitive_player_spotlights= df['rate_competitive_player_spotlights_encoder'].value_counts()
ax= sns.barplot(y=df_rate_competitive_player_spotlights.values, x=df_rate_competitive_player_spotlights.index)
ax.set(xticklabels=['Meh','Blank' ,'Okay', 'Like em', 'Love em'])
ax.set(xlabel='Rate: Competitive Player Spotlights Threads', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_rate_competitive_team_discussion= df['rate_competitive_team_discussion_encoder'].value_counts()
ax= sns.barplot(y=df_rate_competitive_team_discussion.values, x=df_rate_competitive_team_discussion.index)
ax.set(xticklabels=['Meh','Blank' ,'Okay', 'Like em', 'Love em'])
ax.set(xlabel='Rate: Competitive Team Discussion Threads', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_rate_hero_discussions= df['rate_hero_discussions_encoder'].value_counts()
ax= sns.barplot(y=df_rate_hero_discussions.values, x=df_rate_hero_discussions.index)
ax.set(xticklabels=['Meh','Blank' ,'Okay', 'Like em', 'Love em'])
ax.set(xlabel='Rate: Hero DiscussionThreads', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_rate_item_discussions= df['rate_item_discussions_encoder'].value_counts()
ax= sns.barplot(y=df_rate_item_discussions.values, x=df_rate_item_discussions.index)
ax.set(xticklabels=['Meh','Blank' ,'Okay', 'Like em', 'Love em'])
ax.set(xlabel='Rate: Item Discussion Threads', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_rate_free_talk_threads= df['rate_free_talk_threads_encoder'].value_counts()
ax= sns.barplot(y=df_rate_free_talk_threads.values, x=df_rate_free_talk_threads.index)
ax.set(xticklabels=['Meh','Blank' ,'Okay', 'Like em', 'Love em'])
ax.set(xlabel='Rate: Free Talk Threads', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_rate_news_megathreads= df['rate_news_megathreads_encoder'].value_counts()
sns.barplot(y=df_rate_news_megathreads.values, x=df_rate_news_megathreads.index)
ax.set(xticklabels=['Meh','Blank' ,'Okay', 'Like em', 'Love em'])
ax.set(xlabel='Rate: News Megathreads', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_rate_news_megathreads= df['rate_news_megathreads_encoder'].value_counts()
sns.barplot(y=df_rate_news_megathreads.values, x=df_rate_news_megathreads.index)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_rate_this_week_learned= df['rate_this_week_learned_encoder'].value_counts()
ax= sns.barplot(y=df_rate_this_week_learned.values, x=df_rate_this_week_learned.index)
ax.set(xticklabels=['Meh','Blank' ,'Okay', 'Like em', 'Love em'])
ax.set(xlabel='Rate: This Week I Learned Threads', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_rate_stupid_questions= df['rate_stupid_questions_encoder'].value_counts()
ax= sns.barplot(y=df_rate_stupid_questions.values, x=df_rate_stupid_questions.index)
ax.set(xticklabels=['Meh','Blank' ,'Okay', 'Like em', 'Love em'])
ax.set(xlabel='Rate: Stupid Questions Threads', ylabel='Count')
plt.show()


# In[ ]:


"I don't care":0, "It shouldn't be done at all":1, 
                        "It should only be allowed for very high ranked players.":2, 
                        "It should be allowed for everyone":3


# In[ ]:


plt.figure(figsize=(12,8))
df_show_case_badge= df['show_case_badge_encoder'].value_counts()
ax= sns.barplot(y=df_show_case_badge.values, x=df_show_case_badge.index)
ax.set(xticklabels=["I don't care", "It should not be done", "Only for very high ranks", "Allowed for Everyone"])
ax.set(xlabel='Rate: Stupid Questions Threads', ylabel='Count')
plt.show()


# In[ ]:


"I don't watch at all":0, 'I only watch The International': 1, 
                            'I watch some events and the highlights for the ones I miss': 2, 'I follow only the major tournaments (more than just The International)': 3, 
                            'I watch nearly every event':4


# In[ ]:


plt.figure(figsize=(12,8))
df_follow_dota_esports= df['follow_dota_esports_encoder'].value_counts()
ax= sns.barplot(y=df_follow_dota_esports.values, x=df_follow_dota_esports.index)
ax.set(xticklabels=["Don't Watch", "Only The International", "Some events/Watch Highlights", "Only the Majors", "Watch All Events"])
ax.set(xlabel='How much do you follow Dota 2 Esports', ylabel='Count')
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_behaviour= df['behaviour'].value_counts()
sns.barplot(y=df_behaviour.values, x=df_behaviour.index)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_solo_party= df['solo_party'].value_counts()
sns.barplot(y=df_solo_party.values, x=df_solo_party.index)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_unranked_ranked= df['unranked_ranked'].value_counts()
sns.barplot(y=df_unranked_ranked.values, x=df_unranked_ranked.index)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_least_favorite_content= df['least_favorite_content'].value_counts()
sns.barplot(y=df_least_favorite_content.values, x=df_least_favorite_content.index)
plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_subreddit_years= df['subreddit_years_encoder'].value_counts()
sns.barplot(y=df_subreddit_years.values, x=df_subreddit_years.index)

plt.show()


# In[ ]:


plt.figure(figsize=(12,8))
df_subreddit_years= df['subreddit_years_encoder'].value_counts()
ax= sns.barplot(y=df_subreddit_years.values, x=df_subreddit_years.index)
ax.set(xticklabels=['<3 Months', '3 - 6 Months', '6 Months - 1 Year', '1 - 2 Years', '2 - 3 Years'
                       ,  '3 - 5 Years',  '+5 Years'])
ax.set(xlabel='How long have you subscribed to the subreddit', ylabel='Count')
plt.show()


# {'Very important':3, 'Sort of important':3, '4 = Very Important':3, '3':2,'2':1, '1 = Not Important':0}

# In[ ]:


df['years_dota2_encoder']= df['years_dota2'].map(years_dota2_coder)


# In[ ]:


df_veteran_encoder= {0:'less than 5 yrs' ,1:'less than 5 yrs', 2:'less than 5 yrs', 3:'less than 5 yrs', 4:'less than 5 yrs', 5:'more than 5 yrs', 6:'more than 5 yrs'}
df['dota_veteran'] = df['years_dota2_encoder'].map(df_veteran_encoder)
plt.figure(figsize=(12,8))
df_subreddit_veteran= df['dota_veteran'].value_counts()
sns.barplot(y=df_subreddit_veteran.values, x=df_subreddit_veteran.index)

plt.show()


# In[ ]:


ax= sns.catplot(x='badge_encoder', kind='count', hue='dota_veteran',data=df)
ax.set(xticklabels=["None", 'Herald', 'Guardian', 'Crusader', 'Archon', 'Legend',
              'Ancient', 'Divine', 'Immortal'])
ax.set(xlabel='Ranked Badge', ylabel='Count')
ax.set_xticklabels(rotation=90)
plt.show()


# In[ ]:


ax= sns.catplot(x='subreddit_check_encoder', kind='count', hue='dota_veteran',data=df)
ax.set(xticklabels=['Only when Trending', 'Once a month', 
                        'Once a week', 'Once a day', 'Multiple times a day'])
ax.set(xlabel='How often do you check the subreddit', ylabel='Count')
ax.set_xticklabels(rotation=90)
plt.show()


# In[ ]:


ax= sns.catplot(x='games_week_encoder', kind='count', hue='dota_veteran',data=df)
ax.set(xticklabels=['None', '0 - 1', '1 - 5', '6 - 10', 
                   '11 - 20', '20 - 30',  '30 - 40', '> 40'])
ax.set(xlabel='Games played per week', ylabel='Count')
ax.set_xticklabels(rotation=90)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




