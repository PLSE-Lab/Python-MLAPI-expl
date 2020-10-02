#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# # Scripts activity in competitions
# 
# The things I want to look at:
# 
# - How do public and private scores of script submissions evolve?
# - How many new versions appear daily, how many contain changes?
# - How many submissions are made from the scripts, how many use the script version currently leading on LB?
# 
# ## Load the data

# In[ ]:


# Competitions - use only those that award points
competitions = (pd.read_csv('../input/Competitions.csv')
                .rename(columns={'Id':'CompetitionId'}))
competitions = competitions[(competitions.UserRankMultiplier > 0)]
# Scriptprojects to link scripts to competitions
scriptprojects = (pd.read_csv('../input/ScriptProjects.csv')
                    .rename(columns={'Id':'ScriptProjectId'}))
# Evaluation algorithms
evaluationalgorithms = (pd.read_csv('../input/EvaluationAlgorithms.csv')
                          .rename(columns={'Id':'EvaluationAlgorithmId'}))
competitions = (competitions.merge(scriptprojects[['ScriptProjectId','CompetitionId']],
                                   on='CompetitionId',how='left')
                            .merge(evaluationalgorithms[['IsMax','EvaluationAlgorithmId']],
                                   on='EvaluationAlgorithmId',how='left')
                            .dropna(subset = ['ScriptProjectId'])
                            .set_index('CompetitionId'))
competitions['ScriptProjectId'] = competitions['ScriptProjectId'].astype(int)
# Fill missing values for two competitions
competitions.loc[4488,'IsMax'] = True # Flavours of physics
competitions.loc[4704,'IsMax'] = False # Santa's Stolen Sleigh
# Scripts
scripts = pd.read_csv('../input/Scripts.csv')
# Script versions
# List necessary columns to avoid reading script versions content
svcols = ['Id','Title','DateCreated','ScriptId',
          'LinesInsertedFromPrevious','LinesDeletedFromPrevious', 
          'LinesChangedFromPrevious','LinesInsertedFromFork', 
          'LinesDeletedFromFork', 'LinesChangedFromFork']
scriptversions = pd.read_csv('../input/ScriptVersions.csv', 
                             usecols=svcols)
scriptversions['DateCreated'] = pd.to_datetime(scriptversions['DateCreated'])
# Determine if a script version contains changes 
#(either from fork parent or from previous version)
isfirst = scriptversions.Id.isin(scripts.FirstScriptVersionId)
scriptversions.loc[isfirst, 'IsChanged'] = scriptversions.loc[isfirst, 
            ['LinesInsertedFromFork', 
             'LinesDeletedFromFork', 
             'LinesChangedFromFork']].any(axis=1)
scriptversions.loc[~(isfirst), 'IsChanged'] = scriptversions.loc[~(isfirst), 
            ['LinesInsertedFromPrevious', 
             'LinesDeletedFromPrevious', 
             'LinesChangedFromPrevious']].any(axis=1)
# Submissions
submissions = pd.read_csv('../input/Submissions.csv')
submissions = submissions.dropna(subset=['Id','DateSubmitted','PublicScore'])
submissions.DateSubmitted = pd.to_datetime(submissions.DateSubmitted)


# Some functions for analyzing scripts activity.

# In[ ]:


def report_script_activity(scriptversions, submissions, ismax):
    scores = pd.DataFrame()
    scores['BestPublic'] = submissions.PublicScore.cummax() if ismax else submissions.PublicScore.cummin()
    scores.loc[scores.BestPublic == submissions.PublicScore, 'BestPrivate'] = submissions.PrivateScore
    scores.BestPrivate = scores.BestPrivate.fillna(method='ffill')
    scores['DateSubmitted'] = submissions['DateSubmitted']
    activity = pd.DataFrame()
    activity['Submissions'] = submissions.groupby(submissions.DateSubmitted.dt.date)['Id'].size()
    activity['SubmissionsBest'] = ((submissions['PublicScore']==scores['BestPublic'])
                                   .groupby(submissions.DateSubmitted.dt.date).sum())
    activity['Versions'] = scriptversions.groupby(scriptversions.DateCreated.dt.date)['Id'].size()
    activity['VersionsChanged'] = scriptversions.groupby(scriptversions.DateCreated.dt.date)['IsChanged'].sum()
    return scores, activity

def plot_script_activity(scores, activity):
    fig, ax = plt.subplots(3,1, figsize=(10,8), sharex=True, gridspec_kw = {'height_ratios':[1,3,1]})
    colors = cm.Blues(np.linspace(0.5, 0.8, 2))
    ax[0].bar(activity.index, activity.Versions,color=colors[0])
    ax[0].bar(activity.index, activity.VersionsChanged,color=colors[1])
    ax[0].set_title('Daily new versions')
    ax[0].legend(['all','with changes'])
    ax[1].plot(scores.DateSubmitted, scores.BestPublic, '-', 
               scores.DateSubmitted, scores.BestPrivate, '-')
    ax[1].set_title('Best public submission scores')
    ax[1].legend(['Public','Private'],loc=4)
    ax[2].bar(activity.index, activity.Submissions,color=colors[0]);
    ax[2].bar(activity.index, activity.SubmissionsBest,color=colors[1]);
    ax[2].set_title('Daily submissions');
    ax[2].legend(['all','best public']);
    return fig, ax


# In[ ]:


def report_competition(competitionId):
    ismax = competitions.loc[competitionId,'IsMax']
    scriptprojectid = competitions.loc[competitionId, 'ScriptProjectId']
    s = scripts.loc[scripts.ScriptProjectId==scriptprojectid,'Id'].values
    v = scriptversions.loc[scriptversions.ScriptId.isin(s)]
    sub = (submissions.loc[submissions.SourceScriptVersionId.isin(v.Id)]
                      .sort_values(by='DateSubmitted'))
    scores, activity = report_script_activity(v,sub,ismax)
    fig, ax = plot_script_activity(scores, activity)
    plt.suptitle(competitions.loc[competitionId,'Title'],fontsize='large')


# In[ ]:


# Recent competitions
competitions.sort_values(by='Deadline',ascending=False)[['Title']].head()


# ## Expedia Hotel Recommendations
# 
# Let's look at scripts activity in the Expedia Hotel Recommendations competition.

# In[ ]:


report_competition(5056)


# Things to note:
# 
# - Public and private LB scores follow each other closely.
# - There are 3 levels of submission scores
#   1. the first is the "most popular local hotel" approach, 
#   1. the second improves on that result by partially using the data leak
#   1. the third and highest is from the "Leakage solution" script by ZFTurbo and its many forks.
# - The number of script submissions increased drastically after a high-scoring script appeared.
# - The number of script submissions is low in the last week of the competition. Are script submissions mostly made by new entrants? Or do people start being more careful with the limited number of submissions left?
# 
# Let's take a closer look at the "Leakage solution" script and its forks.

# In[ ]:


# find scriptIds of scripts forked from ancestors, and of their forks, and of their forks...
def find_descendants(ancestors, scripts, scriptversions):
    if len(ancestors) == 0:
        return np.array([],dtype=int)
    ancestors_versions = scriptversions.loc[scriptversions.ScriptId.isin(ancestors),'Id']
    children = scripts.loc[scripts.ForkParentScriptVersionId.isin(ancestors_versions.values),'Id'].values
    return np.concatenate((children, find_descendants(children, scripts, scriptversions)))
# find scripts with most descendants in a competition
def find_most_forked_scripts(competitionId, n = 5):
    print('Most forked scripts in {}'.format(competitions.loc[competitionId,'Title']))
    # Find scripts project id
    projectId = competitions.loc[competitionId,'ScriptProjectId']
    # Read in scripts and scriptversions data
    s = scripts.loc[(scripts.ScriptProjectId==projectId)].copy()
    v = scriptversions.loc[scriptversions.ScriptId.isin(s.Id)]
    origmask = s.ForkParentScriptVersionId.isnull()
    s.loc[origmask,'Nforks'] = s.loc[origmask,'Id'].apply(lambda x,s,v: find_descendants([x],s,v).shape[0],args=(s,v))
    return s[['Id','UrlSlug','Nforks']].sort_values(by='Nforks',ascending=False).head(n)


# 

# In[ ]:


find_most_forked_scripts(5056)


# ### The Leakage solution
# 
# The 'Leakage solution' script by @ZFTurbo combined the most popular local hotel approach with exploiting the data leak. There are 371 scripts descended from it. Variations they made included:
# 
# - tuning the weights of booking and click events
# - tuning the weights of old versus new events
# - including other grouping strategies and weighting their contributions
# - probably something else as well (drop a comment to add something)
# 
# How did the scores evolve?

# In[ ]:


def report_competition_script(competitionId, scriptId):
    ismax = competitions.loc[competitionId,'IsMax']
    children = find_descendants([scriptId],scripts,scriptversions)
    family = np.append(children,[scriptId])
    v = scriptversions.loc[scriptversions.ScriptId.isin(family)]
    sub = (submissions.loc[submissions.SourceScriptVersionId.isin(v.Id)]
                      .sort_values(by='DateSubmitted'))
    scores, activity = report_script_activity(v,sub,ismax)
    fig, ax = plot_script_activity(scores, activity)
    scriptname = scripts.loc[scripts.Id==scriptId,'UrlSlug'].values[0]
    competitionname = competitions.loc[competitionId,'Title']
    title = '{} script and all its forks\n{}'.format(scriptname, competitionname)
    plt.suptitle(title,fontsize='large')


# In[ ]:


report_competition_script(5056, 60666)


# A zoom in to this subset of scripts reveals that
# 
# - 20-40 variations of the model appeared every day, trying new ideas and parameters.
# - Both private and public scores of best public submission improved as a result of this model tuning. 
# - The gap between public and private scores was widening over time which may indicate some public LB overfitting.
# - Submissions made from scripts followed the leaderboard closely, half or more of script submissions came from the current best-scoring script.
# 
# ### The history of forks
# 
# What if we look at the history of forks of this script? I want to visualize how new versions appear, see their scores and number of submissions from each one.

# In[ ]:


def plot_script(scriptId, ax, x=0, vmin=0.49, vmax = 0.502):
    ax.set_title('The history of forks of {}'.format(s.loc[scripts.Id==scriptId,'UrlSlug'].values[0]),
                 fontsize='x-large')
    versions = v.loc[v.ScriptId==scriptId].sort_values(by='DateCreated', ascending=False)
    ax.plot(versions.DateCreated.values,
            np.ones(versions.shape[0])*x, 
            'k-',zorder=1, linewidth=0.5)
    ax.scatter(versions.DateCreated.values, 
               np.ones(versions.shape[0])*x, 
               s = 2*versions.Nsubmissions.values,
               c = versions.PublicScore.values,
               cmap = cm.rainbow,marker='o',alpha=0.9,
               vmin = vmin, zorder=2,vmax=vmax)
    n = 1
    for versionId in versions.index:
        versionDate = versions.loc[versionId,'DateCreated']
        desc = s.loc[s.ForkParentScriptVersionId==versionId]
        if desc.shape[0] == 0:
            continue
        desc = desc.sort_values(by='Id',ascending=False)
        for script in desc.Id.values:
            forkversion = desc.loc[desc.Id==script,'FirstScriptVersionId'].values[0]
            forkversionDate = v.loc[forkversion,'DateCreated']
            ax.plot([versionDate, forkversionDate],
                    [x,x+n],
                    'k-',zorder=1, linewidth=0.5,alpha = 0.5)
            nd = plot_script(script, ax, x=x+n)
            n += nd
    return n


# In[ ]:


scriptId=60666
children = find_descendants([scriptId],scripts,scriptversions)
family = np.append(children,[scriptId])
s = scripts.loc[scripts.Id.isin(family)]
v = scriptversions.loc[scriptversions.ScriptId.isin(family)].set_index('Id')
sub = (submissions.loc[submissions.SourceScriptVersionId.isin(v.index)]
                  .sort_values(by='DateSubmitted'))
v['Nsubmissions'] = sub.groupby('SourceScriptVersionId').size()
v['PublicScore'] = sub.groupby('SourceScriptVersionId')['PublicScore'].agg('first')


# There's probably some library for this but I have created a function to plot the tree of forks using matplotlib. 
# 
# - Horizontal levels are scripts
# - Markers denote script versions, date created is their x coordinate
# - Color is public LB score (redder is better)
# - Size is the number of submissions from this version

# In[ ]:


fig, ax = plt.subplots(figsize=(12,12))
n = plot_script(60666, ax)
oneday = pd.to_timedelta(1, unit='day')
ax.set_xlim(v.DateCreated.min()-oneday, 
            competitions.loc[5056,'Deadline']);
ax.set_ylim(-10, n+10);


# The original script is the lowermost line. Its last version appeared on May the 10th. The later model developments took place in forks. It seems like the best scoring fork or version gains initiative and receives a lot of forks in turn leading to a gradual improvement of scores. There is also probably some exchange of ideas between simultaneously developing scripts (like those long lines at y=55 and y=125).
# 
# What other interesting things might we see in this structure? Leave a comment if you have an idea =).
# 
# 
# ## Santander Customer Satisfaction
# 
# Santander Customer Satisfaction was a large competition that suffered a really big leaderboard shakeup. What can scripts activity tell us about it?

# In[ ]:


competitionId = 4986
report_competition(competitionId)


# - After an initial jump there is very little improvement in scores throughout the competition.
# - The improvement in script scores on March 23rd leads to an increase in the number of script submissions.
# - After that there is a long period of stagnation where despite 100-200 new versions of scripts appearing every day the scores stay largely the same.
# - In the last eight days of the competition there is a lot of new script versions appearing. Kagglers desperate for some improvement started postprocessing their models' predictions with handcrafted rules. As a result the public LB scores are improving and the private scores do the opposite - a clear case of public LB overfitting.
# - There is a spike in script submissions 8 days before the competition deadline. Here we probably see script submissions playing the role of sample submission, a means for people who haven't yet started working on a competition to make a claim before the first submission deadline.

# In[ ]:


find_most_forked_scripts(competitionId)


# In[ ]:


projectId = competitions.loc[competitionId,'ScriptProjectId']
s = scripts.loc[(scripts.ScriptProjectId==projectId)]
v = scriptversions.loc[scriptversions.ScriptId.isin(s.Id)].set_index('Id')
sub = (submissions.loc[submissions.SourceScriptVersionId.isin(v.index)]
                  .sort_values(by='DateSubmitted'))
v['Nsubmissions'] = sub.groupby('SourceScriptVersionId').size()
v['PublicScore'] = sub.groupby('SourceScriptVersionId')['PublicScore'].agg('first')


# In[ ]:


fig, ax = plt.subplots(figsize=(12,12))
n = 0
for script in [49934, 43840]:
    n += plot_script(script, ax,x=n,vmin=0.841, vmax=v.PublicScore.max())
ax.set_xlim(competitions.loc[competitionId,'DateEnabled'], 
            competitions.loc[competitionId,'Deadline']);
ax.set_ylim(-10,n+10);
ax.set_title('The history of forks of popular Santander scripts',fontsize='large');


# ## Some conclusions
# 
# - Posting of a well-performing script in a competition leads to a multitude of forks and new versions appearing. Variations on the script's model start playing leapfrog game on the Leaderboard.
# - The better scoring forked scripts or versions gain initiative and receive a lot of forks in turn.
# - Appearing of a high scoring script results in an increase in script submissions number, [McScriptface](https://www.kaggle.com/dvasyukova/d/kaggle/meta-kaggle/scripty-mcscriptface-the-lazy-kaggler) seizing his chance of a good submission.
# - This type of crowdsourced model tuning runs a risk of overfitting to the public LB (Santander shakeup).
# - Script submissions seem to be used by people as sample submission - a way to show that they are going to participate in a competition (less submissions in the last week, a spike the day before first entry deadline).
# 
# Too bad we won't see the Mad Scripts Battles of Facebook checkins predictions here =).
# 
# In case of any questions or suggestions regarding this analysis, please leave a comment. If you want to look at other competitions in a similar way - feel free to fork this notebook.
