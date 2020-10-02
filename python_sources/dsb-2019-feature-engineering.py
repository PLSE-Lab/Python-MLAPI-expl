#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('time', '', '#dependencies\nimport numpy as np\nimport pandas as pd\nimport matplotlib.pyplot as plt\nfrom multiprocessing import Pool\n\n#load training data\ntrain_data = pd.read_csv("../input/data-science-bowl-2019/train.csv")\nspecs = pd.read_csv("../input/data-science-bowl-2019/specs.csv")\n#find the range of installation_id\'s wanted\ntarget_population = train_data[train_data[\'installation_id\'].isin(train_data[train_data[\'type\'] == \'Assessment\'][\'installation_id\'])][\'installation_id\'].value_counts()\ntarget_population = target_population[target_population <= 3000][target_population> 49]\n#save these ids for later usage\ntraining_sample_ids = target_population.index\ndel target_population')


# # Exploring Datat Set
# 
# ## How many installation_id's exist?
# 
# The basis for this experiment is to use a target entity history with the application "PBS KIDS Measure Up!", the reason for using the term "entity" over a kid is that we only have a **installation_id** which relates to a single computer entity using their application. So we could have several people using the single **installation_id** we just don't know.
# 
# But that is our starting point so lets explore just the id's too see how many exist

# In[ ]:


train_data['installation_id'].describe()


# So we have 17000 unique `installation_id`'s and we can see that one , `f1c21eda` has over 50,000 activities recorded which seems like a lot given the context of the application being educational and kids between the ages 3-6. So lets take a look at the histogram plot of the number of times we see a `installation_id` and see if that is fact a lot.

# In[ ]:


#get the counts for entity records
entity_counts = train_data['installation_id'].value_counts()
#create a hist plot of values
fig,ax = plt.subplots(2,2,figsize=(15,15))
plt.subplot(2,2,1)
ax = entity_counts.plot(kind="hist",
                        bins= 200,
                        title="F1 : Plot of occuring installation_id's in dataset")
#this seems to be skew heavly to the left with our mega user so lets change the range
plt.subplot(2,2,2)
ax = entity_counts.plot(kind="hist",
                        bins= 200,
                        title="F2 : subset of occuring installation_id's between 0-7000",
                        range=[0,7000])
#still seems to be skewed heavly to the left so lets break it down further
plt.subplot(2,2,3)
ax = entity_counts.plot(kind="hist",
                        bins= 200,
                        title="F3 : subset of occuring installation_id's between 0-3000",
                        range=[0,3000])
plt.subplot(2,2,4)
ax = entity_counts.plot(kind="hist",
                        bins= 200,
                        title="F4 : subset of occuring installation_id's between 0-1000",
                        range=[0,1000])
plt.show()
#finally show the value_counts
print(train_data['installation_id'].value_counts().describe())
del entity_counts


# Now with the given information above and graphs we can see that the count ratio of `installation_id` is heavly skwed to the right with 75% of users having less that 1000 entries and the bottom 25% having less than 7 activities. So we might have to reduce the desired amount of these to include in our training as they might have noise included which was memtioned in the data description for the competition. So lets see how the above information changes when we look at `installation_id`'s that have a assessment peice (our target variable for prediction)

# In[ ]:


target_population = train_data[train_data['installation_id'].isin(train_data[train_data['type'] == 'Assessment']['installation_id'])]
#get the counts for entity records
entity_counts = target_population['installation_id'].value_counts()
#create a hist plot of values
fig,ax = plt.subplots(2,2,figsize=(15,15))
plt.subplot(2,2,1)
ax = entity_counts.plot(kind="hist",
                        bins= 200,
                        title="F5 : Plot of occuring installation_id's in dataset")
#this seems to be skew heavly to the left with our mega user so lets change the range
plt.subplot(2,2,2)
ax = entity_counts.plot(kind="hist",
                        bins= 200,
                        title="F6 : subset of occuring installation_id's between 0-7000",
                        range=[0,7000])
#still seems to be skewed heavly to the left so lets break it down further
plt.subplot(2,2,3)
ax = entity_counts.plot(kind="hist",
                        bins= 200,
                        title="F7 : subset of occuring installation_id's between 0-3000",
                        range=[0,3000])
plt.subplot(2,2,4)
ax = entity_counts.plot(kind="hist",
                        bins= 200,
                        title="F8 : subset of occuring installation_id's between 0-1000",
                        range=[0,1000])
plt.show()
#finally show the value_counts
print(target_population['installation_id'].value_counts().describe())
del target_population,entity_counts


# The whole behaviour and skew for the histogram has changed how that we are looking at `installation_id` that have assessment pieces, it seems that most amount of users, tried an assessment piece without trying anything else ( maybe they are just that good ? ) but the bottom range for 25% of the entity bases has dramatically increased to 481 activities and 75% range now sits at 2320 activities but we still have that crazy 50,000 activity user.
# 
# To get a good representation of id's in my training set I am now only going to look at users with more than 50 activities and less than 3000. Why thoses numbers ? Well anyone with less than 50 activities isn't going to have much of history for us to use to predict I believe ( at this stage ) and having more than 3000 activties seems to be unnatural for the whole population and could introduce some basis into our models if included for these heavly outliers.
# 
# So lets get one more look into `installation_id`'s with our new range.
# * installation_id's include must have the following conditions met
#     *  have at least 50 activities
#     *  have at least one assessment piece in that history
#     *  have no more than 3000 activities to sotp outliers affecting training

# In[ ]:


#find the range of installation_id's wanted
target_population = train_data[train_data['installation_id'].isin(train_data[train_data['type'] == 'Assessment']['installation_id'])]['installation_id'].value_counts()
target_population = target_population[target_population <= 3000][target_population> 49]
#save these ids for later usage
training_sample_ids = target_population.index
#get the counts for entity records
entity_counts = target_population
#create a hist plot of values
fig,ax = plt.subplots(2,2,figsize=(15,15))
plt.subplot(2,2,1)
ax = entity_counts.plot(kind="hist",
                        bins= 200,
                        title="F9 : Plot of occuring installation_id's in dataset")
#this seems to be skew heavly to the left with our mega user so lets change the range
plt.subplot(2,2,2)
ax = entity_counts.plot(kind="hist",
                        bins= 200,
                        title="F10 : subset of occuring installation_id's between 2000-3000",
                        range=[2000,3000])
#still seems to be skewed heavly to the left so lets break it down further
plt.subplot(2,2,3)
ax = entity_counts.plot(kind="hist",
                        bins= 200,
                        title="F11 : subset of occuring installation_id's between 1000-2000",
                        range=[1000,2000])
plt.subplot(2,2,4)
ax = entity_counts.plot(kind="hist",
                        bins= 200,
                        title="F12 : subset of occuring installation_id's between 50-1000",
                        range=[50,1000])
plt.show()
#finally show the value_counts
print(target_population.describe())
del target_population,entity_counts


# Looks like have no crazy frquency for a single count and we have a nice degrading skew to the right and we have only lost 900 unique `installation_id`'s so hopefully that is enough to get a positive value for a our model evulation in the future. But that is only the start, `installation_id`'s are used to group a entities game sessions back to a single entity so the next step to check out how many game sessions a entity has and some descriptive information to use in our model. To find game sessions we have a atrribute called `game_session` which has a unique value.
# 
# ## Game sessions
# 
# So lets use the `installation_id` found in our previous investigations and look into how many game sessions they have.

# In[ ]:


#get the number of unique gamesession per installation id's
target_population = train_data[train_data['installation_id'].isin(training_sample_ids)].groupby(['installation_id']).game_session.nunique()
print(target_population)
print("Stats of game sessions")
print(target_population.describe())
print("how many have less than 5 session?")
print(target_population[target_population < 5].describe())
print(target_population[target_population < 5].head())
#lets check out a histogram of game_sessions
fig,ax = plt.subplots(2,2,figsize=(15,15))
plt.subplot(2,2,1)
target_population.plot(kind="hist",
                       bins=200,
                       title="F13 : Distrubution of gamesession id's")
plt.subplot(2,2,2)
target_population.plot(kind="hist",
                       bins=200,
                       title="F14 : Distrubution of gamesession id's between 0,200",
                       range=[0,200])
plt.subplot(2,2,3)
target_population.plot(kind="hist",
                       bins=200,
                       title="F15 : Distrubution of gamesession id's between 0,100",
                       range=[0,100])
del target_population


# So we have our average number of game sessions peaking around 20 and we have some much more active users having up to 600 game sessions. So there is a good variety of values in this attribute for us to include in the training set, so lets make a couple of variables to represent game_sessions that a entity has. We might want to find the average/min/max duration a entity has for their game sessions, average/min/max events in a game session , the amount of game sessions that this entity has and often a game session has an assessment.
# 
# So lets find these distrubutions and see how they look.
# 
# ## Game Session Durations

# In[ ]:


#get the target population grouped by install and game session
target_population = train_data[train_data['installation_id'].isin(training_sample_ids)].groupby(['installation_id','game_session'])
#get the time duration for a single session (game_time is recorded through out the session so we take the max)
session_duration = ((target_population['game_time'].max()/1000)/60) # in minutes
#only include sessions where session took longer than 1 millisecond
session_duration = session_duration[session_duration > 0]
#plot min,avg,max,std
fig,ax = plt.subplots(2,2,figsize=(15,15))
#plot average
plt.subplot(2,2,1)
session_duration.groupby('installation_id').mean().plot(kind="hist",
                                                        bins=100,
                                                       title="F16 : Histogram of average session durations",
                                                       range=[0,50])
plt.xlabel("duration in minutes")
#plot std
plt.subplot(2,2,2)
session_duration.groupby('installation_id').std().plot(kind="hist",
                                                        bins=100,
                                                       title="F17 : Histogram of std session durations",
                                                       range=[0,50])
plt.xlabel("duration in minutes")
#plot max
plt.subplot(2,2,3)
session_duration.groupby('installation_id').max().plot(kind="hist",
                                                       bins=100,
                                                       title="F18 : Histogram of max session durations",
                                                      range=[0,400])
plt.xlabel("duration in minutes")
#plot min
plt.subplot(2,2,4)
session_duration.groupby('installation_id').min().plot(kind="hist",
                                                       bins=100,
                                                       title="F19 : Histogram of min session durations")
plt.xlabel("duration in minutes")
plt.show()
del session_duration,target_population


# ## Events in game sessions

# The minimun events in seen in all game sessions, doesn't show anything of interest so instead I have investigated the unique events and total events using the mean and max counts.

# In[ ]:


#get the target population grouped by install and game session
target_population = train_data[train_data['installation_id'].isin(training_sample_ids)].groupby(['installation_id','game_session'])['event_id']
#get counts of events and uniques
uniques = target_population.nunique().groupby('installation_id')
counts = target_population.count().groupby('installation_id')
#plot min,avg,max,uniques
fig,ax = plt.subplots(2,2,figsize=(15,15))
#plot average
plt.subplot(2,2,1)
counts.mean().plot(kind="hist",
                   bins=100,
                   title="F20 : Histogram of average events for entity's sessions",
                   range=[0,350])
plt.xlabel("number of installation_id's")
#plot std
plt.subplot(2,2,2)
uniques.mean().plot(kind="hist",
             bins=100,
             title="F21 : Histogram of  average unique events for entity's sessions")
plt.xlabel("number of installation_id's")
#plot max
plt.subplot(2,2,3)
counts.max().plot(kind="hist",
                  bins=100,
                  title="F22 : Histogram of max events for entity's sessions")
plt.xlabel("number of installation_id's")
#plot min
plt.subplot(2,2,4)
uniques.max().plot(kind="hist",
                  bins=20,
                  title="F23 : Histogram of max unique events for entity's sessions")
plt.xlabel("number of installation_id's")
plt.xticks(range(0,26,5))
plt.show()
del target_population,uniques,counts


# So what do we see going on here ? We can see that our average events per normal game session range from 0 to 100 and that our maximun seen for a `installation_id` ranges from  0 to 400 but if we look at the number of unique events per game session, we see a clear range of between 5 to 20 and an average of 4 to 8. Oddly enough the number of unique events seem to be very close to a normal distrubution while the frequency of just events is skewed to the right.
# 
# This could highlight that user tend to like the same event and often repeat that event in a single session and that they will interaction only within one area. But without more information on the topic for these activties it is hard if this is true or not, perphas the events recorded in the dataset are mixed between interaction and a popup showing up.
# 
# But what about our target activities, the assessment types , lets take a look into the same metrics but only including the assessment activities.

# In[ ]:


#get the target population grouped by install and game session, but only for assessment activities
target_population = train_data[train_data['type'] == 'Assessment']
target_population= target_population[target_population['installation_id'].isin(training_sample_ids)].groupby(['installation_id','game_session'])['event_id']
#get counts of events and uniques
uniques = target_population.nunique().groupby('installation_id')
counts = target_population.count().groupby('installation_id')
#plot min,avg,max,uniques
fig,ax = plt.subplots(2,2,figsize=(15,15))
#plot average
plt.subplot(2,2,1)
counts.mean().plot(kind="hist",
                   bins=100,
                   title="F24 : Histogram of average events(assessment) for entity's sessions",
                   range=[0,250])
plt.xlabel("number of installation_id's")
#plot std
plt.subplot(2,2,2)
uniques.mean().plot(kind="hist",
             bins=20,
             title="F25 : Histogram of  average unique events(assessment) for entity's sessions")
plt.xlabel("number of installation_id's")
#plot max
plt.subplot(2,2,3)
counts.max().plot(kind="hist",
                  bins=100,
                  title="F26 : Histogram of max events(assessment) for entity's sessions",
                  range=[0,400])
plt.xlabel("number of installation_id's")
#plot min
plt.subplot(2,2,4)
uniques.max().plot(kind="hist",
                  bins=20,
                  title="F27 : Histogram of max unique events(assessment) for entity's sessions")
plt.xlabel("number of installation_id's")
plt.xticks(range(0,26,5))
plt.show()
del target_population,uniques,counts


# When only looking at the assessed activities we can that our average for the amount per session has kept it shape but in all other  dimensions it has changed (the frequency in all dimensions have changed as well but that is to be expected ). Odd enough though the number of unique assessment that occurs in one sitting is very different to the number of unique events for sitting, it would seem that users are more interested in streaming multiple assessment together rather than replaying similar events.
# 
# ## Game activity levels
# 
# So the final part for this investigations was that we wanted to find out how often a user would start a game session and how many of these would have a assessment item in them. So again lets look into the how many sessions a user has and how often it includes a assessment.
# 

# In[ ]:


#get the target population grouped by install and collect unique game_sessions
counts = train_data[train_data['installation_id'].isin(training_sample_ids)].groupby(['installation_id'])['game_session'].nunique()
#plot min,avg,max,uniques
fig,ax = plt.subplots(1,2,figsize=(15,7.5))
#plot how many game sessions
plt.subplot(1,2,1)
counts.plot(kind="hist",
                   bins=200,
                   title="F28 : Histogram of how many game sessions for each entity")
plt.xlabel("number of installation_id's")
#remake counts to only include assessment game sesisons
counts = train_data[train_data['installation_id'].isin(training_sample_ids)][train_data['type']=='Assessment'].groupby(['installation_id'])['game_session'].nunique()
#plot how many game sessions had assessment
plt.subplot(1,2,2)
counts.plot(kind="hist",
                   bins=25,
                   title="F29 : Histogram of how many game sessions had asessment for each entity",
                   range=[2,25])
plt.xlabel("number of installation_id's")
plt.xticks(range(0,27,2))
del counts


# We can see a sharp drop in the amount of game sessions that have assessment over the amount of session that do,dropping by almost a factor of ten. These numbers might be helpful to understand how we can predict the `accuracy_group` but I believe that looking into the actual assessment peices themselves and finding more information about their previoues is likely to be more useful. 
# 
# ## Investigation Outcomes
# 
# First we found that `3367` unique `installation_id`'s are going to be useful for the exploration of models to suit the needs of the task. Afterwards we looked into the first link in the dataset `game_session` id's and explored how these related to the history of an entity using the application. We looked into how many `game_session`s a entity has, how often they included a assessment piece , how many events a entity would completed in a game session and how many unique events per session , how many unique assement events a entity would attempt and how many assessment events would occur in a game session.
# 
# From this I have concluded that I might be able to use these factors to find some weak relationship with the accuracy group so they have decided to make the following features for a training set.
# 
# * Features included `installation_id` as index :
#     * nu_game_sessions
#     * nu_game_assessment_sessions
#     * avg/max_unique_assessment_in_sessions
#     * avg/max_assessment_in_sessions
#     * avg/max_unique_events_in_sessions
#     * avg/max_events_in_sessions
#     * avg/max/min/std_session_durations
#     * `event_title`'s as column name and how often it appear in the history of the `installation_id` as its value (taken from my early "Feature Engineering")
#     
# So lets build a function to take the given train.csv and turn it into our feature dataset!
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'def simple_features(data,training_set=True):\n    if training_set:\n        #first reduce installation_id\'s down to having at least one assessment and having between 50 and 3000 events\n        target_population = data[data[\'installation_id\'].isin(data[data[\'type\'] == \'Assessment\'][\'installation_id\'])][\'installation_id\'].value_counts()\n        target_population = target_population[target_population <= 3000][target_population> 49]\n        #save these ids for later usage\n        wanted_install_ids = target_population.index\n        #reduce dataset to have only wanted \n        data = data[data[\'installation_id\'].isin(wanted_install_ids)]\n        #free memory\n        del target_population\n    #make a dataframe for our returned new transform\n    feature_set =pd.DataFrame()\n    #find number of game sessions that installation_ids have and add them to the feature set\n    feature_set = pd.concat([feature_set,\n                             data.groupby([\'installation_id\'])[\'game_session\'].nunique()\n                            ])\n    #### Game Sessions\n    feature_set_columns = ["nu_game_sessions"]\n    #find number of games that included a assessment for each installation_id and add them to feature set\n    feature_set = pd.concat([feature_set,\n                             data[data[\'type\']==\'Assessment\'].groupby([\'installation_id\'])[\'game_session\'].nunique()\n                            ],\n                           axis=1)\n    feature_set_columns.append("nu_game_assessment_sessions")\n    #find numbner of average assessment events for each installation_id and add them\n    feature_set = pd.concat([feature_set,\n                             data[data[\'type\']==\'Assessment\'].groupby([\'installation_id\',\'game_session\'])[\'event_id\'].count().groupby(\'installation_id\').mean()\n                            ],\n                           axis=1)\n    #### Game Assessment\n    feature_set_columns.append("avg_assessment_in_sessions")\n    #find numbner of max assessment events for each installation_id and add them\n    feature_set = pd.concat([feature_set,\n                             data[data[\'type\']==\'Assessment\'].groupby([\'installation_id\',\'game_session\'])[\'event_id\'].count().groupby(\'installation_id\').max()\n                            ],\n                           axis=1)\n    feature_set_columns.append("max_assessments_in_sessions")\n    #find numbner of average unique assessment events for each installation_id and add them\n    feature_set = pd.concat([feature_set,\n                             data[data[\'type\']==\'Assessment\'].groupby([\'installation_id\',\'game_session\'])[\'event_id\'].nunique().groupby(\'installation_id\').mean()\n                            ],\n                           axis=1)\n    feature_set_columns.append("avg_unique_assessment_in_sessions")\n    #find numbner of max assessment events for each installation_id and add them\n    feature_set = pd.concat([feature_set,\n                             data[data[\'type\']==\'Assessment\'].groupby([\'installation_id\',\'game_session\'])[\'event_id\'].nunique().groupby(\'installation_id\').max()\n                            ],\n                           axis=1)\n    feature_set_columns.append("max_unique_assessments_in_sessions")\n    #find numbner of average assessment events for each installation_id and add them\n    feature_set = pd.concat([feature_set,\n                             data.groupby([\'installation_id\',\'game_session\'])[\'event_id\'].count().groupby(\'installation_id\').mean()\n                            ],\n                           axis=1)\n    #### Game Events\n    feature_set_columns.append("avg_events_in_sessions")\n    #find numbner of max assessment events for each installation_id and add them\n    feature_set = pd.concat([feature_set,\n                             data.groupby([\'installation_id\',\'game_session\'])[\'event_id\'].count().groupby(\'installation_id\').max()\n                            ],\n                           axis=1)\n    feature_set_columns.append("max_events_in_sessions")\n    #find numbner of average unique assessment events for each installation_id and add them\n    feature_set = pd.concat([feature_set,\n                             data.groupby([\'installation_id\',\'game_session\'])[\'event_id\'].nunique().groupby(\'installation_id\').mean()\n                            ],\n                           axis=1)\n    feature_set_columns.append("avg_unique_events_in_sessions")\n    #find numbner of max assessment events for each installation_id and add them\n    feature_set = pd.concat([feature_set,\n                             data[data[\'type\']==\'Assessment\'].groupby([\'installation_id\',\'game_session\'])[\'event_id\'].nunique().groupby(\'installation_id\').max()\n                            ],\n                           axis=1)\n    feature_set_columns.append("max_unique_events_in_sessions")\n    #### Game Session Durations\n    durations_mins = (data.groupby([\'installation_id\',\'game_session\'])[\'game_time\'].max()/1000)/60\n    #remove zero durations for stats\n    durations_mins = durations_mins[durations_mins > 0]\n    #add average duration for game sessions for installation_ids\n    feature_set = pd.concat([feature_set,\n                             durations_mins.groupby(\'installation_id\').mean()\n                            ],\n                           axis=1)\n    feature_set_columns.append("avg_session_durations")\n    #add min duration for game sessions for installation_ids\n    feature_set = pd.concat([feature_set,\n                             durations_mins.groupby(\'installation_id\').min()\n                            ],\n                           axis=1)\n    feature_set_columns.append("min_session_durations")\n    #add max duration for game sessions for installation_ids\n    feature_set = pd.concat([feature_set,\n                             durations_mins.groupby(\'installation_id\').max()\n                            ],\n                           axis=1)\n    feature_set_columns.append("max_session_durations")\n    #add std duration for game sessions for installation_ids\n    feature_set = pd.concat([feature_set,\n                             durations_mins.groupby(\'installation_id\').std()\n                            ],\n                           axis=1)\n    feature_set_columns.append("std_session_durations")\n    ### Events occured in history\n    ### add occurrences of events_titles for this installation_id\n    feature_set = pd.concat([feature_set,\n                             data.groupby([\'installation_id\',\'title\'])[\'title\'].count().unstack().fillna(value=0)\n                            ],\n                           axis=1)\n    feature_set_columns += list(data.groupby([\'installation_id\',\'title\'])[\'title\'].count().unstack().columns)\n    #add column names to feature set\n    feature_set.columns = feature_set_columns\n    return feature_set')


# Now we have a function we can drop and drag into a submission kernel to transform the train.csv into a feature set for some of the`installation_id`'s and use them to predict the last seen assessment's `accucary_group`. The function takes a pandas dataframe and returns a new one, optional if you want the recreate the training_set selection I have used you can use it as is or if you want to transform the test.csv or just want to change the selection critera yuo can pass the optional parameter `training_set = False` and the function will use all seen `installation_id`'s.
# 
# Of note for usage is that you need to take out the last seen assessment in the data before passing it to this function or you will start to have target leakage (even if its just a bit) in the returned training set.For the `test.csv` you shoudn't have to worry as it will be swapped at runtime with a private dataset which has only activities up to the last activity before the predicted assessment.
# 
# # Targets for training labels
# 
# Now that we have a function which can produce a training set based on the data_set we supply it, we need one last thing before we start to use it, a target variable for the next assessment piece. This target variable is going to the `accuracy_group` for the next assessment piece so it will either be `0,1,2,3` depending on how many attempts were taken to get a correct response. In the notebook's re-run for testing on the leaderboard, the given test.csv will be swaped and it will include `installation_id` sets where the history has been truncated randomally along the journey for this `installation_id` so we need to replicate this process in our training set to build a good model.
# 
# So we need to make a training set for each nth assessment piece a `installation_id` has using only the history before that starting event for that piece of assessment. Then we need to calculate the `accuracy_group` for that assessment, truncate the history before the event and transform the data_set to have our previous found variables plus the `accuracy_group` for a single row in our training set.

# This function takes a group of `event_data` for a single `event_id` and returns the assessment score based on the number of correct and incorrect repsonses.

# In[ ]:


def get_accuracy(desired_group):
    num_correct = desired_group['event_data'].str.contains('"correct":true').sum()
    num_incorrect = desired_group['event_data'].str.contains('"correct":false').sum()
    #decision tree for deciding the accuracy_group of the next seen assessment piece
    if num_correct > 0 :
        if num_incorrect == 0 :
            accuracy = 3
        elif num_incorrect == 1 :
            accuracy = 2
        elif num_incorrect == 2:
            accuracy = 1
        else :
            accuracy = 0
    else :
        accuracy = 0
    return accuracy


# This function here is going to produce a reduced training dataset like we have before with using the whole train.csv but instead we are going to create several entries per `installation_id`. These entries will split the the existing history into smaller subsets before a seen assessment piece much like the test.csv will have in a hope to then use these smaller histories to predict the next assessment outcome.

# In[ ]:


def reduce_group(group,n_assessment):
    ###function to as a group based on installation_id down to its history before n1th seen assessment
    #sort group based on timestamp
    group = group.sort_values("timestamp")
    #find the timestamp of the nth assessment piece
    try :
        #find assessment groups
        assessment_pieces = group[group['type']=='Assessment'][group['event_code'].isin([4100,4110])].groupby(['game_session','event_id'],sort=False)
        #get the nth group
        desired_group= None
        if n_assessment == -1 :
            try : 
                #check to see if a group exists
                if len(assessment_pieces.groups) < 1:
                    desired_group = group.tail(1)
                else:
                    desired_group = assessment_pieces.last()
            except Exception as e:
                print(e)
                return pd.DataFrame(columns=group.columns)
        else:
            for n_group,assessment in enumerate(assessment_pieces):
                if n_group == n_assessment-1:
                    desired_group = assessment[1]
                    break
        #check to see if we found the desired nth assessment group
        if type(desired_group) == type(None):
            #print("installation_id didn't have {} assessment group".format(n_assessment))
            return pd.DataFrame(columns=group.columns)
        #find the first timestamp in the nth assessment group
        if n_assessment == -1:
            timestamp = group['timestamp'].tail(1).values[0]
        else:
            timestamp = desired_group['timestamp'].values[0]
    except IndexError as e:
        #print("Index Error occured while handling history")
        return pd.DataFrame(columns=group.columns)
    #find the correct accucary group
    #Filter the event_codes based on what type of assessment it is
    if desired_group['title'].isin(['Bird Measurer (Assessment)']).any():
        desired_group = desired_group[desired_group['event_code'] == 4110]
    else:
        desired_group = desired_group[desired_group['event_code'] == 4100]
    #check that the group isn't now empty after filtering on type of assessment
    if n_assessment == -1:
        pass
    elif len(desired_group['event_code'].values) < 1:
        #print("group was empty after filtering")
        return pd.DataFrame(columns=group.columns)
    #return group's reduced dataframe
    if not(n_assessment == -1):
        reduced_group = group[group['timestamp'] < timestamp]
    else:
        reduced_group = group
    #add the accuracy_group
    reduced_group['accuracy_group'] = get_accuracy(desired_group)
    if (reduced_group.count() < 1).any():
        print("row had no entries")
    #add how many previous assessment pieces this history has
    if n_assessment == -1:
        reduced_group['previous_assessments'] = len(assessment_pieces)
    else:
        reduced_group['previous_assessments'] = n_assessment - 1
    del desired_group,timestamp,assessment_pieces
    return reduced_group


# Lets make a wrapper that is going to use the simple_feature func and reduce_group, keeping the extra two colums that reduce_group added `accuracy_group` and `previous_assessments`

# In[ ]:


def reduce_and_transform(data,history_length):
    #group by installation_id and then transform
    data = data.groupby(['installation_id'],sort=False)
    data = data.apply(lambda x:reduce_group(x,history_length)).reset_index(drop=True)
    #get additional columns
    target_variable = data.groupby("installation_id",sort=False).apply(lambda x: x['accuracy_group'].unique()[0])
    previous_assessments = data.groupby("installation_id",sort=False).apply(lambda x:x['previous_assessments'].unique()[0])
    #get features
    features = simple_features(data,training_set=False)
    #make DataFrame and set columns
    transform = pd.concat([features,target_variable,previous_assessments],axis=1,sort=False)
    transform.columns = list(features.columns)+["accuracy_group","previous_assessments"]
    del data,target_variable,previous_assessments,features
    return transform


# But firstly, what if I just wanted the get the longest connection possible (important for the test case as we can't have multiple predictions for a `installation_id`) the below snippet shows how to use a full dataset and get one single journey for each `installation_id`

# In[ ]:


get_ipython().run_cell_magic('time', '', 'target_data = train_data\nreduce_and_transform(target_data,-1).fillna(0)')


# Now lets build a full dataset with our predicted value with all history lengths up to 16 (the peak of our previous analysis histogram on unique events with assessment).

# In[ ]:


get_ipython().run_cell_magic('time', '', 'target_data = train_data[train_data[\'installation_id\'].isin(training_sample_ids)]\nfull_data_set = reduce_and_transform(target_data,1)\nprint("Frist pass completed")\n#start compiling other history lengths\nfor length in range(2,17):\n    #get new rows for this length of history\n    full_data_set = pd.concat([reduce_and_transform(target_data,length),full_data_set],sort=False,axis=0)\n    print("{} pass completed".format(length))\nfull_data_set.to_csv("user-interactions.csv")')


# Let us take a quick look at our final results, by checking out the behaviour for our accuracy groups and how many different lengths we have.

# In[ ]:


plt.figure()
ax = full_data_set['accuracy_group'].plot(kind="hist",
                                     title="breakdown of accuracy groups")
plt.show()
plt.figure()
ax = full_data_set['previous_assessments'].plot(kind="hist",
                                     title="breakdown of previous_assessments lengths")
plt.show()


# ## Parameter Check for corelation
# 
# So now that we have our set of features, lets check now well are they performing against the `accuracy_group` of the row and if we have any strongly corelated or assoicated features already for our model to jump on.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from pandas.plotting import scatter_matrix\nprint("starting setup for scatter matrix")\nscatter_list = []\nscatter_number = 0\nfor column in full_data_set.columns:\n    if scatter_number == 0 :\n        scatter_list.append([])\n        scatter_list[-1].append(column)\n        scatter_number += 1\n    elif scatter_number < 5 :\n        scatter_number += 1\n        scatter_list[-1].append(column)\n    else:\n        scatter_list[-1].append(\'accuracy_group\')\n        scatter_number = 1\n        scatter_list.append([])\n        scatter_list[-1].append(column)\n#check that the last group has accuracy_group\nif not(\'accuracy_group\' in scatter_list[-1]):\n    scatter_list[-1].append(\'accuracy_group\')\nprint(scatter_list)\nprint("starting to scatter plot columns")\nfor column_set in scatter_list:\n    plt.figure()\n    target_data = full_data_set[column_set]\n    target_data.columns = range(len(target_data.columns))\n    scatter_matrix(target_data,\n                   alpha=0.3,\n                   figsize= (len(column_set),len(column_set)),\n                   diagonal=\'kde\'\n                  )\n    title = ""\n    for idx,value in enumerate(column_set):\n        title += "{} : {} \\n".format(idx,value)\n    plt.subplot(len(column_set),len(column_set),round(len(column_set)/2))\n    plt.title(title)\n    plt.show()')


# So we don't have any clear assoications with any numerical value in our current set, in most cases we have overlap between all four classes we want to predict. After testing a bit with some baseline models (using basic sklearn models) I could get roughly 0.07 score against the leaderboard. So not great , we have to keep looking for more...

# # Using the event data in the histories

# lets take at some of the attributes we can have in our event_data for events. I want to look for some attributes to agument the individuals with the most recent interactions the user has been involved with.

# When looking into the type column for our events in those with `clip` as type didn't have anything additional in their event data so I have excluded that type, we can't use the assessment as we might not have any but we could add a new column to transform and reduce function at a later date so lets look into Activity and Game and see what unique `event_data` we have to use to make some features to feed our models.

# In[ ]:


train_data['type'].unique()


# So we have four types of events going on

# In[ ]:


unique_entries = []
def extract_uniques(row):
    #adds values to global list returns the unique list in column as it goes
    global unique_entries
    for value in row:
        if not(value in unique_entries):
            unique_entries.append(value)
    return unique_entries

def reset_uniques():
    global unique_entries
    unique_entries = []


# Lets check out all the combinations of `event_data`'s attributes and value when `type == 'Activity'`

# In[ ]:


Activity_attributes = train_data[train_data['installation_id'].isin(training_sample_ids)][train_data['type']=="Activity"]['event_data'].str.findall(r'\"([a-zA-Z_\s]+)\":\"([a-zA-Z_\s]+)\"')
Activity_attributes = Activity_attributes.apply(extract_uniques)
Activity_attributes.values[0]


# So we have ten major keys and a large combination of values. It looks like the keys that are going to be helpful are going to be media_type and maybe identifier if we are luckly as the  others seem to be more around the instances of the activities rather than informative aspects of the activity.

# ### Activity type event_data

# So lets start digging into records that were `type == activity` and the two keys to find features.

# #### media_type key

# So lets look into the amount of activities that had a media component in the history and how we could extract them out. Below I am going to look at the media_type of activity if it has that attribute as it might relate to the user reading or watching some informative stuff that will help with assessment.

# In[ ]:


media_type = train_data[train_data['installation_id'].isin(training_sample_ids)]    [train_data['type']=="Activity"]    [['event_data','installation_id']]    .groupby("installation_id")    .apply(lambda x:x['event_data'].str.findall(r'\"media_type":\"([a-zA-Z_\s]+)\"')).astype(str)    .reset_index()    .groupby(['installation_id','event_data'])    .count()    .unstack()    .fillna(value=0)
media_type.columns = media_type.columns.droplevel()
media_type = media_type.drop(['[]'],axis=1)
new_columns = []
#clean up columns
for value in media_type.columns:
    new_columns.append(str(value).replace("[","")                                   .replace("]","")                                   .replace("'","")                                   + "_event"
                      )
media_type.columns = new_columns
media_type.columns.name = "type of media act"
#lets check out our frame
print(media_type.head(25))


# In[ ]:


#lets take a look at the frequency maps of these events
for column in media_type.columns:
    plt.figure()
    media_type[column].plot(kind="hist",
                                      title=column,
                                      bins=100)
    plt.show()


# ####  identifier key

# In[ ]:


identifier_type = train_data[train_data['installation_id'].isin(training_sample_ids)]    [train_data['type']=="Activity"]    [['event_data','installation_id']]    .groupby("installation_id")    .apply(lambda x:x['event_data'].str.findall(r'\"identifier":\"([a-zA-Z_\s]+)\"')).astype(str)    .reset_index()    .groupby(['installation_id','event_data'])    .count()    .unstack()    .fillna(value=0)
identifier_type.columns = identifier_type.columns.droplevel()
identifier_type = identifier_type.drop(['[]'],axis=1)
new_columns = []
#clean up columns
for value in identifier_type.columns:
    new_columns.append(str(value).replace("[","")                                   .replace("]","")                                   .replace("'","")                                   .replace("event","")                                   + "_trigger"
                      )
identifier_type.columns = new_columns
identifier_type.columns.name = "type of identifier act"
#lets check out our frame
identifier_type.describe()


# In[ ]:


#lets take a look at the frequency maps of these events
for column in identifier_type.columns:
    plt.figure()
    identifier_type[column].plot(kind="hist",
                                      title=column,
                                      bins=100)
    plt.show()


# I am only to take a handful of these triggers from the event_data, mainly due to the following have distrubutions outside of zero or the trigger related to a positive feedback to the user.

# In[ ]:


maybe_useful = ['Dot_GreatJob_trigger','Dot_FillItUp_trigger','Dot_DragMoldPlace_trigger','Dot_AllDoneTapThis_trigger','Dot_SoCool_trigger',
                'addToYourCollection_trigger','Buddy_Incoming_trigger','Dot_Amazing_trigger','Dot_TrySomethingNew_trigger','Dot_TryWall_trigger',
               'andItsFull_trigger','niceJob_trigger','ohWow_trigger','wowSoCool_trigger','thatLooksSoCool_trigger']
identifier_type = identifier_type[maybe_useful]
identifier_type.describe()


# ### Clip type event_data

# In[ ]:


train_data[train_data['installation_id'].isin(training_sample_ids)][train_data['type']=="Clip"]['event_data'].unique()


# So `type == 'Clip'` doesn't have anything we can get  out of besides seeing how many `Clip` events we have so I will make a little code template below to get that out

# In[ ]:


clips_per_user = train_data[train_data['installation_id'].isin(training_sample_ids)]                            [train_data['type']=="Clip"]                            [['event_data','installation_id']]                            .groupby("installation_id")                            .count()
clips_per_user.columns = ['clips_seen']
clips_per_user.head(5)


# ### Game type event_data
# 
# So now lets do the same for game types to find out if they have any useful event_data extracts we can do

# In[ ]:


train_data[train_data['installation_id'].isin(training_sample_ids)][train_data['type']=="Game"]['event_data'].unique()


# Just from the brief look at the unique values for the event_data, it seems that we may have some more useful attributes included in the `event_data`

# In[ ]:


Activity_attributes = train_data[train_data['installation_id'].isin(training_sample_ids)][train_data['type']=="Game"]['event_data'].str.findall(r'\"([a-zA-Z_\s]+)\":\"([a-zA-Z_\s]+)\"')
Activity_attributes = Activity_attributes.apply(extract_uniques)
Activity_attributes.values[0]


# So looking at this we have a couple of key that are interesting to investigate.Those keys that I am going to explore are `'media_type','identifier','movie_id','toy_earned','exit_type'`
# 
# #### media_type

# In[ ]:


media_type = train_data[train_data['installation_id'].isin(training_sample_ids)]    [train_data['type']=="Game"]    [['event_data','installation_id']]    .groupby("installation_id")    .apply(lambda x:x['event_data'].str.findall(r'\"media_type":\"([a-zA-Z_\s]+)\"')).astype(str)    .reset_index()    .groupby(['installation_id','event_data'])    .count()    .unstack()    .fillna(value=0)
media_type.columns = media_type.columns.droplevel()
media_type = media_type.drop(['[]'],axis=1)
new_columns = []
#clean up columns
for value in media_type.columns:
    new_columns.append(str(value).replace("[","")                                   .replace("]","")                                   .replace("'","")                                   + "_event"
                      )
media_type.columns = new_columns
media_type.columns.name = "type of media act"
#lets check out our frame
print(media_type.head(25))


# In[ ]:


#lets take a look at the frequency maps of these events
for column in media_type.columns:
    plt.figure()
    media_type[column].plot(kind="hist",
                                      title=column,
                                      bins=100)
    plt.show()


# The `animation_event` when `type == 'Game'` seems to have a nice distrubution that spreads over all the records so this feature might hold  some useful assoications when  building our models in the future but `audio_event` is  mainly not existing in the records we have.

# #### identifier

# In[ ]:


identifier_type = train_data[train_data['installation_id'].isin(training_sample_ids)]    [train_data['type']=="Game"]    [['event_data','installation_id']]    .groupby("installation_id")    .apply(lambda x:x['event_data'].str.findall(r'\"identifier":\"([a-zA-Z_\s]+)\"')).astype(str)    .reset_index()    .groupby(['installation_id','event_data'])    .count()    .unstack()    .fillna(value=0)
identifier_type.columns = identifier_type.columns.droplevel()
identifier_type = identifier_type.drop(['[]'],axis=1)
new_columns = []
#clean up columns
for value in identifier_type.columns:
    new_columns.append(str(value).replace("[","")                                   .replace("]","")                                   .replace("'","")                                   .replace("event","")                                   + "_trigger"
                      )
identifier_type.columns = new_columns
identifier_type.columns.name = "type of identifier act"
#lets check out our frame
identifier_type.describe()


# In[ ]:


#lets take a look at the frequency maps of these events
for column in identifier_type.columns:
    plt.figure()
    identifier_type[column].plot(kind="hist",
                                      title=column,
                                      bins=100)
    plt.show()


# Look at the histogram plots of all the `identifier` event_attributes across our records, we don't seen any particular value standing out at all. In most cases the value is rarely  used by more than 500 unique installation uniques more than once. If we build any models off these attributes we are  going to have some huge swings as in most cases it  will just be a zero feild with very little range to make any sort of decision so I am going to drop this key for when `type == 'Game'`.
# 
# #### movie_id

# In[ ]:


movie_type = train_data[train_data['installation_id'].isin(training_sample_ids)]    [train_data['type']=="Game"]    [['event_data','installation_id']]    .groupby("installation_id")    .apply(lambda x:x['event_data'].str.findall(r'\"movie_id":\"([a-zA-Z_\s]+)\"')).astype(str)    .reset_index()    .groupby(['installation_id','event_data'])    .count()    .unstack()    .fillna(value=0)
movie_type.columns = movie_type.columns.droplevel()
if '[]' in movie_type.columns:
    movie_type = movie_type.drop(['[]'],axis=1)
new_columns = []
#clean up columns
for value in movie_type.columns:
    new_columns.append(str(value).replace("[","")                                   .replace("]","")                                   .replace("'","")                                   + "_movie"
                      )
movie_type.columns = new_columns
movie_type.columns.name = "type of movie"
#lets check out our frame
movie_type.head(25)


# In[ ]:


#lets take a look at the frequency maps of these events
for column in movie_type.columns:
    plt.figure()
    movie_type[column].plot(kind="hist",
                                      title=column,
                                      bins=100)
    plt.show()


# This one is tough, it seems  like thoses with the prefix `scrubadub` are linked to a particular type of game on the app while the  `intro/outro` solo prefix might be more general so lets merge this down to just an intro and outro movie feild and see what the distrubution is like.

# In[ ]:


movie_type['Sum_intro_movie'] = 0
movie_type['Sum_outro_movie'] = 0
for column in movie_type.columns:
    if 'intro' in  column or 'Intro' in  column :
        movie_type['Sum_intro_movie'] += movie_type[column]
    elif 'outro' in column or 'Outro' in column:
        movie_type['Sum_outro_movie'] += movie_type[column]
        
plt.figure()
movie_type['Sum_intro_movie'].plot(kind="hist",
                                   title="Summarised  intro  movie",
                                   bins=50)
plt.show()
plt.figure()
movie_type['Sum_outro_movie'].plot(kind="hist",
                                   title="Summarised  intro  movie",
                                   bins=50)
plt.show()


# # What is next ?
# 
# Now that I have some descriptive information points about the history of the `installation_id`'s activities and game sessions, I should get some quantitive information points about assessments in `game_session`'s or even just generally across all seen `game_session`'s. Then maybe even drill it down to the type of the assessment as that list seems to very small of about 5 `event_code`'s from looking around at other notebooks and the data itself.
# 
# But more of that soonish....

# # Previous work
# 
# Below is unpolish work that I was using to create training labels and assessment score outcomes for a basemodel for submissions.

# In[ ]:


get_ipython().run_cell_magic('time', '', '#get the previous history for a user \nuser_labels = []\nsort_data = train_data\nassessment = sort_data[sort_data[\'type\'] == \'Assessment\'][sort_data[\'event_code\'].isin([4100,4110])].groupby([\'game_session\',\'title\',\'event_id\'])\n#loop through groups of users,game session and assement pieces\nfor key,group in assessment:\n    #create new row\n    user_labels.append({})\n    user_labels[-1]["game_session"] = key[0]\n    user_labels[-1]["installation_id"] = group[\'installation_id\'].unique()\n    user_labels[-1]["title"] = group[\'title\'].astype(str).unique()[0]\n    user_labels[-1]["num_correct"] = group[\'event_data\'].str.contains(\'"correct":true\').sum()\n    user_labels[-1]["num_incorrect"] = len(group[\'event_data\']) - group[\'event_data\'].str.contains(\'"correct":true\').sum()\n    user_labels[-1]["accuracy"] = user_labels[-1]["num_correct"]/len(group[\'event_data\'])\n    #label the activity\n    if user_labels[-1]["num_correct"] > 0:\n        if len(group[\'event_data\']) < 2:\n            user_labels[-1]["accuracy_group"] = 3\n        elif len(group[\'event_data\']) < 3:\n            user_labels[-1]["accuracy_group"] = 2\n        else:\n            user_labels[-1]["accuracy_group"] = 2\n    else:\n        user_labels[-1]["accuracy_group"] = 0\n#make a full dataframe for all users\nassemented_activties = pd.DataFrame(data=user_labels)\nassemented_activties = assemented_activties.sort_values(\'installation_id\')\nassemented_activties.to_csv("assessed_activities.csv")')


# In[ ]:


get_ipython().run_cell_magic('time', '', '#setup data for test run with simple model\nlabels = pd.read_csv("../input/data-science-bowl-2019/train_labels.csv")\n#group by install_id\ngrouped_outcomes = labels.groupby("installation_id")\n#collect the last accuracy_group for this install_id\noutcomes = []\nfor install_id,group in grouped_outcomes:\n    outcomes.append({})\n    outcomes[-1]["installation_id"] = str(install_id)\n    outcomes[-1]["accuracy_group"] = group[\'accuracy_group\'].values[-1]\n#make y data\ny = pd.DataFrame(data=outcomes)\ny = y.set_index("installation_id")\ny.to_csv("user-outcomes.csv")\ndel labels')

