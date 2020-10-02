#!/usr/bin/env python
# coding: utf-8

# # Path Naive Bayes Model

# # Problem Formulation

# We are predicting an accuracy group for students on a given assessment using the event history of their interaction with the PBS KIDS Measure Up! app.  Taking inspiration from the beautiful kernels from [J Hogg](https://www.kaggle.com/c/data-science-bowl-2019/discussion/123102) and [Buffalo Spdwy](https://www.kaggle.com/barnwellguy/eye-candy-player-activity-sequence-visualization), let's assume that a student's progression from module to module within the app follows a Markov process such that the probability that they will move to module $x$ at time $t$ only depends on which module the completed at time $t-1$.  Then we can compute the probability of a specific sequence of modules (or the student's path through the website) as:

# $$P( \boldsymbol{x})=P(x_0,x_1,\ldots,x_T) = P(x_0)\displaystyle\prod_{t=1}^{T} P(x_t | x_{t-1})$$

# One hypothesis to test is whether students that perform better on the assessments take different paths than those from other accuracy groups.  To do so, let's now condition the transition probabilties also on the student's accuracy group.  This would then allow us to calculate a posterior probability of a student acheiving an accuracy group of $y$ given a path history $ \boldsymbol{x}$.  As a type of Naive Bayes classifier, we would estimate that the student's accuracy group would be the one with the largest posterior probability based on their path through the site.

# $$\arg\max_y P(y| \boldsymbol{x}) \propto P(y)P(x_0|y)\displaystyle\prod_{t=1}^{T} P(x_t | x_{t-1}, y)$$

# *Aside:* We should also note that this interpretation also lends itself well to a Hidden Markov Model as well allowing for a student's latent accuracy group potential to evolve over time (using the module type as the observations and accuracy group potential at time step $t$ as the hidden state).

# The benefits of this model are that:
# * It is simple
# * It is fast
# * It is deterministic
# * There are no paramaters to train, and only one hyperparameter
# * It is very interperatable
#   * We can see which paths through the program lead to the highest accuracy groups
#   * The transition probabilities of the highest peforming students could be used by the app to help suggest the next module for students.

# # Key Assumptions

# ### Model
# 
# * Student behavior exhibits the Markov property
# * Students in different accuracy group have different transition probabilities
# * The students' accuracy groups are strongly correlated with the type of assessment to be graded
#   * We will construct distinct models for each assessment type
# * This kernel is only for a demonstration of a simple sequence model so there is limited model evaluation, i.e., I'm not doing any CV
# 
# ### Data
# To make the data manageable, we will make some strong assumptions in formatting a data for predicition.
# 
# * Let's define all of the events between each assessment as an "episode"
# * We will only calculate the path probabilities for each episode
#   * We will ignore all path data prior to the most recent assessment before the predicted assessment
# * For simplicity, we will only consider each `game_session` as a state
#   * An alternative is to use all of the individual events as states

# # Potential Drawbacks

# Some key drawbacks with this approach could be:
# * By limiting the predictions to just each episode, we lose the benefit of all event history data prior to the last assessment
#   * Has the student attempted the assessment title under observation?
#   * If so, how many times and how did they perform?
#   * Has the student attempted any other assessment titles?
#   * If so, how many times and how did they perform?
#   * How much experience does the student have with using the site?
# * By enforcing the Markov property on model, it will have a short memory
#   * Do transition probabilities depend only on the current state?
#   * How important is what the student was doing two states ago to their next state?
#   * ...or all states since account creation?
# * By only looking at the `game_session` titles, we lose the benefit of using their behavior within each module as a feature.
#   * How deeoly did the student engage in the games?
#   * How does the student perform in the activities?
#   * How long does the student spend listening to instruction?
# 
# These drawbacks point to potential avenues for improving a model of this type in the future.

# # Data Preparation

# Now for the good stuf, the code...

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


DATA_DIR='/kaggle/input/data-science-bowl-2019'


# In[ ]:


train = pd.read_csv(os.path.join(DATA_DIR,'train.csv'))
test = pd.read_csv(os.path.join(DATA_DIR,'test.csv'))
train_labels = pd.read_csv(os.path.join(DATA_DIR,'train_labels.csv'))


# In[ ]:


# Recreate the train_labels.csv file for episodes in the training data

def extract_accuracy_group(df: pd.DataFrame) -> pd.DataFrame:
    # Regex strings for matching Assessment Types
    assessment_4100 = '|'.join(['Mushroom Sorter',
                                'Chest Sorter',
                                'Cauldron Filler',
                                'Cart Balancer'])
    assessment_4110 = 'Bird Measurer'
    
    # 1. Extract all assessment scoring events
    score_events = df[((df['title'].str.contains(assessment_4110)) & (df['event_code']==4110)) |                      ((df['title'].str.contains(assessment_4100)) & (df['event_code']==4100))]
    
    # 2. Count number of correct vs. attempts
    # 2.a. Create flags for correct vs incorrect
    score_events['num_correct'] = 1
    score_events['num_correct'] = score_events['num_correct'].where(score_events['event_data'].str.contains('"correct":true'),other=0)
    
    score_events['num_incorrect'] = 1
    score_events['num_incorrect'] = score_events['num_incorrect'].where(score_events['event_data'].str.contains('"correct":false'),other=0)
    
    # 2.b. Aggregate by `installation_id`,`game_session`,`title`
    score_events_sum = score_events.groupby(['installation_id','game_session','title'])['num_correct','num_incorrect'].sum()
    
    # 3. Apply heuristic to convert counts into accuracy group
    # 3.a. Define heuristic
    def acc_group(row: pd.Series) -> int:
        if row['num_correct'] == 0:
            return 0
        elif row['num_incorrect'] == 0:
            return 3
        elif row['num_incorrect'] == 1:
            return 2
        else:
            return 1
        
    # 3.b. Apply heuristic to count data
    score_events_sum['accuracy_group'] = score_events_sum.apply(acc_group,axis=1)
    
    return score_events_sum.reset_index()


# In[ ]:


test_labels = extract_accuracy_group(test)


# These functions are used to group the events and game sessions into episodes.  Each episode takes on an `episode_session` which the is `game_session` of the assessment taken at the end of episode.

# In[ ]:


def build_episode_game_sessions(df):
    return pd.DataFrame(index=df['game_session'])

def build_starts_only(df):
#     return df[df['event_code']==2000]
    return df.groupby(['installation_id','game_session']).last()

def build_start_end_times(df, df_labels):
    start_end_times = pd.merge(left=df, right=df_labels,left_on='game_session', right_index=True)        .groupby(['installation_id','game_session'])        .first()['timestamp']
    start_end_times = start_end_times.reset_index().sort_values(by=['installation_id','timestamp'])
    start_end_times.columns = ['installation_id','episode_session','end_time']
    start_end_times['start_time'] = start_end_times.groupby('installation_id')['end_time'].shift(1,fill_value='2018-09-11T18:56:11.918Z')
    return start_end_times

def append_times_to_labels(labels, start_end_times):
    new_labels = pd.merge(left=labels,
                          right=start_end_times,
                          left_on=['game_session','installation_id'],
                          right_on=['episode_session','installation_id'])
    return new_labels.drop('game_session',axis=1)

def add_labels_to_sessions(sessions, labels_with_times):
    outer = pd.merge(left=sessions.reset_index(),
                     right=labels_with_times,
                     left_on='installation_id',
                     right_on='installation_id',
                     suffixes=('','_episode') )
    labeled_sessions = outer[(outer['timestamp']>=outer['start_time']) & (outer['timestamp']<=outer['end_time'])]
    
    labeled_session_ids = pd.DataFrame(index=labeled_sessions['game_session'],
                                       data=np.ones(len(labeled_sessions)),
                                       columns=['has_label'])
    unlabeled_sessions = pd.merge(sessions.reset_index(),labeled_session_ids,how='left',left_on='game_session',right_index=True)
    unlabeled_sessions = unlabeled_sessions[unlabeled_sessions['has_label']!=1].drop('has_label',axis=1)
    return labeled_sessions, unlabeled_sessions

def build_session_labels(events, labels):
    episodes_game_sessions = build_episode_game_sessions(labels)
    starts = build_starts_only(events)
    start_end_times = build_start_end_times(starts, episodes_game_sessions)
    labels_with_times = append_times_to_labels(labels, start_end_times)
    return add_labels_to_sessions(starts, labels_with_times)


# ## Create consolidated labeled session Dataframes

# In[ ]:


train_bm_starts, train_no_asmt = build_session_labels(train, train_labels)


# We should note that we are using all of the labeled episodes from both the training and test sets.

# In[ ]:


test_bm_starts, test_to_predict = build_session_labels(test, test_labels)
bm_starts = pd.concat([train_bm_starts, test_bm_starts], sort=True)


# ## Convert Strings to Indices

# In[ ]:


idx_to_titles = list(set(bm_starts['title'])) + ['<START>','<END>']
titles_to_idx = {title: idx for idx, title in enumerate(idx_to_titles)}
bm_starts['title_idx'] = bm_starts['title'].map(titles_to_idx)


# In[ ]:


idx_to_asmt = list(set(bm_starts['title_episode']))
asmt_to_idx = {title: idx for idx, title in enumerate(idx_to_asmt)}
bm_starts['asmt_idx'] = bm_starts['title_episode'].map(asmt_to_idx)


# In[ ]:


asmt_to_idx


# ## Create Session Series

# Let's collect all of the game sessions into a list for each episode.

# In[ ]:


bm_session_columns = ['installation_id','episode_session',
        'accuracy_group', 'asmt_idx']
bm_episodes = bm_starts.groupby(bm_session_columns)['title_idx'].aggregate(lambda x: [len(idx_to_titles)-2] + list(x) ).reset_index()


# In[ ]:


bm_episodes.head()


# ## Build Probability Tables

# Here we calculate the transition probabilities by assessment and by accuracy group (assuming that the `accuracy group` of the `episode_session` is pertinent to our prediction).  It should be noted that we use Laplace smoothing to help with rare transitions - as such our one hyperparameter is this pseudo count, or how much smoothing we should use.

# In[ ]:


num_asmt, num_scores, num_titles = len(idx_to_asmt), 4, len(idx_to_titles)
transition_matrix = np.ones((num_asmt, num_scores, num_titles, num_titles),dtype=np.float32)

asmt_idx = list(bm_episodes['asmt_idx'])
accuracy_group = list(bm_episodes['accuracy_group'])
paths = list(bm_episodes['title_idx'])

for asmt, acc, path in zip(asmt_idx, accuracy_group, paths):
    for prior, current in zip(path[:-1],path[1:]):
        transition_matrix[asmt, acc, prior, current] += 1
        
transition_matrix = transition_matrix / transition_matrix.sum(axis=-1).reshape(*transition_matrix.shape[:-1],1)

He we calculate a prior probability of a given `accuracy_group` for each assessment type.
# In[ ]:


def build_priors(sessions):
    by_asmt_by_acc = sessions.groupby(['asmt_idx','accuracy_group'])['episode_session'].count().unstack(-1).values
    by_asmt = np.sum(by_asmt_by_acc,axis=-1).reshape(-1,1)
    return by_asmt_by_acc/by_asmt

priors = build_priors(bm_episodes)


# # Define Model

# In[ ]:


from sklearn.metrics import cohen_kappa_score

class PathNaiveBayes:
    def __init__(self, priors_, transitions_, start_, end_):
        self.priors = priors_
        self.transitions = transitions_
        self.start = start_
        self.end = end_
        
    def predict(self, asmt, seq):
        # Check if the provide sequence has a start code
        if seq[0] != self.start:
            seq = [self.start] + seq + [self.end]
            
        # Use sum of log probabilities instead of multiply probabilities directly to help avoid underflow
        
        # Initialize with our prior probability
        log_prob = np.log(self.priors[asmt])
        
        # Calculate the Markov chain probability of the user's path
        for prev, current in zip(seq[:-1],seq[1:]):
            log_prob += np.log(self.transitions[asmt,:,prev,current])
            
        # Our prediction is the accuracy score with the highest posterior log probability
        return {"prediction": np.argmax(log_prob), "probabilities": log_prob }
    
    def evaluate(self, asmts, seqs, scores, verbose=False):
        correct, total = np.zeros((self.priors.shape[0])), np.zeros((self.priors.shape[0]))
        preds = []
        for asmt, seq, score in zip(asmts, seqs, scores):
            pred = self.predict(asmt, seq)['prediction']
            if pred == score:
                correct[asmt] += 1
            total[asmt] += 1
            preds.append(pred)
            if verbose:
                print(f"Asmt:{asmt} True:{score} Pred:{pred}")
        acc_per_asmt = correct/total
        total_acc = np.sum(correct)/np.sum(total)
        kappa = cohen_kappa_score(preds,scores,weights='quadratic')
        
        return {'predictions': preds,
               'accuracy_by_asmt': acc_per_asmt,
               'accuracy': total_acc,
               'kappa': kappa}


# In[ ]:


model = PathNaiveBayes(priors,transition_matrix, titles_to_idx['<START>'], titles_to_idx['<END>'] )


# # Make Predictions

# In[ ]:


## Results on Training Data


# In[ ]:


results = model.evaluate(asmts=bm_episodes['asmt_idx'],
               scores=bm_episodes['accuracy_group'],
              seqs=bm_episodes['title_idx'])
results.pop('predictions')
print(results)


# ### Build prediction - final assessment tables

# In[ ]:


test.set_index(['installation_id','timestamp'])
last_entries = test.assign(rn=test.sort_values(['timestamp'], ascending=False)            .groupby(['installation_id'])            .cumcount() + 1)            .query('rn == 1')            .sort_values(['installation_id'])
last_entries[['installation_id','title']].to_csv('test_final_title.csv',index=False)


# In[ ]:


submission = pd.read_csv('test_final_title.csv')
submission['asmt_idx'] = submission['title'].map(asmt_to_idx)
submission_asmt = pd.DataFrame(index=submission['installation_id'], data=submission['asmt_idx'].values, columns=['asmt_idx'])


# ## Get last prediction episode

# We assume that all events not related to a labeled episode belong to the episode under evaluation.

# In[ ]:


test_to_predict['title_idx'] = test_to_predict['title'].map(titles_to_idx)
test_to_predict = pd.merge(left=test_to_predict,right=submission_asmt,left_on='installation_id',right_index=True)
test_to_predict


# In[ ]:


test_session_columns = ['installation_id','asmt_idx']
test_episodes = test_to_predict.groupby(test_session_columns)['title_idx'].aggregate(lambda x: [len(idx_to_titles)-2] + list(x) ).reset_index()
test_episodes


# In[ ]:


def predict_row(row):
    return model.predict(row['asmt_idx'],row['title_idx'])['prediction']

test_episodes['accuracy_group'] = test_episodes.apply(predict_row,axis=1)
test_episodes


# In[ ]:


test_episodes[['installation_id','accuracy_group']].to_csv('submission.csv',index=False)


# In[ ]:


test_episodes.groupby(['asmt_idx','accuracy_group'])['installation_id'].count().unstack(-1)


# # Conclusion

# This model only performs slightly better in terms of a QWK than a baseline of using only the prior probabilities by assessment type.  While the model's Dory problem (i.e., short memory) should be addressed by using longer portions of the user's history, there appears to be some incremental predictive power to only analyzing the probability of the path immediately prior to the assessment to be taken.
