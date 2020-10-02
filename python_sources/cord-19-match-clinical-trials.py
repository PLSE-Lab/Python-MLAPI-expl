#!/usr/bin/env python
# coding: utf-8

# Here we combine the information about clinical trials gathered in [this great dataset](https://www.kaggle.com/panahi/covid-19-international-clinical-trials) with the CORD-19 dataset by matching trial ids found in the title, abstract or the full text of the papers.

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 200)


# # Load Data

# We use the dataset that I've created in [another notebook](https://www.kaggle.com/danielwolffram/cord-19-create-dataframe). 

# In[ ]:


df = pd.read_csv('/kaggle/input/cord-19-create-dataframe-may-12-update/cord19_df.csv')


# In[ ]:


df.head()


# In[ ]:


ct_df = pd.read_csv('/kaggle/input/covid-19-international-clinical-trials/data/ClinicalTrials.gov_COVID_19.csv')

ct_df.shape, ct_df.columns


# In[ ]:


ict_df = pd.read_csv('/kaggle/input/covid-19-international-clinical-trials/data/ICTRP_COVID_19.csv')

ict_df.shape, ict_df.columns


# In[ ]:


ict_df.head(3)


# # Combine both dataframes

# Let's pick several columns that seem interesting and appear in both files...

# In[ ]:


ict = ict_df[['TrialID', 'web address', 'Study type', 'Study design', 'Intervention', 'Primary outcome']]
ct = ct_df[['NCT Number', 'URL', 'Study Type', 'Study Designs', 'Interventions', 'Outcome Measures']]


# ... and rename them...

# In[ ]:


ict.columns = ['id', 'url', 'study_type', 'study_design', 'intervention', 'outcome']
ct.columns = ['id', 'url', 'study_type', 'study_design', 'intervention', 'outcome']


# ... so we can easily combine them:

# In[ ]:


all_trials = ict.append(ct, ignore_index=True)


# In[ ]:


all_trials.head()


# # Drop Duplicates

# In[ ]:


# all_trials[all_trials.id.duplicated(keep=False)].sort_values('id').head()


# In[ ]:


all_trials.id.duplicated().sum()


# Some trial ids appear in both data sets, we drop the duplicates.

# In[ ]:


all_trials.drop_duplicates(subset='id', keep='last', inplace=True)


# In[ ]:


all_trials.shape


# # Search Trial ID in Papers

# We need to come up with some regular expression to search for trial ids from different sources.

# In[ ]:


all_trials.id.str[:6].value_counts()


# In[ ]:


all_trials.id


# In[ ]:


all_trials.id[all_trials.id.str.startswith('EUCTR')] # use this to see the patterns


# These should work:

# In[ ]:


reg_nct = 'NCT[0-9]{8}'
reg_chi = 'ChiCTR[0-9]{10}'
reg_eu = 'EUCTR[0-9]{4}-[0-9]{6}-[0-9]{2}-[A-Z]{2}'
reg_ir = 'IRCT[0-9]+N[0-9]{1,2}'
reg_isrctn = 'ISRCTN[0-9]{8}'
reg_jprn = 'JPRN-[0-9a-zA-Z]+'
reg_tctr = 'TCTR[0-9]{11}'
reg_actrn = 'ACTRN[0-9]{14}'
reg_drks = 'DRKS[0-9]{8}'
reg_nl = 'NL[0-9]{4}'

registries = [reg_nct, reg_chi, reg_eu, reg_ir, reg_isrctn, reg_jprn, reg_tctr, reg_actrn, reg_drks, reg_nl]

reg = ('|').join(registries)
reg = r'({})'.format(reg)

reg


# Sanity checks

# In[ ]:


pd.Series(['The trial has been registered in Chinese Clinical Trial Registry (ChiCTR2000029981). blabla NCT04275245', 'NCT04275245']).str.findall(reg)


# In[ ]:


all_trials.id.str.extract(reg)[0]


# Great, we don't lose any of the trial ids:

# In[ ]:


(all_trials.id == all_trials.id.str.extract(reg)[0]).sum()


# Now we extract all trial ids that we can find in the title, abstract or text body.

# In[ ]:


trials = (df.title.fillna('') + ' ' + df.abstract.fillna('') + ' ' + df.body_text.fillna('')).str.findall(reg)


# In[ ]:


df['trial_id'] = trials.apply(lambda x: list(dict.fromkeys(x))) # remove multiple occurences


# Total number of trial ids we found:

# In[ ]:


sum([len(x)!=0 for x in trials])


# In[ ]:


# trials.notnull().sum()


# In our papers that are marked as covid-19-papers we found 361 trial ids. (We will see later that not all of them are found in the registered trials)

# In[ ]:


df[[len(x)!=0 for x in trials] & df.is_covid19].shape


# We now want to match the trial ids we found in the papers with the registered trials.

# In[ ]:


all_trials.set_index('id', inplace=True)


# In[ ]:


trials_df = pd.DataFrame(trials, columns=['trials'])


# Only keep trial ids that were found in the trial registries

# In[ ]:


trials_df.trials = trials_df.trials.apply(lambda x: [i for i in pd.Index(x).intersection(all_trials.index)])


# In[ ]:


trials_df.set_index(df.cord_uid, inplace=True)


# In[ ]:


trials_df = trials_df[trials_df.trials.str.len() != 0]


# In[ ]:


all_trials.loc[all_trials.index.intersection(['NCT04276688', 'NCT04319172', 'NCT04319172'])].url.values.tolist()


# In[ ]:


def get_urls(df, trial_ids):
    idx = pd.Index(trial_ids)
    if len(idx) > 0:
        return df.loc[idx].url.values.tolist()
    else:
        return None


# In[ ]:


trials_df['trial_url'] = trials_df.trials.apply(lambda x: get_urls(all_trials, x))


# In[ ]:


# trials_df = trials_df.dropna(subset=['trial_url'])


# In[ ]:


trials_df.loc['00s3wgem']


# In[ ]:


trials_df.head()


# In[ ]:


trials_df.shape


# In[ ]:


len(df.set_index('cord_uid')[df.set_index('cord_uid').is_covid19==True].index.intersection(trials_df.index))


# In[ ]:


df.set_index('cord_uid').loc['00s3wgem'].url


# We found 279 Covid-19-papers that mention registered (Covid-19) trials, some of them appear multiple times. Keep in mind though that not all papers in CORD-19 specifically deal with Covid-19. (For now, we don't consider other registered trials that might appear in non-covid19 papers)

# In[ ]:


df.is_covid19.sum()


# Let's export our dataframe with paper_id + trial ids and urls.

# In[ ]:


trials_df


# In[ ]:


trials_df.to_csv('trial_urls.csv')


# ### Load like this to read in the lists

# In[ ]:


a = pd.read_csv('trial_urls.csv', index_col='cord_uid' , converters={'trial_url': eval})


# In[ ]:


a.head()


# # Exploded Format

# In[ ]:


len(trials_df.explode(column='trials')), len(trials_df.explode(column='trial_url'))


# In[ ]:


trials_exploded = trials_df.explode(column='trials')

trials_exploded.trial_url = trials_df.explode(column='trial_url').trial_url.values

trials_exploded.head()


# In[ ]:


trials_exploded.to_csv('trials_exploded.csv')


# In[ ]:


# def get_trials(df, trial_ids):
#     idx = df.index.intersection(trial_ids)
#     if len(idx) > 0:
#         return df.loc[idx]
#     else:
#         return None

# trials_df['trial_info'] = trials_df.trials.apply(lambda x: get_trials(all_trials, x))

# trials_df = trials_df.set_index(df.paper_id).dropna(subset=['trial_info'])

# trials_df.loc['188e7ff1e260864c89f266b5597de26d69a84660'].trial_info

# trials_df.shape

# trials_df.to_csv('trials_df.csv')

# a = pd.read_csv('trials_df.csv', index_col='paper_id', converters={'trial_info': eval}) # how to load this?

