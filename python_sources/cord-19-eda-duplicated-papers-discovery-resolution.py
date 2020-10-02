#!/usr/bin/env python
# coding: utf-8

# # Intro
# In this notebook I'd like to focus on discovering and cleaning duplicated papers in the dataset, in hope to help avoid garbage-in-garbage-out scenario for all the awesome analytics being developed by our community.

# # Plan
# 1. [Libraries](#Libraries)
# 2. [Discovering and cleaning duplicated papers](#Discovering-and-cleaning-duplicated-papers)
#   * [Using *text* field](#Using-text-field)
#   * [Using *abstract* field](#Using-abstract-field)
#   * [Using *affiliations* field](#Using-affiliations-field)
#   * [Using *authors* field](#Using-authors-field)
#   * [Using *title* field](#Using-title-field)
# 3. [Summary of duplicates cleaning](#Summary-of-duplicates-cleaning)
#   * [How the discovered duplicates were processed](#How-the-discovered-duplicates-were-processed)
#   * [Counts of removed duplicates per source dataset](#Counts-of-removed-duplicates-per-source-dataset)
#   * [Counts of merged duplicates per source dataset](#Counts-of-merged-duplicates-per-source-dataset)
# 4. [Storing the results for further reuse](#Storing-the-results-for-further-reuse)

# # Libraries

# In[ ]:


import os
import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import time
from IPython.display import display
pd.options.display.max_colwidth = 120
get_ipython().run_line_magic('matplotlib', 'inline')


# # Discovering and cleaning duplicated papers
# I'm going to leverage output of the nice [xhlulu's extraction work](https://www.kaggle.com/xhlulu/cord-19-eda-parse-json-and-generate-clean-csv).

# In[ ]:


input_root_dir = '../input/cord-19-eda-parse-json-and-generate-clean-csv'
input_files_list = [f for f in os.listdir(input_root_dir) if os.path.splitext(f)[1] == '.csv']
df_list = []
for input_file in input_files_list:
    df_next = pd.read_csv(os.path.join(input_root_dir, input_file))
    df_next['source_dataset'] = input_file
    df_list.append(df_next)
df_all_papers = pd.concat(df_list).reset_index(drop=True)


# In[ ]:


df_all_papers.head(3)


# In[ ]:


df_all_papers.describe()


# In[ ]:


print(f"Number of missing texts: {df_all_papers['text'].isnull().sum()}")


# No paper has missing text, it's great.

# In[ ]:


print(f"Number of missing titles: {df_all_papers['title'].isnull().sum()}")


# In[ ]:


print(f"Number of missing authors: {df_all_papers['authors'].isnull().sum()}")


# By comparing `count` and `unique` we can see symptoms of duplicated entries. However, `unique` counts differ for different fields. Based on the main `text` field it seems like if almost all papers were unique. Yet, the `title` and `authors` fields suggest otherwise. 
# 

# ## Using *text* field
# First, let's drop duplicated `text`.

# In[ ]:


source_datasets_of_duplicates = list(df_all_papers.loc[df_all_papers['text'].duplicated(keep=False), 'source_dataset'].values)
duplicated_text_bool_idx = df_all_papers['text'].duplicated()
df_all_papers = df_all_papers[np.logical_not(duplicated_text_bool_idx)]
print(f'{sum(duplicated_text_bool_idx)} papers with duplicated text were removed.')


# Let's keep track of the performed changes.

# In[ ]:


removed_duplicates_total_count = 10


# Ideally, out of the remaining duplicates we'd like to leave the most informative entries. E.g., if a duplicated paper has entries with and without abstract, we'd prefer to leave the one with abstract.

# ## Using *abstract* field

# In[ ]:


missing_abstract_bool_idx = df_all_papers['abstract'].isnull()
print(f"Number of missing abstracts: {missing_abstract_bool_idx.sum()}")


# We can see that quite a few papers have abstract missing. Let's check groups of papers with duplicated abstracts. Let's count words in the `text` field of the papers.

# In[ ]:


indices = df_all_papers.loc[~df_all_papers['abstract'].isnull() & df_all_papers['abstract'].duplicated(keep=False),
                  'abstract'].sort_values().index

def get_word_count(text):
    # when missing
    if isinstance(text, float):
        return 0
    return len(text.replace('\n', ' ').split())
    
def check_duplicates_by_sorted_indices(indices, duplication_field, max_number_of_groups=50, max_group_size=20, return_shown_indices=False):
    cols = ['title', 'authors', 'abstract', 'text']
    if not duplication_field in cols:
        cols += [duplication_field]
    duplicates_to_check = df_all_papers.loc[indices, cols]    
    duplication_field_i = duplicates_to_check.columns.to_list().index(duplication_field)    
    row_i = 0
    duplication_count = 0
    group_i = 0
    if return_shown_indices:
        indices_shown = []
    while row_i + duplication_count + 1 < len(duplicates_to_check) and group_i < max_number_of_groups:
        while (row_i + duplication_count + 1 < len(duplicates_to_check) and 
               duplicates_to_check.iloc[row_i, duplication_field_i] == duplicates_to_check.iloc[row_i + duplication_count + 1, duplication_field_i]):
            duplication_count += 1 
        group_i += 1
        if duplication_count + 1 > max_group_size:
            continue
        
        print(f'Group {group_i + 1} of potential duplicates')
        dups_group = duplicates_to_check.iloc[row_i: row_i + duplication_count + 1]
        if return_shown_indices:
            indices_shown.extend(dups_group.index)
        display(dups_group)
        text_lens = dups_group['text'].map(get_word_count).values
        if reduce(lambda x, y: x == y, text_lens):
            print('Text have equal length.')
        else:
            print(f"Text lengths are {', '.join(map(str, text_lens))}.")
        print('='*90)
        row_i += duplication_count + 1
        duplication_count = 0
    if return_shown_indices:
        return indices_shown
                  
check_duplicates_by_sorted_indices(indices, 'abstract')


# *Observations*: 
# * the first group seems to correspond to supplementary materials of the same paper,
# * the second group of the "dummy" short abstract contain different papers. The same holds for groups 6, 7.
# 
# The remaining true duplicates could have been detected when focusing on authors.
# To sum up, instead of cleaning starting with duplicated abstracts we should start from the more reliable field.

# ## Using *affiliations* field

# Affiliations provide detailed info on authors. Therefore, when possible let's use `affiliations` to find duplicates. Otherwise, let's use the `authors` field.
# 
# *a technical note: to speed up processing, I'm going to avoid pandas apply and use map instead. Whenever more columns would be required simultaneously, I'll just create a helping pandas series containing the required fields*

# In[ ]:


missing_affiliations_count = df_all_papers['affiliations'].isnull().sum()
print(f"Number of missing affiliations: {missing_affiliations_count}")
non_missing_affiliations = df_all_papers.loc[np.logical_not(df_all_papers['affiliations'].isnull()), 'affiliations']
duplicated_affiliations = non_missing_affiliations[non_missing_affiliations.duplicated()].unique()
print(f"Number of duplicated affiliations: {len(duplicated_affiliations)}")


# Let's focus on the duplicated affiliations. Either those are just papers from the perfectly same team, or potential duplicates.

# In[ ]:


duplicated_affiliations_bool_idx = df_all_papers['affiliations'].isin(duplicated_affiliations)
title_affilation_series = (df_all_papers['title'].map(lambda x: [x]) +
                           df_all_papers['affiliations'].map(lambda x: [x]))
removal_candidates_title_affiliation = title_affilation_series[duplicated_affiliations_bool_idx & 
                                                               title_affilation_series.map(tuple).duplicated(keep=False)]


# In[ ]:


print(f'Number of duplicates: {len(removal_candidates_title_affiliation)}.')


# Not too many, let's verify the duplicates manually by text.

# In[ ]:


indices = (removal_candidates_title_affiliation
           .map(lambda x: ', '.join(filter(lambda i: not isinstance(i, float), x)))
           .sort_values().index)

check_duplicates_by_sorted_indices(indices, 'affiliations')


# Observations: some article pairs have the same number of words, it's obvious that they are duplicates with some insignificant differences in text (otherwise they would have been dropped as duplicated `text` already). Some papers have slight difference in word counts, suggesting different versions. However, in some cases it seems that we have supplementary materials stored as a separate paper. I've manually examined these cases, let me summarize some insights.
# 
# *Observation 1*: the supplementary materials might contain important info, like the stats etc., but unfortunately we might not have the data itself, as in the case of the item above. Might be worth going through the most relevant papers manually.
# 
# *Observation 2*: supplementary materials have abstract missing.
# As supplementary materials can obviously provide important clues about values pieces of information (like the one below without actual figures yet with mentions of contacts/contact durations ect.), let's keep it as well. It might indicate to download the full paper and add information from the tables.

# I'll do the following: if the text lengths are the same, then I'll drop the either the 2nd one or the one without abstract; if the the text lengths are different, we clean data only if both titles are not missing. In this case if both abstracts are present, then I'll drop the shorter paper. When a shorter paper has no abstract, I'll concatenate the texts.

# In[ ]:


merged_to_parent_papers_total_count = 0
source_datasets_of_merged_supps = []
source_datasets_of_parents_for_merged_supps = []

def drop_strategy_developed_for_affiliations(indices):
    global merged_to_parent_papers_total_count
    global removed_duplicates_total_count
    global source_datasets_of_duplicates
    global source_datasets_of_merged_supps
    global source_datasets_of_parents_for_merged_supps
    
    merged_to_parent_papers_total_count_before = merged_to_parent_papers_total_count
    indices_to_drop = []
    duplicates_to_check = df_all_papers.loc[indices, ['title', 'abstract', 'text']]
    for row_i in range(0, len(duplicates_to_check), 2):
        dups_pair = duplicates_to_check.iloc[row_i: row_i + 2]
        text_lens = dups_pair['text'].map(get_word_count).values
        missing_abstracts = dups_pair['abstract'].map(lambda x: isinstance(x, float)).values
        if text_lens[0] == text_lens[1]:
            if missing_abstracts[0]:
                indices_to_drop.append(duplicates_to_check.index[row_i])
            else:
                indices_to_drop.append(duplicates_to_check.index[row_i + 1]) # including the case when both abstracts are present
            source_datasets_of_duplicates.extend(df_all_papers.loc[dups_pair.index, 'source_dataset'].values)
        else:
            missing_titles = dups_pair['title'].map(lambda x: isinstance(x, float)).values
            shorter_paper_i = np.argmin(text_lens)
            if not missing_titles[0] and not missing_titles[1]:
                shorter_paper_index = duplicates_to_check.index[row_i + shorter_paper_i]
                indices_to_drop.append(shorter_paper_index)
                merged_to_parent_papers_total_count += 1
                source_datasets_of_merged_supps.append(df_all_papers.loc[shorter_paper_index, 'source_dataset'])
                if not missing_abstracts[(shorter_paper_i+1) % 2] and missing_abstracts[shorter_paper_i]:
                    longer_paper_index = duplicates_to_check.index[row_i + (shorter_paper_i+1) % 2]
                    df_all_papers.loc[longer_paper_index, 'text'] += ' ' + df_all_papers.loc[shorter_paper_index, 'text']
                    source_datasets_of_parents_for_merged_supps.append(df_all_papers.loc[longer_paper_index, 'source_dataset'])
    df_all_papers.drop(indices_to_drop, inplace=True)
    number_of_merged_papers = merged_to_parent_papers_total_count - merged_to_parent_papers_total_count_before
    removed_duplicates_total_count += len(indices_to_drop) - number_of_merged_papers
    print(f'{len(indices_to_drop) - number_of_merged_papers} items were removed.')
    print(f'{number_of_merged_papers} items were merged to parents.')

drop_strategy_developed_for_affiliations(indices)


# ## Using *authors* field
# Let's repeat the same for the `authors` field, which is a less-informative version of the `affiliations`. 
# 
# *In the vain of Andrej Karpathy's advice not to generalize too early, I'm not creating general functions straight away for the one-time cleaning (even though I'm in fact copy-pasting.. but I'm not going to couple any other field with title, so no practical reason to generalize). Btw, if by any chance some of us hasn't read the Karpathy's [Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) yet, I'd recommend to do so, it's priceless!*

# In[ ]:


non_missing_authors = df_all_papers.loc[np.logical_not(df_all_papers['authors'].isnull()), 'authors']
duplicated_authors = non_missing_authors[non_missing_authors.duplicated()].unique()
duplicated_authors_bool_idx = df_all_papers['authors'].isin(duplicated_authors)
title_authors_series = (df_all_papers['title'].map(lambda x: [x]) +
                        df_all_papers['authors'].map(lambda x: [x]))
removal_candidates_title_author = title_authors_series[duplicated_authors_bool_idx & 
                                                       title_authors_series.map(tuple).duplicated(keep=False)]
print(f'Number of duplicates: {len(removal_candidates_title_author)}.')


# In[ ]:


indices = (removal_candidates_title_author
           .map(lambda x: ', '.join(filter(lambda i: not isinstance(i, float), x)))
           .sort_values().index)

check_duplicates_by_sorted_indices(indices, 'authors')


# Except for the last group all observations made during cleaning based on affiliations hold. Let's repeat the cleaning.

# In[ ]:


drop_strategy_developed_for_affiliations(indices)


# Regarding the last group from initial manual investigation, even though similar lengths suggested that we might be dealing with the same paper, manual check verified that those are two different papers.

# ## Using *title* field
# After cleaning based on more reliable combinations of authors(affiliations) + title, let's check if some duplicates can be still spotted based on `title` field solely.

# In[ ]:


indices = df_all_papers.loc[~df_all_papers['title'].isnull() & df_all_papers['title'].duplicated(keep=False),
                            'title'].sort_values().index
print(f'Number of papers with non-unique title: {len(indices)}.')              


# In[ ]:


check_duplicates_by_sorted_indices(indices, 'title', max_number_of_groups=20)


# Based on the gained intuition from the inspection above, let's check only groups of size 2 with titles longer than 11 words.

# In[ ]:


indices = df_all_papers.loc[~df_all_papers['title'].isnull() & 
                            df_all_papers['title'].duplicated(keep=False) & 
                            (df_all_papers['title'].map(get_word_count) > 11),
                            'title'].sort_values().index
indices = check_duplicates_by_sorted_indices(indices, 'title', max_group_size=2, return_shown_indices=True)


# In[ ]:


drop_strategy_developed_for_affiliations(indices)


# # Summary of duplicates cleaning

# ## How the discovered duplicates were processed

# In[ ]:


fig, ax = plt.subplots(figsize=(7, 7))

# credits: https://stackoverflow.com/questions/6170246/how-do-i-use-matplotlib-autopct
def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.0f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

processed_duplicates_counts = [removed_duplicates_total_count, merged_to_parent_papers_total_count]
ax.pie(processed_duplicates_counts, labels=['Duplicates removed', 'Merged to parent as supplementary material'], 
       autopct=make_autopct(processed_duplicates_counts), shadow=True, startangle=90)
ax.axis('equal')
_ = ax.set_title('How the discovered duplicates were processed', fontsize=20)


# ## Counts of removed duplicates per source dataset

# In[ ]:


duplicate_source_dataset_counts = pd.Series(source_datasets_of_duplicates).value_counts()
duplicate_source_dataset_counts.plot(kind='barh',
                                     title='Counts of removed duplicates per dataset', 
                                     color=['dodgerblue'])


# ## Counts of merged duplicates per source dataset

# In[ ]:


merged_source_dataset_counts = pd.Series(source_datasets_of_merged_supps).value_counts()
merged_source_dataset_counts.plot(kind='barh',
                                  title='Counts of the merged shorter papers per dataset', 
                                  color=['dodgerblue'])


# In[ ]:


parent_for_merged_source_dataset_counts = pd.Series(source_datasets_of_parents_for_merged_supps).value_counts()
parent_for_merged_source_dataset_counts.plot(kind='barh',
                                             title='Counts of the merged longer papers per dataset', 
                                             color=['dodgerblue'])


# # Storing the results for further reuse

# In[ ]:


output_file_name = f"master_dataset_cleaned_{time.strftime('%Y%m%d_%H%M')}.csv"
df_all_papers.to_csv(output_file_name, index=None)

