#!/usr/bin/env python
# coding: utf-8

# # **Identifying and searching papers that reference antiviral medicines**
# In this kernel, I use lists of drugs generated from [RxNorm](https://mor.nlm.nih.gov/RxNav/)/[RxNav](https://mor.nlm.nih.gov/RxNav/)/[RxClass](https://mor.nlm.nih.gov/RxClass/) to identify a subset of papers that may be of further interest for addressing the questions posed in the vaccines/therapeutics task. From this subset of papers, I create a dataframe with useful features for querying. As an example for what can be done with this dataframe, I present the specific use case of searching mentions of these drugs for words related to animal models and drug efficacy.

# In[ ]:


import numpy as np
import pandas as pd
import os
import json
import glob
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Use RxNav to find classes of drugs that are of interest
# I've chosen to start with antivirals and antimalarials (primarily due to the present interest in chloroquine) shown in RxNav, but also recent experimental antivirals being discussed.

# In[ ]:


potential_antivirals = ['abacavir', 'abacavir / dolutegravir / Lamivudine', 'abacavir / Lamivudine', 'abacavir / Lamivudine / Zidovudine', 'Acyclovir', 'adefovir', 'Amprenavir', 'asunaprevir', 'Atazanavir', 'Atazanavir / cobicistat', 'Baloxavir marboxil', 'bictegravir / emtricitabine / tenofovir alafenamide', 'boceprevir', 'brivudine', 'Cidofovir', 'cobicistat / darunavir', 'cobicistat / darunavir / emtricitabine / tenofovir alafenamide', 'cobicistat / elvitegravir / emtricitabine / tenofovir alafenamide', 'cobicistat / elvitegravir / emtricitabine / tenofovir disoproxil','daclatasvir', 'darunavir', 'dasabuvir', 'dasabuvir / ombitasvir / paritaprevir / Ritonavir', 'Delavirdine', 'Didanosine', 'dolutegravir', 'dolutegravir / Lamivudine', 'dolutegravir / Rilpivirine', 'DORAVIRINE', 'DORAVIRINE / Lamivudine / tenofovir disoproxil', 'efavirenz', 'efavirenz / emtricitabine / tenofovir disoproxil', 'efavirenz / Lamivudine / tenofovir disoproxil', 'elbasvir', 'elbasvir / grazoprevir', 'elvitegravir', 'emtricitabine', 'emtricitabine / Rilpivirine / tenofovir alafenamide', 'emtricitabine / Rilpivirine / tenofovir disoproxil', 'emtricitabine / tenofovir alafenamide', 'emtricitabine / tenofovir disoproxil','enfuvirtide', 'entecavir', 'etravirine', 'famciclovir', 'fosamprenavir', 'Foscarnet', 'Ganciclovir', 'glecaprevir / pibrentasvir', 'grazoprevir', 'ibalizumab', 'Idoxuridine', 'Indinavir', 'Inosine Pranobex', 'Lamivudine', 'Lamivudine / Nevirapine / Stavudine', 'Lamivudine / Nevirapine / Zidovudine', 'Lamivudine / tenofovir disoproxil', 'Lamivudine / Zidovudine', 'ledipasvir / sofosbuvir', 'letermovir', 'lopinavir / Ritonavir', 'lysozyme', 'maraviroc', 'moroxydine', 'Nelfinavir', 'Nevirapine', 'ombitasvir / paritaprevir / Ritonavir', 'Oseltamivir', 'penciclovir', 'peramivir', 'raltegravir', 'Ribavirin', 'Rilpivirine', 'Rimantadine', 'Ritonavir', 'Saquinavir', 'simeprevir', 'sofosbuvir', 'sofosbuvir / velpatasvir', 'sofosbuvir / velpatasvir / voxilaprevir', 'Stavudine', 'Tecovirimat', 'telaprevir', 'telbivudine', 'tenofovir alafenamide', 'tenofovir disoproxil', 'tipranavir', 'tromantadine', 'valacyclovir', 'valganciclovir', 'Vidarabine', 'Zalcitabine', 'Zanamivir', 'Zidovudine'] + ['Amodiaquine', 'artemether', 'artemether / lumefantrine', 'artesunate', 'Chloroquine', 'halofantrine', 'Hydroxychloroquine', 'Mefloquine', 'Primaquine', 'Proguanil', 'Pyrimethamine', 'Quinine', 'tafenoquine'] + ['remdesivir', 'galidesivir', 'favipiravir']


# Next, I split compound medicines into their constituent agents, normalize capitalization, and reduce to the unique set of agents to create a set of search terms.

# In[ ]:


antiviral_search_terms = []
for drug in potential_antivirals:
    if ' / ' in drug:
        antiviral_search_terms.extend([component.lower() for component in drug.split(' / ')])
    else:
        antiviral_search_terms.append(drug.lower())
antiviral_search_terms = list(set(antiviral_search_terms))


# ### Load papers and search
# I'm opting to work with the `body_text` only (approximately 13.2K documents of 29.5K); it's a list of dictionaries containing a `text` key. I join these together to construct the full text. First, I'll do a simple verificaiton that this method is reliably retrieving documents by checking the character counts.

# In[ ]:


paper_lengths = []
for paper_path in glob.glob('/kaggle/input/CORD-19-research-challenge/2020-03-13/*/*/*.json'):
    with open(paper_path, 'r') as f:
        paper = json.loads(f.read())
    paper_text = '\n'.join(item['text'] for item in paper['body_text'])
    paper_lengths.append(len(paper_text))
_ = plt.hist(paper_lengths, bins=[i*1000 for i in range(100)])
plt.show()


# OK, there are some that are quite short, but it looks like it's working.

# In[ ]:


papers_referring_to_antivirals = []
papers_searched = 0
papers_matching = 0
for paper_path in glob.glob('/kaggle/input/CORD-19-research-challenge/2020-03-13/*/*/*.json'):
    with open(paper_path, 'r') as f:
        paper = json.loads(f.read())
    paper_text = '\n'.join(item['text'] for item in paper['body_text']).lower()
    if any([term in paper_text for term in antiviral_search_terms]):
        papers_referring_to_antivirals.append(paper_path)
        papers_matching += 1
    papers_searched += 1
    if papers_searched % 1000 == 0:
        print(papers_searched, papers_matching)


# Encouraging - over 10% of the papers reference one or more of the medicines! I'll write out the full text from this subset of papers to my working directory.

# In[ ]:


for paper_path in papers_referring_to_antivirals:
    filename = paper_path.split('/')[-1].split('.')[0] + '.txt'
    with open(filename, 'w') as f:
        with open(paper_path, 'r') as g:
            paper = json.loads(g.read())
        f.write('\n'.join(item['text'] for item in paper['body_text']).lower())


# ### Find out which agents are mentioned in each paper
# Here I use a count vectorizer with the antiviral search terms as the dictionary. I don't particularly care about the frequency that the agents show up in any one paper, so I'll "flatten" the vectorization to True (agent was found in paper) / False (agent was not found in paper).

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(input='filename', lowercase=True, vocabulary=antiviral_search_terms)


# In[ ]:


output = cv.fit_transform(glob.glob('*.txt'))
presence_of_term = np.where(output.toarray() > 0, True, False)


# Now I'll create a dataframe with the information about which drugs appear in each paper. The `sha` values will be used as the merge key.

# In[ ]:


antiviral_df = pd.DataFrame(presence_of_term, columns=antiviral_search_terms)
antiviral_df['sha'] = [fname.split('.')[0] for fname in glob.glob('*.txt')]


# I'll load the summary table and merge it with the dataframe of papers with mentions of antivirals.

# In[ ]:


index_df = pd.read_csv('/kaggle/input/CORD-19-research-challenge/2020-03-13/all_sources_metadata_2020-03-13.csv')
output_df = index_df.merge(antiviral_df, on='sha', how='inner')
output_df.index = output_df['sha']


# Turns out there are duplicate `sha`s in the dataset (you can check via `output_df['sha'].value_counts()`); I'll de-dupe the output dataframe by iterating through the dataframe and keeping track how many prior times the `sha` has been seen so far. Afterwards I'll add this as a new column and restrict `output_df` to places where the number of repeats of the `sha` are 0.

# In[ ]:


sha_repeats_dict = {}
times_sha_repeats = []
for sha, row in output_df.iterrows():
    if sha in sha_repeats_dict:
        sha_repeats_dict[sha] += 1
    else:
        sha_repeats_dict[sha] = 0
    times_sha_repeats.append(sha_repeats_dict[sha])


# In[ ]:


output_df['times_sha_repeated'] = times_sha_repeats
output_df = output_df[output_df['times_sha_repeated'] == 0]


# ### Simple examples

# #### What drugs appear in the most papers?
# Let's look at the top 20 drugs.

# In[ ]:


output_df[antiviral_search_terms].sum().sort_values(ascending=False)[:20]


# #### Which papers refer to large numbers of the drugs?
# Let's create a new column in the dataframe looking at the number of distinct drugs mentioned in the paper.

# In[ ]:


output_df['drugs_mentioned'] = output_df[antiviral_search_terms].sum(axis=1)


# In[ ]:


output_df.sort_values('drugs_mentioned', ascending=False)[:20][['title', 'publish_time', 'doi', 'drugs_mentioned']]


# ### A more complicated example: in what context are the drugs being mentioned?
# For each article, I'll grab the sentences referring to antivirals and concatenate them to make a pseudo-summary of the paper. I'll use `spaCy` to handle the identification of sentences.

# In[ ]:


import spacy
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000 # there's at least one paper that exceeds the default max_length of 1000000


# In[ ]:


def get_article_text(sha):
    with open(sha + '.txt', 'r') as f:
        return f.read()


# The following block of code steps through each row (article) of the dataframe. I use the `sha` value to point to the fulltext for the paper, retrieve it, and tokenize it. Then, I identify each token that is one of the antivirals, and store the sentence it was in. The sentences in each paper are concatenated (separated by newline characters) and stored as a new column in the dataframe. This part is a little slow to execute (tens of minutes).

# In[ ]:


relevant_sentences = []
i = 0
for sha, row in output_df.iterrows():
    if i % 100 == 0:
        print(i)
    i += 1
    relevant_sentences_in_paper = []
    drugs_in_paper = [drug for drug, drug_in_paper in row[antiviral_search_terms].iteritems() if drug_in_paper]
    doc = nlp(get_article_text(sha))
    for token in doc:
        if token.text in drugs_in_paper:
            relevant_sentences_in_paper.append(token.sent.text)
    relevant_sentences.append('\n'.join(list(set(relevant_sentences_in_paper))))
output_df['relevant_sentences'] = relevant_sentences


# ### Output the dataframe as a CSV
# This may serve as a useful starting point for those looking to do a deep-dive into text analysis of papers referring to antiviral treatments.

# In[ ]:


output_df.to_csv('antiviral_paper_table.csv', index=False)


# ## Use case: finding useful sections of articles mentioning drugs
# **[Favipiravir](https://en.wikipedia.org/wiki/Favipiravir)** has been in the news as a drug with potential to treat COVID-19. What has been said in recent (2019-2020) papers? We can query the dataframe to identify this in the following way:
# * Take all papers that mention favipiravir (`output_df['favipiravir'] == True`)
# * Combine this with a query that looks at `publish_time` (note that quite a few papers are without; could consider trying to extract publication dates from `doi` where available) - we want it to contain either 2019 or 2020
# 
# From the remaining articles, we can output the sentences found in `relevant_sentences` by putting them in a list, newline-joining them, and printing:

# In[ ]:


recent_favipiravir_df = output_df[(output_df['favipiravir']) & (output_df['publish_time'].notnull()) & ((output_df['publish_time'].str.contains('2019')) | (output_df['publish_time'].str.contains('2020')))]


# In[ ]:


print('\n\n'.join(recent_favipiravir_df['relevant_sentences'].tolist()))


# OK, that's quite a few... what if I only look at sentences where there are terms like `effic`, `activ`, `effect` (the stem will pick up the words I'm interested in for these first three), or `model` (i.e. animal model)? I'll write a function that searches for sentences containing terms of interest and return those in a formatted "report".

# In[ ]:


def search_article_subset(subset_df, search_terms):
    search_results = ''
    for sha, row in subset_df.iterrows():
        sentences_matching_search = []
        # sorry for the egregious list comprehension - each article's relevant sentences are newline separated sentences. we want to check each sentence against each search term.
        matching_sentences = [sentence for sentence in row['relevant_sentences'].split('\n') if any([st in sentence for st in search_terms])]
        if matching_sentences != []:
            sentences_matching_search.extend(matching_sentences)
        if sentences_matching_search != []:
            search_results += '** article sha: {}\n'.format(sha)
            search_results += '** article title: {}\n'.format(row['title'])
            search_results += '\n\n'.join(sentences_matching_search)
            search_results += '\n\n'
    return search_results


# In[ ]:


key_terms = ['effic', 'activ', 'effect', 'model']


# ### OK, let's try it out - what do our recent articles say about favipiravir in regards to animal models and efficacy?

# In[ ]:


print(search_article_subset(recent_favipiravir_df, key_terms))


# In[ ]:




