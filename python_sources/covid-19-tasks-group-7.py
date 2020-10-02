#!/usr/bin/env python
# coding: utf-8

# # Covid Round 2 - Group 7
# 
# After trying various advanced data extraction methods like NLP, I found that I was generally able to get better results using simpler, hand-crafted regular expression searches for each question in the study task. For most searches, I used a series of "ranked regexes" to scan through the abstracts or full body texts looking for particular patterns, where the regexes were ordered from most restrictive and most likely relevant to more common but less likely relevant.
# 
# There are three main steps:
# 
# 1. Import the metadata and search for covid synonyms in the titles and abstracts; drop anything that doesn't appear covid-related.
# 2. For each particular task or topic, search for specific keywords in titles and abstracts
# 3. Using this minimized dataset, use regular expressions to search the full article text and extract the relevant passages
# 
# This is still a work in progress.

# # Preparing the data
# 
# First we load the metadata and clean it up; ignoring anything that isn't covid-related, dropping duplicates, and ignoring anything that is missing actual article text.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re # regular expression for text matching
import json
import os


# In[ ]:


#load the files
allmeta = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv', low_memory=False)
print('All metadata: ', allmeta.shape)

#We can see here that several columns are almost entirely null, so we could probably drop those
#allmeta.isna().sum().plot(kind='bar', stacked=True)

#Drop duplicates
covmeta = allmeta.drop_duplicates(subset=('cord_uid','title'), keep='first')
#Drop articles with no full text
covmeta = covmeta.dropna(subset=('pdf_json_files','pmc_json_files'), how='all')
#Drop articles with no abstract
covmeta = covmeta.dropna(subset=('abstract',))
#Drop articles from before 2020
covmeta = covmeta[covmeta['publish_time'].str.contains('2020')]
print('Clean metadata: ', covmeta.shape)

allmeta = None


# In[ ]:


def search_column(df, column, regex):
    return df[df[column].str.contains(regex, case=False, flags=re.IGNORECASE)]

#search just for abstracts that mention covid-19
covmeta_regex = r'covid|-cov-2|cov2|ncov|sars-cov-2|wuhan|novel\scoronavirus|coronavirus\s2019'
covmeta = search_column(covmeta, 'abstract', covmeta_regex)
print("Covid metadata:", covmeta.shape)

covmeta.head()


# Next we load the full text linked to the remaining metadata.

# In[ ]:


#load all the full text; this will take a while
covfiles = []
for index, row in covmeta.iterrows():
    path = ''
    if isinstance(row['pmc_json_files'], str):
        #print("PMC: ", row['pmc_json_files'])
        path = row['pmc_json_files'].split(';')[0]
    else:
        #print("PDF: ", row['pdf_json_files'])
        if isinstance(row['pdf_json_files'], str):
            path = row['pdf_json_files'].split(';')[0]
            
    if path:
        with open(os.path.join('/kaggle/input/CORD-19-research-challenge/', path), 'rb') as file:
            json_data = json.load(file)
            json_data['meta_id'] = row['cord_uid']
            covfiles.append(json_data)

print(covfiles[0]['body_text'][1]['text'])
covtext = pd.DataFrame(covfiles)
print("Text data:", covtext.shape)

covfiles = None

#abstract is always null so we can drop it
#covtext.isna().sum().plot(kind='bar', stacked=True)
covtext = covtext.drop('abstract', axis=1)
#covtext.head()
covtext.head()


# Here we add a reformatted version of the body text and an array of sentences to make it easier to search.

# In[ ]:


#Note: This function based on https://www.kaggle.com/mlconsult/round-2-working-example-material-studies-covid-19
def format_body(body_text):
    texts = [(di['section'], di['text']) for di in body_text]
    texts_di = {di['section']: "" for di in body_text}
    
    for section, text in texts:
        texts_di[section] += text

    body = ""

    for section, text in texts_di.items():
        body += section.replace('\n', ' ')
        body += "\n\n"
        body += text.replace('\n', ' ')
        body += "\n\n"
    
    return body

covtext['reformatted_body'] = covtext.body_text.apply(format_body)

def split_sentences(text):
    reformatted = re.sub(r'\. |\.(?!\d)', '.\n', text)
    #reformatted = text.replace('. ', '.\n')
    return reformatted.split('\n')

covtext['sentences'] = covtext.reformatted_body.apply(split_sentences)

covtext.head()


# Here we set up some common helper functions to allow us to search for similar task data.
# 

# In[ ]:


def quickprint(text):
    return print(f'{text}\n')

def search_title(df, regex):
    return df[df['title'].str.contains(regex, case=False, flags=re.IGNORECASE)]

def search_title_and_abstract(df, regex):
    return df[df['title'].str.contains(regex, case=False, flags=re.IGNORECASE) | 
              df['abstract'].str.contains(regex, case=False, flags=re.IGNORECASE)]

def get_sentence_match(regex, sentences):    
    for s in sentences:
        if re.search(regex, s, flags=re.IGNORECASE):
            return s 
        
def get_best_match(ranked_regexes, sentences):
    for regex in ranked_regexes:
        s = get_sentence_match(regex, sentences)
        if s != None:
            return s
        
def get_journal(row):
    if isinstance(row.journal, str):
        return row.journal
    return row.source_x

#returns a single tag for the first match found
def get_exclusive_tag_from_text(text, searches_and_tags):
    for search_and_tag in searches_and_tags:
        if re.search(search_and_tag[0], text, flags=re.IGNORECASE):
            return search_and_tag[1]
    return ''

#returns a list of tags for all matches found
def get_tags_from_text(text, searches_and_tags):
    tags = []
    for search_and_tag in searches_and_tags:
        if re.search(search_and_tag[0], text, flags=re.IGNORECASE):
            tags.append(search_and_tag[1])
            continue
    return tags

def get_tags_string_from_text(text, searches_and_tags):
    return ', '.join(get_tags_from_text(text, searches_and_tags))

def regex_shuffle2(term1, term2):
    return f'({term1}.*{term2})|({term2}.*{term1})'

def regex_shuffle3(term1, term2, term3):
    return f'({term1}.*{term2}.*{term3})|({term1}.*{term3}.*{term2})|({term2}.*{term1}.*{term3})|({term2}.*{term3}.*{term1})|({term3}.*{term1}.*{term2})|({term3}.*{term2}.*{term1})'

study_search_and_tags = [[r'\bSystematic review\b', 'Systematic review and meta-analysis'],
                         [r'\bProspective observational\b', 'Prospective observational study'],
                         [r'\bRetrospective observational\b', 'Retrospective observational study'],
                         [r'\bCross.?sectional\b', 'Cross-sectional study'],
                         [r'\bEcological regression\b', 'Ecological regression'],
                         [r'\bCase series\b', 'Case series'],
                         [r'\bExpert review\b', 'Expert review'],
                         [r'\bEditorial\b', 'Editorial'],
                         [r'\bSimulation\b', 'Simulation']]
        
def get_study_from_row(row):
    study = get_exclusive_tag_from_text(row.abstract, study_search_and_tags)
    if study != '' and study != 'Simulation':
        return study
    #fulltext = covtext[covtext.meta_id == row.cord_uid].reformatted_body.item()
    return get_exclusive_tag_from_text(row.reformatted_body, study_search_and_tags)

severity_search_and_tags = [[r'\bsevere(?! acute)\b', 'Severe'],
                            [r'\bmild\b', 'Mild'],
                            [r'\bmoderate\b', 'Moderate'],
                            [r'\bvaried\b', 'Varied'],
                            [r'(\bnon.?icu)|(\bnon.?intensive care unit)', 'Non-ICU'],
                            [r'(\b(?! non.?)ICU\b)|(\b(?! non.?)intensive care unit\b)', 'ICU']]

def get_severity_from_text(text):
    return get_tags_string_from_text(text, severity_search_and_tags)                            


# # Task-Specific Output: Group 7
# 
# Here we set up output tables based on the task samples
# 

# In[ ]:


taskdir = '../input/CORD-19-research-challenge/Kaggle/target_tables/7_therapeutics_interventions_and_clinical_studies/'
output_examples = []
for filename in os.listdir(taskdir):
    example = pd.read_csv(taskdir+filename)
    print(filename," Index: ",len(output_examples), " Shape: ", example.shape)
    output_examples.append(example)


# # What is the efficacy of novel therapeutics being tested currently?
# 
# First we look at the example data and make an output file based on it.

# In[ ]:


q1example = output_examples[0]
q1output = pd.DataFrame(columns=q1example.columns)
q1example['Study'].apply(quickprint)
q1example.head()


# **Drugs Used**
# 
# Here I'm using the drug nomenclature table from wikipedia: https://en.wikipedia.org/wiki/Drug_nomenclature as a regex search to try to find drug names within the text.

# In[ ]:


drug_tags = [
'\w+afil',
'\w\w\w+ast', #two characters to avoid "fast", "east"
'\w+axine',
'\w+barb\w*',
'\w+caine',
'\w+ciclib',
'\w+cillin',
'\w+grel.*',
'\w+imsu',
'\w+icin', '\w+ilin', '\w+ycin', #instead of just '-in', which was too vague
'\w+amine', '\w+adine', '\w+asine', '\w+crine', '\w+quine', #instead of just '-ine', which was too vague
'\w+lisib',
'\w+lukast',
'\w+mab',
'\w+olol',
'\w+oxacin',
'\w+oxetine',
'\w+parib',
'\w+prazole',
'\w+pril',
'\w+prost\w*',
'\w+sartan',
'\w+tide',
'\w+tinib',
'\w+vastatin',
'\w+vec',
'\w+vir',
'\w+xaban',
'\w+ximab',
'\w+zumab',
'cef\w+',
'dexa\w+',  #added to catch things like dexamethasone
'\w+metha\w+'] #added to catch things like dexamethasone

not_drugs = ['breast', 'contrast', 'april', 'nucleotide', 'dinucleotide', 'peptide', 'examine']

def generate_drug_regex():
    words = []
    for word in drug_tags:
        words.append(f'\\b{word}\\b')
    return "|".join(words)

drug_regex_text = generate_drug_regex()
drug_regex = re.compile(drug_regex_text, flags=re.IGNORECASE)
print(drug_regex.pattern)
        
def get_drugs(text):
    drug_matches = re.findall(drug_regex,text)
    drug_matches = set(map(str.lower, drug_matches))
    for word in not_drugs:
        drug_matches.discard(word)
    return ", ".join(drug_matches)


# Search for articles that might be related to the task.

# In[ ]:


q1search = r'treatment|drug|medicin|therap|trial|' + drug_regex_text
q1 = search_title(covmeta, q1search)
print('Q1 metadata shape: ', q1.shape)
q1.sample(5).title.apply(quickprint)
q1 = q1.join(covtext.set_index('meta_id'), on='cord_uid')
q1.head()


# In[ ]:


q1.head(5).title.apply(quickprint)
q1.head(5).url.apply(quickprint)
print()


# **Therapeutics Used**
# 
# Here we'll use the drugs list created before as well as some other terms to search for the specific treatments discussed.

# In[ ]:


#example
q1.head(10).reformatted_body.apply(get_drugs).apply(print)
print()


# **Sample Size**
# 
# Use a numeric and keyword search to try to guess the sample size used.

# In[ ]:


samplesize_regexes=[r'\s\d+\s.*sample size',
                    r'sample size.*\s\d+\b',
                    r'\s\d+\s.*(patient|case|participant|distribution)',
                    r'\s\d+\s.*(drug|trial|therap|medic|study)',
                    r'(patient|case|participant|distribution|drug|trial|therap|medic|study).*\s\d+\b']
                     
def get_samplesize(row):
    return get_best_match(samplesize_regexes, row.sentences)

#example
q1.head(10).apply(get_samplesize, axis=1).apply(quickprint)
print()


# **Severity of Disease**
# 
# Here we'll use keywords to search for the severity

# In[ ]:


#example
q1.head(10).reformatted_body.apply(get_severity_from_text).apply(print)
print()


# **Outcome or conclusion**
# 
# Search the text for an excerpt about the conclusion.

# In[ ]:


term1 = r"(conclu|demonstrat|effective|succe|fail|finding|outcome|proof)"
term2 = r"(treat|therap|drug|medic|approach|trial|strategy)"
term3 = f'({drug_regex_text})'
solution_regexes=[regex_shuffle3(term1, term2, term3),
                  regex_shuffle2(term1, term3),
                  regex_shuffle2(term1, term2),
                  term1,
                  regex_shuffle2(term2, term3)]
                     
def get_solutions(row):
    return get_best_match(solution_regexes, row.sentences)

#example
q1.head(10).apply(get_solutions, axis=1).apply(quickprint)
print()


# **Primary Endpoint**
# 
# Find the primary endpoint or measurement.

# In[ ]:


term1 = r"(\bprimary\sendpoint\b)"
term2 = r"endpoint"
endpoint_regexes=[term1,
                  term2]
                     
def get_primary_endpoint(row):
    return get_best_match(endpoint_regexes, row.sentences)

#example
q1.head(10).apply(get_primary_endpoint, axis=1).apply(quickprint)
print()


# **Clinical Improvement**
# 
# Whether or not the patients showed improvement.

# In[ ]:


term1 = r'(fail(s|ed|\b))|(not.*improve)'
term2 = r'(patient|case|trail|result)'
failure_regexes = [regex_shuffle2(term1, term2),
                  term1]
                     
def get_failed_to_improve(row, do_print):
    sentence = get_best_match(failure_regexes, row.sentences)
    if sentence != None:
        if do_print:
            print(sentence)
        return True
    return False

term1 = r'((?! not.*)improve|(?! (not|fail|poor).*)recover)'
success_regexes = [regex_shuffle2(term1, term2),
                   term1]

def get_did_improve(row, do_print):
    sentence = get_best_match(success_regexes, row.sentences)
    if sentence != None:
        if do_print:
            print(sentence)
        return True
    return False

def get_improvement(row):
    if get_failed_to_improve(row, False):
        return 'N'
    if get_did_improve(row, False):
        return 'Y'
    return ''

#example
q1.head(10).apply(lambda x: get_failed_to_improve(x, True), axis=1).apply(print)
q1.head(10).apply(lambda x: get_did_improve(x, True), axis=1).apply(print)
q1.head(10).apply(get_improvement, axis=1).apply(print)
print()


# Finally we use the combined functions above to generate the output file.

# In[ ]:


for index, row in q1.iterrows():
    newrow = [index, row.publish_time, row.title, row.url, get_journal(row), get_drugs(row.reformatted_body), get_samplesize(row), get_severity_from_text(row.reformatted_body), get_solutions(row), get_primary_endpoint(row), get_improvement(row), get_study_from_row(row), 'Unknown']
    q1output.loc[index] = newrow
    
q1output.head()


# In[ ]:


q1output.to_csv('What is the efficacy of novel therapeutics being tested currently_.csv')


# # What is the best method to combat the hypercoagulable state seen in COVID-19?
# 
# Now we'll look at the second question and generate a new output file.

# In[ ]:


q2example = output_examples[1]
q2output = pd.DataFrame(columns=q2example.columns)
q2example['Study'].apply(quickprint)
q2example.head()


# In[ ]:


q2search = r'coagul|plasm|thromb|hypox|pulmonar|tachycard|\bcardi[ao]|\barter|\bplatelet|\bblood|\bclot' #+ drug_regex_text
q2 = search_title(covmeta, q2search)
print('Q2 metadata shape: ', q2.shape)
q2.sample(5).title.apply(quickprint)
q2 = q2.join(covtext.set_index('meta_id'), on='cord_uid')
q2.head()


# Finally we will create the second output file based on the results.

# In[ ]:


for index, row in q2.iterrows():
    newrow = [index, row.publish_time, row.title, row.url, get_journal(row), get_study_from_row(row), get_drugs(row.reformatted_body), get_samplesize(row), get_severity_from_text(row.reformatted_body), get_solutions(row), get_primary_endpoint(row), get_improvement(row), 'Unknown']
    q2output.loc[index] = newrow
    
q2output.head()


# In[ ]:


q2output.to_csv('What is the best method to combat the hypercoagulable state seen in COVID-19_.csv')

