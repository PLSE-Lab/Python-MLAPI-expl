#!/usr/bin/env python
# coding: utf-8

# # Get a Software Engineer Job at Big Tech Companies
# 
# > Authors: [Anita Chang](https://github.com/a19990506), [Ariel Hsu](https://github.com/aarriiel), [Marco Wang](https://github.com/aesophor)
# >
# > Supervisor: [Pecu Tsai](https://github.com/pecu)
# >
# > Github Repo: https://github.com/aesophor/NTU-CSX-DataScience2019
# > 
# > Please refer to **[this external page](https://github.com/aesophor/NTU-CSX-DataScience2019/blob/master/hw4-final-project/README_EN.md)** to see all these graphs without code.
# 
# ## Intention 
# As a CS major student, working as a **software engineer** at one of the big tech companies has always been my dream. 
# However, I wonder...
# 
# * What if I'm not graduated from a prestigious university?
# * What if I don't have any work experience?
# * Being familiar with which programming languages (besides C++/Java/Python) gives me an edge?
# * Which frameworks/skills are preferred in Google/Facebook/Apple?
# 
# Therefore, my friends and I decided to collect all job descriptions relavant to the keyword `software` 
# from the websites of **Google, Facebook and Apple** ([web scrapers source code](https://github.com/aesophor/NTU-CSX-DataScience2019/tree/master/hw4-final-project/data)), then perform some analysis and visualization. Not
# sure if you would like it, but we do hope that our work can benefit other people (even if just a few!)
# 
# The data used in this kernel were collected on Jan 22, 2019. 
# 
# ## Outline
# 1. **Google**    
#   1.1 Degree Requirements    
#   1.2 Work Experience Requirements    
#   1.3 Frequently Appeared Programming Languages in Job Descriptions    
#   1.4 Keyword Extraction: TF-IDF Wordclouds    
#   1.5 Keyword Extraction: Term-Term Matrix Visualization    
# 2. **Facebook**    
#   2.1 Degree Requirements    
#   2.2 Work Experience Requirements    
#   2.3 Frequently Appeared Programming Languages in Job Descriptions    
#   2.4 Keyword Extraction: TF-IDF Wordclouds    
#   2.5 Keyword Extraction: Term-Term Matrix Visualization    
# 3. **Apple**    
#   3.1 Degree Requirements    
#   3.2 Work Experience Requirements    
#   3.3 Frequently Appeared Programming Languages in Job Descriptions    
#   3.4 Keyword Extraction: TF-IDF Wordclouds    
#   3.5 Keyword Extraction: Term-Term Matrix Visualization
# 4. **Global Job Opportunities at Google, Facebook and Apple**
# 5. **Conclusion**
#     
# ## 1. Google
# ### 1.1 Degree Requirements
# 

# In[ ]:


import re
import nltk
import pandas
import matplotlib
import numpy as np
from collections import OrderedDict, Counter
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image


# In[ ]:


df = pandas.read_csv('../input/raw-data/google_jobs.csv')
df.head(10)


# Here we'll create the following function (since it will be reused for many times).

# In[ ]:


def count_keywords_freq(df: pandas.DataFrame, col_name: str, keywords: list, none=False):
    """ Given a list of keywords and count their frequency in the specified pandas dataframe.
    :param df: target pandas dataframe.
    :param col_name: target column name.
    :param keywords: a list of keywords.
    :param none: count the items that contain no specified keywords at all.
    :return: keyword frequency dict.
    """
    freq = {keyword: 0 for keyword in keywords}    
    for keyword in keywords:
        freq[keyword] = df[col_name].str.contains(keyword, regex=False).sum()
    
    if none is True:
        freq['None'] = 0
        for col in df[col_name]:
            freq['None'] += 0 if type(col) is str and any(w in col for w in keywords) else 1
    
    return freq


# In[ ]:


keywords = ['PhD', 'Master', 'MBA', 'BA', 'BS', 'Bachelor']

# Count keyword frequency.
min_degree_reqs = count_keywords_freq(df, 'minimum_qual', keywords, none=True)
pref_degree_reqs = count_keywords_freq(df, 'preferred_qual', keywords, none=False)

print("Min: " + str(min_degree_reqs))
print("Pref: " + str(pref_degree_reqs))


# In[ ]:


# Convert the above dicts into pandas DataFrames.
min_degree_df = pandas.DataFrame.from_dict(min_degree_reqs, orient='index', columns=['Count'])
pref_degree_df = pandas.DataFrame.from_dict(pref_degree_reqs, orient='index', columns=['Count'])

min_degree_df


# In[ ]:


# Define bar colors
colors = ['#958090', '#83A2BE', '#7DAEA9', '#B4BF86', '#CBB079', '#B77A76', '#707070', '#AAAAAA']
min_labels = list(min_degree_reqs.keys())
min_values = list(min_degree_reqs.values())
pref_labels = list(pref_degree_reqs.keys())
pref_values = list(pref_degree_reqs.values())

plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.bar(min_labels, min_values, color=colors, width=0.5)
plt.xlabel('Degree')
plt.ylabel('Number of jobs')
plt.title('Minimum Degree Requirements')

plt.subplot(122)
plt.bar(pref_labels, pref_values, color=colors, width=0.5)
plt.xlabel('Degree')
plt.ylabel('Number of jobs')
plt.title('Preferred Degree Requirements')

plt.show()


# Most jobs at Google require at least a **Bachelor** degree, but they prefer applicants with **further education** (e.g., Master and PhD). However, there are 131 jobs that doesn't mention degree requirements at all.

# ### 1.2 Work Experience Requirements
# The following function works this way: For each column under the specified column name,
# * 3 years -> extract `3`
# * 3-5 years -> extract `5`
# * 5+ years -> extract `5`

# In[ ]:


def extract_experience(df: pandas.DataFrame, col_name: str, end_year=20):
    """ Extract years of experiences required.
    :param df: target dataframe.
    :param col_name: name of the column that contains strings
                     like `4 years of experience in ...`
    :param end_year: the last year in the list returned.
    :return: a list of years of exp required (indexed by years)
    """
    exp_list = [0] * (end_year + 1)
    
    for col in df[col_name]:
        if type(col) is not str:
            continue
        exp_required = re.findall('\d+ year', col)
        exp_required = exp_required + re.findall('\d+\+ year', col)
        year = 0 if not exp_required else int(exp_required[0].replace(' year', '').replace('+', ''))
        exp_list[year] += 1
        
    return exp_list


# In[ ]:


min_exp_list = extract_experience(df, 'minimum_qual')
pref_exp_list = extract_experience(df, 'preferred_qual')


# In[ ]:


colors = ['#958090', '#83A2BE', '#7DAEA9', '#B4BF86', '#CBB079', '#B77A76', '#707070', '#AAAAAA']
labels = np.arange(len(min_exp_list))

plt.figure(figsize=(16, 5))
plt.subplot(121)
plt.bar(np.arange(11), min_exp_list[0:11], color=colors, width=0.5)
plt.xticks(labels[0:11])
plt.xlabel('Year(s)')
plt.ylabel('Number of jobs')
plt.title('Minimum Experience Requirements')

plt.subplot(122)
plt.bar(labels[0:11], pref_exp_list[0:11], color=colors, width=0.5)
plt.xticks(labels[0:11])
plt.xlabel('Year(s)')
plt.ylabel('Number of jobs')
plt.title('Preferred Experience Requirements')

plt.show()


# Lots of job positions doesn't mention work experience. However, my guess is that I'll need some practical experience in building systems (since getting a software engineer job position at Google is quite difficult). We'll create wordclouds later, so let's carry on for now.

#  ### 1.3 Frequently Appeared Programming Languages in Job Descriptions

# In[ ]:


lang_colors = {
    'C++': '#F34B7D',
    'Java': '#B07219',
    'Python': '#3572A5',
    'JavaScript': '#F1E05A',
    'Go': '#375EAB',
    'PHP': '#4F5D95',
    'SQL': '#494D5C',
    'Ruby': '#701516',
    'Swift': '#FFAC45',
    'Kotlin': '#F18E33',
    'C#': '#178600',
    'Objective C': '#438EFF'
}

langs = lang_colors.keys()


# In[ ]:


# Count keyword frequency.
min_lang_reqs = count_keywords_freq(df, 'minimum_qual', langs)
pref_lang_reqs = count_keywords_freq(df, 'preferred_qual', langs)

# Some manual correction.
min_lang_reqs['Java'] -= min_lang_reqs['JavaScript']
pref_lang_reqs['Java'] -= pref_lang_reqs['JavaScript']
min_lang_reqs['Go'] -= df['minimum_qual'].str.contains('Google').sum()
pref_lang_reqs['Go'] -= df['preferred_qual'].str.contains('Google').sum()

# Sort the dicts.
min_lang_reqs = dict(sorted(min_lang_reqs.items(), key=lambda kv: kv[1], reverse=True))
pref_lang_reqs = dict(sorted(pref_lang_reqs.items(), key=lambda kv: kv[1], reverse=True))

# Create DataFrame from dict.
min_lang_df = pandas.DataFrame.from_dict(min_lang_reqs, orient='index', columns=['Count'])
pref_lang_df = pandas.DataFrame.from_dict(pref_lang_reqs, orient='index', columns=['Count'])

pref_lang_reqs


# In[ ]:


min_labels = list(min_lang_reqs.keys())
min_values = list(min_lang_reqs.values())
min_colors = [lang_colors[k] for k, v in min_lang_reqs.items()]

pref_labels = list(pref_lang_reqs.keys())
pref_values = list(pref_lang_reqs.values())
pref_colors = [lang_colors[k] for k, v in pref_lang_reqs.items()]


plt.figure(figsize=(21, 5))
plt.subplot(121)
plt.bar(min_labels, min_values, color=min_colors, width=0.5)
plt.title('Top 10 Programming Languages in Minimum Quals')

plt.subplot(122)
plt.bar(pref_labels, pref_values, color=pref_colors, width=0.5)
plt.title('Top 10 Programming Languages in Preferred Quals')

plt.show()


# I wonder what Google engineers are using Python for, so I randomly pick three job positions and inpsect their preferred qualifications.
# 
# Also it seems that I need to 
# * master at least one general purpose programming language, such as Java, C/C++, ..., etc. 
# * have experience with Unix/Linux open source developments && kernel/device drivers.
# * have experience building large software systems.
# * have competent communication skills (both written and verbal).

# In[ ]:


df[df['preferred_qual'].str.contains('Python')]['preferred_qual'].tolist()[0:3]


# ### 1.4 Keyword Extraction: TF-IDF Wordclouds
# We want to know what are the most important keywords in the job descriptions, so here we'll extract keywords from minimum qualifications, preferred qualifications and responsibilities, and then create three wordclouds using TF-IDF.

# In[ ]:


def import_terms(tokenizer: nltk.tokenize.MWETokenizer, term_file_path: str):
    """ Import all user-defined untokenizable terms from a file into nltk MWETokenizer.
    :param tokenizer: nltk MWETokenizer instance.
    :param text_file_path: path to the file.
    """
    with open(term_file_path, 'r') as f:
        for line in f:
            tokenizer.add_mwe(line.strip().split())


# In[ ]:


def tokenize(tokenizer: nltk.tokenize.MWETokenizer, s: str, lowercase=True, preserve_case_words=[]):
    """ Tokenize given string using nltk MWETokenizer.
    :param case: convert all tokens into lowercase.
    :param exclude_words: words that should preserve their cases.
    :return: a list of tokens.
    """
    tokens = nltk.tokenize.word_tokenize(s)
    tokens = tokenizer.tokenize(tokens)
    
    # Remove tokens that are either purely digits or purely punctuations.
    tokens = list(filter(lambda token: not token.isdigit() and re.search('[a-zA-Z]', token), tokens))

    # Since nltk MWETokenizer will not split tokens that contain a slash,
    # we'll have to do it ourselves.
    for token in tokens:
        if '/' in token:
            tokens += token.split('/')
            tokens.remove(token)
            
    # Lowercase conversion.
    tokens = [token.lower() if token not in preserve_case_words else token for token in tokens]
    return tokens


# In[ ]:


def create_word_freq_dict(tokenizer, df, col_name, lowercase=True, preserve_case_words=[]):
    """ Create a word frequency dict
    :param tokenizer: nltk MWETokenizer.
    :param df: source pandas dataframe.
    :param col_name: name of the column to create wfm from.
    :param lowercase: convert all tokens into lowercase.
    :param preserve_case_words: words that should preserve their cases.
    :return: a word frequency dict (dict of dict, separated by job indices).
    """
    freq = {}
    
    for i, col in enumerate(df[col_name]):
        if type(col) is str:
            freq[i] = {}
            words = tokenize(tokenizer, col, lowercase=True, preserve_case_words=preserve_case_words)
            for word in words:
                if word in freq:
                    freq[i][word] += 1
                else:
                    freq[i][word] = 1
                
    return freq


# In[ ]:


def create_wfm(word_frequency_dict: dict):
    """ Create word frequency matrix from the specified word frequency dict """
    dwf_list = [pandas.DataFrame(list(freq.values()), index=freq.keys()) for freq in word_frequency_dict.values()]
    wfm = pandas.concat(dwf_list, axis=1, sort=True)
    wfm = np.transpose(wfm).fillna(0)
    wfm.index = word_frequency_dict.keys()
    return wfm


# In[ ]:


# Initialize nltk MWETokenizer.
tokenizer = nltk.tokenize.MWETokenizer(separator=' ')
import_terms(tokenizer, '../input/cs_terms_and_stopwords/cs_terms.txt')

# Words that needs to preserve case.
preserve_case_words = list(langs) + ['.Net', '.NET']
min_qual_wfd = create_word_freq_dict(tokenizer, df, 'minimum_qual', True, preserve_case_words)
pref_qual_wfd = create_word_freq_dict(tokenizer, df, 'preferred_qual', True, preserve_case_words)
resp_qual_wfd = create_word_freq_dict(tokenizer, df, 'responsibilities', True, preserve_case_words)

list(min_qual_wfd[0].items())[0:10]


# In[ ]:


min_qual_wfm = create_wfm(min_qual_wfd)
pref_qual_wfm = create_wfm(pref_qual_wfd)
resp_wfm = create_wfm(resp_qual_wfd)

# Row: job, Column: word frequency
pref_qual_wfm.head(5)


# In[ ]:


def create_tfm(wfm):
    tfm = wfm.copy()
    for i in range(0, len(tfm)):
        tfm.iloc[i] = tfm.iloc[i] / tfm.iloc[i].sum()
    return tfm


# In[ ]:


min_qual_tfm = create_tfm(min_qual_wfm)
pref_qual_tfm = create_tfm(pref_qual_wfm)
resp_tfm = create_tfm(resp_wfm)

min_qual_tfm.head()


# In[ ]:


min_qual_df = (min_qual_wfm > 0).sum()
pref_qual_df = (pref_qual_wfm > 0).sum()
resp_df = (resp_wfm > 0).sum()


# In[ ]:


N = len(df)
N


# In[ ]:


def create_tfidfm(tfm, N, df):
    tfidfm = tfm.copy()
    for i in range(0, len(tfidfm)):
        # Add 0.01 so that those extremely frequent words won't be completely neglected.
        tfidfm.iloc[i] = tfidfm.iloc[i] * np.log10(N / df) + 0.01
    return tfidfm


# In[ ]:


min_qual_tfidfm = create_tfidfm(min_qual_tfm, N, min_qual_df)
pref_qual_tfidfm = create_tfidfm(pref_qual_tfm, N, pref_qual_df)
resp_tfidfm = create_tfidfm(resp_tfm, N, resp_df)


# In[ ]:


mask = np.array(Image.open('../input/wordcloud-mask/google-icon.png'))
plt.figure(figsize=(18,6))

plt.subplot(131)
tfidf_dict = min_qual_tfidfm.to_dict(orient='records')
wordcloud = WordCloud(mask=mask, background_color="white").fit_words(tfidf_dict[0])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Minimum qual',size=24)

plt.subplot(132)
tfidf_dict = pref_qual_tfidfm.to_dict(orient='records')
wordcloud = WordCloud(mask=mask, background_color="white").fit_words(tfidf_dict[0])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Preferred qual',size=24)

plt.subplot(133)
tfidf_dict = resp_tfidfm.to_dict(orient='records')
wordcloud = WordCloud(mask=mask, background_color="white").fit_words(tfidf_dict[0])
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.title('Responsibilities',size=24)

plt.show()


# * programming: Java/JSP/OOP/modular/asp.net/JavaScript/CSS/HTML/Ajax
# * tools: gwt/xul/xaml/flex
# * goal: accelerate/next-generation/redesign/user interface

# Now we can create the following function which allows us to create wordclouds easily!

# In[ ]:


def create_wordclouds(dataframe, mask_img_path, *col_names):
    """ Create wordclouds from the given column names """
    mask = np.array(Image.open(mask_img_path))
    plt.figure(figsize=(18,6))
    
    # Initialize nltk MWETokenizer.
    tokenizer = nltk.tokenize.MWETokenizer(separator=' ')
    import_terms(tokenizer, '../input/cs_terms_and_stopwords/cs_terms.txt')
    
    # Words that needs to preserve case.
    preserve_case_words = list(langs) + ['.Net', '.NET']
    
    for i, col_name in enumerate(col_names, 1):
        wfd = create_word_freq_dict(tokenizer, dataframe, col_name, True, preserve_case_words)
        wfm = create_wfm(wfd)
        tfm = create_tfm(wfm)
        df = (wfm > 0).sum()
        N = len(dataframe)
        tfidfm = create_tfidfm(tfm, N, df)
        
        plt.subplot(1, len(col_names), i)
        tfidf_dict = tfidfm.to_dict(orient='records')
        wordcloud = WordCloud(mask=mask, background_color="white").fit_words(tfidf_dict[0])
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.margins(x=0, y=0)
        plt.title(col_name, size=24)

    plt.show()
    
# Usage: create_wordclouds(df, '../input/wordcloud-mask/google-icon.png', 'minimum_qual', 'preferred_qual', 'responsibilities')


# ### 1.5 Keyword Extraction: Term-Term Matrix Visualization
# We also created a term-term matrix visualization. For each job position, we extract the top 3 most important keywords (we borrowed a smart stopwords list from [here](https://github.com/aneesha/RAKE/blob/master/SmartStoplist.txt)), create a term-term matrix and perform visualization with Gephi.
# 
# Reference: https://zhuanlan.zhihu.com/p/30023387?fbclid=IwAR1XnO0HXkhLMDrdQrS9dCYew9uk5aucQZzptR96JZWLnKW2nF9I3EVlMzM

# In[ ]:


keyword_df = df.copy()

# Replace NaN columns with empty strings.
keyword_df = keyword_df.replace(np.nan, '', regex=True)

# Split job titles by comma into separate columns.
keyword_df['title'] = keyword_df['title'].str.split(',', expand=True)[0]

# Combine minimum_qual, preferred_qual and responsibilities into a single column.
keyword_df['text'] = list(keyword_df['minimum_qual'] + keyword_df['preferred_qual'] + keyword_df['responsibilities'])

# Drop unused columns.
keyword_df = keyword_df.drop(['location', 'minimum_qual', 'preferred_qual', 'responsibilities'], axis=1)
keyword_df.head()


# In[ ]:


def get_keywords(tokens, num, stopwords=[]):
    if len(stopwords) > 0:
        for stopword in stopwords:
            for token in tokens:
                if token == stopword:
                    tokens.remove(token)
    return [t[0] for t in Counter(tokens).most_common(num)]


# In[ ]:


stopwords = []
with open('../input/cs_terms_and_stopwords/stopwords.txt', 'r') as f:
    for line in f:
        if not line.startswith('#'):
            stopwords.append(line.strip())


# In[ ]:


# Clean up the text.
tokenizer = nltk.tokenize.MWETokenizer(separator=' ')
import_terms(tokenizer, '../input/cs_terms_and_stopwords/cs_terms.txt')
keyword_df['text'] = keyword_df['text'].apply(lambda s: ' '.join(tokenize(tokenizer, s, lowercase=True, preserve_case_words=['Go', '.NET'])))

# Create keyword columns.
keyword_df['keyword'] = list(','.join(get_keywords(tokenize(tokenizer, col, lowercase=True, preserve_case_words=['Go']), 3, stopwords)) for col in keyword_df['text'])
keyword_df.head(10)


# In[ ]:


keywords_array=[]
for index, row in keyword_df.iterrows():
    keywords = row['keyword'].split(',')
    for kw in keywords:
        keywords_array.append((kw.strip(' '), row['keyword']))

kw_df = pandas.DataFrame(keywords_array).rename(columns={0:'keyword', 1:'keywords'})
kw_df.head(10)


# In[ ]:


document = kw_df.keywords.tolist()
names = kw_df.keyword.tolist()

document_array = []
for item in document:
    items = item.split(',')
    document_array.append((items))

occurrences = OrderedDict((name, OrderedDict((name, 0) for name in names)) for name in names)

# Find the co-occurrences:
for l in document_array:
    for i in range(len(l)):
        for item in l[:i] + l[i + 1:]:
            occurrences[l[i]][item] += 1

co_occur = pandas.DataFrame.from_dict(occurrences)
co_occur.head(10)


# Finally, export the term-term matrix to a csv file and import it as a `spreadsheet` in Gephi.

# In[ ]:


# Write to CSV and import it into Gephi as spreadsheet.
#co_occur.to_csv('keyword_co_occur_google.csv')


# ![](https://raw.githubusercontent.com/aesophor/NTU-CSX-DataScience2019/master/hw4-final-project/out_graphs/google_keyword_relations.png)

# We'll also wrap all of these in a function.

# In[ ]:


def create_term_term_matrix(dataframe):
    keyword_df = dataframe.copy()

    # Replace NaN columns with empty strings.
    keyword_df = keyword_df.replace(np.nan, '', regex=True)

    # Split job titles by comma into separate columns.
    keyword_df['title'] = keyword_df['title'].str.split(',', expand=True)[0]

    # Combine minimum_qual, preferred_qual and responsibilities into a single column.
    keyword_df['text'] = list(keyword_df['minimum_qual'] + keyword_df['preferred_qual'] + keyword_df['responsibilities'])

    # Drop unused columns.
    keyword_df = keyword_df.drop(['location', 'minimum_qual', 'preferred_qual', 'responsibilities'], axis=1)
    keyword_df.head()
    
    stopwords = []
    with open('../input/cs_terms_and_stopwords/stopwords.txt', 'r') as f:
        for line in f:
            if not line.startswith('#'):
                stopwords.append(line.strip())
                
    # Clean up the text.
    tokenizer = nltk.tokenize.MWETokenizer(separator=' ')
    import_terms(tokenizer, '../input/cs_terms_and_stopwords/cs_terms.txt')
    keyword_df['text'] = keyword_df['text'].apply(lambda s: ' '.join(tokenize(tokenizer, s, lowercase=True, preserve_case_words=['Go', '.NET'])))
    
    # Create keyword columns.
    keyword_df['keyword'] = list(','.join(get_keywords(tokenize(tokenizer, col, lowercase=True, preserve_case_words=['Go']), 3, stopwords)) for col in keyword_df['text'])
    
    keywords_array=[]
    for index, row in keyword_df.iterrows():
        keywords = row['keyword'].split(',')
        for kw in keywords:
            keywords_array.append((kw.strip(' '), row['keyword']))

    kw_df = pandas.DataFrame(keywords_array).rename(columns={0:'keyword', 1:'keywords'})
    
    document = kw_df.keywords.tolist()
    names = kw_df.keyword.tolist()

    document_array = []
    for item in document:
        items = item.split(',')
        document_array.append((items))

    occurrences = OrderedDict((name, OrderedDict((name, 0) for name in names)) for name in names)

    # Find the co-occurrences:
    for l in document_array:
        for i in range(len(l)):
            for item in l[:i] + l[i + 1:]:
                occurrences[l[i]][item] += 1

    return pandas.DataFrame.from_dict(occurrences)


# In summary, in order to have advantages over other applicants at Google...
# * have at least a bachelor degree, but further education is preferred.
# * master at least one general purpose programming language, such as Java, C/C++, etc.
# * have experience with [GWT](http://www.gwtproject.org/overview.html), JSP, OOP, etc.
# * have experience with Unix/Linux open source developments && kernel/device drivers.
# * have experience building large software systems.
# * have competent communication skills (both written and verbal).
# 
# Also it seems that Google cares a lot about UX and is looking forward to redesigning UI.

# ## 2. Facebook
# ### 2.1 Degree Requirements

# In[ ]:


df = pandas.read_csv('../input/raw-data/facebook_jobs.csv')
df.head(10)


# In[ ]:


keywords = ['PhD', 'Master', 'MBA', 'BA', 'BS', 'Bachelor']

# Count keyword frequency.
min_degree_reqs = count_keywords_freq(df, 'minimum_qual', keywords, none=True)
pref_degree_reqs = count_keywords_freq(df, 'preferred_qual', keywords, none=False)

print("Minimum: " + str(min_degree_reqs))
print("Preferred: " + str(pref_degree_reqs))


# In[ ]:


min_degree_df = pandas.DataFrame.from_dict(min_degree_reqs, orient='index', columns=['Count'])
pref_degree_df = pandas.DataFrame.from_dict(pref_degree_reqs, orient='index', columns=['Count'])

min_degree_df


# In[ ]:


colors = ['#958090', '#83A2BE', '#7DAEA9', '#B4BF86', '#CBB079', '#B77A76', '#707070', '#AAAAAA']

min_labels = list(min_degree_reqs.keys())
min_values = list(min_degree_reqs.values())

pref_labels = list(pref_degree_reqs.keys())
pref_values = list(pref_degree_reqs.values())


plt.figure(figsize=(15, 5))

plt.subplot(121)
plt.bar(min_labels, min_values, color=colors, width=0.5)
plt.xlabel('Degree')
plt.ylabel('Number of jobs')
plt.title('Minimum Degree Requirements')

plt.subplot(122)
plt.bar(pref_labels, pref_values, color=colors, width=0.5)
plt.xlabel('Degree')
plt.ylabel('Number of jobs')
plt.title('Preferred Degree Requirements')

plt.show()


# It seems that lots of job positions doesn't mention work experience. We'll explore further.

# ### 2.2 Work Experience Requirements

# In[ ]:


# Initialize a list with 0 from index 0 to 20.
min_exp_list = extract_experience(df, 'minimum_qual')
pref_exp_list = extract_experience(df, 'preferred_qual')


# In[ ]:


colors = ['#958090', '#83A2BE', '#7DAEA9', '#B4BF86', '#CBB079', '#B77A76', '#707070', '#AAAAAA']
labels = np.arange(len(min_exp_list))

plt.figure(figsize=(16, 5))

plt.subplot(121)
plt.bar(np.arange(11), min_exp_list[0:11], color=colors, width=0.5)
plt.xticks(labels[0:11])
plt.xlabel('Year(s)')
plt.ylabel('Number of jobs')
plt.title('Minimum Experience Requirements')

plt.subplot(122)
plt.bar(labels[0:11], pref_exp_list[0:11], color=colors, width=0.5)
plt.xticks(labels[0:11])
plt.xlabel('Year(s)')
plt.ylabel('Number of jobs')
plt.title('Preferred Experience Requirements')

plt.show()


# Hmm... so some job positions don't require educational degree and work experience. I found an interesting example which doesn't mention educational degree, but **requires practical experience** from the applicant (e.g., C/C++, open source, Linux userspace, Linux kernel skills). Personally, these skills can be self-taught or gathered from previous work experience.

# In[ ]:


print(df[df['title'] == 'Software Engineer, Linux Userspace'].iloc[0].minimum_qual)


# In[ ]:


print(df[df['title'] == 'Software Engineer, Linux Userspace'].iloc[0].preferred_qual)


# In[ ]:


print(df[df['title'] == 'Software Engineer, Linux Userspace'].iloc[0].responsibilities)


# ### 2.3 Frequently Appeared Programming Languages in Job Descriptions

# In[ ]:


# Count keyword frequency.
min_lang_reqs = count_keywords_freq(df, 'minimum_qual', langs)
pref_lang_reqs = count_keywords_freq(df, 'preferred_qual', langs)

# Some manual correction.
min_lang_reqs['Java'] -= min_lang_reqs['JavaScript']
pref_lang_reqs['Java'] -= pref_lang_reqs['JavaScript']

# Sort the dicts.
min_lang_reqs = dict(sorted(min_lang_reqs.items(), key=lambda kv: kv[1], reverse=True))
pref_lang_reqs = dict(sorted(pref_lang_reqs.items(), key=lambda kv: kv[1], reverse=True))

# Create DataFrame from dict.
min_lang_df = pandas.DataFrame.from_dict(min_lang_reqs, orient='index', columns=['Count'])
pref_lang_df = pandas.DataFrame.from_dict(pref_lang_reqs, orient='index', columns=['Count'])

min_lang_reqs


# In[ ]:


min_labels = list(min_lang_reqs.keys())
min_values = list(min_lang_reqs.values())
min_colors = [lang_colors[k] for k, v in min_lang_reqs.items()]

pref_labels = list(pref_lang_reqs.keys())
pref_values = list(pref_lang_reqs.values())
pref_colors = [lang_colors[k] for k, v in pref_lang_reqs.items()]


plt.figure(figsize=(21, 5))

plt.subplot(121)
plt.bar(min_labels, min_values, color=min_colors, width=0.5)
plt.title('Top 10 Programming Languages in Minimum Quals')

plt.subplot(122)
plt.bar(pref_labels, pref_values, color=pref_colors, width=0.5)
plt.title('Top 10 Programming Languages in Preferred Quals')

plt.show()


# In Facebook, PHP is used quite often! Anyway, let's dig deeper and create wordclouds!

# ### 2.4 Keyword Extraction: TF-IDF Wordclouds

# In[ ]:


create_wordclouds(df, '../input/wordcloud-mask/fb-icon.png', 'minimum_qual', 'preferred_qual', 'responsibilities')


# * library: boost/openmp
# * tools: git
# * machine learning: openml
# * hardware related: i2c/uart/spi/low-level/device drivers
# * parallel computation: mpi/cuda

# ### 2.5 Keyword Extraction: Term-Term Matrix Visualization

# In[ ]:


co_occur = create_term_term_matrix(df)
#co_occur.to_csv('keyword_co_occur_facebook.csv')


# ![](https://raw.githubusercontent.com/aesophor/NTU-CSX-DataScience2019/master/hw4-final-project/out_graphs/facebook_keyword_relations.png)

# ## 3. Apple
# ### 3.1 Degree Requirements

# In[ ]:


df = pandas.read_csv('../input/raw-data/apple_jobs.csv')
df.head(10)


# In[ ]:


keywords = ['PhD', 'Master', 'MBA', 'BA', 'BS', 'Bachelor']

# Count keyword frequency.
degree_reqs = count_keywords_freq(df, 'education&experience', keywords, none=True)

print("Education & Experience: " + str(degree_reqs))


# In[ ]:


degree_df = pandas.DataFrame.from_dict(degree_reqs, orient='index', columns=['Count'])
degree_df


# In[ ]:


colors = ['#958090', '#83A2BE', '#7DAEA9', '#B4BF86', '#CBB079', '#B77A76', '#707070', '#AAAAAA']

degree_labels = list(degree_reqs.keys())
degree_values = list(degree_reqs.values())

plt.figure(figsize=(15, 5))
plt.bar(degree_labels, degree_values, color=colors, width=0.5)
plt.xlabel('Degree')
plt.ylabel('Number of jobs')
plt.title('Degree Requirements')
plt.show()


# ### 3.2 Work Experience Requirements

# In[ ]:


min_exp_list = extract_experience(df, 'minimum_qual', end_year=25)


# In[ ]:


colors = ['#958090', '#83A2BE', '#7DAEA9', '#B4BF86', '#CBB079', '#B77A76', '#707070', '#AAAAAA']
labels = np.arange(len(min_exp_list))

plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1) # Drawing the 1st subplot.
plt.bar(np.arange(26), min_exp_list[0:26], color=colors, width=0.5)
plt.xticks(labels[0:26])
plt.xlabel('Year(s)')
plt.ylabel('Number of jobs')
plt.title('Experience Requirements')
for i, v in enumerate(min_exp_list[0:26]):
    plt.text(i - 0.3, v+25, str(v), color='black')
plt.show()


# Lots of job positions mention "BS" and no experience. I found an interesting example, let me sum it up real quick:
# * Bachelor's degree in CS or equivalent is required; Master's degree preferred
# * programming paradigm: OOP
# * have an understand of: scalable distributed systems (scalability and performance matters a lot)
# * core skills: strong problem solving/debugging/communicate effectively (written and verbal)
# * mindsets: results oriented/deadline driven/have desire to work in dynamic and challenging environment

# In[ ]:


print(df[df['title'] == 'Software Engineer'].iloc[11].minimum_qual)


# In[ ]:


print(df[df['title'] == 'Software Engineer'].iloc[11].responsibilities)


# In[ ]:


print(df[df['title'] == 'Software Engineer'].iloc[11]['education&experience'])


# ### 3.3 Frequently Appeared Programming Languages in Job Descriptions

# In[ ]:


# Count keyword frequency.
min_lang_reqs = count_keywords_freq(df, 'minimum_qual', langs)
pref_lang_reqs = count_keywords_freq(df, 'preferred_qual', langs)

# Sort the dicts.
min_lang_reqs = dict(sorted(min_lang_reqs.items(), key=lambda kv: kv[1], reverse=True))
pref_lang_reqs = dict(sorted(pref_lang_reqs.items(), key=lambda kv: kv[1], reverse=True))

# Create DataFrame from dict.
min_lang_df = pandas.DataFrame.from_dict(min_lang_reqs, orient='index', columns=['Count'])
pref_lang_df = pandas.DataFrame.from_dict(pref_lang_reqs, orient='index', columns=['Count'])

print(min_lang_reqs)
print(pref_lang_reqs)


# In[ ]:


min_labels = list(min_lang_reqs.keys())
min_values = list(min_lang_reqs.values())
min_colors = [lang_colors[k] for k, v in min_lang_reqs.items()]

pref_labels = list(pref_lang_reqs.keys())
pref_values = list(pref_lang_reqs.values())
pref_colors = [lang_colors[k] for k, v in pref_lang_reqs.items()]


plt.figure(figsize=(21, 5))

plt.subplot(121)
plt.bar(min_labels, min_values, color=min_colors, width=0.5)
plt.title('Top 10 Programming Languages in Minimum Quals')

plt.subplot(122)
plt.bar(pref_labels, pref_values, color=pref_colors, width=0.5)
plt.title('Top 10 Programming Languages in Preferred Quals')

plt.show()


# ### 3.4 Keyword Extraction: TF-IDF Wordclouds

# In[ ]:


create_wordclouds(df, '../input/wordcloud-mask/apple-icon.png', 'minimum_qual', 'preferred_qual', 'responsibilities')


# * criteria: hands-on experience
# * management: pdm/lims/tech-stack/ERP/OEM
# * familiar with: python/perl/\*nix
# * ability: debugging/deployment/schematics/assist

# ### 3.5 Keyword Extraction: Term-Term Matrix Visualization

# In[ ]:


co_occur = create_term_term_matrix(df)
#co_occur.to_csv('keyword_co_occur_apple.csv')


# ![](https://raw.githubusercontent.com/aesophor/NTU-CSX-DataScience2019/master/hw4-final-project/out_graphs/apple_keyword_relations.png)

# ## 4. Global Job Opportunities at Google, Facebook and Apple
# We are students from Taiwan, and we harbor ambitions of working abroad at a big tech company in a team with some amazing software engineers! Therefore, we decide to visualize all job opportunities in a world map.

# In[ ]:


df_apple = pandas.read_csv('../input/raw-data/apple_jobs.csv')
df_google= pandas.read_csv('../input/raw-data/google_jobs.csv')
df_facebook= pandas.read_csv('../input/raw-data/facebook_jobs.csv')


# In[ ]:


loc_apple = []
loc_google = []
loc_facebook = []

for a in df_apple['location']:
    loc_apple.append(a)
for b in df_google['location']:
    loc_google.append(b)
for c in df_facebook['location']:
    loc_facebook.append(c)
    
print(loc_google[0:5])


# In[ ]:


import pickle

loc_google_dic = {}
num_google_dic = {}
loc_facebook_dic = {}
num_facebook_dic = {}
loc_apple_dic = {}
num_apple_dic = {}

with open('../input/map-data/map_locations_data.pkl', 'rb') as f:
    data = pickle.load(f)
    loc_google_dic = data['loc_google_dic']
    num_google_dic = data['num_google_dic']
    loc_facebook_dic = data['loc_facebook_dic']
    num_facebook_dic = data['num_facebook_dic']
    loc_apple_dic = data['loc_apple_dic']
    num_apple_dic = data['num_apple_dic']


# In[ ]:


def merge_two_dicts(x, y):
    """ Given two dicts, merge them into a new dict as a shallow copy """
    z = x.copy()
    z.update(y)
    return z


# In[ ]:


company_locations_colors = {
    'google': 'green',
    'apple': 'red',
    'facebook': 'blue'
}


# In[ ]:


loc_arr = []
num_arr = []
long_arr = []
lati_arr = []

for key, value in loc_apple_dic.items():
    loc_arr.append(key)
    num_arr.append(num_apple_dic[key])
    long_arr.append(loc_apple_dic[key][1])
    lati_arr.append(loc_apple_dic[key][0])
company_apple_locations={'apple': {'long':long_arr,'lati':lati_arr,'num':num_arr,'loc':loc_arr }}


# In[ ]:


loc_arr = []
num_arr = []
long_arr = []
lati_arr = []

for key, value in loc_google_dic.items():
    loc_arr.append(key)
    num_arr.append(num_google_dic[key])
    long_arr.append(loc_google_dic[key][1])
    lati_arr.append(loc_google_dic[key][0])
company_google_locations={'google': {'long':long_arr,'lati':lati_arr,'num':num_arr,'loc':loc_arr }}


# In[ ]:


loc_arr = []
num_arr = []
long_arr = []
lati_arr = []

for key, value in loc_facebook_dic.items():
    loc_arr.append(key)
    num_arr.append(num_facebook_dic[key])
    long_arr.append(loc_facebook_dic[key][1])
    lati_arr.append(loc_facebook_dic[key][0])
company_facebook_locations={'facebook': {'long':long_arr,'lati':lati_arr,'num':num_arr,'loc':loc_arr }}


# In[ ]:


tem = merge_two_dicts(company_apple_locations,company_google_locations)
company_locations = merge_two_dicts(tem,company_facebook_locations)
company_locations['google']['loc'][0]


# In[ ]:


import pandas as pd # Reading csv file 
from shapely.geometry import Point # Shapely for converting latitude/longtitude to geometry
import geopandas as gpd # To create GeodataFrame

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
base = world.plot(color='white', edgecolor='black',figsize=(18,16))
base.legend(['google'], title='legend')


lon = company_locations['google']['long']
lat = company_locations['google']['lati']
key = company_locations['google']['loc']
num = company_locations['google']['num']
geometry = [Point(xy) for xy in zip(lon, lat)]
    
# Coordinate reference system : WGS84
crs = {'init': 'epsg:4326'}
    
# Creating a Geographic data frame 
for a in range(0,len(lon)):
    gdf_google_dot = gpd.GeoDataFrame(key, crs=crs, geometry=geometry)
    size = 5 + num[a]*0.4
    base.scatter(lon[a],lat[a], marker='o', color='green',s=size)
    
base.set_title('Google Job Opportunities')
base.legend(['Google'])


# In[ ]:


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
base = world.plot(color='white', edgecolor='black',figsize=(18,16))
base.legend(['apple'], title='legend')


lon = company_locations['apple']['long']
lat = company_locations['apple']['lati']
key = company_locations['apple']['loc']
num = company_locations['apple']['num']
geometry = [Point(xy) for xy in zip(lon, lat)]
    
# Coordinate reference system : WGS84
crs = {'init': 'epsg:4326'}
    
# Creating a Geographic data frame 
for a in range(0,len(lon)):
    gdf_apple_dot = gpd.GeoDataFrame(key, crs=crs, geometry=geometry)
    size = 5 + num[a]*0.4
    base.scatter(lon[a],lat[a], marker='o', color='red',s=size)
    
base.set_title('Apple Job Opportunities')
base.legend(['Apple'])


# In[ ]:


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
base = world.plot(color='white', edgecolor='black',figsize=(18,16))
base.legend(['facebook'], title='legend')


lon = company_locations['facebook']['long']
lat = company_locations['facebook']['lati']
key = company_locations['facebook']['loc']
num = company_locations['facebook']['num']
geometry = [Point(xy) for xy in zip(lon, lat)]
    
# Coordinate reference system : WGS84
crs = {'init': 'epsg:4326'}
    
# Creating a Geographic data frame 
for a in range(0,len(lon)):
    gdf_facebook_dot = gpd.GeoDataFrame(key, crs=crs, geometry=geometry)
    size = 5 + num[a]*0.4
    base.scatter(lon[a],lat[a], marker='o', color='blue',s=size)
    
base.set_title('Facebook Job Opportunities')
base.legend(['Facebook'])


# In[ ]:


world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
base = world.plot(color='white', edgecolor='black',figsize=(18,16))
base.legend(['google', 'apple', 'facebook'], title='legend')

for company, color in company_locations_colors.items():
    lon = company_locations[company]['long']
    lat = company_locations[company]['lati']
    key = company_locations[company]['loc']
    num = company_locations[company]['num']
    geometry = [Point(xy) for xy in zip(lon, lat)]
    
    df = pandas.DataFrame(
    {'City':key,
     'Company':company,
     'Number of jobs':num})
    
    # Coordinate reference system : WGS84
    crs = {'init': 'epsg:4326'}
    
    # Creating a Geographic data frame 
    if company is 'google':
        gdf_google = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
        gdf_google.plot(ax=base, marker='o', color=color, markersize=20)
    if company is 'apple':
        gdf_apple = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
        gdf_apple.plot(ax=base, marker='o', color=color, markersize=20)
    if company is 'facebook':
        gdf_facebook = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
        gdf_facebook.plot(ax=base, marker='o', color=color, markersize=20)
    
base.set_title('Google, Apple, Facebook Global Job Opportunities')
base.legend(['Google', 'Apple', 'Facebook'])


# Now we're going to create an enhanced version of this map with the ability to interact with its users!

# In[ ]:


from bokeh.io import output_file, output_notebook, show
from bokeh.plotting import figure, output_file, show, ColumnDataSource
output_notebook()


# In[ ]:


def convert_GeoPandas_to_Bokeh_format(gdf):
    """
    Function to convert a GeoPandas GeoDataFrame to a Bokeh
    ColumnDataSource object.
    
    :param: (GeoDataFrame) gdf: GeoPandas GeoDataFrame with polygon(s) under
                                the column name 'geometry.'
                                
    :return: ColumnDataSource for Bokeh.
    """
    gdf_new= gdf.drop('geometry', axis=1).copy()

    gdf_new['x'] = gdf.apply(getGeometryCoords, 
                             geom='geometry', 
                             coord_type='x', 
                             shape_type='polygon', 
                             axis=1)
    
    gdf_new['y'] = gdf.apply(getGeometryCoords, 
                             geom='geometry', 
                             coord_type='y', 
                             shape_type='polygon', 
                             axis=1)
    
    return ColumnDataSource(gdf_new)


def getGeometryCoords(row, geom, coord_type, shape_type):
    """
    Returns the coordinates ('x' or 'y') of edges of a Polygon exterior.
    
    :param: (GeoPandas Series) row : The row of each of the GeoPandas DataFrame.
    :param: (str) geom : The column name.
    :param: (str) coord_type : Whether it's 'x' or 'y' coordinate.
    :param: (str) shape_type
    """
    
    # Parse the exterior of the coordinate
    if shape_type == 'polygon':
        try:
            exterior = row[geom].exterior
        except:
            exterior = row[geom].geom[0].exterior
        
        if coord_type == 'x':
            # Get the x coordinates of the exterior
            return list( exterior.coords.xy[0] )    
        
        elif coord_type == 'y':
            # Get the y coordinates of the exterior
            return list( exterior.coords.xy[1] )

    elif shape_type == 'point':
        exterior = row[geom]
    
        if coord_type == 'x':
            # Get the x coordinates of the exterior
            return  exterior.coords.xy[0][0] 

        elif coord_type == 'y':
            # Get the y coordinates of the exterior
            return  exterior.coords.xy[1][0]


# In[ ]:


world.head()


# In[ ]:


def explode(gdf):
    gs = gdf.explode()
    gdf2 = gs.reset_index().rename(columns={0: 'geometry'})
    gdf_out = gdf2.merge(gdf.drop('geometry', axis=1), left_on='level_0', right_index=True)
    gdf_out = gdf_out.set_index(['level_0', 'level_1']).set_geometry('geometry')
    gdf_out.crs = gdf.crs
    return gdf_out


# In[ ]:


world_Source = convert_GeoPandas_to_Bokeh_format(explode(world))


# In[ ]:


from bokeh.models import (
    Range1d,
    GeoJSONDataSource,
    HoverTool,
    LinearColorMapper,
    GMapPlot, GMapOptions, ColumnDataSource, 
    Circle, DataRange1d, PanTool, WheelZoomTool, BoxSelectTool
)


# In[ ]:


gdf_google['x'] = gdf_google.apply(getGeometryCoords, 
                                 geom='geometry', 
                                 coord_type='x', 
                                 shape_type='point',
                                 axis=1)
                                 
gdf_google['y'] = gdf_google.apply(getGeometryCoords, 
                                 geom='geometry', 
                                 coord_type='y', 
                                 shape_type='point',
                                 axis=1)

gdf_google = gdf_google.drop(['geometry'],axis=1)

point_source_google = ColumnDataSource(data=dict(x=gdf_google['x'],
                                      y=gdf_google['y'],
                                      City=gdf_google['City'].values,
                                      Company=gdf_google['Company'].values,
                                      NumberofJobs=gdf_google['Number of jobs'].values))


# In[ ]:


gdf_apple['x'] = gdf_apple.apply(getGeometryCoords, 
                                 geom='geometry', 
                                 coord_type='x', 
                                 shape_type='point',
                                 axis=1)
                                 
gdf_apple['y'] = gdf_apple.apply(getGeometryCoords, 
                                 geom='geometry', 
                                 coord_type='y', 
                                 shape_type='point',
                                 axis=1)

gdf_apple = gdf_apple.drop(['geometry'],axis=1)

point_source_apple = ColumnDataSource(data=dict(x=gdf_apple['x'],
                                      y=gdf_apple['y'],
                                      City=gdf_apple['City'].values,
                                      Company=gdf_apple['Company'].values,
                                      NumberofJobs=gdf_apple['Number of jobs'].values))


# In[ ]:


gdf_facebook['x'] = gdf_facebook.apply(getGeometryCoords, 
                                 geom='geometry', 
                                 coord_type='x', 
                                 shape_type='point',
                                 axis=1)
                                 
gdf_facebook['y'] = gdf_facebook.apply(getGeometryCoords, 
                                 geom='geometry', 
                                 coord_type='y', 
                                 shape_type='point',
                                 axis=1)

gdf_facebook = gdf_facebook.drop(['geometry'],axis=1)

point_source_facebook = ColumnDataSource(data=dict(x=gdf_facebook['x'],
                                      y=gdf_facebook['y'],
                                      City=gdf_facebook['City'].values,
                                      Company=gdf_facebook['Company'].values,
                                      NumberofJobs=gdf_facebook['Number of jobs'].values))


# In[ ]:


TOOLS = "pan,wheel_zoom,box_zoom,reset,hover,save"


# In[ ]:


Elevated = figure(title="Google, Apple, Facebook Locations across the Globe",
            tools=TOOLS,
            x_axis_location=None, 
            y_axis_location=None, plot_width=900, plot_height=450)   


# In[ ]:


Elevated.multi_line('x', 
                    'y', 
                    source=world_Source, 
                    color="gray", 
                    line_width=1)


# In[ ]:


Elevated.circle('x', 
                'y', 
                source=point_source_google, 
                size=5,
                color='green')


# In[ ]:


Elevated.circle('x', 
                'y', 
                source=point_source_apple, 
                size=5,
                color='red')


# In[ ]:


Elevated.circle('x', 
                'y', 
                source=point_source_facebook, 
                size=5,
                color='blue')


# In[ ]:


hover = Elevated.select_one(HoverTool)
hover.point_policy = "follow_mouse"

TOOLTIPS = """
    <div>
        <div>
            <span style="font-size: 15px; font-weight: bold;">City:</span>
            <span style="font-size: 17px; font-weight: bold;">@City</span>
        </div>
        <div>
            <span style="font-size: 15px;">Company:</span>
            <span style="font-size: 15px; color: #696;">@Company</span>
        </div>
        <div>
            <span style="font-size: 15px;">Location:</span>
            <span style="font-size: 15px; color: #696;">($x, $y)</span>
        </div>
        <div>
            <span style="font-size: 15px;">NumberofJobs:</span>
            <span style="font-size: 15px; color: #696;">@NumberofJobs</span>
        </div>
    </div>
"""
hover.tooltips=TOOLTIPS


# In[ ]:


show(Elevated)


# ## 5. Conclusion
# Conclusively, in order to have an advantage among all job applicants at Google, Facebook and Apple...
# * Having at least a **Bachelor degree** would be safer (unless you have some jaw-dropping personal projects); **further education is a plus**!
# * Having **practical experience** is vital. Previous work experience isn't necessary for every job, but most job positions metion lots of skills you need to have, and I believe a good way to build up those experience is through **personal projects**.
# * In big tech companies, they care a lot about system **scalability, stability and performance** (every msec counts).
# * Having **competent communication skills** (both written and verbal) is important, since you will be working in a TEAM with lots of amazing engineers!
# * Google cares a lot about UI/UX and is currently looking forward to redesigning it. Facebook has lots of job positions related to hardware and firmware. In Apple, some well-known management methodologies (e.g., pdm, lims, tech-stack, ERP...) are mentioned.
# * master at least one general purpose programming language, such as Java, C/C++, Python etc. Being very good at one programming language is very important.
# * have experience with **OOP** and Unix/Linux **open source** developments.
# ---
# **Last but not least, a shoutout to our supervisor, Pecu Tsai, as well as her TAs, who spent their time training us the skills we need!**
