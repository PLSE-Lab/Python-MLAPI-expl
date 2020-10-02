#!/usr/bin/env python
# coding: utf-8

# # The City of Los Angeles Job Posting Diversity Analysis
# 
# As many of its employees become eligible for retirement, the City of Los Angeles is interested in finding ways to improve diversity among their applicant pool.
# 
# Compared to other data science competitions (where the objective function we are trying to optimize is "given from on high"), the question of how to improve recruitment requires a different thought pattern -- we must consider how people interact with job postings and prioritize actionable/interpretable results when constructing our model and making recommendations.

# Given the more subjective nature of "goodness of a piece of text", a cursory look through the postings may suggest some good lines of inquiry. After having taken a look, there are some immediate thoughts/suggestions/lines of inquiry that may be worth pursuing:
# 
# 1. The typical job posting's length and diction feel like a barrier to entry.
# 
# Some off-the-cuff recommendations:
# 
# 1. *Show information that is relevant and actionable to applicants to reduce friction.* For example, for relevant mutually exclusive information, there could be two modes: (1) promotional applicants (within City of LA); (2) non-City of LA prospecitve applicants. Also, perhaps have a collapsed "FAQ" portion to the job listing in lieu of the potentially opaque "Notes" listed underneath headings. If there are liability concerns, the original text can perhaps be indicated as a page prior to application submission. But for many, it is not critical to know that, for example, the examination is based on a validation study. Such information would likely be ignored, or make one wonder what is meant by "validation study", or else simply add to the length of the job listing (increasing the likelihood of applicants either skimming the listing or forgoing the application process altogether).
# 
# After further analysis, I aim to show whether and with what effect size post diction and (especially) length, among other factors, change application rates across different groups.

# # Initial Plan
# 
# The City of Los Angeles (LA) is interested in greater diversity. Using race/ethnicity as a (tenuous) proxy, we can see how job application proportions correspond to local 2010 census data. (This of course assumes that we are interested in having the employee composition reflect that of the 2010 local census data; populations may have shifted in the meantime and applicants are not limited to being/having been in the LA area.)
# 
# Census dataset is available on Kaggle (https://www.kaggle.com/cityofLA/los-angeles-census-data). Job applicant data is from https://data.lacity.org/A-Prosperous-City/Job-Applicants-by-Gender-and-Ethnicity/mkf9-fagf and contains information on some applications during Fiscal Years (FY) 2013-2014 and 2014-2015.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import namedtuple

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from datetime import datetime
import re
import os
import itertools # mainly for `chain`
import spacy # for dependency trees, word vectors, etc.
import plotnine as plt
from tqdm import tqdm

# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Los Angeles Census Data
# 
# But first, for something completely different!
# 
# This focuses on census data from 2010 to see what the relative gender/ethnicity breakdowns of the city are.

# In[ ]:


census_dir = os.listdir('../input/los-angeles-census-data/')
census_path = '../input/los-angeles-census-data/'
# census_dir


# In[ ]:


census_df = pd.read_csv(
    os.path.join(census_path, 'census-data-by-council-district.csv')
) # use 'council-district' because Hispanic data is intact
# census_df.head()


# In[ ]:


gender_fracs = (
    census_df.select(lambda x: 'pop' in x and 'male' in x.lower(), axis = 'columns')
     .aggregate(sum)
     /
        census_df['Pop2010'].aggregate(sum)
).rename('gender_fracs')
# print(gender_fracs)
gender_fracs.plot.bar()


# In[ ]:


# # hacking together something to make ggplot be agreeable -- faux DF
# test_df = pd.DataFrame(gender_fracs)
# test_df['index'] = test_df.index

# (plt.ggplot(test_df, plt.aes('index','gender_fracs')) +
#  plt.geom_bar(stat = 'identity')
# )


# In[ ]:


# get breakdown by race
race_counts = (census_df.select(lambda x: 'pop' in x and not 'male' in x.lower(), axis = 'columns')
 .sum(axis='index') # counts by race
).rename('race_counts')

# have both counts and fractions in same dataframe
race_df = pd.DataFrame(data = {'race_counts': race_counts, 
                     'race_fracs': race_counts/census_df['Pop2010'].aggregate(sum)
                    }
            )
# print(race_df)
race_df['race_fracs'].plot.bar()


# The self-reported ethnicity data will be used to compare against the composition of applicants to previous job postings. (**Note** that these fractions need not sum to $1$.)
# 
# From the plots, we see that 
# 1. there was a fairly even split between males and females, and
# 2. the most commonly reported ethnicities were (1) white; (2) Hispanic; (3) Asian; (4) black.

# # Previous Job Applicant Data
# 
# These come from the LA City website as listed above.
# 
# There may even be some correspondence between job postings here and job postings in the provided data! Worth looking into, especially when looking for disparities in desired applicant makeup. (Of course, we do not have information about who was eventually hired.)

# In[ ]:


applicant_dir = os.path.join('..','input', 'city-of-los-angeles-job-applicant-composition')
raw_applicant_df = pd.read_csv(os.path.join(applicant_dir, 'rows.csv'))
# remove those with missing entries, see how big the difference is
applicant_df = raw_applicant_df.dropna()
print("The ratio of cleaned-up entries (no NaN's) to raw entries is {0:.4f}.".
      format(len(applicant_df)/len(raw_applicant_df)))
applicant_df


# In[ ]:


# split string column via 'extract'
# ala https://stackoverflow.com/a/21296915
regexp_extractor = r'.*?(?P<Job_Title>[0-9]*[a-zA-Z ]+).*?(?P<Job_Number>\d+)'
# test = applicant_df['Job Description'].str.extract(regexp_extractor)


applicant_df = (applicant_df['Job Description'].str.extract(regexp_extractor)
 .join(
     applicant_df.drop(['Fiscal Year', 'Job Number', 'Job Description'], axis = 'columns'),
      how = 'inner'
     )
)
# merge lines with same job title *and* job number
applicant_df = applicant_df.groupby(['Job_Title', 'Job_Number'], as_index = False).sum()


# In[ ]:


ethnicity_indices = ['Black', 'Hispanic', 'Asian',
       'Caucasian', 'American Indian/ Alaskan Native', 'Filipino',
       'Unknown_Ethnicity']
gender_indices = ['Male', 'Female', 'Unknown_Gender']

ethnicity_nums = (applicant_df
 .groupby(applicant_df['Job_Number']
         )
#  .apply(matcher))
 .sum()
 [ethnicity_indices]
 .aggregate(sum, axis = 'index')
) 
ethnicity_nums.plot.bar()
# probably dubious; if diversity (wrt ethnicity) has been an issue


# The above may make measures of diversity (with respect to having a high number of applicants from underrepresented minorities) seem good. But it's worth remembering that
# 
# 1. This is incomplete data from circa 2014. 
# 2. [Simpson's Paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox) may be at play, where things may look "good" in the aggregate but are in fact "bad" at the stratified level, where it actually matters (or vice-versa). For example, perhaps URMs are overrepresented in applying to less influential positions. Indeed, it's worth noting the most popular applications:

# In[ ]:


n = 10
df_popular = applicant_df.sort_values('Apps Received', axis = 'index', ascending = False).head(n)
print("The top {0} most popular positions account for {1:.1%} of all applications."
     .format(n, df_popular['Apps Received'].sum()/applicant_df['Apps Received'].sum()))
df_popular.plot(x='Job_Title', y = 'Apps Received', kind='bar')


# In[ ]:


ethnicity_df_popular = df_popular.select(lambda x: x in ethnicity_indices, axis = 'columns').sum()
(ethnicity_df_popular/ethnicity_df_popular.sum()).plot(kind = 'bar')


# Many of the most applied-for positions (Customer Service Representative, Meter Reader) do not have much impact on larger operations of the City's decision-making, and we find an overrepresentation of underrepresented minorities within these positions. Indeed, it looks like Simpson's paradox strikes, and we will have to be wary of mindless aggregation.

# # City of LA Job Bulletins: A Whole Lot of Cleanup
# 
# There is quite a bit of data wrangling to be done in order to build our desired structured database/CSV. We shall use regular expression matching (using `re`) and dependency tree parsing (using a `spacy` model).
# 
# NB: We could perhaps use a deep-learning model like [BERT](https://arxiv.org/abs/1810.04805) to try and pull out the semantic understandings in a more black-box manner. But given the highly structured nature of the provided job postings, BERT seems to bring in more complexity and compute requirements than is really warranted (especially if fine-tuning is required), and we may end up having to massage BERT's outputs afterward regardless.

# In[ ]:


cityofla_root = '../input/data-science-for-good-city-of-los-angeles/cityofla/CityofLA'
bulletins_dir = os.path.join(cityofla_root, 'Job Bulletins')
additional_dir = os.path.join(cityofla_root, 'Additional data')
# os.listdir(bulletins_dir)[:10]


# In[ ]:


# let's see what fields we need to fill
# the specification itself
spec_csv = pd.read_csv(os.path.join(additional_dir, 'kaggle_data_dictionary.csv'))
# the provided example
example_csv = pd.read_csv(os.path.join(additional_dir, 'sample job class export template.csv'))
entries = list(spec_csv['Field Name'].unique())
entries


# In[ ]:


job_titles = []
with open(os.path.join(additional_dir, 'job_titles.csv')) as f:
    job_titles = [job.strip().title() for job in f.readlines()]
job_titles[:10]


# In[ ]:


# example_csv.head()


# So the sample CSV would like to have multiple rows for each observation (job listing).
# 
# We will first collect as much information as we can in the most convenient manner into one row for the purposes of analysis.

# In[ ]:


n_char = ' #newline# ' #to replace newlines
t_char = ' #tab# ' #to replace tab

def load_file(fn,root_dir = bulletins_dir,  newline_repl = n_char, tab_repl = t_char, encoding = 'utf-8'):
    """
    Load text from filename.
    Replace newlines ('\r\n' or '\n' or '\r') and tabs ('\t') with newline_repl and tab_repl, respectively.
    Return dictionary with fields:
    - 'fn' : original filename
    - 'text': text of file, as substituted above
    """
    with open(os.path.join(bulletins_dir, fn), mode = 'r', encoding = encoding) as f:
        text = (f.read()
                .replace('\r\n', '\n').replace('\r', '\n').replace('\n', newline_repl)
                .replace('\t', tab_repl))
    
    return {'fn': fn, 'text': text}

test_fn = "ADMINISTRATIVE ANALYST 1590 060118.txt"
d = load_file(test_fn, encoding = 'latin-1')


# ## Text Extraction: Regular Expression City
# 
# Let's try and pull out useful chunks, i.e., chunks that 
# 
# 1. chunks that are directly related to a field requested in the specification; or
# 2. chunks which imply a certain context in which some assumptions can be made (for example: if under the "Requirements" tab, a mention of "experience" probably means that whatever is mentioned in that sentence is a requirement for the position)

# In[ ]:


marker_newheading = n_char + '[A-Z ]{4,}' + n_char # probably if all-caps are floating on their own, it's a heading
    # it seems related notes follow each heading with "NOTES:" (with a colon ':')
    # we'll keep these under the same heading for now, perhaps will separate later

pattern_job_title = '(?P<text>^[A-Z ]+)' # because one contiguous string and job is always first, this is enough
pattern_class_code = '(?P<label>[cC]lass [cC]ode)\s*:?\s*(?P<text>[0-9]+)' #'\s' = whitespace (includes ':' in this case)
pattern_open_date = '(?P<label>[oO]pen [dD]ate)\s*:?\s*(?P<text>[0-9-]+)' # should parse via datetime

pattern_chunk_duties = 'DUTIES(?P<text>.*?)' + marker_newheading
pattern_chunk_salary = 'SALARY\s*?(' + n_char + ')*(?P<text>.*?)' + marker_newheading # further processing if DWP is mentioned
pattern_chunk_requirements =  ('(?P<label>REQUIREMENT(S?)(\/MINIMUM QUALIFICATION(S?))?).*?' 
                        + n_char 
                        + '(?P<text>.*?)'
                        + marker_newheading)
pattern_chunk_process_notes = 'REQUIREMENT(S?)(.*?)NOTE(S?)(:?)(?P<text>.*?)WHERE TO APPLY' # Requirement Notes follow requirements section

pattern_is_college = '((Graduation)|(degree from)) .*? (?P<text>(college)|(university))'


# pattern_exp_length_unspecified = r'(?P<unit>(year)|(month)|(day)|(hour))(s?).*?(experience)'


# a dictionary we'll be able to iterate through later
# re.compile() for some optimization benefits
pattern_dict = {k:re.compile(v) for (k,v) in globals().items() if k.startswith('pattern') and type(v) == str}


# In[ ]:


def split_list(l, start_func, end_func = None):
    """Given list, return three slices of the list (pre,mid,post):
    - first element of mid is the first element where start_func evaluates to True
    - last element of mid is the element *preceding* the elemtn where end_func evaluates to True,
    or the end of the list if no such element exists.
    
    If instead end_func is set to None (default), will add to mid
    only for the contiguous elements for which start_func evaluates to True
    
    e.g.: start_func = (lambda x: x > 2), end_func = (lambda x: x < 7)
    split_list([1,2,9,13,9,4],start_func,end_func) -> [[1,2],[9,13,9],[4]]
    """
    if end_func is None:
        def end_func(item):
            return not start_func(item)
    
    pre,mid,post = [],[],[]
    start_mid = False
    stop_mid = False
    # stop_mid is a "higher-priority" flag than start_mid
    # so check/set that flag first in loop
    for item in l:
        if stop_mid or (start_mid and end_func(item)):
            stop_mid = True
            post.append(item)
        elif not start_mid:
            if not start_func(item):
                pre.append(item)
            elif (not stop_mid and start_func(item)):
                start_mid = True
                mid.append(item)
        elif (start_mid and not end_func(item)):
            mid.append(item)

    return [pre,mid,post]
            
# unit test
start_func = lambda x: x > 2
end_func = lambda x: x < 7
assert split_list([1,2,9,13,9,4], start_func, end_func) == [[1, 2], [9, 13, 9], [4]]

end_func = None
assert split_list([1,2,9,12,0,-1,12], start_func) == [[1, 2], [9, 12], [0, -1, 12]]


# In[ ]:


def _process_simple(text, key, pattern):
    """Basic method for simple regex processing.
    Returns a dictionary of {key: result of regexp.groupdict()['text'].strip()}
    after reverting 'text' to unsubstituted version
    """
    try:
        match_text = re.search(pattern, text).groupdict()['text']
        m = match_text.replace(n_char, '\n')
        return {key:m.strip()}
    except AttributeError: # no match! So `groupdict` is not an attribute
        return {key:None}

def process_job_title(text):
    return _process_simple(text, 'job_title', pattern_job_title)

def process_class_code(text):
    return _process_simple(text, 'class_code', pattern_class_code)

def process_duties(text):
    return _process_simple(text, 'duties', pattern_chunk_duties)

def process_is_college_required(text):
    res = _process_simple(text, 'is_college_required', pattern_is_college)
    res['is_college_required'] = True if res['is_college_required'] else False
    return res


# In[ ]:


def process_open_date(text, in_fmt = "%m-%d-%y", out_fmt = '%m/%d/%Y'):
    try:
        date_str = re.search(pattern_open_date, text).groupdict()['text']
        date = datetime.strptime(date_str, in_fmt)
        # desired output format: 10/27/2017
        return {'open_date':date.strftime(out_fmt)}
    except:
        # it's a different format than expected
        # we're going to be lazy for now
        return {'open_date':None}


# In[ ]:


def process_salary(text):
    """
    Given full text,
    Return dict of
    - keys: 'la' or 'dwp'
    - values: list of tuples for start and end ranges (as ints)
    
    (Would break if salaries are mentioned in more than two lines.)
    """
    sal_match_str = (r'\$(?P<salary_start>[0-9,]+) to \$(?P<salary_end>[0-9,]+)')
    dwp_str = 'Department of Water and Power'
    
    pre_dict = _process_simple(text, 'sal', pattern_chunk_salary)
    salary_text = pre_dict['sal']
    if not salary_text: # no salary given
        return None
    
#     salary_text = pattern_dict['pattern_chunk_salary'].search(text).groupdict()['text']
#     text_list = salary_text.split(n_char)
    text_list = salary_text.split('\n') # _process_simple replaced n_char with '\n'
    text_list = [s for s in text_list if '$' in s]
    sal_tups_dict = {s:re.findall(sal_match_str, s) for s in text_list}
    sal_tups_dict = {k:v for k,v in sal_tups_dict.items() if v != []}
    
    temp = {}
    # clean up into dictionary of lists
    for k,sal_list in sal_tups_dict.items():
        # 'clean up' sal_list
        nums_list = []
        for tup in sal_list:
            num_range = [int(re.sub(',','',t)) for t in tup]
            nums_list.append(num_range)
        
        if dwp_str in k:
            temp['dwp'] = nums_list
        else:
            temp['la'] = nums_list
            
    # now we flatten into dictionary of "simple" key-value pairs
    # for inclusion into a dataframe
    res = {}
    label_prefixes = ['start', 'end']
    for key, l_val in temp.items():
        for i,tup in enumerate(l_val):
            for j,val in enumerate(tup):
                label = 'salary_{0}_{1}_listing_{2}'.format(label_prefixes[j], key, i+1)
                res[label] = val
                
    return res


# In[ ]:


def preprocess_requirements(text):
    """
    Separate full text into list of sentences corresponding to different requirements or subrequirements.
    Returns a dictionary of
    requirement_subheading:sent (e.g. '1a': (subsection 1a))
    as well as a key 'misc' containing a string not tied to a specific (sub)heading.
    """
    sents = [s for s in 
             pattern_dict['pattern_chunk_requirements'].search(text).groupdict()['text'].split(n_char) 
             if len(s) > 0]
    
    num_str = '123456789'
    split_sents = {}
    pre = sents
    
    for s in num_str:
        split_str = '\A(?P<label>[{}a-z])\.'.format(s) # match with strings start with e.g. '{s}.' and also any 'a.','b.', 
        splitter = re.compile(split_str)
        pre,mid,post = split_list(pre, start_func = splitter.search)
        # split_sents[s] = mid
        # build specific labels (some repeated searches but oh well) 
        labels = [(s + splitter.search(sent).groupdict()['label']) 
                  if s != splitter.search(sent).groupdict()['label']
                  else s for sent in mid
                 ]
        split_sents.update(zip(labels, mid))
        pre.extend(post)
    
    split_sents['misc'] = '\n'.join(pre) # keep any "uncaptured" lines as a long str

    
    
    return split_sents


def process_majors(text):
    """
    Returns a string of majors relevant to the listing, separated by '|'
    ('' if no relevant majors).
    """
    dict_preprocessed = preprocess_requirements(text)
    
#     dict_processed = {}
#     for key, val in dict_preprocessed.items():
#         try:
#             dict_processed[key] = get_major_requirement(val)
#         except:
#             print('key ={}, val = {}'.format(key, val))
#             raise
    
    dict_processed = {key:get_major_requirement(val) for key,val in dict_preprocessed.items()}
    temp = [v for v in list(dict_processed.values()) if v]
    majors_list = ['|'.join(v) for v in temp]
    # flatten list
#     majors_list = itertools.chain.from_iterable(majors_list)
    majors_list = [m for m in majors_list if m != '']
    return {'majors':'|'.join(majors_list)}
        


# In[ ]:


file = load_file('SYSTEMS ANALYST 1596 102717.txt')
# file = load_file('ADMINISTRATIVE ANALYST 1590 060118.txt')
text = file['text']


# ## Text Extraction: Dependency tree parsing (A ton of tree traversal)
# 
# We'll work with the dependency trees generated by Spacy to aid in grabbing more complex fields.

# In[ ]:


import spacy


# In[ ]:


# en_nlp = spacy.load('en')
en_nlp = spacy.load('en_core_web_lg')


# If properly loaded, Spacy comes with pre-trained dependency parsers. These dependency trees (while not perfect) do a pretty good job on the sentence level and will be helpful in extracting the more complicated pieces of information requested for our final CSV.
# 
# An example of a parsed sentence is shown below:

# In[ ]:


s = """1. Graduation from an accredited four-year college or university with a major in Computer Science, Philosophy of Science, Information Systems, or Geographical Information Systems; or"""
doc = en_nlp(s)
spacy.displacy.render(doc)


# In[ ]:


def print_node(tok):
    """Print human-readable information on Spacy Token."""
    print("Text:{}, POS:{}, tag:{}, dep:{}".format(tok.orth_, tok.pos_, tok.tag_, tok.dep_))


# In[ ]:


def text_to_num(textnum):
    """
    Converts text to number.
    Currently a brute-force/dictionary-based implementation for some
    single-token words (handles "zero" through "twenty")
    (Should handle current use-case.)
    """
    num_dict = {
        'zero': 0,
        'one':1,
        'two':2,
        'three':3,
        'four':4,
        'five':5,
        'six':6,
        'seven':7,
        'eight':8,
        'nine':9,
        'ten':10,
        'eleven':11,
        'twelve':12,
        'thirteen':13,
        'fourteen':14,
        'fifteen':15,
        'sixteen':16,
        'seventeen':17,
        'eighteen':18,
        'nineteen':19,
        'twenty':20
    }
    if textnum in num_dict.keys():
        return num_dict[textnum]
    
    # maybe a str'd number like "1,000"
    s = re.sub(',','',textnum)
    try:
        return int(s)
    except ValueError:
        # still no good, huh? ...
        print("text_to_num didn't work! Number: {}".format(textnum))
    return None


# In[ ]:


test = """
1. Graduation from an accredited four-year college or university with a major in Computer Science, Information Systems, or Geographical Information Systems; or
2. Graduation from an accredited four-year college or university and two years of full-time paid experience in a class at the level of Management Assistant which provides experience in:
a. the development, analysis, implementation or major modification of new or existing computer-based information systems or relational databases; or
b. performing cost benefit, feasibility and requirements analysis for a large-scale computer-based information system; or
c. performing system implementation and support activities including software and hardware acquisition, installation, modifications to system configuration, system and application upgrade installation; or
3. Two years of full-time paid experience as a Systems Aide with the City of Los Angeles; and
a. Satisfactory completion of four courses, of at least three semester or four quarter units each, in Information Systems, Systems Analysis, or a closely related degree program, professional designation, or certificate program from an accredited college or university.
b. At least three of the courses must be from the core courses required in the program, and one course may be from either the required core courses or the prescribed elective courses of the program.  A course in systems analysis and design is especially desired, but not required.
"""

test_doc = en_nlp(test)

s = """1. Graduation from an accredited four-year college or university with a major in Computer Science, Philosophy of Science, Information Systems, or Geographical Information Systems; or"""
doc = en_nlp(s)


# In[ ]:


def find_lemma(lemma, doc, start = 0):
    """
    NOTE: Must use the lemma form.
    type(lemma) == str, 
    type(doc) == spacy.Doc (or some other iterable containing spacy.Tokens)
    (if str, doc is converted to a Doc)
    
    Returns the *first* instance of token with lemma from `lemma` in the 
    document itself if found, else None.
    'start' determines the index from which token of the doc to start looking
    (this is a search *to the right*).
    """
    
    for token in doc[start:]:
        if token.lemma_ == lemma:
            return token
    return None

def is_proper_context(lemmas, doc, need_all = False):
    """
    Sees whether lemmas (str) are found in doc.
    Returns boolean True if yes.
    If need_all flag is True, only return True if all lemmas are in doc.
    """
    if need_all:
        for lemma in lemmas:
            if find_lemma(lemma, doc) is None:
                return False
        return True
    else:
        for lemma in lemmas:
            if find_lemma(lemma, doc):
                return True
        return False
        


# In[ ]:


# for seeing if a driver's license is required
# TODO: Seems that with a bit more abstractness, could be used to
# deal with more questions regarding context
def get_drivers_license_requirement(sent):
    """
    Checks if sent (a spacy.Doc or str) suggests a driver's license is required.
    Returns None if unknown from sentence, 'P' if possibly required, 'R' if required, 'N' if not required.
    """
    if type(sent) == str:
        sent = en_nlp(sent)
    
    ret = None
    
    target = find_lemma('require', sent)
    if target is None or not is_proper_context(["driver", "license"], doc = sent, need_all = True):
        return None # may not be relevant
    
    
    is_negated = ((sum(['neg' == t.dep_ for t in target.children]) % 2) == 1)
    is_ambiguous = (sum(['may' == t.lemma_ for t in target.children]) > 0)
    
    if is_ambiguous:
        ret = 'P'
    else:
        ret = 'N' if is_negated else 'R'
    
    return ret

# unit test
assert get_drivers_license_requirement(sent = en_nlp("This position may not require a driver's license.")) == 'P'

def process_drivers_license_requirement(text):
    try:
        sents = re.search(pattern_chunk_process_notes,text).groupdict()['text']
    except:
        return {'drivers_license_req':''}
    sents = sents.split(n_char)
    temp = [get_drivers_license_requirement(s) for s in sents]
    temp = [t for t in temp if t]
#     print(temp)
    # TODO: really hacky/questionable
    try:
        temp = temp[0:1] # only keep first mention
    except: # nothing in list -- ''.join(temp) will return '' without extra processing
        pass
    return {'drivers_license_req':''.join(temp)}

# process_drivers_license_requirement(text)


# In[ ]:


def traverse_tree(start_node, is_match, process_node, top_down = True):
    """
    Traverse tree, starting from start_node.
    If is_match(node) returns True, perform process_node(node).
    Otherwise, keep searching.
    If top_down is True (default), traverse top-down (root to leaves).
    If False, travel up the tree (from passed-in leaf toward root).
    
    Returns a list of process_node(node) results.
    """
    # traverse down from 'major' until the first proper-noun tag (t.tag_ == 'NNP') is found
    # then use children's (t.dep_ == 'compound') [merge tokens] and (t.dep_ == 'conj') [separate entity]
    # to build spans for the majors.
    # NOTE: This will almost surely fail for long majors with commas and "and"s
    # Will try triaging with whether 'or' vs 'and' conj is in children of "major"
    
    # going down or up?
    cont = ''
    if top_down:
        cont = 'children'
    else:
        cont = 'ancestors'
    
    stack = [start_node]
    result = []
    cur = None
    while len(stack) > 0:
        cur = stack.pop()   
        if is_match(cur): # "start" of interesting span
            result.append(process_node(cur))
        else:
            stack.extend(list(cur.__getattribute__(cont)))
#             stack.extend(cur.children) # keep hunting
    return result


# In[ ]:


# target = find_lemma('major', doc)
# ## have moved functions into major_requirement
# # res = traverse_tree(target, is_match, process_node) 
# # res # one too many layers
# # list(itertools.chain.from_iterable(res))


# In[ ]:


def get_major_requirement(sent):
    """
    (If type(sent) == str, will convert to spacy.Doc first)
    
    Check which majors are mentioned as satisfying the requirement.
    Returns None if irrelevant sentence, otherwise a list of the majors (list of str).
    """
    if type(sent) == str:
        sent = en_nlp(sent)
    
    if not (is_proper_context(['graduation'], sent) and is_proper_context(['college', 'university'], sent)):
        return None
    
    node_major = find_lemma('major', sent)
    
    if not node_major:
        return None
    # have a "major *in*" {major}
#     target = node_major
    try:
        target = [n for n in node_major.children if n.orth_ == 'in'][0]
    except IndexError: # no specific major
        return None
    
    # Specific is_match and process_node functions for traverse_tree, for this use-case
    
    def is_match(node):
        return (node.tag_ == 'NNP')

    def process_node(cur):
        # NOTE: With the spacy builtins token.conjuncts and token.subtree, this is more procedural/less recursive

    #             print("cur:{}".format(cur))
        ll_res = []
        conj_list = [cur]
        conj_list.extend(cur.conjuncts)
    #             print(conj_list)
        for conj in conj_list:
            chunk_list = [child for child in conj.children if child not in conj_list]  # make chunks mutually exclusive
    #                 print("Chunk List:{}".format(chunk_list))
            res = [list(child.subtree) for child in chunk_list] # get ME subtrees
            res.append([conj])
    #                 print(temp)
            res = list(itertools.chain.from_iterable(res)) # flatten list
    #                 print("After:{}".format(temp))
                # add coordinating conjunctive if exists
            for node in res:
                if node.dep_ == 'cc' and node.orth_ == 'or':
                    ll_res.append('OR')
                # need to remove separately -- can't modify list during above iteration
            res = [node for node in res if node.orth_ != 'or'] 
            res = sorted(res, key= lambda tok: tok.i)
    #                 print("finally:{}".format(temp))
            ll_res.append(res)
        return ll_res
    
    temp = traverse_tree(target, is_match, process_node) # has one too many layers
    temp = list(itertools.chain.from_iterable(temp))
        # TODO: buggy... May need to clean up process_node. 
        # Currently, list representation works though
#     return [sent[t[0].i:t[-1].i+1] if type(t) == list else t for t in temp] 
    
    def postprocess(node_list):
        temp = []
        for l in node_list:
            if l == 'OR':
                continue
            l_t = [t.orth_ for t in l]
            temp.append(' '.join(l_t))
        return [t.strip(' ,;') for t in temp]
    
    return postprocess(temp)


# In[ ]:


s = """1. Graduation from an accredited four-year college or university with a major in Computer Science, Philosophy of Science, Information Systems, or Geographical Information Systems; or"""
doc = en_nlp(s)

# get_major_requirement(doc)


# In[ ]:


job_fields = ['node', # a reference back to the Token in the Doc
             'name', # plaintext (may not match node.orth_, which is only one word)
             'experience_type', # type of experience
             'duration' # how long the experience must be
                        ]

Job = namedtuple('Job', job_fields)


# In[ ]:




def get_node_modifiers(node):
    """
    Pass in Spacy.tokens.
    Return whether job/experience must be full-time or part-time.
    Returns Span of Spacy.tokens relevant to type of experience, or None if no such tokens exist.
    
    (Note: Quality of output depends on accuracy of Spacy's dependency tree)
    """
    
    # we'll want the full subtrees of relevant nodes
    # will use traverse_tree again, but will filter nodes beforehand
    
    # want to return relevant nodes
    def has_wanted_dep(node):
        return (node.dep_ == 'compound' or node.dep_ == 'amod')
    
    def get_subtree_and_node(node):
        temp = list(node.subtree)
        temp.append(node)
        return temp
    
    # we assume that relevant information are left children of 'experience'
    # if dep_ is "compound" or "amod"
    res = []
    for tok in node.lefts:
        res.extend(
            traverse_tree
                   (tok, is_match = has_wanted_dep, process_node = get_subtree_and_node)
                  )
    if len(res) == 0:
        return None
    # flatten and sort
    res = sorted(list(set(itertools.chain.from_iterable(res))), key = lambda tok: tok.i)
    res = node.doc[res[0].i:res[-1].i+1]
    return res



def get_associated_time(node):
    """Returns time (number of years) as a float.
    
    If no explicit time is attached to node, return None.
    """
    #something like
    
    time_periods = {'year': 1, 
                    'month': 1/12, 
#                     'week': 1/(12*4),
#                     'day': 1/(12*4*5), 
#                     'hour': 1/(12*4*5*8)
                   }
    # find the node associated with the time (associated with job)
    
#     print_node(node)
    nodes = [tok for tok in node.ancestors if tok.lemma_ in time_periods.keys()]
    # maybe node itself is a time period?
    if node.orth_ in time_periods.keys():
        nodes.append(node)
    if not nodes: # maybe the dependency tree is broken (has happened)
        nodes = [find_lemma(time, node.doc) for time in time_periods.keys()]
        nodes = [n for n in nodes if n]
#     print(nodes)
#     nodes = [find_lemma(tok.lemma_, tok.doc) 
#              for tok in list(time_periods.keys())
#              if tok.lemma_ in list(job.ancestors)]
#     nodes = [n for n in nodes if n is not None]
    # how to choose "most important" time?
    # the one that's closest to the root
    # and because parent.ancestors is a subset of child.ancestors
    # we can simply choose the node with the smallest len(ancestors)
    try:
        time_node = sorted(nodes, key = lambda tok: len(list(tok.ancestors)))[0]
    except: # no explicit period of time stated
#         print("Problem in get_associated_time! Sentence: {}".format(node.doc))
        return None
    
    # now check whether there are numerical modifiers attached to this node
    prefactor = get_associated_number(time_node)
    # multiply this * time_periods[time.orth_] to get number of years
    return prefactor * time_periods[time_node.lemma_]



def get_associated_number(node, verbose = False):
    """Use Spacy dependency tree to match number with node. Return number as int or float.
    Return 0 if 'no' is attached to node.
    Return 1 if no explicit number is attached to node (presumably 'a','an','the' is attached).
    (*This is an assumption*)
    
    Else return number.
    """
    # first see if there's a determiner
    try:
        det_nodes = [n for n in node.children if n.dep_ == 'det']
        det_node = det_nodes[0]
        if det_node.lemma_ == 'no':
            return 0
    except: # no det match
        pass
    try:
        number_nodes = [n for n in node.children if n.dep_ == 'nummod']
        num_node = number_nodes[0]
        return text_to_num(num_node.lemma_)
    except:
        pass
    # other cases, just return 1
    return 1

# unit tests
s = """1. Six months of perfectly normal and not at all weird experience with the City of Los Angeles as a Management Assistant or Management Aide interpreting and applying State and Federal regulations, City ordinances, the City Administrative Code and/or the City Charter; or"""
doc = en_nlp(s)
test = find_lemma('experience', doc)
m = get_node_modifiers(test)
assert m.orth_ == 'perfectly normal and not at all weird'

s = """1. Six months of unpaid volunteer experience with the City of Los Angeles as a Management Assistant or Management Aide interpreting and applying State and Federal regulations, City ordinances, the City Administrative Code and/or the City Charter; or"""
doc = en_nlp(s)
test = find_lemma('Assistant', doc)
assert get_associated_time(test) == 0.5

s = """1. Two years of unpaid volunteer experience with the City of Los Angeles as a Management Assistant or Management Aide interpreting and applying State and Federal regulations, City ordinances, the City Administrative Code and/or the City Charter; or"""
doc = en_nlp(s)
test = find_lemma('year', doc)
assert get_associated_number(test) == 2

s = """1. Years of unpaid volunteer experience with the City of Los Angeles as a Management Assistant or Management Aide interpreting and applying State and Federal regulations, City ordinances, the City Administrative Code and/or the City Charter; or"""
doc = en_nlp(s)
test = find_lemma('year', doc)


# In[ ]:


# # currently breaks below...
# # probably because of sorting list in func,
# # but making key from unsorted list in first instance
# # so should have sorted list passed in/do so here first
# def memoize(func):
#     memoize_dict = {}
#     def mem_func(*args, **kwargs):
#         # build a key by converting everything into a bunch of tuples
#         key = []
#         for arg in args:
#             if type(arg) == list:
#                 key.append(tuple(arg))
#                 continue
#             key.append(arg)
#         for k in sorted(kwargs):
#             val = kwargs[k]
#             if type(val) == list:
#                 val = tuple(val)
#             key.append((k, val))
#         key = tuple(key)
            
#         if key in memoize_dict:
#             return memoize_dict[key]
#         else:
#             memoize_dict[key] = func(*args,**kwargs)
#     return mem_func

# @memoize
def keep_largest_discrete_intervals(vals, start_func, end_func, crit, acc = 0):
    """
    Given a list (vals) and a function which, 
    when acted on each entry in vals,
    provides the val's start (start_func) and end (end_func) interval portions,
    
    Return a list of non-overlapping values with maximal total value
    (as assessed by crit()).
    """
    # sounds like a greedy dynamic programming algorithm to me
    if len(vals) == 0:
        return vals, acc
    elif len(vals) == 1:
        return vals, acc + crit(vals[0])
    
    # sort by start time
    vals = sorted(vals, key = lambda val: start_func(val))
    val_0 = crit(vals[0])
    
    vals_skip, acc_skip = (keep_largest_discrete_intervals
                    (vals[1:], start_func, end_func, crit, acc)
                          )
    vals_keep, acc_keep = (
                 keep_largest_discrete_intervals
                     ([v for v in vals if start_func(v)>= end_func(vals[0])],
                         start_func, end_func, crit, acc + val_0)
                           )
    if acc_skip > acc_keep: # better to skip
        return vals_skip, acc_skip
    else: # tack on vals[0] along vals_keep before returning
        vals_keep.insert(0, vals[0])
        return vals_keep, acc_keep

ll = [(0,2),(8, 15), (0,3), (1,7), (2,9)]
ll= sorted(ll, key = lambda val: val[0])

# # test case
# keep_largest_discrete_intervals(vals = ll, 
#                                 start_func = lambda x: x[0], 
#                                 end_func = lambda x: x[1], 
#                                 crit = lambda x: x[1] - x[0], 
#                                 )


# In[ ]:


s = """2.  Graduation from an accredited four-year college or university with any major and six months of full-time paid experience (1,000 hours) providing recreation and leisure services for an agency or organization that conducts professional recreation programs; or
"""
doc = en_nlp(s)
# spacy.displacy.render(doc)


# In[ ]:


def get_previous_experience(sent):
    """
    sent = Spacy.Doc pertaining to job (if not relevant, returns None)
    
    Return a Job NamedTuple with the following fields:
    - node: a node comprising part of the job
    - name: a string of the full job name
    - experience_type: a Spacy.Span of type of experience
    - time: the length (in years) of required time
    """
    if type(sent) == str:
        sent = en_nlp(sent)
    
    within_la = True
    context_words = ['experience']
    if not is_proper_context(context_words, sent, need_all=True):
        return None
    
    # now just regex for the position
    s = sent.text
    
    # regexp search plus cleanup
    # if there was overlapping segments, keep only the largest one
    prev_jobs = [re.search(job, s) for job in job_titles if re.search(job, s)]
    prev_jobs, _ = keep_largest_discrete_intervals(prev_jobs, 
                                start_func = lambda x: x.start(), 
                                end_func = lambda x: x.end(), 
                                crit = lambda x: x.end() - x.start()
                                                  )
    # now get just the strings
    prev_jobs = [m.group() for m in prev_jobs]
    within_la = bool(prev_jobs) # if no job titles were found, maybe generic experience
    #     print("prev_jobs = {}".format(prev_jobs))
    if within_la:
        # match jobs to Spacy.tokens to look for time-period
        toks_dict = {}
        for job in prev_jobs:
            # TODO: really should retokenize sentence...
            lemma = job.split(' ')[-1] # just pick a word -- will be going up subtree
            cur_i = 0
            temp = find_lemma(lemma, sent)
            while temp in toks_dict.keys(): # collision -- keep looking
                cur_i = temp.i + 1 # start from the word following the previous collision
                temp = find_lemma(lemma, sent, start = cur_i)
            if temp:
                toks_dict[temp] = job
    
#     print(toks_dict)
    # build Job namedtuples
    job_tuples_list = []
    if within_la:
        for job_node, job_name in toks_dict.items():
            exp_type = get_node_modifiers(find_lemma('experience', sent))
            time = get_associated_time(job_node)
            job_tuples_list.append(Job(job_node,job_name,exp_type,time))
    else: # will have dummy job_names/job_nodes
        job_node = find_lemma('experience', sent)
        job_name = 'generic'
        exp_type = get_node_modifiers(job_node)
        time = get_associated_time(job_node)
        job_tuples_list.append(Job(job_node,job_name,exp_type,time))
            
    return job_tuples_list


# In[ ]:


s = """1. Two years of full-time paid professional experience as a Customer Service Representative in budgetary analysis and control, administrative analysis and research, systems and procedures analysis, or personnel administration; or"""
doc = en_nlp(s)
# get_previous_experience(doc)


# In[ ]:


def process_experience(text, verbose = False):
    """From text, get CSV-addable dict of experience information"""
    rel_text = preprocess_requirements(text)

    temp = [get_previous_experience(sent) for sent in rel_text.values() if sent]
    temp = [t for t in temp if t] # remove empties
    temp = itertools.chain.from_iterable(temp)

    # now iterate to make some key-value pairs
    res = {}
    j_fields = job_fields.copy()
    j_fields.remove('node') # not interested in token when making dict
    
    for i, job_tuple in enumerate(temp):
        for field in j_fields:
            label = 'prev_job_{num}_{field}'.format(num=i+1, field=field)
            try:
                val = job_tuple.__getattribute__(field).orth_ # convert Node to str
            except:
                val = job_tuple.__getattribute__(field)
            res[label] = val
            
    return res
        


# In[ ]:


# process_experience(text)


# In[ ]:


s = """1. Three years of full-time paid experience with the City of Los Angeles as a Management Assistant or Management Aide interpreting and applying State and Federal regulations, City ordinances, the City Administrative Code and/or the City Charter; or"""
doc = en_nlp(s)
job_node = get_previous_experience(doc)[0]


# In[ ]:


# text = file['text']
# pattern_dict['pattern_chunk_salary'].search(text).groupdict()


# In[ ]:


# TODO: does this work for all listings?
def perform_processing(fn):
    """
    For a job listing (given via filename),
    extract all relevant information and place into dictionary.
    """
    entry = load_file(fn, encoding = 'latin-1')
    text = entry['text']
    # now a slew of function calls and updates to `entry`
    funs = [fun for key,fun in globals().items() if key.startswith('process_')]
    
    # some preprocessing
    text = text.replace("-time paid experience", "-time experience")
    
    for fun in funs:
        entry.update(fun(text))
    
    # untransform whitespace
    entry['text'] = text.replace(n_char, '\n').replace(t_char, '\t') 
    
    return entry


# In[ ]:


job_fns = os.listdir(bulletins_dir)
df_list = []
for fn in tqdm(job_fns):
    try:
        df_list.append(perform_processing(fn))
    except:
        print("problem_fn:{}".format(fn))
#         raise
bulletins_df = pd.DataFrame.from_records(df_list)


# In[ ]:


# from collections import namedtuple

# NamedDict = namedtuple('NamedDict',['name','dict'])

# # close = ['education', 'college', 'school', 'university', 'apprenticeship', 'apprentice']
# # far = ['banana', 'information', 'systems', 'bye', 'dean']



# def get_sims(target, word_list, nlp = en_nlp, verbose = True):
#     """
#     Get similarities between target and word_list.
#     If either are str, they are converted to spacy.Doc's using nlp model.
#     Returns a NamedTuple with name=target, dict={words in wordlist: similarity_scores}
#     """
#     if type(word_list) == str:
#         word_list = nlp(word_list)
#     if type(target) == str:
#         target = nlp(target)
    
# #     sim_scores = [target.similarity(word) for word in word_list]
# #     if verbose:
# #         print("Similarity with {0}: {1}".format(target[0].orth_, sim_scores))
#     d = {word.orth_:target.similarity(word) for word in word_list}
#     if verbose:
#         print("Similarity with '{0}': {1}".format(target[0].orth_, d))
# #     d['_TARGET'] = target[0].orth_
    
#     nd = NamedDict(name = target[0].orth_, dict = d)

#     return nd

# # sims = get_sims("education", test_doc)
# # max(sims.dict.values())

# # print(show_)

# # sims_close = [en_nlp('college').similarity(en_nlp(word)) for word in close]
# # print("Close words: {}".format(sims_close))

# # sims_far = [en_nlp('college').similarity(en_nlp(word)) for word in far]
# # print("Far words: {}".format(sims_far))


# # Merging our datasets
# 
# Phew! Structuring data is exhausting work! (And certainly can be continued further, but we shall leave it here for now.)
# 
# Let's combine this newly minted DataFrame with our dataset regarding applicant data, create some relevant features, and see what we notice.
# 
# The goal will be to calculate a generalized linear model for a Bernoulli/Binomial family to predict what fraction of a job postings will have applicants from underrepresented ethnic backgrounds (URMs), based on:
# 
# - a job posting's length (`text_length`)
# - the minimum required previous job experience (`min_experience_duration`)
# - whether a college degree requirement is mentioned in the listing (`is_college_required`)
# 
# ... and a few other factors (such as average salary offered) in an attempt to mitigate [omitted variable bias](https://en.wikipedia.org/wiki/Omitted-variable_bias). We will then investigate the estimates for the parameter coefficients corresponding to the above features to see their effect on applicant composition. (That is, we'll see how much the above features affect who applies.)
# 
# **In simple English**: We're going to use a model that maps the variables above to a fraction in $(0,1)$. We'll use that model to see how much each variable contributes.

# In[ ]:


def calc_min_experience(df):
    return (df.select(lambda x: 'prev_job' in x and 'duration' in x, axis = 'columns')
     .fillna(100000) # temporary remove nans for min operation
     .aggregate(min, axis = 'columns')
     .replace(100000, np.nan) # reinstate nan's
     .rename('min_experience_duration')
    )


# In[ ]:


# def calc_salary_mean(df, edge, loc):
#     """
#     Calculate salary mean
#     edge in ['start', 'end'],
#     loc in ['la','dwp']
#     """
#     s = 'salary_{}_{}'.format(edge, loc)
#     return (
#         df.select(lambda x: s in x, axis = 'columns').
#      fillna(0).aggregate(sum, axis = 'columns') # sum all non-null values in a row
#     /
#     (~df.select(lambda x: s in x, axis = 'columns').isnull())
#      .apply(sum, axis = 'columns') # ...and divide by the number of non-null values in that row
#     ).rename('mean_{}'.format(s))

# edges = ['start','end']
# locs = ['la','dwp']

# temp_list = []
# # temp_df = pd.DataFrame()

# temp_list.append(calc_min_experience(bulletins_df))

# for edge in edges:
#     for loc in locs:
#         temp_list.append(calc_salary_mean(bulletins_df, edge, loc))
# #         label = 'salary_mean_{}_{}'.format(edge, loc)
# #         temp_df.insert(loc = len(temp_df.columns), column = label, value = calc_salary_mean(bulletins_df,edge,loc))
# # temp_list.append(temp_df)
# temp_df = pd.concat(temp_list, axis = 'columns') # pd.concat() more efficient than repeated updates


# In[ ]:


bulletins_df = (bulletins_df.assign(
    has_dwp_listing = lambda x: x['salary_start_dwp_listing_1'].notna())
)


# In[ ]:


def calc_salary_mean(df, edge):
    """
    Calculate salary mean
    edge in ['start', 'end']
    """
    s = 'salary_{}'.format(edge)
    return (
        df.select(lambda x: s in x and not 'mean' in x, axis = 'columns').
     fillna(0).aggregate(sum, axis = 'columns') # sum all non-null values in a row
    /
    (~df.select(lambda x: s in x and not 'mean' in x, axis = 'columns').isnull())
     .apply(sum, axis = 'columns') # ...and divide by the number of non-null values in that row
    ).rename('mean_{}'.format(s))

edges = ['start','end']

temp_list = []
temp_list.append(calc_min_experience(bulletins_df))

for edge in edges:
    temp_list.append(calc_salary_mean(bulletins_df, edge))

temp_df = pd.concat(temp_list, axis = 'columns') # pd.concat() more efficient than repeated updates


# In[ ]:


temp_df = (temp_df.assign(
    mean_salary_range = lambda x:x['mean_salary_end'] - x['mean_salary_start'])
)


# In[ ]:


bulletins_df = (bulletins_df.assign(text_length = lambda x: x['text'].str.len()))
bulletins_df = bulletins_df.merge(temp_df, left_index = True, right_index = True, how = 'outer')


# In[ ]:


# calculate the response variable 
urm_labels = ['Black','Hispanic','American Indian/ Alaskan Native', 'Filipino']
applicant_df['frac_URM'] = applicant_df.apply(lambda x: sum(x[urm_labels])/x['Apps Received'], axis = 'columns')


# In[ ]:


features = ['min_experience_duration', 'is_college_required', 'text_length', 
            'mean_salary_start','mean_salary_range']
responses = ['frac_URM']


# dataframe with only features we'll want to use in our model (and job title as index)
df_model = applicant_df.merge(bulletins_df, how = 'left', left_on = 'Job_Number', right_on = 'class_code')
# df_model = applicant_df.merge(bulletins_df, how = 'left', left_on = 'Job_Title', right_on = 'job_title')


# a slightly cleaner way to deal with duplicates (multiple job listings in bulletins)
# could be to figure out a way to "average" the relevant values
# we shall go with a simpler approach and just drop the duplicate listing
#
# Future directions could be to see how the listing changed and why,
# but unfortunately we do not have information about applicant information coregistered with date,
# so there wouldn't be much more analysis we could do (e.g. pairwise comparison)
df_model = df_model.drop_duplicates(subset = ['Job_Number'])
applicant_df = applicant_df.drop_duplicates(subset = ['Job_Number'])


print("After merging, we have {} entries, which is {:.2%} of the total number of provided job postings.".format(
    len(df_model), len(df_model)/len(bulletins_df)))


# A quick view at the distribution of job posting lengths:

# In[ ]:


(plt.ggplot(df_model, plt.aes(x = 'text_length'))
 + plt.geom_histogram(bins = 20, fill = 'gray', color = 'black') # doesn't perfectly follow ggplot syntax :'(
 + plt.ggtitle("Job Posting Length Distribution")
 + plt.labs(x = "Length of Posting", y = "Frequency")
)


# Well, some positions still contain missing values in the features we're interested in. Rather than drop these jobs from the analysis (every observation counts!), we will impute "reasonable" values for these values by averaging other jobs with similar topic and filled-in value. (This will be done via a combination of [Latent Dirichlet Allocation](http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) to separate the postings into related groups, and a k-nearest-neighbors imputation using `fancyimpute`.)

# In[ ]:


import fancyimpute


# In[ ]:


from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# In[ ]:


num_topics = 10
lda = LatentDirichletAllocation(n_components = num_topics, random_state = 42)

vectorizer = CountVectorizer()
text_vecs = vectorizer.fit_transform(bulletins_df['text']) # convert docs to document-word matrix
lda_model = lda.fit(text_vecs) # use document-word matrix to create n-category generative model
text_sims = lda_model.transform(text_vecs) # use generative model to predict probability of doc falling into topic

# take argmax for each list, assign that as the text's "category"
# group by category
df_sims = pd.DataFrame.from_records(text_sims)
# df_sims.apply([max, np.mean, np.var], axis = 'rows') # see how informative categories are vs others


# In[ ]:


df_impute = df_sims.merge(df_model[features + responses 
#                                    + ['Apps Received'] # newly added; impute for statsmodels
                                   # turns out, no missing values to impute!
                                  ], left_index = True, right_index = True)


# In[ ]:


# first, how much are we imputing?
for label in features:
    print("Fraction of missing values in {}: {:.3f}".format(
        label, sum(df_model[label].isna())/len(df_model))
         )


# In[ ]:


imputer = fancyimpute.IterativeImputer()


df_imputed = pd.DataFrame(imputer.fit_transform(df_impute), columns = df_impute.columns)
# the imputation considered 'is_college_required' a float rather than a factor
# replace via cutoff
df_imputed['is_college_required'] = df_imputed.apply(lambda x: x['is_college_required'] > 0.5, axis = 'columns')


# # Model building
# 
# Now we build our model! For greater explainability, we use `statsmodels` to generate our model.

# In[ ]:


# # sklearn attempt

# from sklearn.linear_model import LassoLarsCV, LarsCV, LassoCV, LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score

# models = ['LassoLarsCV', 'LarsCV', 'LassoCV', 'LinearRegression']
# models_dict = {}

# for model in models:
#     reg = globals()[model]() #default
#     reg = reg.fit(df_imputed[features], np.ravel(df_imputed[responses]))
#     models_dict[model] = reg

# for key, model in results_dict.items():
#     print("{} R2 value: {}".format(key, model.score(df_imputed[features], np.ravel(df_imputed[responses]))))
    

# model_str = 'LinearRegression'
# reg = models_dict[model_str]

# print("The {} model has a mean-squared-error (MSE) of {:.4f} and a coefficient-of-determination (R^2) of {:.2%}"
#      .format(model_str, 
#             mean_squared_error(np.ravel(df_imputed[responses]), reg.predict(df_imputed[features])),
#              r2_score(np.ravel(df_imputed[responses]), reg.predict(df_imputed[features]))
# ))
# dict(zip(features, reg.coef_))


# ## Statsmodels

# In[ ]:


import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[ ]:


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# df_tomerge = pd.DataFrame(scaler.fit_transform(df_imputed[features].values), columns = df_imputed[features].columns)

# df_stat = sm.tools.add_constant(df_tomerge)
# df_stat['frac_URM'] = df_imputed['frac_URM']


# In[ ]:


# # analyze only those observations with with no missing values
# df_stat = df_model[features + responses].dropna()

# analyze dataset using imputed values
df_stat = df_imputed.drop(labels =[0,1,2,3,4,5,6,7,8,9], axis = 1)


# In[ ]:


df_stat = sm.tools.add_constant(df_stat)
# df_stat = df_stat.drop('const', axis = 'columns')
df_stat['is_college_required'] = df_stat.apply(lambda x: int(x['is_college_required']), axis = 1)
df_stat['Apps_Received'] = applicant_df['Apps Received']


# In[ ]:


df_stat = df_stat.dropna()


# In[ ]:


# df_stat.corr()


# In[ ]:


# "successes" and "failures" of applicant diversity
# to be used to weight the different observations appropriately in GLM
binom_counts = np.array([applicant_df.apply(lambda x: sum(x[urm_labels]), axis = 'columns'),
 applicant_df.apply(lambda x: x['Apps Received'] - sum(x[urm_labels]), axis = 'columns')
]).T


# In[ ]:


lm_apps_received = smf.ols(formula = 
        "Apps_Received ~ text_length + is_college_required \
        + min_experience_duration + \
        mean_salary_start + mean_salary_range",
        data = df_stat
       ).fit()
print(lm_apps_received.summary())


# In[ ]:


df_stat.keys()


# In[ ]:


lm_frac_urm = (smf.GLM(
#                 endog = binom_counts
                endog = df_stat[responses]
              , exog = df_stat[features + ['const']]
              , family = sm.families.Binomial()
#               , var_weights = df_stat['Apps_Received']  # weight job by number of received apps 
#                                                         # (maybe questionable? Probably gives overconfident estimates?)
             )
      .fit())
print("Weighting each listing equally...")
print(lm_frac_urm.summary())


# In[ ]:


lm_frac_urm_weighted = (smf.GLM(
#                 endog = binom_counts
                endog = df_stat[responses]
              , exog = df_stat[features + ['const']]
              , family = sm.families.Binomial()
              , var_weights = df_stat['Apps_Received']  # weight job by number of received apps 
                                                        # (maybe questionable? Probably gives overconfident estimates?)
             )
      .fit())
print("Weighting each listing by number of applicants...")
print(lm_frac_urm_weighted.summary())


# # Discussion and Future Directions
# 
# Hmm, curious... 
# 
# For our diversity prediction model, if we weight each job listing evenly, the specified features do not appear strongly associated with the fraction of URM applicants, except for a *negative* association with `mean_salary_start` (the larger the mean salary, the smaller the fraction of URM applicants). Weighting each job listing by number of applicants gives similar-magnitude weights to the features but is probably overconfident in its estimates.
# 
# I am hesitant to make conclusions based on this analysis, based on a warning made when seeing how well we could predict the number of applications by the previously mentioned features. There is likely collinearity issues with the chosen features (hence a "high condition number") or some other issue in the specification, leading to questionable estimates. We've attempted different data curation (dropping observations with missing values), investigating correlations between variables, trying different regression variables (dropping different variables off the regression), but no luck.
# 
# Given more time, it would be worth investigating the cause of the high condition number before making any conclusions. Despite this issue, it would also be worth seeing how different clusters of job postings (clustered via, for example, topic) differ from one another in required previous experience, education requirements, salary, etc.
# 
# I look forward to seeing what other Kagglers have been able to pull off with this data! I hope the City of Los Angeles finds use in these analyses.
