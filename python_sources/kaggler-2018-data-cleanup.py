#!/usr/bin/env python
# coding: utf-8

# * [Introduction](#Introduction)
# * [The data](#The-data)
# * [Objectives](#Objectives)
# * [Imports and file loading](#Imports-and-file-loading)
# * [Exploration of data contents and format](#Exploration-of-data-contents-and-format)
# * [First observations](#First-observations)
# * [Cleaning checklist](#Cleaning-checklist)
# * [First fixes](#First-fixes)
# * [New findings, new problems](#New-findings,-new-problems)
# * [More question types](#More-question-types)
# * [A function that determines the question type](#A-function-that-determines-the-question-type)
# * [A function that checks the data](#A-function-that-checks-the-data)
# * [Data gone missing in Q34](#Data-gone-missing-in-Q34)
# * [Trolling attempt in Q35?](#Trolling-attempt-in-Q35?)
# * [Text data hiding out in Q47](#Text-data-hiding-out-in-Q47)
# * [And it's a wrap!](#And-it's-a-wrap!)
# * [Lessons learned](#Lessons-learned)

# # Introduction
# Welcome to my first public kernel where I'll do some thorough **data cleaning** of the Kaggler 2018 survey dataset. Why would I focus on this aspect? My initial idea for a project was the role of gender for data scientists. I wanted to automatically pinpoint interesting, significant differences between men and women and let that initial wide scan be a guide for subsequent deeper analysis.  I realized however that implementing this without too much hardcoding was a bit of a project in itself, so this is it!  In addition, a main goal for me was to get familiar with the Python library pandas and data cleaning is the perfect task for that.

# # The data<a name="The-data"></a>
# In October 2018, thousands of Kagglers participated in a survey, answering questions about themselves in their role as data scientists. See the full description of the survey [here](https://www.kaggle.com/kaggle/kaggle-survey-2018/home).  The data consists of the following files:
# 
# * `multipleChoiceResponses.csv`: all survey response data except free form text responses
# * `freeFormResponses.csv`: the free form text responses, order reshuffled.
# *  `SurveySchema.csv`: conditions for who gets to answer which questions
# 
# ** I will focus on the main dataset** `multipleChoiceResponses.csv` and will leave `freeFormResponses.csv` for the NLP lovers out there. The contents of  `SurveySchema.csv` was not obvious to me initially, as is clear from [these questions](https://www.kaggle.com/kaggle/kaggle-survey-2018/discussion/71315) I had. In any case, it would mainly be useful for figuring out why some people haven't responded to certain questions. As is clear from the discussion thread linked above, this can happen for either of the following reasons: the respondent
# 1. isn't eligible to answer (e.g. a question about the current job for someone who is unemployed)
# 2. aborts the survey before answering all questions
# 3. skips a question (made possible by mistake for certain questions).
# 
# 
# # Objectives 
# My goal is to automatically generate basic statistics and plots from the dataset to get an overview. To achieve this I need to:
# * understand what different types of question there are
# * understand how data is encoded for each question
# * decide what basic plots and statistics to generate for each question type
# * write code for extracting data for all questions and presenting it compactly as plots and statistics
# 
# For the last step to work smoothly, **the data needs to be clean and tidy**.

# # Imports and file loading<a name="Imports-and-file-loading"></a>
# We start off with some basic imports and file loading.

# In[ ]:


import numpy as np
import pandas as pd
import os, re
from collections import OrderedDict

base_dir = ("../input")
df = pd.read_csv(os.path.join(base_dir,"multipleChoiceResponses.csv"))
df_text = pd.read_csv(os.path.join(base_dir,"freeFormResponses.csv"))
df_survey = pd.read_csv(os.path.join(base_dir, "SurveySchema.csv"))


# # Exploration of data contents and format<a name="Exploration-of-data-contents-and-format"></a>
# The main dataset is stored in `df`.  A first visual inspection gives an idea of what's in it:

# In[ ]:


display(df.head())


# Looking at the columns names and frequencies provides a more quantitative overview.

# In[ ]:


n=20
print('First {} columns: {}'.format(n, ', '.join(df.columns.values[:n])))
def count_column_name_formats(df):
    print('Unique column name format:\n{}'.format(df.columns.str.replace('[0-9]+','?').value_counts()))
count_column_name_formats(df)


# # First observations<a name="First-observations"></a>
# With this type of simple analysis and visual inspection I was able to postulate the following about the contents and format of the dataset and compile it into the following pretty boring but useful list. Note my use of `?` as a placeholder for an integer.
# * The first row  stores strings describing the column data, typically a question string and the info about the answer options.
# * The first column stores the time the respondent spent on taking the survey.
# * Questions are labelled by `Q?`.
# * Two main types of questions:
#     * "select one", e.g. `Q1`:
#     > "What is your gender?"
#     * "select any" ("select all that apply"), e.g. `Q11`:
#     > Select any activities that make up an important part of your role at work: (Select all that apply)*
# * Columns named `Q?` belong to "select one"-type questions.
#     * Possible values are `NaN` or either of the answer strings for the question (categorical)
#     * A`NaN`value means the question was not answered.
# * Columns named `Q?_Part_?` belong "select any"-type questions.
#     * Each column has info about if a certain answer option was selected.
#     * The answer that the column corresponds to can be found at the end of the string on the first row. E.g. in `Q11_Part_1`:
#     > *Select any activities that make up an important part of your role at work: (Select all that apply) - Selected Choice - Analyze and understand data to influence product or business decisions*
#     * There are two possible values: `NaN` or the answer string of the column (binary categorical)
#     * A value of `NaN` means either that the option was not selected or the question was not answered. 
# * Columns with names suffixed by `_TEXT`  has data on whether additional text was provided or not.
#     * `OTHER_TEXT` generally refers to text for the selection "Other" .
#     *  `Part_?_TEXT`  refers to text for one out of multiple selections having text data.
#     * A value of `-1` means no text data was provided.
#     * A non-negative integer value  means that text data was provided (stored in the free text file).
# * There are two column types which only occur once, `Q?_MULTIPLE_CHOICE` and `Q?_OTHER`. Is there something special about them?
# 
# Phew, that's quite a list!

# # Cleaning checklist<a name="Cleaning-checklist"></a>
# Typical things to look out for:
# * typos or inconsistent naming in strings
# * inconsistent types (e.g. string vs numerical values).
# * broken constraints on numerical values, e.g. a negative time interval.
# 
# In my experience, adding checks (e.g. `assert`) is a great way to detect problems. This might also catch errors in your own initial assumptions or understanding about the data.
# 
# I find it can also be convenient to:
# * change naming conventions
# * sort out data from metadata
# * split data into multiple datasets if they'll be analyzed independently, or only a subset is of interest.

# # First fixes<a name="First-fixes"></a>
# Based on these first observations, there is already some stuff to do:
# * The single column with name format `Q?_OTHER` is a typo and should be `Q?_OTHER_TEXT` to be consistent.
# * Similarly for `Q?_MULTIPLE_CHOICE`; it belongs to a "select one" question and should be renamed to `Q?`.
# * Split `_TEXT` columns and store separately.
# * Split out the first row with info about the column (metadata) and store separately.
# * Split the survey times from the actual questions and store separately. Make sure times are numeric and `>=0`.
# * The`_TEXT`columns mix string and numeric types. Change to numeric and set `dtype`.  Note that even though this data won't be used  here, it's worth ensuring that they contain what we think they do.
# * Make sure the only negative value of `_TEXT` columns is  `-1` (if there are others I wouldn't know how to interpret them).
# * Set `dtype` of `Q?` and `Q?_Part_?` columns to `category`  to save some space.
# * Make sure `Q1` has `> 1` categories and `Q?_Part_?` has exactly 1 category (not counting `NaN`-values).

# In[ ]:


# Back up in case we want to go back to the raw data.
df_raw = df.copy()


# Rename columns:

# In[ ]:


# Rename columns
# Q?_OTHER --> Q?_OTHER_TEXT
df = df.rename(columns = lambda s: re.sub('^(Q[1-9]+[0-9]*_OTHER)$',r'\1_TEXT',s))

# Q?_MULTIPLE_CHOICE --> Q?
df = df.rename(columns = lambda s: re.sub('^(Q[1-9]+[0-9]*)_MULTIPLE_CHOICE$',r'\1',s))


# Extract headers from data and split datasets:

# In[ ]:


# Split out metadata and times
column_headers, df = df.iloc[0], df.iloc[1:]
times, df = df.iloc[:,0], df.iloc[:,1:]

text_cols = df.filter(regex='_TEXT$').columns
# The text columns contain information about if there is text in freeFromResponses.csv or not.
# but since the data has been randomized I don't see how it would every be useful.
df_text_before_randomizing =  df.filter(regex='_TEXT$')
df.drop(columns=df_text_before_randomizing.columns, inplace = True)


# Change types:

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Times are numeric\ntimes = pd.to_numeric(times)\n\n# Text columns are integer (essentially boolean)\ndf_text_before_randomizing = df_text_before_randomizing.apply(pd.to_numeric)\n\n# Convert remaining colums to categorical\ndf = df.astype('category')")


# I checked the time here because before I split out the text columns from `df` this `dtype` conversions to `category` took > 10 s. I haven't investigated why it's so much faster doing `astype` without filtering on columns, maybe someone knows?
# 
# Finally, make value checks based on the column and question type:

# In[ ]:


assert (times >=0).all(), "Times should be non-negative."
assert ((df_text_before_randomizing == -1) | (df_text_before_randomizing >= 0)).all().all(),"Values in text columns should equal -1 or be non-negative."

# Q? columns are for "select one"-type questions
select_one_cols = df.filter(regex='^Q[1-9]+[0-9]*$').columns

# Q?_Part_? columns are for select-any questions 
select_any_cols =  df.filter(regex='^Q[1-9]+[0-9]*_Part_[1-9]+[0-9]*$').columns

assert (df[select_one_cols].apply(lambda c: len(c.cat.categories), axis=0) > 1).all(),"select-one column `Q?_Part_?` should have more than 1 (non-NaN) category."
assert (df[select_any_cols].apply(lambda c: len(c.cat.categories), axis=0) == 1).all(),"select-any column `Q?_Part_?` should have exactly 1 (non-NaN) category."


#  The assertion for the number of categories being equal to 1 in `Q?_Part_?` failed! This means something is wrong in the data or with my assumptions. A closer look at the reason:

# In[ ]:


ncat = df[select_any_cols].apply(lambda c: len(c.cat.categories), axis=0)
problem_ncat = ncat[ncat != 1]
print("Number of categories/column:\n{}".format(problem_ncat.sort_values(ascending=False)))

# "Hack" to compactly extract the question label from the columns without changing their order,
# there is probably a better way... suggestions?
problem_questions =  list(OrderedDict({idx[:idx.find('_')]: '' for idx in problem_ncat.index}))

print("Questions with != 1 category: {}".format(', '.join(problem_questions)))


# What are these questions?

# In[ ]:


first_cols = OrderedDict()
for col in problem_ncat.index:
    q = re.sub('^(Q[1-9]+[0-9]*).*$',r'\1', col)
    if q in first_cols:
        continue
    else:
        first_cols[q] = col

for question, col in first_cols.items():
    print(' - '.join([question, column_headers[col]]))


# # New findings, new problems<a name="New-findings,-new-problems"></a>
# From the output above we learn:
# * `Q34`, `Q35` are of a different type than I knew about, where percentages are to be specified by the respondent. For this, columns should have `dtype` `float64` rather than `category`
# * `Q38` had zero categories in two columns, corresponding to "Towards Data Science Blog" and "Analytics Vidhya", meaning not a single person selected them. This is surprising to me at least, since they pop up in my scope every now and then. And is it just a coincidence that it's the two last columns that are zero...? In any case, this shows that my check was too strict, assuming there is "enough" data everywhere.
# *  `Q39` and `Q41` also correspond to a new question type: multiple similar "select one" questions are bunched together under the same question label. Each column corresponds to a different "select one" question.
# * `Q47` is more fishy!  One column, `Q47_Part_16` out of at least 16 has >100 categories while the rest has 1. What's going on?
# 
# I'll now go through these issues in more detail and and fix them one-by-one.

# # More question types<a name="More-question-types"></a>
# For `Q34`, `Q35`, `Q39`, `Q41` the problem were my initial assumptions. We see that the data has two more question types with a different format:
# * "percentages", `Q34` and `Q35`.
#     * the respondent is asked to fill in the relative contribution of different components to some task.
#     * each `Q?_Part_?` contains a percentage value
#     * Values are or a number or `NaN` if the question was not answered.
#     * numerical constraints: 1) each value >=0 and <= 100;  2) sum of values = 100.
#     * `dtype`: `float64`
# * "multiple select one", `Q39` and `Q41`
#     * multiple similar select-one questions are listed under the same question label.
#     * each `Q?_Part_?` is of the same format as  a select-one `Q?` column.
#     * possible answers identical for different questions so they can be compared (e.g. `Q39`: `'Much better', 'Much worse', 'Neither better nor worse','No opinion; I do not know', 'Slightly better', or 'Slightly worse'`).
#     * `dtype`: `category`

# # A function that determines the question type<a name="A-function-that-determines-the-question-type"></a>
# Remembering my goal of automatically (without hardcoding) determining the question type from the data, one realizes that the column names are not enough since different question types all have `Q?_Part_?` columns. But, as we saw with `Q38` looking at the data itself is  also a fragile approach since it assumes a certain amount of data for it to be efficient. Therefore, I'll use the column formats together with the question string itself to determine the question type, and then use the data for consistency checks.

# In[ ]:


# Recognized question types
QTYPES = ['select one', 'select any', 'multiple select one', 'percentages']

def determine_question_type(q, headers):
    assert 'QTYPES' in globals(), "Question types QTYPES needs to be defined"

    # Only checks columns 'Q?' or 'Q?_Part_?'
    q_headers = headers.filter(regex='{}_Part_[1-9]+[0-9]*$|^{}$'.format(q,q))

    ncols = len(q_headers)
    if ncols == 0:
        raise ValueError('no columns found for {}'.format(q))
        
    # The conditions for determining the question type are not fully determinant.
    # Therefore the types are listed in order of the strictness of the condition.
    header_0 = q_headers.iloc[0].lower()
    
    has_one_col = (ncols == 1)
    consistent_qtype = OrderedDict({'select one': has_one_col, 
                                    'select any': not has_one_col and 'select all that apply' in header_0,
                                    'percentages': not has_one_col and '100%' in header_0,
                                    'multiple select one': not has_one_col and (ncols > 1)})

    assert set(consistent_qtype).issubset(set(QTYPES)), 'question type should be globally known'

    # Return the first consistent type
    for qtype, is_consistent in consistent_qtype.items():
        if is_consistent:
            return qtype;

    raise ValueError('Could not determine question type')


# We can now map each question to its type.

# In[ ]:


# Question labels Q?. Not so elegant, but extracts the unique question label and maintaining the order.
qs = list(OrderedDict({re.sub('^(Q[1-9]+[0-9]*).*$',r'\1', col): None for col in df.columns}))

# Keep two convenience dicts
# One mapping the Q? to question type
question_type = dict()
for q in qs:
    question_type[q] = determine_question_type(q, column_headers)
print(' | '.join( ': '.join ([str(k),str(v)]) for k, v in question_type.items()))

# and one mapping to the question string.
questions = dict()
for q in qs:
    # Extract the question string from the first header starting with q
    for col, header in zip(column_headers.index, column_headers):
        if col.startswith(q):
            questions[q] = header.partition(' - ')[0]
            break;
    assert q in questions, "question for {} not found".format(q)


# and change the value of `dtype` for all "percentages" questions to numeric.

# In[ ]:


# Helper function. Return columns for question q in dataFrame df.
def cols_for_question(q, df):
    return df.filter(regex='^{}$|^{}_.*$'.format(q, q)).columns

# Set dtype to numeric for all "percentages" questions.
for q, qtype in question_type.items():
    if (qtype == 'percentages'):
        for col in cols_for_question(q, df):
            df[col] = pd.to_numeric(df[col])                        


# # A function that checks the data<a name="A-function-that-checks-the-data"></a>
# After determining the question type, the data can be checked to make sure it's consistent with assumptions and constraints.

# In[ ]:


def check_data(q, df, headers):
    # Note format for output
    note = '{q} at {loc} type \"{qtype}\" {msg}'
        
    qtype = determine_question_type(q, headers)

    # Check only on non-text columns
    cols = df.filter(regex='^{}$|^{}_Part_[1-9]+[0-9]*$'.format(q, q)).columns
    ncats =  df[cols].apply(lambda col: len(col.astype('category').cat.categories), axis=0)
    
    if qtype == 'select one':
        # <= 1 categories in this case is suspicious, but could be due to
        # skewed or not enough data so just give a note.
        for col, ncat in zip(cols, ncats):
            if ncat <= 1:
                loc = "column {}".format(col)
                msg = "has {ncat} categories".format(ncat=ncat)
                print(note.format(q=q, qtype=qtype, loc=loc, msg=msg))
    elif qtype == 'select any':
        # Typically 1 category, but could be 0 if there is limited data
        for col, ncat in zip(cols, ncats):
            loc = "column {}".format(col)
            msg = "has {ncat} categories".format(ncat=ncat)
            if ncat < 1:
                print(note.format(q=q, qtype=qtype, loc=loc, msg=msg))
            if ncat > 1:
                raise ValueError(note.format(q=q, qtype=qtype, loc=loc, msg=msg))
    elif qtype == 'multiple select one':
        # Theoretically, each column should have the same categories,
        # but this may not be true for the actual data, so just give a note.
        df_cats = df[cols].apply(lambda col: col.astype('category').cat.categories, axis=0)
        
        # Check if categories in all columns are equal by comparing to the union of all categories.
        cats_union = set(df_cats.values.flatten())
        for col in cols:
            cats = set(df_cats[col])
            if cats != cats_union:
                loc = "column {}".format(col)
                msg = "has categories:\n{cats}\n but union with other columns is\n{union}"                .format(cats=cats, union=cats_union)
                print(note.format(q=q, qtype=qtype, loc=loc, msg=msg))
    elif qtype == 'percentages':
        # Values should be numeric
        try:
            df_num = df[cols].apply(pd.to_numeric)
        except ValueError as err:
            loc = "column {}".format(col)
            msg = "failed to convert to numerical type"
            print(note.format(q=q, qtype=qtype, loc=loc, msg=msg))
            raise
        
        # Percentage values should be >=0 and <=100
        in_range = df_num.isna() | ((df_num <= 100) & (df_num >= 0))
        if not in_range.all().all():
            # find the data (rows) which are out of range
            col_not_in_range = ~(in_range.all(axis=1))
            row_not_in_range = ~(in_range.all(axis=0))
        
            rows = col_not_in_range.index[col_not_in_range].values
            cols = row_not_in_range.index[row_not_in_range].values
            loc = "columns {cols}, rows {rows}".format(cols=cols, rows=rows)
            msg = "has values out of allowed range [0,100]"
            raise ValueError(note.format(q=q, qtype=qtype, loc=loc, msg=msg))
        # Values should add to 100.
        sums = df_num.dropna().sum(axis=1)
        if not (sums == 100).all():
            bad_sums = sums[sums != 100]
            bad_indices = bad_sums.index

            note_alt_2 = 'Note: at {loc} type \"{qtype}\" {msg}'
            loc = "rows {}".format(bad_indices.values)
            msg = "has incorrect sums {} (not equal to 100)".format(bad_sums.values)
            raise ValueError(note.format(q=q, qtype=qtype, loc=loc, msg=msg))
    else:
        raise ValueError("Unknown question type {}".format(qtype))


# Now check all questions.

# In[ ]:


# Print notes and errors found by check_data()
for q in qs:
    try:
        check_data(q, df, column_headers)
    except ValueError as err:
        print(' '.join([str(err),'(error)']))


# Some of this we already knew, we still need to figure out `Q47`,  but the value errors of `Q34` and `Q35` are new. Good to find out about! 

# # Data gone missing in Q34<a name="Data-gone-missing-in-Q34"></a>
# We can look at the suggested fishy data point with index 3 to verify that the percentages (in `Part_1` to `Part_6`) do not add to 100.

# In[ ]:


# Print basic info about question Q34
q = 'Q34'
df_q = df.filter(cols_for_question(q=q, df=df))
i_suspect = 3

# Header for question q
headers_q = column_headers[df_q.columns]

# Print question
print(': '.join([q, questions[q]]))

# Print column info.
# Each column maps to a component associated with a percentage value.
print(' | '.join(['column', 'component', 'value (%)']))
perc_sum = 0
for col, header, val in zip(headers_q.index, headers_q, df_q.loc[i_suspect]):
    component = header.split(' - ')[-1]
    print(' | '.join([col, component, str(val)]))
    perc_sum += val
print("Total sum = {}".format(str(perc_sum)))


# Ok... What could be missing? Our only chance is that it was by mistake stored in a `_TEXT` columns, which we prevously split off into a separate table.  Indeed, we find a missing column for the option "Other" there. However, we see that these values are not the missing percentages (since they are not in [0,100]).

# In[ ]:


cols = cols_for_question(q=q, df=df_text_before_randomizing)
print(' | '.join(['column', 'component']))
for col in cols:
    component  =  column_headers[col].split( ' - ')[-1]
    print(' | '.join([col, component]))

display(df_text_before_randomizing[cols].describe())


# We can now understand the origin of the problem. The percentage values of the component "Other" should have been put in a column `Q34_Part_7` which was however incorrectly called`Q34_OTHER_TEXT`. Because the naming implies text data, its contents have ended up being split out, reshuffled and put into `freeFormResponses.csv`. Indeed, in `df_text['Q34_OTHER_TEXT']` we find such fitting numerical data, not text!

# In[ ]:


print("Some statistics:")
cols = cols_for_question(df=df_text, q=q)
print(df_text[cols].iloc[1:].apply(pd.to_numeric).describe())


# The reshuffling however means that we can't make use of that data anymore. Nonetheless, because there is only one column missing, it's not neeeded since we know what value makes the sum correct. So the tasks now are:
# * add column `Q34_Part_7` to `df`
# * fill it with missing percentage  by enforcing the column sum to equal 100.
# * delete columns `Q34_OTHER_TEXT`

# In[ ]:


# Add the missing column
old_cols = cols_for_question(df=df, q=q)
new_col = '{q}_Part_7'.format(q=q)

# Add what's missing from 100%
df[new_col] = 100 - df[old_cols].sum(1)

# Rows with NaN summed to zero above, but should be NaN also in the new column.
has_nan_on_row = df.loc[:,old_cols].isna().any(1)
df.loc[has_nan_on_row, new_col] = np.nan


# #  Trolling attempt in Q35?<a name="Trolling-attempt-in-Q35"></a>
# Now to `Q35`. Respondent no. 2823 managed to sneak in some illegitimate percentage values; apparently it was possible!

# In[ ]:


q, i ='Q35', 2823
display(df.loc[i, cols_for_question(df=df, q=q)])


# I'm just going to remove you, sorry!

# In[ ]:


df.drop(i, inplace = True) # too easy!


# # Text data hiding out in Q47<a name="Text-data-hiding-out-in-Q47?"></a>
# Finally, we get to sorting out the fishy column of`Q47`. This is a "select any"-type question:

# In[ ]:


q = 'Q47'
print(questions[q])


# and we expect each column to be binary categorical variable (`NaN` or the answer string), but the last "part" had 127 categories. Let's look at the values:

# In[ ]:


cols = cols_for_question(df=df, q=q)
display(df[cols[-1]].dropna().head(10))


# Oops! (And hello Siraj!) This is text data that should've been reshuffled and put into `freeFormResponses.csv`, but somehow this column got mislabeled; it should have been `Q47_OTHER_TEXT`.  Because it now didn't get the `_TEXT` suffix, it didn't either get treated as text in the post-processing.  To fix this I'll rename the column and move the text data to the text dataset (and change to correct `dtype`).

# In[ ]:


# Fix the typo by renaming the column
old_name, new_name = 'Q47_Part_16', 'Q47_OTHER_TEXT'
df.rename(columns={old_name: new_name}, inplace=True)

# Copy data to text dataset
df_text[new_name] = df[new_name].astype('object')

# Just for consistency, keep a column in the text dataFrame that was split off,
# changing NaN to -1 and strings to 0.
df_text_before_randomizing[new_name] = df[new_name].replace(to_replace = r'.*', regex=True, value='0').fillna(-1).astype('int64')

# Finally remove it from main data
df = df.drop(columns=[new_name])


# # And it's a wrap!<a name="And-it's-a-wrap!"></a>
# A final check on the values shows that the data is now clean.

# In[ ]:


# Print notes and errors found by check_data()
for q in qs:
    try:
        check_data(q, df, column_headers)
    except ValueError as err:
        print(' '.join([str(err),'(error)']))


# # Lessons learned<a name="Lessons-learned"></a>
# When you're handed a dataset without being able talk to directly the people who compiled the dataset, it can really be a detective work to even understand what the data *is*. In the case of the Kaggle survey, there were several different question types and they were encoded in different ways, sometimes with typos in either the data or the format. 
# 
# The simple workflow I used here to explore and clean the data has been:
# * First, visually inspect the data
# * Describe the data format  (e.g. as a list of the different column types and constraints on their content)
# * Decide how you want to analyze the data and extract the relevant data.
# * Perform automated tests based on the description of the data you made. If the tests fail either the test is wrong or the data is wrong.
# * Iterate!
# 
# Cleaning is certainly not the most fun part of data analysis, but I think it's good practice carry out this step systematically at the get-go with so you don't get surprises and mess to sort out later on!
