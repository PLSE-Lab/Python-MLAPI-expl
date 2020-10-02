#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.preprocessing import minmax_scale, scale
import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


survey_schema = pd.read_csv('../input/SurveySchema.csv')
free_form = pd.read_csv('../input/freeFormResponses.csv')
multiple_choice = pd.read_csv('../input/multipleChoiceResponses.csv')


# In[ ]:


all_na_values = (multiple_choice.loc[1:, :].isna().sum() == (len(multiple_choice) - 1))
print('Dropped columns ', multiple_choice.columns[all_na_values].tolist(),
      ' from multipleChoiceResponses.csv because are all na')
multiple_choice = multiple_choice.loc[:, ~all_na_values]


# In[ ]:


# Many support functions
def cols_part_questionary(q):
    return multiple_choice.columns[multiple_choice.columns.str.startswith(q) & multiple_choice.columns.str.contains('_Part_')]

def get_cols_values(values):
    return values.iloc[1:, :].apply(lambda col: col[col.first_valid_index()])

def plot_correlation_question(*qs):
    q_multiple_answers = []
    for q in qs:
        q_multiple_answers += cols_part_questionary(q).tolist()
    q_choices = multiple_choice.loc[1:, q_multiple_answers]
    map_index = q_choices.apply(lambda col: col[col.first_valid_index()], axis=0).reset_index(drop=True)
    if not map_index.str.match('[0-9]+').any():
        q_choices = ~q_choices.isna()
#       plt.subplots(figsize=(15, 11))
#         plt.title(survey_schema.loc[0, q])
        sns.heatmap(cramerv(q_choices, q_choices.columns.tolist()),
                    xticklabels=map_index.values,
                    yticklabels=map_index.values)
        plt.show()
    return map_index, q_choices

def plot_countbar(responses, values):
    counts = values.sum()
    counts.index = responses.values
    plt.subplots(figsize=(10, 7))
    ax = sns.barplot(counts.index, counts.values)
    ax.set_xticklabels(counts.index.values, rotation=90)
    return ax

def plot_cross_correlation(q1, q2):
    q1 = cols_part_questionary(q1).tolist()
    q2 = cols_part_questionary(q2).tolist()
    q_choices = multiple_choice.loc[1:, q1 + q2]
    map_index = q_choices.apply(lambda col: col[col.first_valid_index()], axis=0).reset_index(drop=True)
    q_choices = ~q_choices.isna()
    corr_matrix = cramerv(q_choices, q1 + q2)
    ax = sns.heatmap(corr_matrix[:len(q1), len(q1):], yticklabels=map_index[:len(q1)], xticklabels=map_index[len(q1):])
    return ax

from scipy.stats import chi2_contingency
def _cramerv(col1: np.array, col2: np.array):
    confusion_matrix = pd.crosstab(col1, col2)
    chi2, p_value = chi2_contingency(confusion_matrix)[0:2]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1))) if not phi2corr == 0 else 0


def cramerv(dataset: pd.DataFrame, cols) -> np.matrix:
    """
    Apply the cramerv test on each pair of columns. The cramerv test is used to
    measure in [0,1] the correlation between categorical variables
    :param dataset: Dataframe where the correlation is measured
    :param cols: the columns of dataset where measure the correlation
    :return: the correlation matrix of cramerv on the specified columns of dataset
    """
    matr = np.matrix([[0] * len(cols)] * len(cols), dtype='float')
    for i in range(len(cols)):
        for j in range(i, len(cols)):
            if i == j:
                matr[i, j] = 1
            else:
                corr = _cramerv(dataset[cols[i]], dataset[cols[j]])
                matr[i, j] = corr
                matr[j, i] = corr
    return matr

def normalized_crosstab(q1, q2):
    """Get the crosstab normalized by q1"""
    crosstab = pd.crosstab(multiple_choice.loc[1:, q1], multiple_choice.loc[1:, q2])
    norm_crosstab = crosstab.divide(crosstab.sum(axis=1), axis='rows')
    return norm_crosstab.T


# Hi! My name is Michele De Vita, i'm Italian and i have a bachelor degree in computer science and now i'm doing a master degree in data science (under the computer science department) at the University of Florence. With this kernel i want to give my impressions about this survey by a person with a computer science background! I hope you enjoy!

# # Data science is a young field
# From the next three plots we can see that the age of kagglers is mostly between 18-30 and they have mostly low experience in the current role (i'm also in) while the companies are still exploring machine learning or not using at all. <br>
# I can also confirm this from my personal experience: in Italy, in my university, there are few people that are doing the master degree in Data Science (in my year we are 4) and there aren't so much professors that are doing research or teaching data science/machine learning. <br>
# There are few people also because there is a high entry barrier: in order to understand and practice it require at least a basic knowledge of statistic, math and computer science.<br>
# A young field has advantages and disadvantages:<br>
# if you are in this field there you have surely few competitors but also less collaborators: this means less progression on research, less people to chat and from a student perspective learn by yourself many things that aren't explained in any course <br>
# Luckly this trend is changing, there are many incentives to invest into data science field nowdays and i think in 10 years (but maybe also less) things changes dramatically

# In[ ]:


multiple_choice.Q2[1:].value_counts().plot.barh(title='Age of kagglers survey respondants')
plt.show()


# In[ ]:


multiple_choice.Q8[1:].value_counts().plot.barh(title=multiple_choice.Q8[0])
plt.show()


# In[ ]:


multiple_choice.Q10[1:].value_counts().plot.barh(title=multiple_choice.Q10[0])
plt.show()


# # Coding stuff during ml projects
# 

# Since i have greatly appretiate the book Clean Code (read it if you hadn't!) i'm very glad that the most importants things cared by data scientist are make the code human-readable and well documented!

# In[ ]:


q49_cols = multiple_choice.loc[:, cols_part_questionary('Q49')]
q49_values = get_cols_values(q49_cols)
counts_q49 = (~q49_cols.isna()).sum()
counts_q49.index = q49_values.values
counts_q49.plot.barh(title=survey_schema.Q49[0])
plt.show()


# ### What is the background of the kagglers
# The latter result is not surprising if we see the background of the kaggle survey partecipants:

# In[ ]:


multiple_choice.loc[1:, 'Q5'].value_counts().plot.barh(title=multiple_choice.Q5[0])
plt.show()


# because the major part of the people are from Computer Science (me too!). <br> It is also surprisingly low the amount of statistician that use Kaggle taking into account
#  that data science is a mix of computer science and statistic. 
# 

# ## Programming languages and data scientist
# ### Most used programming languages
# We know without plots that Python and R are the most widely used programming languages in the field of data scientist. <br>
# In this survey we observe a different story but this is caused by the sample we are observing:  the major part are computer scientist so they mostly like use Python instead of R

# In[ ]:


q16_cols = multiple_choice.loc[1:, cols_part_questionary('Q16')]
q16_values = get_cols_values(q16_cols)
counts_q16 = (~q16_cols.isna()).sum()
counts_q16.index = q16_values.values
counts_q16.plot.barh(title=survey_schema.Q16[0])
plt.show()


# In[ ]:


multiple_choice.Q18[1:].value_counts().plot.barh(title=multiple_choice.Q18[0])
plt.show()


# We can observe that Python is the most used language but it also the most recomended -  in the latter case is also vastly suggested in proportion to the others 

# ### Use of programming language
# Now let's explore the principal use of programming languages:
# - We can note that SQL has a strong correlation with  "Analyze and understand data to influence product or business decisions". In my opinion this is caused by the fact that sql is used a lot in business context and, for example, not so much in reseach
# - Also R has a significative correlation with the latter, because R is mainly used by statistician and they care a lot understand data, causal relations and so on
# - The latter hypothesis is corfirmed also by the correlation with SAS/STATA
# - It is also interesting note that research use a lot Python because popularity and the presence of Deep Learning libraries because DL is the main research area today in ML
# - Also  C/C++ and MATLAB are used for research purpuose. The choice of C/C++ implies more difficult to code compared to Python (e.g. manual memory management) but allow more speed of execution and also a more control of what happend under the hood.
# - Scala has a significative correlation with "Building a data pipeline" thanks to Spark library
# 
# ##### Technical notes:
# This heatmap is extracted from the correlation matrix calculated with [CramerV](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V) correlation coefficient beween the multiple choice responses between question 11 and 16.  If you have seen the csv the matrix involve the columns of type "Q11_Part_&#65121;" and "Q16_Part_&#65121;, then from the square correlation matrix i've extracted the rectangular part that involve only the correlation between the possible responses in Q11 and Q16

# In[ ]:


ax = plot_cross_correlation('Q11', 'Q16')
plt.show()


# In[ ]:


crosstab = pd.crosstab(multiple_choice.Q17[1:], multiple_choice.Q5[1:])
norm_crosstab = crosstab.divide(crosstab.sum(axis=1), axis='rows')
ax = sns.heatmap(norm_crosstab.T)
ax.set_xlabel('')
ax.set_ylabel('')

plt.show()


# In[ ]:


ax = sns.heatmap(normalized_crosstab('Q17', 'Q10'))
# sns.heatmap(pd.crosstab(multiple_choice.Q17[1:], multiple_choice.Q10[1:]).T)
plt.show()


# ### Earning by programming language
# 
# ##### Tecnical notes:
# Since Python is the preponderant programming langauge an heatmap of "Earning by programming language" give light colours only to Python programming language so i thought that is more logical plot a normalized version of earning, in math terms every row is divided by the frequency of the language

# In[ ]:


counts = multiple_choice.Q17[1:].value_counts()
q9_true_order = [0, 1, 5,8, 10, 12, 14, 15, 16, 17, 2, 3, 4 ,6 , 7, 9, 11, 13, 18]
crosstab = (pd.crosstab(multiple_choice.Q17[1:], multiple_choice.Q9[1:]))
crosstab = crosstab.iloc[:, q9_true_order]
crosstab = crosstab.rename({crosstab.columns[-1]:"I don't want to disclose"}, axis='columns')
norm_crosstab = crosstab.divide(crosstab.sum(axis=1), axis='rows')
fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(norm_crosstab, ax=ax)
plt.tight_layout()
ax.set_xlabel('')
ax.set_ylabel('')
plt.show()


# ### IDE used by language

# In[ ]:


ax = plot_cross_correlation('Q13', 'Q16')
plt.show()


# ## Machine learning frameworks
# Nowdays exists many machine learning frameworks to do all kinds of machine learning algorithms. Most of them are for Python or R, so this cause a strong preference to this two languages in the field of Data Science.<br>
# Some are generic collection of machine learning algorithms while some others implements very well a single powerful algorithm. In the first set we have, for example, mlr and Scikit-learn for R and python respectively while in the second set we have for example randomForest, lightgbm 

# ### Most popular frameworks

# In[ ]:


mlf_col = 'Q19'


# In[ ]:


mlf_cols = cols_part_questionary(mlf_col)
mlf_counts = (~multiple_choice.loc[1:, mlf_cols].isna()).sum()
cols_values = get_cols_values(multiple_choice.loc[:, mlf_cols])
mlf_counts.index = cols_values.values
mlf_counts.plot.barh()
plt.show()


# In[ ]:


ax = plot_cross_correlation('Q11', mlf_col)
ax.set_title(survey_schema.Q11[0])
plt.show()


# ### Machine learning frameworks used in production:
# Interesting note: A 33% of the people who use CNTK and Mxnet (Deep learning libraries) have in production a ML model for more than two year

# In[ ]:


ax = sns.heatmap(normalized_crosstab('Q20', 'Q10'))
plt.show()


# ## Stationarity in the same company
# While i was exploring the data i noted a pattern in many CramerV correlation matrix: in different topics there is the same tendency to have an high correlation between products from the same company

# #### Cloud computing

# In[ ]:


_ = plot_correlation_question('Q27')
del _


# #### DBMS
# Is interesting to note that there is a correlation "square" between the most popular DBMS

# In[ ]:


plt.subplots(figsize=(8, 5))
_ = plot_correlation_question('Q29')
del _


# #### Machine learning products
# Machine learning products are different from the last cited Machine learning frameworks because have some kind of web-interface or it is a software program to run locally.<br>
# Personally i don't like too much for the lack of flexibility but i admit they are good for anyone who don't want going too technical and deep inside machine learning field

# In[ ]:


plt.subplots(figsize=(10, 7))
_ = plot_correlation_question('Q28')
del _


# ## The university effectiveness
# There are a modest group of people that dedicate all  or the most of time to the university but if we see the results of question 40, a better expertize is demonstrated more likely from indipendent projects rather than academic achievements.<br>
# I don't say that university is a waste of time, i also do actually a master degree, but it is important in my opinion differentiate the kind of things we do in university: for example instead of dedicate three hours to the study for an exam you can study two hours and the last remaining you can watch a youtube video about autoencoders, or partecipate into a kaggle competition 

# In[ ]:


legend_cols = multiple_choice.loc[0, cols_part_questionary('Q35')].str.split(' - ').tolist()[:-1]
legend_cols = list(map(lambda el: el[1], legend_cols))
time_use = multiple_choice.loc[1:, cols_part_questionary('Q35')[:-1]].fillna(0).astype('float')
time_use.columns = legend_cols
time_use.plot.hist(bins=15, title=survey_schema.Q35[0])
plt.xlim((1, 100))
plt.show()


# In[ ]:


multiple_choice.Q40[1:].value_counts().plot.barh(title=multiple_choice.Q40[0])
plt.show()

