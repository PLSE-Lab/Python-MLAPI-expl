#!/usr/bin/env python
# coding: utf-8

# # Dataset Creation: Fusing survey data from 2014-2018
# 
# ## Introduction
# This kernel describes analysis of the 'mental health in technology' survey performed in 2014, by the non-profit corporation named "Open Sourcing Mental Illness" (OSMI). This survey was one of the largest efforts to collect data about mental health amongst technology workers in the US. 
# 
# The questions covered in this dataset cover three general areas:
# * **Demographic**: for example the state the person lives or works in and their gender. Much to the dismay of categorical attribute lovers everywhere, the "gender" column was left as a free-form text input.
# * **Social impact and perceptions of mental health**: For example, how a person may be treated by their co-workers or senior management after disclosing a mental health issue. 
# * **Diagnosis**: Whether the person has received an official diagnoses of a mental health issue. After inspecting these columns, a significant class imbalance can be seen in favour of people without a diagnosis. This makes sense. It is also quite useful to know if you were planning on using this dataset for a classification task. 
# 
# However, whilst the dataset has value in isolation, it is clear that the social landscape surrounding mental health has changed over the last 4 years. Luckily, this survey has also been performed in 2016, 2017 and 2018 by OSMI. Not 2015 though. Clearly not the year for exposing your psychological ecology to the world. Regardless, aggregating this data to create a dataset across the years is far more useful in terms of insight compared to a single year alone.
# 
# However, a roadblock to this goal was the lack of standardisation of the survey questions year to year. One such example is:
# 
# **2014**
# > "Would you be willing to discuss a mental health issue with a coworker?"
# 
# **2016**
# > "Would you be willing to discuss a mental health issue with your coworkers?"
# 
# As you can imagine, merging these two attributes based on a complete string match would result in *two* unique columns being created rather than one. So, the aim of this notebook is to describe how I attempted to homogenise these attributes. 
# 
# ## Main motivation
# * What are some methods to combine tabular data, where the attributes (e.g survey questions) refer to the same phenomena
# 
# ## Topics covered
# * Natural language processing (word vectors, similarity, text cleaning)
# * Treating missing data
# 
# # Imports

# In[ ]:


import numpy as np 
import pandas as pd
import plotly.graph_objs as go
import spacy
import re
from IPython.display import display, HTML
import plotly.plotly as py
import plotly.offline as pyo
import plotly.graph_objs as go
import os
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)


# # 1. Aggregating the data across years
# 
# To perform a flat aggregation, we will first standardise the "low-hanging fruit" attributes. For example, we can rename "What is your age?" to simply "age". I also noticed that there were a few `HTML` artifacts in some of the questions, e.g `<br></br>`, so these were removed too. 

# In[ ]:


data_paths = {
    '2014': '../input/mental-health-in-techology-survey-2014-and-2016/survey_2014.csv',
    '2016': '../input/mental-health-in-techology-survey-2014-and-2016/survey_2016.csv',
    '2017': '../input/osmi-mental-health-in-tech-survey-2017/OSMI Mental Health in Tech Survey 2017.csv',
    '2018': '../input/osmi-mental-health-in-tech-survey-2018/OSMI Mental Health in Tech Survey 2018.csv'
}

raw_data = {
    date: pd.read_csv(path, index_col=False) for date, path in data_paths.items()
}


# In[ ]:


def clean_columns(dataframe, year):

    dataframe.columns = map(str.lower, dataframe.columns)

    # Remove HTML artifacts
    dataframe.rename(columns=lambda colname: re.sub('</\w+>', '', colname), inplace=True)
    dataframe.rename(columns=lambda colname: re.sub('<\w+>', '', colname), inplace=True)

    # Standardise demographic questions
    dataframe.rename(columns={'what is your age?': 'age', 'what is your gender?': 'gender',
                              'what is your race?': 'race'},
                     inplace=True)

    # Following the 2014 convention where 'country' refers to country of living
    unused_columns = ['country_work', 'state_work', 'timestamp']
    for column in unused_columns:
        if column in dataframe.columns:
            dataframe.drop(columns=column, inplace=True)

    dataframe['year'] = year

    if {'#', 'start date (utc)', 'submit date (utc)', 'network id'}.issubset(set(dataframe.columns)):
        dataframe.drop(columns=['#', 'start date (utc)', 'submit date (utc)', 'network id'], inplace=True)

    # Drop duplicated columns
    dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()]

    dataframe.reset_index(inplace=True, drop=True)

    return dataframe


# In[ ]:


for dataset in ['2014', '2016', '2017', '2018']:
    raw_data[dataset] = clean_columns(raw_data[dataset], dataset)
    
initial_concat = pd.concat([raw_data['2014'], raw_data['2016']], ignore_index=True, sort=True)
interm_concat = pd.concat([raw_data['2017'], raw_data['2018']], ignore_index=True, sort=True)
survey_dataframe = pd.concat([initial_concat, interm_concat], ignore_index=True, sort=True)

del initial_concat, interm_concat
display(HTML(survey_dataframe.head(3).to_html()))


# Now we have a "super" dataframe, which is the aggregate of all of the years following a join. Columns that can be merged, e.g "age" and "gender" have been, whilst all unique columns have been appended.
# 
# Let us visualise the distribution of missing data.

# In[ ]:


def generate_missing_value_heatmap(dataframe):
    """
    Generates a plotly heatmap to graphically display missing values in a pandas dataframe

    :param dataframe: Pandas dataframe, missing values should be of type `numpy.nan`
    :type dataframe: Pandas DataFrame

    :return:
    :rtype: Python dictionary with keys 'data' and 'layout', containing a plotly.Heatmap and plotly.Layout object.

    """
    val_array = np.array(dataframe.fillna(-99).values)
    val_array[np.where(val_array != -99)] = 0
    val_array[np.where(val_array == -99)] = 1

    data = [
        go.Heatmap(
            z=val_array,
            x=dataframe.columns,
            y=dataframe.index,
            colorscale='Reds',
           hovertemplate='Question: %{x}\n Missing?: %{z}'
        )
    ]

    layout = go.Layout(
    title=dict(text="Missing data heatmap (red values indicate missing). Hover to see responses with missing values.",
               font=dict(size=24)),
    autosize = True, 
    xaxis=dict(
        showticklabels=False, 
        ticks="", 
        showgrid=False,
        zeroline=False,
        automargin=False,
        tickmode='array',
    ),
    yaxis=dict(
        autorange=True,
        tickmode='array',
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks="",
        automargin=False,
        showticklabels=False),
    )

    fig = dict(data=data, layout=layout)

    return fig


# In[ ]:


iplot(generate_missing_value_heatmap(survey_dataframe))


# Ouch. Although a large quantity of data **is** missing, it can be seen that there are two reasons explaining the rest of it:
# 
# 1. After merging the datasets, columns that ask the same question, with a slightly different wording/spelling are treated as seperate columns across the years
# 2. Many columns are relating to a diagnosis of a mental health disorder, and are simply named for example "adhd", or "schizophrenia". These are also duplicated throughout a single survey e,g "schiqophrenia.1, schizophrenia.2..." in response to different questions
# 
# Let's see if addressing these attributes makes a significant difference in the missing value distribution.

# # 2. Merging attributes using word vector similarity

# ## 2.a Word vectors: A brief introduction
# Vector space models are used in the field of natural language processing to encode natural words as continuous vectors. These word "embeddings" are named as such, as these models are based on the principle that words that occur in similar contexts share some underlying meaning. A novel aspect of word vectors is their ability to let a person compute the *similarity* between words, by calculating the distance between their embeddings in the vector space. For example in the below figure, we can see that "sir" is much closer to "man" and "king" than it is to "madam", of which it is also related to.
# 
# ![Word vector example](https://nlp.stanford.edu/projects/glove/images/man_woman.jpg)
# > "Image used from the Stanford page [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
# 
# For this kernel, a vector space model from SpaCy will be used to compute a **similarity score** between each question in the survey. This way, we can identify which attributes are the most similar, and see if we can merge on these too. Unfortunately, SpaCy models (anything larger than the 'small' class) are not able to be loaded into kaggle kernels. I used the `en_core_wb_md` offline to create the similarity matrix between all of the survey questions and have included the function that I used. For the purposes of this notebook however, I have uplodaded the pre-computed similarity matrix with the supporting raw 2014 dataset.

# In[ ]:


# !python -m spacy download en_core_web_md

# word_vecs = spacy.load('en_core_web_md')


# For those interested, here is the code I used to generate the similarity matrix. 

# In[ ]:


def compute_similarity_matrix(documents, word_vecs):
    if not isinstance(documents, list):
        raise ValueError("Documents must be a list of strings")
    else:
        similarity_matrix = np.zeros(shape=(len(documents), len(documents)))

        for question_i in range(len(documents)):
            for question_j in range(len(documents)):
                if question_i == question_j:
                    continue
                else:
                    question_i_vec = word_vecs(documents[question_i])
                    question_j_vec = word_vecs(documents[question_j])
                    similarity_matrix[question_i, question_j] = question_i_vec.similarity(question_j_vec)
        return similarity_matrix
    
def remove_punctuation(documents):
    punctuation = re.compile('[\?\(\)\.\,]')
    if isinstance(documents, list):
        clean_documents = [''] * len(documents)
        for i, document in enumerate(documents):
            clean_documents[i] = re.sub(punctuation, '', document)
    elif isinstance(documents, str):
        clean_documents = re.sub(punctuation, '', documents)
    else:
        raise ValueError("Documents must be a list of strings or a string")

    return clean_documents


# In[ ]:


header = remove_punctuation(list(survey_dataframe.columns))
# This is a costly operation, and needs a SpaCy model with word vectors to be performed.
# similarity_matrix = compute_similarity_matrix(header, word_vecs)

# In kernel mode, let us just load this from the pre-computed weights
similarity_matrix = np.load('../input/mental-health-in-techology-survey-2014-and-2016/similarity_matrix.npy')


# ## 2.b Plotting the similarity matrix

# In[ ]:


def generate_similarity_heatmap(similarity_matrix, labels=None):
    
    if labels is None or not isinstance(labels, list):
        labels = np.arange(similarity_matrix.shape[0])
    elif similarity_matrix.shape[0] != similarity_matrix.shape[1]:
        raise ValueError("Please provide a square matrix")
    else:
        pass
    
    data = go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='Hot',
        )

    layout = go.Layout(
        title=dict(text="Zoomable Similarity Matrix (hover to see the similarity between each label)",
                   font=dict(size=24)),
        autosize = True, 
        xaxis=dict(
            showticklabels=False, 
            ticks="", 
            showgrid=False,
            zeroline=False,
            automargin=False,
            tickmode='array',
        ),
        yaxis=dict(
            autorange=True,
            tickmode='array',
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="",
            automargin=False,
            showticklabels=False),
    )

    fig = dict(data=[data], layout=layout)
    return fig


# Now we can visualise the similarity between all of the questions. It can be seen from this plot that questions such as "adhd.1", "adhd.2" obviously have a near perfect similarity score. As hoped for, questions such as  "Would you be willing to discuss a mental health issue with your coworkers?" and "Would you be willing to discuss a mental health issue with a coworker?" also have a very high score (~>0.95)

# In[ ]:


iplot(generate_similarity_heatmap(similarity_matrix, labels=header))


# ## 2.c Merging attribute columns based on similarity
# 
# **WIP.**

# # Conclusion
# Following this notebook, an interesting facet of this new dataset I want to explore is transforming the "gender" column from a continuous/freeform text column into a categorical attribute. Some example classes would be "cisgender male/female", "transgender male/female", "non-binary/genderfluid" and "other gender expression". This way, a correlation or similar analysis could be performed to investigate the relationship between gender expression and mental health issues within this demographic.
# 
# Thanks for reading.
# 
