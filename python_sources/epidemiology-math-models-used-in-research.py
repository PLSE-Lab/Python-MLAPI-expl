#!/usr/bin/env python
# coding: utf-8

# # Epidemiology math models used in Research (work in progress!)

# I would like to ackowledge Andy White and his excellent notebook [COVID-19 Thematic tagging with Regular Expressions](https://www.kaggle.com/ajrwhite/covid-19-thematic-tagging-with-regular-expressions#kln-97). This notebook is just an expansion of it and relies on his philosphy. 
# 
# **This notebook can help you to evaluate and discover new math models for the forecasting competition!**
# 

# Leave a like if you like the notebook. 
# High five for all the math-mates around here! :)

# ## Contents
# 
# 1. Introduction
# 2. Goals
# 3. Setup
# 4. Choosing the models
# 5. Getting the papers
# 6. Extracting the parameters used.
# 7. Bibliography

# ## Introduction

# ### If you want a good book about this topic: [Mathematical Models in Epidemiology](https://doi.org/10.1007/978-1-4939-9828-9)
# 
# *(From Wikipedia, the free encyclopedia)*
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/bd/SIR-Modell.svg/1280px-SIR-Modell.svg.png)
# ### Definition
# **Mathematical models** can project how infectious diseases progress to show the likely outcome of an epidemic and help inform public health interventions. Models use basic assumptions and/or collected statistics along with mathematics to **find parameters** for various infectious diseases and use those parameters to calculate the effects of different interventions, like mass vaccination programmes. **The modelling can help decide which intervention/s to avoid and which to trial, or can predict future growth patterns, etc.**
# 
# ### Types of epidemic models
# 
# - **Stochastic:** means being or having a random variable. A stochastic model is a tool for estimating probability distributions of potential outcomes by allowing for random variation in one or more inputs over time. Stochastic models depend on the chance variations in risk of exposure, disease and other illness dynamics.
# - **Deterministic:** When dealing with large populations, as in the case of covid, **deterministic or compartmental mathematical models are often used**. In a deterministic model, individuals in the population are assigned to different subgroups or compartments, each representing a specific stage of the epidemic. **Letters such as M, S, E, I, and R are often used to represent different stages**. The transition rates from one class to another are mathematically expressed as derivatives, hence the model is formulated using differential equations. While building such models, it must be assumed that the population size in a compartment is differentiable with respect to time and that the epidemic process is deterministic. In other words, the changes in population of a compartment can be calculated using only the history that was used to develop the model
# 
# ### Example: SIR model
# 
# 
# ### Common models:
# - SIR [link](http://idmod.org/docs/general/model-sir.html)
# - SEIR [link](https://sites.me.ucsb.edu/~moehlis/APC514/tutorials/tutorial_seasonal/node4.html)
# - SEIRS [link](http://www.public.asu.edu/~hnesse/classes/seirs.html)
# - SIS [link](https://sites.me.ucsb.edu/~moehlis/APC514/tutorials/tutorial_seasonal/node2.html)
# - SIRS [link](http://idmod.org/docs/general/model-sir.html)
# - SEIS [link](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)
# - MSIR [link](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)
# - MSEIR [link](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)
# - MSEIRS [link](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology)
# - SEIRV [link](https://doi.org/10.3934/mbe.2020148)
# - SIRV [link](http://)
# - SIQR [link](http://)
# - SVEIS [link](http://)
# - SEIQS [link](http://)
# 
# ### Main focus:
# 
# Our main focus is to discover how the epidemic is being modeled. As it is said we are going to use the most common letters in order to find acronyms in the research papers. 
# 
# ### Work in progress
# - Use thw whole text
# - cleaning text a bit more in order to avoid mistakes.
# - Getting parameter information of the models used.
# - Getting information about math models that are based on network dynamics.
# - Getting information about any stochastic model used in bibliography.
# - Conclusions.

# ## Goals: 
# - Find the **common math models and variations** that are being used in order to predict or analize COVID19 (that way if you are making some model or improving one, you can fastly access to bibliography about the model you are using).
# - Get the **parameter information** regarding each model in order to set up any simulation quicly or contrast your model.
# - Check the issues that some of the classical approaches may have in order to prevent possible changes in predicted trends.
# 

# ## Setup

# Getting the data

# In[ ]:


# Data libraries
import pandas as pd
import re
import pycountry

# Visualisation libraries
import plotly.express as px
import plotly.graph_objects as go

get_ipython().run_line_magic('matplotlib', 'inline')

# Load data
metadata_file = '../input/CORD-19-research-challenge/metadata.csv'
df = pd.read_csv(metadata_file,
                 dtype={'Microsoft Academic Paper ID': str,
                        'pubmed_id': str})

def doi_url(d):
    if d.startswith('http'):
        return d
    elif d.startswith('doi.org'):
        return f'http://{d}'
    else:
        return f'http://doi.org/{d}'
    
df.doi = df.doi.fillna('').apply(doi_url)

print(f'loaded DataFrame with {len(df)} records')


# Helpers functions

# In[ ]:


# Helper function for filtering df on abstract + title substring
def abstract_title_filter(search_string):
    return (df.abstract.str.lower().str.replace('-', ' ').str.contains(search_string, na=False) |
            df.title.str.lower().str.replace('-', ' ').str.contains(search_string, na=False))
def full_text_filter(search_string):
    return (df.abstract.str.lower().str.replace('-', ' ').str.contains(search_string, na=False) |
            df.title.str.lower().str.replace('-', ' ').str.contains(search_string, na=False)|
            df.full_text.str.lower().str.replace('-', ' ').str.contains(search_string, na=False))


# In[ ]:


# Helper function which counts synonyms and adds tag column to DF
def count_and_tag(df: pd.DataFrame,
                  synonym_list: list,
                  tag_suffix: str) -> (pd.DataFrame, pd.Series):
    counts = {}
    df[f'tag_{tag_suffix}'] = False
    for s in synonym_list:
        synonym_filter = abstract_title_filter(s)
        counts[s] = sum(synonym_filter)
        df.loc[synonym_filter, f'tag_{tag_suffix}'] = True
    return df, pd.Series(counts)

# Helper function which counts synonyms and adds tag column to DF
def count_and_tag_full_text(df: pd.DataFrame,
                  synonym_list: list,
                  tag_suffix: str) -> (pd.DataFrame, pd.Series):
    counts = {}
    df[f'tag_{tag_suffix}'] = False
    for s in synonym_list:
        synonym_filter = full_text_filter(s)
        counts[s] = sum(synonym_filter)
        df.loc[synonym_filter, f'tag_{tag_suffix}'] = True
    return df, pd.Series(counts)


# In[ ]:


# Helper function for Cleveland dot plot visualisation of count data
def dotplot(input_series, title, x_label='Count', y_label='Regex'):
    subtitle = '<br><i>Hover over dots for exact values</i>'
    fig = go.Figure()
    fig.add_trace(go.Scatter(
    x=input_series.sort_values(),
    y=input_series.sort_values().index.values,
    marker=dict(color="crimson", size=12),
    mode="markers",
    name="Count",
    ))
    fig.update_layout(title=f'{title}{subtitle}',
                  xaxis_title=x_label,
                  yaxis_title=y_label)
    fig.show()


# In[ ]:


# Function for printing out key passage of abstract based on key terms
def print_key_phrases(df, key_terms, n=5, chars=300):
    for ind, item in enumerate(df[:n].itertuples()):
        print(f'{ind+1} of {len(df)}')
        print(item.title)
        print('[ ' + item.doi + ' ]')
        try:
            i = len(item.abstract)
            for kt in key_terms:
                kt = kt.replace(r'\b', '')
                term_loc = item.abstract.lower().find(kt)
                if term_loc != -1:
                    i = min(i, term_loc)
            if i < len(item.abstract):
                print('    "' + item.abstract[i-30:i+chars-30] + '"')
            else:
                print('    "' + item.abstract[:chars] + '"')
        except:
            print('NO ABSTRACT')
        print('---')


# #### As it is said, there are some papers that are not related to covid. Eventhough these papers could be interesting to analyze in the parameter section, we omit them for now.

# In[ ]:


covid19_synonyms = ['covid',
                    'coronavirus disease 19',
                    'sars cov 2', # Note that search function replaces '-' with ' '
                    '2019 ncov',
                    '2019ncov',
                    r'2019 n cov\b',
                    r'2019n cov\b',
                    'ncov 2019',
                    r'\bn cov 2019',
                    'coronavirus 2019',
                    'wuhan pneumonia',
                    'wuhan virus',
                    'wuhan coronavirus',
                    r'coronavirus 2\b']


# In[ ]:


df, covid19_counts = count_and_tag(df, covid19_synonyms, 'disease_covid19')


# In[ ]:


covid19_counts.sort_values(ascending=False)


# In[ ]:


novel_corona_filter = (abstract_title_filter('novel corona') &
                       df.publish_time.str.startswith('2020', na=False))
print(f'novel corona (published 2020): {sum(novel_corona_filter)}')
df.loc[novel_corona_filter, 'tag_disease_covid19'] = True


# In[ ]:


df[df.tag_disease_covid19].publish_time.str.slice(0, 4).value_counts(dropna=False)


# In[ ]:


df.loc[df.tag_disease_covid19 & ~df.publish_time.str.startswith('2020', na=True),
       'tag_disease_covid19'] = False


# ## Choosing the most common math models

# The most common math model acronyms are:
# - 's.i.r', 'sir',
# -    'seir','s.e.i.r',
# -    'sis','s.i.s',
# -    'sirs','s.i.r.s',
# -    'seis', 's.e.i.s',
# -    'seir','s.e.i.r',
# -  'msir', 'm.s.i.r.'
# -    'mseir', 'm.s.e.i.r',
# -   'seirv','s.e.i.r.v',
# -    'nac', 'nac', 
# -    'nac seirv','nac s.e.i.r.v',
# -    'sirv','s.i.r.v',
# -    'siqr','s.i.q.r',
# -   'sveis','s.v.e.i.s',
# -    'seiqs','s.e.i.q.s',
# -    'iar','i.a.r',
# -    'network model',

# We first check all the abbreviatures and then we can go one by one

# In[ ]:


repr_synonyms = [
    'math',
    's.i.r', 'sir',
    'seir','s.e.i.r',
    'sis','s.i.s',
    'sirs','s.i.r.s',
    'seis', 's.e.i.s',
    'seirs','s.e.i.rs',
    'msir', 'm.s.i.r.',
    'mseir', 'm.s.e.i.r',
    'mseirs','m.s.e.i.r.s',
    'seirv','s.e.i.r.v',
    'nac', 'nac', 
    'nac seirv','nac s.e.i.r.v',
    'sirv','s.i.r.v',
    'siqr','s.i.q.r',
    'sveis','s.v.e.i.s',
    'seiqs','s.e.i.q.s',
    'iar','i.a.r',
    'network model',
]


# Adding the results to dataframe and plotting

# In[ ]:


df, repr_counts = count_and_tag(df,repr_synonyms, 'Math_models_used')
dotplot(repr_counts, 'Math models by title / abstract metadata')


# In[ ]:


repr_counts.sort_values(ascending=False)


# Let's check how many of them are related to covid19

# In[ ]:


n_math = (df.tag_disease_covid19 & df.tag_Math_models_used).sum()
n_math


# ## So now that we know there are a lot of articles from these models, let's make a deeper analysis, model by model

# ### MSIR

# In[ ]:


msir_synonyms = [
    'msir','m.s.i.r',
]
df, msir_counts = count_and_tag(df,msir_synonyms, 'MSIR')
dotplot(msir_counts, 'MSIR models by title / abstract metadata')
msir_counts.sort_values(ascending=False)
n_msir = (df.tag_disease_covid19 & df.tag_MSIR).sum()
n_msir
n_msir_no_covid_filter = (df.tag_MSIR).sum()
n_msir_no_covid_filter


# Unfortunately, there is no match with SIRV model. It looks that "no one" (in our database) has used it.

# ### SIRV

# In[ ]:


sirv_synonyms = [
    'sirv','s.i.r.v',
]
df, sirv_counts = count_and_tag(df,sirv_synonyms, 'SIRV')
dotplot(sirv_counts, 'SIRV models by title / abstract metadata')
sirv_counts.sort_values(ascending=False)
n_sirv = (df.tag_disease_covid19 & df.tag_SIRV).sum()
n_sirv
n_sirv_no_covid_filter = (df.tag_SIRV).sum()
n_sirv_no_covid_filter


# Unfortunately, there is no match with SIRV model. It looks that "no one" (in our database) has used it.

# ### SEIS

# In[ ]:


seis_synonyms = [
    'seis', 's.e.i.s',
]
df, seis_counts = count_and_tag(df,seis_synonyms, 'SEIS')
dotplot(seis_counts, 'SEIS models by title / abstract metadata')
seis_counts.sort_values(ascending=False)
n_seis = (df.tag_disease_covid19 & df.tag_SEIS).sum()
n_seis
n_seis_no_covid_filter = (df.tag_SEIS).sum()
n_seis_no_covid_filter


# printing the title

# In[ ]:


print(df[df.tag_disease_covid19 & df.tag_SEIS]['title'])


# ### SIRS

# In[ ]:


sirs_synonyms = [
    'sirs','s.i.r.s',
]
df, sirs_counts = count_and_tag(df,sirs_synonyms, 'SIRS')
dotplot(sirs_counts, 'SIRS models by title / abstract metadata')
sirs_counts.sort_values(ascending=False)
n_sirs = (df.tag_disease_covid19 & df.tag_SIRS).sum()
n_sirs
n_sirs_no_covid_filter = (df.tag_SIRS).sum()
n_sirs_no_covid_filter 


# In[ ]:


print(df[df.tag_disease_covid19 & df.tag_SIRS]['title'])


# ### SEIR
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/SEIR.PNG/800px-SEIR.PNG)

# In[ ]:


seir_synonyms = [
    'seir','s.e.i.r',
]
df, seir_counts = count_and_tag(df,seir_synonyms, 'SEIR')
dotplot(seir_counts, 'SEIR models by title / abstract metadata')
seir_counts.sort_values(ascending=False)
n_seir = (df.tag_disease_covid19 & df.tag_SEIR).sum()
n_seir

n_seir_no_covid_filter = (df.tag_SEIR).sum()
n_seir_no_covid_filter


# It is remarkable to see that a interesting amount of articles related to SEIR models are from COVID19

# In[ ]:


print(df[df.tag_disease_covid19 & df.tag_SEIR]['title'])


# ### SIR
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8a/SIR.PNG/600px-SIR.PNG)

# In[ ]:


sir_synonyms = [
    'sir','s.i.r',
]
df, sir_counts = count_and_tag(df,sir_synonyms, 'SIR')
dotplot(sir_counts, 'SIR models by title / abstract metadata')
sir_counts.sort_values(ascending=False)
n_sir = (df.tag_disease_covid19 & df.tag_SIR).sum()
n_sir

n_sir_no_covid_filter = (df.tag_SIR).sum()
n_sir_no_covid_filter


# In[ ]:


print(df[df.tag_disease_covid19 & df.tag_SIR]['title'])


# ### SIS
# ![](https://upload.wikimedia.org/wikipedia/commons/thumb/1/16/SIS.PNG/400px-SIS.PNG)

# In[ ]:


sis_synonyms = [
    'sis','s.i.s',
]
df, sis_counts = count_and_tag(df,sis_synonyms, 'SIS')
dotplot(sis_counts, 'SIS models by title / abstract metadata')
sis_counts.sort_values(ascending=False)
n_sis = (df.tag_disease_covid19 & df.tag_SIS).sum()
n_sis
n_sis_no_covid_filter = (df.tag_SIS).sum()
n_sis_no_covid_filter


# In[ ]:


print(df[df.tag_disease_covid19 & df.tag_SIS]['title'])


# ## Final plot

# In[ ]:


from bokeh.io import output_file, show, output_notebook
from bokeh.plotting import figure
output_notebook()
models_name = ['SIR','SIS','SIRV','SEIS','SEIR','SIRS','MSIR']
counts = [n_sir,n_sis,n_sirv,n_seis,n_seir,n_sirs,n_msir]
p = figure(x_range=models_name, plot_height=250, title="Model Counts")
p.xgrid.grid_line_color = None
p.y_range.start = 0
p.vbar(x=models_name, top=counts, width=0.9)
show(p)

counts


# In[ ]:


from bokeh.io import output_file, show, output_notebook
from bokeh.plotting import figure
output_notebook()
models_name = ['SIR','SIS','SIRV','SEIS','SEIR','SIRS','MSIR']
counts_no = [n_sir_no_covid_filter,n_sis_no_covid_filter,n_sirv_no_covid_filter,n_seis_no_covid_filter,n_seir_no_covid_filter,n_sirs_no_covid_filter,n_msir_no_covid_filter]
p = figure(x_range=models_name, plot_height=250, title="Model Counts without COVID19 filter")
p.xgrid.grid_line_color = None
p.y_range.start = 0
p.vbar(x=models_name, top=counts_no, width=0.9)
show(p)
counts_no


# ## Parameter extraction

# ### SIR

# In[ ]:


df_SIR = df[df.tag_disease_covid19 & df.tag_SIR]
df_SIR.head()
df_SIR.shape


# SIR model usually uses two parameters: beta and gamma. Let's try to find them.

# In[ ]:


parameter_sir_synonyms = [
    'beta','betta',
    'gama','gamma'
]
df, parameter_sir_counts = count_and_tag(df,parameter_sir_synonyms, 'parameter_sir')
dotplot(parameter_sir_counts, 'parameter_sir by text metadata')
parameter_sir_counts.sort_values(ascending=False)
n = (df.tag_disease_covid19 & df.tag_SIR & df.tag_parameter_sir).sum()
n


# In[ ]:


print_key_phrases(df[df.tag_disease_covid19 & df.tag_SIR & df.tag_parameter_sir], parameter_sir_synonyms, n=52, chars=500)


# In[ ]:





# ## Conclusions

# ### - It is interesting that, although SIR and SIS are separate models (one admits recoveries while the other does not) they are the main two used for this purpose.
# The main reason may be that large variations can be made on them, both in the way of modeling and in the parameters, due to their simplicity of approach. However, these simplifications are also accompanied by possible errors when using these models in prediction.
# 
# ### - On the other hand, it is interesting to highlight the SEIR model, (whose phases are susceptible, exposed, sick and recovered) has been widely used for this particular epidemic.
# 
# ### - In any case, the use of models with recovered or cannot be identified with the generalized attitude of treating recoveries as individuals that cannot be infected. However, if this disease is prolonged in time, they could be infected again, so the use of the R (recovered) phases should be taken with caution.
# 
# ### - I know SEIRV model has been used quite nicely in Wuhan outbreak, furthermore MSEIRS model could be quite interesting to see as well, given that the immunity in the R class would be temporary, so that individuals would regain their susceptibility when the temporary immunity ended. 

# ## Extra bibliography

# In[ ]:





# In[ ]:




