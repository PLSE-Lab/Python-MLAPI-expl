#!/usr/bin/env python
# coding: utf-8

# # **INTRODUCTION**
# 
# This work is to help the medical community answer the posted question in Kaggle: [What is known about vaccines and therapeutics?](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks) The resulting model enables to understand and keep up with the large amount of literature contained in the provided dataset, specifically:
# 
# 1. Effectiveness of drugs being developed and tried to treat COVID-19 patients.
# 2. Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.
# 3. Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.
# 4. Exploration of use of best animal models and their predictive value for a human vaccine.
# 5. Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.
# 6. Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.
# 7. Efforts targeted at a universal coronavirus vaccine.
# 8. Efforts to develop animal models and standardize challenge studies
# 9. Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers
# 10. Approaches to evaluate risk for enhanced disease after vaccination
# 11. Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]
# 
# At the end of this notebook, the question asked is answered by means of each of the aspects listed above. In each case, the original content of the article that most represents the processed aspect is shown.
# 
# To accomplish the goal, the selected approach was to perform text mining on input data, by applying the latest advances in natural language processing (NLP). This was realized by the following steps:
# 
#     1. Obtain input data and pre-process it to facilitate analysis.
#     2. Extract the key terms from the task description.
#     3. Match key terms with text contents.
#     4. Group and quantify the findings.
#     5. Show the documents that answer the task questions.
#         
# The advantage of using NLP to abord this problem is that it is based on language-specific models, saving time and resources for text analysis.
# 
# The key terms was extracted from the task description in the step 2, resulting in a set of rule-based patterns.
# 
# The matching of key terms (step 3) was applied through topic  classification with Spacy library. This work through large sets of the unstructured data to match the patterns obtained in step 2. It is a very fast and scalable process that preferably uses the GPU resource.
# 
# This approach presents the limitation of synonymy, where multiple words and phrases have the same or similar meaning. To counter this, great care was taken in selecting keywords to make up the vocabulary of terms (e.g. COVID-19, SARS-CoV-2, 2019-nCov, SARS Coronavirus 2, or 2019 Novel Coronavirus).
# 
# In other hand, since the [original input data](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge) it is stored in JSON single files, whose structure is likely too complex to directly perform the analysis, this notebook uses the pre-processed data from the dataset [CORD-19-CSV](https://www.kaggle.com/huascarmendez1/cord19csv), also of same authorship of this.
# 
# The preprocessing of the data further consisted of filtering the documents that specifically talk about the covid-19 disease and its other names, as well as that they dealt with related risk factors, among other general data review, counting and cleaning activities.
# 
# Finally, as it is clear that the results presented here are not final, it is recommended to assume them as a starting point for a complete understanding of each of the aspects that it tries to address.

# In[ ]:


import os, glob, pandas as pd


# ## Load Data

# In[ ]:


# Paths

input_dir = os.path.abspath('/kaggle/input/')
articles_dir = input_dir + '/cord19csv/'


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nli_df = []\n\nfor jt in [\'pdf\',\'pmc\']:\n    path = f\'{articles_dir}/{jt}*.csv\'\n    files = glob.glob(path)\n    \n    for file in files:\n        if jt == "pdf":            \n            df_pdf = pd.read_csv(file, index_col=None, header=0)\n            li_df.append(df_pdf)\n        else:\n            df_pmc = pd.read_csv(file, index_col=None, header=0)        \n            li_df.append(df_pmc)\n\n# Combine all papers dataframes in one\ndf = pd.concat(li_df, axis=0, ignore_index=True, sort=False)')


# In[ ]:


df.shape


# In[ ]:


df.head()


# Pre-process input data

# In[ ]:


# Drop duplicated documents by paper_id
df.drop_duplicates(subset="paper_id", keep='first', inplace=True)


# In[ ]:


# Drop duplicated documents by text
df.drop_duplicates(subset="doc_text", keep='first', inplace=True)
df.shape


# ## Pattern Matching
# Objective: classify all articles according to key terms.

# In[ ]:


# Create the lists of key terms

terms_group_id = "vaccines"

terms1 = [
    "Vaccines and therapeutics",
    "Efforts of vaccines",
    "Vaccines",
    "Efforts of therapeutics",
    "Therapeutics",
    "Therapeutics being developed",
    "Therapeutics being tried",
    "Effectiveness of drugs",
    "Drugs",
    "Drugs being developed",
    "Drugs being tried",
    "Clinical trials",
    "Bench trials",
    "Investigate less common viral inhibitors",
    "Naproxen",
    "Clarithromycin",
    "Minocyclinethat",
    "Exert effects on viral replication",
    "Exert effects"
]

terms2 = [
    "Potential complication",
    "Antibody-dependent enhancement",
    "Vaccine recipients",
    "Ade"
]

terms3 = [
    "Best animal models",
    "Predictive value",
    "Human vaccine"
]

terms4 = [
    "Capabilities to discover a therapeutic",
    "Not vaccine",
    "Clinical effectiveness studies",
    "Discover therapeutics",
    "include antiviral agents",
    "Antiviral agents"
]

terms5 = [
    "Alternative models",
    "Alternative",
    "Aid decision makers",
    "Prioritize and distribute scarce",
    "Prioritize scarce",
    "Distribute scarce",
    "Scarse",
    "Newly proven therapeutics",
    "Proven",
    "Production capacity",
    "Capacity",
    "Equitable distribution",
    "Timely distribution"
]

terms6 = [
    "Universal coronavirus vaccine",
    "Universal vaccine",
    "Coronavirus vaccine"
]

terms7 = [
    "Efforts to develop animal models",
    "Standardize challenge studies",
    "Challenge studies"
]

terms8 = [
    "Efforts to develop prophylaxis",
    "Develop prophylaxis",
    "Prophylaxis",
    "Healthcare workers"
]

terms9 = [
    "Approaches to evaluate risk",
    "Risk after vaccination",
    "After vaccine","After vaccination"
]

terms10 = [
    "Assays to evaluate vaccine",
    "Immune response",
    "Process development"
]

terms = terms1 + terms2 + terms3 + terms4 + terms5 
terms += terms6 + terms7 + terms8 + terms9 + terms10


# In[ ]:


import spacy

# Perform NLP operations on GPU, if available.
spacy.prefer_gpu()

# Load Spacy english model
nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])
nlp.max_length = 5000000


# In[ ]:


# Create matcher and patterns

from spacy.matcher import PhraseMatcher

# Create a Matcher to case insensitive text matching
matcher = PhraseMatcher(nlp.vocab, attr='LEMMA') 

# Create patterns from terms
patterns = [nlp(d) for d in terms]
matcher.add(terms_group_id, None, *patterns)


# In[ ]:


# Defines the matcher

def cord_19_matcher(sample_pct):   
    # variables to test: test_limt is the total of docs to test; 
    # 0 = test off
    
    test_limit = 0
    counter = 0

    docs = df.sample(frac = sample_pct/100) if sample_pct < 100 else df
    tdocs = str(len(docs))

    print(f"{tdocs} documents to proccess...")
        
    # Maximun allowed length of string text document
    max_tlen = 100000

    # initialize array and total found variables
    findings_arr = []

    # loop all articles to match terms
    for idx, row in docs.iterrows():
        try:
            paper_id = row['paper_id']
            doc_text = row["doc_text"]            
            
            doc = nlp(doc_text)

            # get the matches
            matches = matcher(doc)

            # process all matches found in text
            if matches:
                for m in matches:
                    m_id, start, end = m[0],m[1],m[2]
                    term_group = nlp.vocab.strings[m_id]
                    term = doc[start:end].text

                    # put finding into json object
                    finding = {
                        "paper_id": paper_id,
                        "term_group": term_group,
                        "term": term
                    }

                    # append finding to findings array
                    findings_arr.append(finding)                

            counter += 1
            if counter % 100 == 0:
                print(f"{counter} documents proccessed")

            # breake loop if test control present
            if test_limit > 0:            
                if counter == test_limit:
                    print(test_limit, "sample count reached")
                    break

        except BaseException as e:
            print("Oops!  Error occurred in document loop.")
            print(str(e))
            print("Continuing...")
            continue
    
    return findings_arr


# Run the matcher

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Set sample parameter = % of papers to proccess\nsample_pct = 100\n#sample_pct = 1.2\n#sample_pct = 10\n\nfindings_arr = cord_19_matcher(sample_pct)\n\ntfound = len(findings_arr)\nprint(tfound, "matches found\\n")')


# In[ ]:


# Put findings array into a dataframe

findings = pd.DataFrame(findings_arr)

# exclude the following terms originally taken in account
#exc = ['term1','term2','term3']
#findings.where(~findings.term.isin(exc), inplace = True)


# In[ ]:


findings.info()


# In[ ]:


findings.head()


# In[ ]:


# Capitalize each term in findings
findings["term"] = findings["term"].str.capitalize()


# Quantify documents by key terms

# In[ ]:


findings['count'] = ''
cnt = findings.groupby('term').count()[['count']]
cnt_s = cnt.sort_values(by='count', ascending=False).copy()


# Display a bar graph and a word cloud with the totals of findings  by key term.

# In[ ]:


# Show the bar chart

ax = cnt_s.plot(kind='barh', figsize=(12,25), 
                legend=False, color="coral", 
                fontsize=16)
ax.set_alpha(0.8)
ax.set_title("What is Known About Vaccines and Therapeutics?",
             fontsize=18)
ax.set_xlabel("Term Appearances", fontsize=16);
ax.set_ylabel("Terms", fontsize=14);
ax.set_xticks([0,200,400,600,800,1000,1200,1400,1600,1800,2000])

# Create a list to collect the plt.patches data
totals = []

# Fill totals list
for i in ax.patches:
    totals.append(i.get_width())

total = sum(totals)

# Set bar labels using the list
for i in ax.patches:
    c = i.get_width()
    cnt = f'{c:,} '
    pct = str(round((c/total)*100, 2)) + '%'
    pct_f = "(" + pct + ")"
    ax.text(c+.3, i.get_y()+.4, cnt + pct_f, 
            fontsize=14, color='dimgrey')

# Invert graph 
ax.invert_yaxis()


# In[ ]:


from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Fill the list of words to show
term_values = ""
for term in findings['term']:
    term_val = str(term).title()
    term_val = term_val.replace(' ','_')
    term_val = term_val.replace('-','_')
    term_values += term_val + ' '

# Generates the wordcloud object
wordcloud = WordCloud(background_color="white",
                      collocations=False).generate(term_values)

# Display the generated image
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.figure(figsize=((10,8)))
plt.show()


# ## Get what the literature reports on the task topic

# Group findings by key term and sort by key term and count. The first document in each group will be part of the response to the task question.

# In[ ]:


findings_sta = findings.groupby(["term", "paper_id"]).size().reset_index(name="count")
findings_sta = findings_sta.sort_values(by=['term','count'], ascending=False)


# In[ ]:


# Helper

def get_doc_text(paper_id):
    doc = df.loc[df["paper_id"]==paper_id].iloc[0]
    return doc["doc_text"]


# In[ ]:


answers = []

for term in terms:    
    term = term.capitalize()
    try:
        f = findings_sta[findings_sta["term"]==term]
        f = f.sort_values("count",ascending=False)
        for fc in f.iterrows():           
            paper_id = fc[1]["paper_id"]                        
            doc_text = get_doc_text(paper_id)
            
            answer = {
                "aspect": terms_group_id,
                "factor": term,
                "paper_id": paper_id,
                "doc_text": str(doc_text)
            }

            answers.append(answer)
            
            break
        
    except BaseException as e:
        print(str(e))
        continue

len(answers)


# In[ ]:


import ipywidgets as widgets
from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, IntSlider


# In[ ]:


item_layout = Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between',
    width= '100%',
    height= '200px'
)


# In[ ]:


# Helpers

def get_text_area(text):
    ta = widgets.Textarea(
        value=str(text),
        placeholder='',
        description='',
        layout=item_layout,
        disabled=True
    )
    return ta

import json

def get_answer_text(factor):
    try:
        factor = factor.capitalize()
        ans = next(x for x in answers if x["factor"] == factor)
        ans = json.dumps(ans["doc_text"]).strip("'").strip('"')
        ans = ans.replace('\\n', '\n\n')
        return ans
    except BaseException:
        return ""
    
def get_question_answer(t_params):
    full_text = ''
    for t_param in t_params:
        try:
            doc_text = get_answer_text(t_param)
            if not doc_text in full_text:
                if len(full_text) > 0:
                    full_text += "\n\n"                
                full_text += doc_text
        except BaseException:
            continue
    
    return full_text


# In[ ]:


t1d = "Effectiveness of drugs being developed and tried to treat COVID-19 patients. Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication."
text = get_question_answer(terms1)
ta1 = get_text_area(text)

t2d = "Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients."
text = get_question_answer(terms2)
ta2 = get_text_area(text)

t3d = "Exploration of use of best animal models and their predictive value for a human vaccine."
text = get_question_answer(terms3)
ta3 = get_text_area(text)

t4d = "Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents."
text = get_question_answer(terms4)
ta4 = get_text_area(text)

t5d = "Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need."
text = get_question_answer(terms5)
ta5 = get_text_area(text)
    
t6d = "Efforts targeted at a universal coronavirus vaccine."
text = get_question_answer(terms6)
ta6 = get_text_area(text)

t7d = "Efforts to develop animal models and standardize challenge studies."
text = get_question_answer(terms7)
ta7 = get_text_area(text)

t8d = "Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers."
text = get_question_answer(terms8)
ta8 = get_text_area(text)

t9d = "Approaches to evaluate risk for enhanced disease after vaccination."
text = get_question_answer(terms9)
ta9 = get_text_area(text)
        
t10d = "Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]."
text = get_question_answer(terms10)
ta10 = get_text_area(text)


# In[ ]:


ac1_tas = [ta1,ta2,ta3,ta4,ta5,ta6,ta7,ta8,ta9,ta10]
ac1 = widgets.Accordion(children=ac1_tas)
ac1.set_title(0, t1d)
ac1.set_title(1, t2d)
ac1.set_title(2, t3d)
ac1.set_title(3, t4d)
ac1.set_title(4, t5d)
ac1.set_title(5, t6d)
ac1.set_title(6, t7d)
ac1.set_title(7, t8d)
ac1.set_title(8, t9d)
ac1.set_title(9, t10d)


#  # What is known about vaccines and therapeutics?
#  
# The answer to this question is distributed in each of the aspects listed at the beginning of this notebook.

# In[ ]:


ac1

