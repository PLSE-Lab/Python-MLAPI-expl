#!/usr/bin/env python
# coding: utf-8

# # **INTRODUCTION**
# 
# This work is to help the medical community answer the posted question in Kaggle: [What do we know about virus genetics, origin, and evolution?](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks) The resulting model enables to understand and keep up with the large amount of literature contained in the provided dataset, specifically:
# 
# 1. Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.
# 2. Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.
# 3. Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.
# 4. Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.
# 5. Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.
# 6. Experimental infections to test host range for this pathogen.
# 7. Animal host(s) and any evidence of continued spill-over to humans
# 8. Socioeconomic and behavioral risk factors for this spill-over
# 9. Sustainable risk reduction strategies
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

terms_group_id = "genetics"

terms1 = [
    "Real-time tracking of whole genomes",
    "Real-time tracking",
    "Whole genomes",
    "Mechanism for coordinating",
    "Dissemination of information",
    "Mechanism for coordinating",
    "Inform of diagnostics",
    "Therapeutics",
    "Track variations of the virus over time",
    "Track variations over time",
    "Track variations",
    "Track variations"
]

terms2 = [
    "Sample sets",
    "Access to geographic sample sets",
    "Geographic sample sets",
    "Geographic distribution",
    "Genomic differences",
    "Strain in circulation",
    "Multi-lateral agreements",
    "Nagoya protocol"
]

terms3 = [
    "Livestock could be infected",
    "Livestock infected",
    "Livestock",
    "Field surveillance",
    "Genetic sequencing",
    "Receptor binding",
    "Reservoir",
    "Farmers infected",
    "Farmers origin",
    "Farmers",
    "Farmers",
    "Surveillance of mixed wildlife",
    "Experimental infections"
]

terms4 = [
    "Animal host(s)",
    "Spill-over to humans",
    "Spill-over"
]

terms5 = ["Spillover"]

terms6 = [
    "Sustainable",
    "Management measures"
]

terms = terms1 + terms2 + terms3 + terms4 + terms5 + terms6


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

ax = cnt_s.plot(kind='barh', figsize=(12,14), 
                legend=False, color="coral", 
                fontsize=16)
ax.set_alpha(0.8)
ax.set_title("What do we know about virus genetics, origin, and evolution?",fontsize=18)
ax.set_xlabel("Term Appearances", fontsize=16);
ax.set_ylabel("Terms", fontsize=14);
ax.set_xticks([0,100,200,300,400,500,600,700,800,900,1000])

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


td1 = "Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time."
text = get_question_answer(terms1)
ta1 = get_text_area(text)

td2 = "Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged."
text = get_question_answer(terms2)
ta2 = get_text_area(text)

td3 = "Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over. Evidence of whether farmers are infected, and whether farmers could have played a role in the origin. Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia. Experimental infections to test host range for this pathogen."
text = get_question_answer(terms3)
ta3 = get_text_area(text)

td4 = "Animal host(s) and any evidence of continued spill-over to humans"
text = get_question_answer(terms4)
ta4 = get_text_area(text)

td5 = "Socioeconomic and behavioral risk factors for this spill-over"
text = get_question_answer(terms5)
ta5 = get_text_area(text)

td6 = "Sustainable risk reduction strategies"
text = get_question_answer(terms6)
ta6 = get_text_area(text)


# In[ ]:


ac1_tas = [ta1,ta2,ta3,ta4,ta5,ta6]
ac1 = widgets.Accordion(children=ac1_tas)
ac1.set_title(0, td1)
ac1.set_title(1, td2)
ac1.set_title(2, td3)
ac1.set_title(3, td4)
ac1.set_title(4, td5)
ac1.set_title(5, td6)


#  # What is known about virus genetics, origin, and evolution?
#              
#  
# The answer to this question is distributed in each of the aspects listed at the beginning of this notebook.

# In[ ]:


ac1

