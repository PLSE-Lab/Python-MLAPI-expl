#!/usr/bin/env python
# coding: utf-8

# # **INTRODUCTION**
# 
# This work is to help the medical community answer the posted question in Kaggle: [What has been published about ethical and social science considerations?](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks) The resulting model enables to understand and keep up with the large amount of literature contained in the provided dataset, specifically:
# 
# 1. Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019
# 2. Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight
# 3. Efforts to support sustained education, access, and capacity building in the area of ethics
# 4. Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.
# 5. Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)
# 6. Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.
# 7. Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media.
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

terms_group_id = "ethical"

terms1 = [
    'efforts to articulate existing ethical principles',
    'efforts to translate existing ethical principles',
    'efforts to articulate standards',
    'efforts to translate standards',
    'to salient issues','ethical principles','standards',
    'salient issues']

terms2 = ['efforts to embed ethics across all thematic areas',
    'embed ethics across all thematic areas',
    'embed ethics','thematic areas','ethics','thematic',
    'engage with novel ethical issues that arise',
    'novel ethical issues',
    'coordinate to minimize duplication of oversight',
    'minimize duplication','duplication of oversight',
    'oversight']
          
terms3 = [    
    'efforts to support sustained education','sustained education',
    'efforts to support sustained access','sustained access',
    'efforts to support sustained capacity','sustained capacity',
    'capacity building']

terms4 = ['efforts to establish a team at who','team at who',
    'integrated within multidisciplinary research',
    'multidisciplinary research','operational platforms',
    'connect with existing global networks','global networks',
    'connect with expanded global networks']
    
terms5 = ['efforts to develop qualitative assessment frameworks',
    'qualitative assessment frameworks','assessment frameworks',
    'systematically collect information','collect information',
    'local barriers','adherence to public health measures',
    'public health measures','prevention and control',
    'secondary impacts','health seeking behaviors','srh']
    
terms6 = ['efforts to identify how the burden of responding to the outbreak',
    'responding to the outbreak','measures affects',
    'measures affects the physical health','physical health',
    'measures affects the psychological health',
    'psychological health','identify the immediate needs',
    'immediate needs']
    
terms7 = ['efforts to identify the underlying drivers of fear',
    'drivers of fear','fear',
    'efforts to identify the underlying drivers of anxiety',
    'drivers of anxiety','anxiety',
    'efforts to identify the underlying drivers of stigma',
    'drivers of stigma','stigma',
    'fuel misinformation','rumor','social media']

terms = terms1 + terms2 + terms3 +         terms4 + terms5 + terms6 + terms7


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

ax = cnt_s.plot(kind='barh', figsize=(12,20), 
                legend=False, color="coral", 
                fontsize=16)
ax.set_alpha(0.8)
ax.set_title("What has been published about ethical and social science considerations?",
             fontsize=18)
ax.set_xlabel("Term Appearances", fontsize=16);
ax.set_ylabel("Terms", fontsize=14);
ax.set_xticks([0,200,400,600,800,1000,1200,1400])

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
            doc_text = get_answer_text(t_param.capitalize())
            if not doc_text in full_text:
                if len(full_text) > 0:
                    full_text += "\n\n"                
                full_text += doc_text
        except BaseException:
            continue
    
    return full_text


# In[ ]:


td1 = 'Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019'
text = get_question_answer(terms1)
ta1 = get_text_area(text)

td2 = 'Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight'
text = get_question_answer(terms2)
ta2 = get_text_area(text)

td3 = 'Efforts to support sustained education, access, and capacity building in the area of ethics'
text = get_question_answer(terms3)
ta3 = get_text_area(text)

td4 = 'Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.'
text = get_question_answer(terms4)
ta4 = get_text_area(text)

td5 = 'Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)'
text = get_question_answer(terms5)
ta5 = get_text_area(text)

td6 = 'Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.'
text = get_question_answer(terms6)
ta6 = get_text_area(text)

td7 = 'Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media.'
text = get_question_answer(terms7)
ta7 = get_text_area(text)


# In[ ]:


ac1_tas = [ta1,ta2,ta3,ta4,ta5,ta6,ta7]
ac1 = widgets.Accordion(children=ac1_tas)
ac1.set_title(0, td1)
ac1.set_title(1, td2)
ac1.set_title(2, td3)
ac1.set_title(3, td4)
ac1.set_title(4, td5)
ac1.set_title(5, td6)
ac1.set_title(6, td7)


#  # What has been published about ethical and social science considerations?

# In[ ]:


ac1

