#!/usr/bin/env python
# coding: utf-8

# ## **Relevant symptoms**
# 
# This notebook shows how to plot a word cloud with the most relevant symptoms showed in the database

# ### Imports

# In[ ]:


import os
import re
import json
from tqdm import tqdm
import pandas as pd
from collections import Counter
from stop_words import get_stop_words
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# In[ ]:


cord_path = '../input/CORD-19-research-challenge'
dirs = ["biorxiv_medrxiv", "comm_use_subset", "noncomm_use_subset", "custom_license"]


# ### Importing the full dataset into a pandas df
# 
# This portion of code iterates over the directories that contain the data. Over each directory it iterates over each file, loads the data and put the title, abstract, authors and full text into a pandas dataframe

# In[ ]:


docs = []
for d in dirs:
    for file in tqdm(os.listdir(f"{cord_path}/{d}/{d}")):
        file_path = f"{cord_path}/{d}/{d}/{file}"
        j = json.load(open(file_path, "rb"))

        title = j["metadata"]["title"]
        authors = j["metadata"]["authors"]

        try:
            abstract = j["abstract"][0]["text"].lower()
        except:
            abstract = ""

        full_text = ""
        for text in j["body_text"]:
            full_text += text["text"].lower() + "\n\n"
        docs.append([title, authors, abstract, full_text])

df = pd.DataFrame(docs, columns=["title", "authors", "abstract", "full_text"])


# Now we filter the dataframe to only the rows that contains the word "symptom" in the full_text column

# In[ ]:


symptoms_df = df[df["full_text"].str.contains("symptom")]


# Symptoms list. Taken from https://www.kaggle.com/cstefanache/nlp-text-mining-disease-behavior[](http://)

# In[ ]:


symptoms = [
    "weight loss","chills","shivering","convulsions","deformity","discharge","dizziness",
    "vertigo","fatigue","malaise","asthenia","hypothermia","jaundice","muscle weakness",
    "pyrexia","sweats","swelling","swollen","painful lymph node","weight gain","arrhythmia",
    "bradycardia","chest pain","claudication","palpitations","tachycardia","dry mouth","epistaxis",
    "halitosis","hearing loss","nasal discharge","otalgia","otorrhea","sore throat","toothache","tinnitus",
    "trismus","abdominal pain","fever","bloating","belching","bleeding","blood in stool","melena","hematochezia",
    "constipation","diarrhea","dysphagia","dyspepsia","fecal incontinence","flatulence","heartburn",
    "nausea","odynophagia","proctalgia fugax","pyrosis","steatorrhea","vomiting","alopecia","hirsutism",
    "hypertrichosis","abrasion","anasarca","bleeding into the skin","petechia","purpura","ecchymosis and bruising",
    "blister","edema","itching","laceration","rash","urticaria","abnormal posturing","acalculia","agnosia","alexia",
    "amnesia","anomia","anosognosia","aphasia and apraxia","apraxia","ataxia","cataplexy","confusion","dysarthria",
    "dysdiadochokinesia","dysgraphia","hallucination","headache","akinesia","bradykinesia","akathisia","athetosis",
    "ballismus","blepharospasm","chorea","dystonia","fasciculation","muscle cramps","myoclonus","opsoclonus","tic",
    "tremor","flapping tremor","insomnia","loss of consciousness","syncope","neck stiffness","opisthotonus",
    "paralysis and paresis","paresthesia","prosopagnosia","somnolence","abnormal vaginal bleeding",
    "vaginal bleeding in early pregnancy", "miscarriage","vaginal bleeding in late pregnancy","amenorrhea",
    "infertility","painful intercourse","pelvic pain","vaginal discharge","amaurosis fugax","amaurosis",
    "blurred vision","double vision","exophthalmos","mydriasis","miosis","nystagmus","amusia","anhedonia",
    "anxiety","apathy","confabulation","depression","delusion","euphoria","homicidal ideation","irritability",
    "mania","paranoid ideation","suicidal ideation","apnea","hypopnea","cough","dyspnea","bradypnea","tachypnea",
    "orthopnea","platypnea","trepopnea","hemoptysis","pleuritic chest pain","sputum production","arthralgia",
    "back pain","sciatica","Urologic","dysuria","hematospermia","hematuria","impotence","polyuria",
    "retrograde ejaculation","strangury","urethral discharge","urinary frequency","urinary incontinence","urinary retention"]


# Now we take the column full_text and iterate over each text. On each text we separate it by sentences and over each sentence we separate it by words only if the sentence contains the word symptom. Then we save all these words in a list.

# In[ ]:


texts = df.full_text.values

all_words = []
for text in texts:
    sentences = re.split('[. ] |\n',text)
    for sentence in sentences:
        sentence = sentence.replace(',', '')
        if ("symptom" in sentence):
            words = sentence.split()
            words = [word for word in words if word  in symptoms]
            all_words.append(words)
            
all_words = [item for sublist in all_words for item in sublist]


# Now we create a dictionary with the word frequencies

# In[ ]:


word_dict = Counter(all_words)


# Finally we plot the wordcloud

# In[ ]:


wc = WordCloud(background_color="black",width=1000, height=800).generate_from_frequencies(word_dict)
fig = plt.figure(figsize=(15,15))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[ ]:




