#!/usr/bin/env python
# coding: utf-8

# # Analyzing the Democratic Debates of 2020
# 
# The following exercise involves some exploratory data analysis, and topic modeling in order to identify congressmen/congresswomen-level statistics, and the topics that they tend to speak about.

# In[ ]:


# Importing the essential libraries for the exercise.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Linking the directory to access the dataset.
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# Reading the data, and exploring its characteristics.
path = "../input/democratic-debate-transcripts-2020/debate_transcripts_v2_2020-02-23.csv"
data = pd.read_csv(path, encoding = "ISO-8859-1")
data[0:5]


# ## Exploratory Data Analysis

# In[ ]:


# Dictionary of the debates.
debate_names = data["debate_name"]
number_of_debates = len(set(debate_names))
print("Total number of democratic debates:", number_of_debates, "debates")

# Dictionary of number of sections.
debate_section = data["debate_section"]
number_of_sections = len(set(debate_section))
print("Maximum number of sections in the debates:", number_of_sections, "sections")

# Dictionary of name of speakers.
dem_speakers = data["speaker"]
number_of_speakers = len(set(dem_speakers))
print("Total number of democratic speakers:",number_of_speakers, "speakers")

# Mean duration of speech.
print("The average speaking time is:",np.mean(data["speaking_time_seconds"]), "seconds")


# In[ ]:


# Sorted dictionary of debates.
debs = dict()
for i in debate_names:
    debs[i] = debs.get(i, 0) + 1

debates = {k: v for k, v in sorted(debs.items(), key=lambda item: item[1])}
 
# Sorted dictionary of sections.
secs = dict()
for i in debate_section:
    secs[i] = secs.get(i, 0) + 1

sections = {k: v for k, v in sorted(secs.items(), key=lambda item: item[1])}

# Sorted dictionary of speakers.
spkrs = dict()
for i in dem_speakers:
    spkrs[i] = spkrs.get(i, 0) + 1

speakers = {k: v for k, v in sorted(spkrs.items(), key=lambda item: item[1])}


# In[ ]:


# Plot of debates
import matplotlib.pyplot as plt

plt.bar(list(debates.keys()), debates.values(), color='red')
plt.title("Histogram of Debates")
plt.xticks(rotation = 90)
plt.rcParams["figure.figsize"] = (10,10)


# In[ ]:


# Plot of speakers' activity
plt.bar(list(speakers.keys()), speakers.values(), color='green')
plt.title("Histogram of Speakers' Activity")
plt.xticks(rotation=90)
plt.rcParams["figure.figsize"] = (20,20)


# ## Topic Modeling with Non-negative Matrix Factorization

# In[ ]:


import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
random_state = 0

# Taking into consideration only nouns so as to identify the topics.
def only_nouns(texts):
    output = []
    for doc in nlp.pipe(texts):
        noun_text = " ".join(token.lemma_ for token in doc if token.pos_ == 'NOUN')
        output.append(noun_text)
    return output


# In[ ]:


# Merging the nouns-only list to the dataset.
data_new = only_nouns(data["speech"])
speech_nouns = pd.DataFrame(data_new)
data["Index"] = data.index
speech_nouns["Index"] = speech_nouns.index
democrat_data = pd.merge(data, speech_nouns, on="Index")
democrat_data.columns = ["debate_name", "debate_section", "speaker", "speech", "speaking_time_seconds", "index", "speech_nouns"]
democrat_data = democrat_data.drop(["index"], axis=1)
democrat_data.head()


# In[ ]:


# Number of topics to extract
n_topics = 10

# Vectorization of the nouns.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vec = TfidfVectorizer(max_features=5000, stop_words="english", max_df=0.95, min_df=2)
features = vec.fit_transform(democrat_data.speech_nouns)

# Non-negative matrix factorization.
from sklearn.decomposition import NMF
cls = NMF(n_components=n_topics, random_state=random_state)
cls.fit(features)


# In[ ]:


# List of unique words
feature_names = vec.get_feature_names()

# Number of top words per topic
n_top_words = 20

for i, topic_vec in enumerate(cls.components_):
    print(i, end=' ')
    for fid in topic_vec.argsort()[-1:-n_top_words-1:-1]:
        print(feature_names[fid], end=' ')
    print()


# ## Highlights of the Democratic debates 2020:
# 
# 1. Joe Biden and Elizabeth Warren gave more speeches that every other congressmen/congresswomen
# 2. All the candidates spoke for an average duration of 17 seconds.
# 3. The most common topics of discussion were: 
#                          (1) Wealth and tax returns of the candidates
#                          (2) Healthcare provisions for the community
#                          (3) Follow-up, opening, closing, and rebuttals 
#                          (4) Changes in policies, and the associated dilemmas
#                          (5) Judiciary, and law
#                          (6) Deportation issues, and the economic impacts
#                          (7) Impeachment of the President
#                          (8) Climate change
#                          (9) Various classes in the society (higher, lower, and middle) and respective issues
#                          (10) Gun laws, and racism
#                          
# The analysis of the democratic debates show that the debates generally center around the wealth status of the candidates and the respective follow-ups/rebuttals/rivalries. The second most prioritized topic is "Healthcare", which is a prime concern for people of all demographics. The ethical dilemmas within the conducts of the American society are less discussed, due to possible lobbying efforts and differing points of view (when the party members know certain topics are too controversial, they tend to not mention it voluntarily in their speeches, but it seems they are asked questions regarding those topics by other members/audience).

# ## Sentiment clarification with Word2Vec Embedding

# In[ ]:


import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

democrat_speeches = list()
lines = data["speech"].tolist()

for line in lines:
    tokens = word_tokenize(line)
    # lowercase
    tokens = [word.lower() for word in tokens]
    # remove punctuation
    table = str.maketrans("","", string.punctuation)
    strip = [w.translate(table) for w in tokens]
    # remove remaining non-alphabets
    words = [word for word in strip if word.isalpha()]
    # filter stop-words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    democrat_speeches.append(words)


# In[ ]:


import gensim

# Train word2vec model
model = gensim.models.Word2Vec(sentences = democrat_speeches, size = 100, window = 5, workers = 4, min_count = 1)
# Vocab size
words = list(model.wv.vocab)
print("Vocabulary size: ", len(words))


# In[ ]:


print("Talked issue #1: Tax")
model.wv.most_similar("tax")


# In[ ]:


print("Talked issue #2: Healthcare")
model.wv.most_similar("healthcare")


# In[ ]:


print("Talked issue #3: Rebuttals")
model.wv.most_similar("rebuttal")


# In[ ]:


print("Talked issue #4: Policy")
model.wv.most_similar("policy")


# In[ ]:


print("Talked issue #5: Law")
model.wv.most_similar("law")


# In[ ]:


print("Talked issue #6: Immigration")
model.wv.most_similar("immigration")


# In[ ]:


print("Talked issue #7: Impeachment")
model.wv.most_similar("impeachment")


# In[ ]:


print("Talked issue #8: Climate")
model.wv.most_similar("climate")


# In[ ]:


print("Talked issue #9: Class")
model.wv.most_similar("class")


# In[ ]:


print("Talked issue #10: Racism")
model.wv.most_similar("racism")

