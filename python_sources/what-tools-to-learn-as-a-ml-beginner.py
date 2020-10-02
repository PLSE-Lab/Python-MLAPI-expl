#!/usr/bin/env python
# coding: utf-8

# **What tools to learn as a ML begginner**
# 
# Machine Learning has been a big thing lately, with numerous frameworks and libraries being used all over the world. For someone who is interested in learning about ML or started studying ML, it can get quite confusing what is actually used and is considered a useful skill to know.
# 
# Here, we'll try to see the most used tools on this field. Since Kaggle is a well known website among people who work with and love Machine Learning, this 2018 survey is a good way to see whats still relevant to know about. This way, someone who wants to start learning and is a little overwhelmed by all these names and programs can get an idea of whats currently dominating the industry and who's using it. 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.tokenize import ToktokTokenizer 
from wordcloud import WordCloud
import squarify
get_ipython().run_line_magic('matplotlib', 'inline')

schema = pd.read_csv("../input/SurveySchema.csv", low_memory=False)
ff = pd.read_csv("../input/freeFormResponses.csv",low_memory=False)
mc = pd.read_csv("../input/multipleChoiceResponses.csv",low_memory=False)


# Let's first get an overview of the participants.

# In[ ]:


gender_question = mc.iloc[1:, 1]
gender_answers = gender_question.value_counts()
plt.style.use("seaborn-white")
plt.figure(figsize = (12, 6))
sns.barplot(gender_answers.index, gender_answers.values, 
           color = "#009688", ec = "k", alpha = .6)
plt.title("Gender of Kaggle's 2018 Survey Participants")
plt.box(False)
plt.yticks([])

for i, v in enumerate(gender_answers.values):
    plt.text(i - .1, v + 300, str(v), fontdict={"fontsize": 11})


# In[ ]:


jobs = []
for i in mc.iloc[1:, 7].values:
    if i == i:
        jobs.append(i)
        
jobs = Counter(jobs)
jobs.most_common()
jobs_count = []
possible_jobs = []
for i in jobs.most_common():
    jobs_count.append(i[1])
    possible_jobs.append(i[0])
    
plt.figure(figsize = (12, 12))
sns.barplot(jobs_count, possible_jobs, color = "#009688", ec = "k", alpha = .6)
plt.title("Current Role Situation of Respondents")
plt.box(False)
plt.xticks([])

for i, v in enumerate(jobs_count):
    plt.text(v + 30, i + .15, str(v))


# ![](https://i.imgur.com/IwfgrVF.png)

# We can see the dominance of males and students here. Also, the amount of people that work on different job positions, some even not as "tech-related" as others.
# On the map, we can also see that the US and India are two of the powerhorses currently (but let's not forget about China). Since I'm brazilian, I highlighted my homeland to show that we are not to be forgotten and have quite a lot of people working or studying to get into this field.
# 
# But okay, we're here to know more about the most used Machine Learning tools.
# 
# Let's start with the programming language. The most used language is by far Python (which is the one I'm using here aswell). Python is a great language to learn, specially for someone who is just getting into programming, not to mention the amount of libraries that focus on Machine Learning. SQL is a great skill to have, because if you are interested in ML, you are interested in data. Your SQL skills will allow you to work with and store this data. In third place, R is one of the most used language for statistical computing and analisys. We can also see that a lot of people use C/C++/Java

# In[ ]:


languages_array = []
for i in range(len(mc) - 1):
    for language in mc.iloc[i + 1, 65:83].values:
        if language == language and language != "None":
            languages_array.append(language)
            
language_count = []
languages = []
for i in Counter(languages_array).most_common():
    languages.append(i[0])
    language_count.append(i[1])
    
plt.figure(figsize = (12, 24))
plt.subplot(2, 1, 1)
sns.barplot(language_count, languages, ec = "k", alpha = .6, color = "#009688")
plt.title("Most Used Programming Languages Among Respondents")
plt.box(False)
plt.xticks([])

for i, v in enumerate(language_count):
    plt.text(v + 100, i + .15, str(v))
    
plt.subplot(2, 1, 2) 
squarify.plot(sizes = language_count, label = languages, alpha = .8, ec = "k", value = language_count, color = ["#009688","#3F51B5","#66BB6A", "#90CAF9", "#616161", "#E0E0E0"])
plt.axis("off")


# Okay, so what about something more "machine learning specific"? Let's see the most used ML frameworks among the participants.

# In[ ]:


frameworks = []
for i in range(len(mc) - 1):
    for j in mc.iloc[i + 1, 88:107].values:
        if j == j and j != "None" and j != "-1":
            frameworks.append(j)
            
frameworks = Counter(frameworks)
frameworks = frameworks.most_common()

frameworks_count = []
possible_frameworks = []

for i in frameworks:
    frameworks_count.append(i[1])
    possible_frameworks.append(i[0])
    
plt.figure(figsize = (12, 24))
plt.subplot(2, 1, 1)
sns.barplot(frameworks_count, possible_frameworks, color = "#009688", ec = "k", alpha = .6)
plt.title("Used Machine Learning Frameworks (Kaggle's 2018 Survey)")
plt.box(False)
plt.xticks([])

for i, v in enumerate(frameworks_count):
    plt.text(v + 100, i + .15, str(v))
    
plt.subplot(2, 1, 2) 
squarify.plot(sizes = frameworks_count, label = possible_frameworks, alpha = .8, ec = "k", value = frameworks_count, color = ["#009688","#3F51B5","#66BB6A", "#90CAF9", "#616161", "#E0E0E0"])
plt.axis("off")


# Remember when I said Python is great for its libraries? Yeah. Here, the top 3 aframeworks are Python based (even though TensorFlow is offered in other languages). Scikit-Learn offers so much in terms of Machine Learning that its #1 spot is not a surprise. When talking Neural Networks, Deep Learning or AI, TensorFlow and Keras show up.
# 
# If you would like to know some more terms to search for and study, let's take a look at some other things some respondants made sure to mention.
# 

# In[ ]:


frameworks_other_answers = []
for i in ff.iloc[1:, 13].values:
    if i == i:
        frameworks_other_answers.append(i)
        
frameworks_other_answers
tok = ToktokTokenizer()
tokens = []
for i in frameworks_other_answers:
    tokens_per_answer = tok.tokenize(i)
    for j in tokens_per_answer:
        tokens.append(j)

wc = WordCloud(width=1280, height=720).generate(' '.join(tokens))
plt.figure(figsize = (25, 14))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of text answers from Q13 (Other Machine Learning Frameworks)", fontsize = 24)
plt.box(False)


# Now, if you're into Machine Learning (and reading this post) you'd probably like to know some data visualization tools. Let's see what Kaggle users most use.
# 

# In[ ]:


dv_libs = []
for i in range(len(mc) - 1):
    for j in mc.iloc[i + 1, 110:123].values:
        if j == j and j != "None" and j != "-1":
            dv_libs.append(j)
            
dv_libs = Counter(dv_libs)
dv_libs = dv_libs.most_common()

dv_libs_count = []
possible_dv_libs = []

for i in dv_libs:
    dv_libs_count.append(i[1])
    possible_dv_libs.append(i[0])

plt.figure(figsize = (12, 24))
plt.subplot(2, 1, 1) 
sns.barplot(dv_libs_count, possible_dv_libs, color = "#009688", ec = "k", alpha = .6)
plt.title("Used Data Visualization Libraries (Kaggle's 2018 Survey)", fontsize = 16)
plt.box(False)
plt.xticks([])

for i, v in enumerate(dv_libs_count):
    plt.text(v + 100, i+.15, str(v))

plt.subplot(2, 1, 2) 
squarify.plot(sizes = dv_libs_count, label = possible_dv_libs, alpha = .8, ec = "k", value = dv_libs_count, color = ["#009688","#3F51B5","#66BB6A", "#90CAF9", "#616161", "#E0E0E0"])
plt.axis("off")
plt.box(False)


# Matplotlib and Seaborn are both Python libraries for data visualization. Considering that more than 25% of the participants use Python (although not exclusively), the top position was expected. Ggplot2 is also a library, but for the R programming language, and it is clearly used a lot aswell, right in there with Seaborn.
# 
# So now we know the most used ML frameworks and data visualization libraries by Kaggle users. If you're getting into ML, specially if you're still a student, I think you'd like to know what types of jobs can take advantage of mastering those skills. 

# In[ ]:


df_framework = mc.iloc[1:, 108]
df_dvlib = mc.iloc[1:, 124]
df_jobs = mc.iloc[1:, 7]
df_gender = mc.iloc[1:, 1]
df = pd.concat([df_gender, df_dvlib, df_jobs, df_framework], axis = 1, )
df.columns = ["Gender", "Visualization Library","Current Role", "Machine Learning Framework"]

def ignore_less_voted_frameworks(x):
    if x != "Scikit-Learn" and x != "TensorFlow" and x != "Keras":
        return None
    else:
        return x
df["Machine Learning Framework"] = df["Machine Learning Framework"].apply(ignore_less_voted_frameworks)

plt.figure(figsize = (12, 6))
sns.countplot("Current Role", data = df, hue = "Machine Learning Framework", ec = "k", palette = "Paired")
plt.xticks(rotation = 60)
plt.title("Current Job Title by Top 3 Machine Learning Framework")
plt.legend(loc=1)
plt.box(False)


# It's pretty clear that Scikit-Learn is really a big thing among Data Scientists. It's pretty popular among students aswell, a sign that Scikit-Learn will continue to be an important skill to have in the future.

# In[ ]:


def ignore_less_voted_libs(x):
    if x != "Matplotlib" and x != "Seaborn" and x != "ggplot2":
        return None
    else:
        return x
    
df["Visualization Library"] = df["Visualization Library"].apply(ignore_less_voted_libs)

plt.figure(figsize = (12, 6))
sns.countplot("Current Role", data = df, hue = "Visualization Library", ec = "k", palette = "Paired")
plt.xticks(rotation = 60)
plt.title("Current Job Title by Top 3 Visualization Libraries")
plt.legend(loc=1)
plt.box(False)


# Regarding data visualization, Matplotlib dominates in most job titles. Statisticians, however, tend to use ggplot2 more. Ggplot2 is an R library, which is, as said before, a statistical language. Ggplot2 is also pretty popular with Data Analysts.
# 
# What about the work environment? IDE's (Integrated Development Environment)? What do Kaggle ML people like to code in?

# In[ ]:


ides_jobs = mc.iloc[1:, 29:44].join(mc.iloc[:, 7])
ides_jobs.head(10)

ds_ides_df = ides_jobs[ides_jobs["Q6"] == "Data Scientist"].iloc[:, :-1]
ds_ides = []
for i in range(len(ds_ides_df) - 1):
    for ide in ds_ides_df.iloc[i + 1, :].values:
        if ide == ide:
            ds_ides.append(ide)
            
ds_ides_counter = Counter(ds_ides).most_common()
ds_ides = []
ds_ides_count = []

for i in ds_ides_counter:
    ds_ides.append(i[0])
    ds_ides_count.append(i[1])
    
se_ides_df = ides_jobs[ides_jobs["Q6"] == "Software Engineer"].iloc[:, :-1]
se_ides = []
for i in range(len(se_ides_df) - 1):
    for ide in se_ides_df.iloc[i + 1, :].values:
        if ide == ide:
            se_ides.append(ide)
            
se_ides_counter = Counter(se_ides).most_common()
se_ides = []
se_ides_count = []

for i in se_ides_counter:
    se_ides.append(i[0])
    se_ides_count.append(i[1])
    
da_ides_df = ides_jobs[ides_jobs["Q6"] == "Data Analyst"].iloc[:, :-1]
da_ides = []
for i in range(len(da_ides_df) - 1):
    for ide in da_ides_df.iloc[i + 1, :].values:
        if ide == ide:
            da_ides.append(ide)
            
da_ides_counter = Counter(da_ides).most_common()
da_ides = []
da_ides_count = []

for i in da_ides_counter:
    da_ides.append(i[0])
    da_ides_count.append(i[1])
    
ds_df = pd.DataFrame(data = {"IDE": ds_ides, "Data Scientist Count": ds_ides_count})
se_df = pd.DataFrame(data = {"IDE": se_ides, "Software Engineer Count": se_ides_count})
da_df = pd.DataFrame(data = {"IDE": da_ides, "Data Analyst Count": da_ides_count})

ds_se_df = pd.merge(ds_df, se_df, on = "IDE")
df = pd.merge(ds_se_df, da_df, on = "IDE")

index = np.arange(len(df["IDE"]))
bar_width = .25
plt.figure(figsize = (12, 6))
sns.set_palette("Paired")
plt.bar(index - bar_width, df["Data Scientist Count"], label = "Data Scientist", width = bar_width, ec = "k")
plt.bar(index, df["Software Engineer Count"], label = "Software Engineer", width = bar_width, ec = "k")
plt.bar(index + bar_width, df["Data Analyst Count"], label = "Data Analyst", width = bar_width, ec = "k")
plt.title("Most Used IDE's by Top 3 Job Titles")
plt.ylabel("Count")
plt.xticks(index, (df["IDE"]), rotation = 45)
plt.legend()
plt.box(False)


# Among the top 3 job titles, we can see that the Jupyter/IPython notebook environment is the most used one. The R language makes another important appearance with RStudio being really used by Data Scientists and Analysts. Software Engineers tend to like Visual Studio and IntelliJ a lot, but even with them Jupyter is huge. 

# In[ ]:


ides_frameworks = mc.iloc[1:, 29:44].join(mc.iloc[:, 88:92])
sklearn_ides_df = ides_frameworks[ides_frameworks["Q19_Part_1"] == "Scikit-Learn"].iloc[:, :-4]
sklearn_ides = []
for i in range(len(sklearn_ides_df) - 1):
    for ide in sklearn_ides_df.iloc[i + 1, :].values:
        if ide == ide:
            sklearn_ides.append(ide)
            
sklearn_ides_counter = Counter(sklearn_ides).most_common()
sklearn_ides = []
sklearn_ides_count = []

for i in sklearn_ides_counter:
    sklearn_ides.append(i[0])
    sklearn_ides_count.append(i[1])
    
tf_ides_df = ides_frameworks[ides_frameworks["Q19_Part_2"] == "TensorFlow"].iloc[:, :-4]
tf_ides = []
for i in range(len(tf_ides_df) - 1):
    for ide in tf_ides_df.iloc[i + 1, :].values:
        if ide == ide:
            tf_ides.append(ide)
            
tf_counter = Counter(tf_ides).most_common()
tf_ides = []
tf_ides_count = []

for i in tf_counter:
    tf_ides.append(i[0])
    tf_ides_count.append(i[1])
    
keras_ides_df = ides_frameworks[ides_frameworks["Q19_Part_3"] == "Keras"].iloc[:, :-4]
keras_ides = []
for i in range(len(keras_ides_df) - 1):
    for ide in keras_ides_df.iloc[i + 1, :].values:
        if ide == ide:
            keras_ides.append(ide)
            
keras_counter = Counter(keras_ides).most_common()
keras_ides = []
keras_ides_count = []

for i in keras_counter:
    keras_ides.append(i[0])
    keras_ides_count.append(i[1])
    
skl_df = pd.DataFrame(data = {"IDE": sklearn_ides, "Scikit-Learn Count": sklearn_ides_count})
tf_df = pd.DataFrame(data = {"IDE": tf_ides, "TensorFlow Count": tf_ides_count})
keras_df = pd.DataFrame(data = {"IDE": keras_ides, "Keras Count": keras_ides_count})
skl_tf_df = pd.merge(skl_df, tf_df, on = "IDE")
df = pd.merge(skl_tf_df, keras_df, on = "IDE")

index = np.arange(len(df["IDE"]))
bar_width = .25
plt.figure(figsize = (12, 6))
sns.set_palette("Paired")
plt.bar(index - bar_width, df["Scikit-Learn Count"], label = "Scikit-Learn", width = bar_width, ec = "k")
plt.bar(index, df["TensorFlow Count"], label = "TensorFlow", width = bar_width, ec = "k")
plt.bar(index + bar_width, df["Keras Count"], label = "Keras", width = bar_width, ec = "k")
plt.title("Most Used IDE's by Top 3 ML Frameworks")
plt.ylabel("Count")
plt.xticks(index, (df["IDE"]), rotation = 45)
plt.legend()
plt.box(False)


# Inside the Machine Learning Kaggle community, clearly the Jupyter environment takes the lead.
# 
# My goal with this kernel was to show Machine Learning enthusiasts what tools are important for someone in this field to learn. It's a lot of information out there and it can get quite stressful trying to know what to focus your time and energy studying.
# 
# Hopefully you can now see the top frameworks and libraries used by  Kaggle users and direct your studies towards whats the current trend among the Machine Learning community.
# 
# I hope you liked it!
