#!/usr/bin/env python
# coding: utf-8

# ## Summary
# 
# #### The purpose of this kernel is to tease out personas of respondents in the 2018 Kaggle ML & Data Science Survey. 
# 
# 
# For this purpose, I rely on the following definition of persona:
# ***
# **Persona** - In marketing, a persona is a fictitious character based on known features of the target audience for a product... (source: Macmillan Dictionary) 
# ***
# Basically what we want to do here is develop some common characteristics of respondents that group intuitively in order to better understand who the respondents are, where they are in their careers, and how much coding, analysis, and machine learning they do. 
# 
# One initial question I had was: are all those who identify as students really just students? Are they people with work experience as well? Are they undergraduate aspiring data scientists or ML research dissertators? 
# 
# A second question that was begged by the [aggregated results](https://www.kaggle.com/paultimothymooney/2018-kaggle-machine-learning-data-science-survey) was what, where, and how non-student respondents were doing data science?
# 
# **Let's see what we can discover!**
# 
# To develop these personas, let's:
# 
#     1. Preprocess the response data - deal with some noisy responses, custom encode questions that have obvious response ordinality (e.g., years of experience), binary encode features from questions containing multiple responses, and change some question/feature namings for convenience.
#     
#     2. Identify some questions we will use for clustering. These clusters will inform us of survey respondent groups (personas).
#     
#     3. Further identify certain questions we will use for inspecting marginal distributions across clusters/personas. 
#     
#     4. Segment survey respondents by self-identified students (Question 6 response "Student", or Question 7 response "I am a student") and non-students.
#     
#     5. Perform k-means clustering on each of the two segments and interpret intuitive personas for each cluster in each segment.
#     
#     6. Generate some heatmaps for each segment in order to interpret the personas that exist and give them some intuitive names.
#     
#     7. Summarize Student and Non-Student Personas from heatmaps and give them some names.
#     
#     8. Quantify characteristics from marginal distributions across personas in each segment to help us better identify with each Kaggle survey persona.

# ### 1. First, let's process the data, so it's easier for us to use for clustering and interpretation...

# In[ ]:


# Import some packages
import numpy as np 
import pandas as pd 
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score as sl
import matplotlib.pyplot as plt
import seaborn as sbn

# Read the survey data and for convenience drop the question row
kd_init = pd.read_csv("../input/multipleChoiceResponses.csv").drop(index=0, axis=0)

# Lets impute some values for NaNs and remove some non-ascii chars for some troublesome Pandas Series
kd_init["Q8"] = kd_init["Q8"].fillna("0-1").astype(str)
kd_init["Q23"] = kd_init["Q23"].fillna("1% to 25% of my time").astype(str)
kd_init["Q24"] = kd_init["Q24"].fillna("< 1 year")
kd_init["Q25"] = kd_init["Q25"].fillna("I have never studied machine learning but plan to learn in the future")
kd_init["Q4"] = kd_init["Q4"].fillna("Masters degree")
kd_init["Q4"] = kd_init["Q4"].astype(str).apply(lambda x: re.sub(r'[^\x00-\x7F]', '', x).replace("/", " "))

# Define the clustering dataframe and modfiy question feature names to be more compact
kd = pd.DataFrame()
feature_names = {"Q4": "Degree Level",
                 "Q5": "Degree Major",
                 "Q8": "Years Experience",
                 "Q11": "Work Activities",
                 "Q12": "Data Analysis Tools Used",
                 "Q16": "Programming Langs Used",
                 "Q17": "Languages Used",
                 "Q19": "ML Frameworks Used",
                 "Q22": "Viz Used",
                 "Q23": "Coding Activity",
                 "Q24": "Coding Experience",
                 "Q25": "ML Experience",
                 "Q30": "Big Data Products",
                 "Q34": "Data Science Project Time",
                 "Q35": "ML Training"}

# Add coarse deep learning & machine learning categories for more granular Questions 3 and 20
US = ["United States of America"]
EU = ["United Kingdom of Great Britain and Northern Ireland", "France", 
      "Germany", "Italy", "Spain", "Netherlands", "Poland", "Sweden", 
      "Norway", "Greece", "Portugal", "Switzerland", "Denmark", 
      "Belgium", "Ireland", "Finland", "Hungary"]
BRICCJ = ["India", "China", "Russia", "Brazil", "Canada", "Japan"]
kd_init["Q3"] = kd_init["Q3"].apply(lambda x: 
                                    "US" if x in US else (
                                        "Europe" if x in EU else (
                                            x if x in BRICCJ else "Other")))

TF = ["TensorFlow"]
Other_DL = ["Keras", "PyTorch", "H2O", "Fastai", "Mxnet", "Caret", "CNTK", "Caffe"]
SKL = ["Scikit-Learn"]
Other_ML = ["Other", "catboost", "lightgbm", "randomForest", "Prophet", "mlr", "Xgboost", "Spark MLlib"]
kd_init["Q20"] = kd_init["Q20"].apply(lambda x: 
                                      "TensorFlow" if x in TF else (
                                          "Other Deep Learning" if x in Other_DL else (
                                              "SKLearn" if x in SKL else (
                                                  "Other Machine Learning" if x in Other_ML else "None"))))

# Custom label encoding for categorical questions with obvious ordinality
# N.B. Q4 has some noise due to no answer responses, so impute mode (master's degree)
ords = {
    "Q4": {
        "No formal education past high school": 0,
        "Some college university study without earning a bachelors degree": 1,
        "Bachelors degree": 2,
        "Professional degree": 3,
        "I prefer not to answer": 4,
        "Masters degree": 4,
        "Doctoral degree": 5
},
    "Q8": {
        "0-1": 0,
        "1-2": 1,
        "2-3": 1,
        "3-4": 2,
        "4-5": 2,
        "5-10": 3,
        "10-15": 4,
        "15-20": 5,
        "20-25": 5,
        "25-30": 5,
        "30 +": 5
    },
    "Q25": {
        "I have never studied machine learning and I do not plan to": 0,
        "I have never studied machine learning but plan to learn in the future": 1,
        "< 1 year": 2,
        "1-2 years": 3,
        "2-3 years": 4,
        "3-4 years": 5,
        "4-5 years": 6,
        "5-10 years": 7,
        "10-15 years": 8,
        "20+ years": 9
    },
    "Q24": {
        "I have never written code and I do not want to learn": 0,
        "I have never written code but I want to learn": 1,
        "< 1 year": 2,
        "1-2 years": 3,
        "3-5 years": 4,
        "5-10 years": 5,
        "10-20 years": 6,
        "20-30 years": 7,
        "30-40 years": 8,
        "40+ years": 9
    },
    "Q23": {
        "0% of my time": 0,
        "1% to 25% of my time": 1,
        "25% to 49% of my time": 2,
        "50% to 74% of my time": 3,
        "75% to 99% of my time": 4,
        "100% of my time": 5
    }
}

# Perform the actual encoding
for keys in ords.keys():
    name = feature_names[keys]
    kd[name] = kd_init[keys].apply(lambda x: ords[keys][x.replace("/", " ")])

# Binary encoding and feature renaming for questions: Q11, Q12, Q34, and Q35
# Q11 - Work Activity
for key in range(1, 7):
    keyn = 'Q11_Part_' + str(key)
    name = feature_names[keyn.split("_")[0]] + "_" + str(kd_init[keyn][kd_init[keyn].notnull()].iloc[0])
    kd[name] = np.where(kd_init[keyn].fillna(0) == 0, 0, 1)

# Q12 - Data Analysis Tools
q_feat = {1: "Spreadsheets", 2: "Advances Stats", 3: "BI Tools", 4: "Development Env",
          5: "Cloud SaaS", 6: "Other"}
for key in range(1, 6):
    keyn = 'Q12_Part_' + str(key) + "_TEXT"
    name = feature_names[keyn.split("_")[0]] + "_" + q_feat[key]
    kd[name] = np.where(kd_init[keyn].fillna(0) == 0, 0, 1)
kd[feature_names["Q12"] + "_Other"] = np.where(kd_init["Q12_OTHER_TEXT"].fillna(0) == 0, 0, 1)

# Q34 - Data Science Project Time
q_feat = {1: "Gathering data", 2: "Cleaning data", 3: "Visualizing data", 4: "Model building and selection",
          5: "Putting the model into production", 6: "Data insights and comms",
          7: "Other"}
for key in range(1, 6):
    keyn = 'Q34_Part_' + str(key)
    name = feature_names[keyn.split("_")[0]] + "_" + q_feat[key]
    kd[name] = np.where(kd_init[keyn].fillna(0) == 0, 0, 1)
kd[feature_names["Q34"] + "_Other"] = np.where(kd_init["Q34_OTHER_TEXT"].fillna(0) == 0, 0, 1)

# Q35 - ML Training (Work, Kaggle, or Uni)?
q_feat = {1: "Self-taught", 2: "Online courses", 3: "Work", 4: "University", 5: "Kaggle", 6: "Other"}
for key in range(1, 7):
    keyn = 'Q35_Part_' + str(key)
    name = feature_names[keyn.split("_")[0]] + "_" + q_feat[key]
    kd[name] = np.where(kd_init[keyn].fillna(0) == 0, 0, 1)

# Add questions for marginal distributions after clustering
kd["Q3"] = kd_init["Q3"]
kd["Q5"] = kd_init["Q5"]
kd["Q6"] = kd_init["Q6"]
kd["Q10"] = kd_init["Q10"]
kd["Q17"] = kd_init["Q17"]
kd["Q20"] = kd_init["Q20"]
kd["Q22"] = kd_init["Q22"]

print(kd.shape)


# ### 2.  and 3. Now, let's identify questions to cluster over and some question marginals to use later and give them some concise names.

# In[ ]:


# Define features used for marginal exploration only
marginal_names = ["Q3", "Q5", "Q6", "Q10", "Q17", "Q20", "Q22"]
marginal_questions = ["Country of Residence", "Undergrad Major", "Current Title", 
                      "Employer Incorporates ML", "Most Used Programming Language",
                      "Most Used ML Framework", "Most Used Viz Tool"]

# Clustering features
cluster_features = [val for val in kd.columns if val not in marginal_names]
print("Cluster Features")
print(cluster_features)


# ### 4. We can segment survey respondents by self-identified students (Question 6 response "Student", or Question 7 response "I am a student") and non-students.

# In[ ]:


# My responses
me = [5, 8, 7, 6, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]


# In[ ]:


# Separate all self-identified students from identified non-students
kd_non_students = kd[(kd['Q6'] != "Student") & (kd_init['Q7'] != "I am a student")]
kd_students = kd[(kd['Q6'] == "Student") | (kd_init['Q7'] == "I am a student")]
print("Student feature space", kd_students.shape)
print("Non-student feature space", kd_non_students.shape)


# ### 5. Now, we can perform k-means clustering on each of the two segments and interpret personas for each cluster in each segment.

# In[ ]:


# Z-score norm features for student and non-student clustering first to avoid any feature dynamic range dominance
stsc = StandardScaler()
kd_st = stsc.fit_transform(kd_students[cluster_features])
kd_nst = stsc.fit_transform(kd_non_students[cluster_features])

# Cluster for students personas - N.B. I added the names for the personas after initial interpretation of the clusters
personas = ['The Aspiring', 'The Coder', 'The Worker']
km = KMeans(n_clusters=3, random_state=222)
kmf = km.fit(kd_st)
st_centroids = kmf.cluster_centers_
st_results = pd.DataFrame(st_centroids, columns=cluster_features, index=personas).T
kd_students['cluster'] = kmf.predict(kd_st)

# Clustering for non-students - N.B. I added the names for the personas after initial interpretation of the clusters
personas_nst = ['The Freshman', 'The Data Engineer', 'The ML Researcher', 'The Practitioner']
km = KMeans(n_clusters=4, random_state=111)
kmn = km.fit(kd_nst)
nst_centroids = kmn.cluster_centers_
nst_results = pd.DataFrame(nst_centroids, columns=cluster_features, index=personas_nst).T
kd_non_students['cluster'] = kmn.predict(kd_nst)

# What is my cluster in Non-Student segment?
my_cluster = kmn.predict(np.asarray(me).reshape(1, -1))
print(my_cluster)

# Compute the silhouette_score is for each of our segments
student_ss_score = sl(kd_st, kd_students['cluster'])
nonstudent_ss_score = sl(kd_nst, kd_non_students['cluster'])
print("Student Clusters Silhouette Score", student_ss_score)
print("Non-Student Clusters Silhouette Score", nonstudent_ss_score)


# From the above Silhouette Score outputs for Student and Non-Student segments, we see that there is some overlap in clusters within each segment. The ideal score would be +1.0 for each segment. We could perhaps do better with our selected features, but 0.27 and 0.21 are positive scores and should suffice here. 

# ### 6. Let's generate some cluster heatmaps for student and non-student segments and interpret and name them.
# ***
# Note that I Z-score normalized the features in the previous step in order to facilitate discussion of clusters in terms of their standard deviations from the norm for individual features. Specifically, where the cells in the heatmap below are annotated > 0.0, that cluster's feature/skill has a value greater than the mean across all clusters (personas) and conversely (less than the mean) for cells that have anotated values < 0.0.
# ***

# In[ ]:


# Heatmaps of student persona centroids
plt.figure(figsize=(18, 15))
ax = sbn.heatmap(st_results, cmap = 'GnBu', annot=True, cbar=True, vmin=-1, vmax=1)
plt.title('Self-Identified Student - Centroids vs Cluster Persona', fontsize=24)
plt.xlabel('Persona', fontsize=24)
plt.ylabel('Skill', fontsize=24)
plt.setp(ax.get_xticklabels(), rotation='30', fontsize=20)
plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=12)
plt.show()
plt.savefig('Identified_Student_Centroids_vs_Cluster_Persona.png')
plt.close()


# ### 7a. Student Heatmap Personas
# ***
# Recall that all of these respondents are self-identified students. As such, we are going to interpret the clusters and heatmap a little differently than the Non-Student clusters. One interesting point to make is that features in the heatmap associated with Analysis Tools dont appear to be differentiated across the clusters. 
# 
# *"The Aspiring"*
# 
# What abvout the column I am calling the "The Aspiring"? Well, first off, it is obvious from the heatmap that this group has lower college degree levels and years of work experience than the mean of all groups. Furthermore, this group is light on ML and Coding experience as well as the amount of time they spend coding (i.e., "Coding Activity" row). Compared to the other personas, they also are much less likely to claim they are performing work activities or common tasks associated with Data Science Project Time (1+ standard deviations below the mean). Lastly, they claim to have much less training across the board than do the other two groups (e.g., the "ML Training_Kaggle", and "ML Training_University" rows )
# 
# *"The Coder"*
# 
# In contrast to The Aspiring, The Coder group is slightly more likely to have degrees beyond a Master's (the mean in this case), and they have more coding and ML experience. Not only do they spend more time in coding activities than The Aspiring group, but they also are much more likely to be engaged in common data science project tasks (e.g., cleaning, gathering, and visualizing data). They are also much more likely to have taken ML course online and at University. 
# 
# *"The Worker"*
# 
# This group stands out against both The Aspiring and The Coder groups for their years of ML Experience, their Work Activities associated with ML (e.g., building prototypes, performing ML research, and running ML services). They don't tend to have more years of overall experience or higher degrees than The Coder group, but they are much more likely to be spending a lot of time on Data Science Project Time work. They also claim to have much more ML training from work, online courses, and university (1+ standard deviations above the mean).

# In[ ]:


# Heatmap of non-student persona centroids
plt.figure(figsize=(18, 15))
ax = sbn.heatmap(nst_results, cmap='GnBu', annot=True, cbar=True, vmin=-0.9, vmax=0.9)
plt.title('Self-Identified Non-Student - Centroids vs Cluster Persona', fontsize=24)
plt.xlabel('Persona', fontsize=24)
plt.ylabel('Skill', fontsize=24)
plt.setp(ax.get_xticklabels(), rotation='30', fontsize=20)
plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=12)
plt.show()
plt.savefig('Identified_Non_Student_Centroids_vs_Cluster_Persona.png')
plt.close()


# ### 7b. Non-student Heatmap Personas
# ***
# Although through a trial and error process, I identified 4 clusters for the Non-Student segment as opposed to three for the Student segment, we see similar trends in the above heatmap. Namely, we see experience and training being a distinguishing factor from group to group. We also see that work-related activities distinguish between groups. As with the Student segment, Analysis Tools do not appear to be differentiating between respondents. 
# 
# *"The Freshman"*
# Similar to The Aspiring group in the Student segment, this group is generally lacking the coding, ML and work experiences of the other groups. They are not likely to be doing data science project work, and they have little ML training. However, they do tend to have 1-2 years of experience, and claim to be doing some work-related data/ML infrastructure and services work. 
# 
# *"The Data Engineer"*
# This group distinguishes itself nicely as very data focused. They have more years of experience and coding experience than The Freshman, and they spend more time coding at work. They don't tend to have higher degrees than The Freshman, but they are more likely to spend time on work activities related to understanding data and its influence on the business and in supporting data infrastructure. They are much less likely to spend time on ML activities compared to any other group. The "Data Science Project Time" tasks they are performing are much more likely to be gathering, cleaning, and visualizing data, than it is putting ML models into production.
# 
# *"The ML Researcher"*
# Compared to The Data Engineer and The Freshman, this group is clearly focused on machine learning as their primary discipline. They are much more likely to have a PhD than the other groups. They have more years of experience, and much more ML experience than the other groups. They claim their largest Work Activity components are running ML services, doing advanced ML research, and building ML prototypes. They appear to be performing an equal amount of all of the Data Science Project Time tasks, but most of their ML training coming from work. 
# 
# *"The Practitioner"*
# Much like The ML Researcher, The Practitioner is spending lots of time on all of the Data Science Project Time tasks, but they spend most of their time on "Putting Models into Production". They have spent time training themselves in ML at University, Work, and Online. They are much less likely to have higher degrees compared to The ML Researcher, and they don't appear to have the same level of ML and coding experience. They are much less likely to be prototying new ML algorithms or doing advanced ML research. 

# ### 8. Quantify characteristics from marginal distributions across personas in each segment to help us better identify with each Kaggle survey persona.
# 
# Let's now plot, for each Student persona and Non-Student persona, distributions for questions: 
# * Q3 - Country of Residence
# * Q5 - Undergrad Major
# * Q17 - Most Used Programming Language
# * Q20 - Most Used ML Framework
# * Q22 - Most Used Viz Tool
# 
# We will also add the following for the Non-Student segment:
# * Q6 - Current Title
# * Q10 - Employer Incorporates ML
# ***
# Note that I normalize the histograms so that each response is plotted as a percentage of total responses for that question. This way, we can compare histograms easily from persona to persona without having to do the math. 

# In[ ]:


# Plot Student marginal histograms for marginal_features questions
marginal_names_st = ["Q3", "Q5", "Q17", "Q20", "Q22"]
marginal_questions_st = ["Country of Residence", "Undergrad Major", "Most Used Programming Language",
                         "Most Used ML Framework", "Most Used Viz Tool"]
for j, key in enumerate(marginal_names_st):
    ct = 1
    plt.figure(figsize=(25, 25))
    for i, cluster in enumerate(personas):
        # print key, cluster
        # print kd_students["cluster"][kd_students['cluster'] == i].count(), kd_students[key][kd_students['cluster'] == i].count(), \
        #    kd_students[key][kd_students['cluster'] == i].count()/float(kd_students["cluster"][kd_students['cluster'] == i].count())

        total_resps = float(kd_students["cluster"][kd_students['cluster'] == i].count())
        count = kd_students[key][kd_students['cluster'] == i].value_counts()

        ax = plt.subplot(len(personas), 1, ct)
        plt.title(cluster + " " + marginal_questions_st[j], fontsize=20)
        plt.barh(count.index, count/float(total_resps))
        plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=18)
        plt.setp(ax.get_xticklabels(), fontsize=18)
        plt.xlim(0, 0.6)
        plt.ylabel("Percent of Total Responses", fontsize=14)
        ct += 1
    plt.show()


# #### Student Segment Histogram Take-Aways
# 
# 
# *"Country of Residence"*
# ***
# The Aspiring are nearly 2 times more likely to be resident in India than in the US or Europe. The Workers are even more likely (3 times) to be resident in India than in US or Europe, and much more likely than China. In contrast, The Coders are more likely to reside in the US than India, Europe, or China.
# 
# *"Undergrad Major"*
# ***
# Perhaps somewhat interesting is that there appears to be very little differentiation between Personas with regard to Undergraduate Major. There are some variations between The Coders (more likely to be mathematics majors and less likely to be computer science majors) and The Workers, but overall, there is a lot of consistency between Student personas when it comes to degree major. 
# 
# *"Most Used Programming Language"*
# ***
# As expected from the progression of experience between The Aspiring persona and The Coders and The Workers, there is significant decrease in likelihood that The Aspiring will respond that they have worked with programming languages (<20%) compared to >90% for the other personas. Between The Coders and The Workers, we do see some variation in that The Workers are somewhat more likely to be using Python as their main language and somewhat less likely to be using R.  
# 
# *"Most Used ML Framework"*
# ***
# As with programming languages, greater than 60% of The Aspiring respond that they use no ML Framework currently. Between The Coders and The Workers, The Coders are somewhat less likely to respond that they predominantly use Sci-kit Learn and Deep Learning frameworks.
# 
# *"Most Used Viz Tool"*
# ***
# Similar to the most used ML Framework Question, The Aspiring are much less likely to be using any visualization tools, compared to either The Coders or The Workers, who are very consistent in their useage of visualization tools. 

# In[ ]:


# Plot Non-Student segment marginal histograms for some questions
for j, key in enumerate(marginal_names):
    ct = 1
    plt.figure(figsize=(25, 25))
    for i, cluster in enumerate(personas_nst):
        # print key, cluster
        # print kd_non_students["cluster"][kd_non_students['cluster'] == i].count(), kd_non_students[key][
        #    kd_non_students['cluster'] == i].count(), \
        #    kd_non_students[key][kd_non_students['cluster'] == i].count() / float(
        #        kd_non_students["cluster"][kd_non_students['cluster'] == i].count())

        total_resps = float(kd_non_students["cluster"][kd_non_students['cluster'] == i].count())
        count = kd_non_students[key][kd_non_students['cluster'] == i].value_counts()

        ax = plt.subplot(len(personas_nst), 1, ct)
        plt.title(cluster + " " + marginal_questions[j], fontsize=16)
        plt.barh(count.index, count / float(total_resps))
        plt.setp(ax.get_yticklabels(), rotation='horizontal', fontsize=18)
        plt.setp(ax.get_xticklabels(), fontsize=18)
        plt.xlim(0, 0.6)
        plt.ylabel("Percent of Total Responses", fontsize=14)
        ct += 1
    plt.show()


# #### Non-Student Segment Histogram Take-Aways
# 
# 
# *"Country of Residence"*
# ***
# For The Freshman and The Data Engineer, we see pretty consistent distributions for country of residence; Other, US, Europe, and India are the most likely countries for respondents, while Japan, Canada, China, Russia, and Brazil are approximately 2 times less likely to occur.  While The ML Researcher group has the same 4 countries as the most likely to occur, this group also shows a marked increase in the likelihood of residing in the US or Europe, and a marked decrease in the likelihood of respondents residing in India (>2 time less likely than for The Freshman). **This could reflect the high concentration of ML research being performed in US and Europe tech companies and academic institutions. ** 
# 
# The Practitioners, by complete contrast, have the highest likelihood of residing in India, with the US being more than 3 times less likely as a response.
# 
# *"Undergrad Major"*
# ***
# As with Country of Residence, there is little differentiation between The Freshman, The Data Engineer, and The Practitioner personas with regard to Undergraduate Major. There are some small variations between The Practitioners (more likely to be Computer Science and Physics/Astronomy majors and less likely to be Information Technology majors) and The Freshman and The Data Engineer, but overall, there is a lot of consistency between these Non-Student personas when it comes to degree major. However, The ML Scientist has a marked increase in the likelihood of having had a Mathematics or Statistics undergraduate major and a marked decrease in the likelihood of having had Information Technology as a major. **This is probably reflective of the US tech industry recruiting ML scientists from academic institutions in scientific disciplines (Physics, Engineering, Statistics, Economics, Mathematics, etc).**
# 
# *"Current Title"*
# ***
# The data here is not very surprising. The Freshman and The Data Engineers are most likely to have the Software Engineer title, while The ML Researchers and Practitioners are most likely to have a Data Scientist tile. The ML Researchers are also ~twice as likely than other personas to have a Research Scientist title. 
# 
# *"Employer Incorporates ML"*
# ***
# Here, the personas separate quite nicely between those whose employers either no not incorporate ML or are exploring using ML (The Freshman and The Data Engineers) and those whose employers have recently started using ML or have well-established methods for ML (The Practitioners and The ML Researchers). 
# 
# *"Most Used Programming Language"*
# ***
# As expected from the progression of experience between The Freshman persona and the other Non-Student personas, there is significant decrease in likelihood that The Freshman will respond that they have worked with programming languages (~30%) compared to >90% for the other personas. While The ML Scientists, The Data Engineers, and The Practitioners all have the highest likelihood of responding that there most used languages are Python, R, and SQL, The Data Engineers have the highest occurrence of SQL amoung the 3 personas. Conversely, The ML Researchers are more likely than The Data Engineers and The Practitioners to be using either Python and R as their main language.
# 
# *"Most Used ML Framework"*
# ***
# As with programming languages, greater than 60% of The Freshman respond that they use no ML Framework currently. Likewise, more than 35% The Data Engineers don't use any ML Frameworks, and only ~30% use Sci-Kit Learn. In stark contrast, some 50+% of The ML Scientists use either Sci-Kit Learn or Other ML Framework, and <15% respond that they don't use any ML Framework. Furthermore, some 30+% of The ML Researchers are using TensorFlow or Other Deep Learning Framework. The Practitioners have a similar distribution to The ML Scientists, but overall have a higher likelihood of not using any ML Framework. 
# 
# *"Most Used Viz Tool"*
# ***
# Similar to the most used ML Framework Question, The Freshman are much less likely to be using any visualization tools, compared to other personas, who are very consistent in their useage of visualization tools. 

# ### P.S. 
# ***
# My survey responses cluster as *The ML Researcher*, but if asked, I'd probably say I identify more with *The Practitioner* persona. Oh well, data science isn't perfect!

# 

# 

# 
