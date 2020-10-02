#!/usr/bin/env python
# coding: utf-8

# # Introduction 
# 
# I recently had a chat with a friend of mine about *what defines a Data Scientist*. Given how much the term was thrown around and how much of a buzz word it's become, i thought it would be a good idea to tackle the question by exploring public datasets from Kaggle. 
# 
# To this effect, I've used job postings from the US and the UK to understand what are the requirements for a Data Scientist. I will also be exploring what are the differences between similar occupations such as 'Data Analyst', 'Machine Learning Engineer' and 'Big Data Engineer', among others.
# 
# Any comments, recommendations and advice on this kernel would be much appreciated!
# 
# *Table of contents* 
# 
# *  [Data Preparation](#Prep)
# *  [Exploratory Analysis](#Exp)
#     * [Word Clouds](#WC) 
#         * [Data Analyst Word Cloud](#dawc)
#         * [Data Scientist Word Cloud](#dswc)
#         * [Machine Learning Word Cloud](#mlwc)
#         * [Big Data Word Cloud](#bdwc)
#     * [N-gram analysis](#Ngr)
#         * [Data Analyst N-gram](#dang)
#         * [Data Scientist N-gram](#dsng)
#         * [Machine learning N-gram](#mlng)
#         * [Big Data N-gram](#bdng)
# * [Conclusion](#End)
# 
# 
# *Credits:*
# 
# Most of the code has been readapted from a [previous kernel](https://www.kaggle.com/spurryag/beginner-attempt-at-nlp-workflow).

# In[ ]:


#Import python libraries
import numpy as np 
import pandas as pd 
import seaborn as sns
import re
from collections import defaultdict


# In[ ]:


# List the files in the directory
import os
print(os.listdir("../input/"))


# # Data Preparation
# 
# <div id="Prep"> 
# 
# </div>
# 
# This step will involve converting the strings in the dataset to the appropriate data type for the analysis and removing certain words for ease.
# 

# In[ ]:


#Import the US based dataset
data_us = pd.read_csv("../input/data-scientist-job-market-in-the-us/alldata.csv")
#Import the UK based dataset
data_uk = pd.read_csv("../input/50000-job-board-record-from-reed-uk/reed_uk.csv")


# In[ ]:


#Select only the position and the associated description for the US based dataset
select_data_us = data_us[["position","description"]]
select_data_uk = data_uk[["job_title","job_description"]]
# rename UK columns
select_data_uk = select_data_uk.rename(index=str, columns={"job_title": "position", "job_description": "description"})


# In[ ]:


# Concatenate resulting dataframes
select_dat = pd.concat([select_data_us,select_data_uk],axis=0)
# Convert to strings
select_dat = select_dat.applymap(str)
# Replace certain strings
select_dat["description"] = select_dat["description"].replace(to_replace='Apply', value="",regex=True)
select_dat["description"] = select_dat["description"].replace(to_replace='apply', value="",regex=True)
select_dat["description"] = select_dat["description"].replace(to_replace='now', value="",regex=True)
select_dat["description"] = select_dat["description"].replace(to_replace='apply now', value="",regex=True)
select_dat["description"] = select_dat["description"].replace(to_replace='Apply Now', value="",regex=True)
select_dat["description"] = select_dat["description"].replace(to_replace='Job Description', value="",regex=True)
select_dat["description"] = select_dat["description"].replace(to_replace='job description', value="",regex=True)
select_dat["description"] = select_dat["description"].replace(to_replace='changes everything', value="",regex=True)
select_dat["description"] = select_dat["description"].replace(to_replace='everything', value="",regex=True)
select_dat["description"] = select_dat["description"].replace(to_replace='data scientist', value="Data Scientist",regex=True)


# In[ ]:


#View the resulting concatenated dataframe
select_dat.head()


# In[ ]:


#Check the resulting shape of the dataframe
select_dat.shape


# In[ ]:


#Select Data Analyst postings from the listings
Analyst = select_dat[select_dat['position'].str.contains("Data Analyst")] 
#View the slice
Analyst.head()


# In[ ]:


#Select Data Scientist postings from the listings
Scientist = select_dat[select_dat['position'].str.contains("Data Scientist")] 
#View the slice
Scientist.head()


# In[ ]:


#Select Machine learning postings from the listings
ML = select_dat[select_dat['position'].str.contains("Machine Learning")] 
#View the slice
ML.head()


# In[ ]:


#Select Big Data postings from the listings
BD = select_dat[select_dat['position'].str.contains("Big Data")] 
#View the slice
BD.head()


# # Exploratory Analysis
# 
# <div id="Exp"> 
# 
# </div>
# 
# 
# Given the NLP nature of the datasets, similar exploration methods will be applied to each dataset to unveil their distinct characteristics. This will include using visualisations such as:
# 
# 1) Word Clouds
# 
# Word clouds can identify trends and patterns that would otherwise be unclear or difficult to see in a tabular format. Frequently used keywords stand out better in a word cloud. Common words that might be overlooked in tabular form are highlighted in larger text making them pop out when displayed in a word cloud.
# 
# 2) N-Gram (Unigram, Bigram and Trigram)
# 
# An n-gram is a contiguous sequence of n items from a given sample of text or speech. Different definitions of n-grams will allow for the identification of the most prevalent words/sentences in the training data and thus help distinguish what comprises insincere and sincere questions.
# 
# It should be noted that prior to displaying individual words or sentences, the text will first be tokenized (based on a desired integer) and then put into a dataframe which will be used to construct side by side plots. Tokenization is, generally, an early step in the NLP process, a step which splits longer strings of text into smaller pieces, or tokens. Larger chunks of text can be tokenized into sentences, sentences can be tokenized into words, etc.

# In[ ]:


#Code for wordcloud (adapted for removal of stop words)

#Code adpted from : https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc

#import the wordcloud package
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#Define the word cloud function with a max of 200 words
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(10,10), 
                   title = None, title_size=20, image_color=False):
    stopwords = set(STOPWORDS)
    #define additional stop words that are not contained in the dictionary
    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
    stopwords = stopwords.union(more_stopwords)
    #Generate the word cloud
    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    #set the plot parameters
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
#ngram function
def ngram_extractor(text, n_gram):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

# Function to generate a dataframe with n_gram and top max_row frequencies
def generate_ngrams(df, n_gram, max_row):
    temp_dict = defaultdict(int)
    for question in df:
        for word in ngram_extractor(question, n_gram):
            temp_dict[word] += 1
    temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(max_row)
    temp_df.columns = ["word", "wordcount"]
    return temp_df

#Function to construct side by side comparison plots
def comparison_plot(df_1,df_2,col_1,col_2, space):
    fig, ax = plt.subplots(1, 2, figsize=(20,10))
    
    sns.barplot(x=col_2, y=col_1, data=df_1, ax=ax[0], color="royalblue")
    sns.barplot(x=col_2, y=col_1, data=df_2, ax=ax[1], color="royalblue")

    ax[0].set_xlabel('Word count', size=14)
    ax[0].set_ylabel('Words', size=14)
    ax[0].set_title('Top 20 Bi-grams in Descriptions', size=18)

    ax[1].set_xlabel('Word count', size=14)
    ax[1].set_ylabel('Words', size=14)
    ax[1].set_title('Top 20 Tri-grams in Descriptions', size=18)

    fig.subplots_adjust(wspace=space)
    
    plt.show()


# ## Word Clouds 
# <div id="WC"> 
# 
# </div>

# ### Data analyst Word Cloud
# <div id="dawc"> 
# 
# </div>
# 

# In[ ]:


#Select descriptions from Analyst
Analyst_desc = Analyst["description"]
Analyst_desc.replace('--', np.nan, inplace=True) 
Analyst_desc_na = Analyst_desc.dropna()
#convert list elements to lower case
Analyst_desc_na_cleaned = [item.lower() for item in Analyst_desc_na]
#remove html links from list 
Analyst_desc_na_cleaned =  [re.sub(r"http\S+", "", item) for item in Analyst_desc_na_cleaned]
#remove special characters left
Analyst_desc_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in Analyst_desc_na_cleaned]
#convert to dataframe
Analyst_desc_na_cleaned = pd.DataFrame(np.array(Analyst_desc_na_cleaned).reshape(-1))
#Squeeze dataframe to obtain series
Analyst_cleaned = Analyst_desc_na_cleaned.squeeze()


# In[ ]:


#run the function on the Data Analyst headlines and Remove NA values for clarity of visualisation
plot_wordcloud(Analyst_cleaned, title="Word Cloud of Data Analyst Descriptions")


# The above "Data Analyst" word cloud indicates that the position has requirements at the graduate level and would be inclined towards doing research using quantitative methods.

# ### Data Scientist Word Cloud 
# 
# <div id="dswc"> 
# 
# </div>
# 

# In[ ]:


#Select descriptions from Scientist
Scientist_desc = Scientist["description"]
Scientist_desc.replace('--', np.nan, inplace=True) 
Scientist_desc_na = Scientist_desc.dropna()
#convert list elements to lower case
Scientist_desc_na_cleaned = [item.lower() for item in Scientist_desc_na]
#remove html links from list 
Scientist_desc_na_cleaned =  [re.sub(r"http\S+", "", item) for item in Scientist_desc_na_cleaned]
#remove special characters left
Scientist_desc_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in Scientist_desc_na_cleaned]
#convert to dataframe
Scientist_desc_na_cleaned = pd.DataFrame(np.array(Scientist_desc_na_cleaned).reshape(-1))
#Squeeze dataframe to obtain series
Scientist_cleaned = Scientist_desc_na_cleaned.squeeze()


# In[ ]:


#run the function on the Data Analyst headlines and Remove NA values for clarity of visualisation
plot_wordcloud(Scientist_cleaned, title="Word Cloud of Data Scientist Descriptions")


# The "Data scientist" word cloud shows that this position seems to be more oriented towards business needs, rather than a typical engineering position,  and has very prominent team component (indicated by the word 'collaborate'). It might also be that the Data scientist leads teams (indicated by the word 'leading') and would thus be a senior position, as compared to the "Data Analyst" position which does not appear to have a fixed position in  hierarchy (senior or non-senior role).

# ### Machine Learning Word Cloud
# 
# <div id="mlwc"> 
# 
# </div>
# 

# In[ ]:


#Select descriptions from ML
ML_desc = ML["description"]
ML_desc.replace('--', np.nan, inplace=True) 
ML_desc_na = ML_desc.dropna()
#convert list elements to lower case
ML_desc_na_cleaned = [item.lower() for item in ML_desc_na]
#remove html links from list 
ML_desc_na_cleaned =  [re.sub(r"http\S+", "", item) for item in ML_desc_na_cleaned]
#remove special characters left
ML_desc_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in ML_desc_na_cleaned]
#convert to dataframe
ML_desc_na_cleaned = pd.DataFrame(np.array(ML_desc_na_cleaned).reshape(-1))
#Squeeze dataframe to obtain series
ML_cleaned = ML_desc_na_cleaned.squeeze()


# In[ ]:


#run the function on the Machine learning headlines and Remove NA values for clarity of visualisation
plot_wordcloud(ML_cleaned, title="Word Cloud of Machine learning positions Descriptions")


# The "machine learning" word cloud appears to be a position geared towards engineering and which has an element of seniority to it. As compared to the "Data Scientist" role which seems to be another senior role, the coding element appears to be more important for the "Machine Learning" job positions, given the "engineer" component in the wordcloud. 

# ### Big Data Word Cloud
# 
# <div id="bdwc"> 
# 
# </div>
# 

# In[ ]:


#Select descriptions from BD_US
BD_desc = BD["description"]
BD_desc.replace('--', np.nan, inplace=True) 
BS_desc_na = BD_desc.dropna()
#convert list elements to lower case
BD_desc_na_cleaned = [item.lower() for item in BS_desc_na]
#remove html links from list 
BD_desc_na_cleaned =  [re.sub(r"http\S+", "", item) for item in BD_desc_na_cleaned]
#remove special characters left
BD_desc_na_cleaned = [re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", item) for item in BD_desc_na_cleaned]
#convert to dataframe
BD_desc_na_cleaned = pd.DataFrame(np.array(BD_desc_na_cleaned).reshape(-1))
#Squeeze dataframe to obtain series
BD_cleaned = BD_desc_na_cleaned.squeeze()


# In[ ]:


#run the function on the Big Data headlines and Remove NA values for clarity of visualisation
plot_wordcloud(BD_cleaned, title="Word Cloud of Big Data positions Descriptions")


# The "Big Data" word cloud appears to be an engineering related role, rather than a typical business role. It seems to have element of analysis, in addition to helping architect data structures a firm. 

# ## N-gram Analysis
# 
# <div id="Ngr"> 
# 
# </div>

# ### Data Analyst N-gram analysis
# 
# <div id="dang"> 
# 
# </div>
# 

# In[ ]:


#Generate unigram for data analyst
Analyst_1gram = generate_ngrams(Analyst_cleaned, 1, 20)
#generate barplot for unigram
plt.figure(figsize=(12,8))
sns.barplot(Analyst_1gram["wordcount"],Analyst_1gram["word"])
plt.xlabel("Word Count", fontsize=15)
plt.ylabel("Unigrams", fontsize=15)
plt.title("Top 20 Unigrams for Data Analyst Descriptions")
plt.show()


# In[ ]:


#Obtain bi-grams and tri-grams (top 20)
Analyst_2gram = generate_ngrams(Analyst_cleaned, 2, 20)
Analyst_3gram = generate_ngrams(Analyst_cleaned, 3, 20)
#compare the bar plots
comparison_plot(Analyst_2gram,Analyst_3gram,'word','wordcount', 0.5)


# The n-gram analysis for the "data analyst" position further indicates that the role can be very general, with responsibilities ranging across data science, big data, analytics and machine learning. It might also be inferred from the above that data analysts (based on the sample dataset) typically have a graduate eductation, with their more experienced counterparts considered to be Senior data analysts.  

# ### Data Scientist N-gram Analysis
# 
# <div id="dsng"> 
# 
# </div>
# 

# In[ ]:


#Generate unigram for data analyst
Scientist_1gram = generate_ngrams(Scientist_cleaned, 1, 20)
#generate barplot for unigram
plt.figure(figsize=(12,8))
sns.barplot(Scientist_1gram["wordcount"],Scientist_1gram["word"])
plt.xlabel("Word Count", fontsize=15)
plt.ylabel("Unigrams", fontsize=15)
plt.title("Top 20 Unigrams for Data Scientist Descriptions")
plt.show()


# In[ ]:


#Obtain bi-grams and tri-grams (top 20)
Scientist_2gram = generate_ngrams(Scientist_cleaned, 2, 20)
Scientist_3gram = generate_ngrams(Scientist_cleaned, 3, 20)
#compare the bar plots
comparison_plot(Scientist_2gram,Scientist_3gram,'word','wordcount', 0.5)


# The n-gram analysis for the "Data Scientist" position further appears to  that the role can is focused on using statistical techniques and machine learning models to analyse large datasets. It can be inferred that, similar to the "Data Analyst" position, the "Data Scientist" has to use skills across the fields of data mining, big data, analytics and machine learning.

# ### Machine Learning Positions N-gram Analysis
# 
# <div id="mlng"> 
# 
# </div>
# 

# In[ ]:


#Generate unigram for ML positions
Scientist_1gram = generate_ngrams(ML_cleaned, 1, 20)
#generate barplot for unigram
plt.figure(figsize=(12,8))
sns.barplot(Scientist_1gram["wordcount"],Scientist_1gram["word"])
plt.xlabel("Word Count", fontsize=15)
plt.ylabel("Unigrams", fontsize=15)
plt.title("Top 20 Unigrams for Machine Learning positions descriptions")
plt.show()


# In[ ]:


#Obtain bi-grams and tri-grams (top 20)
ML_2gram = generate_ngrams(ML_cleaned, 2, 20)
ML_3gram = generate_ngrams(ML_cleaned, 3, 20)
#compare the bar plots
comparison_plot(ML_2gram,ML_3gram,'word','wordcount', 0.5)


# The n-gram analysis for the "Machine Learning" positions reveal that they are indeed an engineering related role, with a degree in computer science being typically required. Additionally, it is also revealed to be a more specific role than its "Data Scientist" counterpart where terms such as deep learning, software development, language processing and artificial intelligence are used. 

# ### Big Data Positions N-gram Analysis
# 
# <div id="bdng"> 
# 
# </div>

# In[ ]:


#Generate unigram for ML positions
BD_1gram = generate_ngrams(BD_cleaned, 1, 20)
#generate barplot for unigram
plt.figure(figsize=(12,8))
sns.barplot(Scientist_1gram["wordcount"],Scientist_1gram["word"])
plt.xlabel("Word Count", fontsize=15)
plt.ylabel("Unigrams", fontsize=15)
plt.title("Top 20 Unigrams for Big Data positions descriptions")
plt.show()


# In[ ]:


#Obtain bi-grams and tri-grams (top 20)
BD_2gram = generate_ngrams(BD_cleaned, 2, 20)
BD_3gram = generate_ngrams(BD_cleaned, 3, 20)
#compare the bar plots
comparison_plot(BD_2gram,BD_3gram,'word','wordcount', 0.5)


# The n-gram analysis for the Big Data roles appear to confirm that they are engineering roles which rely on using big data technologies (map reduce) and analytics. They seem to require a computer science related educated and seem to encompass some of the responsibilities of the "Data Scientists".

# # Conclusion
# <div id="End"> 
# 
# </div>

# In summary, this analysis indicates that for the below positions, the folllowing appear to be hold:
# 
# * Data Analyst positions - Graduate level education for an entry level position (relative to the other positions analysed in this kernel) which requires knowledge of data science, big data, analytics and machine learning. 
# 
# * Data Scientist positions- Focused towards meeting business needs and leading teams to meet the latter. It is tyically required to use statistical techniques and machine learning models to analyse large datasets. Similar to the "Data Analyst" position, the "Data Scientist" has to use skills across the fields of data mining, big data, analytics and machine learning.
# 
# * Machine Learning positions -  Engineering focused role, with a degree in computer science being typically required. It appears to be more specific role than its "Data Scientist" counterpart where terms such as deep learning, software development, language processing and artificial intelligence are used. 
# 
# * Big Data positions -  Engineering roles which rely on using big data technologies (map reduce) and analytics. They seem to require a computer science related educated and seem to encompass some of the responsibilities of the "Data Scientists".
# 
# Further improvements to this kernel could be to:
# 
# * Clean the descriptions to particularly outline which responsibilities and qualifications are required for each position. This would help to remove company related descriptions and focus on what actually is required of each position. 
