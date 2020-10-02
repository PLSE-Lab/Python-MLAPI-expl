#!/usr/bin/env python
# coding: utf-8

# # Introduction

# CareerVillage.org is a nonprofit that crowdsources career advice for underserved youth. Founded in 2011 in four classrooms in New York City, the platform has now served career advice from 25,000 volunteer professionals to over 3.5M online learners. The platform uses a Q&A style similar to StackOverflow or Quora to provide students with answers to any question about any career.                     
# 
# In this competition, we develop a **Content Based Recommender**  to recommend relevant questions to the professionals who are most likely to answer them.                     
# 
# The Contentbased recommender makes use of the concept that if a professional has answered a question, then the professional would be recommended **SIMILAR questions**.
# 
# **Step 1 - Preprocessing - Questions Title and Body**            
# Extract features using the Question Body and Title. Make a feature vector with these attributes for each of the questions              
# 
# **Step 2 - Preprocessing Questions Tags**        
# Get the tags associated with the questions. Create a Tags vector for each of the questions
# 
# **Step 3 - Create the Questions Feature Vector**   
# Create a feature vector consisting of Question Body , Title and Tags by merging the Questions Body , Title and Tags
# 
# At the end of Step 3, we have for all questions a feature vector consisting of Body, Title and Tags
# 
# **Step 4 - Get Questions answered by a professional**              
# Get all the questions associated with a professional.
# 
# **Step 5 -  Find Similiar Questions**              
# Find Similiar Questions which have been answered by the professional. **Cosine Similarity** is used to calculate the similarity between the questions.
# 
# **Step 6 - Questions based on Professional Tags**       
# The professionals are associated with Tags. The questions are also associated with Tags. In this step, we recommend questions based on the same tags associated with professionals and questions. This last step also helps professionals who have not started answering any questions.            

# # Proof through Scenarios       
# 
# The above mentioned steps have been coded below. This is also demonstrated with **Three Scenarios**. The demonstration with the three examples provide proof that this strategy works

# In[ ]:


import gc
import numpy as np 
import pandas as pd 
import scipy
import time

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[ ]:


#Read the relevant files
questions =  pd.read_csv("../input/questions.csv")
tags = pd.read_csv("../input/tags.csv")
tag_questions= pd.read_csv('../input/tag_questions.csv')
professionals = pd.read_csv("../input/professionals.csv")
answers = pd.read_csv("../input/answers.csv")
tag_users = pd.read_csv('../input/tag_users.csv')


# In[ ]:


questions2 = questions.copy()


# **Glimpse of the professionals data**             
# 
# This is to display the professionals data. This will be used for recommending questions to the professionals 

# In[ ]:


professionals.head(10)


# # Step  1  - Preprocessing - Questions Title and Body

# The questions data has Title and Body. **TF-IDF** technique is used to extract meaningful features from the **questions title and text**
# 
# A **document** in this case is the combination of questions title and text. 
# 
# From the book [5 Algorithms Every Web Developer Can Use and Understand](https://lizrush.gitbooks.io/algorithms-for-webdevs-ebook/content/chapters/tf-idf.html)     
# >    TF-IDF computes a weight which represents the importance of a term inside a document. 
# >    It does this by comparing the frequency of usage inside an individual document as opposed to the entire data set (a collection of documents).
# 
# The importance increases proportionally to the number of times a word appears in the individual document itself--this is called Term Frequency. However, if multiple documents contain the same word many times then you run into a problem. That's why TF-IDF also offsets this value by the frequency of the term in the entire document set, a value called Inverse Document Frequency.
# 
# ## The Math
# 
# >  TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document)         
# 
# IDF(t) = log_e(Total number of documents / Number of documents with term t in it).         
# 
# Value = TF * IDF

# A derived column `Text` is made up of the questions Title and questions Body.

# In[ ]:


# Preprocessing of text data
textfeats = ["questions_title","questions_body"]
for cols in textfeats:
    questions[cols] = questions[cols].astype(str) 
    questions[cols] = questions[cols].astype(str).fillna('') 
    questions[cols] = questions[cols].str.lower() 
    questions[cols] = questions[cols].str.replace('nan','')

questions['Text'] = questions['questions_title'] + ' ' + questions['questions_body']


# In[ ]:


cols = ['Text']
n_features = [5000]


# # Create the TF-IDF matrix    
# 
# We create the TF-IDF Matrix using the column Text and the number of features is 5000. This is configurable through the variable `n_features`     

# In[ ]:


for c_i, c in tqdm(enumerate(cols)):
    
    # The function TfidfVectorizer converts a collection of raw documents 
    # to a matrix of TF-IDF features
    tfidf = TfidfVectorizer(
        max_features=n_features[c_i],
        strip_accents='unicode',
         analyzer='word',
         lowercase=True, # Convert all uppercase to lowercase
         stop_words='english',
        norm='l2',
        )
    tfidf.fit(questions[c])
    tfidf_feature_names = tfidf.get_feature_names()
    tfidf_train = np.array(tfidf.transform(questions[c]).toarray(), dtype=np.float16)

    for i in range(n_features[c_i]):
        questions[c + '_tfidf_' + tfidf_feature_names[i]] = tfidf_train[:, i]
        
    del tfidf, tfidf_train
    gc.collect()


# In[ ]:


questions_profile = questions.drop(['Text','questions_author_id',
                                           'questions_title','questions_date_added',
                                           'questions_body'], axis=1)


# ## Glimpse of the questions profile    
# 
# The following shows a glimpse of the questions which is made of features of question body and question title   
# 

# In[ ]:


questions_profile.iloc[10:20,300:310]


#  # Step 2 - Preprocessing Questions Tags

# This section merges the questions data along with the tags data. A question is associated with various tags

# In[ ]:


questions3 = questions2


# In[ ]:


questions_tags = pd.merge(questions3, tag_questions,
                          how= 'left', 
                          left_on = 'questions_id', 
                          right_on = 'tag_questions_question_id')

questions_tags_name = pd.merge(questions_tags, tags,
                               how= 'left', 
                               left_on = 'tag_questions_tag_id', 
                               right_on = 'tags_tag_id')


# In[ ]:


questions_profile_tags = questions_tags_name[['questions_id','tags_tag_name']]
questions_profile_tags['NumberOfEntries'] = 1
questions_profile_tags_original = questions_profile_tags.copy()


# Most common tags in Questions

# In[ ]:


n =  50
topn_tags = questions_profile_tags_original.groupby(['tags_tag_name'])['tags_tag_name']. count(). sort_values(ascending = False).head(n)


# We retain the question profile tags dataframe to contain tags which are **NOT the most n popular**

# In[ ]:


notintopn_tags = ~questions_profile_tags_original['tags_tag_name'].isin(topn_tags.index)
questions_profile_tags_original = questions_profile_tags_original[notintopn_tags]


# In[ ]:


start = time.time()
questions_profile_tags = pd.pivot_table(questions_profile_tags, 
                                        values='NumberOfEntries', 
                                        index = ['questions_id'], 
                                        columns=['tags_tag_name'], 
                                        aggfunc=np.sum)
end = time.time()
end - start


# # Step 3 - Create Questions Feature Vector  
# Merge the question tags with the question feature matrix made before of various words from question title and question text

# In[ ]:


questions_profile_tags = questions_profile_tags.reset_index()
questions_profile_tags = questions_profile_tags.fillna(0)

questions_profile_complete = pd.merge(questions_profile,questions_profile_tags,
                                      left_on = 'questions_id', 
                                      right_on = 'questions_id')


# In[ ]:


questions_profile_complete.iloc[10:20,7000:7010]


# # Step 4 - Questions answered by a Professional

# A professional may have answered some questions. We build a matrix of professionals and questions. We would use the fact that if a professional has answered a question, then the professional can be recommended **Similiar Questions**

# In[ ]:


professionals_questions = pd.merge(professionals, answers, 
                                   how = 'left' ,
                                   left_on = 'professionals_id', 
                                   right_on = 'answers_author_id')


# In[ ]:


#Function to get all questions answered by a professional
def get_questions(professional):
    
    """
    Summary
    --------------
    Function to get all questions answered by a professional
    
    professional  =  the professional id
   
    """
    questions = professionals_questions[professionals_questions.professionals_id == professional].answers_question_id
    questions = pd.DataFrame(questions).rename(index=str, columns={"answers_question_id": "question_id"})
    return(questions)   


# # Step 5 - Similarity Between Questions

# In[ ]:


questions_profile_complete = questions_profile_complete.set_index('questions_id')


# In[ ]:


#Function to recommend questions for the professional
def recommend_questions(questions,topn = 100,topn1 = 50):
    
    """
    Summary
    --------------
    recommend_questions recommends questions for the professional
    
    questions  =  the questions which the professional has answered
    topn       =  the top n cosine values to be considered
    toppn1     =  the number of questions to be recommended for each question
    
    returns the recommended questions for the professional
    
    """
    questions_list = []
    cosine_values = []
    len_questions = len(questions)
    
    #Check to see if the professional has answered questions
    if(len_questions > 0):
        #Create a sparse matrix of the COMPLETE question profile consisting of title , body  and tags
        b = scipy.sparse.csr_matrix(questions_profile_complete)
        
    for i in range(len_questions):
        #get the question from the question profile
        try:
            question = questions_profile_complete.loc[questions.iloc[i].question_id]        
        except KeyError:
            # do nothing
            pass
        
        #Create a sparse matrix of the question profile consisting of title , body  and tags
        a = scipy.sparse.csr_matrix(question.values)
        # Find the cosine similarities
        cosine_similarities = cosine_similarity(a,b)
        # Find the similar indices
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        similar_questions = sorted([(questions_profile_complete.iloc[i,:],cosine_similarities[0,i]) for i in similar_indices],key=lambda x: -x[1])
             
        question_name = questions.iloc[i].question_id
        
        if(len(similar_questions) > topn1):
            length_similar = topn1
        else:
            length_similar = len(similar_questions)
        
        for j in range(length_similar):
            if(similar_questions[j][0].name != question_name):
                questions_list.append(similar_questions[j][0].name)
                cosine_values.append(similar_questions[j][1])
                
    return(questions_list,cosine_values)   


# # Recommended Questions

# ## Scenario 1 

# Here we choose a professional with professional id `0c673e046d824ec0ad0ebe012a0673e4` and run through the steps mentioned above to recommend questions

# In[ ]:


professional = '0c673e046d824ec0ad0ebe012a0673e4'
questions_all = get_questions(professional)
questions_all.head()


# The number of questions answered by the professional is shown below

# In[ ]:


len(questions_all)


# We modify the professional questions by choosing only 10 questions instead of 39 questions answered by the professional. This is done to check if the professional answers these 10 questions, can we recommend correctly the questions answered by the professional

# In[ ]:


questions = pd.DataFrame(questions_all.head(10))
questions.head()


# We now find the questions which have been recommended by the professionals.

# In[ ]:


questions_list,cosine_values = recommend_questions(questions)
cols_dictionary = {'questions': questions_list,'cosine_values': cosine_values}


# In[ ]:


questions_list_df = pd.DataFrame(cols_dictionary)
questions_list_df = questions_list_df.sort_values('cosine_values', ascending = False)
questions_list_df.head(10)


# ### Hit rate of the recommendation

# In[ ]:


questions_list_set =  set(questions_list) 
len(questions_list_set.intersection(questions_all.question_id))


# `Using 10 questions , we could recommend 9 more questions out of 39 questions.` This shows the technique works.

# # Professionals and Tags

# This section merges the professionals data along with the tags data. A user in `Career Village` has the option of associating with various tags

# In[ ]:


profile_tags = pd.merge(professionals, tag_users,
                          how= 'left', 
                          left_on = 'professionals_id', 
                          right_on = 'tag_users_user_id')

profile_tags_name = pd.merge(profile_tags, tags,
                               how= 'left', 
                               left_on = 'tag_users_tag_id', 
                               right_on = 'tags_tag_id')

profile_tags_name = profile_tags_name[['professionals_id','tags_tag_name']]

profile_tags_name = profile_tags_name.fillna(0)


# In[ ]:


profile_tags_name.head()


# # Step 6 - Recommend using Professional tags and Question Tags

# Suppose a professional joins Career Village. The person has not answered any question. We can still recommend questions to the professional by matching the tags of the professional with the tags of the questions.

# In[ ]:


def recommend_questions_using_tags(professional,questions_list = []):
    
    """
    This function recommends questions using the tags of the professional
    and the tags associated with the questions.          
    
    """
    professional_tags = profile_tags_name[profile_tags_name.professionals_id == professional]
    tags =pd.DataFrame(pd.unique(profile_tags_name[profile_tags_name.professionals_id == professional].tags_tag_name))
    tags['tags_tag_name'] = tags[0]
    print(tags)
    questions = pd.merge(tags,questions_profile_tags_original, 
                                   how = 'inner' ,
                                   left_on = 'tags_tag_name', 
                                   right_on = 'tags_tag_name')
    questions_list_tags = questions.questions_id.tolist()
    questions_list.extend(questions_list_tags)
    questions_set = set(questions_list)
    
    return(questions_set)


# In[ ]:


list_qs = list(questions.question_id)


# In[ ]:


questions_set = recommend_questions_using_tags(professional,list_qs)


# In[ ]:


len(questions_set)


# **Hit rate after considering Tags matching**

# In[ ]:


len(questions_set.intersection(questions_all.question_id))


# Using the Professionals tags and the Questions tags, more questions were matched. The cosine similarity technique along with the Tag matching of professionals and questions therefore works.

# # Scenario 2      
# Here we choose a professional with professional id `977428d851b24183b223be0eb8619a8c` and run through the steps mentioned above to recommend questions

# In[ ]:


professional = '977428d851b24183b223be0eb8619a8c'
questions_all = get_questions(professional)
questions_all


# We modify the professional questions by choosing only 5 question instead of 22 questions answered by the professional. This is done to check if the professional answers these 5 questions, can we recommend correctly the questions answered by the professional

# In[ ]:


questions = pd.DataFrame(questions_all.head(5))
questions


# We now find the questions which have been recommended by the professionals.

# In[ ]:


questions_list,cosine_values = recommend_questions(questions)
cols_dictionary = {'questions': questions_list,'cosine_values': cosine_values}


# In[ ]:


questions_list_df = pd.DataFrame(cols_dictionary)
questions_list_df = questions_list_df.sort_values('cosine_values', ascending = False)
questions_list_df.head(10)


# **Hit Rate of the Recommendation**

# In[ ]:


questions_list_set =  set(questions_list) 
questions_list_set.intersection(questions_all.question_id)


# The professional had answered `22` questions. We used `5` of the questions answered by the professional and was able to recommend `8` other questions which the professional had answered. This again proves the technique of Cosine similarity is fruitful in recommending questions to professionals        

# **Recommend using Professional tags and Question Tags**

# In[ ]:


list_qs = list(questions.question_id)


# In[ ]:


list_qs


# In[ ]:


questions_set = recommend_questions_using_tags(professional,list_qs)


# In[ ]:


len(questions_set)


# # Scenario 3   
# 
# Here we choose a professional with professional id `81999d5ad93549dab55636a545e84f2a` and run through the steps mentioned above to recommend questions

# In[ ]:


professional = '81999d5ad93549dab55636a545e84f2a'
questions_all = get_questions(professional)
questions_all


# This professional has answered 3 questions.We modify the professional questions by choosing only 1 question instead of 3 questions answered by the professional. This is done to check if the professional answers these 1 question, can we recommend correctly the questions answered by the professional

# In[ ]:


data = {'question_id': ['2c7bb1973510493aa8daf75e08bbe773']}
questions = pd.DataFrame.from_dict(data)
questions.head()


# We now recommend the questions for the professional using the 1 question

# In[ ]:


questions_list,cosine_values = recommend_questions(questions)
cols_dictionary = {'questions': questions_list,'cosine_values': cosine_values}
questions_list_df = pd.DataFrame(cols_dictionary)
questions_list_df = questions_list_df.sort_values('cosine_values', ascending = False)
questions_list_df.head(10)


# **Hit Rate of the Recommendation**

# In[ ]:


questions_list_set =  set(questions_list) 
questions_list_set.intersection(questions_all.question_id)


# The cosine similarity did not provide any matching recommendation

# **Recommend using Professional tags and Question Tags**

# In[ ]:


list_qs = list(questions.question_id)
questions_set = recommend_questions_using_tags(professional,list_qs)
len(questions_set)


# Using tags, the number of questions increases to **121**

# **Hit Rate after Tags matching**

# In[ ]:


len(questions_set.intersection(questions_all.question_id))


# `After the professionals tag matching with the questions tag matching, we could match the other questions correctly`

# # Another approach - Collaborative filtering     
# 
# We use the Collaborative filtering method to recommend questions [here](https://www.kaggle.com/ambarish/careervillage-collaborativefiltering). However the results are not promising using collaborative filtering.

# # Summary       
# We have made a **Content based recommender** and seen how we can recommend questions using the similarity of questions answered by the professional. If the professional has not answered any questions, then we match the tags of the professional with the tags associated with the questions to recommend questions.         
# 
# ## Implementation details and Future Explorations
# This can be deployed in **Production** easily to get results.  The Three scenarios prove that there is merit in using this technique for more focussed recommendations. The code tries to follow the PEP8 documentation standards and would help the implementers to comphrehend easily.
# 
# The questions profile calculation( the question profile has all the questions title , body , tags ) does not take much time. For production this can be precalculated. If a new question arrives, the question feature vector can be calculated and added to the full question profile vector.
# 
# Career Village can clean the tags associated with the questions and professionals for much better accuracy.

# # Questions and Answers

# <hr/>
# *  **Question - Did you decide to predict Pros given a Question, or predict Questions given a Pro? Why?   **           
# 
# The model chooses to predict Questions given a Professional. The idea came when I joined as a professional, I was recommended certain questions.Now when a new question arrives , the program can run and can recommend this question to professionals.         
# 
# <hr/>
# *  **Question - Does your model address the "cold start" problem for Professionals who just signed up but have not yet answered any questions? How'd you approach that challenge?**       
# 
# Step 6 of the algorithm addresses the cold start problem. The professionals are associated with Tags. The questions are also associated with Tags. In this step, we recommend questions based on the same tags associated with professionals and questions. This last step also helps professionals who have not started answering any questions.     
# <hr/>
# * **Question - Did your model have any novel approaches to ensuring that "no question gets left behind"?  **      
# 
# The last step ensures that the matching of the tags of the professional and the questions ensures that no questions are left behind.
# 
# <hr/>
# 
# * **Question - What types of models did you try, and why did you pick the model architecture that you picked?  **    
# 
# The models tried were the **Content Based Recommender** with Professionals and Questions tag matching  and the **Collaborative Filtering** method. The first method **Content Based Recommender** with Professionals and Questions tag matching gave better accuracy.         
# 
# <hr/>
# * **Question -  Did you engineer any new data features? Which ones worked well for your model? **
# 
# In the Content Based Recommender, we engineered TF-IDF features from the Questions Body and Title. This we had combined with the Questions tags. This helps us to extract important words from the Questions Body and Title.
# 
# <hr/>
# 
# * **Question -  Is there anything built into your approach to protect certain Professionals from being "overburdened" with a disproportionately high share of matches?**          
# 
# In Step 6 , we match the Professionals tag with the Questions tag. In order that the Professionals are not overburdened, we remove the Top N tags of the questions from matching. This is presently configured to 50. The developer at Career Village can configure this to any desired value. This ensures that Professionals from being "overburdened" with a disproportionately high share of matches
# 
# <hr/>
# 
# * **Question -  What do you wish you could use in the future (that perhaps we could try out on our own as an extension of your work)?**
# 
# This program could be deployed in Production. Career Village can clean the tags associated with the questions and professionals for much better accuracy. In future more templatized questions will help in doing better question similarity
# 
# 

# In[ ]:




