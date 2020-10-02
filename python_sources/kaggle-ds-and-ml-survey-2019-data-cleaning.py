#!/usr/bin/env python
# coding: utf-8

# # Cleaning Up the Kaggle DS and ML Survey 2019
# ##### A step-by-step approach to dealing with messy data and structuring it to your needs

# ## What is Data Cleaning?
# - According to **Wikipedia**, *"Data cleansing or data cleaning is the process of detecting and correcting (or removing) corrupt or inaccurate records from a record set, table, or database and refers to identifying incomplete, incorrect, inaccurate or irrelevant parts of the data and then replacing, modifying, or deleting the dirty or coarse data."*
# - Basically put, **"Data Cleaning can be thought of as a fundamental step in data science that helps convert available data into a form more suitable for the analysis task at hand."**
# 
# Taking that *easier-to-comprehend* definition to heart, I have tried cleaning up the [2019 Kaggle ML and DS Survey](https://www.kaggle.com/c/kaggle-survey-2019) data for a pet-project I am trying to work on. 

# ## The Aim of this Notebook
# 
# - Outline a "Step-by-Step" procedure to **Data Cleaning** on the [2019 Kaggle ML and DS Survey](https://www.kaggle.com/c/kaggle-survey-2019) data
# - Generate a **clean version** of "multiple_choice_responses.csv" (Can be found in the **Output** section)
# - This clean version has only 35 columns (34 question-responses and 1 for the timestamp; contrary to the 246 columns in the original *multiple_choice_responses.csv* (There is **no loss** of important data during this conversion)
# - Provide **re-usable code samples** so that anybody could use similar functions for their own projects (or maybe just improve my functions and help me learn in the process)

# ## Key Considerations while dealing with "Messy Data"
# 
# ### The "Aim" of your Analysis
# - The kind of data cleaning performed and the extent of data cleaning performed is dependent on the kind of analysis you want to perform
# - Data Cleaning is always going to be **"specific"** to the questions you want to answer
# 
# ### Understanding "Important" Data
# - While data cleaning involves removing messy values and modifying them, it is important to identify which of these values are "un-important" and can be removed
# - Just because a particular feature has about 30% missing values, it is not possible to make a mechanical call as to whether the feature must be kept or removed; we will need to identify the importance of that feature to the data we have
# - Many a time, we can convert **messy-looking** values to better formats by writing a bit of code. Eg : "500 Dollars" might seem like a bad value if we want to perform a calculation with it. So, it's better to convert it to "500" for correct computation. (Make sure it's integer)
# 
# ### Identifying the "Causes" of Messy Data
# - Identifying the cause will help shape a better solution to data cleaning
# - At times, human error or bias can be a cause of bad data and other times, faulty apparatus can engender poor data entry
# - If we know the cause, we can definitely figure out the cure!

# **NOTE:**
# - I shall be using the terms "column" and "feature" interchangeably in the course of this notebook

# Withour further ado, I shall step right into the data cleaning process!

# ## Data Cleaning Begins here!

# In[ ]:


# Loading necessary libraries

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Load the relevant datasets

mcr = pd.read_csv("../input/kaggle-survey-2019/multiple_choice_responses.csv") # responses
ques = pd.read_csv("../input/kaggle-survey-2019/questions_only.csv") # questions
txtr = pd.read_csv("../input/kaggle-survey-2019/other_text_responses.csv") # text responses
ss = pd.read_csv("../input/kaggle-survey-2019/survey_schema.csv")


# In[ ]:


# Inspect mcr using head()

mcr.head()


# ### The "*print_all*" function
# 
# - **Motivation :** Since each question in the survey is very long, printing them would cause the truncation of these long values and the truncated portion would be replaced with **...**. This is will not help us observe all the values clearly.
# - **Aim :** To print out the contents of all cells in a given row without truncation
# - **Modules needed :** pandas
# - **Application :** Here, we use it to print all the questions (in row 0 of mcr) in their full length

# In[ ]:


# Function to print the contents of all cells in a given row of a dataframe in their entire length

def print_all(df, r_num):
    """
    Task:
    To print all the content of the cells in a given row without truncation
    
    Input :
    df <- Dataframe
    r_num <- Row number (starts at 0)
    
    Output :
    Prints out all values for the input row across all attributes with corresponding headers
    """
    row = df.iloc[r_num]
    pd.set_option('display.max_rows', len(row))
    pd.set_option('display.max_colwidth', -1)
    print(row)
    pd.reset_option('display.max_rows')


# In[ ]:


# Inspecting Multiple Choice Responses (We need Questions, so that's why we are passing 0 as the row number)
print_all(mcr,0)


# There were only 34 questions in the survey in total. But, the responses dataframe (mcr) has a whopping 246 columns. Why did this happen?
# <br>
# > It's because several questions have their answers split over multiple columns. Example : "Select all that apply" and "Other Text" questions. We can notice two important patterns from mcr.head():
# * **Select all that apply** questions are in columns encoded as Qi_Part_k (i = question number, k = choice number)
# * **Other Text** responses are under the columns titled Qi_OTHER_TEXT (i = question number)

# **NOTE :** Text Responses have been encoded to maintain anonymity of respondents (as on the survey data description). So, removing the OTHER_TEXT columns from mcr makes better sense.

# ### The "*col_rem_by_exp*" function
# 
# - **Motivation :** Certain columns in a dataset are of a similar type and this type might not be be useful for analysis. And often, columns of the similar type share a common "string expression" in their titles. So, we can remove them all together based on this expression.
# - **Aim :** To remove all the columns in a dataframe based on a shared expression
# - **Modules needed :** pandas
# - **Application :** Here, we use it to remove all the "OTHER_TEXT" columns from mcr

# In[ ]:


# Function to remove columns from a dataset based on an expression in the column name

def col_rem_by_exp(df, exp):
    """
    Task:
    To remove columns from a dataset based on an expression in the column name
    
    Input :
    df <- Dataframe
    exp <- The string expression that is common to all columns that need to be removed
    
    Output :
    Returns a dataframe with the removed columns
    """
    removable_cols = []
    for i in df.columns:
        if (exp in i):
            removable_cols.append(i)
    return (df.drop(removable_cols, axis=1))


# In[ ]:


# Removing "OTHER_TEXT" columns

mcr = col_rem_by_exp(mcr, "_OTHER_TEXT")
print_all(mcr,0)


# All columns with "OTHER_TEXT" responses have been removed now, as seen from the output generated.

# ### Which are the "Select all that apply" questions ?
# 
# Here, we shall segregate all the questions that have a "Select all that apply" option and store them for later

# ### The "*select_features*" function
# 
# - **Motivation :** Certain columns in a dataset might share a common "string expression" in their titles. So, we can group them all together based on this expression.
# - **Aim :** To make a list of all the columns in a dataframe based on a shared expression
# - **Modules needed :** pandas
# - **Application :** Here, we use it to make a list of all columns with "_Part_" in their column titles (Because that's the pattern of questions with a "Select all that apply" option

# In[ ]:


# Function to segregate all the questions that have the "Select all that apply" option
def select_features(df, exp):
    """
    Task:
    To group columns from a dataset based on an expression in the column name
    
    Input :
    df <- Dataframe
    exp <- The string expression that is common to all columns that need to be aggregated
    
    Output :
    Returns the list of all features with the common expression
    """
    feature_list = []
    for i in mcr.columns:
        if ("_Part_" in i):
            q = i
            pos_ = q.index('_')
            q_no = int(q[:pos_][1:])
            if(q_no not in feature_list):
                feature_list.append(q_no)
    return feature_list


# In[ ]:


select_all_ques = select_features(mcr, "_Part_")
print('"Select all that apply" questions :')
print(select_all_ques)


# ### The "*response_combine*" function
# 
# - **Motivation :** The "Select all that apply" type questions allows for each respondent to choose more than one option for a given question. In the survey data, each of these options are encoded as separate features. (*That explains the huge 240+ features in the dataframe*)
# - **Aim :** To combine responses of "Select all that apply" questions
# - **Modules needed :** pandas
# - **Application :** Here, we use it to combine all the options chosen by a respondent for a given question into a single feature
# 
# > **CAVEAT :** This function when applied to the dataframe takes a long time to run (over 10 minutes). That's a drawback of this function and I will try to see if I can improve it in the future.

# In[ ]:


def response_combine(df, q_num):
    """
    Task:
    To combine responses of "Select all that apply" questions
    
    Input :
    df <- Multiple choice response survey
    q_num <- Question number whose responses need to be combined
    
    Output :
    > List of lists...each list corresponds to a row and all the options selected by that respondent are grouped together in it
    > Leave out the first list (it just groups the headers) once you get the output
    """
    # Identify the PARTS of the given question number
    resp_cols = []
    for i in df.columns:
        if (('Q'+str(q_num)) in i):
            resp_cols.append(i)
            
    # Aggregate all the responses of a given respondent
    responses = []
    for i in range(df.shape[0]):
        l = list(df[resp_cols].iloc[i])
        cleaned_responses = [choice for choice in l if str(choice) != 'nan']
        responses.append(cleaned_responses)
    
    # Create a dataframe of these aggregated responses, merge them with the original dataframe and delete the PARTS
    header = ("Q"+str(q_num))
    temp_df = pd.DataFrame(dict({header:responses}))
    df = df.drop(resp_cols, axis=1)
    final_df = pd.concat([df, temp_df], axis=1, sort=False)
    
    return (final_df)


# In[ ]:


# Cleaning the complete dataframe
clean_mcr = response_combine(mcr,select_all_ques[0])
for q in select_all_ques[1:]:
    clean_mcr = response_combine(clean_mcr,q)
    
print("The shape of the cleaned dataframe is :",clean_mcr.shape)


# The new dataset only has 35 columns (1 for the time taken and 34 for questions). This is much cleaner and compact.

# ### It's not over! Yet...
# Okay, so with the **clean_mcr** generated, it looks as though we have cleaned our data well. But, there is an issue that we have introduced in the course of our analysis.
# 
# - The response_combine() function removes the questions with "Select all that apply" from their original positions in the dataframe, combines all the responses for a given question and appends these new clean features at the end of the new dataframe.
# - This means that we need to make sure that the "Questions" that we provide to each of these new features are also ordered correctly. Else, there will be a mixup of column headers and values. And, **THAT IS GOING TO BE BAD!**
# 
# To avoid such a problem, I have written a few lines of code in the cell beneath that takes care of this issue.

# In[ ]:


"""Fixing the Column Headers"""

# list of all questions whose positions have changed after response_combine()
pos_changed_ques = [("Q"+str(x)) for x in select_all_ques]

# dropping the position-changed-questions from the main dataframe
questions = list(ques.loc[0,pos_changed_ques])
new_ques = ques.drop(pos_changed_ques, axis=1)

# using the concept of dataframes concatenation to create "new_ques" from "ques"
## new_ques is a modified version of ques that orders questions like how they have been modified in clean_mcr
temp_dict = {}
for i in range(len(pos_changed_ques)):
    temp_dict[pos_changed_ques[i]] = questions[i]
temp_ques = pd.DataFrame(temp_dict, index=[0])
new_ques = pd.concat([new_ques,temp_ques], axis=1)

# new_ques
new_ques


# In[ ]:


""" Final Clean Up """

# Rename columns
clean_mcr.columns = list(new_ques.iloc[0,:])

# Drop the first row
clean_mcr = clean_mcr.drop([0])

# Drop the first column (it's not needed)
# clean_mcr = clean_mcr.drop(clean_mcr.columns[[0]], axis=1)

# Save clean_mcr as an output dataset
clean_mcr.to_csv("clean_multiple_choice_responses.csv")


# ### RESULT
# **clean_mcr** is our final, cleaned survey dataframe.<br>
# You can download it from the "Output" feature.

# ### FUTURE WORK
# - Use this cleaned dataset for my project
# - Try and automate the data cleaning process above and apply it to the previous year's Kaggle DS and ML surveys
# 
# ### CONCLUSION
# 
# And that's it! My data cleaning for the task I needed is done (as of now). Hopefully, I was able to perform my task correctly, but if there is anything that I have not done well enough, I would love to know about it so that I can improve this notebook :)
