#!/usr/bin/env python
# coding: utf-8

# # 1. Questions

# In[ ]:


# define Questions

Q1 = "Does the workstation have a valve?"
Q2 = "What action may I use to open the valve?"
Q3 = "Is the valve related to Pipe1?"
Q4 = "What is the data type of the input data for the valve?"
Q5 = "What is the status of the pump?"
Q6 = "How to use the emergency stop?"
Q7 = "What action may I use to close the valve?"


question_list = [Q1, Q2, Q3, Q4, Q5, Q6, Q7]


# # 2. Answers

# In[ ]:


# read in jsonld file and use it to answer questions
import json

# read file
with open('../input/siemens-task-2-dataset/W3C_WoT_Thing_Description.jsonld', 'r') as myfile:
    data=myfile.read()

# parse file
data_dict = json.loads(data)

# print(json.dumps(data_dict, indent = 4))


# In[ ]:


# Answers to questions
A1 = data_dict['properties']['ValveStatus']['description']
A2 = data_dict['actions']['OpenValve']['description']
A3 = 'ValveStatus: isPropertyOf: @id' + data_dict['properties']['ValveStatus']['isPropertyOf']['@id']
A4 = data_dict['actions']['OpenValve']['input']['type']
# or A4 = data_dict['actions']['CloseValve']['input']['type']
A5 = data_dict['properties']["PumpStatus"]["description"]
A6 = data_dict['actions']['EmergencyStop']['input']['description']
A7 = data_dict['actions']['CloseValve']['description']

# put answers in a list
A_list = [A1, A2, A3, A4, A5, A6, A7]


# # 3. Use spacy to Get a Vector Representation of Questions

# In[ ]:


# import spacy and load large library
import spacy
nlp = spacy.load('en_core_web_lg')


# In[ ]:


# use word2vec embedding of spacy 
# get 300 dim vectors out of Questions
Q1_vector = nlp(Q1).vector
Q2_vector = nlp(Q2).vector
# replace Pipe1 by Pipe 1, since for Pipe1 there exist no word embedding, 
# which makes it otherwise meaningless (get just maped to 0 vector)
Q3_vector = nlp(Q3.replace('Pipe1', 'Pipe 1')).vector    
Q4_vector = nlp(Q4).vector
Q5_vector = nlp(Q5).vector
Q6_vector = nlp(Q6).vector
Q7_vector = nlp(Q7).vector

q_vec = [Q1_vector, Q2_vector, Q3_vector, Q4_vector, Q5_vector,
         Q6_vector, Q7_vector]

# shift qestions, so that they are located around the origin
import numpy as np
shift_vec =  np.mean(np.array(q_vec), axis = 0)
q_vec_shift = q_vec - shift_vec


# # 4. Define Measure of Similarity

# *Cosine Similarity*
# $$\frac{\vec x \cdot \vec y}{|\vec x|\cdot |\vec y|} = \cos(\sphericalangle(\vec x,\vec y))$$

# In[ ]:


# define cosine similarity (typical measure for similarity in NLP context)
# and use this as measure of similarity
# same as space does with its similarity function
def cosine_similarity(array1, array2):
    return np.dot(array1, array2) / np.sqrt(np.dot(array1, array1) * np.dot(array2, array2))


# # 5. Define Function, which Answers Questions

# In[ ]:


# define here the function or class (at first a fct later maybe a class)
def answer_function(your_question):
    your_question = your_question.replace('Pipe1', 'Pipe 1')
    your_question = your_question.replace('pipe1', 'Pipe 1')    
    sim_list = []
    for question in q_vec_shift:
        sim_list.append(cosine_similarity(question, nlp(your_question).vector - shift_vec))
        
    print(sim_list)
    return A_list[np.argmax(np.array(sim_list))]


# # 6. Test Function with Some Questions

# In[ ]:


# example Questions:
# Status of pump?
# Hey robot, do we have a valve?
# How to open valve?
# Relationship of pipe1 to valve
# Input data of valve?
# STOP!!!
# How can I close the valve
answer_function('how open valve')


# # 7. Some Thoughts about Goodness and Improvement

# * works quite well
# * got only for "open valve?" a wrong answer ("How open valve?" is answered right)

# **Possible Improvements**
# 
# * Provide more possible questions for one answer, should increase accuracy of providing the right answer.
# * Possibility of answering more questions at once (e.g. "Is there a valve and is it related to Pipe1?")
# * May add some threshold of cosine similarity. If question is below that threshold provid a answer like "Sorry I don't understand your question, maybe you mean (most similar question)".
# * Possiblility to say something like "That was not my question", if one got a false answer. Then the machine should answer something like "Oh sorry for that, maybe you meant (second most similar question)". If than one answers "Yes" the machine should provide the answer to the second most similar question. If one answers "No" one can start the game from begining.
# * Possibility to provide two answers if the question is quite similar to 2 or more questions at once.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




