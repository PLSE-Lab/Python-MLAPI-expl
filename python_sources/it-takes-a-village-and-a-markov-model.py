#!/usr/bin/env python
# coding: utf-8

#  <div style="text-align:center"><span style="font-size:2em"> It Takes a Village - CareerVillage Initial Analysis</div></style>
#  <div style="text-align:center">By Maya Shaked</div>

# ## Our task at hand is to develop a method to recommend relevant questions to the professionals who are most likely to answer them.

# In[ ]:


import pandas as pd
import sys
import math

# loading data
answers = pd.read_csv("../input/answers.csv")
comments = pd.read_csv("../input/comments.csv")
emails = pd.read_csv("../input/emails.csv")
group_memberships = pd.read_csv("../input/group_memberships.csv")
groups = pd.read_csv("../input/groups.csv")
matches = pd.read_csv("../input/matches.csv")
professionals = pd.read_csv("../input/professionals.csv")
questions = pd.read_csv("../input/questions.csv")
school_memberships = pd.read_csv("../input/school_memberships.csv")
students = pd.read_csv("../input/students.csv")
tag_questions = pd.read_csv("../input/tag_questions.csv")
tag_users = pd.read_csv("../input/tag_users.csv")
tags = pd.read_csv("../input/tags.csv")


# # ** So, let's first explore some relevant data!**

# In[ ]:


#  Let's see what's going on in the `questions` dataframe
questions.head(1)


# In[ ]:


# our data description lets us know that "Answers are what this is all about! Answers get posted in response to questions. Answers can only be posted by users who are registered as Professionals." 
# So, let's check out our `answers` data followed by our `professionals` data
answers.head(1)


# In[ ]:


professionals.head(1)


# In[ ]:


# We eventually will need to know what professional answered what question. Lets combine the `professionals` data and our answers by JOINing with `answers` 
answers_with_professions = pd.merge(answers, professionals, left_on = "answers_author_id", right_on = "professionals_id")

# Now, we can connect our questions/tags data with answers/professions by JOINing on the question_id
q_and_a = pd.merge(questions, answers_with_professions, left_on = "questions_id", right_on = "answers_question_id")

q_and_a.head(1)


# # Let's generate a Markov model for each professional
# 
# ### Given a question string of text from an unidentified professional, we will use the Markov model to assess the likelihood that it was uttered by the particular professional to which that particular Markov model corresponds. If we have built models for different professionals (based on our training data), then we will have likelihood values for each, and will choose the professional with the highest likelihood as the best responder to the question.

# # First, we must build a Hash Table (with linear probing) class to hold our model's k-grams

# In[ ]:


TOO_FULL = 0.5
GROWTH_RATIO = 2


class Hash_Table:

    def __init__(self,cells,defval):
        '''
        Construct a new hash table with a fixed number of cells equal to the
        parameter "cells", and which yields the value defval upon a lookup to a
        key that has not previously been inserted
        '''
        
        self.defval = defval
        self.size = cells
        self.slots = [[None, self.defval] for x in range(self.size)]
        self.occupied = 0

    def hash(self, string):

        return hash(string)


    def lookup(self,key):
        '''
        Retrieve the value associated with the specified key in the hash table,
        or return the default value if it has not previously been inserted.
        '''

        hashed = self.hash(key)
        ind = hashed % self.size
        og_ind = ind
        val = self.defval

        while self.slots[ind][0] != None:

            if self.slots[ind][0] == key:
                val = self.slots[ind][1]
                break

            else:
                ind = self.probe(ind)

                if ind == og_ind:
                    break

        return val


    def update(self,key,val):
        '''
        Change the value associated with key "key" to value "val".
        If "key" is not currently present in the hash table,  insert it with
        value "val".
        '''

        hashed = self.hash(key)
        ind = hashed % self.size

        if self.slots[ind][0] == key:

            self.slots[ind][1] = val

        elif self.slots[ind][0] == None:
            
            self.occupied += 1
            self.slots[ind][0] = key
            self.slots[ind][1] = val

        else:

            while (self.slots[ind][0] != None and self.slots[ind][0] != key):

                ind = self.probe(ind)

            if self.slots[ind][0] == None:
                self.slots[ind][0] = key
                self.slots[ind][1] = val
                self.occupied += 1

            elif self.slots[ind][0] == key:
                self.slots[ind][1] = val

        if self.occupied / self.size > TOO_FULL:

            self.rehash()

        pass


    def rehash(self):
        '''
        Grows our hash table by the given GROWTH_RATIO and rehashes all keys
        '''

        self.size = self.size * GROWTH_RATIO
        self.occupied = 0

        keys = []
        for [k, v] in self.slots:
            if k != None:
                keys.append([k, v])

        self.slots =  [[None, self.defval] for x in range(self.size)]

        for [k, v] in keys:
            
            self.update(k, v)

        pass


    def probe(self, prev_ind):
        '''
        Gives the next index in our hash table
        '''
        
        if prev_ind + 1 < self.size:

            return prev_ind + 1

        else:

            return 0


# # Now we can build our Markov class off of our Hash_Table class

# In[ ]:


HASH_CELLS = 57

class Markov:

    def __init__(self,k,s):
        '''
        Construct a new k-order Markov model using the statistics of string "s"
        '''

        self.k = k
        self.table = Hash_Table(HASH_CELLS, 0)
        self.s = s

        self.update_all_ks(self.k)
        self.update_all_ks(self.k + 1)

    def update_all_ks(self, k):
        '''
        Adds counts of all substrings of length k to our hash table
        '''

        looped_s = self.s[-k + 1:] + self.s

        for i in range(len(self.s)):
            to_add = looped_s[i : i + k]
            val = self.table.lookup(to_add)
            self.table.update(to_add, val + 1)

        pass


    def log_probability(self,s):
        '''
        Get the log probability of string "s", given the statistics of
        character sequences modeled by this particular Markov model
        This probability is *not* normalized by the length of the string.
        '''

        logprob = 0

        looped_s = s[-self.k :] + s

        smoothing_num = len(set(self.s))


        for i in range(len(s)):
            full_seq = looped_s[i : i + self.k + 1]
            to_check_seq = looped_s[i : i + self.k]

            numerator = self.table.lookup(full_seq) + 1
            denominator = self.table.lookup(to_check_seq) + smoothing_num

            if denominator > 0:
                logprob += math.log(numerator / denominator)

        return logprob


# # Now it's time for us to go back to our data, clean them up, and build all our models

# In[ ]:


def gen_models_for_professions(question_data, k):

    all_professions = list(question_data.professionals_industry.unique())

    models = {}
    for profession in all_professions:
        profession_questions = question_data.loc[q_and_a['professionals_industry'] == profession]["questions_body"]
        all_questions = ""
        for question in profession_questions:
            all_questions += question.lower() + " "
        models[profession] = Markov(k, all_questions)
        
    return(models)


# # Now that we have stored our models, we need to be able to actually determine the best professional to answer based on a given question. 
# 

# In[ ]:


def determine_profession(models, string_to_test):
    
    logprobs = {}
    
    for profession in models:
        markov_model = models[profession]
        logprob = markov_model.log_probability(string_to_test)
        logprobs[profession] = logprob
        
    return(min(logprobs, key = logprobs.get))
    # gotta remember that low log-probability is actually high probability!
    


# # Let's check out how it works!

# In[ ]:


models = gen_models_for_professions(q_and_a, 5)

print(determine_profession(models, "I'm sad, can someone help me with depression?"))
print(determine_profession(models, "How do I get financial aid for college"))
print(determine_profession(models, "Where should I buy school supplies like notebooks and pencils?"))
print(determine_profession(models, "What are the best educational TV shows?"))
print(determine_profession(models, "How hard is it to get a position as a Data Scientist at a top tech company?"))

