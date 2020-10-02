#!/usr/bin/env python
# coding: utf-8

# # Creating your own chatbot: RelaBot

# In[ ]:


import en_core_web_lg # Large SpaCy model for English language
import numpy as np
import re # regular expressions
import spacy # NLU library

from collections import defaultdict
from sklearn.svm import SVC # Support Vector Classification model


# In[ ]:


output_format = "IN: {input}\nOUT: {output}\n" + "_"*50


# ## Version 1: Use exact matches

# In[ ]:


# hard-coded exact questions
responses_exact = {
    "what would you like to eat tonight?": "Pasta with salmon and red pesto please!",
    "what time will you be home tonight?": "I will be home around 6 pm.",
    "default": "I love you too!"
}

def respond_exact(text):
    response = responses_exact.get(text.lower(), responses_exact['default'])
    return(output_format.format(input=text, output=response))


# In[ ]:


print(respond_exact("What would you like to eat tonight?"))
print("_"*50)
print(respond_exact("What time will you be home tonight?"))
print("_"*50)
print(respond_exact("I just found out my boss is leaving the company."))


# ### Assignment 1: Add a default response 

# ### Assignment 2: Think of ways to extend this functionality

# In[ ]:





# ## Version 2: Pattern Matching

# In[ ]:


# Define keywords that can help determine the intent 
intent_keywords = {
    'dinner_preference': ['eat', 'dinner', 'food', 'cook', 'craving'],
    'arrival_time': ['time', 'when', 'get here', 'be home']
}
# Create a dictionary of patterns
patterns = {intent: re.compile('|'.join(keys)) for intent, keys in intent_keywords.items()}

# Define a function to find the intent of a message
def get_intent_re(message):
    for intent, pattern in patterns.items():
        # Check if the pattern occurs in the message 
        if pattern.search(message):
            return(intent)
    else:
        return('default')

responses_re = {
    "dinner_preference":"Pasta with salmon and red pesto please!",
    "arrival_time": "I will be home around 6 pm.",
    "default":"I like you too!"
}

def respond_re(text):
    response = responses_re.get(get_intent_re(text))
    return(output_format.format(input=text, output=response))


# In[ ]:


print(respond_re("what would you like to eat tonight?"))
print(respond_re("what time will you be home tonight?"))
print(respond_re("I just food out my boss is leaving the company."))


# ### Assignment 3: Improve the chatbot's recognition capability

# ## Version 3: Machine Learning

# ### Step 3.1: Finding out what he / she wants

# In[ ]:


# Create training data
training_sentences = [
    "What would you like to have for dinner?",
    "What do you want to eat tonight?",
    "I don't know what to cook tonight.",
    "Do you have any cravings?",
    "Can I get you something to eat?", 
    "What time will you be home?",
    "How much longer will you be?",
    "When can we expect you to get here?",
    "What's taking you so long?",
    "At what hour will you be here?"
    
]
training_intents = [
    "dinner_preference",
    "dinner_preference",
    "dinner_preference",
    "dinner_preference",
    "dinner_preference",
    "arrival_time",
    "arrival_time",
    "arrival_time",
    "arrival_time",
    "arrival_time"   
]


# In[ ]:


# this may take a couple of seconds
nlp = en_core_web_lg.load()


# In[ ]:


# Initialize the array with zeros: X
X_train = np.zeros((len(training_sentences), 
              nlp('sentences').vocab.vectors_length))

for i, sentence in enumerate(training_sentences):
    # Pass each each sentence to the nlp object to create a document
    doc = nlp(sentence)
    # Save the document's .vector attribute to the corresponding row in X
    X_train[i, :] = doc.vector


# In[ ]:


# Create a support vector classifier
clf = SVC(C=1, gamma="auto", probability=True)

# Fit the classifier using the training data
clf.fit(X_train, training_intents)

#Yes, a lot can be done here to check / improve model performance! We will leave that for another day!


# In[ ]:


def get_intent_ml(text):
    doc = nlp(text)
    return(clf.predict([doc.vector])[0])


# ### Step 3.2 Figure out how to deal with it.

# In[ ]:


responses_ml = {
    "dinner_preference":"Pasta with salmon and red pesto please!",
    "arrival_time": "I will be home around 6 pm.",
    "default":"I like you too!"
}

def respond_ml(text):
    response = responses_ml.get(get_intent_ml(text), responses_ml["default"])
    return(output_format.format(input=text, output=response))


# In[ ]:


print(respond_ml("what would you like to eat tonight?"))
print(respond_ml("what time will you be home tonight?"))
print(respond_ml("l"))


# ### Assignment 4: Add default 

# ### Assignment 5: Add variety to your answers, so he / she will think it's you

# ### Assignment 6: Extract context from the sentence:
# #### What day of the week is the question about?

# ### 7: Bonus Assignments
# 
# 1. add variety to food
# 2. stem words in put text to improve recognition
# 3. combine patterns
# 4. ask for specification

# ## Extra input for assignments

# In[ ]:


# add this to responses dict: "default": "I love you too!"
# in the predict function: if the model is not too sure about the intent, return the string "default"
    # There is a function that gives the probabilities for each of the possible outputs
    # If the maximum probability is low, one might say that the model is not sure about the intent
    # Note! This idea should work, but relies on the functionality of the predict_proba function:
    # for the SVC model, the predict_proba function does not give meaningfull results for small datasets:
    # http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC.predict_proba

def get_intent_ml_2(text):
    """
        Returns the intent from a given text, unless the model is not sure, in which case 'default' is returned
    """
    doc = nlp(text)
    max_proba = max(clf.predict_proba([doc.vector])[0])
    if(max_proba == 0.5):
        return('default')
    else:
        return(clf.predict([doc.vector])[0])

def respond_ml_2(text):
    response = responses_ml.get(get_intent_ml_2(text), responses_ml["default"])
    return(output_format.format(input=text, output=response))


# In[ ]:


print(respond_ml(  'flowers'))
print(respond_ml_2('flowers'))


# In[ ]:


def get_all_entities(text):
    """
        Get all entities in a given text, in a text: label_ dictionary
    """
    doc = nlp(text)
    
    d = defaultdict(list)
    for ent in doc.ents:
        d[ent.label_].append(ent.text)
    return(d)


# In[ ]:


test_ents = get_all_entities('what would you like to eat tonight?, or next tuesday or wednesday fish football Bengals')
print(sorted(test_ents.items()))


# In[ ]:


policy = {
    ("dinner_preference", "time and date"): "I want to eat pasta",
    ("dinner_preference", "time only"): "I want to eat pasta",
    ("dinner_preference", "date only"): "I want to eat pasta",
    ("dinner_preference", "none"): "When?",
    ("arrival_time", "time and date"): "I will be home at six",
    ("arrival_time", "time only"): "I will be home at six",
    ("arrival_time", "date only"): "I will be home at six",
    ("arrival_time", "none"): "When?",
    ("default", "none"): "What do you want?",
}


# In[ ]:


def respond_ml_3(text):
    """Check for specification of date and time
        If not specified, ask for clarification
    """
    intent = get_intent_ml_2(text)
        
    if intent != 'default':
        entities = get_all_entities(text)
        if 'TIME' in entities and 'DATE' in entities:
            specification = 'time and date'
            time = ' and '.join(entities['DATE']) + ' at ' + ' and '.join(entities['TIME'])
        elif 'TIME' in entities:
            specification = 'time only'
            time = ' and '.join(entities['TIME'])
        elif 'DATE' in entities:
            specification = 'date only'
            time = ' and '.join(entities['DATE'])
        else:
            specification = 'none'
            time = ""
    else:
        specification = 'none'
        time = ""
    
    response = policy.get((intent, specification)) + ' ' + time
    return(output_format.format(input=text, output=response))


# In[ ]:


preferences {"monday" :"pancakes"}


# In[ ]:


print(respond_ml_3('what would you like to eat next wednesday thursday and friday?'))


# In[ ]:




