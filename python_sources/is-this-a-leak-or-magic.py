#!/usr/bin/env python
# coding: utf-8

# # Did any one wonder about the reason: why the selected_text has a lot of noise? 
#  
# # Well, I was one of the Kagglers who thought about this question and I was 100% sure that explaining the reason of these noisy targets is the secret for jumping in the top LB. After all, it is hosted by Kaggle and I don't think that Kaggle wanted to use noisy labels to challenge us(this is not helpful in real life).
# 
# # I was asking myself a lot of questions and I made a lot of hypothisis to explain the errors. Until the day, I found the original data in a shared kernel ([here](https://www.kaggle.com/jonathanbesomi/private-test-not-that-private-afterall))
# 
# # When comparing the competition data with the original data I realized that all the mentions in the tweets were removed (@Mohamed @love123 @azeikhff ect...) from the text and I remarked that the majority of the tweets which were filtred from these mentions have noisy labels. Starting from this point I found a way to reverse engineer the selected_text from the correct one to the noisy style lol.
# 
# # Let's try to print some examples and try to find some points in common.
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import os


# In[ ]:


def find_all(word, text):
    import re
    word = word.replace(".","\.")
    word = word.replace(")","\)")
    word = word.replace("(","\(")
    word = word.replace("?","\?")
    word = word.replace("!","\!")
    word = word.replace("*","\*")
    word = word.replace("$","\$")
    word = word.replace("[","\[")
    word = word.replace("]","\]")
    word = word.replace("+","\+")
    return [m.start() for m in re.finditer(word, text)]
               
def extract_end_index(text, selected_text):
    i=0
    last_word = selected_text.split()[-1]
    index_last_word = find_all(last_word, text)
    n_occ = len(index_last_word)
    
    selected_text_split = selected_text.split()
    text_split = text.split()
    n_end = 0
    if len(selected_text_split)==len(text_split):
        return len(text)
    for j, elm in enumerate(text_split[len(selected_text_split):]):
        i = j + len(selected_text_split)
        if elm == last_word :
            n_end +=1
            if text_split[j+1:i+1] == selected_text_split:
                break
   
    return index_last_word[n_end-1] + len(selected_text_split[-1])

def extract_start_index(text, selected_text):
    first_word = selected_text.split()[0]
    index_first_word = find_all(first_word, text)
    n_occ = len(index_first_word)
    
    selected_text_split = selected_text.split()
    text_split = text.split()
    n_start = 0
    for i, elm in enumerate(text_split):
        if (first_word !=elm) and (first_word in elm):
            n_start += elm.count(first_word)
        if elm == first_word :
            n_start +=1
            if text_split[i:i+len(selected_text_split)] == selected_text_split:
                break
    return index_first_word[n_start-1]

def jaccard(str1, str2): 
    a = set(str(str1).lower().split()) 
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def pp_v2(text, predicted, spaces):
    text = text.lower()
    predicted = predicted.strip()
    try : 
        index_start = extract_start_index(text,predicted)
        index_end = extract_end_index(text,predicted)
        if text[index_start:index_end]=="":
            return predicted
    except:
        return predicted
  
    if spaces == 1:
        return text[max(0,index_start-1):index_end]
    elif spaces == 2:
        return text[max(0,index_start-2):index_end]
    elif spaces == 3:
        return text[max(0,index_start-3):index_end-1]
    elif spaces == 4:
        return text[max(0,index_start-4):index_end-2]
    else:
        return predicted


# # We will need these functions later

# In[ ]:


original_data = pd.read_csv("../input/emotion/text_emotion.csv")
competition_data = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
my_predictions = pd.read_csv("../input/tweets-predictions/prediction_examples.csv")


# # This function will help us to extract some samples that are impossible to predicted using a word level tokenization

# In[ ]:


def impossible_to_predict(text, selected_text):
    text = str(text)
    selected_text = str(selected_text)
    
    text = set(text.split())
    selected_text = set(selected_text.split())
    
    return not selected_text.issubset(text)


# In[ ]:


competition_data["is_impossible"] = competition_data.apply(lambda x:impossible_to_predict(x.text, x.selected_text), axis=1)
competition_data["is_impossible"].sum()


# # As you can see we have 2905 samples at least that are impossible to predict. 
# # Let's print them and print their original tweet

# In[ ]:


competition_data[competition_data["is_impossible"]==True]


# # The rows number 27470, 27476, 27477 have the same problem which is an extra letter at the beginning

# # index = 27470

# In[ ]:


print(competition_data.loc[27470].text)


# In[ ]:


t = "did you fall asleep??"
t= t.lower()
print(original_data[original_data.content.str.lower().str.contains(t.lower())].content.values[0])
print(competition_data.loc[27470].text)


# # index = 27476

# In[ ]:


print(competition_data.loc[27476].text)


# In[ ]:


t = "wish we could come see u"
t= t.lower()
print(original_data[original_data.content.str.lower().str.contains(t.lower())].content.values[0])
print(competition_data.loc[27476].text)


# # index = 27477

# In[ ]:


print(competition_data.loc[27477].text)


# In[ ]:


t = "wondered about rake to."
t= t.lower()
print(original_data[original_data.content.str.lower().str.contains(t.lower())].content.values[0])
print(competition_data.loc[27477].text)


# # Can you see a common point? Yes!! I do.
# # All of these sentences have extra spaces. Look at the beginning of each sentence... It has an extra space... Not only that but for the index = 27477 we have a second extra space ".  The client" There are 2 spaces between the point and "the client" and it should be one space.
# # Lets calculate the number of extra spaces for all sentences

# In[ ]:


def calculate_spaces(text, selected_text):
    text = str(text)
    selected_text = str(selected_text)
    index = text.index(selected_text)
    x = text[:index]
    try:
        if x[-1]==" ":
            x= x[:-1]
    except:
        pass
    l1 = len(x)
    l2 = len(" ".join(x.split()))
    return l1-l2


# In[ ]:


competition_data["extra_spaces"] =  competition_data.apply(lambda x:calculate_spaces(x.text, x.selected_text), axis=1)


# In[ ]:


competition_data[competition_data.extra_spaces==2].head(20)


# # Can you see the trick? look at the noise in the selected_text above...
# # Congrats you find the magic. Let's do a further step

# In[ ]:


competition_data[competition_data.extra_spaces>3].head(20)


# # still noise but this time the noise is more intense (3 extra letters instead of one in the left and missing letter in the right)
# 
# # Can you now elaborate the magic?
# 
# # Here is the reverse of the noise:
# 

# In[ ]:


# 1. Calculate the number of extra spaces in the text. We will call this n_extra_spaces
# 2. Shift your predicted_selected_text n_extra_spaces in the beginning to the left
# 3. Shift your predicted_selected_text max((n_extra_spaces-2),0) in the end to the left


# # Let's apply it to some predictions

# In[ ]:


my_predictions


# In[ ]:


print(f"Score without reversing the trick is {my_predictions.score.mean()}")


# # Let's apply the postprocessing to build the noise

# In[ ]:


def calculate_spaces(text, selected_text):
    text = str(text)
    selected_text = str(selected_text)
    text = text.lower()
    selected_text = selected_text.lower().strip()
    index = extract_start_index(text, selected_text)
    x = text[:index]
    try:
        if x[-1]==" ":
            x= x[:-1]
    except:
        pass
    l1 = len(x)
    l2 = len(" ".join(x.split()))
    return l1-l2
my_predictions["extra_spaces"] =  my_predictions.apply(lambda x:calculate_spaces(x.text, x.predicted), axis=1)


# In[ ]:


my_predictions["new_selected"] = my_predictions.apply(lambda x: pp_v2(x.text, x.predicted,x.extra_spaces), axis=1)


# In[ ]:


my_predictions["new_score"] = my_predictions.apply(lambda x: jaccard(x.selected_text, x.new_selected), axis=1)


# In[ ]:


print(f"Score after rebuilding the noise is : {my_predictions.new_score.mean()}")


# # I hope a part of the trick is well explained here.
# # The question now: is this magic or leak or what?
# # Thank you

# In[ ]:




