#!/usr/bin/env python
# coding: utf-8

# ##### Transform your data from,
# * I am sleepyyyyy!!          --> I am sleepy!!
# * that was meeeaaannn.       --> that was mean.
# * tiiiiiiired! Going to bed! --> tired! Going to bed!
# 
# 
# ##### And then map the predicted data back to the original data as follows (considering the examples above),
# * sleepy!! --> sleepyyyyy!!
# * that was mean. --> that was meeeaaannn.
# * tired! --> tiiiiiiired!

# In[ ]:


import random
import pandas as pd
from nltk.corpus import words


# In[ ]:


class DataPreprocessor():
    def __init__(self, df):
        self.df = df.copy()
        self.text = self.df.text.values
        self.text = self.text.copy()
        self.originals = self.text.copy()
        self.selected_text = self.df.selected_text.values
        self.id = self.df.textID.values
        self.is_mod = []
        self.mods = {}
        self.corpus = words.words()
        self.corpus += ['lol', 'll']

    def preprocess(self, selected_text=False):
        text = None
        ts = selected_text
        if ts:
            text = self.selected_text
        else:
            text = self.text
        
        for i in range(len(text)):
            s = text[i]
            s1 = ""
            temp = ""
            modified = False
            
            for ind in range(len(s)):
                if s[ind].isalnum():
                    temp += s[ind]
                else:
                    temp = temp
                    temp1 = temp
                    if self.has_3_conts(temp):
                        if temp not in self.corpus:
                            new = self.normalize(temp)
                            if modified:
                                self.mods[self.id[i]][new] = temp
                            else:
                                modified = True
                                self.is_mod.append(self.id[i])
                                self.mods[self.id[i]] = {new: temp1}
                            temp = new
                    s1 += temp + s[ind]
                    temp = ""
            
            if s[-1].isalnum():
                temp = temp
                temp1 = temp
                if self.has_3_conts(temp):
                    if temp not in self.corpus:
                        new = self.normalize(temp)
                        if modified:
                            self.mods[self.id[i]][new] = temp
                        else:
                            modified = True
                            self.is_mod.append(self.id[i])
                            self.mods[self.id[i]] = {new: temp1}
                        temp = new
                s1 += temp
                temp = ""
            
            if ts:
                self.selected_text[i] = s1
            else:
                self.text[i] = s1
        if ts:
            self.df.selected_text = self.selected_text
        else:
            self.df.text = self.text
    
    def has_3_conts(self, word):
        for i in range(2, len(word)):
            if word[i] == word[i-1] == word[i-2]:
                return True
        return False
    
    def normalize(self, word):
        reps = []
        j = 0
        prev = word[0]
        for i in range(1, len(word)):
            if word[i] != prev:
                if j != i-1:
                    reps.append((j, i))
                j = i
                prev = word[j]
        if j != len(word)-1:
            reps.append((j, len(word)))
        if len(reps) == 1:
            temp1 = word[:reps[0][0]] + word[reps[0][0]] + word[reps[0][1]:]
            temp2 = word[:reps[0][0]] + word[reps[0][0]] * 2 + word[reps[0][1]:]
            temp1 = temp1.lower()
            temp2 = temp2.lower()
            if temp2 in self.corpus:
                return temp2
            else:
                return temp1
        if len(reps) == 2:
            temp1 = word[:reps[0][0]] + word[reps[0][0]] + word[reps[0][1]:reps[1][0]] + word[reps[1][0]] + word[reps[1][1]:]
            temp2 = word[:reps[0][0]] + word[reps[0][0]]*2 + word[reps[0][1]:reps[1][0]] + word[reps[1][0]]*2 + word[reps[1][1]:]
            temp3 = word[:reps[0][0]] + word[reps[0][0]] + word[reps[0][1]:reps[1][0]] + word[reps[1][0]]*2 + word[reps[1][1]:]
            temp4 = word[:reps[0][0]] + word[reps[0][0]]*2 + word[reps[0][1]:reps[1][0]] + word[reps[1][0]] + word[reps[1][1]:]
            temp1 = temp1.lower()
            temp2 = temp2.lower()
            temp3 = temp3.lower()
            temp4 = temp4.lower()
            for i in self.corpus:
                if i == temp4:
                    return temp4
                elif i == temp3:
                    return temp3
                elif i == temp2:
                    return temp2
                elif i == temp1:
                    return temp1
            return temp1
        else:
            temp = ""
            for i in range(1, len(word)):
                if word[i] != word[i-1]:
                    temp += word[i-1]
            temp += word[-1]
            return temp
        
    def postprocess(self, df):
        df = df.copy()
        st = df.selected_text.values
        ids = df.textID.values
        
        for i in range(len(st)):
            assert ids[i] == self.id[i]
            if ids[i] in self.is_mod:
                s1 = st[i]
                s2 = self.text[i]
                s3 = self.originals[i]
                s1 = list(s1.split(' '))
                s2 = list(s2.split(' '))
                s3 = list(s3.split(' '))
                for ind in range(len(s3)):
                    if s2[ind] == s1[0]:
                        if s2[ind:ind+len(s1)-1] == s1[:-1]:
                            st[i] = ' '.join(s3[ind:ind+len(s1)])
                            break
        
        df.selected_text = st
        return df


# In[ ]:


df = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)


# In[ ]:


preprocessor = DataPreprocessor(df)


# In[ ]:


preprocessor.preprocess()
preprocessor.preprocess(selected_text=True)


# In[ ]:


# sentence wise modifications
# Note: here the values in the dict are the original words and the keys are the converted words
for i, id in enumerate(preprocessor.mods):
    if i > 25:
        break
    print(id + ": " + str(preprocessor.mods[id]))


# In[ ]:


# original VS edited data
t1 = df.selected_text.values
t2 = preprocessor.df.selected_text
for i in range(50):
    if preprocessor.id[i] in preprocessor.is_mod:
        print(t1[i])
        print(t2[i], end="\n\n")


# In[ ]:


# suppose the predictions are exactly equal to selected_text of the preprocessed data
preds = preprocessor.df.selected_text.values
submission = df.drop(["text", "sentiment"], axis=1)
submission.selected_text = preds


# In[ ]:


# This maps the outputs with the original data back
submission = preprocessor.postprocess(submission)


# In[ ]:


# you can see we have the original data back
# Note: because of the noice in the raw data(half words in the selected_text column of the training data, etc.)
# some of the convertion below will not be exactly equal to the original data, but with the actual prediction 
# form the model you can self engineer your model to not result in half words and thus the overall process 
# will become consistent.
t1 = preds
t2 = submission.selected_text.values
t3 = df.selected_text.values

for i in range(100):
    # check if the datapoint has been modified by the preprocessor
    if preprocessor.id[i] in preprocessor.is_mod:
        print("predicted data: " + t1[i])
        print("converted data: " + t2[i])
        print("original data : " + t3[i], end="\n\n")


# In[ ]:




