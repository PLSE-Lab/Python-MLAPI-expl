#!/usr/bin/env python
# coding: utf-8

#  # ||Predicting the gender of a person by training your model with some sample names (Male and female names).||

# ![](https://i.gifer.com/7uH6.gif)

# # ||Execution time||

# In[ ]:


#Build a small application which will take input a name of a person and 
#tell if the name is female or male , choosing features which our algorithm will use to identify 
#whether the person is male or female.

# A very simple function which returns the last letter of the name 
def gender_features_part1(word):
    word = str(word).lower()
    return {'last_letter': word[-1:]}


# In[ ]:


#Checking the last letter of name "sam"?????
print(gender_features_part1('Sam'))


# In[ ]:


# sample of names using the nltk built-in module
from nltk.corpus import names as names_sample
import random
names = [(name, 'male') for name in names_sample.words('male.txt')] + [(name, 'female') for name in
                                                                      names_sample.words('female.txt')]


# In[ ]:


print(names)


# In[ ]:


# run mult times we get shuffled names to reduce bias
random.shuffle(names)
for name,gender in names[:100]:
   print('Name: ', name, '    Gender:',gender)


# In[ ]:


# make a feature set for all the names
feature_sets = [(gender_features_part1(name.lower()), gender) for name, gender in names]

for dict,gender in feature_sets[:20]:
    print(dict,gender)


# In[ ]:


print(len(feature_sets))


# In[ ]:


# Making a testing data set and training data set
train_set = feature_sets[3000:]
test_set = feature_sets[:3000]


# In[ ]:


#Training dataset!!!!!!
print(train_set)


# In[ ]:


#Testing dataset!!!!!!!!!!
print(test_set)


# In[ ]:


import nltk
# Now we use the Naive Buyes classifier and train it using the train set
classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[ ]:


# Test 1.......
print(classifier.classify(gender_features_part1('Justin')))


# In[ ]:


# Test 2.......
print(classifier.classify(gender_features_part1('Sonali')))


# In[ ]:


# Test 3.......
print(classifier.classify(gender_features_part1('Ram')))


# In[ ]:


# Test 4.......
print(classifier.classify(gender_features_part1('Sita')))


# In[ ]:


# Test 5.......
print(classifier.classify(gender_features_part1('Hanuman')))


# In[ ]:


#Testing the accuracy of our classifier
print(nltk.classify.accuracy(classifier, test_set)*100)


# In[ ]:


#Checking the accuracy of training dataset
print(nltk.classify.accuracy(classifier, train_set)*100)


# In[ ]:


# show_most_informative_features function
# no of features we want to see - default value of 10

print(classifier.show_most_informative_features())


#  # Guys if you like it then please **VOTE UP**
#  
#  
# ![](https://media3.giphy.com/media/3SLnytgfJTaxy/giphy.gif) 

# In[ ]:




