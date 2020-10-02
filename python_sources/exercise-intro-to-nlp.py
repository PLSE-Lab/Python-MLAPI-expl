#!/usr/bin/env python
# coding: utf-8

# **[Natural Language Processing Home Page](https://www.kaggle.com/learn/natural-language-processing)**
# 
# ---
# 

# # Basic Text Processing with Spacy
#     
# You're a consultant for [DelFalco's Italian Restaurant](https://defalcosdeli.com/index.html).
# The owner asked you to identify whether there are any foods on their menu that diners find disappointing. 
# 
# <img src="https://i.imgur.com/8DZunAQ.jpg" alt="Meatball Sub" width="250"/>
# 
# Before getting started, run the following cell to set up code checking.

# In[ ]:


import pandas as pd

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.nlp.ex1 import *
print('Setup Complete')


# The business owner suggested you use diner reviews from the Yelp website to determine which dishes people liked and disliked. You pulled the data from Yelp. Before you get to analysis, run the code cell below for a quick look at the data you have to work with.

# In[ ]:


# Load in the data from JSON file
data = pd.read_json('../input/nlp-course/restaurant.json')
data.head()


# The owner also gave you this list of menu items and common alternate spellings.

# In[ ]:


menu = ["Cheese Steak", "Cheesesteak", "Steak and Cheese", "Italian Combo", "Tiramisu", "Cannoli",
        "Chicken Salad", "Chicken Spinach Salad", "Meatball", "Pizza", "Pizzas", "Spaghetti",
        "Bruchetta", "Eggplant", "Italian Beef", "Purista", "Pasta", "Calzones",  "Calzone",
        "Italian Sausage", "Chicken Cutlet", "Chicken Parm", "Chicken Parmesan", "Gnocchi",
        "Chicken Pesto", "Turkey Sandwich", "Turkey Breast", "Ziti", "Portobello", "Reuben",
        "Mozzarella Caprese",  "Corned Beef", "Garlic Bread", "Pastrami", "Roast Beef",
        "Tuna Salad", "Lasagna", "Artichoke Salad", "Fettuccini Alfredo", "Chicken Parmigiana",
        "Grilled Veggie", "Grilled Veggies", "Grilled Vegetable", "Mac and Cheese", "Macaroni",  
         "Prosciutto", "Salami"]


# # Step 1: Plan Your Analysis

# Given the data from Yelp and the list of menu items, do you have any ideas for how you could find which menu items have disappointed diners?
# 
# Think about your answer. Then run the cell below to see one approach.

# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_1.solution()


# # Step 2: Find items in one review
# 
# You'll pursue this plan of calculating average scores of the reviews mentioning each menu item.
# 
# As a first step, you'll write code to extract the foods mentioned in a single review.
# 
# Since menu items are multiple tokens long, you'll use `PhraseMatcher` which can match series of tokens.
# 
# Fill in the `____` values below to get a list of items matching a single menu item.

# In[ ]:


import spacy
from spacy.matcher import PhraseMatcher

index_of_review_to_test_on = 14
text_to_test_on = data.text.iloc[index_of_review_to_test_on]

# Load the SpaCy model
nlp = spacy.blank('en')

# Create the tokenized version of text_to_test_on
review_doc =nlp(text_to_test_on)

# Create the PhraseMatcher object. The tokenizer is the first argument. Use attr = 'LOWER' to make consistent capitalization
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

# Create a list of tokens for each item in the menu
menu_tokens_list = [nlp(item) for item in menu]

# Add the item patterns to the matcher. 
# Look at https://spacy.io/api/phrasematcher#add in the docs for help with this step
# Then uncomment the lines below 

# 
matcher.add("MENU",None, *menu_tokens_list)           
# Just a name for the set of rules we're matching to
#            None,              # Special actions to take on matched words
#            ____  
#           )

# Find matches in the review_doc
# matches = ____
matches = matcher(review_doc)

# Uncomment to check your work
#q_2.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_2.hint()
#q_2.solution()


# After implementing the above cell, uncomment the following cell to print the matches.

# In[ ]:


for match in matches:
  print(f"Token number {match[1]}: {review_doc[match[1]:match[2]]}")


# # Step 3: Matching on the whole dataset
# 
# Now run this matcher over the whole dataset and collect ratings for each menu item. Each review has a rating, `review.stars`. For each item that appears in the review text (`review.text`), append the review's rating to a list of ratings for that item. The lists are kept in a dictionary `item_ratings`.
# 
# To get the matched phrases, you can reference the `PhraseMatcher` documentation for the structure of each match object:
# 
# >A list of `(match_id, start, end)` tuples, describing the matches. A match tuple describes a span `doc[start:end]`. The `match_id` is the ID of the added match pattern.

# In[ ]:


from collections import defaultdict

# item_ratings is a dictionary of lists. If a key doesn't exist in item_ratings,
# the key is added with an empty list as the value.
item_ratings = defaultdict(list)

for idx, review in data.iterrows():
    doc = nlp(review.text)
    # Using the matcher from the previous exercise
    matches = matcher(doc)
    # Create a set of the items found in the review text
    found_items = set([doc[match[1]:match[2]] for match in matches])
    
    # Update item_ratings with rating for each item in found_items
    # Transform the item strings to lowercase to make it case insensitive
    for item in found_items:
        item_ratings[str(item).lower()].append(review.stars)

q_3.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_3.hint()
q_3.solution()


# # Step 4: What's the worst reviewed item?
# 
# Using these item ratings, find the menu item with the worst average rating.

# In[ ]:


# Calculate the mean ratings for each menu item as a dictionary
mean_ratings = {item: sum(ratings)/len(ratings) for item, ratings in item_ratings.items()}

# Find the worst item, and write it as a string in worst_text. This can be multiple lines of code if you want.
worst_item = sorted(mean_ratings, key=mean_ratings.get)[0]

q_4.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_4.hint()
q_4.solution()


# In[ ]:


# After implementing the above cell, uncomment and run this to print 
# out the worst item, along with its average rating. 

print(worst_item)
print(mean_ratings[worst_item])


# # Step 5: Are counts important here?
# 
# Similar to the mean ratings, you can calculate the number of reviews for each item.

# In[ ]:


counts = {item: len(ratings) for item, ratings in item_ratings.items()}

item_counts = sorted(counts, key=counts.get, reverse=True)
for item in item_counts:
    print(f"{item:>25}{counts[item]:>5}")


# Here is code to print the 10 best and 10 worst rated items. Look at the results, and decide whether you think it's important to consider the number of reviews when interpreting scores of which items are best and worst.

# In[ ]:


sorted_ratings = sorted(mean_ratings, key=mean_ratings.get)

print("Worst rated menu items:")
for item in sorted_ratings[:10]:
    print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} \tcount: {counts[item]}")
    
print("\n\nBest rated menu items:")
for item in sorted_ratings[-10:]:
    print(f"{item:20} Ave rating: {mean_ratings[item]:.2f} \tcount: {counts[item]}")


# Run the following line after you've decided your answer.

# In[ ]:


# Check your answer (Run this code cell to receive credit!)
q_5.solution()


# # Keep Going
# 
# Now that you are ready to combine your NLP skills with your ML skills, **[see how it's done](https://www.kaggle.com/matleonard/text-classification)**.

# ---
# **[Natural Language Processing Home Page](https://www.kaggle.com/learn/natural-language-processing)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
