#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print("Sorry no code examples here as it is written in Rust. Source code is available in gitlab link below.")


# # Ponygram - ngram pony prediction by using phrase lines
# ## Introduction
# This program tries to predict MLP-characters by phrase lines. The data used is 'clean_dialog.csv'. Also some other general statistics is analyzed.
# 
# Note: Prediction accuracy is far from optimal, and ngram implementation is very basic. Also the source code is documented poorly. The reason this simple project was published to Kaggle, is to say a _Thank You_ to Liu Renyu who gathered this amazing data set. It was a great fun to play with, even though nothing that special was achieved.
# 
# Program is written in Rust (Rust is like C++ but way less painful). Source code available here: 
# https://gitlab.com/tolvanea/sgn-44006_artificial_intelligence_weekly_exercises/tree/master/ex4
# 
# ## Methods
# This program uses ngram-text classification with simulatenous n=1 and n=2. The ngram works in word-level integer encoded vocabulary.
# See more: https://en.wikipedia.org/wiki/N-gram
# 
# Ngram is not optimal algorithm for predicting pony by line, because most prase lines are quite short (< 20 words), and there is simply just not enough data (i.e. word-pairs) to do accurate probabilistic matching. Also there are a lot of different characters, with very total few spoken lines.
# 
# All lines shorter than 20 words are left out from test-data to increase the effectiveness of ngram.
# 
# Note: I did barely read how ngrams really work, and then just quickly threw together something. Therefore this is not an orthodox or exact implementation of ngram, but just fun play-around proof-of-concept. I did also some arbitary modifications to algorihm that seemed to produce better results (e.g. probability of word pair, when there exists no such mathces in the data). 
# 
# ## Results
# These are some outputs of program.
# 
# Top 10 words in the show. Total word count: 443095
# 
#     Prob: 0.02925,  Count: 12961,  - you             
#     Prob: 0.02910,  Count: 12896,  - i               
#     Prob: 0.02880,  Count: 12759,  - the             
#     Prob: 0.02569,  Count: 11384,  - to              
#     Prob: 0.01857,  Count:  8228,  - a               
#     Prob: 0.01596,  Count:  7072,  - and             
#     Prob: 0.01318,  Count:  5839,  - of              
#     Prob: 0.01256,  Count:  5567,  - it              
#     Prob: 0.01092,  Count:  4837,  - that            
#     Prob: 0.00982,  Count:  4350,  - is           
#     
# Top 10 characters with highest word count. Total 841 characters.
# 
#     Count: 61718,  - Twilight Sparkle
#     Count: 36833,  - Rarity
#     Count: 34584,  - Rainbow Dash
#     Count: 34178,  - Pinkie Pie
#     Count: 33371,  - Applejack
#     Count: 24385,  - Fluttershy
#     Count: 24090,  - Spike
#     Count: 15516,  - Others
#     Count: 14519,  - Starlight Glimmer
#     Count: 13998,  - Apple Bloom
#     
# Top 10 writers with highest word count. Total 66 writers.
# 
#     Count: 32832,  - Amy Keating Rogers
#     Count: 31078,  - M. A. Larson
#     Count: 30731,  - Meghan McCarthy
#     Count: 29558,  - Josh Haber
#     Count: 24802,  - Dave Polsky
#     Count: 21231,  - Cindy Morrow
#     Count: 18276,  - Nick Confalone
#     Count: 16711,  - Joanna Lewis & Kristine Songco
#     Count: 16671,  - Michael Vogel
#     Count: 11224,  - Gillian M. Berrow
# 
# Here's two examples: 
# 1. Predicted correct pony and wrong writer, 
# 2. Predicted the pony in top 5, and predicted correct writer.
# 
# (Note: 'P_log' means logarithm of probability.)
# 
#     Epsiode: Bridle Gossip
#     Pony: Twilight Sparkle
#     Writer: Amy Keating Rogers
#     Phrase: "No no no no no! None of these books have a cure! Ugh! There has to be a real reason for this! An illness? An allergy?!"
#     Most probable ponies:
#         Rank:  1. P_log:-516.4, Twilight Sparkle
#         Rank:  2. P_log:-522.0, Starlight Glimmer
#         Rank:  3. P_log:-542.0, Rainbow Dash
#         Rank:  4. P_log:-543.2, Fluttershy
#         Rank:  5. P_log:-543.6, Spike
#         Rank:  6. P_log:-550.0, Applejack
#         Rank:  7. P_log:-551.7, Sweetie Belle
#         Rank:  8. P_log:-556.9, Pinkie Pie
#         Rank:  9. P_log:-560.5, Rarity
#         Rank: 10. P_log:-574.4, Apple Bloom
#     Most probable writers:
#         Rank:  1. P_log:-533.0, Meghan McCarthy
#         Rank:  2. P_log:-550.8, Kaita Mpambara
#         Rank:  3. P_log:-554.2, M. A. Larson
#         Rank:  4. P_log:-554.3, Michael Vogel
#         Rank:  5. P_log:-554.5, Josh Haber
#         Rank:  6. P_log:-562.1, Joanna Lewis & Kristine Songco
#         Rank:  7. P_log:-563.9, F.M. De Marco; story by Meghan McCarthy
#         Rank:  8. P_log:-563.9, Gillian M. Berrow
#         Rank:  9. P_log:-564.2, Amy Keating Rogers
#         Rank: 10. P_log:-564.8, Lauren Faust
# 
#     -----------------------------------------------------
# 
#     Epsiode: Boast Busters
#     Pony: Trixie
#     Writer: Chris Savino
#     Phrase: "Hah! You think you're better than the Great and Powerful Trixie? You think you have more magical talent? Well, come on, show Trixie what you've got. Show us all."
#     Most probable ponies:
#         Rank:  1. P_log:-538.9, Starlight Glimmer
#         Rank:  2. P_log:-548.5, Snips
#         Rank:  3. P_log:-585.1, Trixie
#         Rank:  4. P_log:-627.9, Twilight Sparkle
#         Rank:  5. P_log:-643.5, Applejack
#         Rank:  6. P_log:-647.5, Apple Bloom
#         Rank:  7. P_log:-655.2, Discord
#         Rank:  8. P_log:-655.2, Vinny
#         Rank:  9. P_log:-657.6, Snails
#         Rank: 10. P_log:-659.9, Rarity
#     Most probable writers:
#         Rank:  1. P_log:-492.6, Chris Savino
#         Rank:  2. P_log:-567.6, Josh Haber
#         Rank:  3. P_log:-596.8, M. A. Larson
#         Rank:  4. P_log:-603.1, Joanna Lewis & Kristine Songco
#         Rank:  5. P_log:-608.8, Josh Haber & Michael Vogel
#         Rank:  6. P_log:-619.6, Amy Keating Rogers
#         Rank:  7. P_log:-620.3, Nick Confalone
#         Rank:  8. P_log:-620.3, Josh Haber & Kevin Lappin
#         Rank:  9. P_log:-626.7, Kevin Lappin
#         Rank: 10. P_log:-637.2, Joanna Lewis & Kristine Songco; story by Meghan McCarthy, Joanna Lewis, & Kristine Songco
# 
# 
# 
#         
# Here's mean values of 239 predictions: 
# (Only considering lines with length >= 20)
# 
#     Predicted right pony:            68 / 235 = 0.29
#     Predicted right writer:          89 / 235 = 0.38
#     Predicted right pony in top 5:   125 / 235 = 0.53
#     Predicted right writer in top 5: 138 / 235 = 0.59
# 
# ## Summary
# This program predicts the correct pony with accuracy ~29%, and the correct pony is in top 5 most probable candidates with accuracy ~38%. Even though accuracy is better than pure random, there's a lot of room for improvement. Also one has to remember that lines are quite short compared to normal ngram-data.
# (Worth noting is that the main 6 characters "mane-6" has the most lines in the show. If I were to guess, predicting only Twilight would probably yield at least 10% accuracy. I did not test that though.)

# In[ ]:




