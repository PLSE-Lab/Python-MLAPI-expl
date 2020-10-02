#!/usr/bin/env python
# coding: utf-8

# NOTE: this could be viewed as my plan when I stopped working on the competition. It was about to undergo severe pruning, which I decided not to do as I had several other projects which would in my estimation generate more value for me. 

# Now it is time to take the work I have done, and information available, and find fastest way to improve score. There are many many more things to try then time to try them all. The challenge is to pick the the most promising things, and have an good method of abondining tasks early(but not too early). 

# ### things I may do today
# 
# * Take a look at the literature on how to do graph reconstruction
#     * so far I have been trying to solve this naively, with only 3 days its time to switch strategies.
#         * Learning naively is better for learning how to solve unsolved problems
# * look through the forum for hints I have missed(again I have not been following as closely as possible to reduce majority bias, though I have looked enough that I am pretty sure I am mostly doing what others are doing) 
# 
# * consider whether I should try using a distance measure suitible for angles instead of euclidean. 
#     * Balance time cost vs likelyhood of improvement. 
# 
# * investigate joining train and test. 
#     * larger leaderboard scores seem to imply there is some leakage between them
#     * add orientation data and see
#         * properly split via group(note if multiple groups in train are actually from same group this wouldn't work how to test this?) 
#         * check CV score, check lb score. PRetty sure this data is available in community kernels. 
#             * seems likely that there is some mixing between train and test given scores but I can't be sure.
#     * spend some time think about it can you prove it?
# * move data to ssd drive, io is getting on my nerves. 
# 
# * compute edge probabilities for all train_node, test_node pairs. should be ~ 28 million edges. 
#     * If I find a more efficient solution then kill this idea. 
# 
# * can we use class probabilities generated without the use of orientation(hopefully eliminating leakage)
#     * concerned about leakage, as I will be using class probability to improve edge probability, then use this to improve class probability
#         * you could iterate on this, but my sense is there is a leakage trap somewhere in here. 
# 
# * some physics based features(motion equations)
# 
# 
# Overall all though I don't know my solution path right now. It might change rapidly as I learn more about shortest path to 100%. 
# 

# > ### things I see and things I think
# 
# * https://www.kaggle.com/ilhamfp31/fast-fourier-transform-denoising
#     * seems like something to add
#     * probably for thursday
#     * keep an eye on full solution generation time in case I want to fully redo it on last day with new preprocessing

# In[ ]:





# I am done with the competition at 2:03 pm tuesday. At least 2 days over what I wanted(saturday night). I will hold my code until the top 100 fill up. Then I see no harm in releasing it.

# 

# 
