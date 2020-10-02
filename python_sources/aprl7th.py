#!/usr/bin/env python
# coding: utf-8

# ### April 7th planning(general) 
# 
# 
# Remaining tasks:
# 
# - Write a function that computes span
#     - args: all linear accleration for entire group.
#     - output, min, max offset from origin, range of movement in x, and y
#     
# - Investigate whether groups are spread between the train and test set
#     - if they are now to use this information. 
#     - additional class info
#     
# - Ask the question: What are hte odds someone get 100% today/by end of comp
#     - id on't know 0.5 ? who knows. 
#     
# - Use manually sequenced groups from train and generate more training data with stride 32
#     - obviously save it somewhere. 
# 
# - Inference code should have online feature generation(if not time, then do stride 16 static features). 
#     - need to decide on features, start with 5 likely ones that can be computed real time. 
#     - save trained model somewhere
#     - implement label smoothing
#     - implement method to break groups apart based on smoothed label
#     
# - make a submission
#     - if score is low(under 95), look for method to improve
#         - data cleaning
#         - use 1d convolution network
#         - more feature
#         - some parameter search(automatic) for gbm
#         
# - save train data after concatting all series in groups
# - save test datat after concating all series in group
# - take 1/2 to 1 hour to look for other hidden puzzles
#     - I have not been trying to find these. The community does good job already. 
#  
#  
#  #### optional stuff, and comments
#  
#  - trim the ends of groups, when last sequence is very far away. 
#  
#  start keeping track of the mistakes I made and reinterprate problem with updated information
#  
# - lack of knowledge about graph caused slowdown. Tried to use Graph first when really DiGraph(networkx) is appropiate for thsi competition.
#     - all edges are directed either from end of sequence to beginning of next, or the reverse. 
# - didn't follow up on sense of confusion quickly enough. 
#     - I realized the mental representation i was using for the problem with undirected graph was not using all the data. I did learn the basics of networkx faster though because I did not get stuck in theory to long.
# 
# - didn't explore enough. 
#      - I spent very little time looking at the data
#      - I don't know how imu work: can we reconstruct physical information about the robot from the sensor readings?
#      - didn't explore physical theory of system(physics) very much. Time constraint mainly. I remembered basic physics equations for motion. 
#      - didn't look for puzzles from competition designer(s). I just hoped that the puzzles were difficult, and had an difficult additive tuning term in it so people in top ten can compete in some manner. 
# 
# - assumed I understood what the competition is testing for(many things idk really)
# - lack of skill in traditional signal processing mathematics removed a decently sized set of options for feature generation
#     - mathematics skill acquisition has large time cost. 
# - A much better understanding of mathematical statistics would have have likely caused me to look for trickier things in the data. 
#    

# In[ ]:




