#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Log Loss is the most important classification metric based on probabilities. 
# 
# It's hard to interpret raw log-loss values, but log-loss is still a good metric for comparing models.  For any given problem, a lower log-loss value means better predictions.
# 
# Log Loss is a slight twist on something called the **Likelihood Function**. In fact, Log Loss is -1 * the log of the likelihood function. So, we will start by understanding the likelihood function.
# 
# The likelihood function answers the question "How likely did the model think the actually observed set of outcomes was." If that sounds confusing, an example should help.  
# 
# # Example
# A model predicts probabilities of `[0.8, 0.4, 0.1]` for three houses.  The first two houses were sold, and the last one was not sold. So the actual outcomes could be represented numeically as `[1, 1, 0]`.
# 
# Let's step through these predictions one at a time to iteratively calculate the likelihood function.
# 
# The first house sold, and the model said that was 80% likely.  So, the likelihood function after looking at one prediction is 0.8.
# 
# The second house sold, and the model said that was 40% likely.  There is a rule of probability that the probability of multiple independent events is the product of their individual probabilities.  So, we get the combined likelihood from the first two predictions by multiplying their associated probabilities.  That is `0.8 * 0.4`, which happens to be 0.32.
# 
# Now we get to our third prediction.  That home did not sell.  The model said it was 10% likely to sell.  That means it was 90% likely to not sell.  So, the observed outcome of *not selling* was 90% likely according to the model.  So, we multiply the previous result of 0.32 by 0.9.  
# 
# We could step through all of our predictions.  Each time we'd find the probability associated with the outcome that actually occurred, and we'd multiply that by the previous result.  That's the likelihood.
# 
# # From Likelihood to Log Loss
# Each prediction is between 0 and 1. If you multiply enough numbers in this range, the result gets so small that computers can't keep track of it.  So, as a clever computational trick, we instead keep track of the log of the Likelihood.  This is in a range that's easy to keep track of. We multiply this by negative 1 to maintain a common convention that lower loss scores are better.
