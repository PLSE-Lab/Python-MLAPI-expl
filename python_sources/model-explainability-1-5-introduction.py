#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Model Explainability methods are powerful tools for your Data Science Toolbox.  
# 
# Explainability unlocks tremendous value organizationally, for Data Science and Analytics teams, as it shifts the focus from being applications driven to insights-driven, and shifts the relationship other departments have with your team from being one which serves their business strategy to one which informs their business strategy. This can be really powerful and can unlock a great deal of value for clients as you offer more value-add services.  
#   
# Generally, for Data Scientists, explanations serve one of five main goals:  
# 1. Debugging  
#     - Machine Learning practitioners can identify if and why models are performing poorly and devise strategies in data augmentation to try to mitigate these challenges. This can also hold true for algorithmic bias, as we can see if the model is making critical decisions based on demographic features like age or gender. 
# 2. Feature Engineering
#     - Explanations about the partial residuals of a model can sometimes point to non-linear features in the data which have not been included and could improve model fit and bias. 
# 3. Directing Future Data Collection
#     - Prototypical explanations can sometimes reveal clusters in your data which can often be better accounted for by missing variables.  
# 4. Informing Human Decision-Making and Organizational Strategy
#     - Explanations can inform Customer Segmentation or can drive initiatives in Behavioural Insurance or Banking.  
# 5. User Trust and Buy-in
#     - Providing explanations to clients can improve buy-in and trust in technology as they have the opportunity to verify a model's decision-making criteria against their own expert knowledge.  
#   
# I had initially had planned to make one giant notebook, but it ended up getting way big, and I there was just way too much to cover, I thought it best to split it into a series. 
# 
# In this series, I will be looking at:
# 1. Permutive methods  
# 2. Global surrogate methods  
# 3. Local kernel methods  
# 4. Gradient methods
# 
# For those looking for more content on Model Explainability, this series will try to follow loosely a  [talk](https://github.com/marcusinthesky/Talks/tree/master/UCT%20Reinforcement%20Learning%20Group/Explainability) I gave a few months back from, which has some fantastic links to articles and resources on methods in explainability.  
#   
# Currently, the black-box explainability landscape is still relatively new, and the optimal API for explainability tools is by no means a decided landscape. I would, however, recommend people look into (tf-explain)[https://tf-explain.readthedocs.io/en/latest/] and [IBM XAI360](https://aix360.readthedocs.io/en/latest/) for their fantastic suite of tools and documentation. 
#   
# # The map
# Generally, methods for explainability can be described across two main axes: model reliance and explanation type.  
#   
# Across Model Reliance, we observe types of model explanations which make increasing assumptions about the properties of the model. Mode Reliance has three main categories of explanation: Directly Interprettable Models, which learn human-understandable explanations directly for explaining particular relationships, White Box Models, which leverage specific properties of a model - such as its weights, thresholds or gradients- to provide explanations, and Box Models, which attempt to give explanations which are agnostic to the particular type of model.  
#   
# Across Explanation Type, we have different categories of model explanations which provide different levels of insight into our model's prediction.  The four main categories of Explanations Type are: Global Explanations, in which explanations apply globally to all features, Reference-points Explanations, which explain a particular model's prediction against some reference points, either the average or some prototypical consumer or customer segment, Prototypical Explanations which try to learn clusters of explanations which apply to particular regions of the input space and can be mixed, and Local Explanations, which learn explanations which attempt to describe only a single prediction at one specific point in the input space. 
# 
# 
# |                        | __Local__       | __Propotypical__       | __Reference-point__                 | __Global__                                                                        |
# |------------------------|-------------|--------------------|---------------------------------|-------------------------------------------------------------------------------|
# | __Black-box__              | LIME + SHAP | Regression Mixture | Greedy-hill Climber             | Permutive Importance + Partial Dependence                                     |
# | __White-box__              |             |                    | Integrated Gradients + DeepLift |                                                                               |
# | __Directly Interpretible__ |             |                    |                                 | Boolean Decision Rules via Column Generation + Generalized Linear Rule Models |
# 
#  
# This by no means is the entire landscape of all possible ways of explaining predictions, but is a helpful framework for understanding a number of standard methods which we will be exploring. 
# 
# # The key
# For those new to explainability, my crucial advice is that if you can choose an explainable model to start with rather than attempt more exotic post-hoc explanations on black-box machine learning models. Explainability can come with trade-offs in model complexity or compute, and explanations may serve different strategic roles in your data science project and to your client. My recommendations would be to use the check-list I gave in the section [For Data Scientist](#For-Data-Scientists) to brainstorm the opportunities explanations may provide you and the domain-expert on your project and evaluate the impact those explanations will have.  Be clear about the assumptions the different approaches make to explainability and their hyperparameters, as well as the interpretability of those explanations to non-Data Scientists. 
# 
# # Conclusion
# For those looking for a warm introduction to post-hoc explanations, I would recommend the [Kaggle tutorials](https://www.kaggle.com/learn/machine-learning-explainability) on model explainability as an excellent starting for many persons in the Data Science, as I would consider this series more of a deep dive on code. For those new, I would also recommend this [ebook](https://christophm.github.io/interpretable-ml-book/global.html) on the model explainability landscape as a friendly reference guide on particular methods. 
