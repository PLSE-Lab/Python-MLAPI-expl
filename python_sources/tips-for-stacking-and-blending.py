#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# As a beginner on Kaggle,  I am always curious about how does people perform the blending? what are the guide lines for blending. 
# In this kernel, I want to share my recent findings from website (all the credits belong to [Soledad](https://www.linkedin.com/pulse/do-you-want-learn-stacking-blending-ensembling-machine-soledad-galli/) and [Kaggle Ensemble Guide](https://mlwave.com/kaggle-ensembling-guide/?lipi=urn%3Ali%3Apage%3Ad_flagship3_pulse_read%3BPZ4T3JLHTu%2BOWNI0d5kFbg%3D%3D)):
# 
# Ensembling or stacking methods are procedures designed to increase predictive performance by blending or combining the predictions of multiple machine learning models. There is a variety of ensembling or stacking methods, from simple ones like voting or averaging the predictions, to building complex learning models (logistic regressions, k-nearest neighbours, boosting trees) using the predictions as features.
# 
# Ensembling or stacking of machine learning model predictions very often beat state-of-the-art academic benchmarks and are widely used to win Kaggle competitions. On the downside, they are usually computationally expensive, but if time and resources are not an issue, a minimal percentage improvement in predictive performance could, for example, help companies save a lot of money.
# 
# Where can you learn about stacking and ensembling of models?
# 
# I have found the Kaggle ensembling guide blog by one of Kaggle top competitors very useful as a starting point to discover the different methods in which predictions from individual models can be combined to improve predictive accuracy. In this blog, the author describes a variety of stacking methods, and how he and his team have used them to win Kaggle competitions. In addition, he provides a number of resources for further reading.
# 
# To perform stacking methods, it is not essential to learn new techniques, algorithms or concepts. Instead, you use what you already know in a creative way to improve the performance of individual algorithms. In this sense, knowing what other scientists have successfully implemented in different predictive scenarios helps understand the variety and the potential of what can be done by combining predictors creatively.
# 
# Some interesting reads include the following (many of them recommended in the Kaggle ensembling guide blog):
# * [Stacked Generalization](https://rd.springer.com/article/10.1007%2FBF00117832), DH Wolpert, Neural Networks, 1992. Wolpert was the first one to introduce the idea of stacking models to improve performance.
# * [Bagging Predictors](https://rd.springer.com/article/10.1023%2FA%3A1018054314350) and [Stacked Regression](https://rd.springer.com/article/10.1007%2FBF00117832), 2 scientific articles from Leo Breiman.
# * [Feature-Weighted linear scaling](https://arxiv.org/pdf/0911.0460.pdf?lipi=urn%253Ali%253Apage%253Ad_flagship3_pulse_read%253BPZ4T3JLHTu%252BOWNI0d5kFbg%253D%253D).
# * [Combining predictors for accurate recommender systems](http://elf-project.sourceforge.net/CombiningPredictionsForAccurateRecommenderSystems.pdf?lipi=urn%3Ali%3Apage%3Ad_flagship3_pulse_read%3BPZ4T3JLHTu%2BOWNI0d5kFbg%3D%3D).
# * [The BigChaos solution to the Netflix Grand Prize](https://www.netflixprize.com/assets/GrandPrize2009_BPC_BigChaos.pdf).
# * [Computer Vision for Head Pose Estimation: Review of a competition](http://vision.cs.tut.fi/data/publications/scia2015_hpe.pdf).
# * [Ensemble Selection from Libraries of Models](http://www.cs.cornell.edu/~caruana/ctp/ct.papers/caruana.icml04.icdm06long.pdf).
# * [The Sum is Greater than the Parts: Ensembling Models of Student Knowledge in Educational Software](http://www.columbia.edu/~rsb2162/PBGH-SIGKDDExp.pdf).
# \
# As always, I hope you enjoy these links, get in touch with questions or suggestions and happy learning together !!!
# 
# 
# 
