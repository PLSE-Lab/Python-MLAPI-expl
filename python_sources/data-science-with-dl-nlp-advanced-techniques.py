#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# # This is a collection of the best Kaggle notebooks (kernels) and other resources (including notebooks (kernels) and posts in discussion from *Prize Competition Winners*) with Advanced Techniques of Data Science (including NLP) by Deep Learning (DL)

# ## Sources:
# ### - Notebooks (kernels) of the Prize Competition Winners
# ### - Notebooks (kernels) of Kaggle Grandmasters, Masters or Experts
# ### - Detailed tutorials of the leading Python libraries
# etc.

# ### Thanks to 
# 
# * @abhinand05, 
# * @arthurtok, 
# * @arunkumarramanan, 
# * @boliu0, 
# * @cdeotte, 
# * @christofhenkel,
# * @cpmpml,
# * @ddanevskyi, 
# * @iezepov, 
# * @itratrahman,
# * @jagannathrk,
# * @jiweiliu, 
# * @kanncaa1, 
# * @kashnitsky, 
# * @kfujikawa, 
# * @leighplt, 
# * @mateiionita, 
# * @mjbahmani,
# * @olegplatonov,
# * @prashant111, 
# * @prokaj, 
# * @pronkinnikita, 
# * @sanjaykr, 
# * @seesee, 
# * @shivamb, 
# * @shahules, 
# * @tonyxu, 
# * @user123454321, 
# * @user189546, 
# * @wowfattie, 
# * @xhlulu, 
# * @yaroshevskiy 
# 
# for their wonderful and helpful notebooks (kernels) or posts which used in my notebook (kernel)!

# <a class="anchor" id="0.1"></a>
# ## Table of Contents
# 
# 1. [Prize Competition Winners: Notebooks (kernels) and Posts with Magic](#1)
#     -  [NLP : Jigsaw Unintended Bias in Toxicity Classification](#1.1)
#     -  [NLP : Gendered Pronoun Resolution](#1.2)
#     -  [NLP : Quora Insincere Questions Classification](#1.3)
#     -  [NLP : TensorFlow 2.0 Question Answering](#1.4)
# 1. [TensorFlow](#2)
# 1. [Keras](#3)
# 1. [PyTorch](#4)
# 1. [Exploratore Data Analyze (EDA)](#5)
# 1. [Fundamentals of Data Science and DL](#6)
# 1. [Interactive DL tools](#7)
#     -  [Neural Network Services](#7.1)
#     -  [MNIST Digit Recognizer](#7.2)
# 1. [Selection of the NN architecture](#8)
# 1. [NLP, analysis and synthesis of text](#9)
#     -  [NLP Tutorials](#9.1)
#     -  [BERT](#9.2)
#     -  [NLTK, WordCloud, Bag of Words, TF IDF, GloVe](#9.3)    
#     -  [LDA, NNMF](#9.4)
#     -  [SpaCy, Gensim](#9.5)
#     -  [Text Generation using LSTM](#9.6)

# ## 1. Prize Competition Winners: Notebooks (kernels) and Posts with Magic <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# ## 1.1. NLP : [Jigsaw Unintended Bias in Toxicity Classification](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) <a class="anchor" id="1.1"></a>
# 
# [Back to Table of Contents](#0.1)

# **The kernel 1 [Wombat Inference Kernel](https://www.kaggle.com/iezepov/wombat-inference-kernel) - 4st place (Private LB) from 3165** 
# 
# **Thanks to @iezepov, @pronkinnikita**
# 
# LSTM, BERT, GPT2CNN and their (23 model and solution) merging.

# ![image.png](attachment:image.png)

# **The kernel 2 [Jigsaw_predict](https://www.kaggle.com/haqishen/jigsaw-predict) - 8st place (Private LB) from 3165** 
# 
# **Thanks to @haqishen**
# 
# Pytorch, Multi-Sample Dropout, 4 model and solution:
# * Bert Small V2 29bin 300seq NAUX,
# * Bert Large V2 99bin 250seq,
# * XLNet 9bin 220seq,
# * GPT2 V2 29bin 350seq NAUX
# 
# and their merging.
# 
# Post about that solution (@haqishen): [8th Place Solution (4 models simple avg)](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/100961)

# ![image.png](attachment:image.png)

# ## 1.2. NLP : [Gendered Pronoun Resolution](https://www.kaggle.com/c/gendered-pronoun-resolution) <a class="anchor" id="1.2"></a>
# 
# [Back to Table of Contents](#0.1)

# **The kernel [Simple logistic regression & BERT [0.27 LB]](https://www.kaggle.com/kashnitsky/simple-logistic-regression-bert-0-27-lb)** 
# 
# The team with author (@kashnitsky, @vlarine, @atanasova, @dennislo, @mateiionita) took **Silver Medal** - 22st place (Private LB) from 838 teams.
# 
# **This kernel basic on the kernel [Taming the BERT - a baseline](https://www.kaggle.com/mateiionita/taming-the-bert-a-baseline)**
# 
# **Thanks to @kashnitsky, @mateiionita**
# 
# BERT, LogisticRegression.

# ## 1.3. NLP : [Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification) <a class="anchor" id="1.3"></a>
# 
# [Back to Table of Contents](#0.1)

# **The kernel [4th place](https://www.kaggle.com/kfujikawa/4th-place) - 4st place (Gold Medal) (Private LB) from 4037 teams** 
# 
# **Thanks to @kfujikawa**
# 
# PyTorch, NLTK, Gensim, LSTM.

# ## 1.4. NLP : [TensorFlow 2.0 Question Answering](https://www.kaggle.com/c/tensorflow2-question-answering) <a class="anchor" id="1.4"></a>
# 
# [Back to Table of Contents](#0.1)

# **The post [1st place solution](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127551) - 1st place (Gold Medal) from 1233 teams**
# 
# **Thanks to @wowfattie**

# **The kernel [submit_full](https://www.kaggle.com/seesee/submit-full) - 2st place (Gold Medal) (Private LB), 1st place (Public LB) from 1233 teams** 
# 
# **Thanks to @seesee**
# 
# TensorFlow, BERT, RoBERTa
# 
# **Comments in post [2nd place solution](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127333)**

# **Post [3rd place solution](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127339) - 3st place (Gold Medal) (Private LB) from 1233 teams** 
# 
# **Thanks to @christofhenkel, @cpmpml**
# 
# PyTorch, RoBERT Large

# **Post [4th place Solution](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127371) - 4st place (Gold Medal) (Private LB) from 1233 teams** 
# 
# **Thanks to @tonyxu**
# 
# TensorFlow, Tried XLNet, Bert Large Uncased/Cased, SpanBert Cased, Bert Large WWM

# **The kernel [Fork of baseline html tokens v5](https://www.kaggle.com/prokaj/fork-of-baseline-html-tokens-v5) - 6st place (Gold Medal) (Private LB) from 1233 teams** 
# 
# **Thanks to @prokaj**
# 
# TensorFlow, BERT
# 
# Comments in post [6th place solution](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127521)

# **The kernel [7th place submission](https://www.kaggle.com/boliu0/7th-place-submission) - 7st place (Gold Medal) (Private LB) from 1233 teams** 
# 
# **Thanks to @boliu0, @jiweiliu**
# 
# TensorFlow, BERT
# 
# Comments in post [7th place solution](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127259)

# **Post [8th place solution](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127545) - 8st place (Gold Medal) (Private LB) from 1233 teams** 
# 
# **Thanks to @olegplatonov**
# 
# TensorFlow, RoBERTa-large and many others technologies and models

# **The kernel [tfqa-bert-train](https://www.kaggle.com/user189546/tfqa-bert-train) - 9st place (Gold Medal) (Private LB) from 1233 teams** 
# 
# **Thanks to @user189546**
# 
# TensorFlow, BERT
# 
# Comments in post [9th place solution](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/128278)

# **The kernel [Solid BERT-joint baseline [0.65/0.66]](https://www.kaggle.com/kashnitsky/solid-bert-joint-baseline-0-65-0-66) - 13st place (Silver Medal) (Private LB) from 1233 teams** 
# 
# **Thanks to @kashnitsky**
# 
# TensorFlow, BERT, many explanations
# 
# Comments in post [Brief summary of 13th place solution (hide the pain Harold)](https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127350) 
# 
# **Thanks for the post to: @ddanevskyi, @kashnitsky, @yaroshevskiy**

# ## to be continued...

# ## 2. TensorFlow <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# **The kernel 1 [TensorFlow Tutorial and Housing Price Prediction](https://www.kaggle.com/arunkumarramanan/tensorflow-tutorial-and-housing-price-prediction)**
# 
# This kernel has Boston Housing Price Prediction from MIT Deep Learning by Lex Fridman and TensorFlow tutorial for Beginners with Latest APIs that was designed by Aymeric Damien for easily diving into TensorFlow, through examples. For readability, it includes both notebooks and source codes with explanation.
# 
# **Many models with TensorFlow (links)**

# **The kernel 2 [Introduction to Tensorflow and Tensorboard](https://www.kaggle.com/sanjaykr/introduction-to-tensorflow-and-tensorboard)**
# 
# Build a Convolutional Neural Network with Tensorflow. The Notebook consists 4 parts:
# 
# * Data Preprocessing
# * Defining CNN architecture
# * Training the Model
# * Visualising Tensorboard Graph and Scalars

# ## 3. Keras <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# **The kernel 1 [Keras tutorial for beginners](https://www.kaggle.com/prashant111/keras-tutorial-for-beginners)**
# 
# ### Keras tutorial
# 
# * What is a backend
# * Keras fundamentals (Keras Sequential model, Keras Functional API)
# * Keras layers (Sequential Model, Layers: Convolutional, MaxPooling, Dense, Dropout)
# * Compile, train and evaluate model
# * Keras in action - Simple Linear Regression example etc.

# **The kernel 2 [MNIST - Deep Neural Network with Keras](https://www.kaggle.com/prashant111/mnist-deep-neural-network-with-keras)**
# 
# In this notebook, author have built a deep neural network on MNIST handwritten digit images to classify them.
# MNIST is called Hello World of Deep Learning.
# So, it is actually an image recognition task.
# It helps you to understand and built a deep neural network with Keras.

# ![image.png](attachment:image.png)

# ## 4. PyTorch <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# **The kernel 1 [PyTorch Tutorial: Dataset. Data preparetion stage.](https://www.kaggle.com/leighplt/pytorch-tutorial-dataset-data-preparetion-stage)**
# 
# Data preparetion. Spet by step
# * Simple Dataset
# * Splitting data into train and validation part
# * Using augmentation for images
# * Adding mask

# **The kernel 2 [Pytorch Tutorial for Deep Learning Lovers](https://www.kaggle.com/kanncaa1/pytorch-tutorial-for-deep-learning-lovers)**
# 
# * Basics of Pytorch
# * Matrices
# * Math
# * Variable
# * Linear Regression
# * Logistic Regression
# * Artificial Neural Network (ANN)
# * Concolutional Neural Network (CNN)
# * Recurrent Neural Network (RNN)

# ## 5. Exploratore Data Analyze (EDA) <a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# **The Kaggle kernels collection [EDA for tabular data: Advanced Techniques](https://www.kaggle.com/vbmokin/eda-for-tabular-data-advanced-techniques)**
# 
# This is a collection of the best Kaggle kernels and other resources with Advanced Techniques of Exploratory Data Analysis (EDA) for tabular data

# ## 6. Fundamentals of Data Science and DL<a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# **The kernel 1 [A Very Comprehensive Tutorial : NN + CNN](https://www.kaggle.com/shivamb/a-very-comprehensive-tutorial-nn-cnn)**
# 
# In this kernel, author have explained the intution about neural networks and how to implement neural networks from scratch in python.

# ![image.png](attachment:image.png)

# **The kernel 2 - the Kaggle kernels collection [Data Science for tabular data: Advanced Techniques](https://www.kaggle.com/vbmokin/data-science-for-tabular-data-advanced-techniques)**
# 
# This is a collection of the best Kaggle kernels and other resources with Advanced Techniques of Data Science for tabular data

# ## 7. Interactive DL tools <a class="anchor" id="7"></a>
# 
# [Back to Table of Contents](#0.1)

# ## 7.1. Neural Network Services <a class="anchor" id="7.1"></a>
# 
# [Back to Table of Contents](#0.1)

# **The web service [Neural Network Playground](http://www.ccom.ucsd.edu/~cdeotte/programs/neuralnetwork.html)**
# 
# **Thanks to @cdeotte**
# 
# ### Interactive Neural Network Playground
# 
# In the example above, the inputs x1 and x2 are fed into two hidden neurons (classifiers). The first divides the feature space with a vertical decision boundary, the second with a hortizonal boundary. These are fed into an output neuron which combines their decisions creating the nonlinear decision boundary pictured. By combining multiple linear decision boundaries the ensemble has the ability to model any shape decision boundary.

# ![image.png](attachment:image.png)

# ## 7.2. MNIST Digit Recognizer <a class="anchor" id="7.2"></a>
# 
# [Back to Table of Contents](#0.1)

# **The web service [MNIST Digit Recognizer](http://www.ccom.ucsd.edu/~cdeotte/programs/MNIST.html)**
# 
# **Thanks to @cdeotte**
# 
# ### Interactive MNIST Digit Recognizer
# 
# Draw a digit between 0 and 9 above and then click classify. A neural network will predict your digit in the blue square above. Your image is 784 pixels (= 28 rows by 28 columns with black=1 and white=0). Those 784 features get fed into a 3 layer neural network; Input:784 - AvgPool:196 - Dense:100 - Softmax:10. The net has 20,600 learned weights hardcoded into this JavaScript webpage. It achieves 98.5% accuracy on the famous MNIST 10k test set and was coded and trained in C. The net is explained here. The best nets are convolutional neural networks and they can achieve 99.8% accuracy. An example coded in Python with Keras and TensorFlow is here.
# 
# Additionally this page allows you to download your hand drawn images. Your images get added to your history showing above to the right. Click 'download' to receive a CSV of digits with or without labels. You can import that CSV into your neural network software for training or testing. The format of the CSV is the same as Kaggle's. Each row is a digit with 784 pixels representing a 28x28 image (rows first). If you download with labels, then each row begins with the label. Assign true labels by clicking images in your history above and then clicking the correct keyboard key number.
# 
# Each image is cropped before classification and before adding to history. If you don't want images cropped, hit the shift key to expose two new buttons. Then hit the button that says, 'Crop Off'. To clear the entire history, simply reload the webpage. To remove one image from history, click it, then click delete.

# ![image.png](attachment:image.png)

# ## 8. Selection of the NN architecture <a class="anchor" id="8"></a>
# 
# [Back to Table of Contents](#0.1)

# **The kernel 1 [CNN Architectures : VGG, ResNet, Inception + TL](https://www.kaggle.com/shivamb/cnn-architectures-vgg-resnet-inception-tl)**
# 
# CNN Architectures : VGG, Resnet, InceptionNet, XceptionNet
# UseCases : Image Feature Extraction + Transfer Learning

# ![image.png](attachment:image.png)

# **The kernel 2 [How to choose CNN Architecture MNIST](https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist)**
# 
# There are so many choices for CNN architecture. How do we choose the best one? First we must define what best means. The best may be the simplest, or it may be the most efficient at producing accuracy while minimizing computational complexity. In this kernel, we will run experiments to find the most accurate and efficient CNN architecture for classifying MNIST handwritten digits.
# 
# The best known MNIST classifier found on the internet achieves 99.8% accuracy!! That's amazing. The best Kaggle kernel MNIST classifier achieves 99.75% posted here. This kernel demostrates the experiments used to determine that kernel's CNN architecture.

# ![image.png](attachment:image.png)

# ## 9. NLP, analysis and synthesis of text <a class="anchor" id="9"></a>
# 
# [Back to Table of Contents](#0.1)

# From the kernel [BERT for Humans: Tutorial+Baseline (Version 2)](https://www.kaggle.com/abhinand05/bert-for-humans-tutorial-baseline-version-2)

# ![image.png](attachment:image.png)

# ## 9.1. NLP Tutorials <a class="anchor" id="9.1"></a>
# 
# [Back to Table of Contents](#0.1)

# **The kernel 3 [NLP Tutorial using Python](https://www.kaggle.com/itratrahman/nlp-tutorial-using-python)**

# ![image.png](attachment:image.png)

# ## 9.2. BERT <a class="anchor" id="9.2"></a>
# 
# [Back to Table of Contents](#0.1)

# **The kernel 1 [BERT for Humans: Tutorial+Baseline (Version 2)](https://www.kaggle.com/abhinand05/bert-for-humans-tutorial-baseline-version-2)**
# 
# 1. Comprehensive BERT Tutorial
# 2. Implementation in Tensorflow 2.0

# ![image.png](attachment:image.png)

# **The kernel 2 [NLP - EDA, Bag of Words, TF IDF, GloVe, BERT](https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert)**
# 
# The kernel sections for BERT basic on kernels:
# * Disaster NLP: Keras BERT using TFHub & tuning
# * [Disaster NLP: Keras BERT using TFHub](https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub) from @xhlulu
# * [Bert starter (inference)](https://www.kaggle.com/user123454321/bert-starter-inference) from @user123454321
# 
# and on resource: https://tfhub.dev/s?q=bert
# 
# 1. Build model using Keras, BERT and TFHub
# 2. PCA transform and visualization
# 3. Confusion matrix
# 4. Submission in competition [Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started)

# ![image.png](attachment:image.png)

# ## 9.3. NLTK, WordCloud, Bag of Words, TF IDF, GloVe <a class="anchor" id="9.3"></a>
# 
# [Back to Table of Contents](#0.1)

# **The kernel 1 - [NLP - EDA, Bag of Words, TF IDF, GloVe, BERT](https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert)**
# 
# This kernel uses the **kernel 2 - [Basic EDA,Cleaning and GloVe](https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove)** of @shahules
# 
# * Import libraries, download data, EDA, NLTK, Data Cleaning
# * WordCloud
# * Bag of Words Counts
# * TF IDF
# * GloVe
# * PCA transform and visualization

# ![image.png](attachment:image.png)

# **The kernel 2 - [Spooky NLP and Topic Modelling tutorial](https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial)**
# 
# * Exploratory Data Analysis (EDA) and Wordclouds - Analyzing the data by generating simple statistics such word frequencies over the different authors as well as plotting some wordclouds (with image masks)
# * Natural Language Processing (NLP) with NLTK (Natural Language Toolkit) - Introducing basic text processing methods such as tokenizations, stop word removal, stemming and vectorizing text via term frequencies (TF) as well as the inverse document frequencies (TF-IDF)

# ## 9.4. LDA, NNMF <a class="anchor" id="9.4"></a>
# 
# [Back to Table of Contents](#0.1)

# **The kernel [Spooky NLP and Topic Modelling tutorial](https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial)**
# 
# Topic Modelling with LDA and NNMF - Implementing the two topic modelling techniques of Latent Dirichlet Allocation (LDA) and Non-negative Matrix Factorization (NMF)

# ![image.png](attachment:image.png)

# ## 9.5. SpaCy, Gensim <a class="anchor" id="9.5"></a>
# 
# [Back to Table of Contents](#0.1)

# **The kernel [Top 3 NLP Libraries Tutorial( NLTK+spaCy+Gensim)](https://www.kaggle.com/jagannathrk/top-3-nlp-libraries-tutorial-nltk-spacy-gensim)**
# 
# You are reading 10 Steps to Become a Data Scientist and are now in the 8th step:
# * Leren Python
# * Python Packages
# * Mathematics and Linear Algebra
# * You are in the 4th step
# * Big Data
# * Data visualization
# * Data Cleaning
# * Tutorial-on-ensemble-learning
# * A Comprehensive ML Workflow with Python
# * Deep Learning
# 
# **SpaCy** is an Industrial-Strength Natural Language Processing in Python.
# 
# **Gensim** is a Python library for topic modelling, document indexing and similarity retrieval with large corpora. Target audience is the natural language processing (NLP) and information retrieval (IR) community.https://github.com/chirayukong/gensim
# * Gensim is a FREE Python library
# * Scalable statistical semantics
# * Analyze plain-text documents for semantic structure
# * Retrieve semantically similar documents. https://radimrehurek.com/gensim/

# ![image.png](attachment:image.png)

# ## 9.6. Text Generation using LSTM <a class="anchor" id="9.6"></a>
# 
# [Back to Table of Contents](#0.1)

# ![image.png](attachment:image.png)

# **The kernel [Beginners Guide to Text Generation using LSTMs](https://www.kaggle.com/shivamb/beginners-guide-to-text-generation-using-lstms)**
# 
# Text Generation is a type of Language Modelling problem. Language Modelling is the core problem for a number of of natural language processing tasks such as speech to text, conversational system, and text summarization. A trained language model learns the likelihood of occurrence of a word based on the previous sequence of words used in the text. Language models can be operated at character level, n-gram level, sentence level or even paragraph level. In this notebook, Author will explain how to create a language model for generating natural language text by implement and training state-of-the-art Recurrent Neural Network.

# ![image.png](attachment:image.png)

# I hope you find this kernel useful and enjoyable.

# Your comments and feedback are most welcome.

# **I will continue this list first of all the decisions of the winners of the competitions and the most voted kernels of Kaggle Kernels Grandmasters, Masters or Experts. I ask those who know such decisions, write about them in the comments**

# [Go to Top](#0)
