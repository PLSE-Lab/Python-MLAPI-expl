#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Data Mining

#     data mining is a very popular area in recent years,and if you look at different angle,we can have diffiernt opinions.I want to have a very little conlcusion after my study.hope who watch this can inpsire me to do better
#     we all know the importance of data,feature engenering,model emsemble etc. matters to our project.but I am not here to talk about powerful algorithms here.I want to talk about different types of data.constructed data,text.audio.picures and even videos.It was so fascinating that because of computers.we can store and share our life and knowledge without any limitaions,and all this made much bearable in the epedemic around the world.
#     
#  
#    
#     
#     

# # 1.constructed data

# * at first,I want to talk about contsructed data,which is normally seen in information system,we store the productid,names,saleprice,totalsales and so on,and I think you must be similar with house price predicitons and titanic predictions,which we need to del with constructed data,and clean them,and using the model to do the right thing.thi stype of data is very important in our lives,because It can help us to reduce tons of meaningless works,and we can focus on our aglorithms and try something better to improve.

# # 2.text data

# * text data.this kind of data is widely used in internet and social media.It was so important and unconstruted.and now we use the data to find the topics (like LDA) that appears in the paper or comments.and we created sentiment dictionary to predict the sentiment(postive or negative) to do the analysis.this is very interesting and widely used in e-commerce.and we do some noraml tasks like clustering and classfication.data cleaning  is a very impotant step to do this job like remove stopwords,puntuations,and using re packages to deal with dirty signals,and useless stuff,and we split tokenize the data,then counter to build a vocaublary.and encode the text to a sequence.and using the model to train the data.and the whole process is awasome.

# # 3.images data

# *    the pictures are easy to seen.because images are normally stored in json including R,G,B values, a seies of numbers in json.and It was not hard to train the pictures.we need to do decompostion of features sometimes,and this can used to improve the performce we use PCA TSNE,or VGG to do the feature stuff.and It can be very useful to do the mining job.

# # 4.audio data

# *  the last one s aduio mining.honestly,I think nowadays with the development of AI,we can easily tranform the audio data to text then we can do the jobs.but now we think the feeling in a audio data can be studied even better.we need to find the meaning of the words,and we also need to analyze the sentiment of speakers.

# # Final words

# all in all the data can vary in a lot of ways.some data are very professional.like in medicine and design and chemistry.I just did some noraml jobs in construted data and text data,little pictures data.of courese,It was closely connected.but I still thnk that we need better algorthms and understandging others' contribution in the kaggle community.I learnt a lot by joining this platform.and we believe the he development of the era ,the algorithms.we can make more better mining not only in clustering ,classfivation,related vlaues,It needs to do better in this field.

# Thank you so much for watching.If you have any thing speacial want to share,I think we should have more conversations in this palces.please being active and passionate about what you are doing.
# keeping going,KAGGLERs!
