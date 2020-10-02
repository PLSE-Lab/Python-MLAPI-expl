#!/usr/bin/env python
# coding: utf-8

# **Collections of Paintings from 50 Influential Artists**
# * Part 1: Exploratory Data Analysis
# * Part 2: Image Classification using Fastai

# *Part 1: Exploratory Data Analysis*

# In[ ]:


import numpy as np
import pandas as pd 
import cv2
from fastai.vision import *
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from glob import glob
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system("pip freeze > '../working/dockerimage_snapshot.txt'")


# In[ ]:


def makeWordCloud(df,column,numWords):
    topic_words = [ z.lower() for y in
                       [ x.split() for x in df[column] if isinstance(x, str)]
                       for z in y]
    word_count_dict = dict(Counter(topic_words))
    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)
    popular_words_nonstop = [w for w in popular_words if w not in stopwords.words("english")]
    word_string=str(popular_words_nonstop)
    wordcloud = WordCloud(stopwords=STOPWORDS,
                          background_color='white',
                          max_words=numWords,
                          width=1000,height=1000,
                         ).generate(word_string)
    plt.clf()
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

def plotImages(artist,directory):
    print(artist)
    multipleImages = glob(directory)
    plt.rcParams['figure.figsize'] = (15, 15)
    plt.subplots_adjust(wspace=0, hspace=0)
    i_ = 0
    for l in multipleImages[:25]:
        im = cv2.imread(l)
        im = cv2.resize(im, (128, 128)) 
        plt.subplot(5, 5, i_+1) #.set_title(l)
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
        i_ += 1

np.random.seed(7)
overview = pd.read_csv('../input/artists.csv')
overviewArtist = overview[['name','paintings']]
overviewArtist = overviewArtist.sort_values(by=['paintings'],ascending=False)
overviewArtist = overviewArtist.reset_index()
overviewArtist = overviewArtist[['name','paintings']]


# In[ ]:


print(os.listdir("../input/images/images/"))


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 15)
plt.imshow(cv2.imread("../input/images/images/Salvador_Dali/Salvador_Dali_86.jpg"))
shutil.copyfile("../input/images/images/Salvador_Dali/Salvador_Dali_86.jpg", "/kaggle/working/Salvador_Dali_86.jpg")           


# In[ ]:


plotImages("Vincent van Gogh","../input/images/images/Vincent_van_Gogh/**")      


# In[ ]:


plotImages("Leonardo da Vinci","../input/images/images/Leonardo_da_Vinci/**")      


# In[ ]:


plotImages("Andy Warhol","../input/images/images/Andy_Warhol/**")      


# In[ ]:


plotImages("Pablo Picasso","../input/images/images/Pablo_Picasso/**")      


# In[ ]:


plotImages("Salvador Dali","../input/images/images/Salvador_Dali/**")


# In[ ]:


plotImages("Jackson Pollock","../input/images/images/Jackson_Pollock/**")


# In[ ]:


plotImages("Raphael","../input/images/images/Raphael/**")      


# In[ ]:


plotImages("Rembrandt","../input/images/images/Rembrandt/**")      


# In[ ]:


plt.figure(figsize=(10,10))
makeWordCloud(overview,'bio',10000000)


# In[ ]:


plt.figure(figsize=(5,5))
nationalityPlot = sns.countplot(y='nationality',data=overview)
nationalityPlot


# In[ ]:


overviewArtist.head(25)


# *Part 2: Image Classification using Fastai*

# In[ ]:


img_dir='../input/images'
path=Path(img_dir)
data = ImageDataBunch.from_folder(path, train=".", 
                                  valid_pct=0.2,
                                  ds_tfms=get_transforms(do_flip=False,flip_vert=False, max_rotate=0,max_lighting=0.3),
                                  size=299,bs=64, 
                                  num_workers=0).normalize(imagenet_stats)
print(f'Classes: \n {data.classes}')
data.show_batch(rows=8, figsize=(40,40))


# In[ ]:


learn = create_cnn(data, models.resnet50, metrics=accuracy,model_dir=Path("/kaggle/working/"),path=Path("."))
learn.fit_one_cycle(3)


# In[ ]:


plt.rcParams['figure.figsize'] = (5, 5)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot()


# In[ ]:


learn.fit_one_cycle(10)


# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(15,15), dpi=60)


# In[ ]:


plt.rcParams['figure.figsize'] = (15, 15)
plt.imshow(cv2.imread("../input/images/images/Salvador_Dali/Salvador_Dali_86.jpg"))


# In[ ]:




