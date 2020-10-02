#!/usr/bin/env python
# coding: utf-8

# ### This kernel was inspired by a tweet from Balaji Srinivasan - https://twitter.com/balajis/status/1207703228516904960
# 
# #### He wanted to know if DL can find correlations between a book's cover or in this case Movie Poster and it's associated Amazon rating/IMDB Score

# ### Imports

# In[ ]:


from fastai.vision import *
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
data = pd.read_csv('/kaggle/input/movie-genre-from-its-poster/MovieGenre.csv',engine='python')
data.head()


# ## Number of Unique IMDB Scores (from 0-10*)
# *Discounting Spinal Tap of course

# In[ ]:


len(data['IMDB Score'].unique())


# ### IMDB Score wise distribution in the dataset

# In[ ]:


import matplotlib.pyplot as plt
fig = plt.gcf()
fig.set_size_inches(15,10)
data['IMDB Score'].value_counts().plot.bar(fig)


# In[ ]:


data.dropna(inplace=True)


# In[ ]:


sns.distplot(data['IMDB Score'])


# ### Customized Image Dataset derived from the original Movie Posters dataset where filenames are named with their IMDB(Score and ID) and uses regex to parse their IMDB score for training labels

# ### Has close to 8.7k images - does increasing the dataset increase the performance of the model?

# In[ ]:


path_img = Path('/kaggle/input/movie-posters/poster_downloads/')
def get_float_labels(file_name):
    return float(re.search('\d.\d',str(file_name)).group())
def get_score_labels(file_name):
    return re.search('\d.\d',str(file_name)).group()


# ## Image Databunch from Fast.ai library
# ### One Image Databunch (data_reg) will act as the training data for a regression approach while the other (data_class) will consider it as a multi-label classification problem

# In[ ]:


data_reg = (ImageList.from_folder(path_img)
 .split_by_rand_pct()
 .label_from_func(get_float_labels, label_cls=FloatList)
 .transform(get_transforms(), size=[300,180])
 .databunch()) 
data_reg.normalize(imagenet_stats)
data_reg.show_batch(rows=3, figsize=(9,6))


# In[ ]:


data_class = (ImageList.from_folder(path_img)
 .split_by_rand_pct()
 .label_from_func(get_score_labels)
 .transform(get_transforms(), size=[300,180])
 .databunch()) 
data_class.normalize(imagenet_stats)
data_class.show_batch(rows=3, figsize=(9,6))


# ## Custom Loss function - inspired from https://medium.com/@btahir/a-quick-guide-to-using-regression-with-image-data-in-fastai-117304c0af90

# In[ ]:


class L1LossFlat(nn.L1Loss):
    "Mean Absolute Error Loss"
    def forward(self, input:Tensor, target:Tensor) -> Rank0Tensor:
        return super().forward(input.view(-1), target.view(-1))


# ## Initialize two trainers with ResNet 50 architecture but one for regression and the other for classification

# In[ ]:


learn_reg = create_cnn(data_reg, models.resnet50)
learn_reg.loss = L1LossFlat


# In[ ]:


learn_class = create_cnn(data_class, models.resnet50,metrics=accuracy)


# In[ ]:


learn_reg.fit_one_cycle(5)


# In[ ]:


learn_class.fit_one_cycle(5)


# ### Image Regression Results

# In[ ]:


learn_reg.show_results(rows=3)


# ### Image Classification Results

# In[ ]:


learn_class.show_results(rows=3)


# ### Scatterplot of Ground Truth and Predictions - Image Regression

# In[ ]:


preds,y,losses = learn_reg.get_preds(with_loss=True)
num_preds = [x[0] for x in np.array(preds)]
num_gt = [x for x in np.array(y)]
scat_data = pd.DataFrame(data={'Predictions':num_preds,'Ground_Truth':num_gt})


# In[ ]:


preds_cl,y_cl = learn_class.get_preds()
labels = np.argmax(preds_cl, 1)
preds_class = [float(data_class.classes[int(x)]) for x in labels]
y_class = [float(data_class.classes[int(x)]) for x in y_cl]
scat_data_cl = pd.DataFrame(data={'Predictions':preds_class,'Ground_Truth':y_class})


# In[ ]:


sns.regplot(x='Predictions',y='Ground_Truth',data = scat_data_cl,lowess=True,scatter_kws={'s':2})


# In[ ]:


sns.regplot(x='Predictions',y='Ground_Truth',data = scat_data,lowess=True,scatter_kws={'s':2})


# In[ ]:


sns.regplot(x='Predictions',y='Ground_Truth',data = scat_data_cl,lowess=True,scatter_kws={'s':2})


# In[ ]:


preds_class,y_class,losses_class = learn_class.get_preds(with_loss=True)
interp = ClassificationInterpretation(learn_class, preds_class, y_class, losses_class)
interp.plot_confusion_matrix()


# ### Some of the most confusing examples for the Image Classification Model

# In[ ]:


interp.plot_top_losses(9, figsize=(15,11))


# ### Testing the model with BvS (IMDB rating - 6.5)
# ![Batman Vs Superman](https://m.media-amazon.com/images/M/MV5BYThjYzcyYzItNTVjNy00NDk0LTgwMWQtYjMwNmNlNWJhMzMyXkEyXkFqcGdeQXVyMTQxNzMzNDI@._V1_QL50_SY1000_CR0,0,675,1000_AL_.jpg**)

# In[ ]:


learn_reg.export('/kaggle/output/')
learn_class.export('/kaggle/output/')
img1 = open_image('/kaggle/input/test-images/test1.jpg')
img2 = open_image('/kaggle/input/test-images/test2.jpg')


# ## Predictions of IR and IC models

# In[ ]:


print("Predicted IMDB Score of Image Regression Model is: ",learn_reg.predict(img1)[0])
print("Predicted IMDB Score of Image Classification Model is: ",learn_class.predict(img1)[0])


# ### Similarly we test with one of 2019's top rated movie - Parasite (IMDB rating - 8.6)
# ![Parasite](https://m.media-amazon.com/images/M/MV5BYWZjMjk3ZTItODQ2ZC00NTY5LWE0ZDYtZTI3MjcwN2Q5NTVkXkEyXkFqcGdeQXVyODk4OTc3MTY@._V1_QL50_SY1000_CR0,0,674,1000_AL_.jpg)

# In[ ]:


print("Predicted IMDB Score of Image Regression Model is: ",learn_reg.predict(img2)[0])
print("Predicted IMDB Score of Image Classification Model is: ",learn_class.predict(img2)[0])


# ### In Conclusion, best not to judge a movie by it's poster? :)
