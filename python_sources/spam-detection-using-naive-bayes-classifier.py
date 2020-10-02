#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## 01-Importing Libraries:

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


# ## 02-Load Data:

# ![](http://)Loading Data 

# In[ ]:


messages = pd.read_csv('../input/sms-spam-collection-dataset/spam.csv',encoding='latin-1')
messages.head()


# It seems like there are some columns, who dont have any value in the data set.we can drop them.
# Hii Friends" Do you guys Love Cricket",Let me know in the comment section. I am from India and I love Cricket.Let me Know about You.

# In[ ]:


messages.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)


# Above line of code drops the unnessary columns,Now lets make Our DataFrame more Interesting by giving some meaningful column names.

# In[ ]:


messages.rename(columns={'v1':'Label','v2':'Message'},inplace = True)
messages.info()


# Here some information about our created dataframe, It seems like there is no NaN or null values present,Let me know if you gus find some. We can clean them .Right...

# ## 03- Visualization:

# Ok now Let's Visualize Our Data.

# As I am a Big Fan "Twilight" , I am using this palette style ,You can use as your wish , I have given a set of Palette styles down ,Play aound

# Possible Palettes are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cividis, cividis_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, twilight, twilight_r, twilight_shifted, twilight_shifted_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r

# In[ ]:


sns.countplot(messages.Label,palette="twilight")
plt.xlabel('Label')
plt.title('Number of ham and spam messages')


# Lets add some percentages to more understanding.

# In[ ]:


ax = sns.countplot(y="Label", data=messages,palette="twilight")
plt.xlabel('Count')
plt.title('Number of ham and spam messages')

total = len(messages['Label'])
for p in ax.patches:
        percentage = '{:.1f}%'.format(100 * p.get_width()/total)
        x = p.get_x() + p.get_width() + 0.02
        y = p.get_y() + p.get_height()/2
        ax.annotate(percentage, (x, y))

plt.show()


# ## 04-Data Cleaning and Preprocessing:

# I will try to make you understand evry line,Please read every comments written after every line of code 

# In[ ]:


ps = PorterStemmer # You can use stemming but here I am using Lemmatization.
lmr = WordNetLemmatizer
corpus = []
for i in range( 0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['Message'][i]) # Cleaning all special characters and numbers, keeping words only
    review = review.lower() # Making all the words to lower case.
    review = review.split()
    
    review = [lmr.lemmatize('word',word) for word in review if word not in set(stopwords.words('english'))]
    # This is list compreshession adding the words which are not avaibale in stopwords.
    review = ' '.join(review)
    corpus.append(review) # adding all words to Corpus


# Stemming->"Stemming is the process of reducing infected or derived words to there word stem, base or root form"-Wikipedia
# But the problem in stemming is some of stemming words dont have any actual meaning.(for example- "intelligence,intelligent,intelligently" becomes "intelgen" which dont have any meaning.
# in otherhand Lemmatization make it "intelligent" which have an actual meaning.

# Ok it's becoming little intense ,,, Let's have some fun.... Anyone can tell me "Pandora" is a name used for one Planate in a very famous Science Friction Movie ,What is the name of the movie ? # Answer it in the comment section

# In[ ]:


# Creating a Bag Of Words model:
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000) # Taking the 5000 important words.
X = cv.fit_transform(corpus).toarray() # Fitting the model to Corpus and converting it to array.


# Lets create a Bag Of words using the messages and consider the 5000 important words.

# Lets's Create Dummi Values for Ham and Spam

# In[ ]:


Y = pd.get_dummies(messages['Label'])
# No need to specify teo catregorical column here we can just define one culumn (ie:if 0-> Ham or if ->1 Spam):
Y = Y.iloc[:,1].values


# Lets see Our X, Is it looking Handsome?
# As it is the independent Variable I am taking it as Hero of Our Movie.(Don't aks me if there are more than one one independent variables, they might be side Heros or Fraiends of Hero)- Got it You Dirty Mind

# In[ ]:


X.shape


# Now it time for Our Heroine, Y

# In[ ]:


Y.shape


# Woew She looks Beautyful...#bornsexy

# ## 05-Spliting Data into train and test:

# In[ ]:


# Spliting the Data into train and test:
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size= 0.2,random_state = 5)


# Test_size = 0.2 mean I am taking 20% of all data to test our model and random_state means that 20% data will be cloose randomly from the main dataset.

# ## 06-Training Model:

# In[ ]:


# Training the model with Naive Bayes Classifier:
from sklearn.naive_bayes import MultinomialNB
spam_detection_model=MultinomialNB().fit(X_train,Y_train)


# In[ ]:


# The predected output of our model
Y_pred = spam_detection_model.predict(X_test)


# Confusion matrix- It show the True +ve,False +ve,True -ve and False -ve values. It mainly helps to find the accuracy of the Classificaton model.

# Okk answer me onething guys ---Which one is more Dengerious ? (False +ve or False -ve or Depends on the situation) If you can explane your ans!!!

# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(Y_test,Y_pred) #It shows the total Right Predictions (960+140) and Wrong Predictions(10+5).
print(confusion_m)


# ## 07-Accuracy:

# Finally Accuracy of Our model.

# In[ ]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(Y_test,Y_pred) # It shows the accuracy of the model.
print (accuracy)


# It looks like we got a good accuracy just using a simple Classifier, we an accuracy approximate to 98 %.OKKK it's GOOD .

# Thank You Guys.ALL the very Best for your Data Science Career.
