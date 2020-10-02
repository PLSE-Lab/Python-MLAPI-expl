#!/usr/bin/env python
# coding: utf-8

# # In this notebook we will learn use of:
# 1. Plotly : One of the best and popular EDA libraries in python/R
# 2. langdetect library: to detect what language is present in text.
# 3. NLTK and wordcloud : Popular libraries for text mining and creating word cloud
# 4. RE: The regular expression library which gives us tools to search, replace and combine patterns in texts, we will explore few feratures in this notebook.
# 5. Scikit learn and Xgboost library to try out predicting ratings based on enrolled certificate type and course difficulty

# # Reading the file

# In[ ]:


import pandas as pd
cs=pd.read_csv(r'/kaggle/input/coursera-course-dataset/coursea_data.csv')


# In[ ]:


cs.describe(include= 'all')


# # Using langdetect function to see what are available languages for courses

# In[ ]:


get_ipython().system('pip install langdetect')
import langdetect
cs['Language'] = cs['course_title'].apply(lambda x: langdetect.detect(x))
cs['Language'].value_counts()


# We see majority of courses are in English followed by distant second spanish, third is french, and 4th is italian, 5th is german and russian

# # Converting k in enrollment to 1000 for better analysis/aggregation

# In[ ]:


cs['enrolled'] = cs['course_students_enrolled'].map(lambda x: str(x)[:-1])


# In[ ]:


cs["enrolled"] = pd.to_numeric(cs["enrolled"])


# In[ ]:


cs["enrolled"]=cs["enrolled"]*1000


# In[ ]:


cs["enrolled"].describe()


# # pie chart for difficulty % using plotly

# In[ ]:


import plotly.express as px
fig = px.pie(cs, values='enrolled', names='course_difficulty')
fig.update_traces(textposition='inside')
fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
fig.show()


# # Does more enrollment leads to better ratings?

# In[ ]:


import plotly.express as px
fig = px.scatter(cs, x="course_rating", y="enrolled", color="course_difficulty",
                 size='course_rating', hover_data=['enrolled'])
fig.show()


# The scatter graph shows enrollment and ratings are not linearly related, the bottom right where high desnity of observations are present do not have very high enrollments.

# # Let explore difficulty vs enrollment vs language using boxplot

# In[ ]:


import plotly.express as px
fig = px.box(cs, x="course_difficulty", y="enrolled",color="Language")
fig.update_traces(quartilemethod="exclusive")
fig.show()


# Key observations from boxplot:
# 1. Only very few advanced courses and all are in English.
# 2. The Italian beginner courses have very high enrollemt variations, one course has over 800K enrollemnts.
# 3. English based courses have almost outliers in every category suggesting there are popular courses across difficulty levels.
# 4. In intermediate level dutch has very high number of enrollments.

# # Exploring text in course title using NLTK and wordcloud

# In[ ]:


from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from wordcloud import WordCloud
cs['course_title_new'] = cs['course_title'].map(lambda x: x.split())
cs['course_title_new']=cs['course_title_new'].apply(lambda x: [item for item in x if item not in stop_words])
cs['course_title_new']=cs['course_title_new'].astype(str)


# In[ ]:


title_count = ','.join(list(cs['course_title_new'].values))


# In[ ]:


wordcloud = WordCloud(background_color="white", max_words=100, contour_width=3, contour_color='steelblue')
wordcloud.generate(title_count)
wordcloud.to_image()


# Python, Data are expected. We also have business, introductionand fundamentals are expected as majority of courses are beginners. We also Machine Learning and Data Science in good numbers.

# # Courses on Python.

# In[ ]:


cs["python"]= cs["course_title"].str.find("Python") 


# In[ ]:


cs.loc[cs['python'] == -1, 'python_yes'] = 0
cs.loc[cs['python'] > -1, 'python_yes'] = 1


# Lets see organisations providing python course.

# In[ ]:


unv_python=cs.groupby(['course_organization'],as_index=False).python_yes.sum()
unv_python


# In[ ]:


unv_python.sort_values(by='python_yes',ascending=False)


# # Lets see courses on ML/AI/DS using regular expression library

# Here we are searching for keywords in course title using re and creating a new column, We will then see which institutes are providing maximum courses.

# In[ ]:


import re
patterns=['AI','Artificial Intelligence','Machine Learning','Data Science','Analytics','Neural Networks','Random Forest',
          'Deep Learning','Reinforcement Learning','Pattern Recognition','Feature Engineering','Kaggle','Data Visualization']
ultimate_pattern = '|'.join(patterns)
def Clean_names(x):
    if re.search(ultimate_pattern, x):
        return 1
    else: 
        return 0
cs['ML_yes'] = cs['course_title'].apply(Clean_names) 


# In[ ]:


cs['ML_yes'].sum()


# In[ ]:


unv_ml=cs.groupby(['course_organization'],as_index=False).ML_yes.sum()
unv_ml.sort_values(by='ML_yes',ascending=False)


# # Exploring top enrolled courses in DS/ML/AI

# In[ ]:


ds_cs=cs[cs['ML_yes']==1]


# In[ ]:


import plotly.express as px
fig = px.bar(ds_cs, x="course_organization", y="enrolled",color="Language")
fig.show()


# We have our top three: John hopkins, deeplearning.ai, and IBM

# # Ratings variations of AI/ML/DS organization courses 

# In[ ]:


import plotly.express as px
fig = px.box(ds_cs, x="course_organization", y="course_rating",color="course_difficulty")
fig.update_traces(quartilemethod="exclusive")
fig.show()


# # Top rated course and speciliazation

# In[ ]:


unv_sp=cs.groupby(['course_organization','course_Certificate_type'],as_index=False).course_rating.mean()
unv_sp.sort_values(by='course_rating',ascending=False)


# # Lets see business courses as well 

# In[ ]:


import re
patterns2=['Business','Management','Leadership','Finance','Accounts','Consulting','Administration']
ultimate_pattern2 = '|'.join(patterns2)
def Clean_names(x):
    if re.search(ultimate_pattern2, x):
        return 1
    else: 
        return 0
cs['B_yes'] = cs['course_title'].apply(Clean_names) 


# In[ ]:


B_cs=cs[cs['B_yes']==1]


# In[ ]:


import plotly.express as px
fig = px.bar(B_cs, x="course_organization", y="enrolled",color="Language")
fig.show()


# The top three B schools by enrollment are Univeristy of Penn, UC Irvine, Macquaire(this has diversity in langauges too)

# # Jaccard Distamce, crude measure of how close text are related :Organization vs course title

# In[ ]:


get_ipython().system('pip install distance')
import distance
jd = lambda x, y: 1 - distance.jaccard(x, y)
sim_unv_course=ds_cs['course_organization'].apply(lambda x: ds_cs['course_title'].apply(lambda y: jd(x, y)))


# In[ ]:


sim_unv_course


# In[ ]:


import numpy as np
(sim_unv_course.values).trace()


# The trace of jaccard distance matrix of DS/AI/ML suggest that course title and university are have high occurence together. 27/83 suggest ~ 33% of similarity.

# # making categories of enrollment to see if something different comes up
# 
# 

# In[ ]:


cs['enrolled_cat'] = pd.cut(cs['enrolled'].astype(int), 5)


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
cs['enrolled_cat']=cs['enrolled_cat'].apply(LabelEncoder().fit_transform)


# In[ ]:


import plotly.express as px
fig = px.box(cs, x="enrolled_cat", y="course_rating",color="course_difficulty")
fig.update_traces(quartilemethod="exclusive")
fig.show()


# Median rating for high enrolled courses moves toward average: Law of averages, less enrolled categories 0 have near 5 ratings for few courses.One less enrolled courses was poorly rated as well.

# # Lets make some ML models to predict ratings using SVM,XGBOOST

# Encoding the difficulty, enrollement category and course type to predict ratings

# In[ ]:


cs_sub=cs[['course_Certificate_type','course_difficulty','enrolled_cat']]


# In[ ]:


cs_sub


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
cs_sub=cs_sub.apply(LabelEncoder().fit_transform)


# In[ ]:


cs_sub


# # XG boost

# In[ ]:


import xgboost as xgb


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(cs_sub, cs['course_rating'], test_size=0.2)


# In[ ]:



xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
model = xg_reg.fit(X_train, Y_train)
import numpy as np
from sklearn.metrics import mean_squared_error

preds = xg_reg.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, preds))
print("RMSE: %f" % (rmse))


# The RMSE 1.4 out of 5 scale rating is very high, probably our tree aren't getting information to split up, we chan check this using reg plot using gpahviz and matplotlib

# Install and set up environment for Graphviz

# In[ ]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin'


# In[ ]:


import matplotlib.pyplot as plt

xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [10, 6]
plt.show()


# In[ ]:


dtrain = xgb.DMatrix(X_train,Y_train)
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
xg_reg = xgb.train(params=params, dtrain=dtrain, num_boost_round=10)


# In[ ]:


import matplotlib.pyplot as plt

xgb.plot_tree(xg_reg,num_trees=9)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()


# The trees are not buiding up suggesting that XGboost hasnt got the enough learning rate and depth of tree as specified in parameters

# # Using Support Vector Machines

# In[ ]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, Y_train)


# We used Radial Basis Function as Kernel as ratings and other variables have non linear relationship. We didnt do hyperparameter tuning and went with defaults.

# In[ ]:


y_pred = regressor.predict(X_test)
rmse = np.sqrt(mean_squared_error(Y_test, y_pred))
print("RMSE: %f" % (rmse))


# This RMSE is way better than XG boost suggesting SVM has fitted data good.

# In[ ]:


error=y_pred-Y_test


# In[ ]:


error


# In[ ]:


min(error),max(error),plt.hist(error)


# 
# Error are nearly normally distributed having less variations and one outlier , the SVM has fitted data better than XG boost.

# More analysis can be done considering ratings to be ordinal ,more feature engineering and exploring more techniques.
